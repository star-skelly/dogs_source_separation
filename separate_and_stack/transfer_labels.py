#!/usr/bin/env python
"""Transfer per-photon labels from the merged fit onto one obsid's reprojected
event file by exact key join. No GMM, no starlets, no feature recomputation.

Requires merged_labels.parquet written by double_gmm.py --mode fit
(columns: time, x, y, energy, ccd_id, is_bg, src_label, weight).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

KEYS = ['time', 'x', 'y', 'energy', 'ccd_id']

p = argparse.ArgumentParser()
p.add_argument('--evt-file', required=True, help='reprojected per-obsid event file')
p.add_argument('--repro-file', default=None, help='header source for saved files (default: --evt-file)')
p.add_argument('--labels', required=True, help='merged_labels.parquet from the fit run')
p.add_argument('--xmin', type=float, required=True)
p.add_argument('--xmax', type=float, required=True)
p.add_argument('--ymin', type=float, required=True)
p.add_argument('--ymax', type=float, required=True)
p.add_argument('--emin', type=float, default=1500)
p.add_argument('--emax', type=float, default=8000)
p.add_argument('--nb-source', type=int, default=4)
p.add_argument('--outdir', default='output')
p.add_argument('--tag', required=True, help='obsid, used as filename prefix')
args = p.parse_args()

os.makedirs(args.outdir, exist_ok=True)
header_file = args.repro_file or args.evt_file

lab = pd.read_parquet(args.labels)
lab[KEYS] = lab[KEYS].astype(np.float64)
dups = lab.duplicated(KEYS).sum()
if dups:
    raise SystemExit(f"{dups} duplicate keys in merged labels; key is not unique, "
                     "add another column (e.g. obsid, expno) before joining")

with fits.open(header_file) as h1:
    out_header = h1['EVENTS'].header.copy()

with fits.open(args.evt_file) as hdul:
    data = hdul['EVENTS'].data
    box = ((data['x'] > args.xmin) & (data['x'] < args.xmax) &
           (data['y'] > args.ymin) & (data['y'] < args.ymax) &
           (data['energy'] > args.emin) & (data['energy'] < args.emax))
    ev = data[box]

    df = Table([ev[c] for c in KEYS], names=KEYS).to_pandas().astype(np.float64)
    m = df.merge(lab, on=KEYS, how='left', validate='many_to_one', indicator=True)
    assert len(m) == len(ev), "merge changed row count/order"
    matched = (m['_merge'] == 'both').to_numpy()
    n_unmatched = int((~matched).sum())
    if n_unmatched:
        print(f"[{args.tag}] WARNING: {n_unmatched}/{len(m)} events not found in merged labels; excluded")

    is_bg = m['is_bg'].fillna(False).astype(bool).to_numpy()
    src_label = m['src_label'].fillna(-1).astype(int).to_numpy()

    def save(sel, name):
        new_hdu = fits.BinTableHDU(data=ev[sel], header=out_header)
        new = fits.HDUList([hdul[0], new_hdu] + [hdul[i] for i in range(2, len(hdul))])
        new.writeto(f"{args.outdir}/{name}", overwrite=True)

    counts = {}
    for i in range(args.nb_source):
        sel = matched & ~is_bg & (src_label == i)
        counts[f"source_{i}"] = int(sel.sum())
        save(sel, f"{args.tag}_source_{i}.fits")
    sel_bg = matched & is_bg
    counts["background"] = int(sel_bg.sum())
    save(sel_bg, f"{args.tag}_source_{args.nb_source}.fits")

# per-event labels + posterior weights, for probability-weighted analysis later
m.drop(columns='_merge').to_parquet(f"{args.outdir}/{args.tag}_labels.parquet")

with open(f"{args.outdir}/{args.tag}_qa.json", "w") as fh:
    json.dump({"obsid": args.tag, "n_events_in_box": int(len(m)),
               "n_matched": int(matched.sum()), "n_unmatched": n_unmatched,
               **counts}, fh, indent=2)
print(f"[{args.tag}] done: {counts}")