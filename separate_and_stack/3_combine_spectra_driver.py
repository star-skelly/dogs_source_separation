import glob, os, re, subprocess
import numpy as np
from astropy.io import fits
from scipy.optimize import linear_sum_assignment

combined_root = "combined"
os.makedirs(combined_root, exist_ok=True)

CENTROID_WEIGHT = 3.0
VARIANCE_WEIGHT = 1.0


def get_component_stats(evtfile):
    """Returns (centroid, spread) for one component's event file.
    ASSUMPTION: matching on spatial (x, y) position. If double_gmm.py actually
    clusters on energy instead, replace this with:
        energy = hdul["EVENTS"].data["energy"]
        return energy.mean(), energy.var()
    """
    with fits.open(evtfile) as hdul:
        data = hdul["EVENTS"].data
        x, y = data["x"], data["y"]
    centroid = np.array([x.mean(), y.mean()])
    spread = x.var() + y.var()  # trace of the 2x2 covariance matrix
    return centroid, spread


# 1. Gather stats for every successfully-extracted (obsid, component)
obsid_components = {}  # obsid -> {comp_idx: {"centroid", "spread", "pi"}}
for pi_path in glob.glob("output/*/*_source_*.pi"):
    m = re.match(r"(\d+)_source_(\d+)\.pi", os.path.basename(pi_path))
    if not m:
        continue
    obsid, comp = m.groups()
    comp = int(comp)
    evtfile = os.path.join(os.path.dirname(pi_path), f"{obsid}_source_{comp}.fits")
    if not os.path.exists(evtfile):
        print(f"[warn] no event file for {pi_path}, skipping in matching")
        continue
    centroid, spread = get_component_stats(evtfile)
    obsid_components.setdefault(obsid, {})[comp] = {
        "centroid": centroid, "spread": spread, "pi": os.path.abspath(pi_path)
    }

# 2. Split off the background component (always the highest index per obsid)
background_by_obsid = {}
source_components_by_obsid = {}
for obsid, comps in obsid_components.items():
    bg_idx = max(comps.keys())
    background_by_obsid[obsid] = comps[bg_idx]
    source_components_by_obsid[obsid] = {i: v for i, v in comps.items() if i != bg_idx}

# 3. Reference obsid = the one with the most successfully-matched source components,
#    so every label has a chance to be anchored
ref_obsid = max(source_components_by_obsid, key=lambda o: len(source_components_by_obsid[o]))
ref_comps = source_components_by_obsid[ref_obsid]
ref_indices = sorted(ref_comps.keys())
ref_centroids = np.array([ref_comps[i]["centroid"] for i in ref_indices])
ref_spreads = np.array([ref_comps[i]["spread"] for i in ref_indices])

# scale terms so centroid (pixels) and spread (pixels^2) are comparable before weighting
centroid_scale = np.linalg.norm(ref_centroids.std(axis=0)) or 1.0
spread_scale = ref_spreads.std() or 1.0

# 4. Match every other obsid's components to reference labels via optimal assignment
matched_label_by_obsid = {ref_obsid: {i: i for i in ref_indices}}

for obsid, comps in source_components_by_obsid.items():
    if obsid == ref_obsid:
        continue
    comp_indices = sorted(comps.keys())
    cost = np.zeros((len(comp_indices), len(ref_indices)))
    for a, ci in enumerate(comp_indices):
        for b, ri in enumerate(ref_indices):
            dc = np.linalg.norm(comps[ci]["centroid"] - ref_comps[ri]["centroid"]) / centroid_scale
            dv = abs(comps[ci]["spread"] - ref_comps[ri]["spread"]) / spread_scale
            cost[a, b] = CENTROID_WEIGHT * dc + VARIANCE_WEIGHT * dv
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {comp_indices[a]: ref_indices[b] for a, b in zip(row_ind, col_ind)}
    matched_label_by_obsid[obsid] = mapping

    # sanity-check printout — eyeball this before trusting the stack
    for a, b in zip(row_ind, col_ind):
        ci, ri = comp_indices[a], ref_indices[b]
        print(f"obsid {obsid}: component {ci} -> label {ri} "
              f"(cost={cost[a, b]:.3f}, centroid={comps[ci]['centroid']}, "
              f"ref_centroid={ref_comps[ri]['centroid']})")

import numpy as np
from collections import defaultdict

# after the matching loop, before building by_component:

matches_by_label = defaultdict(list)  # label -> list of (obsid, orig_idx, centroid, cost)
for obsid, comps in source_components_by_obsid.items():
    mapping = matched_label_by_obsid[obsid]
    for orig_idx, label in mapping.items():
        matches_by_label[label].append(
            (obsid, orig_idx, comps[orig_idx]["centroid"], comps[orig_idx]["spread"])
        )

print("\n=== Per-label match diagnostics ===")
flagged = []
for label, entries in sorted(matches_by_label.items()):
    centroids = np.array([e[2] for e in entries])
    mean_c = centroids.mean(axis=0)
    dists = np.linalg.norm(centroids - mean_c, axis=1)
    median_dist = np.median(dists)
    print(f"\nlabel {label}: n={len(entries)}, mean_centroid={mean_c}, "
          f"median_dist={median_dist:.2f}px, max_dist={dists.max():.2f}px")
    for (obsid, orig_idx, centroid, spread), d in zip(entries, dists):
        # flag anything notably far from its own label's consensus centroid
        if d > max(3.0, 3 * median_dist):
            print(f"  [FLAG] obsid {obsid} comp {orig_idx}: {d:.2f}px from label mean "
                  f"(centroid={centroid})")
            flagged.append((obsid, orig_idx, label))

print(f"\n{len(flagged)} matches flagged for review.")

# 5. Rebuild by_component using matched labels for sources, fixed bucket for background
by_component = {}
for obsid, comps in source_components_by_obsid.items():
    mapping = matched_label_by_obsid[obsid]
    for orig_idx, info in comps.items():
        label = mapping[orig_idx]
        by_component.setdefault(str(label), []).append(info["pi"])

for obsid, info in background_by_obsid.items():
    by_component.setdefault("background", []).append(info["pi"])

# 6. combine_spectra per matched component, same as before
for comp, files in sorted(by_component.items()):
    listfile = os.path.abspath(f"{combined_root}/component_{comp}_src.lis")
    with open(listfile, "w") as fh:
        fh.write("\n".join(sorted(files)))

    outroot = os.path.abspath(f"{combined_root}/source_{comp}")
    cmd = [
        "combine_spectra",
        f"src_spectra=@{listfile}",
        f"outroot={outroot}",
        "clobber=yes",
        "verbose=1",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)