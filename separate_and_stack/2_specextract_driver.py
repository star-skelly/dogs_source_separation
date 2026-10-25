import glob, os, subprocess
import re
from astropy.io import fits

EMIN = 1500
EMAX = 8000
XMIN, XMAX, YMIN, YMAX = 4085, 4120, 4085, 4120

xcen, ycen = (XMIN + XMAX) / 2, (YMIN + YMAX) / 2
width, height = XMAX - XMIN, YMAX - YMIN
region_filter = f"[sky=rotbox({xcen},{ycen},{width},{height},0)]"

MIN_COUNTS = 20

def find_calfile(obsid, kind):
    """kind = 'bpix1' or 'msk1'. Handles .fits and .fits.gz — CIAO/CFITSIO
    reads gzipped FITS transparently, no need to decompress first."""
    subdir = "primary" if kind == "bpix1" else "secondary"
    patterns = [
        f"acisi_img/{obsid}/{subdir}/acisf{int(obsid):05d}_*{kind}.fits*",
        f"acisi_img/{obsid}/primary/acisf{int(obsid):05d}_*{kind}.fits*",
        f"acisi_img/{obsid}/secondary/acisf{int(obsid):05d}_*{kind}.fits*",
        f"{obsid}/repro/acisf{int(obsid):05d}_*{kind}.fits*",
        f"acisi_img/*{obsid}*{kind}*.fits*",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]
    return None

def find_asol_file(obsid):
    patterns = [
        f"acisi_img/*{obsid}*asol*",
        f"acisi_img/{obsid}/primary/*asol*",
        f"acisi_img/{obsid}/secondary/*asol*",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None

failures = []
skipped = []

for evtfile in sorted(glob.glob("output/*/*_source_*.fits")):
    if not re.search(r"/\d+_source_\d+\.fits$", evtfile):
        continue
    outroot = os.path.splitext(evtfile)[0]
    obsid = evtfile.split("/")[1]

    with fits.open(evtfile) as hdul:
        data = hdul["EVENTS"].data
        n_counts = ((data["energy"] > EMIN) & (data["energy"] < EMAX)).sum()

    if n_counts < MIN_COUNTS:
        print(f"[skip] {evtfile}: only {n_counts} counts in {EMIN}:{EMAX} eV (< {MIN_COUNTS})")
        skipped.append((evtfile, n_counts))
        continue

    asol_file = find_asol_file(obsid)
    if asol_file is None:
        print(f"[skip] no asol file found for obsid {obsid}")
        failures.append((evtfile, "no asol file"))
        continue

    bpix_file = find_calfile(obsid, "bpix1")
    if bpix_file is None:
        print(f"[skip] no bpix file found for obsid {obsid}")
        failures.append((evtfile, "no bpix file"))
        continue

    msk_file = find_calfile(obsid, "msk1")
    if msk_file is None:
        print(f"[skip] no mask file found for obsid {obsid}")
        failures.append((evtfile, "no mask file"))
        continue

    cmd = [
        "specextract",
        f"infile={evtfile}{region_filter}",
        f"outroot={outroot}",
        f"asp={asol_file}",
        f"badpixfile={bpix_file}",
        f"mskfile={msk_file}",
        "weight=yes",
        "correctpsf=no",
        f"energy_wmap={EMIN}:{EMAX}",
        "grouptype=NONE",
        "binspec=NONE",
        "clobber=yes",
    ]
    print(f"Running ({n_counts} counts): {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        last_line = e.stderr.strip().splitlines()[-1] if e.stderr else str(e)
        print(f"[FAILED] {evtfile}: {last_line}")
        failures.append((evtfile, e.stderr))

print(f"\n{len(skipped)} skipped for low counts, {len(failures)} failed extraction.")

if skipped or failures:
    with open("output/specextract_log.txt", "w") as fh:
        fh.write("=== Skipped (low counts) ===\n")
        for f, n in skipped:
            fh.write(f"{f}: {n} counts\n")
        fh.write("\n=== Failed ===\n")
        for f, reason in failures:
            fh.write(f"{f}\n{reason}\n\n")