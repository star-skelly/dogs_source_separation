import glob, os, re, subprocess

IMG_DIR = "acisi_img"
XMIN, XMAX, YMIN, YMAX = 4085, 4120, 4085, 4120  # common box, reprojected frame

reproj_files = sorted(glob.glob(os.path.join(IMG_DIR, "*_reproj_evt.fits")))

for f in reproj_files:
    base = os.path.basename(f)
    m = re.match(r"(\d+)_reproj_evt\.fits", base)
    if not m:
        continue  # skips merged_evt.fits
    obsid = m.group(1)

    outdir = f"output/{obsid}"
    os.makedirs(outdir, exist_ok=True)

    cmd = ["python", "transfer_labels.py",
       "--evt-file", f, "--labels", "../models/merged_labels.parquet",
       "--xmin", str(XMIN), "--xmax", str(XMAX), "--ymin", str(YMIN), "--ymax", str(YMAX),
       "--outdir", outdir, "--tag", obsid]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)