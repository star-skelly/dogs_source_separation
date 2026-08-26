import glob, os, re, subprocess
from collections import defaultdict

combined_root = "combined"
os.makedirs(combined_root, exist_ok=True)

# Component index is consistent across obsids now (classified against the
# fixed merged-fit model), so grouping by filename is sufficient — no
# centroid/covariance matching needed.
by_component = defaultdict(list)
for pi_path in glob.glob("output/*/*_source_*.pi"):
    m = re.match(r"(\d+)_source_(\d+)\.pi", os.path.basename(pi_path))
    if not m:
        continue
    obsid, comp = m.groups()
    by_component[comp].append((obsid, os.path.abspath(pi_path)))

# Sanity check before combining — see how many obsids actually contribute
# to each component. Worth eyeballing before trusting the stack.
print("=== Contributions per component ===")
for comp, entries in sorted(by_component.items()):
    obsids = sorted(o for o, _ in entries)
    print(f"component {comp}: {len(entries)} obsids — {obsids}")

for comp, entries in sorted(by_component.items()):
    files = [f for _, f in entries]
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