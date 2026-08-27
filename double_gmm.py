import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import joblib

from astropy.io import fits
from astropy.table import Table

import pandas as pd
import pyStarlet_master_2D1D as pys
from astropy import wcs as pywcs

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d

import torch

# TODO: test known spectra as background to subtract before fitting
# TODO: with evt files, make a spectrum
# TODO: use extended ARF/RMF (not point source) -- experiment with background
# TODO: sherpa package? (for low res spec, which is what we have)
# TODO: characterize background! what's going on with the plasma, radial profile from counts histogram (take xy position -> WCS, calc center of Sag A* and compute the radius then brightness/spectra going out in radius)
    # compare radial profile for background + separated black hole
# then you could load into xspec!

# DECLARE CONSTANTS
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--evt-file', required=True, help='reprojected per-obsid event file')
parser.add_argument('--repro-file', required=True, help='original per-obsid repro evt2 file (for header)')
parser.add_argument('--xmin', type=float, required=True)
parser.add_argument('--xmax', type=float, required=True)
parser.add_argument('--ymin', type=float, required=True)
parser.add_argument('--ymax', type=float, required=True)
parser.add_argument('--emin', type=float, default=1500)   # adjust to your existing EMIN default
parser.add_argument('--emax', type=float, default=8000)  # adjust to your existing EMAX default
parser.add_argument('--outdir', default='output')
parser.add_argument('--tag', required=True, help='obsid, used as filename prefix')
parser.add_argument('--mode', choices=['fit', 'classify'], default='fit',
    help="'fit': fit fresh GMMs — run this exactly once, on the merged file. "
         "'classify': load models from --models-dir and classify this obsid's events against them, no fitting.")
parser.add_argument('--models-dir', default='models')
args = parser.parse_args()

EVT_FILE   = args.evt_file
REPRO_FILE = args.repro_file
XMIN, XMAX = args.xmin, args.xmax
YMIN, YMAX = args.ymin, args.ymax
EMIN, EMAX = args.emin, args.emax
OUTDIR     = args.outdir
TAG        = args.tag
MODE = args.mode
MODELS_DIR = args.models_dir
os.makedirs(MODELS_DIR, exist_ok=True)

# EVT_FILE = 'acisi_merged.fits' # ../acisi_merged.fits for markov stuff
# REPRO_FILE = '3392/repro/acisf03392_repro_evt2.fits'
# OUTDIR = 'output'
# TAG = 'base'
# XMIN = 4085
# XMAX = 4120
# YMIN = 4085
# YMAX = 4120

#EVT_FILE = '3392/repro/acisf03392_repro_evt2.fits'
#XMIN = 4110
#XMAX = 4150 
#YMIN = 4050
#YMAX = 4110

# EMIN = 1500
# EMAX = 8000

VMIN = 0.5
VMAX = 1e3

BINX = 128
BINY = 128
BINE = 50

VERBOSE = True

NUM_LVL = 3
LVL_START = 2

NB_SOURCE = 4

os.makedirs(OUTDIR, exist_ok=True)

hdu = fits.open(EVT_FILE)
evt_data = hdu[1].data

cols = ['energy', 'x', 'y', 'ccd_id']
df = Table([evt_data[c] for c in cols], names=cols, dtype=[np.float64, np.float64, np.float64, np.float64]).to_pandas()

subset = df[(df['x'] > XMIN) & (df['x'] < XMAX) & \
            (df['y'] > YMIN) & (df['y'] < YMAX) & \
            (df['energy'] > EMIN) & (df['energy'] < EMAX)]
print(len(subset))
fig = plt.figure(figsize=(5, 2)) 
ax1 = fig.add_subplot(1, 2, 1)
h2d = ax1.hist2d(subset['x'], subset['y'], bins=(BINX, BINY), 
                 cmap='plasma', norm=LogNorm())
ax1.set_title('Spatial Distribution')
fig.colorbar(h2d[3], ax=ax1)

ax2 = fig.add_subplot(1, 2, 2)
ax2.hist(subset['energy']/1.e3, bins=BINE, histtype='step', lw=2)
ax2.set_xlabel('keV', size=14)
ax2.set_title('Energy Spectrum')

plt.tight_layout()
plt.savefig("output/0_input_data.png")

def plot_split(sources, figname, source_names=None):
    """Plotting helper function. Plots spatial and spectral sources

    Parameters
    ---------- 
    sources : list[df]
        input list of pandas dataframes to plot
    figname : string
        filename to save under, will be saved in output/ folder
    """
    nb_source = len(sources)
    fig, axes = plt.subplots(2, nb_source, figsize=(3 * nb_source, 6))
    
    for i in range(nb_source):
        ax_top = axes[0, i]
        h = ax_top.hist2d(
            sources[i]['x'],
            sources[i]['y'],
            bins=(BINX, BINY),
            range=[[XMIN, XMAX], [YMIN, YMAX]],
            weights=sources[i]['weight'],
            density=False,
            cmap='magma',
            norm=LogNorm(vmin=VMIN, vmax=VMAX)
        )
        ax_top.set_title(f'Source {i} Probability', size=12)
        ax_top.xaxis.set_visible(False)
        ax_top.yaxis.set_visible(False)

        ax_bottom = axes[1, i]
        ax_bottom.hist(
            sources[i]['energy']/1e3,
            bins=BINE,
            histtype='step',
            color='blue'
        )
        ax_bottom.set_title(f'Source {i} Spectrum', size=12)
        ax_bottom.set_xlabel('keV')
        ax_bottom.set_ylabel('Event Counts')
    plt.tight_layout()
    plt.savefig(f"output/{figname}")

def starlet_cube(subset, lvl_start=LVL_START, num_lvl=NUM_LVL, include_raw=True, fitted_model=None):
    """
    Builds a spectral cube and applies starlet transformations. 
    Modifies subset to include the starlet level information and 
    returns a spectral cube with the starlet transformed energy

    Parameters
    ---------- 
    subset : df
        input pandas dataframe loaded from fits file
    lvl_start : int
        first starlet level, defaults to LVL_START defined above (usually 1, because lvl 0 starlet is too fine)
    num_lvl : int
        number of starlet levels to use, defaults to NUM_LVL defined above (usually 2, because more than that is too broad)
    include_raw : bool
        whether or not to include the raw energy, unaffected by starlet transformations

    Returns
    ---------- 
    starlet_cube : numpy array
        4D cube containing energy, x, y over different starlet levels
    e_lvls : list
        the energy levels chosen
    """

    # TODO: check this works for include_raw=False

    spectral_cube, edges = np.histogramdd(subset[['energy', 'x', 'y']].values, bins=(BINE, BINX, BINY))

    total_counts = spectral_cube.sum()
    if fitted_model:
        spectral_cube = 2 * np.sqrt(spectral_cube + 3/8) # anscombe transform for sparse data

    if total_counts > 0:
        spectral_cube = spectral_cube / total_counts

    starlet_cube = pys.Starlet_Forward3D(spectral_cube,J=4)[:,:,:,lvl_start:lvl_start+num_lvl]

    if include_raw:
        raw_energy_level = spectral_cube[..., np.newaxis]
        combined_cube = torch.from_numpy(np.concatenate([raw_energy_level, starlet_cube], axis=-1)).permute(0, 3, 1, 2)

    dims = ['energy', 'x', 'y']
    indices = []
    for i, col in enumerate(dims):
        # Find bin index (subtract 1 for 0-indexing)
        idx = np.digitize(subset[col].values, edges[i]) - 1
        # Clip to stay within cube boundaries
        idx = np.clip(idx, 0, starlet_cube.shape[i] - 1)
        indices.append(idx)
    e_idx, x_idx, y_idx = indices

    e_lvls = []

    if include_raw: e_lvls.append(['energy'])

    for lvl in range(num_lvl):
        subset[f'starlet_{lvl}'] = starlet_cube[e_idx, x_idx, y_idx, lvl]
        e_lvls.append(f'starlet_{lvl}')
    
    if include_raw: starlet_cube = combined_cube
    
    return starlet_cube, e_lvls

def gmm_fitting(ncomp, table=subset, e_lvls = ['energy', 'starlet_0', 'starlet_1'], fitted_model=None):
    """Fits a GMM to a table, given the number of components and the energy levels to include.
    
    Parameters
    ---------- 
    ncomp : int
        number of components to fit the GMM to
    table: df
        data to fit to
    e_lvls: list
        levels to consider (column names of df to include)

    Returns
    ---------- 
    probs : array
        the probability that a specific pixel is part of that cluster
    labels : array
        the most likely cluster for a specific pixel
    centers : list
        x, y pairs for the center of each component
    std_devs : list
        x, y pairs for the spread of each component
    fitted_model : dict or None
        {'gmm': GaussianMixture, 'scaler': StandardScaler} already fit elsewhere.
        If given, skips fitting — only transforms + classifies `table` against it.
    """
    if fitted_model is None:
        std_scaler = StandardScaler()
        scaled_df = std_scaler.fit_transform(table[['x', 'y', *e_lvls]])
        gmm = GaussianMixture(n_components=ncomp, covariance_type='full', random_state=0)
        gmm.fit(scaled_df)
    else:
        std_scaler = fitted_model['scaler']
        gmm = fitted_model['gmm']
        scaled_df = std_scaler.transform(table[['x', 'y', *e_lvls]])  # transform only
        ncomp = gmm.n_components

    probs = gmm.predict_proba(scaled_df)
    labels = gmm.predict(scaled_df)

    centers_scaled = gmm.means_
    centers_original = std_scaler.inverse_transform(centers_scaled)
    covariances = gmm.covariances_
    std_devs = [] 

    if VERBOSE:
        for i, center in enumerate(centers_original):
            print(f"Cluster {i} Center: x={center[0]:.2f}, y={center[1]:.2f}")

    for i in range(ncomp):
        v, w = np.linalg.eigh(covariances[i][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi
        v = np.sqrt(v)
        std_devs.append([v[0], v[1], 180 + angle])
        if VERBOSE:
            print(f"Cluster {i} Spread (Scaled Units): x_std={std_devs[i][0]:.2f}, y_std={std_devs[i][1]:.2f}, theta={std_devs[i][2]:.2f}")

    return probs, labels, centers_original, std_devs, gmm, std_scaler

def bg_fit(ncomp=2, e_lvls = ['energy', 'starlet_0', 'starlet_1'], fitted_model=None, forced_bg_second=None):
    """Specific function for splitting the image into background and sources.
        Operates by splitting mostly along the starlet levels, which is activated
        in regions where there seems to be a star-like shape.

    Parameters
    ---------- 
    ncomp : int
        defaults to 2 (background, sources)
    e_lvls: list
        levels to consider (column names of df to include)

    Returns
    ---------- 
    table_bg : df
        the data that is part of the background, with a 'weight' column 
        for probability of each pixel belonging to background.
    table_sourcs : df
        data that belongs to a source, with a 'weight' column for 
        probability of each pixel belonging to a source rather than background.
          
    """
    probs, labels, centers, cov, gmm, scaler = gmm_fitting(ncomp, e_lvls=e_lvls, fitted_model=fitted_model)
    
    table_bg = subset[labels == 0][[*e_lvls, 'x', 'y']].copy()
    table_sources = subset[labels == 1][[*e_lvls, 'x', 'y']].copy()
    table_bg['weight'] = probs[labels == 0, 0]
    table_sources['weight'] = probs[labels == 1, 1]

    print("Background table should be larger than sources table.")
    plot_split([table_bg, table_sources], f"{TAG}_1_bg_sources_split.png", ["Background", "Foreground"])
    
    # TODO: better way of determining bg/sources that doesn't rely on size
    # ... image variation loss? but then we have to convert to images first...
    
    if forced_bg_second is not None:
        bg_second = forced_bg_second       # classify mode: use the merged fit's decision
    else:
        bg_second = len(table_bg) < len(table_sources)   # fit mode: decide once, here

    bg_mask = (labels == 0)
    if bg_second:
        bg_mask = ~bg_mask

    if bg_second:
        return table_sources, table_bg, bg_mask, gmm, scaler, bg_second
    return table_bg, table_sources, bg_mask, gmm, scaler, bg_second

def source_fit(table_sources, nb_source=3, e_lvls=["energy"], fitted_model=None):
    """Specific function for splitting the sources df into each source.
        Operates in energy level only. Defaults to 3 sources for our use case

    Parameters
    ---------- 
    table_sources : df
        table containing the source data (already split from background)
        should have 'weight' column with probabilities
    nb_source: int
        number of objects to split into

    Returns
    ---------- 
    sources : list
        the list of dataframes for each source, each with their own 'weight' column
        corresponding to [belongs to source i] weight * [belongs to a source] weight
    """
    probs, labels, centers, std_devs, gmm, scaler = gmm_fitting(
        nb_source, table=table_sources, e_lvls=e_lvls, fitted_model=fitted_model
    )
    sources = []
    masks = []
    for i in range(nb_source):
        source_table = table_sources[labels == i][['energy', 'x', 'y']].copy()
        source_table['weight'] = probs[labels == i, i] * table_sources[labels == i]['weight']
        sources.append(source_table)
        masks.append(labels == i)
    return sources, centers, std_devs, masks, gmm, scaler

def mask_source_fit(table_sources, nb_source=3):
    """Specific function for splitting the sources df into each source.
        Operates in energy level only. Defaults to 3 sources for our use case

    Parameters
    ---------- 
    table_sources : df
        table containing the source data (already split from background)
        should have 'weight' column with probabilities
    nb_source: int
        number of objects to split into

    Returns
    ---------- 
    masks : list
        the list of masks for each source, compatible with saving as fits file
    """
    probs, labels, centers, std_devs = gmm_fitting(nb_source, table=table_sources, e_lvls=["energy"])
    masks = []
    for i in range(nb_source):
        masks.append(labels == i)
    return masks

def save_with_bgmask(mask, bgmask, out, template):
    """Save events list as fits file, filtered by mask 
        & bgmask, using the template file WCS

    Parameters
    ---------- 
    mask : list
        list of 1s & 0s for each photon, whether it belongs to a specific source
    bgmask : list
        list of 1s & 0s for each photon, whether it belongs to background
    out: str
        name of outfile
    template: str
        name of input fits file to be used as a template for saving (with proper WCS)
    """
    with fits.open(template) as hdul:
        events_hdu = hdul['EVENTS']        
        original_data = events_hdu.data
        spatial_mask = (original_data['x'] > XMIN) & (original_data['x'] < XMAX) & \
                    (original_data['y'] > YMIN) & (original_data['y'] < YMAX) & \
                    (original_data['energy'] > EMIN) & (original_data['energy'] < EMAX)

        subset_data = original_data[spatial_mask][~bgmask]
        filtered_data = subset_data[mask]
        new_table_hdu = fits.BinTableHDU(data=filtered_data, header=events_hdu.header)
        
        new_hdul = fits.HDUList([hdul[0], new_table_hdu])

        for i in range(2, len(hdul)):
            new_hdul.append(hdul[i])
            
        new_hdul.writeto(f"{OUTDIR}/{out}", overwrite=True)

def save_with_masks(mask, out, template):
    """Save events list as fits file, filtered by mask, using the template file WCS

    Parameters
    ---------- 
    mask : list
        list of 1s & 0s for each photon, whether it belongs to a specific source
    out: str
        name of outfile
    template: str
        name of input fits file to be used as a template for saving (with proper WCS)
    """
    with fits.open(template) as hdul:
        events_hdu = hdul['EVENTS']        
        original_data = events_hdu.data
        spatial_mask = (original_data['x'] > XMIN) & (original_data['x'] < XMAX) & \
                    (original_data['y'] > YMIN) & (original_data['y'] < YMAX) & \
                    (original_data['energy'] > EMIN) & (original_data['energy'] < EMAX)

        subset_data = original_data[spatial_mask][mask]
        new_table_hdu = fits.BinTableHDU(data=subset_data, header=events_hdu.header)
        new_hdul = fits.HDUList([hdul[0], new_table_hdu])
        for i in range(2, len(hdul)):
            new_hdul.append(hdul[i])
        new_hdul.writeto(f"{OUTDIR}/{out}", overwrite=True)

# try faking the fits file evt with single file header
def save_with_bgmask_fake(mask, bgmask, out):
    """Save events list as fits file, with faked WCS so specextract works

    Parameters
    ---------- 
    mask : list
        list of 1s & 0s for each photon, whether it belongs to a specific source
    bgmask : list
        list of 1s & 0s for each photon, whether it belongs to background
    out: str
        name of outfile
    template: str
        name of input fits file to be used as a template for saving (with proper WCS)
    """
    single_file_events_hdu = []
    with fits.open(REPRO_FILE) as hdul1:
        single_file_events_hdu = hdul1['EVENTS'] 
        
    with fits.open(EVT_FILE) as hdul:
        events_hdu = hdul['EVENTS']        
        original_data = events_hdu.data
        spatial_mask = (original_data['x'] > XMIN) & (original_data['x'] < XMAX) & \
                    (original_data['y'] > YMIN) & (original_data['y'] < YMAX) & \
                    (original_data['energy'] > EMIN) & (original_data['energy'] < EMAX)

        subset_data = original_data[spatial_mask][~bgmask]
        filtered_data = subset_data[mask]
        new_table_hdu = fits.BinTableHDU(data=filtered_data, header=single_file_events_hdu.header)
        
        new_hdul = fits.HDUList([hdul[0], new_table_hdu])

        for i in range(2, len(hdul)):
            new_hdul.append(hdul[i])
            
        new_hdul.writeto(f"{OUTDIR}/{out}", overwrite=True)

def save_with_masks_fake(mask, out):
    """Save events list as fits file, filtered by mask, 
        with faked WCS so specextract works

    Parameters
    ---------- 
    mask : list
        list of 1s & 0s for each photon, whether it belongs to a specific source
    out: str
        name of outfile
    """
    single_file_events_hdu = []
    with fits.open(REPRO_FILE) as hdul1:
        single_file_events_hdu = hdul1['EVENTS'] 

    with fits.open(EVT_FILE) as hdul:
        events_hdu = hdul['EVENTS']        
        original_data = events_hdu.data
        spatial_mask = (original_data['x'] > XMIN) & (original_data['x'] < XMAX) & \
                    (original_data['y'] > YMIN) & (original_data['y'] < YMAX) & \
                    (original_data['energy'] > EMIN) & (original_data['energy'] < EMAX)

        subset_data = original_data[spatial_mask][mask]
        new_table_hdu = fits.BinTableHDU(data=subset_data, header=single_file_events_hdu.header)
        new_hdul = fits.HDUList([hdul[0], new_table_hdu])
        for i in range(2, len(hdul)):
            new_hdul.append(hdul[i])
        new_hdul.writeto(f"{OUTDIR}/{out}", overwrite=True)

bg_fitted_model = None
src_fitted_model = None
forced_bg_second = None
if MODE == 'classify':
    bg_saved = joblib.load(f"{MODELS_DIR}/bg_model.joblib")
    src_saved = joblib.load(f"{MODELS_DIR}/src_model.joblib")
    bg_fitted_model = {"gmm": bg_saved["gmm"], "scaler": bg_saved["scaler"]}
    src_fitted_model = {"gmm": src_saved["gmm"], "scaler": src_saved["scaler"]}
    forced_bg_second = bg_saved["bg_second"]

cube, e_lvls = starlet_cube(subset, fitted_model=bg_fitted_model)

print("starlet_0 range:", subset['starlet_0'].min(), subset['starlet_0'].max())
print("starlet_1 range:", subset['starlet_1'].min(), subset['starlet_1'].max())
print("n events:", len(subset))

MIN_SOURCE_EVENTS = 20  # same as for specextract's count check (not absolute, can be tuned)

table_bg, table_sources, bg_mask, bg_gmm, bg_scaler, bg_second = bg_fit(
    e_lvls=['energy', 'starlet_0', 'starlet_1'],
    fitted_model=bg_fitted_model,
    forced_bg_second=forced_bg_second,
)

if len(table_sources) < MIN_SOURCE_EVENTS:
    print(f"[SKIP] {TAG}: only {len(table_sources)} events classified as source "
          f"(< {MIN_SOURCE_EVENTS}) — nothing usable to split further.")
    os.makedirs(OUTDIR, exist_ok=True)
    with open(f"{OUTDIR}/{TAG}_qa.json", "w") as fh:
        json.dump({"obsid": TAG, "n_events_in_box": len(subset),
                   "n_background": int(bg_mask.sum()),
                   "n_source": int(len(table_sources)),
                   "skipped": True, "reason": "insufficient source events"}, fh, indent=2)
    sys.exit(0)  # clean exit — not an error, just nothing to extract here


if len(table_bg) >= MIN_SOURCE_EVENTS:
    bg_nb_src = 4
    bg_split_srcs, ctrs, stddv, _, _, _ = source_fit(table_bg, bg_nb_src, e_lvls=['energy', 'starlet_0', 'starlet_1'])
    plot_split([*bg_split_srcs, table_bg], f"{TAG}_2_bg_starlet_split_{bg_nb_src}sources.png")
else:
    print(f"[{TAG}] skipping bg-starlet diagnostic plot: table_bg has only {len(table_bg)} events")

split_sources, centers, std_dev, src_masks, src_gmm, src_scaler = source_fit(
    table_sources, NB_SOURCE, e_lvls=["energy"], fitted_model=src_fitted_model
)
plot_split([*split_sources, table_bg],
    f"{TAG}_2_split_{NB_SOURCE}sources.png", 
    ["IRS 13", "Pulsar Wind Nebula Core", "Sgr A*", "Pulsar Wind Nebula Tail", "Background"])

with fits.open(EVT_FILE) as hdul:
    solved_wcs = pywcs.WCS(hdul[0].header)

for i, source in enumerate(src_masks):
    save_with_bgmask_fake(source, bg_mask, f"{TAG}_source_{i}.fits")
save_with_masks_fake(bg_mask, f"{TAG}_source_{i+1}.fits")

if MODE == 'fit':
    joblib.dump({"gmm": bg_gmm, "scaler": bg_scaler, "bg_second": bg_second}, f"{MODELS_DIR}/bg_model.joblib")
    joblib.dump({"gmm": src_gmm, "scaler": src_scaler}, f"{MODELS_DIR}/src_model.joblib")

    component_metadata = []
    for idx, (center, std) in enumerate(zip(centers, std_dev)):
        component_metadata.append({
            "component": idx, "x": float(center[0]), "y": float(center[1]),
            "std_x": float(std[0]), "std_y": float(std[1]), "theta_deg": float(std[2]),
        })
    with open(f"{MODELS_DIR}/global_components.json", "w") as fh:
        json.dump(component_metadata, fh, indent=2)
    print(f"Saved global models to {MODELS_DIR}/. bg_second={bg_second}")
else:
    qa = {"obsid": TAG, "n_events_in_box": len(subset),
          "n_background": int(bg_mask.sum()), "n_source": int((~bg_mask).sum())}
    with open(f"{OUTDIR}/{TAG}_qa.json", "w") as fh:
        json.dump(qa, fh, indent=2)