import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table

import pandas as pd
import pyStarlet_master_2D1D as pys


from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d

import torch

# DECLARE CONSTANTS

EVT_FILE = 'acisi_merged.fits'
XMIN = 4085
XMAX = 4120
YMIN = 4080
YMAX = 4120

EMIN = 2000
EMAX = 6000

VMIN = 0.5
VMAX = 1e3

BINX = 64
BINY = 64
BINE = 100

VERBOSE = False

NUM_LVL = 2
LVL_START = 1

hdu = fits.open(EVT_FILE)
evt_data = hdu[1].data

cols = ['energy', 'x', 'y', 'ccd_id']
df = Table([evt_data[c] for c in cols], names=cols).to_pandas()

subset = df[(df['x'] > XMIN) & (df['x'] < XMAX) & \
            (df['y'] > YMIN) & (df['y'] < YMAX) & \
            (df['energy'] > EMIN) & (df['energy'] < EMAX)]

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

def plot_split(sources, figname):
    nb_source = len(sources)
    fig, axes = plt.subplots(2, nb_source, figsize=(3 * nb_source, 6))

    for i in range(nb_source):
        # --- ROW 1: SPATIAL (TOP) ---
        ax_top = axes[0, i]
        h = ax_top.hist2d(
            sources[i]['x'],
            sources[i]['y'],
            bins=(BINX, BINY),
            range=[[XMIN, XMAX], [YMIN, YMAX]],
            weights=sources[i]['weight'],
            cmap='plasma',
            norm=LogNorm(vmin=VMIN, vmax=VMAX)
        )
        ax_top.set_title(f'Spatial Source {i}')
        ax_top.xaxis.set_visible(False)
        ax_top.yaxis.set_visible(False)

        # --- ROW 2: SPECTRAL (BOTTOM) ---
        ax_bottom = axes[1, i]
        ax_bottom.hist(
            sources[i]['energy']/1e3,
            bins=BINE,
            histtype='step',
            color='crimson'
        )
        ax_bottom.set_title(f'Source {i} Energy Spectrum')
        ax_bottom.set_xlabel('keV')
        ax_bottom.set_ylabel('Weighted Counts')
    plt.tight_layout()
    plt.savefig(f"output/{figname}")

def starlet_cube(subset, lvl_start=LVL_START, num_lvl=NUM_LVL, include_raw=True):
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

def gmm_fitting(ncomp, table=subset, e_lvls = ['energy', 'starlet_0', 'starlet_1']):
    std_scaler = StandardScaler()
    scaled_df = std_scaler.fit_transform(table[['x', 'y', *e_lvls]])

    gmm = GaussianMixture(
        n_components=ncomp,
        covariance_type='full',   # most flexible; try 'diag' for faster
        random_state=0
    )

    gmm.fit(scaled_df)
    probs = gmm.predict_proba(scaled_df)
    labels = gmm.predict(scaled_df) 

    return probs, labels

def bg_fit(ncomp=2, e_lvls = ['energy', 'starlet_0', 'starlet_1']):
    probs, labels = gmm_fitting(ncomp)
    
    table_bg = subset[labels == 0][[*e_lvls, 'x', 'y']].copy()
    table_sources = subset[labels == 1][[*e_lvls, 'x', 'y']].copy()

    table_bg['weight'] = probs[labels == 0, 0]
    table_sources['weight'] = probs[labels == 1, 1]

    print("Check these are not too different")
    print(f"Sky background table size: {len(table_bg)}")
    print(f"Sky source table size: {len(table_sources)}")

    plot_split([table_bg, table_sources], "1_bg_sources_split.png")
    return table_bg, table_sources

def source_fit(table_sources, nb_source=3):
    probs, labels = gmm_fitting(nb_source, table=table_sources, e_lvls=["energy"])
    sources = []
    for i in range(nb_source):
        sources.append(table_sources[labels == i][['energy', 'x', 'y']].copy())
        sources[-1]['weight'] = probs[labels == i, i]
    return sources


cube, e_lvls = starlet_cube(subset)
table_bg, table_sources = bg_fit()
nb_source = 3
split_sources = source_fit(table_sources, nb_source)
plot_split([*split_sources, table_bg], f"2_split_{nb_source}sources.png")

    


    