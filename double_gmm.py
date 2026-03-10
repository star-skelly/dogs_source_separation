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

# TODO: test known spectra as background to subtract before fitting
# TODO: with evt files, make a spectrum
# TODO: use extended ARF/RMF (not point source) -- experiment with background
# TODO: sherpa package? (for low res spec, which is what we have)
# TODO: characterize background! what's going on with the plasma, radial profile from counts histogram (take xy position -> WCS, calc center of Sag A* and compute the radius then brightness/spectra going out in radius)
    # compare radial profile for background + separated black hole
# then you could load into xspec!

# DECLARE CONSTANTS

EVT_FILE = 'acisi_merged.fits'
XMIN = 4085
XMAX = 4120
YMIN = 4080
YMAX = 4120

EMIN = 2000
EMAX = 8000

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
df = Table([evt_data[c] for c in cols], names=cols, dtype=[np.float64, np.float64, np.float64, np.float64]).to_pandas()

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
    lables : array
        the most likely cluster for a specific pixel
    """
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
    probs, labels = gmm_fitting(ncomp)
    
    table_bg = subset[labels == 0][[*e_lvls, 'x', 'y']].copy()
    table_sources = subset[labels == 1][[*e_lvls, 'x', 'y']].copy()

    table_bg['weight'] = probs[labels == 0, 0]
    table_sources['weight'] = probs[labels == 1, 1]

    print("Background table should be larger than sources table.")
    plot_split([table_bg, table_sources], "1_bg_sources_split.png")
    
    # TODO: better way of determining bg/sources that doesn't rely on size
    # ... image variation loss? but then we have to convert to images first...
    
    if len(table_bg) < len(table_sources):
        return table_sources, table_bg
    
    return table_bg, table_sources

def source_fit(table_sources, nb_source=3):
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
    probs, labels = gmm_fitting(nb_source, table=table_sources, e_lvls=["energy"])
    sources = []
    for i in range(nb_source):
        source_table = table_sources[labels == i][['energy', 'x', 'y']].copy()
        source_table['weight'] = probs[labels == i, i] * table_sources[labels == i]['weight']
        sources.append(source_table)
    return sources


def save_df_as_fits(source_df, filename):
    from astropy.io import fits
from astropy.table import Table

def save_df_as_fits(source_df, filename):
    astropy_table = Table.from_pandas(source_df[['energy', 'x', 'y', 'weight']])
    hdr0 = fits.getheader(EVT_FILE, 0)
    hdr1 = fits.getheader(EVT_FILE, 1).copy()
    new_hdr1 = fits.Header()
    
    for key, value in hdr1.items():
        if any(x in key for x in ['TELESCOP', 'INSTRUME', 'OBS_ID', 'DATE', 'EXPOSURE']):
            new_hdr1[key] = value
            
        # REMAP WCS: if it was for Col 11 (x), make it Col 2
        elif '11' in key:
            new_key = key.replace('11', '2')
            new_hdr1[new_key] = value
            
        # REMAP WCS: if it was for Col 12 (y), make it Col 3
        elif '12' in key:
            new_key = key.replace('12', '3')
            new_hdr1[new_key] = value

    primary_hdu = fits.PrimaryHDU(header=hdr0)
    evt_hdu = fits.BinTableHDU(data=astropy_table, header=new_hdr1)
    evt_hdu.name = "EVENTS"
    hdul = fits.HDUList([primary_hdu, evt_hdu])
    hdul.writeto(f'output/{filename}', overwrite=True)
    print(f"Success! Saved to output/{filename}")

cube, e_lvls = starlet_cube(subset)
table_bg, table_sources = bg_fit()
nb_source = 4
split_sources = source_fit(table_sources, nb_source)
plot_split([*split_sources, table_bg], f"2_split_{nb_source}sources.png")

# Save all as fits events files
for i, source in enumerate([*split_sources, table_bg]):
    save_df_as_fits(source, f'source_{i}.fits')