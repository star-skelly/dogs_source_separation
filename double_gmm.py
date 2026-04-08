import numpy as np
import matplotlib.pyplot as plt

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

EVT_FILE = 'acisi_merged.fits' # ../acisi_merged.fits for markov stuff
REPRO_FILE = '3392/repro/acisf03392_repro_evt2.fits'
XMIN = 4085
XMAX = 4120
YMIN = 4080
YMAX = 4120

#EVT_FILE = '3392/repro/acisf03392_repro_evt2.fits'
#XMIN = 4110
#XMAX = 4150 
#YMIN = 4050
#YMAX = 4110

EMIN = 1500
EMAX = 8000

VMIN = 0.5
VMAX = 1e3

BINX = 64
BINY = 64
BINE = 50

VERBOSE = True

NUM_LVL = 3
LVL_START = 1

NB_SOURCE = 4

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
        ax_top = axes[0, i]
        h = ax_top.hist2d(
            sources[i]['x'],
            sources[i]['y'],
            bins=(BINX, BINY),
            range=[[XMIN, XMAX], [YMIN, YMAX]],
            weights=sources[i]['weight'],
            cmap='viridis',
            norm=LogNorm(vmin=VMIN, vmax=VMAX)
        )
        ax_top.set_title(f'Spatial Source {i}')
        ax_top.xaxis.set_visible(False)
        ax_top.yaxis.set_visible(False)

        ax_bottom = axes[1, i]
        ax_bottom.hist(
            sources[i]['energy']/1e3,
            bins=BINE,
            histtype='step',
            color='blue'
        )
        ax_bottom.set_title(f'Source {i} Energy Spectrum')
        ax_bottom.set_xlabel('keV')
        ax_bottom.set_ylabel('Event Counts')
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
    labels : array
        the most likely cluster for a specific pixel
    centers : list
        x, y pairs for the center of each component
    std_devs : list
        x, y pairs for the spread of each component
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

    centers_scaled = gmm.means_
    centers_original = std_scaler.inverse_transform(centers_scaled) # since we used standard scaler, have to transform back
    covariances = gmm.covariances_
    std_devs = [] # should be a list of [std_x, std_y, theta]
    
    if VERBOSE:
        for i, center in enumerate(centers_original):
            print(f"Cluster {i} Center: x={center[0]:.2f}, y={center[1]:.2f}")

    for i in range(ncomp):
        v, w = np.linalg.eigh(covariances[i][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = np.sqrt(v)

        std_devs.append([v[0], v[1], 180 + angle])
    
        if VERBOSE: 
            print(f"Cluster {i} Spread (Scaled Units): x_std={std_devs[i][0]:.2f}, y_std={std_devs[i][1]:.2f}, theta={std_devs[i][2]:.2f}")

    return probs, labels, centers_original, std_devs

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
    probs, labels, centers, cov = gmm_fitting(ncomp, e_lvls=e_lvls)
    
    table_bg = subset[labels == 0][[*e_lvls, 'x', 'y']].copy()
    table_sources = subset[labels == 1][[*e_lvls, 'x', 'y']].copy()

    table_bg['weight'] = probs[labels == 0, 0]
    table_sources['weight'] = probs[labels == 1, 1]

    print("Background table should be larger than sources table.")
    plot_split([table_bg, table_sources], "1_bg_sources_split.png")
    
    # TODO: better way of determining bg/sources that doesn't rely on size
    # ... image variation loss? but then we have to convert to images first...
    
    bg_second = len(table_bg) < len(table_sources)

    bg_mask = (labels == 0)
    if bg_second:
        bg_mask = ~bg_mask

    if bg_second:
        return table_sources, table_bg, bg_mask
    
    return table_bg, table_sources, bg_mask

def source_fit(table_sources, nb_source=3, e_lvls=["energy"]):
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
    probs, labels, centers, std_devs = gmm_fitting(nb_source, table=table_sources, e_lvls=e_lvls)
    sources = []
    for i in range(nb_source):
        source_table = table_sources[labels == i][['energy', 'x', 'y']].copy()
        source_table['weight'] = probs[labels == i, i] * table_sources[labels == i]['weight']
        sources.append(source_table)
    return sources, centers, std_devs

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
            
        new_hdul.writeto(f"output/{out}", overwrite=True)

def save_with_masks(mask, out, template):
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
        new_hdul.writeto(f"output/{out}", overwrite=True)

# try faking the fits file evt with single file header
def save_with_bgmask_fake(mask, bgmask, out):
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
            
        new_hdul.writeto(f"output/{out}", overwrite=True)
def save_with_masks_fake(mask, out):
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
        new_hdul.writeto(f"output/{out}", overwrite=True)

cube, e_lvls = starlet_cube(subset)
table_bg, table_sources, bg_mask = bg_fit(e_lvls = ['energy', 'starlet_0', 'starlet_1'])

bg_nb_src = 4
bg_split_srcs, ctrs, stddv  = source_fit(table_bg, bg_nb_src, e_lvls = ['energy', 'starlet_0', 'starlet_1'])
plot_split([*bg_split_srcs, table_bg], f"2_bg_starlet_split_{bg_nb_src}sources.png")

split_sources, centers, std_dev = source_fit(table_sources, NB_SOURCE)
src_masks = mask_source_fit(table_sources, NB_SOURCE)
plot_split([*split_sources, table_bg], f"2_split_{NB_SOURCE}sources.png")

# Save all as fits events files
with fits.open(EVT_FILE) as hdul:
    solved_wcs = pywcs.WCS(hdul[0].header)

for i, source in enumerate(src_masks):
    save_with_bgmask_fake(source, bg_mask, f"source_{i}.fits")

save_with_masks_fake(bg_mask, f"source_{i+1}.fits")