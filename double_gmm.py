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

EVT_FILE = '3392/repro/acisf03392_repro_evt2.fits' #'acisi_merged.fits'
XMIN = 4110 #4085
XMAX = 4150 #4120
YMIN = 4050 #4080
YMAX = 4110 #4120

EMIN = 300
EMAX = 8000

VMIN = 0.5
VMAX = 1e3

BINX = 64
BINY = 64
BINE = 60

VERBOSE = True

# num_lvl + num_start = 5
NUM_LVL = 2
LVL_START = 2

NB_SOURCE = 4

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
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)

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
    probs, labels, centers, std_devs = gmm_fitting(nb_source, table=table_sources, e_lvls=["energy"])
    sources = []
    for i in range(nb_source):
        source_table = table_sources[labels == i][['energy', 'x', 'y']].copy()
        source_table['weight'] = probs[labels == i, i] * table_sources[labels == i]['weight']
        sources.append(source_table)
    return sources, centers, std_devs

def save_df_as_fits(source_df, filename, template_evt_file):
    astropy_table = Table.from_pandas(source_df[['energy', 'x', 'y', 'weight']])
    hdr0 = fits.getheader(template_evt_file, 0)
    hdr1 = fits.getheader(template_evt_file, 1).copy()
    
    new_hdr1 = hdr1.copy()
    for key in list(new_hdr1.keys()):
        if any(key.startswith(p) and key[len(p):].isdigit() for p in ['TTYPE', 'TFORM', 'TUNIT', 'TDISP', 'TDIM', 'TNULL', 'TLMIN', 'TLMAX', 'TCRVL', 'TCRPX', 'TCDLT', 'TCTYP', 'TCUNI']):
            del new_hdr1[key]
    
    mapping = {
        '2': '11', # Map new X (2) to old X (11)
        '3': '12'  # Map new Y (3) to old Y (12)
    }
    
    wcs_prefixes = ['TCRVL', 'TCRPX', 'TCDLT', 'TCTYP', 'TCUNI', 'TLMIN', 'TLMAX']
    
    for new_col, old_col in mapping.items():
        for pre in wcs_prefixes:
            old_key = f"{pre}{old_col}"
            if old_key in hdr1:
                new_hdr1[f"{pre}{new_col}"] = hdr1[old_key]
    
    primary_hdu = fits.PrimaryHDU(header=hdr0)
    evt_hdu = fits.BinTableHDU(data=astropy_table, header=new_hdr1)
    evt_hdu.name = "EVENTS"
    hdul = fits.HDUList([primary_hdu, evt_hdu])
    hdul.writeto(f'output/{filename}', overwrite=True)

def final_wcs_injection(target_fits, wcs_obj):
    """
    Manually 'wires' the WCS object into the target FITS table 
    specifically for CIAO specextract compatibility.
    """
    with fits.open(target_fits, mode='update') as hdul:
        # Access the EVENTS table (usually index 1)
        hdr = hdul['EVENTS'].header
        
        # 1. Map the RA/Dec center (CRVAL)
        # wcs_obj.wcs.crval[0] is RA, [1] is Dec
        hdr['CRVAL2'] = wcs_obj.wcs.crval[0]
        hdr['CRVAL3'] = wcs_obj.wcs.crval[1]
        
        # 2. Map the Pixel Center (CRPIX)
        # Chandra physical coords are usually centered at 4096.5
        hdr['CRPIX2'] = wcs_obj.wcs.crpix[0]
        hdr['CRPIX3'] = wcs_obj.wcs.crpix[1]
        
        # 3. Map the Pixel Scale (CDELT)
        # Standard ACIS is ~0.492 arcsec/pixel => 0.00013666 deg
        cdelt = wcs_obj.wcs.get_cdelt()
        hdr['CDELT2'] = cdelt[0]
        hdr['CDELT3'] = cdelt[1]
        
        # 4. Mandatory CIAO "Handshake" Keywords
        # This tells specextract: 'Columns 2 and 3 are the Sky'
        hdr['CTYPE2'] = 'x'
        hdr['CTYPE3'] = 'y'
        hdr['WCSTY2'] = 'PHYSICAL'
        hdr['WCSTY3'] = 'PHYSICAL'
        
        # This points to the Coordinate System defined in ObsID 3392
        hdr['ACSYS4'] = 'SKY:ASC-FP-1.1'
        
        # 5. The "Security Seal" (Data Subspace)
        # This is the final step to make dmcoords work
        hdr['DSTYP1'] = 'time'
        hdr['DSVAL1'] = 'TABLE'
        hdr['DSTYP2'] = 'x,y'
        hdr['DSVAL2'] = 'specnum:1'
        
        hdul.flush()
    print("WCS successfully wired to Columns 2 and 3.")

cube, e_lvls = starlet_cube(subset)
table_bg, table_sources = bg_fit(e_lvls = ['energy', 'starlet_0'])
split_sources, centers, std_dev = source_fit(table_sources, NB_SOURCE)
bg_split_sources, cntrs, stddv = source_fit(table_bg, NB_SOURCE)
plot_split([*split_sources, table_bg], f"2_split_{NB_SOURCE}sources.png")
plot_split(bg_split_sources, f"2_split_{NB_SOURCE}bg.png")

with fits.open(EVT_FILE) as hdul:
    solved_wcs = pywcs.WCS(hdul[1].header)

for i, source in enumerate([*split_sources, table_bg]):
    save_df_as_fits(source, f"source_{i}.fits", EVT_FILE)
    final_wcs_injection(f"output/source_{i}.fits", solved_wcs)