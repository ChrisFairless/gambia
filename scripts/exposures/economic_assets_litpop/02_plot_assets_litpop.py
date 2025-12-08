from pathlib import Path
import os
import shutil
import pycountry as pc
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from climada.entity import Exposures

data_root_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/economic_assets_litpop/')

country_list = ['Gambia']  # Set lists of countries to combine the exposures

figsize=(9, 13)

overwrite = True

# ------------------------------------------------------------------------------

hdf5_dir = Path(data_root_dir, 'exp/')
tif_dir = Path(data_root_dir, 'tif/')
plot_dir = Path(data_root_dir, 'plots/')

if not os.path.exists(hdf5_dir):
    raise FileNotFoundError(f'No data folder at {hdf5_dir} – please create')

os.makedirs(tif_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

print('Converting LitPop exposure data to tif and plotting')

for country in country_list:
    print(country)

    country_iso = pc.countries.get(name=country).alpha_3
    
    hdf5_filename = f'economic_assets_litpop_{country_iso}.h5'
    tif_filename = f'economic_assets_litpop_{country_iso}.tif'
    png_filename = f'economic_assets_litpop_{country_iso}.png'

    hdf5_path = Path(hdf5_dir, hdf5_filename)
    tif_path = Path(tif_dir, tif_filename)
    png_path = Path(plot_dir, png_filename)

    if os.path.exists(png_path) and not overwrite:
        print('Plot already exists, skipping')
        continue

    exp = Exposures.from_hdf5(hdf5_path)

    fig, ax = plt.subplots(
            nrows=1, ncols=1,
            figsize=figsize,
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

    ax = exp.plot_raster(
        save_tiff=tif_path,
        raster_f=lambda x: x,
        label="Economic assets (USD)",
        axis=ax,
        fill=False,
        adapt_fontsize=True,
    )
    fig.savefig(png_path, dpi=300)
