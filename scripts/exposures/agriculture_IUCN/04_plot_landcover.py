from pathlib import Path
import os
import shutil
import numpy as np
import pycountry as pc
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from climada.entity import Exposures
from climada.hazard import Centroids
import climada.util.coordinates as u_coord

data_root_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/agriculture_IUCN/')

country_list = ['Gambia']

overwrite = True

figsize = (27, 13)
zoom = 10

# ------------------------------------------------------------------------------

hdf5_dir = Path(data_root_dir, 'exp/')
tif_dir = Path(data_root_dir, 'tif/')
plot_dir = Path(data_root_dir, 'plots/')

if not os.path.exists(hdf5_dir):
    raise FileNotFoundError(f'No data folder at {hdf5_dir} – please create')

os.makedirs(tif_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

print('Converting IUCN land cover exposure data to tif and plotting')

exposure_file_list = os.listdir(hdf5_dir)

fig, axes = plt.subplots(
    nrows=len(exposure_file_list), ncols=1,
    figsize=figsize,
    subplot_kw={'projection': ccrs.PlateCarree()}
)
axes = axes.flatten()

for i, filename in enumerate(exposure_file_list):
    print(f'Working on {filename}')

    file_path = Path(hdf5_dir, filename)
    tif_path = Path(tif_dir, f"{file_path.stem}.tif")
    plot_path = Path(plot_dir, f"{file_path.stem}.png")

    if os.path.exists(plot_path) and not overwrite:
        print('Plot already exists, skipping')
        continue

    print('... reading data')
    exp = Exposures.from_hdf5(file_path)
    n_points = exp.gdf.shape[0]
    if n_points > 0:
        print(f'... plotting {n_points} exposure points')
        axes[i] = exp.plot_raster(
            save_tiff=tif_path,
            raster_f=lambda x: x,
            label=file_path.stem,
            axis=axes[i],
            fill=False,
            adapt_fontsize=True
        )

fig.savefig(plot_path, dpi=300)
