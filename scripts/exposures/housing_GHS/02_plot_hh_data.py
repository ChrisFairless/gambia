from pathlib import Path
import os
import pycountry
from pathlib import Path
import cartopy.feature as cfeature
from climada.entity import Exposures
from climada.util.constants import DEF_CRS
from climada.util import coordinates as u_coords


country= 'Gambia'
exposure_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/housing_GHS/exp/')
tif_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/housing_GHS/tif/')
plot_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/housing_GHS/plots/')
figsize=(20, 10)
overwrite = True

print(f'Plotting household data for {country}')
country_iso = pycountry.countries.get(name=country).alpha_3

exposure_file = Path(exposure_dir,  f'ghs_hh_{country_iso}.hdf5')
if not os.path.exists(exposure_file):
    raise FileNotFoundError(f'No exposure file found at {exposure_file} – please create this data first')
if not os.path.exists(plot_dir.parent):
    raise FileNotFoundError(f'No output folder {plot_dir.parent} – create this first')


exp = Exposures.from_hdf5(exposure_file)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(tif_dir, exist_ok=True)
plot_file = Path(plot_dir, f'ghs_hh_{country_iso}.png')
tif_file = Path(tif_dir, f'ghs_hh_{country_iso}.tif')
if os.path.exists(plot_file) and not overwrite:
    raise FileExistsError(f'Plot file {plot_file} already exists and overwrite is False – skipping processing')

title = f'GHS households for {country}'

# ax = exp.plot_basemap(
#         ignore_zero=False,
#         pop_name=False,
#         buffer=10000.0,
#         extend="neither",
#         figsize=figsize,
#         title=title
#     )

ax = exp.plot_raster(
        save_tiff=tif_file,
        raster_f=lambda x: x,
        label="Household count",
        figsize=figsize,
        fill=False,
        # title=title
    )
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.COASTLINE)

ax.figure.savefig(plot_file)