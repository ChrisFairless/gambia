from pathlib import Path
import os
import shutil
import numpy as np
import geopandas as gpd
import pycountry as pc
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from climada.entity import Exposures
from climada.hazard import Centroids
import climada.util.coordinates as u_coord

# Copied from the other downscaling scripts. TODO make a method for this

data_root_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/manufacturing_NCCS/')

regrid_mask_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/population_GHS/exp/')

country_list = ['Gambia']  # Set lists of countries to combine the exposures

overwrite = True
write_csv_too = True

threshold = 0.14  # set to none to estimate automatically. Relies on a few grid points being present
# Underlying data is 0.1 degree resolution so we set it so root 2 x 0.1

# ------------------------------------------------------------------------------

hdf5_dir = Path(data_root_dir, 'raw/')
out_dir = Path(data_root_dir, 'exp/')
csv_dir = Path(data_root_dir, 'csv/')

if not os.path.exists(hdf5_dir):
    raise FileNotFoundError(f'No data folder at {hdf5_dir} – please create')

os.makedirs(out_dir, exist_ok=True)
if write_csv_too:
    os.makedirs(csv_dir, exist_ok=True)

print('Downscaling NCCS manufacturing exposures to populated areas.')  # Best we can do for now, sorry

for country in country_list:
    print(f'Working on {country}')

    country_iso = pc.countries.get(name=country).alpha_3
    
    hdf5_filename = f'global_noxemissions_2011_above_100t_0.1deg_ISO3_values_Manfac_scaled_{country_iso}.h5'
    mask_filename = f'ghs_pop_{country_iso}.hdf5'
    out_filename = f'manufacturing_nccs_downscaled_{country_iso}.hdf5'
    csv_filename = f'manufacturing_nccs_downscaled_{country_iso}.csv'

    hdf5_path = Path(hdf5_dir, hdf5_filename)
    mask_path = Path(regrid_mask_dir, mask_filename)
    out_path = Path(out_dir, out_filename)
    csv_path = Path(csv_dir, csv_filename)

    if os.path.exists(out_path) and not overwrite:
        print('Downscaled exposure already exists, skipping')
        continue

    print('Reading manufacturing data')
    exp = Exposures.from_hdf5(hdf5_path)
    print('Reading population data')
    mask = Exposures.from_hdf5(mask_path)
    mask_gdf = mask.gdf
    mask_gdf.value = np.nan

    print('Regridding to GHS grid and masking')
    if threshold is None:
        threshold = u_coord.estimate_matching_threshold(np.stack([exp.gdf.geometry.y.values, exp.gdf.geometry.x.values], axis=1)) * np.power(2, 0.5)  # Good enough for now
    idx = u_coord.match_coordinates(
        np.stack([mask_gdf.geometry.y.values, mask_gdf.geometry.x.values], axis=1),
        np.stack([exp.gdf.geometry.y.values, exp.gdf.geometry.x.values], axis=1),
        distance="approx",
        threshold=threshold
    )
    mask_gdf['centr'] = idx
    mask_gdf['count'] = mask_gdf.groupby(idx)['centr'].transform('count')
    mask_gdf = mask_gdf[idx >= 0]
    mask_gdf['raw_value'] = exp.gdf['value'].iloc[mask_gdf['centr']].values
    mask_gdf['value'] = mask_gdf['raw_value'] / mask_gdf['count']
    mask_gdf = gpd.GeoDataFrame(mask_gdf[['region_id', 'value', 'geometry']])

    assert(np.abs(mask_gdf.value.sum() - exp.gdf.value.sum()) < 1)
    mask.set_gdf(mask_gdf)

    print('Writing outputs')
    mask.write_hdf5(out_path)

    if write_csv_too:
        mask.gdf.to_csv(csv_path)


