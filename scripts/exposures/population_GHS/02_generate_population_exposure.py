from pathlib import Path
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pycountry
import json
from pathlib import Path

from climada.entity import Exposures
from climada.util.constants import DEF_CRS
from climada.util import coordinates as u_coords

from climada_gambia.utils_raster import cropped_exposure_from_tifs
from climada_gambia.data.exposure_totals import exposure_totals

# Crude script to generate population data from the Gridded Population of the World dataset.

country = 'Gambia'
download_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/population_GHS/raw/')
output_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/population_GHS/exp/')
file_list = []      # Leave empty to loop through all files. Otherwise give a list to speed things up.
overwrite = True

exp_units = "people"

# ------------------------------------------------------------------------------

print(f'Generating population data for {country}')

target_population = exposure_totals['population'] 

if not os.path.exists(download_dir):
    raise FileNotFoundError(f'Download directory {download_dir} does not exist')
if not os.path.exists(output_dir.parent):
    raise FileNotFoundError(f'Output directory {output_dir.parent} does not exist')
os.makedirs(output_dir, exist_ok=True)

if len(file_list) == 0:
    file_list = list(download_dir.glob('*.tif'))

if len(file_list) == 0:
    raise FileNotFoundError(f'No .tif files found in {download_dir}')

filename_roots = set(['_'.join(f.stem.split('_')[0:-2]) for f in file_list])
if len(filename_roots) > 1:
    raise ValueError(f'Multiple different datasets appear to be in the folder – please check: {filename_roots}')

country_iso = pycountry.countries.get(name=country).alpha_3
country_num = int(pycountry.countries.get(name=country).numeric)
out_path = Path(output_dir,  f'ghs_pop_{country_iso}.hdf5')
if os.path.exists(out_path) and not overwrite:
    raise FileExistsError(f'Output file {out_path} already exists and overwrite is False – skipping processing')



source_crs = file_list[0].stem.split('_')[5]
source_crs = f'EPSG:{source_crs}'
    
exp = cropped_exposure_from_tifs(
    filepath_list=file_list,
    country=country,
    drop_zeros=True,
    band=1,
    src_crs=source_crs,
    attrs={'value_unit': exp_units},
    verbose=True
)

if exp.gdf.shape[0] == 0:
    print('... no non-zero population values found')
else:
    total_value = exp.value.sum()
    scale = target_population / total_value
    print(f"Scaling exposure by {scale} to match the target value")
    new_gdf = exp.gdf
    new_gdf.value = new_gdf.value * scale
    exp.set_gdf(new_gdf)      

print(f'Writing to {out_path}')
exp.write_hdf5(out_path)
