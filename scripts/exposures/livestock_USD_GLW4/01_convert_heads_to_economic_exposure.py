import pandas as pd
import numpy as np
import geopandas as gpd
import os
import copy
import pycountry
from pathlib import Path

from climada.entity import Exposures

country_list = ['Gambia']
data_dir = '/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/livestock_heads_GLW4/exp/'
output_dir = '/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/livestock_USD_GLW4/exp/'

# TODO move this into a data file
# figures from wikipedia plus some guesses: look into this more carefully
target_values = {
    'all': 0.15 * 2.77e9
}

# ------------------------------------------------------------------------------

if not os.path.exists(data_dir):
    raise FileNotFoundError(f'No data folder at {data_dir} – please create')
os.makedirs(output_dir, exist_ok=True)

for exposure_type, target in target_values.items():
    print(f'Creating economic exposure layer for livestock: {exposure_type}')

    for country in country_list:
        country_iso = pycountry.countries.get(name=country).alpha_3
        exp_in_path = Path(data_dir, f'glw4_{exposure_type}_{country_iso}_5as.hdf5')
        exp_out_path = Path(output_dir, f'glw4_USD_{exposure_type}_{country_iso}_5as.hdf5')
        exp = Exposures.from_hdf5(exp_in_path)
        new_gdf = copy.deepcopy(exp.gdf)
        new_gdf['value'] = new_gdf['value'] / new_gdf['value'].sum() * target

        output = Exposures(
            data=new_gdf,
            description=f'GLW4 {exposure_type} livestock economic exposure for {country}',
            value_unit='USD'
        )
        exp.write_hdf5(exp_out_path)

print("Done")