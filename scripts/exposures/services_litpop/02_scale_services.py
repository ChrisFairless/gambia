import pandas as pd
import numpy as np
import geopandas as gpd
import os
import copy
import pycountry
from pathlib import Path

from climada.entity import Exposures

country_list = ['Gambia']
data_dir = '/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/services_NCCS/exp/'

# TODO move this into a data file
# figures from wikipedia, look into this more carefully
target_value = 0.19 * 2.77e9

# ------------------------------------------------------------------------------

exp_dir = Path(data_dir, 'exp')

print(f'Renormalising services')

for country in country_list:
    country_iso = pycountry.countries.get(name=country).alpha_3
    exp_path = Path(data_dir, f'services_litpop_{country_iso}.h5')
    exp = Exposures.from_hdf5(exp_path)
    new_gdf = copy.deepcopy(exp.gdf)
    new_gdf['value'] = new_gdf['value'] / new_gdf['value'].sum() * target_value
    exp.set_gdf(new_gdf)
    exp.write_hdf5(exp_path)

print("Done")