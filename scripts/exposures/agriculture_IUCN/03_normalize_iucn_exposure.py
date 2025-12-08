import pandas as pd
import numpy as np
import geopandas as gpd
import os
import copy
import pycountry
from pathlib import Path

from climada.entity import Exposures

country_list = ['Gambia']
data_dir = '/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/agriculture_IUCN/'

# TODO move this into a data file
# figures from wikipedia plus some guesses: look into this more carefully
target_values = {
    'agriculture': 0.15 * 2.77e9,  # 15% of GDP
    'arable': 0.15 * 2.77e9
}
# Alternately normalise to total area which is 2427 km2

# ------------------------------------------------------------------------------

exp_dir = Path(data_dir, 'exp')


for land_use, target in target_values.items():
    print(f'Renormalising land use: {land_use}')

    for country in country_list:
        country_iso = pycountry.countries.get(name=country).alpha_3
        exp_path = Path(exp_dir, f'iucn_{land_use}_{country_iso}.hdf5')
        exp = Exposures.from_hdf5(exp_path)
        new_gdf = copy.deepcopy(exp.gdf)
        new_gdf['value'] = new_gdf['value'] / new_gdf['value'].sum() * target
        exp.set_gdf(new_gdf)
        exp.write_hdf5(exp_path)

print("Done")