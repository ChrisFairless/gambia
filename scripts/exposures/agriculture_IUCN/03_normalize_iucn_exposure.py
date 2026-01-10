import pandas as pd
import numpy as np
import geopandas as gpd
import os
import copy
import pycountry
from pathlib import Path

from climada.entity import Exposures
from climada_gambia.utils_total_exposed_value import get_total_exposed_value

country_list = ['Gambia']
data_dir = '/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/agriculture_IUCN/'

# ------------------------------------------------------------------------------

exp_dir = Path(data_dir, 'exp')

total_exposed = get_total_exposed_value(exposure_type='agriculture', usd=True)
target_values = {
    'agriculture': total_exposed,
    'arable': total_exposed
}

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