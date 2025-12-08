import pandas as pd
import numpy as np
import geopandas as gpd
import os
import copy
import pycountry
import rasterio
from pathlib import Path

import climada.util.coordinates as u_coords
from climada.entity import Exposures

from climada_gambia.utils_raster import cropped_exposure_from_tifs

country_list = ['Gambia']
data_dir = '/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/agriculture_IUCN/'

write_csv_too = True

# ------------------------------------------------------------------------------

exposures_dict = {
    'arable': {
        'types': ['Arable Land'],
        'ids': [1401],
    },
    'pasture': {
        'types': ['Pastureland'],
        'ids': [1402],
    },
    'plantation': {
        'types': ['Plantations'],
        'ids': [1403]
    },
    'agriculture': {
        'types': ['Arable Land', 'Plantations'],
        'ids': [1401, 1403]
    }
}

raw_dir = Path(data_dir, 'raw')
out_dir = Path(data_dir, 'exp')
csv_dir = Path(data_dir, 'csv')

if not os.path.exists(raw_dir):
    raise FileNotFoundError(f'Not found: please create and add data to {raw_dir}')

os.makedirs(out_dir, exist_ok=True)
if write_csv_too:
    os.makedirs(csv_dir, exist_ok=True)


for land_use, info in exposures_dict.items():
    print(f'Working on land use: {land_use}')
    raw_path_list = [
        Path(raw_dir, f'iucn_habitatclassification_fraction_lvl2__{id}_{t}__ver004.tif')
        for id, t in zip(info['ids'], info['types'])
    ]
    for p in raw_path_list:
        if not os.path.exists(p):
            raise FileNotFoundError(f'No IUCN data found at {p} – please download it first')

    attrs = {
        'description': f'IUCN land cover fraction for {land_use}',
        'value_unit': 'area (dimensionless)'
        }

    for country in country_list:
        country_iso = pycountry.countries.get(name=country).alpha_3
        exp = cropped_exposure_from_tifs(raw_path_list, country, drop_zeros=True, band=1, attrs=attrs, verbose=True)

        out_path = Path(out_dir, f'iucn_{land_use}_{country_iso}.hdf5')
        exp.write_hdf5(out_path)

        if write_csv_too:
            csv_path = Path(csv_dir, f'iucn_{land_use}_{country_iso}.csv')
            exp.gdf.to_csv(csv_path)

print("Done")