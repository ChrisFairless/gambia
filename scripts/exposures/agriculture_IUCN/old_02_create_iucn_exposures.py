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

from utils_raster import cropped_exposure_from_tifs

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
        country_num = int(pycountry.countries.get(name=country).numeric)
        print(f'Working on {country} {country_iso} {country_num}')

        geom = u_coords.get_country_geometries(
            country_names = [country_iso],
            # extent = (0, 360, -90, 90),
            # center_crs = True
        ).geometry
        geom_gdf = gpd.GeoDataFrame(geometry=geom)
        geom_bounds = geom_gdf.total_bounds  # minx, miny, maxx, maxy    
        
        exp_list = []
        for p in raw_path_list:
            print(f"Working on {p.name}")
            print("Creating rasterio Window")
            with rasterio.open(p) as src:
                win = from_bounds(geom_bounds[0], geom_bounds[1], geom_bounds[2], geom_bounds[3],
                                  transform=src.transform)
                # expand window to integer offsets/lengths
                win = win.round_offsets().round_lengths()
        
            print('Loading IUCN data')


            this_exp = Exposures.from_raster(
                    p,
                    band=1,
                    src_crs='EPSG:4326',
                    window=win,
                    attrs=attrs
                )
            exp_list.append(this_exp)

        print("Concatenating")
        exp = Exposures.concat(exp_list)
        if len(raw_path_list) > 1:
            new_gdf = exp.gdf.reset_index().groupby('geometry', as_index=False).agg('sum')
            new_gdf = gpd.GeoDataFrame(new_gdf)
            exp.set_gdf(new_gdf)

        print('Masking to country')
        n_exp_before = exp.gdf.shape[0]
        print(f'Starting with {n_exp_before} exposure points')
        chunk_size = 1000000
        n_chunks = int(np.ceil(n_exp_before / chunk_size))
        filtered_chunks = []
        countries_present = set()

        for i in range(n_chunks):
            print(f'... processing chunk {i+1} of {n_chunks}')
            chunk = exp.gdf.iloc[i*chunk_size:(i+1)*chunk_size].copy()
            chunk["region_id"] = u_coords.get_country_code(
                chunk.geometry.y,
                chunk.geometry.x,
                gridded=True
            )
            chunk_countries_present = set(chunk['region_id'].values)
            countries_present = countries_present.union(chunk_countries_present)
            chunk = chunk[chunk['region_id'] == country_num]
            filtered_chunks.append(chunk)

        exp.set_gdf(pd.concat(filtered_chunks, ignore_index=True))
        n_exp_after = exp.gdf.shape[0]
        print(f'... removed {n_exp_before - n_exp_after} out of country points, {n_exp_after} points remain')
        # print(f'... countries present before: {countries_present}.')
        n_exp_before = n_exp_after

        print('Removing zero values')
        exp.set_gdf(exp.gdf[exp.gdf['value'] != 0])
        n_exp_after = exp.gdf.shape[0]
        print(f'... removed {n_exp_before - n_exp_after} zero-value points, {n_exp_after} points remain')
        n_exp_before = n_exp_after

        out_path = Path(out_dir, f'iucn_{land_use}_{country_iso}.hdf5')
        exp.write_hdf5(out_path)

        if write_csv_too:
            csv_path = Path(csv_dir, f'iucn_{land_use}_{country_iso}.csv')
            exp.gdf.to_csv(csv_path)

print("Done")