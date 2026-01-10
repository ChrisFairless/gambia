import pandas as pd
import geopandas as gpd
import os
import copy
import pycountry
from pathlib import Path

import climada.util.coordinates as u_coords
from climada.entity import Exposures

country_list = ['Gambia']
data_dir = '/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/livestock_GLW4/'

# ------------------------------------------------------------------------------

species_list = ['all', 'buffalo', 'chicken', 'cattle', 'goats', 'pigs', 'sheep']

raw_dir = Path(data_dir, 'raw')
out_dir = Path(data_dir, 'exp')
os.makedirs(out_dir, exist_ok=True)

for species in species_list:
    print(f'Working on {species}')
    glw_raw_path = Path(raw_dir, f'glw4_{species}_5as.tif')
    if not os.path.exists(glw_raw_path):
        raise FileNotFoundError(f'No GLW4 data found at {glw_raw_path} – please run 01_download_GLW4_data.py first')


    print('Loading GLW4 data')
    glw = Exposures.from_raster(
            glw_raw_path,
            band=1,
            src_crs='EPSG:4326',
            geometry=None,   # can't get this working for now
            attrs={
                'description': 'GLW4 2020 all livestock combined',
                'ref_year': 2020,
                'value_unit': 'heads per km2'
            }
    )

    for country in country_list:
        country_iso = pycountry.countries.get(name=country).alpha_3
        print(f'Working on {country} {country_iso}')
        geom = u_coords.get_country_geometries(
            country_names = [country_iso],
            # extent = (0, 360, -90, 90),
            # center_crs = True
        ).geometry
        geom_gdf = gpd.GeoDataFrame(geometry=geom)
        geom_bounds = geom_gdf.total_bounds  # minx, miny, maxx, maxy    

        glw_country = copy.deepcopy(glw)
        gdf_country = glw_country.data.clip(mask=geom)
        glw_country.set_gdf(gdf_country)

        out_path = Path(out_dir, f'glw4_{species}_{country_iso}_5as.hdf5')
        glw_country.write_hdf5(out_path)

print("Done")