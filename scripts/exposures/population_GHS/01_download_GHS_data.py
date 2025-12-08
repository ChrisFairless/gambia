import requests
import os
from pathlib import Path

from climada.entity.exposures.litpop import get_gpw_file_path

# We'll use the Global Human Settlement Layer to get population counts

projection = 'WGS84'  # 'Mollweide' or 'WGS84'
resolution = 3  # Values are 100, 1000 (metres) for Mollweide, and 3 or 30 (arcseconds) for WGS84

tile_list = ['R8_C17']   # Tile covering Gambia.
                  # See the GHS site here to get the right tile: https://human-settlement.emergency.copernicus.eu/download.php
                  # alternately, edit this script to work with a download of the whole globe

download_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/population_GHS/raw/')
overwrite = False

if projection == 'WGS84':
    crs = '4326'
    resolution = str(resolution) + 'ss'
if projection == 'Mollweide':
    crs = '54009'
    resolution = str(resolution)

url_base = 'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/'
geometry_folder = f'GHS_POP_E2020_GLOBE_R2023A_{crs}_{resolution}/'
tiles_folder = 'V1-0/tiles/'

if not os.path.exists(download_dir):
    raise FileNotFoundError(f'No download folder at {download_dir} – please create')

for tile in tile_list:
    file_name_root = f'GHS_POP_E2020_GLOBE_R2023A_{crs}_{resolution}_V1_0_{tile}'
    file_name_zip = file_name_root + '.zip'
    file_name_tif = file_name_root + '.tif' 
    url = url_base + geometry_folder + tiles_folder + file_name_zip
    out_path_zip = Path(download_dir, file_name_zip)
    out_path_tif = Path(download_dir, file_name_tif)

    if os.path.exists(out_path_tif) and not overwrite:
        print(f'File already exists, skipping download: {file_name_zip}')
        continue

    if os.path.exists(out_path_zip):
        print(f'Downloaded zip already exists, skipping download: {file_name_zip}')
    else:
        try:
            response = requests.get(url)
            print(f'Downloading {file_name_zip}')
            if response.ok:
                with open(out_path, mode="wb") as f:
                    f.write(response.content)
        except Exception as e:
            print(response.content)
            print(f'Error downloading {file_name_zip}: {e}')
            continue

    try:
        os.system(f"unzip -o {out_path} -d {download_dir}")
        os.remove(out_path)
    except Exception as e:
        print(f'Error unzipping {file_name}: {e}')
        continue


