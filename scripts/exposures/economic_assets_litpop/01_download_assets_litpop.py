from pathlib import Path
import os
import shutil
import pycountry as pc
from climada.util.api_client import Client

download_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/economic_assets_litpop/exp/')

country_list = ['Gambia']



# ------------------------------------------------------------------------------

if not os.path.exists(download_dir.parent.parent):
    raise FileNotFoundError(f'No data folder at {download_dir.parent.parent} – please create')
os.makedirs(download_dir, exist_ok=True)

print('Downloading LitPop to use as asset data')

client = Client()

for country in country_list:
    print(country)
    country_iso = pc.countries.get(name=country).alpha_3
    exp = client.get_litpop(country)

    filename = f'economic_assets_litpop_{country_iso}.h5'
    exp.write_hdf5(Path(download_dir, filename))