from pathlib import Path
import os
import shutil
import pycountry as pc
from climada.util.api_client import Client

download_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/services_NCCS/exp/')

country_list = ['Gambia']

services_GDP = {
    # TODO align with other data sources in the project!!
    'Gambia': 0.583 * 2.77 * 1e9  # USD from Wikipedia and IMF
}


# ------------------------------------------------------------------------------

if not os.path.exists(download_dir.parent.parent):
    raise FileNotFoundError(f'No data folder at {download_dir.parent.parent} – please create')
os.makedirs(download_dir, exist_ok=True)

print('Downloading LitPop data to use as services data')

client = Client()

for country in country_list:
    print(country)
    country_iso = pc.countries.get(name=country).alpha_3
    exp = client.get_litpop(country)
    new_gdf = exp.gdf.copy()
    new_gdf['value'] = new_gdf['value'] * services_GDP[country] / new_gdf['value'].sum()
    exp.set_gdf(new_gdf)

    filename = f'services_litpop_{country_iso}.h5'
    exp.write_hdf5(Path(download_dir, filename))