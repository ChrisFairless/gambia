from pathlib import Path
import os
import shutil
import pycountry as pc

from nccs.pipeline.direct.direct import get_sector_exposure

download_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/energy_NCCS/raw/')

country_list = ['Gambia']

# ------------------------------------------------------------------------------

if not os.path.exists(download_dir.parent.parent):
    raise FileNotFoundError(f'No data folder at {download_dir.parent.parent} – please create')
os.makedirs(download_dir, exist_ok=True)

print('Copying over NCCS manufacturing exposure data')

for country in country_list:
    print(country)
    country_iso = pc.countries.get(name=country).alpha_3
    exp = get_sector_exposure('energy', country)
    filename = f'energy_nccs_{country_iso}.h5'
    exp.write_hdf5(Path(download_dir, filename))