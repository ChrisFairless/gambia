from pathlib import Path
import os
import shutil
import pycountry as pc

from nccs.pipeline.direct.direct import get_sector_exposure

download_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/manufacturing_NCCS/hdf5/')

country_list = ['Gambia', 'Senegal']

# ------------------------------------------------------------------------------

if not os.path.exists(download_dir.parent.parent):
    raise FileNotFoundError(f'No data folder at {download_dir.parent.parent} – please create')
os.makedirs(download_dir, exist_ok=True)

print('Downloading NCCS manufacturing exposure data')

for country in country_list:
    print(country)
    country_iso = pc.countries.get(name=country).alpha_3
    exp = get_sector_exposure('manufacturing', country)
    filename = f'global_noxemissions_2011_above_100t_0.1deg_ISO3_values_Manfac_scaled_{country_iso}.h5'
    exp.write_hdf5(Path(download_dir, filename))