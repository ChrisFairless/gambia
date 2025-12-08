from pathlib import Path
import os
import pycountry
from pathlib import Path
import cartopy.feature as cfeature
from climada.entity import Exposures
from climada.util.constants import DEF_CRS
from climada.util import coordinates as u_coords


country = 'Gambia'
hh_size = 9.5

print(f'Creating household data for {country}')
country_iso = pycountry.countries.get(name=country).alpha_3
exposure_pop_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/population_GHS/exp/')
exposure_hh_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/housing_GHS/exp/')
exposure_pop_file = Path(exposure_pop_dir,  f'ghs_pop_{country_iso}.hdf5')
exposure_hh_file = Path(exposure_hh_dir,  f'ghs_hh_{country_iso}.hdf5')

if not os.path.exists(exposure_pop_file):
    raise FileNotFoundError(f'No exposure file found at {exposure_pop_file} – please run the population_GHS scripts first')
os.makedirs(exposure_hh_dir, exist_ok=True)

exp_pop = Exposures.from_hdf5(exposure_pop_file)
new_gdf = exp_pop.gdf.copy()
new_gdf['value'] = new_gdf['value'] / hh_size

exp_hh = Exposures(
    data=new_gdf,
    description=f'Household exposure for {country}',
    value_unit='number of households'
)

exp_hh.write_hdf5(exposure_hh_file)