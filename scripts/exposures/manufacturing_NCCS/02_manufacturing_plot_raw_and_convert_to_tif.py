from pathlib import Path
import os
import shutil
import pycountry as pc
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from climada.entity import Exposures


data_root_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/manufacturing_NCCS/')

country_list = [['Gambia', 'Senegal']]  # Set lists of countries to combine the exposures

figsize=(9, 13)

# ------------------------------------------------------------------------------

hdf5_dir = Path(data_root_dir, 'raw/')
tif_dir = Path(data_root_dir, 'tif_raw/')
plot_dir = Path(data_root_dir, 'plots_raw/')

if not os.path.exists(hdf5_dir):
    raise FileNotFoundError(f'No data folder at {hdf5_dir} – please create')

os.makedirs(tif_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

print('Converting NCCS manufacturing exposure data to tif')

for countries in country_list:
    print(countries)

    country_iso_list = [pc.countries.get(name=country).alpha_3 for country in countries]
    country_iso_string = '_'.join(country_iso_list)
    
    file_root = 'global_noxemissions_2011_above_100t_0.1deg_ISO3_values_Manfac_scaled'
    hdf5_filename_list = [f'{file_root}_{iso}.h5' for iso in country_iso_list]
    png_filename_list = [f'{file_root}_{iso}.png' for iso in country_iso_list]
    tif_filename_list = [f'{file_root}_{iso}.tif' for iso in country_iso_list]
    combined_hdf5_filename = f'manufacturing_combined_{country_iso_string}.h5'
    combined_tif_filename = f'manufacturing_combined_{country_iso_string}.tif'
    combined_png_filename = f'manufacturing_combined_{country_iso_string}.png'

    exp_list = []
    for country, hdf5_filename, tif_filename, png_filename in zip(countries, hdf5_filename_list, tif_filename_list, png_filename_list):
        print(country)
        hdf5_path = Path(hdf5_dir, hdf5_filename)
        tif_path = Path(tif_dir, tif_filename)
        png_path = Path(plot_dir, png_filename)

        exp = Exposures.from_hdf5(hdf5_path)

        fig, ax = plt.subplots(
                nrows=1, ncols=1,
                figsize=figsize,
                subplot_kw={'projection': ccrs.PlateCarree()}
            )

        ax = exp.plot_raster(
            save_tiff=tif_path,
            raster_f=lambda x: x,
            label="Manufacturing production (USD)",
            axis=ax,
            fill=False,
            adapt_fontsize=True,
        )
        fig.savefig(png_path, dpi=300)

        exp_list.append(exp)
    
    if len(exp_list) > 1:
        print(f'Combining exposures for {", ".join(countries)}')
        combined_exp = Exposures.concat(exp_list)
        combined_hdf5_path = Path(hdf5_dir, combined_hdf5_filename)
        combined_tif_path = Path(tif_dir, combined_tif_filename)
        combined_png_path = Path(plot_dir, combined_png_filename)
        combined_exp.write_hdf5(combined_hdf5_path)

        fig, ax = plt.subplots(
                nrows=1, ncols=1,
                figsize=figsize,
                subplot_kw={'projection': ccrs.PlateCarree()}
            )

        ax = combined_exp.plot_raster(
            save_tiff=combined_tif_path,
            raster_f=lambda x: x,
            label="Manufacturing production (USD)",
            axis=ax,
            figsize=figsize,
            fill=False,
            adapt_fontsize=True,
        )
        fig.savefig(combined_png_path, dpi=300)

    