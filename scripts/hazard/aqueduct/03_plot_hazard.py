import pycountry as pc
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path

from climada.hazard import Hazard
from climada.util.plot import plot_from_gdf

# Loop through downloaded Aqueduct datasets and plot

country_list = ['Gambia']       # Set to None to plot all countries in the folder

data_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/hazard/aqueduct/clipped/')
plot_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/hazard/aqueduct/plots/')

overwrite = True
figsize = (27, 13)

# --------------------------------------------------------------

os.makedirs(plot_dir, exist_ok=True)

if not country_list:
    country_list = [p.stem.split('_')[0] for p in data_dir.glob('*.hdf5')]

country_iso_list = [pc.countries.get(name=country).alpha_3 for country in country_list]

for country, country_iso in zip(country_list, country_iso_list):
    file_list = list(data_dir.glob(f'{country_iso}*.hdf5'))
    for file in file_list:
        out_filename = f'{file.stem}.png'
        if not overwrite and os.path.exists(Path(plot_dir, out_filename)):
            print(f'Plots already exist: skipping {out_filename}')
            continue
        print(f'Plotting {file.stem}')
        _, _, scenario, model, year = file.stem.split('_')
        haz = Hazard.from_hdf5(file)
        n_events = len(haz.event_id)
        rps = [int(f) for f in haz.event_id]
        n_rps = len(set(haz.frequency))

        if model != 'ALL':
            print(f'n_events: {n_events}')
            fig, axs = plt.subplots(
                nrows=int(np.ceil(n_events/2)), ncols=2,
                sharex=True, sharey=True,
                figsize=figsize,
                subplot_kw={'projection': ccrs.PlateCarree()}
            )

            axs = axs.flatten()
            for i, (event_name, ax) in enumerate(zip(haz.event_name, axs)):
                print(f'event: {event_name}')
                # im_val = haz.intensity[i, :].toarray().transpose()
                # alpha = np.where(np.isnan(im_val), 0, np.where(im_val == 0, 0, 1)) 
                # if np.all(np.isnan(im_val)):
                #     print("Only missing values: skipping")
                #     continue
                ax = haz.plot_intensity(
                        event=event_name,
                        axis=ax,
                        # alpha=alpha
                    )
            fig.suptitle(f'''
                Aqueduct {year} river flood depth for
                {country} under {scenario} scenario,
                {model} model
                ''')
            plt.tight_layout()
            plt.savefig(Path(plot_dir, out_filename))

        else:
            fig, axs = plt.subplots(
                nrows=1, ncols=1,
                figsize=figsize,
                subplot_kw={'projection': ccrs.PlateCarree()}
            )
            gdf, label, column_label = haz.local_exceedance_intensity(return_periods=rps)
            axs = plot_from_gdf(
                gdf=gdf,
                title_subplots=column_label,
                axis=axs
            )
            fig.suptitle(f'''
                Aqueduct {year} expected river flood depths for
                {country} under {scenario} scenario, across all models
                ''')
            plt.tight_layout()
            plt.savefig(Path(plot_dir, out_filename))
