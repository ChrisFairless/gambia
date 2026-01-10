from pathlib import Path
import os
import numpy as np
import pycountry
import matplotlib.pyplot as plt
from pathlib import Path
from climada.entity import Exposures
from climada.util.constants import DEF_CRS
from climada.util import coordinates as u_coords
from climada.util.plot import make_map
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx


subplots = (7, 1)
figsize = (12, 20)

country_list = ['Gambia']
exposure_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/livestock_GLW4/exp/')
plot_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/livestock_GLW4/plots/')
overwrite = True
# cmap = 'viridis'
cmap = 'magma_r'

# ------------------------------------------------------------------------------

species_list = ['all', 'buffalo', 'chicken', 'cattle', 'goats', 'pigs', 'sheep']
os.makedirs(plot_dir, exist_ok=True)

for country in country_list:
    print(f'Plotting livestock data for {country}')
    country_iso = pycountry.countries.get(name=country).alpha_3

    plot_file = Path(plot_dir, f'glw4_{country_iso}.png')

    # fig, axes = plt.subplots(
    #     subplots[0], subplots[1],
    #     figsize=figsize,
    #     sharex=True, sharey=True,
    #     # subplot_kw={'projection': ccrs.PlateCarree()}
    # )

    fig, axes, _ = make_map(num_sub=subplots, figsize=figsize)

    axes = axes.flatten()

    for i, species in enumerate(species_list):
        exposure_file = Path(exposure_dir,  f'glw4_{species}_{country_iso}_5as.hdf5')
        if not os.path.exists(exposure_file):
            raise FileNotFoundError(f'No exposure file found at {exposure_file} – please run 02_crop_GLW4_to_country.py first')

        exp = Exposures.from_hdf5(exposure_file)
        if os.path.exists(plot_file) and not overwrite:
            raise FileExistsError(f'Plot file {plot_file} already exists and overwrite is False – skipping processing')
        
        total_value = exp.gdf.value.sum()

        axes[i].add_feature(cfeature.BORDERS)
        axes[i].add_feature(cfeature.COASTLINE)
        _ = exp.plot_basemap(
                    ignore_zero=False,
                    pop_name=False,
                    buffer=20000.0,
                    extend="neither",
                    # url=ctx.providers.OpenStreetMap.Mapnik,
                    zoom=10,
                    title=f'{species.capitalize()}: {np.floor(total_value)} k heads',
                    adapt_fontsize=False,
                    axis=axes[i],
                    cmap=cmap
                )
        axes[i].axis('off')
        axes[i].axis('off')
    
    plt.tight_layout()
    fig.savefig(plot_file)