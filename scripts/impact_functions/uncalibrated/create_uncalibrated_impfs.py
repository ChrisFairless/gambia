import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from climada.entity import ImpactFunc, ImpactFuncSet
from climada_petals.entity.impact_funcs.river_flood import ImpfRiverFlood, SECTOR_CO_ID, REGION_CO_ID


# Create first-guess impact functions
data_dir = '/Users/chrisfairless/Data/UNU/gambia2025/impact_functions/uncalibrated/'

# Copied from CLIMADA Petals for reference

# SECTOR_CO_ID = {
#     "Residential": 1,
#     "Commercial": 2,
#     "Industrial": 3,
#     "Transport": 4,
#     "Infrastructure": 5,
#     "Agriculture": 6,
# }
# REGION_CO_ID = {
#     "africa": 10,
#     "asia": 20,
#     "europe": 30,
#     "northamerica": 40,
#     "oceania": 50,
#     "southamerica": 60,
# }
# VALID_REGIONS = "Africa, Asia, Europe, North America, Oceania, South America"

sector_map_dict = {
    "housing": "residential",
    "manufacturing": "industrial",
    "services": "residential",
    "agriculture": "agriculture",
    "livestock": "agriculture",
    "energy": "industrial",
    "roads": "industrial"
}

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,10))
axs = axs.flatten()

for i, (sector, sector_map) in enumerate(sector_map_dict.items()):
    print(f'Creating uncalibrated river flood impact function for sector {sector}')
    impf_sector = ImpfRiverFlood.from_jrc_region_sector(
        region="Africa",
        sector=sector_map
    )
    df = pd.DataFrame(dict(
        intensity = impf_sector.intensity,
        paa = impf_sector.paa,
        mdd = impf_sector.mdd,
        id = 1
    ))
    out_path = Path(data_dir, f'impf_river_flood_{sector}_uncalibrated.csv')
    df.to_csv(out_path, index=False)

    axs[i] = impf_sector.plot(axis=axs[i])

for i in range(len(sector_map_dict), len(axs)):
    fig.delaxes(axs[i])

plot_path = Path(data_dir, f'impf_river_flood_uncalibrated.png')
plt.savefig(plot_path)