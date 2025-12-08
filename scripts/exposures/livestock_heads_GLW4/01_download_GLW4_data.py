import requests
import os
from pathlib import Path

# We'll use the Gridded Livestock of the World for livestock density

# TODO: overlay pastoral land as a way to downscale cattle (but not chicken)

download_dir = Path('/Users/chrisfairless/Data/UNU/gambia2025/inputs/exposures/livestock_heads_GLW4/raw/')
overwrite = False

# ------------------------------------------------------------------------------

downloads = {
    'all': 'https://storage.googleapis.com/fao-gismgr-glw4-2020-data/DATA/GLW4-2020/MAPSET/D-DA/GLW4-2020.D-DA.GLEAM3-ALL-LU.tif',
    'buffalo': 'https://storage.googleapis.com/fao-gismgr-glw4-2020-data/DATA/GLW4-2020/MAPSET/D-DA/GLW4-2020.D-DA.BFL.tif',
    'chicken': 'https://storage.googleapis.com/fao-gismgr-glw4-2020-data/DATA/GLW4-2020/MAPSET/D-DA/GLW4-2020.D-DA.BFL.tif',
    'cattle': 'https://storage.googleapis.com/fao-gismgr-glw4-2020-data/DATA/GLW4-2020/MAPSET/D-DA/GLW4-2020.D-DA.CTL.tif',
    'goats': 'https://storage.googleapis.com/fao-gismgr-glw4-2020-data/DATA/GLW4-2020/MAPSET/D-DA/GLW4-2020.D-DA.GTS.tif',
    'pigs': 'https://storage.googleapis.com/fao-gismgr-glw4-2020-data/DATA/GLW4-2020/MAPSET/D-DA/GLW4-2020.D-DA.PGS.tif',
    'sheep': 'https://storage.googleapis.com/fao-gismgr-glw4-2020-data/DATA/GLW4-2020/MAPSET/D-DA/GLW4-2020.D-DA.SHP.tif'
}


if not os.path.exists(download_dir.parent):
    raise FileNotFoundError(f'No download folder at {download_dir.parent} – please create')

os.makedirs(download_dir, exist_ok=True)
print("Downloading GLW4 livestock data")

for species, url in downloads.items():
    print(f"...{species}")
    out_path = Path(download_dir, f'glw4_{species}_5as.tif')
    if os.path.exists(out_path) and not overwrite:
        print(f'File already exists, skipping download: {out_path}')
        continue
    else:
        response = requests.get(url)
        if response.ok:
            with open(out_path, mode="wb") as f:
                f.write(response.content)
        else:
            raise ValueError('Download failed')

print("Downloads complete")
