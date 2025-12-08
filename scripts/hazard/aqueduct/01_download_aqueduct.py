import requests
import os
from pathlib import Path
import pandas as pd

# Download raw aqueduct data

# The code in this folder is adapted from the scripts here
# https://bitbucket.org/celsiuspro/nccs-supply-chain/src/main/resources/impacts/slr_intermediate/
# They were written to work with coastal flood data, so there were adjustments, mostly for the file naming
# If you're interested in scaling this up for a large/global study in future, the scripts are designed for that, so
# I recommend re-adapting them

aqueduct_url = 'https://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/'
data_dir = '/Users/chrisfairless/Data/UNU/gambia2025/inputs/hazard/aqueduct/'

rp_list = [2, 5, 10, 25, 50, 100, 250, 500, 1000]

filename_base_list = [
    'inunriver'
]
scenario_list = [
    'historical',
    'rcp4p5',
    'rcp8p5'
]
year_list = [
    # '2030',
    '2050',
    '2080'
]

output_directory = Path(data_dir, 'raw')
if not os.path.exists(output_directory):
    raise FileNotFoundError(f'No output folder at {output_directory} – please create')

def download_file(filename, base_url, outdir):
    out_path = Path(outdir, filename)
    response = requests.get(base_url + filename)
    if response.ok:
        with open(out_path, mode="wb") as f:
            f.write(response.content)
    else:
        raise ValueError('Download failed')
    return out_path


download_list = []

for filename_base in filename_base_list:
    for scenario in scenario_list:
        if scenario == 'historical':
            scenario_year_list = ['1980']
            model_list = ['WATCH']
        else:
            scenario_year_list = year_list
            model_list = ['NorESM1-M', 'GFDL-ESM2M', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM-CHEM']

        for model in model_list:
            model_str = model.rjust(14, '0')
            for year in scenario_year_list:
                print(f'Looking at {filename_base} {scenario} {model} {year}')
                for rp in rp_list:
                    rp_str = str(rp).rjust(5, '0')
                    filename = f'{filename_base}_{scenario}_{model_str}_{year}_rp{rp_str}.tif'
                    if os.path.exists(Path(output_directory, filename)):
                        print(f'Already exists: {filename}')
                        out_path = Path(output_directory, filename)
                    else:
                        print(f'Downloading: {filename}')
                        out_path = download_file(filename, aqueduct_url, output_directory)
                    d = dict(
                        filename_base = filename_base,
                        scenario = scenario,
                        model = model,
                        year = year,
                        rp = rp,
                        url = aqueduct_url + filename,
                        local_path = str(out_path)
                    )
                    download_list = download_list + [d]

df = pd.DataFrame(download_list)
df.to_csv(Path(data_dir, 'aqueduct_download_metadata.csv'))