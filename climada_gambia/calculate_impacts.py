#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np

from climada.hazard import Hazard
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.engine import ImpactCalc
from climada.util.coordinates import estimate_matching_threshold

from climada_gambia.check_inputs import check_node, check_enabled_node
from climada_gambia.config import CONFIG
from climada_gambia import utils_config

# CONF_PATH = "/Users/chrisfairless/Library/CloudStorage/OneDrive-Personal/Projects/UNU/gambia2025/climada_gambia/conf.json"

HAZARD_MAP = {
    "flood": "FL"
}

country = 'Gambia'

def calculate_impacts(impf_dict, data_dir, output_dir=None, overwrite=True):
    hazard_type = impf_dict["hazard_type"]
    hazard_source = impf_dict["hazard_source"]
    exposure_type = impf_dict["exposure_type"]
    exposure_source = impf_dict["exposure_source"]
    impact_type = impf_dict["impact_type"]
    os.makedirs(impf_dict["impact_dir"], exist_ok=True)
    output_dir = output_dir if output_dir else impf_dict["impact_dir"]

    haz_filepath_list = utils_config.gather_hazard_metadata(hazard_type, hazard_source, flatten=True)
    exp = None
    impf_set = impfset_from_csv(impf_dict["impf_file_path"], hazard_type=hazard_type)
    impf_id = impf_set.get_ids(haz_type=HAZARD_MAP[hazard_type])
    if len(impf_id) > 1:
        raise ValueError("Multiple impact function IDs found in impact function set. I wasn't expecting this.")
    impf_id = impf_id[0]
    impact_path_list = []

    # This will need refactoring if we ever create different exposures for the future
    for i, haz_filepath in enumerate(haz_filepath_list):
        if len(haz_filepath_list) > 1:
            print(f'... processing hazard file {i+1} / {len(haz_filepath_list)}')

        impact_path = Path(output_dir, f'impact_{impact_type}_{exposure_type}_{exposure_source}_{hazard_source}_{Path(haz_filepath).stem}.hdf5')
        if os.path.exists(impact_path) and not overwrite:
            print(f'... file exists already, skipping')
            continue

        haz = Hazard.from_hdf5(haz_filepath)
        exp = exp if exp else get_exposures(impf_dict["exposure_node"], impf_dict["exposure_dir"], scenario="present", impf_id=impf_id)

        # TO DO come back and work on this: we probably want to downscale the low-res exposures!!
        threshold_exp = estimate_matching_threshold(haz.centroids.coord)
        threshold_haz = estimate_matching_threshold(np.stack([exp.gdf.geometry.y.values, exp.gdf.geometry.x.values], axis=1))
        threshold = max(threshold_exp, threshold_haz)

        exp.assign_centroids(haz, distance="euclidean", threshold=threshold, overwrite=True)
        imp = ImpactCalc(exp, impf_set, haz).impact(save_mat=True, assign_centroids=False)
        print(f'... writing impact to {impact_path}')
        imp.write_hdf5(impact_path)
        impact_path_list.append(impact_path)

    impf_dict['impact_files'] = impact_path_list
    return impf_dict


def get_exposures(exposure_node, exp_dir, scenario="present", impf_id=None):
    exp_files = exposure_node["files"]
    if not isinstance(exp_files, list):
        exp_files = [exp_files]
    return Exposures.concat([get_one_exposure(Path(exp_dir, fn), impf_id) for fn in exp_files])


def get_one_exposure(filepath, impf_id=None):
    if filepath.suffix == ".hdf5" or filepath.suffix == ".h5":
        exp = Exposures.from_hdf5(filepath)
    elif filepath.suffix == ".csv":
        exp = Exposures.from_csv(filepath)
    elif filepath.suffix == ".tif" or filepath.suffix == ".tiff":
        exp = Exposures.from_raster(filepath)
    else:
        raise ValueError(f"Unknown exposure file format: {filepath.suffix}")

    if impf_id is None and 'impf_' in exp.gdf.columns:
        return exp

    gdf = exp.gdf
    gdf['impf_'] = impf_id
    exp.set_gdf(gdf)
    return exp


def impfset_from_csv(filepath, hazard_type):
    df = pd.read_csv(filepath)
    impf_list = []
    for id in df['id'].unique():
        df_id = df[df['id'] == id]
        impf = ImpactFunc(
            haz_type = HAZARD_MAP[hazard_type],
            id = id,
            intensity = df['intensity'],
            mdd = df['mdd'],
            paa = df['paa']
        )
        impf_list.append(impf)
    return ImpactFuncSet(impf_list)


def main(overwrite):
    # conf_path = Path(CONF_PATH)
    # if not conf_path.exists():
    #     raise FileExistsError(f"conf.json not found at {conf_path}. Adjust CONF_PATH or location.")

    # with conf_path.open("r", encoding="utf-8") as f:
    #     conf = json.load(f)

    conf = CONFIG
    data_dir = Path(conf.get("data_dir"))
    output_dir = Path(conf.get("output_dir"))
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f'Please create an output directory at {output_dir}')

    # Gather all impact calculations:
    impf_list = utils_config.gather_impact_function_metadata()

    # Exposure files tend to be larger than hazard files, so we load each exposure once, then loop through hazards
    for impf_dict in impf_list:
        print("-----------------------------------------------------")
        print(f"Calculating impacts for {impf_dict['exposure_type']}: {impf_dict['exposure_source']} - {impf_dict['hazard_type']}: {impf_dict['hazard_source']}")

        if not impf_dict["exposure_node"]:
            print(' MISSING: No exposure configuration found as specified in impact functions. Skipping')
            continue

        if not impf_dict["hazard_node"]:
            print(' MISSING: No hazard configuration found as specified in impact functions. Skipping')
            continue

        try:
            _ = calculate_impacts(
                impf_dict,
                data_dir=data_dir,
                output_dir=None,
                overwrite=overwrite
            )
        except Exception as e:
            print(' ERROR: Failed to calculate impacts')
            print(f'{e}')
            raise e
            continue


if __name__ == "__main__":
    main(overwrite=True)