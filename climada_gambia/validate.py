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
from climada_gambia.metadata_impact import MetadataImpact
from climada_gambia.utils_exposures import get_exposures
from climada_gambia.impact_function_manager import impfset_from_csv


# CONF_PATH = "/Users/chrisfairless/Library/CloudStorage/OneDrive-Personal/Projects/UNU/gambia2025/climada_gambia/conf.json"

def validate_impacts(impf_dict, data_dir, output_dir, overwrite):
    """Validate impact calculations.
    
    Args:
        impf_dict: MetadataImpact instance or dict
        data_dir: Base data directory
        output_dir: Output directory
        overwrite: Whether to overwrite existing files
    """
    impf_dict = MetadataImpact(impf_dict)
    hazard_type = impf_dict["hazard_type"]
    hazard_source = impf_dict["hazard_source"]
    exposure_type = impf_dict["exposure_type"]
    exposure_source = impf_dict["exposure_source"]
    
    haz_filepath_list = utils_config.gather_hazard_metadata(hazard_type, hazard_source, flatten=True)
    exp = None
    impf_set = impfset_from_csv(impf_dict["file_path"], hazard_type=impf_dict["hazard_abbr"])
    impf_id = impf_set.get_ids(haz_type=impf_dict["hazard_abbr"])
    if len(impf_id) > 1:
        raise ValueError("Multiple impact function IDs found in impact function set. I wasn't expecting this.")
    impf_id = impf_id[0]

    # This will need refactoring if we ever create different exposures for the future
    for i, haz_filepath in enumerate(haz_filepath_list):
        if len(haz_filepath_list) > 1:
            print(f'... processing hazard file {i+1} / {len(haz_filepath_list)}')

        output_impact_dir = impf_dict.impact_output_dir(create=True)
        output_path = impf_dict.get_output_impact_path(haz_filepath)
        if os.path.exists(output_path) and not overwrite:
            print(f'... file exists already, skipping')
            continue

        haz = Hazard.from_hdf5(haz_filepath)
        exp = exp if exp else get_exposures(impf_dict["exposure_node"], impf_dict.exposure_dir(), scenario="present", impf_id=impf_id)

        # TO DO come back and work on this: we probably want to downscale the low-res exposures!!
        threshold_exp = estimate_matching_threshold(haz.centroids.coord)
        threshold_haz = estimate_matching_threshold(np.stack([exp.gdf.geometry.y.values, exp.gdf.geometry.x.values], axis=1))
        threshold = max(threshold_exp, threshold_haz)

        exp.assign_centroids(haz, distance="euclidean", threshold=threshold)
        imp = ImpactCalc(exp, impf_set, haz).impact(save_mat=True, assign_centroids=False)
        print(f'... writing impact to {output_path}')
        imp.write_hdf5(output_path)


def main(overwrite=False):
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
    impf_list = utils_config.gather_impact_calculation_metadata()

    # Exposure files tend to be larger than hazard files, so we load each exposure once, then loop through hazards
    for impf_dict in impf_list:
        print("-----------------------------------------------------")
        print(f"Validating impacts for {impf_dict['exposure_type']}: {impf_dict['exposure_source']} - {impf_dict['hazard_type']}: {impf_dict['hazard_source']}")

        if not impf_dict["exposure_node"]:
            print(' MISSING: No exposure configuration found as specified in impact functions. Skipping')
            continue

        if not impf_dict["hazard_node"]:
            print(' MISSING: No hazard configuration found as specified in impact functions. Skipping')
            continue

        try:
            validate_impacts(
                impf_dict,
                data_dir=data_dir,
                output_dir=output_dir,
                overwrite=overwrite
            )
        except Exception as e:
            print(' ERROR: Failed to calculate impacts')
            print(f'{e}')
            raise e
            continue


if __name__ == "__main__":
    main()