#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import os
import copy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from climada_gambia import utils_config
from climada_gambia.calculate_impacts import calculate_impacts
from climada_gambia.analyse_impacts import analyse_exceedance
from climada_gambia.impact_function_manager import ImpactFunctionManager
from climada_gambia.metadata_calibration import MetadataCalibration
from config import CONFIG



analysis_name = "manufacturing_doubling"

parameters = {
    'x_scale': 1,
    'y_scale': 2,
    'thresholds': {
        'affected': 0.05,
        'damaged': 0.3,
        'destroyed': 0.6
    }
}

fit_thresholds_automatically = True

# control
# parameters = {
#     'x_scale': 1,
#     'y_scale': 1,
#     'thresholds': {
#         'affected': 0.1,
#         'damaged': 0.5,
#         'destroyed': 0.8
#     }
# }

impf_filter = {
    'hazard_type': 'flood',
    'hazard_source': 'aqueduct',
    # 'exposure_type': 'housing',
    # 'exposure_type': 'agriculture',
    # 'exposure_type': 'livestock',
    'exposure_type': 'manufacturing',
    'exposure_source': 'NCCS',
    'impact_type': 'economic_loss'
}

path_builder = MetadataCalibration(
    config=CONFIG,
    analysis_name=analysis_name
)
working_dir = path_builder.calibration_working_dir(analysis_name)


def simulate(impf_dict, parameters, scale_impacts):
    """Simulate impacts with given parameters.
    
    Args:
        impf_dict: MetadataImpact instance
        parameters: Dictionary of calibration parameters as specified above
        scale_impacts: Whether to scale impacts as specified by the parameters
    """
    data_dir = Path(CONFIG["data_dir"])

    # Adjust for this run
    print("Adjusting the impact function")
    hazard_abbr = impf_dict["hazard_abbr"]
    
    # Use ImpactFunctionManager to load and scale the impact function
    manager = ImpactFunctionManager(impf_dict.impact_function_path(), hazard_abbr)
    impf = manager.load_impf()
    impf_scaled = manager.apply_scaling(impf, parameters["x_scale"], parameters["y_scale"])
        
    temp_impf_file_path = path_builder.calibration_temp_impf_path(working_dir)
    impf_dict['impf_file_path'] = temp_impf_file_path
    manager.impf_to_csv(impf_scaled, temp_impf_file_path)

    if "thresholds" in impf_dict.keys():
        for sub_impact, thresh in parameters['thresholds'].items():
            if sub_impact in impf_dict["thresholds"].keys():
                print(f"Adjusting the {sub_impact} threshold")
                impf_dict["thresholds"][sub_impact] = thresh

    if not fit_thresholds_automatically:
        impf_dict['impact_dir'] = working_dir
        impf_dict = calculate_impacts(
            impf_dict,
            scenario="present",
            scale_impacts=scale_impacts,
            fit_thresholds=None,
            write_extras=True,
            overwrite=True
        )
        _, scores = analyse_exceedance(impf_dict, data_dir, working_dir, scenario="present", write_extras=True, overwrite=True)
        print("SCORES:")
        print(scores)
    
    else:
        for fit_thresholds in ["lower", "mid", "upper"]:
            impf_dict_thresh = copy.deepcopy(impf_dict)
            output_dir = path_builder.calibration_output_subdir(working_dir, fit_thresholds)
            impf_dict_thresh['impact_dir'] = output_dir
            os.makedirs(output_dir, exist_ok=True)
            impf_dict_thresh = calculate_impacts(
                impf_dict_thresh,
                scenario="present",
                scale_impacts=scale_impacts,
                fit_thresholds=fit_thresholds,
                write_extras=True,
                overwrite=True
            )
            _, scores = analyse_exceedance(impf_dict_thresh, scenario="present", write_extras=True, overwrite=True)
            print(f"SCORES: (rp_level {fit_thresholds})")
            print(scores)            



def main(overwrite, scale_impacts):
    if not os.path.exists(working_dir.parent):
        raise FileNotFoundError(f'Please create an output directory at {working_dir.parent}')
    os.makedirs(working_dir, exist_ok=True)

    impf_dict_list = utils_config.gather_impact_calculation_metadata(filter=impf_filter)
    assert len(impf_dict_list) == 1, f'Expected one impact function for filter {impf_filter}, found {len(impf_dict_list)}'
    impf_dict = impf_dict_list[0]

    simulate(impf_dict, parameters)



if __name__ == "__main__":
    main(overwrite=True, scale_impacts=False)