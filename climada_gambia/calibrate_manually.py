#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from climada.engine import Impact

from climada_gambia import utils_config
from climada_gambia import calculate_impacts 
from config import CONFIG


parameters = {
    'v_thresh': 0,
    'v_half': 1.5,
    'v_max': 1,
    'thresh_affected': 0.1,
    'thresh_damaged': 0.5,
    'thresh_destroyed': 0.8
}

model_structure = {
    'impact_type': 
}

working_dir = Path(CONFIG["output_dir"], "calibration", "temp")

impf_filter = {
    'hazard_type': 'flood',
    'hazard_source': 'aqueduct',
    'exposure_type': 'housing',
    'exposure_source': 'GHS',
    'impact_type': 'economic_loss'
}
impf_dict_list = utils_config.gather_impact_function_metadata(filter=impf_filter)
assert len(impf_dict_list) == 1, f'Expected one impact function for filter {impf_filter}, found {len(impf_dict_list)}'
impf_dict = impf_dict_list[0]


def simulate(impf_dict, parameters):
    impf_dict = calculate_impacts.calculate_impacts(
        impf_dict,
        data_dir=Path(CONFIG["data_dir"]),
        output_dir=working_dir,
        overwrite=True
    )




def calibrate(impf_dict, data_dir, output_dir, overwrite):
    exposure_type = impf_dict["exposure_type"]
    exposure_source = impf_dict["exposure_source"]
    hazard_source = impf_dict["hazard_source"]

    output_impfs_dir = Path(data_dir, "impact_functions", "calibrated")
    output_working_dir = Path(output_dir, "calibration", "working")

    if not os.path.exists(output_impfs_dir.parent):
        raise FileNotFoundError(f'Please create an impact functions directory at {output_impfs_dir.parent}')
    os.makedirs(output_impfs_dir, exist_ok=True)

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f'Please create an output directory at {output_dir}')
    os.makedirs(output_working_dir, exist_ok=True)

    for scenario, haz_node in impf_dict["hazard_node"].items():
        # print(f"... reading scenario {scenario}")
        haz_filepath_list = haz_node['files']
        if not isinstance(haz_filepath_list, list):
            haz_filepath_list = [haz_filepath_list]
        haz_filepath_list = [fp for fp in haz_filepath_list if not '_ALL_' in fp]  # don't analyse combined impacts from all events
        
        for i, haz_filepath in enumerate(haz_filepath_list):
            impact_path = Path(impact_dir, f'impact_{exposure_type}_{exposure_source}_{hazard_source}_{Path(haz_filepath).stem}.hdf5')
            if not os.path.exists(impact_path):
                raise FileNotFoundError(f'Impact data is missing: {impact_path}')

            imp = Impact.from_hdf5(impact_path)
            rp_data = [
                {
                    "scenario": scenario,
                    "hazard_filepath": Path(haz_filepath).stem,
                    "exposure_type": exposure_type,
                    "exposure_source": exposure_source,
                    "unit": imp.unit,
                    "frequency": freq,
                    "impact": i
                } for freq, i in zip(
                    imp.frequency,
                    imp.at_event
                )
            ]
            curves = curves + rp_data
    
    curves = pd.DataFrame(curves)

    if os.path.exists(exceedance_plot_path) and not overwrite:
        print('... plot already exists, just extracting exceedance values')
        return curves
    
    _, axis = plt.subplots(1, 1, figsize=figsize)
    axis.set_title(f"Exceedance curve for {exposure_type}: {exposure_source}")
    axis.set_ylabel(f"Impact ({imp.unit})")
    rp = [1/f for f in imp.frequency]
    if log_frequency:
        axis.set_xlabel(f"Exceedance frequency ({self.frequency_unit})")
        axis.set_xscale("log")
    else:
        axis.set_xlabel("Return period (year)")
    
    for scenario in impf_dict["hazard_node"].keys():
        scenario_df = curves[curves['scenario'] == scenario]
        for haz_filepath in set(scenario_df['hazard_filepath']):
            sub_df = scenario_df[scenario_df['hazard_filepath'] == haz_filepath]
            axis.plot(rp, sub_df['impact'], color=COLOURS[scenario]['normal'])
        
        scenario_mean = scenario_df.groupby('frequency')['impact'].agg('mean')
        axis.plot(scenario_mean.index**-1, scenario_mean.values, color=COLOURS[scenario]['strong'], linewidth=2)
    
    axis.figure.savefig(exceedance_plot_path)
    return curves




def main(overwrite=False):
    conf = CONFIG
    data_dir = Path(conf.get("data_dir"))
    output_dir = Path(conf.get("output_dir"))
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f'Please create an output directory at {output_dir}')

    results_dir = Path(output_dir, 'calibration')
    os.makedirs(results_dir, exist_ok=True)

    impf_list = utils_config.gather_impact_function_metadata(filter={"calibrated_string": "uncalibrated"})
    calibration_targets = pd.read_csv(Path(data_dir))

    for impf_dict in impf_list:
        print("-----------------------------------------------------")
        print(f"Calibrating impacts for {impf_dict['exposure_type']}: {impf_dict['exposure_source']} - {impf_dict['hazard_type']}: {impf_dict['hazard_source']}")

        if not impf_dict["exposure_node"]:
            print(' MISSING: No exposure configuration found as specified in impact functions. Skipping')
            continue

        if not impf_dict["hazard_node"]:
            print(' MISSING: No hazard configuration found as specified in impact functions. Skipping')
            continue

        try:
            this_rp_data = plot_exceedance(
                impf_dict,
                data_dir=data_dir,
                output_dir=output_dir,
                overwrite=overwrite
            )
            all_rp_data.append(this_rp_data)
        except Exception as e:
            print(f" ERROR: Failed to visualise exceedance curves impacts for {impf_dict['hazard_type']}: {impf_dict['hazard_source']} – {impf_dict['exposure_type']}: {impf_dict['exposure_source']}")
            print(f'{e}')
            raise e
            continue
    
    if len(all_rp_data) > 0:
        print("Writing output CSV")
        all_rp_data = pd.concat(all_rp_data, ignore_index=True)
        all_rp_data.to_csv(csv_path)
    else:
        print(f"No data found for calibration status {calibrated_string}")




if __name__ == "__main__":
    main()