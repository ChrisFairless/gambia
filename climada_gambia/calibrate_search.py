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
from climada_gambia.utils_observations import load_observations
from climada_gambia.config import CONFIG
from climada_gambia.impact_function_manager import ImpactFunctionManager
from climada_gambia.metadata_calibration import MetadataCalibration
from climada_gambia.metadata_impact import MetadataImpact

analysis_name = "temp_calibration/manufacturing_search"

parameters = {
    'x_scale': np.arange(0.3, 1.6, 0.3),
    'y_scale': np.arange(0.8, 2.4, 0.5),
    'thresholds': {
        'affected': 0.1,
        'damaged': 0.5,
        'destroyed': 0.6
    }
}

impf_filter = {
    'hazard_type': 'flood',
    'hazard_source': 'aqueduct',
    # 'exposure_type': 'housing',
    # 'exposure_source': 'GHS',
    # 'exposure_type': 'agriculture',
    # 'exposure_source': 'IUCN',
    # 'exposure_type': 'livestock',
    # 'exposure_source': 'GLW4',
    'exposure_type': 'manufacturing',
    'exposure_source': 'NCCS',
    'impact_type': 'economic_loss'
}

# =====================================================


def simulate(impf_dict, parameters, scale_impacts):
    results_list = []
    for x in parameters['x_scale']:
        for y in parameters['y_scale']:
            print('---------------------------------------------------')
            print(f'PARAMETER COMBINATION x_scale: {x}  y_scale: {y}')
            parameters_one = {
                'x_scale': x,
                'y_scale': y,
                'thresholds': parameters['thresholds']
            }
            results_list.append(run_one_parameter_combo(impf_dict, parameters_one, scale_impacts))
    results = pd.concat(results_list)
    calibration_dict = MetadataCalibration(analysis_name=analysis_name)
    out_path = calibration_dict.calibration_search_csv_path(analysis_name, create=True)
    results.to_csv(out_path, index=False)

def plot_calibration_results(impf_dict: MetadataCalibration, parameters: dict):
    calibration_dict = MetadataCalibration(analysis_name=impf_dict["analysis_name"])
    out_path = c.calibration_search_csv_path(create=False)
    results = pd.read_csv(out_path)

    for rp_level in ["lower", "mid", "upper"]:
        rp_df = results[results['rp_level'] == rp_level]
        rp_df['score'] = np.log10(rp_df['score'])
        xs = np.sort(rp_df['x_scale'].unique())
        ys = np.sort(rp_df['y_scale'].unique())
        pivot = rp_df.pivot(index='y_scale', columns='x_scale', values='score')
        pivot = pivot.reindex(index=ys, columns=xs)
        arr = pivot.values

        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(arr, cmap='viridis', interpolation='nearest', aspect='auto')

        xr = ax.get_xlim()
        yr = ax.get_ylim()
        ax.set_xticks(np.arange(max(xr)), minor=False)
        ax.set_yticks(np.arange(max(yr)), minor=False)
        ax.grid(which='minor', snap=False, color='k', linestyle='-', linewidth=1)
        ax.tick_params(which='major', bottom=False, left=False)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.set_xticklabels(xs)
        ax.set_yticklabels(ys)

        # add colorbar (legend) showing the mapped values; label indicates we plotted log10(score)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('log10(score)')

        plot_path = calibration_dict.calibration_search_plot_path(rp_level, create=True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)

    print("Running the 'most successful' parameter combo to save as the final output")
    for rp_level in ["lower", "mid", "upper"]:
        impf_dict_optimised = copy.deepcopy(impf_dict)
        params_df = results[results['rp_level'] == rp_level]
        ix_score = np.nanargmin(params_df['score'].values)
        params = params_df.iloc[ix_score]
        impf_dict_optimised['thresholds'] = params["thresholds"]

        temp_impf_file_path = calibration_dict.calibration_temp_impf_path(temp_str=f'optimised_{rp_level}', create=True)
        manager = ImpactFunctionManager(impf_dict_optimised.impact_function_path(), impf_dict_optimised['hazard_abbr'])
        impf = manager.load_impf()
        impf_scaled = manager.apply_scaling(impf, params["x_scale"], params["y_scale"])
        manager.impf_to_csv(impf_scaled, temp_impf_file_path)
        impf_dict_optimised['impf_file_path'] = temp_impf_file_path

        out_dir = calibration_dict.calibration_output_subdir(f'optimised_{rp_level}', create=True)
        impf_dict_optimised['impact_dir'] = out_dir
        os.makedirs(out_dir, exist_ok=True)
        _ = calculate_impacts(
            impf_dict_optimised,
            scenario="present",
            scale_impacts=scale_impacts,
            fit_thresholds=None,
            write_extras=True,
            overwrite=True
        )


def run_one_parameter_combo(impf_dict_in, parameters, scale_impacts):
    """Run calibration for one parameter combination.
    
    Args:
        impf_dict_in: MetadataImpact instance
        parameters: Dictionary of calibration parameters
        scale_impacts: Whether to scale impacts
    """
    impf_dict = MetadataImpact(copy.deepcopy(impf_dict_in.to_dict()))

    assert "thresholds" in impf_dict.keys()
    analysis_name = impf_dict_in["analysis_name"]
    calibration_dict = MetadataCalibration(analysis_name=analysis_name)
    
    manager = ImpactFunctionManager(impf_dict.impact_function_path(), impf_dict["hazard_abbr"])
    impf = manager.load_impf()
    impf_scaled = manager.apply_scaling(impf, parameters["x_scale"], parameters["y_scale"])
    temp_impf_file_path = calibration_dict.calibration_temp_impf_path(create=True)
    manager.impf_to_csv(impf_scaled, temp_impf_file_path)

    impf_dict['impact_dir'] = calibration_dict.calibration_working_dir(create=True)
    impf_dict['impf_file_path'] = temp_impf_file_path

    for sub_impact, thresh in parameters['thresholds'].items():
        if sub_impact in impf_dict["thresholds"].keys():
            print(f"Adjusting the {sub_impact} threshold")
            impf_dict["thresholds"][sub_impact] = thresh

    results_list = []
    for rp_level in ["lower", "mid", "upper"]:
        impf_dict_out = calculate_impacts(
            impf_dict,
            scenario="present",
            scale_impacts=scale_impacts,
            fit_thresholds=rp_level,
            write_extras=False,
            overwrite=True
        )
        if impf_dict_out["scores"] is not None:
            score = impf_dict_out["scores"].loc['TOTAL', f'weighted_cost_{rp_level}']
        else:
            score = None
        result = {
            "exposure_type": impf_dict_out["exposure_type"],
            "exposure_source": impf_dict_out["exposure_source"],
            "x_scale": parameters["x_scale"],
            "y_scale": parameters["y_scale"],
            "rp_level": rp_level,
            "score": score,
            "thresholds": impf_dict_out["thresholds"]
        }
        results_list.append(result)

    return pd.DataFrame(results_list)


def main(overwrite, scale_impacts, analysis_name):
    impf_dict_list = utils_config.gather_impact_calculation_metadata(filter=impf_filter, analysis_name=analysis_name)
    assert len(impf_dict_list) == 1, f'Expected one impact function for filter {impf_filter}, found {len(impf_dict_list)}'
    impf_dict = impf_dict_list[0]

    simulate(impf_dict, parameters, scale_impacts=scale_impacts)
    plot_calibration_results(impf_dict, parameters)
    print("Done")



if __name__ == "__main__":
    main(overwrite=True, scale_impacts=False, analysis_name=analysis_name)