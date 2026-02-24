#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import os
import copy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from functools import partial
from typing import Dict
from matplotlib import pyplot as plt
from climada.entity import ImpactFuncSet


from climada_gambia import utils_config
from climada_gambia.calculate_impacts import calculate_impacts, RP_LEVELS
from climada_gambia.analyse_impacts import analyse_impf_exceedance
from climada_gambia.utils_observations import load_observations
from climada_gambia.impact_function_manager import ImpactFunctionManager
from climada_gambia.config import CONFIG
from climada_gambia.metadata_calibration import MetadataCalibration
from climada_gambia.metadata_impact import MetadataImpact
from matplotlib.colors import to_hex


overwrite = True
ANALYSIS_NAME = "calibration"
PARALLELISE = 5
SAVE_SEARCH_IMPACTS = False   # Save storage

parameters = {
    'scale_x': np.arange(0.2, 2.9, 0.2),
    'scale_y': np.arange(0.6, 3.1, 0.2),
    # 'thresholds': {
    #     'affected': 0.1,
    #     'damaged': 0.5,
    #     'destroyed': 0.6
    # }
}

impf_filter = {
    'hazard_type': 'flood',
    'hazard_source': 'aqueduct',
    'exposure_type': 'housing',
    'exposure_source': 'GHS',
    # 'exposure_type': 'agriculture',
    # 'exposure_source': 'IUCN',
    # 'exposure_type': 'livestock',
    # 'exposure_source': 'GLW4',
    # 'exposure_type': 'manufacturing',
    # 'exposure_source': 'NCCS',
    'impact_type': 'economic_loss'
}

impf_filter = {}

# =====================================================


def simulate(impf_dict, parameters, overwrite=True):

    calibration_dict = MetadataCalibration(analysis_name=impf_dict["analysis_name"])
    out_path = calibration_dict.calibration_search_csv_path(create=True)
    # if os.path.exists(out_path) and not overwrite:
    #     print(f" Calibration results already exist at {out_path}, skipping simulation.")
    #     return

    # Create list of parameter combinations
    param_combinations = [
        {
            'scale_x': x,
            'scale_y': y,
            # 'thresholds': parameters['thresholds']
        }
        for x in parameters['scale_x']
        for y in parameters['scale_y']
    ]
    
    worker_func = partial(run_one_parameter_combo, impf_dict_in=impf_dict, overwrite=overwrite)
    
    # Run in parallel
    if PARALLELISE is not None and PARALLELISE > 1:
        n_workers = max(1, PARALLELISE)
        print(f"Multiprocessing pool initialized with {n_workers} workers")
        with multiprocessing.Pool(processes=n_workers) as pool:
            results_list = pool.map(worker_func, param_combinations)
    else:
        results_list = []
        for param_set in param_combinations:
            results_list.append(worker_func(param_set))

    results = pd.concat(results_list)
    results.to_csv(out_path, index=False)



def plot_calibration_results(impf_dict: MetadataImpact):
    calibration_dict = MetadataCalibration(analysis_name=impf_dict["analysis_name"])
    out_path = calibration_dict.calibration_search_csv_path(create=False)
    results = pd.read_csv(out_path, dtype={"exposure_type": str, "exposure_source": str, "scale_x": float, "scale_y": float, "rp_level": str, "score": float, "thresholds": object})
    results['thresholds'] = results['thresholds'].apply(lambda x: eval(x))

    for rp_level in RP_LEVELS:
        rp_df = results[results['rp_level'] == rp_level]
        rp_df['score'] = np.log10(rp_df['score'])
        xs = np.sort(rp_df['scale_x'].unique())
        ys = np.sort(rp_df['scale_y'].unique())
        pivot = rp_df.pivot(index='scale_y', columns='scale_x', values='score')
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
        ax.set_xticklabels([f'{x:.2f}' for x in xs])
        ax.set_yticklabels([f'{y:.2f}' for y in ys])

        # add x indicating chosen parameter combo (for both the 'best score' combo and the 'best score with a little judgement' combo)
        optimal_params_raw = read_calibrated_scaling_factors(impf_dict, rp_level, a_little_judgement=False)
        optimal_params_final = read_calibrated_scaling_factors(impf_dict, rp_level, a_little_judgement=True)
        print(f"Optimal parameters for RP level {rp_level} - best score: scale_x={optimal_params_raw['scale_x']}, scale_y={optimal_params_raw['scale_y']}")
        x_idx = np.where(xs == optimal_params_raw['scale_x'])[0][0]
        y_idx = np.where(ys == optimal_params_raw['scale_y'])[0][0]
        ax.scatter(x_idx, y_idx, color='grey', marker='x', s=100, label='Best score')
        if optimal_params_final['scale_x'] != optimal_params_raw['scale_x'] or optimal_params_final['scale_y'] != optimal_params_raw['scale_y']:
            print(f"Optimal parameters for RP level {rp_level} - best score with judgement: scale_x={optimal_params_final['scale_x']}, scale_y={optimal_params_final['scale_y']}")
            x_idx_final = np.where(xs == optimal_params_final['scale_x'])[0][0]
            y_idx_final = np.where(ys == optimal_params_final['scale_y'])[0][0]
            ax.scatter(x_idx_final, y_idx_final, color='white', marker='x', s=100, label='Best score adjusted')
            
        # add colorbar (legend) showing the mapped values; label indicates we plotted log10(score)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('log10(score)')

        plot_path = calibration_dict.calibration_search_plot_path(rp_level, create=True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)

    # Plot calibrated impact functions
    impfs_optimised = {}
    for rp_level in RP_LEVELS:
        impf_dict_optimised = read_calibrated_scaling_factors(impf_dict, rp_level, a_little_judgement=True)
        impf_dict_optimised["analysis_name"] = f"{impf_dict['analysis_name']}/calibrated_{rp_level}"
        impf_manager = ImpactFunctionManager(impf_dict_optimised.impact_function_path(), impf_dict_optimised["hazard_abbr"])
        impf = impf_manager.load_impf(scale_x=impf_dict_optimised["scale_x"], scale_y=impf_dict_optimised["scale_y"])
        impf.id = rp_level
        impfs_optimised[rp_level] = impf

    impf_orig = impf_manager.load_impf()
    impf_orig.id = 'uncalibrated'
    impfs_optimised['uncalibrated'] = impf_orig

    impfs_optimised_set = ImpactFuncSet(impfs_optimised.values())
    fig = impfs_optimised_set.plot()
    plt.savefig(Path(impf_dict.analysis_output_dir(create=False), f"impact_functions_calibrated.png"))

    # Plot all fitted impact functions
    fig, axes = plt.subplots(3, 1)
    axes = axes.flatten()

    for ax, rp_level in zip(axes, RP_LEVELS):
        impf_dict_temp = copy.deepcopy(impf_dict)
        impf_dict_temp["analysis_name"] = f"{impf_dict['analysis_name']}/calibrated_{rp_level}"
        impf_manager = ImpactFunctionManager(impf_dict_temp.impact_function_path(), impf_dict_temp["hazard_abbr"])
        impf = impf_manager.load_impf(scale_x=1, scale_y=1)

        title = "RP level: %s" % rp_level
        ax.set_xlabel("Intensity (" + impf.intensity_unit + ")")
        ax.set_ylabel("Impact (%)")
        ax.set_title(title)

        # Map the score to a color according to viridis, where the best score (lowest) is dark purple and the worst score (highest) is yellow; we can do this by normalising the score to be between 0 and 1, where 0 corresponds to the best score and 1 corresponds to the worst score, and then using this normalised score to get a color from the viridis colormap
        params_df = read_calibration_logs(impf_dict, rp_level)
        params_df['score'] = np.log10(params_df['score'])
        params_df = params_df.sort_values('score', ascending=False)
        max_score = params_df['score'].max()
        min_score = params_df['score'].min()
        norm_scores = (params_df['score'] - min_score) / (max_score - min_score)
        params_df['colour'] = [to_hex(plt.cm.viridis(score)) for score in norm_scores]

        for _, params in params_df.iterrows():
            ax.plot(impf.intensity * params['scale_x'], impf.mdd * params['scale_y'] * 100, color=params['colour'])

        ax.plot(impf.intensity, impf.mdd * 100, color='red', linewidth=1, label='Original IMPF')
        ax.plot(impfs_optimised[rp_level].intensity, impfs_optimised[rp_level].mdd * 100, color='lightblue', linewidth=1, label='Calibrated IMPF')

        ax.set_xlim((impf.intensity.min(), impf.intensity.max()))
        ax.set_ylim((0, impf.mdd.max() * 100 * params_df['scale_y'].max() * 1.05))

    plt.savefig(Path(impf_dict.analysis_output_dir(create=False), f"impact_functions_search.png"))


def run_optimal_parameters(impf_dict: MetadataImpact, overwrite=True):
    for rp_level in RP_LEVELS:
        print(f"Running the 'most successful' parameter combo for RP {rp_level} to save as final output")
        impf_dict_optimised = read_calibrated_scaling_factors(impf_dict, rp_level, a_little_judgement=True)
        impf_dict_optimised["analysis_name"] = f"{impf_dict['analysis_name']}/calibrated_{rp_level}"

        _ = calculate_impacts(
            impf_dict_optimised,
            scenario=None,
            scale_impacts=True,
            fit_thresholds=False,
            write_extras=True,
            overwrite=overwrite
        )
        _, _ = analyse_impf_exceedance(impf_dict=impf_dict_optimised, scenario=None, make_plots=True, write_extras=True, overwrite=overwrite)
    

def read_calibration_logs(impf_dict, rp_level) -> pd.DataFrame:
    calibration_dict = MetadataCalibration(analysis_name=impf_dict["analysis_name"])
    out_path = calibration_dict.calibration_search_csv_path(create=False)
    results = pd.read_csv(out_path, dtype={"exposure_type": str, "exposure_source": str, "scale_x": float, "scale_y": float, "rp_level": str, "score": float, "thresholds": object})
    results['thresholds'] = results['thresholds'].apply(lambda x: eval(x))    
    params_df = results[results['rp_level'] == rp_level]
    return params_df


def read_calibrated_scaling_factors(impf_dict_in, rp_level, a_little_judgement=True) -> Dict:
    impf_dict = copy.deepcopy(impf_dict_in)
    params_df = read_calibration_logs(impf_dict, rp_level)
    ix_score = np.nanargmin(params_df['score'].values)
    if a_little_judgement:
        # Check if there are other parameter combinations with similar scores (within a few % of the best score) and if so, 
        # choose the one with scale_x and scale_y closest to 1 to avoid overfitting
        sensitivity = 0.05
        best_score = params_df['score'].values[ix_score]

        # Move parameters closer to scale_x = 1 and scale_y = 1, within the sensitivity range
        if False:
            similar_ix = np.where(params_df['score'].values <= best_score * (1 + sensitivity))[0]
            if len(similar_ix) > 1:
                d_similar = (params_df.iloc[similar_ix]['scale_x'].values - 1)**2 + (params_df.iloc[similar_ix]['scale_y'].values - 1)**2
                ix_score = similar_ix[np.argmin(d_similar)]

        # Only move parameters _down_ towards scale_y = 1 within the sensitivity range, otherwise keep as is
        similar_ix = np.where(params_df['score'].values <= best_score * (1 + sensitivity))[0]
        if len(similar_ix) > 1:
            d_similar = np.abs(params_df.iloc[similar_ix]['scale_y'].values - 1)
            ix_score = similar_ix[np.argmin(d_similar)]
            
    params = params_df.iloc[ix_score]
    for param in ['scale_x', 'scale_y', 'thresholds']:
        impf_dict[param] = params[param]
    return impf_dict



def run_one_parameter_combo(parameters, impf_dict_in, overwrite=True):
    """Run calibration for one parameter combination.
    
    Args:
        impf_dict_in: MetadataImpact instance
        parameters: Dictionary of calibration parameters
    """
    print(f'PARAMETER COMBINATION scale_x: {parameters["scale_x"]}  scale_y: {parameters["scale_y"]}')
    temp_analysis_name = f'{impf_dict_in["analysis_name"]}/search/x{parameters["scale_x"]}_y{parameters["scale_y"]}'
    impf_dict = MetadataImpact(impf_dict=impf_dict_in, analysis_name=temp_analysis_name)
    impf_dict["scale_x"] = parameters["scale_x"]
    impf_dict["scale_y"] = parameters["scale_y"]

    assert "thresholds" in impf_dict.keys()
    
    # for sub_impact, thresh in parameters['thresholds'].items():
    #     if sub_impact in impf_dict["thresholds"].keys():
    #         print(f"Adjusting the {sub_impact} threshold")
    #         impf_dict["thresholds"][sub_impact] = thresh

    results_list = []
    exceedance_out_path = impf_dict.exceedance_type_csv_path(create=True)
    if not os.path.exists(exceedance_out_path) or overwrite:
        impf_dict_out = calculate_impacts(
            impf_dict,
            scenario="present",
            scale_impacts=True,
            fit_thresholds=True,
            write_extras=False,
            overwrite=overwrite 
        )
    else:
        print("Found existing impact calculation outputs, skipping impact calculation step")
        impf_dict_out = copy.deepcopy(impf_dict)

    fitted_thresholds = {}
    if len(impf_dict_out["thresholds"]) > 0:
        fitted_thresholds_path = impf_dict.fitted_thresholds_file_path(create=False)
        assert os.path.exists(fitted_thresholds_path), f"Fitted thresholds file not found at {fitted_thresholds_path}: was an earlier calculation interrupted? Delete the parent folder if so and try again."
        fitted_thresholds_df = pd.read_csv(fitted_thresholds_path)
        
        for i_type in impf_dict_out["thresholds"].keys():
            fitted_thresholds[i_type] = {}
            df_type = fitted_thresholds_df.loc[fitted_thresholds_df['impact_type'] == i_type]
            for rp_level in RP_LEVELS:
                df_rp = df_type.loc[df_type['rp_level'] == rp_level]
                thresh_value = df_rp['threshold_final'].values[0]
                fitted_thresholds[i_type][rp_level] = thresh_value
    _, scores_df = analyse_impf_exceedance(impf_dict_out, scenario="present", make_plots=False, write_extras=False, overwrite=False)

    for rp_level in RP_LEVELS:
        if scores_df is not None:
            score = scores_df.loc['TOTAL', f'weighted_cost_{rp_level}']
        else:
            score = None
        result = {
            "exposure_type": impf_dict_out["exposure_type"],
            "exposure_source": impf_dict_out["exposure_source"],
            "scale_x": parameters["scale_x"],
            "scale_y": parameters["scale_y"],
            "rp_level": rp_level,
            "score": score,
            "thresholds": {i_type: fitted_thresholds[i_type][rp_level] for i_type in fitted_thresholds.keys()}
        }
        results_list.append(result)
    
    if not SAVE_SEARCH_IMPACTS:
        haz_filepath = Path(impf_dict["hazard_node"]["present"]["files"])
        impacts_dir = impf_dict.impact_output_dir(create=False)
        _ = [os.remove(impacts_path) for impacts_path in impacts_dir.glob("*.hdf5")]
        _ = [os.remove(impacts_path) for impacts_path in impacts_dir.glob("*.h5")]

    return pd.DataFrame(results_list)


def main(overwrite, analysis_name, impf_filter):
    impf_dict_list = utils_config.gather_impact_calculation_metadata(filter=impf_filter, analysis_name=analysis_name)
    if len(impf_filter) > 0:
        assert len(impf_dict_list) == 1, f'Expected one impact function for filter {impf_filter}, found {len(impf_dict_list)}'
    print(f"Building calibration searches for {len(impf_dict_list)} impact calculations:")
    analyses_list = [f"{analysis_name}/{impf_dict['exposure_type']}_{impf_dict['exposure_source']}" for impf_dict in impf_dict_list]
    print(analyses_list)
    impf_dict_list = [MetadataImpact(impf_dict=impf_dict, analysis_name=n) for impf_dict, n in zip(impf_dict_list, analyses_list)]

    for impf_dict in impf_dict_list:
        print("===================================================")
        print(f"Starting calibration search for impact function:")
        print(f" Hazard: {impf_dict['hazard_type']} - {impf_dict['hazard_source']}")
        print(f" Exposure: {impf_dict['exposure_type']} - {impf_dict['exposure_source']}")
        print(f" Impact type: {impf_dict['impact_type']}")

        plot_path = MetadataCalibration(analysis_name=impf_dict["analysis_name"]).calibration_search_plot_path(rp_level='upper', create=False)
        simulate(impf_dict, parameters, overwrite=overwrite)
        run_optimal_parameters(impf_dict, overwrite=overwrite)
        plot_calibration_results(impf_dict)
    print("Done")



if __name__ == "__main__":
    main(overwrite=overwrite, analysis_name=ANALYSIS_NAME, impf_filter=impf_filter)