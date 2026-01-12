#!/usr/bin/env python3
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
import logging
import copy

from climada.hazard import Hazard
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.engine import ImpactCalc
from climada.util.coordinates import estimate_matching_threshold

from climada_gambia.utils_total_exposed_value import get_total_exposed_value, get_total_exposed_units
from climada_gambia.utils_observations import load_observations
from climada_gambia.config import CONFIG
from climada_gambia import utils_config
from climada_gambia.analyse_impacts import compare_obs, get_curves, analyse_exceedance
from climada_gambia.impact_function_manager import ImpactFunctionManager
from climada_gambia.metadata_impact import MetadataImpact

logging.getLogger("climada").setLevel(logging.WARNING)

# CONF_PATH = "/Users/chrisfairless/Library/CloudStorage/OneDrive-Personal/Projects/UNU/gambia2025/climada_gambia/conf.json"

max_iters = 16


valid_thresholds = ["affected", "damaged", "destroyed"]

def calculate_impacts(impf_dict_in, scenario, fit_thresholds, scale_impacts=False, write_extras=True, overwrite=True):
    """Calculate impacts for a given impact calculation configuration.
    
    Args:
        impf_dict_in: MetadataImpact instance containing impact calculation configuration information
        scenario: Scenario name (e.g., 'present', 'RCP4.5_2050')
        fit_thresholds: Return period level for threshold fitting
        scale_impacts: Whether to scale impacts
        write_extras: Whether to write additional outputs
        overwrite: Whether to overwrite existing files
    """
    # Deep copy the underlying dict to avoid mutating the original
    impf_dict = MetadataImpact(copy.deepcopy(impf_dict_in.to_dict()))
    
    hazard_type = impf_dict["hazard_type"]
    hazard_source = impf_dict["hazard_source"]
    exposure_type = impf_dict["exposure_type"]
    exposure_source = impf_dict["exposure_source"]
    impact_type = impf_dict["impact_type"]
    hazard_abbr = impf_dict["hazard_abbr"]
    
    output_dir = impf_dict["impact_dir"]
    os.makedirs(output_dir, exist_ok=True)

    haz_filepath_dict = utils_config.gather_hazard_metadata(hazard_type, hazard_source, flatten=False)
    if scenario is not None:
        haz_filepath_dict = {scenario: haz_filepath_dict[scenario]}

    exp = None
    if scale_impacts:
        scale_impf = impf_dict.get("scale_impf", 1.0)
    else:
        scale_impf = 1

    # Load impact function using ImpactFunctionManager
    impf_manager = ImpactFunctionManager(impf_dict["impf_file_path"], hazard_abbr)
    impf = impf_manager.load_impf(scale_mdd=scale_impf)
    impf_set = ImpactFuncSet([impf])
    impf_id = impf.id

    all_observations = load_observations(
        exposure_type=exposure_type,
        impact_type=None,
        load_exceedance=True,
        load_supplementary_sources=True
    )
    impact_type_list = list(set([impf_dict["impact_type"]]) | set(all_observations["impact_type"]) | set(impf_dict["thresholds"].keys()))

    impact_path_list = []

    # This will need refactoring if we ever create different exposures for the future
    for scenario, haz_filepath_list in haz_filepath_dict.items():
        if scenario == "present":
            assert len(haz_filepath_list) == 1, "Some of this calibrating math assumes only one hazard file for the present"

        for i, haz_filepath in enumerate(haz_filepath_list):
            if len(haz_filepath_list) > 1:
                print(f'... processing hazard file {i+1} / {len(haz_filepath_list)}')

            impact_path_test_list = [
                impf_dict.impact_file_path(Path(haz_filepath).stem, t)
                for t in impact_type_list
            ]
            if np.all([os.path.exists(fp) for fp in impact_path_test_list]) and not overwrite:
                print(f'... all impact files exist already, skipping')
                continue

            haz = Hazard.from_hdf5(haz_filepath)
            exp = exp if exp else get_exposures(impf_dict["exposure_node"], impf_dict["exposure_dir"], scenario="present", impf_id=impf_id)

            # TO DO come back and work on this: we probably want to downscale the low-res exposures!!
            threshold_exp = estimate_matching_threshold(haz.centroids.coord)
            threshold_haz = estimate_matching_threshold(np.stack([exp.gdf.geometry.y.values, exp.gdf.geometry.x.values], axis=1))
            threshold = max(threshold_exp, threshold_haz)
            exp.assign_centroids(haz, distance="euclidean", threshold=threshold, overwrite=True)

            # Calculate main impact – economic value or displacement
            exp_temp = exp.copy()
            if impf_dict["impact_type"] == 'economic_loss':
                total_exposed_value = get_total_exposed_value(exposure_type, usd=True)
                exp_temp.value_unit = 'USD'
            else:
                total_exposed_value = get_total_exposed_value(exposure_type, usd=False)
                exp_temp.value_unit = get_total_exposed_units(exposure_type, usd=False)
            exp_temp = scale_exposures(exp_temp, total_exposed_value)
            imp = ImpactCalc(exp_temp, impf_set, haz).impact(save_mat=True, assign_centroids=False)
            
            impact_path = impf_dict.impact_file_path(Path(haz_filepath).stem, impact_type)
            print(f'... writing impact to {impact_path}')
            imp.write_hdf5(impact_path)
            impact_path_list.append(impact_path)

            # Calculate threshold impacts - affected/damaged/destroyed
            total_exposed_value = get_total_exposed_value(impf_dict["exposure_type"], usd=False)
            exp_temp = exp.copy()
            exp_temp = scale_exposures(exp_temp, total_exposed_value)
            exp_temp.value_unit = get_total_exposed_units(exposure_type, usd=False)
            assert exp_temp.gdf['impf_'].values[0] == 1, 'Ah, this code assumed that all the impact functions involved have ID 1'
            hazard_abbr = impf_dict['hazard_abbr']

            for i_type in impact_type_list:
                # There are two cases here: calculating for pre-specified thresholds, and searching for an optimal value
                parameter_search = False
                if fit_thresholds is not None:
                    observations = load_observations(
                        exposure_type=exposure_type,
                        impact_type=i_type,
                        load_exceedance=True,
                        load_supplementary_sources=True
                    )
                    if observations.shape[0] > 0:
                        parameter_search = True
                    else:
                        print(f"No observations for {exposure_type} - {i_type}: skipping threshold parameter search")

                if not parameter_search:
                    if i_type not in impf_dict["thresholds"].keys():
                        continue
                    threshold = impf_dict["thresholds"][i_type]
                    impf_step_set = impf_manager.create_step_function(impf, threshold)
                    imp = ImpactCalc(exp_temp, impf_step_set, haz).impact(save_mat=True, assign_centroids=False)
                else:
                    if i_type not in valid_thresholds:
                        continue
                    threshold, imp, _, _ = guess_threshold(i_type, impf_dict, impf, haz, exp, observations, rp_level=fit_thresholds)
                    impf_dict["thresholds"][i_type] = threshold

                impact_path = impf_dict.impact_file_path(Path(haz_filepath).stem, i_type)
                print(f'... writing impact to {impact_path}')
                imp.write_hdf5(impact_path)
                impact_path_list.append(impact_path)

            if scenario=="present":
                print('...Calculating score')  # Since total score is different from score for fitting single thresholds
                _, scores = analyse_exceedance(impf_dict, scenario="present", write_extras=write_extras, overwrite=True)
                impf_dict["scores"] = scores if scores.shape[0] > 0 else None
            else:
                impf_dict["scores"] = None

    impf_dict['impact_files'] = impact_path_list
    return impf_dict




def guess_threshold(threshold_name, impf_dict, impf, haz, exp, observations, rp_level):
    count = 1
    guesses = []
    scores = []
    next_guess = 0.5 if (threshold_name not in impf_dict["thresholds"].keys()) else impf_dict["thresholds"][threshold_name]
    improvement = None
    best_guess = None
    best_imp = None
    assert observations.shape[0] > 0
    assert rp_level in ["lower", "mid", "upper"]

    while (count <= max_iters) and (improvement is None or improvement > 0.001):
        previous_score = None if len(scores) == 0 else scores[-1]
        print(f'... iterating observations fitting: step {count} - guess {next_guess} - previous score {previous_score}')
        imp, guesses, scores = evaluate_one_guess(impf_dict, impf, haz, exp, observations, threshold_name, rp_level, guesses, scores, next_guess)
        next_guess = choose_next_guess(guesses, scores)
        improvement = calc_improvement(scores)
        if scores[-1] == np.min(scores):
            best_imp = imp
            best_guess = guesses[-1]
        count = count + 1

    best_score = np.min(scores)  
    working = {'guesses': guesses, 'scores': scores}      
    print(f'Done: selected threshold {best_guess} after {len(scores)} iterations. Final score: {best_score}')

    return best_guess, best_imp, best_score, working


def evaluate_one_guess(impf_dict, impf, haz, exp, observations, threshold_name, rp_level, guesses, scores, next_guess):
    impf_step_set = ImpactFunctionManager.create_step_function(impf, next_guess)
    imp = ImpactCalc(exp, impf_step_set, haz).impact(save_mat=True, assign_centroids=False)
    total_exposed = exp.value.sum()
    curves = get_curves(scenario="present", impf_dict=impf_dict, imp=imp, impact_type=threshold_name, haz_filepath=None, total_exposed=total_exposed)
    _, score_df = compare_obs(impf_dict, curves, observations)
    this_score = score_df.loc[threshold_name, f"weighted_cost_{rp_level}"]

    guesses.append(next_guess)
    scores.append(this_score)
    return imp, guesses, scores


def choose_next_guess(guesses, scores):
    if len(guesses) == 0:
        return 0.5
    if len(guesses) == 1:
        return guesses[0]/2

    i_sort = np.argsort(np.array(guesses))
    guesses_sorted = np.array(guesses)[i_sort]
    scores_sorted = np.array(scores)[i_sort]
    ix_min = np.argmin(scores_sorted)
    if ix_min == 0:
        return guesses_sorted[0]/2
    if ix_min == len(guesses_sorted)-1:
        return (guesses_sorted[ix_min] + 1)/2
    d_guesses = np.diff(guesses_sorted)
    if d_guesses[ix_min-1] > d_guesses[ix_min]:
        return (guesses_sorted[ix_min - 1] + guesses_sorted[ix_min]) / 2
    return (guesses_sorted[ix_min + 1] + guesses_sorted[ix_min]) / 2


def calc_improvement(scores):
    if len(scores) < 2:
        return None
    best_yet = np.min(scores[:-1])
    return np.abs((scores[-1] - best_yet)/best_yet)


def scale_exposures(exp, total_exposed_value):
    gdf_new = exp.gdf.copy()
    gdf_new['value'] = gdf_new['value'] * total_exposed_value / gdf_new['value'].sum()
    exp.set_gdf(gdf_new)
    return exp


def get_exposures(exposure_node, exp_dir, scenario="present", impf_id=None):
    exp_files = exposure_node["files"]
    if not isinstance(exp_files, list):
        exp_files = [exp_files]
    exp = Exposures.concat([get_one_exposure(Path(exp_dir, fn), impf_id) for fn in exp_files])
    return exp



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



def main(overwrite, scale_impacts):
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
    for analysis_name in set(conf["default_analysis_name"], conf["uncalibrated_analysis_name"]):
        print(f"=== Running analysis: {analysis_name} ===")

        for impf_dict_analysis in impf_list:
            impf_dict = copy.deepcopy(impf_dict_analysis)
            impf_dict['analysis_name'] = analysis_name

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
                    scenario=None,
                    scale_impacts=scale_impacts,
                    fit_thresholds=None,
                    write_extras=True,
                    overwrite=overwrite
                )
            except Exception as e:
                print(' ERROR: Failed to calculate impacts')
                print(f'{e}')
                raise Exception(e)
                # continue


if __name__ == "__main__":
    main(overwrite=True, scale_impacts=False)