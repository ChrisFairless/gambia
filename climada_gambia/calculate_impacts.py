#!/usr/bin/env python3
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
import logging
import copy
from typing import Union, Dict, Any, List, Tuple, Optional

from climada.hazard import Hazard
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.engine import ImpactCalc, Impact
from climada.util.coordinates import estimate_matching_threshold

from climada_gambia.utils_total_exposed_value import get_total_exposed_value, get_total_exposed_units
from climada_gambia.utils_observations import load_observations, RP_LEVELS
from climada_gambia.config import CONFIG
from climada_gambia import utils_config
from climada_gambia.scoring import ScoringEngine, squared_error
from climada_gambia.analyse_impacts import make_exceedance_curve
from climada_gambia.impact_function_manager import ImpactFunctionManager
from climada_gambia.metadata_impact import MetadataImpact, VALID_THRESHOLD_IMPACT_TYPES

logging.getLogger("climada").setLevel(logging.WARNING)

ANALYSIS_NAME = "uncalibrated"
overwrite = True

max_iters: int = 16
SAVE_MAT: bool = False   # Save impact matrices: these are not needed anywhere but you might want them for mapping

def calculate_impacts(
    impf_dict_in: Union[MetadataImpact, Dict[str, Any]], 
    scenario: Optional[str], 
    fit_thresholds: Optional[str], 
    scale_impacts: bool = False, 
    write_extras: bool = True,
    overwrite: bool = True
) -> MetadataImpact:
    """Calculate impacts for a given impact calculation configuration.
    
    Args:
        impf_dict_in: MetadataImpact instance containing impact calculation configuration information
        scenario: Scenario name (e.g., 'present', 'RCP4.5_2050')
        fit_thresholds: Whether to fit thresholds. True fits to observation data. False values defined in the config.
        scale_impacts: Whether to scale impacts
        write_extras: Whether to write additional outputs
        overwrite: Whether to overwrite existing files
    """
    if isinstance(impf_dict_in, MetadataImpact):
        impf_dict = impf_dict_in  # MetadataImpact already encapsulates its data safely
    else:
        impf_dict = MetadataImpact(copy.deepcopy(impf_dict_in))
    
    hazard_type = impf_dict["hazard_type"]
    hazard_source = impf_dict["hazard_source"]
    exposure_type = impf_dict["exposure_type"]
    exposure_source = impf_dict["exposure_source"]
    impact_type = impf_dict["impact_type"]
    hazard_abbr = impf_dict["hazard_abbr"]
    
    haz_filepath_dict = utils_config.gather_hazard_metadata(hazard_type, hazard_source, flatten=False)
    if scenario is None:
        scenario_list = list(haz_filepath_dict.keys())
        if "present" in scenario_list:
            scenario_list = ["present"] + [s for s in scenario_list if s != "present"]  # We need present first
    else:
        assert scenario in haz_filepath_dict.keys(), f"Scenario '{scenario}' not found in hazard configuration for {hazard_type} - {hazard_source}"
        scenario_list = [scenario]
    
    if scale_impacts:
        scale_x = impf_dict.get("scale_x", 1.0)
        scale_y = impf_dict.get("scale_y", 1.0)
    else:
        scale_x = 1.0
        scale_y = 1.0

    # Load impact function using ImpactFunctionManager
    impf_manager = ImpactFunctionManager(impf_dict.impact_function_path(), hazard_abbr)
    impf = impf_manager.load_impf(scale_x=scale_x, scale_y=scale_y)
    impf_set = ImpactFuncSet([impf])
    impf_id = impf.id

    all_observations = load_observations(
        exposure_type=exposure_type,
        impact_type=None,
        load_exceedance=True,
        load_supplementary_sources=True
    )
    impact_type_list = impf_dict.impact_type_list(observations=all_observations)

    exp = get_exposures(impf_dict["exposure_node"], impf_dict.exposure_dir(), scenario="present", impf_id=impf_id)
    observations_exist = {
        i_type: all_observations[all_observations['impact_type'] == i_type].shape[0] > 0
        for i_type in impact_type_list
    }

    distance_match_threshold = None
    
    # This will need refactoring if we ever create different exposures for the future
    for scenario in scenario_list:
        haz_filepath_list = haz_filepath_dict[scenario]
        if scenario == "present":
            assert len(haz_filepath_list) == 1, "Some of this calibrating math assumes only one hazard file for the present"

        for i, haz_filepath in enumerate(haz_filepath_list):
            # TODO We could parallelise this loop, but not the (scenario, haz_filepath) tuples, since we have to process "present" first
            # Also it would be nice to split this calculation into smaller chunks, but doing it as one monster lets us avoid
            # assigning centroids to the same hazard-exposure pair more than once
            if len(haz_filepath_list) > 1:
                print(f'... processing hazard file {i+1} / {len(haz_filepath_list)}')

            impact_path_test_list = [
                impf_dict.impact_file_path(Path(haz_filepath).stem, t, create=True)
                for t in impact_type_list
            ]
            if np.all([os.path.exists(fp) for fp in impact_path_test_list]) and not overwrite:
                print(f'... all impact files seem to exist already, skipping calculations')
                continue

            haz = Hazard.from_hdf5(haz_filepath)

            # TODO come back and work on this: we probably want to downscale the low-res exposures!!
            if not distance_match_threshold:  # We assume all hazard files in all calculations have the same resolution
                distance_match_threshold_haz = estimate_matching_threshold(haz.centroids.coord)
                distance_match_threshold_exp = estimate_matching_threshold(np.stack([exp.gdf.geometry.y.values, exp.gdf.geometry.x.values], axis=1))
                distance_match_threshold = max(distance_match_threshold_exp, distance_match_threshold_haz)
            exp.assign_centroids(haz, distance="euclidean", threshold=distance_match_threshold, overwrite=True)

            # Calculate main impact – economic value or displacement
            if impf_dict["impact_type"] == 'economic_loss':
                total_exposed_value = get_total_exposed_value(exposure_type, usd=True)
                value_unit = 'USD'
            else:
                total_exposed_value = get_total_exposed_value(exposure_type, usd=False)
                value_unit = get_total_exposed_units(exposure_type, usd=False)
            exp_temp = scale_exposures(exp, total_exposed_value, copy=True)
            exp_temp.value_unit = value_unit
            imp = ImpactCalc(exp_temp, impf_set, haz).impact(save_mat=SAVE_MAT, assign_centroids=False)
            
            impact_path = impf_dict.impact_file_path(Path(haz_filepath).stem, impact_type, create=True)
            print(f'... writing impact to {impact_path}')
            imp.write_hdf5(impact_path)

            # Calculate threshold impacts - affected/damaged/destroyed
            total_exposed_value = get_total_exposed_value(impf_dict["exposure_type"], usd=False)
            exp_temp = scale_exposures(exp, total_exposed_value, copy=True)
            exp_temp.value_unit = get_total_exposed_units(exposure_type, usd=False)
            assert exp_temp.gdf['impf_'].values[0] == 1, 'Ah, this code assumed that all the impact functions involved have ID 1'

            # Fit impact thresholds if scenario == "present" and it's required
            relevant_thresholds = list(set(impact_type_list).intersection(VALID_THRESHOLD_IMPACT_TYPES))
            fitted_thresholds = {i_type: {} for i_type in relevant_thresholds}
            if scenario == "present" and fit_thresholds and len(relevant_thresholds) > 0:
                print('Fitting thresholds to observations')
                
                for i_type in relevant_thresholds:
                    # There are two cases here: calculating for pre-specified thresholds, and searching for an optimal value
                    if not observations_exist[i_type]:
                        print(f"No observations for {exposure_type} - {i_type}: skipping threshold parameter search")
                        continue
                    if i_type not in impf_dict["thresholds"].keys():
                        raise ValueError(f"No predefined threshold for impact type '{i_type}' in impact function configuration for {exposure_type}. This will be required for later comparisons with observations: please specify one in the config file.")

                    # Looks like we're fitting thresholds for this impact type
                    observations = all_observations[all_observations['impact_type'] == i_type]
                    for rp_level in RP_LEVELS:
                        threshold, _, _, _ = guess_threshold(i_type, impf_dict, impf, haz, exp, observations, rp_level)
                        fitted_thresholds[i_type][rp_level] = threshold
                
                write_fitted_thresholds_output(impf_dict, fitted_thresholds, overwrite=overwrite)
                # impf_dict["fitted_thresholds"] = fitted_thresholds
            
            # Calculate impacts for each threshold type
            for i_type in relevant_thresholds:
                unfitted_threshold = impf_dict["thresholds"][i_type]
                chosen_threshold = fitted_thresholds[i_type].get('mid', unfitted_threshold)
                impf_step_set = impf_manager.create_step_function(impf, chosen_threshold)
                imp = ImpactCalc(exp_temp, impf_step_set, haz).impact(save_mat=SAVE_MAT, assign_centroids=False)
                impact_path = impf_dict.impact_file_path(Path(haz_filepath).stem, i_type, create=True)
                print(f'... writing impact to {impact_path}')
                imp.write_hdf5(impact_path)

                for rp_level in RP_LEVELS:
                    fitted_threshold = fitted_thresholds[i_type].get(rp_level, None)
                    if not fitted_threshold:
                        continue
                    impf_step_set = impf_manager.create_step_function(impf, fitted_threshold)
                    imp = ImpactCalc(exp_temp, impf_step_set, haz).impact(save_mat=SAVE_MAT, assign_centroids=False)
                    impact_path = impf_dict.impact_rp_level_file_path(Path(haz_filepath).stem, i_type, rp_level, create=True)
                    print(f'... writing impact to {impact_path}')
                    imp.write_hdf5(impact_path)

    return impf_dict


def guess_threshold(
    threshold_name: str, 
    impf_dict: Union[MetadataImpact, Dict[str, Any]], 
    impf: ImpactFunc, 
    haz: Hazard, 
    exp: Exposures, 
    observations: pd.DataFrame, 
    rp_level: str
) -> Tuple[float, Impact, float, Dict[str, List[float]]]:
    count = 1
    guesses = []
    scores = []
    next_guess = 0.5 if (threshold_name not in impf_dict["thresholds"].keys()) else impf_dict["thresholds"][threshold_name]
    improvement = None
    best_guess = None
    best_imp = None
    assert observations.shape[0] > 0
    assert rp_level in RP_LEVELS

    while (count <= max_iters) and (improvement is None or improvement > 0.001):
        previous_score = None if len(scores) == 0 else scores[-1]
        # print(f'... iterating observations fitting: step {count} - guess {next_guess} - previous score {previous_score}')
        imp, this_score = evaluate_one_guess(impf_dict, impf, haz, exp, observations, threshold_name, rp_level, next_guess)
        guesses.append(next_guess)
        scores.append(this_score)
        next_guess = choose_next_guess(guesses, scores)
        improvement = calc_improvement(scores)
        if scores[-1] == np.min(scores):
            best_imp = imp
            best_guess = guesses[-1]
        count = count + 1

    best_score = np.min(scores)  
    working = {'guesses': guesses, 'scores': scores}      
    print(f'Done: selected {threshold_name} threshold {best_guess} after {len(scores)} iterations. Final score: {best_score}')
    return best_guess, best_imp, best_score, working


def evaluate_one_guess(
    impf_dict: Union[MetadataImpact, Dict[str, Any]], 
    impf: ImpactFunc, 
    haz: Hazard, 
    exp: Exposures, 
    observations: pd.DataFrame, 
    threshold_name: str, 
    rp_level: str, 
    next_guess: float
) -> Tuple[Impact, float]:
    impf_step_set = ImpactFunctionManager.create_step_function(impf, next_guess)
    imp = ImpactCalc(exp, impf_step_set, haz).impact(save_mat=SAVE_MAT, assign_centroids=False)
    total_exposed = exp.value.sum()
    curves = make_exceedance_curve(scenario="present", impf_dict=impf_dict, imp=imp, impact_type=threshold_name, haz_filepath=None, total_exposed=total_exposed)
    
    # Use ScoringEngine to compare observations
    engine = ScoringEngine(cost_function=squared_error)
    _, score_df = engine.compare_to_observations(curves, observations, impf_dict)
    this_score = score_df.loc[threshold_name, f"weighted_cost_{rp_level}"]
    return imp, this_score


def choose_next_guess(guesses: List[float], scores: List[float]) -> float:
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


def calc_improvement(scores: List[float]) -> Optional[float]:
    if len(scores) < 2:
        return None
    best_yet = np.min(scores[:-1])
    return np.abs((scores[-1] - best_yet)/best_yet)


def scale_exposures(exp: Exposures, total_exposed_value: float, copy: bool = True) -> Exposures:
    gdf_new = exp.gdf.copy() if copy else exp.gdf
    gdf_new['value'] = gdf_new['value'] * total_exposed_value / gdf_new['value'].sum()
    exp.set_gdf(gdf_new)
    return exp


def get_exposures(
    exposure_node: Dict[str, Any], 
    exp_dir: Path, 
    scenario: str = "present", 
    impf_id: Optional[int] = None
) -> Exposures:
    if scenario != "present":
        raise NotImplementedError("Currently only 'present' scenario exposures are supported")
    exp_files = exposure_node["files"]
    if not isinstance(exp_files, list):
        exp_files = [exp_files]
    exp = Exposures.concat([get_one_exposure(Path(exp_dir, fn), impf_id) for fn in exp_files])
    return exp


def get_one_exposure(filepath: Path, impf_id: Optional[int] = None) -> Exposures:
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


def write_fitted_thresholds_output(
    impf_dict: MetadataImpact, 
    fitted_thresholds: Dict[str, Dict[str, float]], 
    overwrite: bool = True
) -> None:
    fitted_output = pd.DataFrame({
            'impact_type': i_type,
            'rp_level': rp_level,
            'prior_threshold': impf_dict["thresholds"][i_type],
            'fitting_performed': rp_level in fitted_thresholds[i_type],
            'fitted_threshold': fitted_thresholds[i_type].get(rp_level, None)
        }
        for rp_level in RP_LEVELS
        for i_type in fitted_thresholds.keys()
    )
    fitted_output['threshold_final'] = fitted_output.apply(
        lambda row: row['fitted_threshold'] if row['fitting_performed'] else row['prior_threshold'],
        axis=1
    )
    fitted_output_path = impf_dict.fitted_thresholds_file_path(create=True)
    fitted_output.to_csv(fitted_output_path, index=False)


def main(analysis_name: str, overwrite: bool, scale_impacts: bool) -> None:
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
                fit_thresholds=False,
                write_extras=True,
                overwrite=overwrite
            )
        except Exception as e:
            print(' ERROR: Failed to calculate impacts')
            print(f'{e}')
            raise Exception(e)
            # continue


if __name__ == "__main__":
    main(analysis_name=ANALYSIS_NAME, overwrite=overwrite, scale_impacts=False)