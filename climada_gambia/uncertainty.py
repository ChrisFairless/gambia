from pathlib import Path
import copy
import scipy as sp
import numpy as np
import pandas as pd
import random
from functools import partial
import os
import matplotlib.pyplot as plt

from climada_gambia.metadata_impact import MetadataImpact
from climada_gambia.impact_function_manager import ImpactFunctionManager
from climada_gambia.utils_observations import RP_LEVELS
from climada_gambia import utils_config
from climada_gambia.calculate_impacts import get_exposures, estimate_matching_threshold, get_total_exposed_value, get_total_exposed_units, scale_exposures
from climada_gambia.calibrate_search import read_calibrated_scaling_factors
from climada_gambia.metadata_calibration import MetadataCalibration
from climada_gambia.analyse_impacts import COLOURS
from climada_gambia.scoring import calc_aal

from climada.hazard import Hazard
from climada.entity import ImpactFuncSet
from climada.engine.unsequa import InputVar, UncOutput, CalcImpact

overwrite = True

ANALYSIS_NAME = "calibration"
N_SIMULATIONS = 2**4
PARALLELISE = True
SEED = 1312
POP_GROWTH_2050 = 4301896 / 2822093 
# Ratio of 2050 population to 2020 population in Gambia
# https://population.un.org/dataportal/data/indicators/50/locations/270/start/1990/end/2101/table/pivotbylocation?df=e7a8df93-4260-4cfa-b003-6266ed070932

IMPF_UNCERTAINTY = 0.3   # how much to vary impact functions
# TODO adjust this by sector based on our confidence in observations

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

plot_exposure_types = ['agriculture', 'livestock', 'manufacturing', 'energy', 'services']

YMAX = 7.5  # force y-axis limit (percent), Set to None to auto-adjust

# Run an uncertainty analysis
# ---------------------------

# Define uncertainty in hazard (based on different models)
def generate_haz_base(i_model, haz):
    all_models = [s.split('_')[0] for s in haz.event_id]
    n_models = len(np.unique(all_models))
    i_model = int(i_model) % n_models
    sel_events = np.array([s for s in haz.event_id if s.startswith(f"{i_model}_")])
    return haz.select(event_id=sel_events)

haz_choice_distr = {
    "i_model": sp.stats.randint(low=0, high=1000),
}


# Define uncertainty in the exposure value
# actually, since we're normalising by total exposure, this is going to mess up our calculations ... let's keep this here for now but not properly use it
def generate_exp_base(exp_scale, exp):
    exp = exp.copy()
    exp.gdf["value"] *= exp_scale
    return exp

exp_scaling_distr = {
    # "exp_scale": sp.stats.uniform(0.75, 0.5) # uniform distribution between 0.75 and 1.25
    "exp_scale": sp.stats.uniform(0.99, 0.02)
}


# Define uncertainty in impact functionss based on uncertainty
def generate_impf_sets_base(i_rp_level: int, y_scale: float, impf_dict: MetadataImpact):
    rp_level = RP_LEVELS[int(i_rp_level)]
    scaling_factors = read_calibrated_scaling_factors(impf_dict, rp_level, a_little_judgement=True)
    scaling_factors['scale_y'] = y_scale * scaling_factors['scale_y']
    impf_manager = ImpactFunctionManager(impf_dict.impact_function_path(), impf_dict["hazard_abbr"])
    return impf_manager.load_impfset(scale_x=scaling_factors["scale_x"], scale_y=scaling_factors["scale_y"])


# This we don't use - it's not really compatible with the way unsequa works. But maybe later
def unused_generate_impf_sets(i_rp_level: int, y_scale: float, impf_dict: MetadataImpact):
    scaling_factors = read_calibrated_scaling_factors(impf_dict, i_rp_level, a_little_judgement=True)
    scaling_factors['scale_y'] = y_scale * scaling_factors['scale_y']
    impf_manager = ImpactFunctionManager(impf_dict.impact_function_path(), impf_dict["hazard_abbr"])
    impf = impf_manager.load_impf(scale_x=scaling_factors["scale_x"], scale_y=scaling_factors["scale_y"])
    impf_id = impf.id
    impf_output = {
        impf_dict['impact_type']: ImpactFuncSet([impf]),
        "thresholds": {}
    }
    for threshold in impf_dict["thresholds"]:
        # adjusts thresholds automatically based on earlier scaling
        impf_thresh = impf_manager.create_step_function(impf, threshold, impf_id)
        impf_output["thresholds"][threshold] = ImpactFuncSet([impf_thresh])
    return impf_output

impact_scaling_distr = {
    "i_rp_level": sp.stats.randint(low=0, high=3),
    "y_scale": sp.stats.uniform(1 - IMPF_UNCERTAINTY, 2 * IMPF_UNCERTAINTY)  # uniform distribution between 0.8 and 1.2
}

def set_rv_seeds(d: dict, seed=SEED):
    for k, v in d.items():
        v.random_state = np.random.RandomState(seed)

# set seed to align number generation:
set_rv_seeds(haz_choice_distr, seed=SEED)
set_rv_seeds(exp_scaling_distr, seed=SEED)
set_rv_seeds(impact_scaling_distr, seed=SEED)




def run_uncertainty_analysis(impf_dict: MetadataImpact, n_simulations: int = N_SIMULATIONS, overwrite: bool = False):
    exp = get_exposures(
        exposure_node = impf_dict['exposure_node'],
        exp_dir = impf_dict.exposure_dir(), 
        impf_id = 1
    )
    haz_filepath_dict = utils_config.gather_hazard_metadata(impf_dict["hazard_type"], impf_dict["hazard_source"], flatten=False)
    distance_match_threshold = None
    try:
        _ = read_calibrated_scaling_factors(impf_dict, RP_LEVELS[0], a_little_judgement=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Calibrated scaling factors not found for impact function at {impf_dict.impact_function_path()}. The uncertainty analysis has to be run on the outputs of a calibration (for now).")

    for scenario, haz_file_list in haz_filepath_dict.items():
        uncertainty_output_paths = impf_dict.uncertainty_results_paths(scenario=scenario, create=True)
        if os.path.exists(uncertainty_output_paths["rps"]) and not overwrite:
            print(f"Uncertainty results file already exists at {uncertainty_output_paths['rps']} and overwrite=False, skipping calculation.")
            continue

        print(f"Setting up uncertainty analysis for scenario: {scenario}")
        # Here we combine all models into a single Hazard file and label each event with which model it came from, so that we can sample by model later.
        # We don't adjust the frequencies since we'll never calculate forr the whole event set.
        haz_all = [Hazard.from_hdf5(haz_file) for haz_file in haz_file_list]
        assert all([np.all(haz.frequency == haz_all[0].frequency) for haz in haz_all]), "All hazard files must have the same frequencies for this sampling method to work."
        haz_model_id = np.concatenate([[i] * len(haz.event_id) for i, haz in enumerate(haz_all)])
        haz_all = Hazard.concat(haz_all)  # we need all models in one file so we can assign points once instead of every time we sample
        haz_all.event_id = np.array([f"{id}_{event_id}" for id, event_id in zip(haz_model_id, haz_all.event_id)])

        haz_single = Hazard.from_hdf5(haz_file_list[0])
        haz_rps = np.array([1/f for f in np.cumsum(np.sort(haz_single.frequency))])[::-1]

        if not distance_match_threshold:  # We assume all hazard files in all calculations have the same resolution
            distance_match_threshold_haz = estimate_matching_threshold(haz_all.centroids.coord)
            distance_match_threshold_exp = estimate_matching_threshold(np.stack([exp.gdf.geometry.y.values, exp.gdf.geometry.x.values], axis=1))
            distance_match_threshold = max(distance_match_threshold_exp, distance_match_threshold_haz)
        exp.assign_centroids(haz_all, distance="euclidean", threshold=distance_match_threshold, overwrite=True)

        if impf_dict["impact_type"] == 'economic_loss':
            total_exposed_value = get_total_exposed_value(impf_dict["exposure_type"], usd=True)
            value_unit = 'USD'
        else:
            total_exposed_value = get_total_exposed_value(impf_dict["exposure_type"], usd=False)
            value_unit = get_total_exposed_units(impf_dict["exposure_type"], usd=False)
        exp = scale_exposures(exp, total_exposed_value, copy=True)
        exp.value_unit = value_unit
        
        # Define uncertainty inputs:
        generate_haz = partial(generate_haz_base, haz=haz_all)
        haz_iv = InputVar(generate_haz, haz_choice_distr)
        
        # For now, no exposure uncertainty
        # generate_exp = partial(generate_exp_base, exp=exp_temp)
        # exp_iv = InputVar(generate_exp, exp_scaling_distr)
        
        generate_impf_sets = partial(generate_impf_sets_base, impf_dict=impf_dict)
        impf_iv = InputVar(generate_impf_sets, impact_scaling_distr)

        calc_imp = CalcImpact(exp, impf_iv, haz_iv)
        output_imp = calc_imp.make_sample(N=N_SIMULATIONS, sampling_kwargs={"skip_values": 2**8})
        n_processes = PARALLELISE if PARALLELISE is not None else 1

        print(f"Running uncertainty analysis for scenario: {scenario}")
        output_imp = calc_imp.uncertainty(output_imp, rp=haz_rps, processes=n_processes)

        print(f"Generating outputs for scenario: {scenario}")
        uncertainty_df = output_imp.get_uncertainty()
        uncertainty_df.to_csv(uncertainty_output_paths["csv"], index=False)

        fig = output_imp.plot_uncertainty()
        plt.savefig(uncertainty_output_paths["plot"])

        fig = output_imp.plot_rp_uncertainty()
        plt.savefig(uncertainty_output_paths["rps"])
        plt.close('all')


def plot_uncertainty_analyses(impf_dict_list: list, quantiles: tuple=(0.1, 0.9), overwrite=False, **kwargs):
    print(f"Plotting uncertainty analyses")
    impf_dict_list = copy.deepcopy(impf_dict_list)
    if plot_exposure_types is not None:
        impf_dict_list = [impf_dict for impf_dict in impf_dict_list if impf_dict['exposure_type'] in plot_exposure_types]

    df_all = gather_uncertainty_results(impf_dict_list, quantiles=quantiles, normalise_by_exposure=False)
    df_all_norm = gather_uncertainty_results(impf_dict_list, quantiles=quantiles, normalise_by_exposure=True)

    sector_sizes = {impf_dict['exposure_type']: get_total_exposed_value(impf_dict['exposure_type'], usd=True) for impf_dict in impf_dict_list}
    sector_sizes_df = pd.DataFrame(sector_sizes.items(), columns=['exposure_type', 'total_exposed'])
    sector_sizes_df['weight'] = sector_sizes_df['total_exposed'] / sector_sizes_df['total_exposed'].sum()
    total_exposed_litpop = get_total_exposed_value('economic_assets', usd=True)
    total_exposed_ratio = total_exposed_litpop / sector_sizes_df['total_exposed'].sum()
    assert np.isclose(sum(sector_sizes_df['weight']), 1.0), "Sector weights do not sum to 1.0"
    print(f"Using an adjustment factor of {total_exposed_ratio:.2f} to normalise impacts to match LitPop")

    impf_dict_all = {
        'exposure_type': 'all',
        'impact_type': 'economic_loss'
    }
    impf_dict_list = [impf_dict_all] + impf_dict_list

    metadata_calibration = MetadataCalibration(analysis_name=ANALYSIS_NAME)
    output_paths = metadata_calibration.calibration_output_paths(create=True)

    # Plot present-day uncertainty for all exposures
    out_path = output_paths["present_comparison"]
    if os.path.exists(out_path) and not overwrite:
        print(f"Present-day comparison plot already exists at {out_path} and overwrite=False, skipping plotting.")
    else:
        fig, axes = plt.subplots(ncols=2, nrows=len(impf_dict_list)//2, figsize=(10, 3*len(impf_dict_list)//2))
        axes = axes.flatten()
        if YMAX is None:
            ymax = df_all_norm[
                    (df_all_norm['scenario'] == 'present') & 
                    (df_all_norm['rp'] <= 100)
                ]['high'].values.max()
        else:
            ymax = YMAX
        for ax, impf_dict in zip(axes, impf_dict_list):
            exposure_type = impf_dict['exposure_type']
            if exposure_type == 'all':
                df_exp = df_all_norm[
                    (df_all_norm['scenario'] == 'present') & 
                    (df_all_norm['rp'] <= 100)
                ].merge(sector_sizes_df, on='exposure_type', how='left').reset_index(drop=True)
                df_exp[['median', 'low', 'high']] = df_exp[['median', 'low', 'high']].mul(df_exp['weight'], axis=0)
                df_exp[['median', 'low', 'high']] = df_exp.groupby(['scenario', 'rp', 'rp_name'])[['median', 'low', 'high']].transform('sum')
                df_exp = df_exp.sort_values('rp')
            else:
                df_exp = df_all_norm[
                    (df_all_norm['exposure_type'] == exposure_type) & 
                    (df_all_norm['scenario'] == 'present') & 
                    (df_all_norm['rp'] <= 100)
                ].sort_values('rp')
            aal = df_all_norm[df_all_norm['rp_name'] == 'aai_agg'].loc[:, 'median'].values[0]
            df_exp = df_exp[df_exp['rp'] >= 0]
            ax.fill_between(df_exp['rp'], df_exp['low'], df_exp['high'], color='blue', alpha=0.3, label=f'{quantiles[0]*100}-{quantiles[1]*100}th percentile range')
            ax.plot(df_exp['rp'], df_exp['median'], color='black', label=f'Median')
            # ax.axhline(aal, color='blue', linestyle='--')
            ax.set_ylim(0, ymax)
            ax.set_xlabel('Return Period (years)')
            ax.set_ylabel(f'Total impact (%)')
            ax.set_title(f'{exposure_type.title()}')

        ax.legend()
        plt.suptitle("Exceedance curves for economic loss: present-day (% impact)", fontsize=16)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close('all')

    # Plot present and future uncertainty for all exposures
    out_path = output_paths["future_comparison"]
    if os.path.exists(out_path) and not overwrite:
        print(f"Present vs future comparison plot already exists at {out_path} and overwrite=False, skipping plotting.")
    else:
        fig, axes = plt.subplots(ncols=2, nrows=len(impf_dict_list)//2, figsize=(10, 3*len(impf_dict_list)//2))
        axes = axes.flatten()
        if YMAX is None:
            ymax = df_all_norm[
                    (df_all_norm['rp'] <= 100)
                ]['high'].values.max()
        else:
            ymax = YMAX
        for ax, impf_dict in zip(axes, impf_dict_list):
            exposure_type = impf_dict['exposure_type']
            # Present
            if exposure_type == 'all':
                df_exp = df_all_norm[
                    (df_all_norm['scenario'] == 'present') & 
                    (df_all_norm['rp'] <= 100) 
                ].merge(sector_sizes_df, on='exposure_type', how='left').reset_index(drop=True)
                df_exp[['median', 'low', 'high']] = df_exp[['median', 'low', 'high']].mul(df_exp['weight'], axis=0)
                df_exp[['median', 'low', 'high']] = df_exp.groupby(['scenario', 'rp', 'rp_name'])[['median', 'low', 'high']].transform('sum')
                df_exp = df_exp.sort_values('rp')
            else:
                df_exp = df_all_norm[
                    (df_all_norm['exposure_type'] == exposure_type) & 
                    (df_all_norm['scenario'] == 'present') & 
                    (df_all_norm['rp'] <= 100)
                ].sort_values('rp')
            aal = df_all_norm[df_all_norm['rp_name'] == 'aai_agg'].loc[:, 'median'].values[0]
            df_exp = df_exp[df_exp['rp'] >= 0]
            ax.fill_between(df_exp['rp'], df_exp['low'], df_exp['high'], color='blue', alpha=0.3, label=f'{quantiles[0]*100}-{quantiles[1]*100}th percentile range (present)')
            ax.plot(df_exp['rp'], df_exp['median'], color='black', label=f'Median (present)')
            # ax.axhline(aal, color='blue', label='AAL (present)', linestyle='--')

            # Future
            if exposure_type == 'all':
                df_exp = df_all_norm[
                    (df_all_norm['scenario'] == 'RCP8.5_2050') & 
                    (df_all_norm['rp'] <= 100)
                ].merge(sector_sizes_df, on='exposure_type', how='left').reset_index(drop=True) 
                df_exp[['median', 'low', 'high']] = df_exp[['median', 'low', 'high']].mul(df_exp['weight'], axis=0)
                df_exp[['median', 'low', 'high']] = df_exp.groupby(['scenario', 'rp', 'rp_name'])[['median', 'low', 'high']].transform('sum')
                df_exp = df_exp.sort_values('rp')
            else:
                df_exp = df_all_norm[
                    (df_all_norm['exposure_type'] == exposure_type) & 
                    (df_all_norm['scenario'] == 'RCP8.5_2050') & 
                    (df_all_norm['rp'] <= 100)
                ].sort_values('rp')
            aal = df_all_norm[df_all_norm['rp_name'] == 'aai_agg'].loc[:, 'median'].values[0]
            df_exp = df_exp[df_exp['rp'] >= 0]
            ax.fill_between(df_exp['rp'], df_exp['low'], df_exp['high'], color='gold', alpha=0.5, label=f'{quantiles[0]*100}-{quantiles[1]*100}th percentile range (RCP8.5 2050)')
            ax.plot(df_exp['rp'], df_exp['median'], color='darkorange', label=f'Median (RCP8.5 2050)')
            # ax.axhline(aal, color='darkorange', label='AAL (RCP8.5 2050)', linestyle='--')

            ax.set_ylim(0, ymax)
            ax.set_xlabel('Return Period (years)')
            ax.set_ylabel(f'Total impact (%)')
            ax.set_title(f'{exposure_type.title()}')

        ax.legend()
        plt.suptitle("Exceedance curves for economic loss: present-day vs 2050 (% impact)", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_paths["future_comparison"])
        plt.close('all')


    # Plot waterfall charts for all exposures
    out_path = output_paths["waterfall"]
    if os.path.exists(out_path) and not overwrite:
        print(f"Present vs future comparison plot already exists at {out_path} and overwrite=False, skipping plotting.")
    else:
        fig, axes = plt.subplots(ncols=2, nrows=len(impf_dict_list)//2, figsize=(10, 3*len(impf_dict_list)//2))
        axes = axes.flatten()
        for ax, impf_dict in zip(axes, impf_dict_list):
            exposure_type = impf_dict['exposure_type']
            # Present
            if exposure_type == 'all':
                curr_risk = df_all[
                    (df_all['rp_name'] == 'aai_agg') &
                    (df_all['scenario'] == 'present')
                ].loc[:, 'median'].values.sum() * total_exposed_ratio
            else:
                curr_risk = df_all[
                    (df_all['exposure_type'] == exposure_type) & 
                    (df_all['rp_name'] == 'aai_agg') &
                    (df_all['scenario'] == 'present')
                ].loc[:, 'median'].values[0]
            risk_dev = curr_risk * POP_GROWTH_2050

            # Future
            if exposure_type == 'all':
                fut_risk = df_all[
                    (df_all['rp_name'] == 'aai_agg') &
                    (df_all['scenario'] == 'RCP8.5_2050')
                ].loc[:, 'median'].values.sum() * total_exposed_ratio * POP_GROWTH_2050
            else:
                fut_risk = df_all[
                    (df_all['exposure_type'] == exposure_type) & 
                    (df_all['rp_name'] == 'aai_agg') &
                    (df_all['scenario'] == 'RCP8.5_2050')
                ].loc[:, 'median'].values[0] * POP_GROWTH_2050

            present_year = 2025
            future_year = 2050
            norm_fact = 1e6
            ax.bar(
                1,
                curr_risk / norm_fact,
                color="peru",
                **kwargs
            )
            ax.text(
                1,
                curr_risk / norm_fact,
                str(f"{curr_risk / norm_fact:.1f}"),
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=12,
                color="k",
            )
            ax.bar(
                2,
                height=(risk_dev - curr_risk) / norm_fact,
                bottom=curr_risk / norm_fact,
                color="yellowgreen",
                **kwargs
            )
            ax.text(
                2,
                curr_risk / norm_fact + (risk_dev - curr_risk) / norm_fact / 2,
                str(f"{(risk_dev - curr_risk) / norm_fact:.1f}"),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
                color="k",
            )
            ax.bar(
                3,
                height=(fut_risk - risk_dev) / norm_fact,
                bottom=risk_dev / norm_fact,
                color="gold",
                **kwargs
            )
            ax.text(
                3,
                risk_dev / norm_fact + (fut_risk - risk_dev) / norm_fact / 2,
                str(f"{(fut_risk - risk_dev) / norm_fact:.1f}"),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
                color="k",
            )
            ax.bar(
                4,
                height=fut_risk / norm_fact,
                color="chocolate",
                **kwargs)
            ax.text(
                4,
                fut_risk / norm_fact,
                str(f"{fut_risk / norm_fact:.1f}"),
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=12,
                color="k",
            )

            ax.set_xticks(np.arange(4) + 1)
            ax.set_xticklabels(
                [
                    "Risk " + str(present_year),
                    "Economic \ndevelopment",
                    "Climate \nchange",
                    "Risk " + str(future_year),
                ]
            )
            ax.set_ylabel("Impact (millions USD)")
            ax.set_title(f'{exposure_type.title()}')

        ax.legend()
        plt.suptitle("Sectoral changes in risk from {:d} to {:d}".format(present_year, future_year), fontsize=16)
        plt.tight_layout()
        plt.savefig(output_paths["waterfall"])
        plt.close('all')


def gather_uncertainty_results(impf_dict_list: list, quantiles: tuple=(0.1, 0.9), normalise_by_exposure: bool = True):
    df_all = []
    for impf_dict in impf_dict_list:
        total_exposure = get_total_exposed_value(impf_dict["exposure_type"], usd=(impf_dict["impact_type"]=='economic_loss'))

        for scenario in impf_dict["hazard_node"].keys():
            uncertainty_output_paths = impf_dict.uncertainty_results_paths(scenario=scenario, create=False)
            if not os.path.exists(uncertainty_output_paths["rps"]):
                raise FileNotFoundError(f"Uncertainty results file does not exist at {uncertainty_output_paths['rps']}, cannot plot results.")

            uncertainty_df = pd.read_csv(uncertainty_output_paths["csv"])
            uncertainty_aai = uncertainty_df['aai_agg']
            uncertainty_df = uncertainty_df.drop(columns=['aai_agg'])
            assert np.all([col.startswith('rp') for col in uncertainty_df.columns]), "Unexpected column names in uncertainty results file, expected columns starting with 'rp'"

            # reverse columns so that they're in ascending order of frequency rather than RP (for easier calculations)
            uncertainty_df = uncertainty_df[uncertainty_df.columns[::-1]]
            rp = pd.Series([float(s[2:len(s)+1]) for s in uncertainty_df.columns])

            # I don't trust CLIMADA's AAI calculations here, so we recalculate
            naive_frequency = np.array([1/x for x in rp])
            naive_frequency_shifted = np.concatenate([[0], naive_frequency[:-1]])
            freq = naive_frequency - naive_frequency_shifted

            partial_calc_aal = partial(calc_aal, frequency=freq)
            aai = uncertainty_df.apply(partial_calc_aal, axis=1)

            print([(x, y) for x, y in zip(uncertainty_aai.values, aai)])

            if normalise_by_exposure:
                norm = total_exposure
            else:
                norm = 1

            u_mean = 100 * uncertainty_df.mean() / norm
            u_median = 100 * uncertainty_df.quantile(0.5) / norm
            u_low = 100 * uncertainty_df.quantile(quantiles[0]) / norm
            u_high = 100 * uncertainty_df.quantile(quantiles[1]) / norm

            df = pd.DataFrame({
                'exposure_type': impf_dict['exposure_type'],
                'scenario': scenario,
                'rp_name': uncertainty_df.columns,
                'rp': rp,
                'mean': u_mean.values,
                'median': u_median.values,
                'low': u_low.values,
                'high': u_high.values
            }).set_index(['exposure_type', 'scenario', 'rp_name', 'rp'])
            df_all.append(df)
    df_all = pd.concat(df_all).reset_index()
    return df_all

def main(analysis_name, impf_filter, overwrite):
    impf_dict_list = utils_config.gather_impact_calculation_metadata(filter=impf_filter, analysis_name=analysis_name)
    if len(impf_filter) > 0:
        assert len(impf_dict_list) == 1, f'Expected one impact function for filter {impf_filter}, found {len(impf_dict_list)}'
    analyses_list = [f"{analysis_name}/{impf_dict['exposure_type']}_{impf_dict['exposure_source']}" for impf_dict in impf_dict_list]
    print(f"Running uncertainty analyses for {len(impf_dict_list)} impact calculations:")
    print(analyses_list)
    impf_dict_list = [MetadataImpact(impf_dict=impf_dict, analysis_name=n) for impf_dict, n in zip(impf_dict_list, analyses_list)]

    # for impf_dict in impf_dict_list:
    #     print("===================================================")
    #     print(f"Uncertainty analysis for impact:")
    #     print(f" Hazard: {impf_dict['hazard_type']} - {impf_dict['hazard_source']}")
    #     print(f" Exposure: {impf_dict['exposure_type']} - {impf_dict['exposure_source']}")
    #     print(f" Impact type: {impf_dict['impact_type']}")
    #     run_uncertainty_analysis(impf_dict, n_simulations=N_SIMULATIONS, overwrite=overwrite)
    
    plot_uncertainty_analyses(impf_dict_list, quantiles=(0.1, 0.9), overwrite=overwrite)
    print("Done")



if __name__ == "__main__":
    main(analysis_name=ANALYSIS_NAME, impf_filter=impf_filter, overwrite=overwrite)    