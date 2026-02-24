#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from climada.entity import Exposures
from climada.engine import Impact

from climada_gambia.check_inputs import check_node, check_enabled_node
from climada_gambia.metadata_calibration import MetadataCalibration
from climada_gambia.metadata_impact import MetadataImpact, VALID_THRESHOLD_IMPACT_TYPES
from climada_gambia import utils_config
from climada_gambia.utils_observations import load_observations, RP_LEVELS
from climada_gambia.utils_total_exposed_value import get_total_exposed_value
from climada_gambia.config import CONFIG
from climada_gambia.scoring import ScoringEngine, squared_error, calc_aal

analysis_name = "uncalibrated"

base_figsize = (16, 7)

overwrite = True

COLOURS = {
    'present': {
        'strong': 'black',
        'normal': 'lightgrey'
    },
    'RCP4.5_2050': {
        'strong': 'darkorange',
        'normal': 'gold'
    },
    'RCP8.5_2050': {
        'strong': 'red',
        'normal': 'pink'
    },
    'observations': {
        'strong': ['mediumblue', 'royalblue', 'teal'],
        'normal': 'lightblue'
    }
}

COLOURS2 = ['teal', 'coral', 'gold', 'lightgrey', 'lightblue', 'lightgreen', 'violet', 'ivory']


def analyse_impf_exceedance(impf_dict: MetadataImpact, scenario=None, make_plots=True, write_extras=True, overwrite=True):
    """Analyse exceedance curves for impacts.
    
    Args:
        impf_dict: MetadataImpact instance containing impact function configuration
        scenario: Scenario name or None for all scenarios
        write_extras: Whether to write additional outputs
        overwrite: Whether to overwrite existing files
    """
    if scenario is None:
        scenario_list = list(impf_dict["scenarios"])
    else:
        scenario_list = scenario

    if not isinstance(scenario_list, list):
        scenario_list = [scenario_list]
    
    # Load observations
    observations = load_observations(
        exposure_type=impf_dict["exposure_type"],
        impact_type=None,
        load_exceedance=True,
        load_supplementary_sources=True
    )
    impact_type = impf_dict["impact_type"]
    impact_type_list = impf_dict.impact_type_list(observations=observations)
    figsize = (base_figsize[0], base_figsize[1] * len(impact_type_list))

    # Calculate exceedance curves
    curves = get_impf_exceedance_curves(impf_dict, scenario_list, impact_type_list, overwrite=overwrite)

    # Plot exceedance curves
    if make_plots:
        plot_impact_type_exceedance(
            impf_dict,
            impact_type,
            curves,
            observations,
            scenario_list,
            figsize
        )

    # Calculate scores against observations
    _, scores = get_scores(curves, observations, impf_dict, save_scores=write_extras, overwrite=overwrite)
    return curves, scores


def plot_impact_type_exceedance(impf_dict: dict, impact_type: str, curves_uncertain: pd.DataFrame, observations: pd.DataFrame, scenario_list: list, figsize: tuple):
    exceedance_plot_path = impf_dict.exceedance_plot_path(impact_type, create=True)
    exceedance_plot_path_zoom = impf_dict.exceedance_plot_path(impact_type, zoom="zoom", create=True)
    exceedance_plot_path_zoom_obs = impf_dict.exceedance_plot_path(impact_type, zoom="zoom_obs", create=True)
    exceedance_plot_path_zoom_obs_fraction = impf_dict.exceedance_plot_path(impact_type, zoom="zoom_obs_fraction", create=True)
        
    if not os.path.exists(exceedance_plot_path) or overwrite:
        one_exceedance_plot(
            impf_dict, 
            curves_all=curves_uncertain,
            plot_var="impact",
            observations_all=None,
            scaling="absolute",
            exceedance_plot_path=exceedance_plot_path,
            rp_max = None,
            figsize=figsize,
            scenario_list=scenario_list
        )
    if not os.path.exists(exceedance_plot_path_zoom) or overwrite:
        one_exceedance_plot(
            impf_dict,
            curves_all=curves_uncertain,
            plot_var="impact",
            observations_all=None,
            scaling="absolute",
            exceedance_plot_path=exceedance_plot_path_zoom,
            rp_max = 100, 
            figsize=figsize,
            scenario_list=scenario_list
        )
    if not os.path.exists(exceedance_plot_path_zoom_obs) or overwrite:
        one_exceedance_plot(
            impf_dict,
            curves_all=curves_uncertain,
            plot_var="impact",
            observations_all=observations,
            scaling="absolute",
            exceedance_plot_path=exceedance_plot_path_zoom_obs,
            rp_max = 100, 
            figsize=figsize,
            scenario_list=["present"]
        )
    if not os.path.exists(exceedance_plot_path_zoom_obs_fraction) or overwrite:
        one_exceedance_plot(
            impf_dict,
            curves_all=curves_uncertain,
            plot_var="impact_fraction",
            observations_all=observations,
            scaling="fraction",
            exceedance_plot_path=exceedance_plot_path_zoom_obs_fraction,
            rp_max = 100, 
            figsize=figsize,
            scenario_list=["present"]
        )


def one_exceedance_plot(
        impf_dict,
        curves_all,
        plot_var,
        observations_all,
        scaling,
        exceedance_plot_path,
        rp_max,
        figsize,
        scenario_list=None
    ):
    
    impact_subtypes = impf_dict.impact_type_list(observations=observations_all)

    if observations_all is not None and observations_all.shape[0] > 0:
        impact_subtypes = sorted(set(impact_subtypes) | set(observations_all['impact_type'].unique()))

    _, axes = plt.subplots(len(impact_subtypes), 1, figsize=figsize)
    if len(impact_subtypes) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    plt.suptitle(f"Exceedance curves for {impf_dict['exposure_type']}: {impf_dict['exposure_source']}")

    for axis, impact_subtype in zip(axes, impact_subtypes):
        curves = curves_all[curves_all['impact_type'] == impact_subtype]
        unit = curves['unit'].iloc[0]
        assert curves['unit'].nunique() == 1, "Multiple units found in curves data"

        axis.set_xlabel("Return period (years)")
        imp_unit = unit if scaling == "absolute" else "fraction"
        axis.set_title(f"{impact_subtype} ({imp_unit})")
        axis.set_ylabel(f"{imp_unit}")
        if rp_max is None:
            rp_max = max(curves['return_period'])
        axis.set_xlim(1, rp_max)
        
        # Plot exceedance curves
        if scenario_list is None:
            scenario_list = impf_dict["scenarios"]

        for scenario in scenario_list:
            scenario_df = curves[curves['scenario'] == scenario]
            for haz_filepath in set(scenario_df['hazard_filepath']):
                sub_df = scenario_df[scenario_df['hazard_filepath'] == haz_filepath]
                axis.plot(sub_df['return_period'], sub_df[plot_var], color=COLOURS[scenario]['normal'], alpha=0.3)

            # Group by exceedance_frequency (= 1/return_period) to get per-RP statistics
            scenario_grouped = scenario_df.groupby('exceedance_frequency').agg(
                **{plot_var: (plot_var, 'mean'), 'event_frequency': ('event_frequency', 'sum')}
            )
            # Normalize event frequency by number of hazard files to avoid double-counting
            n_haz_files = scenario_df['hazard_filepath'].nunique()
            if n_haz_files > 1:
                scenario_grouped['event_frequency'] /= n_haz_files
            scenario_mean = scenario_grouped[plot_var]
            scenario_aal = calc_aal(impact=scenario_grouped[plot_var].values, event_frequency=scenario_grouped['event_frequency'].values)

            n_lines = scenario_df[['hazard_filepath', 'rp_level']].nunique().values[0]

            if n_lines <= 5:
                axis.fill_between(scenario_mean.index**-1, range_min, range_max, color=COLOURS[scenario]['normal'], alpha=0.2, label='Uncertainty range')
                range_min = scenario_df.groupby(['exceedance_frequency']).agg({plot_var: 'min'}).reset_index()[plot_var].values
                range_max = scenario_df.groupby(['exceedance_frequency']).agg({plot_var: 'max'}).reset_index()[plot_var].values
            else:
                axis.fill_between(scenario_mean.index**-1, quantile_0_2, quantile_0_8, color=COLOURS[scenario]['normal'], alpha=0.2, label='80% uncertainty range')

            axis.plot(scenario_mean.index**-1, scenario_mean.values, color=COLOURS[scenario]['strong'], linewidth=2, label=f'Exceedance: {scenario}')
                quantile_0_2 = scenario_df.groupby(['exceedance_frequency'])[plot_var].quantile(0.2).values
                quantile_0_8 = scenario_df.groupby(['exceedance_frequency'])[plot_var].quantile(0.8).values
            axis.hlines(scenario_aal, xmin=1, xmax=rp_max, color=COLOURS[scenario]['normal'], linestyle='--', linewidth=1)
            axis.plot(1, scenario_aal, marker='s', color=COLOURS[scenario]['normal'], markersize=4, label=f"AAL modelled: {float('%.3g' % scenario_aal)}")

        # plot observations    
        observations = None
        if observations_all is not None and observations_all.shape[0] > 0:
            observations = observations_all[observations_all['impact_type'] == impact_subtype]

        if observations is not None and observations.shape[0] > 0:
            ix_aal = observations['impact_statistic'] == 'aal'
            ix_event = observations['impact_statistic'] == 'event'
            ix_rp = observations['impact_statistic'] == 'rp'

            if plot_var == "impact":
                obs_var = 'value'
            if plot_var == "impact_fraction":
                obs_var = 'value_fraction'

            original_exposure_types = observations['original_exposure_type'].unique()
            n_colours = len(original_exposure_types)

            # Plot individual event observations
            observations_event = observations[ix_event]
            for i, obs_exposure_type in enumerate(original_exposure_types):
                observations_event_subset = observations_event[observations_event['exposure_type'] == obs_exposure_type]
                for _, row in observations_event_subset.iterrows():
                    rp_lower = row['rp_lower']
                    rp_mid = row['rp_mid']
                    rp_upper = row['rp_upper']
                    value = row[obs_var]
                    axis.plot([rp_lower, rp_upper], [value, value], color=COLOURS['observations']['strong'][i], linestyle='--', linewidth=1)
                    axis.plot(rp_mid, value, marker='o', color=COLOURS['observations']['strong'][i], markersize=4, label=f"Observation: {obs_exposure_type}")
            
            # Plot AALs we want to compare to
            observations_aal = observations[ix_aal]
            for i, obs_exposure_type in enumerate(original_exposure_types):
                observations_aal_subset = observations_aal[observations_aal['exposure_type'] == obs_exposure_type]
                for _, row in observations_aal_subset.iterrows():
                    value = row[obs_var]
                    axis.hlines(value, xmin=1, xmax=rp_max, color=COLOURS['observations']['normal'], linestyle='--', linewidth=1)
                    axis.plot(1, value, marker='s', color=COLOURS['observations']['normal'], markersize=4, label=f"AAL observation: {float('%.3g' % value)}")
        
            # Plot RP curves as observations
            observations_rp = observations[ix_rp]
            for i, obs_exposure_type in enumerate(observations_rp['exposure_type'].unique()):
                observations_rp_subset = observations_rp[observations_rp['exposure_type'] == obs_exposure_type]
                axis.plot(observations_rp_subset['rp_mid'], observations_rp_subset[obs_var], color=COLOURS['observations']['strong'][i], linestyle='--', linewidth=1, label=f'RP curve prior')

        axis.legend(loc="lower right")

    plt.savefig(exceedance_plot_path)
    plt.close(axis.figure)
    
    # if observations_path is not None:
    #     pd.concat(observations_collection).to_csv(observations_path)


def get_impf_exceedance_curves(impf_dict: dict, scenario_list: list, impact_type_list: list, overwrite: bool = True) -> pd.DataFrame:
    curves_path = impf_dict.exceedance_type_csv_path(create=True)
    if os.path.exists(curves_path) and not overwrite:
        curves = pd.read_csv(curves_path)
        curves = curves[curves['scenario'].isin(scenario_list) & curves['impact_type'].isin(impact_type_list)]
        return curves

    scenario_list_all = list(impf_dict["scenarios"])
    if scenario_list is None:
        scenario_list = scenario_list_all

    observations_all = load_observations(
        exposure_type=impf_dict["exposure_type"],
        impact_type=None,
        load_exceedance=True,
        load_supplementary_sources=True
    )
    impact_type_list_all = impf_dict.impact_type_list(observations=observations_all)
    if impact_type_list is None:
        impact_type_list = impact_type_list_all

    total_exposed_value = get_total_exposed_value(impf_dict["exposure_type"], usd=False)
    total_exposed_usd = get_total_exposed_value(impf_dict["exposure_type"], usd=True)
    curves = []
    
    # Collect what data we can, based on what's been run earlier in the analysis
    # A little uncertain: depending on the setup not all scenarios and impact types may have been calculated so they won't be returned.
    for scenario in scenario_list_all:
        haz_node = impf_dict["hazard_node"][scenario]
        # print(f"... reading scenario {scenario}")
        haz_filepath_list = haz_node['files']
        if not isinstance(haz_filepath_list, list):
            haz_filepath_list = [haz_filepath_list]
        haz_filepath_list = [fp for fp in haz_filepath_list if not '_ALL_' in fp]  # don't analyse combined impacts from all events
        
        for i, haz_filepath in enumerate(haz_filepath_list):
            for impact_subtype in impact_type_list_all:
                this_is_economic = (impact_subtype == 'economic_loss')  # clumsy: find a smarter way to combine different plot types
                total_exposed = total_exposed_usd if this_is_economic else total_exposed_value

                rp_data = None
                if impact_subtype in VALID_THRESHOLD_IMPACT_TYPES:
                    for rp_level in RP_LEVELS:
                        impact_path = impf_dict.impact_rp_level_file_path(Path(haz_filepath).stem, impact_subtype, rp_level, create=True)
                        if os.path.exists(impact_path):
                            imp = Impact.from_hdf5(impact_path)
                            rp_data = make_exceedance_curve(scenario, impf_dict, imp, impact_subtype, haz_filepath, total_exposed)
                            rp_data['rp_level'] = rp_level
                            curves.append(rp_data)
                if rp_data is None:  # Either not a threshold-based impact type, or no threshold-based impacts found (e.g. no observations to train, or user didn't request_)
                    impact_path = impf_dict.impact_file_path(Path(haz_filepath).stem, impact_subtype, create=True)
                    if not os.path.exists(impact_path):
                        if impact_subtype in impact_type_list and scenario in scenario_list:
                            raise FileNotFoundError(f'Requested impact data is missing: {impact_path}. ')
                        continue
                    imp = Impact.from_hdf5(impact_path)
                    rp_data = make_exceedance_curve(scenario, impf_dict, imp, impact_subtype, haz_filepath, total_exposed)
                    rp_data['rp_level'] = np.nan
                    curves.append(rp_data)

    curves = pd.concat(curves)
    curves.to_csv(curves_path, index=False)

    curves = curves[curves['scenario'].isin(scenario_list) & curves['impact_type'].isin(impact_type_list)]
    return curves


def make_exceedance_curve(scenario, impf_dict, imp, impact_type, haz_filepath, total_exposed):
    event_freqs = np.array(imp.frequency)
    assert np.all(np.diff(event_freqs) <= 0)  # ascending RP, descending frequency

    # Compute exceedance frequencies from event frequencies.
    # Events at the same return period share the same event_frequency.
    # Exceedance frequency = cumulative sum of event frequencies from rarest to most frequent.
    unique_event_freqs = np.unique(event_freqs)  # sorted ascending by numpy

    # annoyingly, some event frequencies are duplicated, but not all, so we have to identify these and correct
    count_event_freqs = np.array([len(event_freqs[event_freqs == f]) for f in unique_event_freqs])
    rep_events = count_event_freqs.min()
    n_freqs = len(event_freqs) / rep_events
    assert int(n_freqs) == n_freqs
    n_freqs = int(n_freqs)

    assert np.all([pd.Series(event_freqs[rep_events * i: rep_events * i + 1]).nunique() == 1 for i in range(0, n_freqs)]), "Event frequencies are not unique within each (assumed) return period"
    unique_event_freqs = [event_freqs[rep_events * i] for i in range(0, n_freqs)]
    total_per_rp = np.array([event_freqs[rep_events * i: rep_events * i + 1].sum() for i in range(0, n_freqs)])   # not really necessary i guess
    exceedance_freqs = np.cumsum(total_per_rp[::-1])[::-1]
    return_periods = 1.0 / exceedance_freqs
    assert set(return_periods) == set([2, 5, 10, 25, 50, 100, 250, 500, 1000]), f"Unexpected aqueduct return periods: {return_periods}"

    # Validate: event frequencies should sum to the max exceedance frequency (= 1/min_RP)
    assert np.isclose(event_freqs.sum(), exceedance_freqs.max(), rtol=1e-4), \
        f"Event frequencies ({event_freqs.sum()}) should sum to max exceedance frequency ({exceedance_freqs.max()})"

    rp_data = [
        {
            "scenario": scenario,
            "hazard_type": impf_dict["hazard_type"],
            "hazard_source": impf_dict["hazard_source"],
            "hazard_filepath": haz_filepath,
            "exposure_type": impf_dict["exposure_type"],
            "exposure_source": impf_dict["exposure_source"],
            "impact_type": impact_type,
            "total_exposed_value": total_exposed,
            "unit": imp.unit,
            "event_frequency": ef,
            "exceedance_frequency": xf,
            "return_period": rp,
            "impact": imp_val,
            "impact_fraction": imp_val / total_exposed
        } for ef, xf, rp, imp_val in zip(
            event_freqs,
            exceedance_freqs,
            return_periods,
            imp.at_event
        )
    ]
    return pd.DataFrame(rp_data)


def get_scores(curves, observations, impf_dict, save_scores: bool, overwrite: bool):
    if observations.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame()
    scores_path = impf_dict.scores_csv_path(create=True)
    scored_observations_path = impf_dict.scored_observations_csv_path(create=True)
    if os.path.exists(scores_path) and not overwrite:
        return pd.read_csv(scored_observations_path), pd.read_csv(scores_path)

    engine = ScoringEngine(cost_function=squared_error)
    scored_observations, scores = engine.compare_to_observations(curves, observations, impf_dict)
    if save_scores:
        scores.to_csv(scores_path, index=False)
        scored_observations.to_csv(scored_observations_path, index=False)
    return scored_observations, scores


def plot_sectoral_exceedances(all_rp_data, sector_comparison_plot_path, figsize, rp_max=None):
    plot_rp_data = all_rp_data[
        (all_rp_data["scenario"] == "present") &
        (all_rp_data["hazard_source"] == "aqueduct") &
        (all_rp_data["impact_type"] == "economic_loss")
    ]
    exp_type_list = plot_rp_data["exposure_type"].unique()
    exp_type_list = set(exp_type_list) - set(['economic_assets'])

    _, axis = plt.subplots(1, 1, figsize=figsize)
    axis.set_xlabel("Return period (years)")
    imp_unit = "%"
    axis.set_ylabel(f"{imp_unit}")

    plt.suptitle(f"The Gambia flood exceedance curves by economic sector")

    for i, exp_type in enumerate(exp_type_list):
        curves = plot_rp_data[plot_rp_data['exposure_type'] == exp_type]
        if rp_max is None:
            rp_max = max(curves['return_period'])
        axis.set_xlim(1, rp_max)
        scenario = "present"

        scenario_grouped = curves.groupby('exceedance_frequency').agg(
            impact_fraction=('impact_fraction', 'mean'),
            event_frequency=('event_frequency', 'sum')
        )
        n_haz_files = curves['hazard_filepath'].nunique()
        if n_haz_files > 1:
            scenario_grouped['event_frequency'] /= n_haz_files
        scenario_mean = scenario_grouped['impact_fraction']
        scenario_aal = calc_aal(impact=scenario_grouped['impact_fraction'].values, event_frequency=scenario_grouped['event_frequency'].values)
        axis.plot(scenario_mean.index**-1, scenario_mean.values, color=COLOURS2[i], linewidth=1.5, label=exp_type.title())
        # axis.hlines(scenario_aal, xmin=1, xmax=rp_max, color=COLOURS[scenario]['normal'], linestyle='--', linewidth=1)
        # axis.plot(1, scenario_aal, marker='s', color=COLOURS[scenario]['normal'], markersize=4, label=f"AAL modelled: {float('%.3g' % scenario_aal)}")

        axis.legend(loc="lower right")

    plt.savefig(sector_comparison_plot_path)
    plt.close(axis.figure)


def main(analysis_name, overwrite=False):
    conf = CONFIG
    data_dir = Path(conf.get("data_dir"))
    output_base_dir = Path(conf.get("output_dir"))
    if not os.path.exists(output_base_dir):
        raise FileNotFoundError(f'Please create an output directory at {output_base_dir}')

    print("======================================================")
    print(f"Working on {analysis_name} data")
    
    impf_list = utils_config.gather_impact_calculation_metadata(analysis_name=analysis_name)

    # Gather all impact calculations:
    all_curves = []
    plot_dir = None

    for impf_dict in impf_list:
        print("-----------------------------------------------------")
        print(f"Visualising impacts for {impf_dict['exposure_type']}: {impf_dict['exposure_source']} - {impf_dict['hazard_type']}: {impf_dict['hazard_source']}")

        if not impf_dict["exposure_node"]:
            print(' MISSING: No exposure configuration found as specified in impact functions. Skipping')
            continue

        if not impf_dict["hazard_node"]:
            print(' MISSING: No hazard configuration found as specified in impact functions. Skipping')
            continue

        csv_path = impf_dict.exceedance_all_csv_path(create=True)  # same for all impact types actually
        if plot_dir is None:
            plot_dir = impf_dict.exceedance_plot_dir(create=True)

        try:
            impf_curves, _ = analyse_impf_exceedance(
                impf_dict,
                scenario=None,
                make_plots=True,
                write_extras=True,
                overwrite=overwrite
            )
            all_curves.append(impf_curves)
        except Exception as e:
            print(f" ERROR: Failed to visualise exceedance curves impacts for {impf_dict['hazard_type']}: {impf_dict['hazard_source']} – {impf_dict['exposure_type']}: {impf_dict['exposure_source']}")
            print(f'{e}')
            raise e
            continue
    
    
    if len(all_curves) > 0:
        print("Writing output CSV")
        all_curves = pd.concat(all_curves, ignore_index=True)
        all_curves.to_csv(csv_path, index=False)

        print("Generating sectoral comparison plots")
        sectoral_plot_path = Path(plot_dir, f"exceedance_sectoral_percentage.png")
        sectoral_plot_path_zoom = Path(plot_dir, f"exceedance_sectoral_percentage_zoom.png")
        plot_sectoral_exceedances(all_curves, sectoral_plot_path, (18, 8), rp_max=None)
        plot_sectoral_exceedances(all_curves, sectoral_plot_path_zoom, (18, 8), rp_max=100)
    else:
        print(f"No data found for analysis {analysis_name}")


if __name__ == "__main__":
    main(analysis_name=analysis_name, overwrite=overwrite)