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
from climada_gambia.paths import MetadataCalibration
from climada_gambia import utils_config
from climada_gambia import utils_observations
from climada_gambia.utils_total_exposed_value import get_total_exposed_value
from climada_gambia.config import CONFIG


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

# Normalised square error – doesn't like targets that are small
# cost_function = lambda modelled, target: ((modelled - target) / target)**2

# Regular square
cost_function = lambda modelled, target: (modelled - target)**2


def analyse_exceedance(impf_dict, scenario=None, write_extras=True, overwrite=True):
    """Analyse exceedance curves for impacts.
    
    Args:
        impf_dict: MetadataImpact instance containing impact function configuration
        scenario: Scenario name or None for all scenarios
        write_extras: Whether to write additional outputs
        overwrite: Whether to overwrite existing files
    """
    exposure_type = impf_dict["exposure_type"]
    exposure_source = impf_dict["exposure_source"]
    hazard_source = impf_dict["hazard_source"]
    impact_type = impf_dict["impact_type"]
    impact_dir = impf_dict["impact_dir"]
    data_dir = impf_dict["data_dir"]
    plot_dir = impf_dict["plot_dir"]
    impact_type_list = [impf_dict["impact_type"]] + list(impf_dict["thresholds"].keys())
    figsize = (base_figsize[0], base_figsize[1] * len(impact_type_list))

    if write_extras:
        if not os.path.exists(plot_dir.parent.parent):
            raise FileNotFoundError(f'Something is wrong with the directory structure. Not found: {plot_dir.parent.parent}')
        os.makedirs(plot_dir, exist_ok=True)

    # exposure_files = impf_dict['exposure_node']['files']
    # exposure_files = exposure_files if isinstance(exposure_files, list) else [exposure_files]
    # exposure_files = [Path(impf_dict['exposure_dir'], fn) for fn in exposure_files]
    # exp = Exposures.concat([Exposures.from_hdf5(epath) for epath in exposure_files])
    # total_exposed_value = exp.gdf['value'].sum()

    total_exposed_value = get_total_exposed_value(exposure_type, usd=False)
    total_exposed_usd = get_total_exposed_value(exposure_type, usd=True)
    if scenario is None:
        scenario_list = list(impf_dict["hazard_node"].keys())
    else:
        scenario_list = scenario

    if not isinstance(scenario_list, list):
        scenario_list = [scenario_list]
        
    curves = []
    for scenario in scenario_list:
        haz_node = impf_dict["hazard_node"][scenario]
        # print(f"... reading scenario {scenario}")
        haz_filepath_list = haz_node['files']
        if not isinstance(haz_filepath_list, list):
            haz_filepath_list = [haz_filepath_list]
        haz_filepath_list = [fp for fp in haz_filepath_list if not '_ALL_' in fp]  # don't analyse combined impacts from all events
        
        for i, haz_filepath in enumerate(haz_filepath_list):
            for impact_subtype in impact_type_list:
                this_is_economic = (impact_subtype == 'economic_loss')  # clumsy: find a smarter way to combine different plot types
                total_exposed = total_exposed_usd if this_is_economic else total_exposed_value
                impact_path = Path(impact_dir, f'impact_{impact_subtype}_{exposure_type}_{exposure_source}_{hazard_source}_{Path(haz_filepath).stem}.hdf5')
                if not os.path.exists(impact_path):
                    raise FileNotFoundError(f'Impact data is missing: {impact_path}')

                imp = Impact.from_hdf5(impact_path)
                rp_data = get_curves(scenario, impf_dict, imp, impact_subtype, haz_filepath, total_exposed)
                curves.append(rp_data)
    
    curves = pd.concat(curves)

    # print("NOT EVEN LOADING OBSERVATIONS!!")
    observations = utils_observations.load_observations(
        exposure_type=impf_dict["exposure_type"],
        impact_type=None,
        load_exceedance=True,
        load_supplementary_sources=True
    )

    if write_extras:
        plot_dir = impf_dict["plot_dir"]
        
        exceedance_plot_path = impf_dict.exceedance_plot_path(impact_type)
        exceedance_plot_path_zoom = impf_dict.exceedance_plot_path(impact_type, zoom="zoom")
        exceedance_plot_path_zoom_obs = impf_dict.exceedance_plot_path(impact_type, zoom="zoom_obs")
        exceedance_plot_path_zoom_obs_fraction = impf_dict.exceedance_plot_path(impact_type, zoom="zoom_obs_fraction")
            
        if os.path.exists(exceedance_plot_path) and not overwrite:
            print('... plot already exists, just extracting exceedance values')
        else:
            plot_exceedance_curves(
                impf_dict, 
                curves_all=curves,
                plot_var="impact",
                observations_all=None,
                unit=imp.unit,
                exceedance_plot_path=exceedance_plot_path,
                rp_max = None,
                figsize=figsize,
                scenario_list=scenario_list
            )
            plot_exceedance_curves(
                impf_dict,
                curves_all=curves,
                plot_var="impact",
                observations_all=None,
                unit=imp.unit,
                exceedance_plot_path=exceedance_plot_path_zoom,
                rp_max = 100, 
                figsize=figsize,
                scenario_list=scenario_list
            )
            plot_exceedance_curves(
                impf_dict,
                curves_all=curves,
                plot_var="impact",
                observations_all=observations,
                unit=imp.unit,
                exceedance_plot_path=exceedance_plot_path_zoom_obs,
                rp_max = 100, 
                figsize=figsize,
                scenario_list=scenario_list
            )
            plot_exceedance_curves(
                impf_dict,
                curves_all=curves,
                plot_var="impact_fraction",
                observations_all=observations,
                unit=imp.unit,
                exceedance_plot_path=exceedance_plot_path_zoom_obs_fraction,
                rp_max = 100, 
                figsize=figsize,
                scenario_list=scenario_list
            )
    # if True:
        # print("SKIPPING OBSERVATION COMPARISONS WHILE WE DEAL WITH A BUG")
        # return curves, pd.DataFrame()

    if observations.shape[0] == 0:
        return curves, pd.DataFrame()

    observations, scores = compare_obs(impf_dict, curves_all=curves, observations_all=observations)

    if write_extras:
        observations_path = Path(plot_dir, f"observations_{impact_type}_{impf_dict['hazard_source']}_{exposure_source}_{exposure_type}.csv")
        observations.to_csv(observations_path, index=False)
        
        scores_path = Path(plot_dir, f"scores_{impact_type}_{impf_dict['hazard_source']}_{exposure_source}_{exposure_type}.csv")
        scores.to_csv(scores_path, index=True)

    return curves, scores


def get_curves(scenario, impf_dict, imp, impact_type, haz_filepath, total_exposed):
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
            "frequency": freq,
            "impact": i,
            "impact_fraction": i / total_exposed
        } for freq, i in zip(
            imp.frequency,
            imp.at_event
        )
    ]
    return pd.DataFrame(rp_data)


def plot_exceedance_curves(impf_dict, curves_all, plot_var, observations_all, unit, exceedance_plot_path, rp_max, figsize, scenario_list=None):
    impact_subtypes = [impf_dict["impact_type"]] + list(impf_dict["thresholds"].keys())
    _, axes = plt.subplots(len(impact_subtypes), 1, figsize=figsize)
    if len(impact_subtypes) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    plt.suptitle(f"Exceedance curves for {impf_dict['exposure_type']}: {impf_dict['exposure_source']}")

    for axis, impact_subtype in zip(axes, impact_subtypes):
        curves = curves_all[curves_all['impact_type'] == impact_subtype]
        curves['rp'] = 1/curves['frequency']

        axis.set_xlabel("Return period (year)")
        imp_unit = "USD" if impact_subtype == "economic_loss" else unit   # clumsy, sorry
        axis.set_title(f"{impact_subtype} ({imp_unit})")
        axis.set_ylabel(f"{imp_unit}")
        if rp_max is None:
            rp_max = max(curves['rp'])
        axis.set_xlim(1, rp_max)
        
        # Plot exceedance curves
        if scenario_list is None:
            scenario_list = impf_dict["hazard_node"].keys()

        for scenario in scenario_list:
            scenario_df = curves[curves['scenario'] == scenario]
            for haz_filepath in set(scenario_df['hazard_filepath']):
                sub_df = scenario_df[scenario_df['hazard_filepath'] == haz_filepath]
                axis.plot(sub_df['rp'], sub_df[plot_var], color=COLOURS[scenario]['normal'])
            
            scenario_mean = scenario_df.groupby('frequency')[plot_var].agg('mean')
            scenario_aal = (scenario_mean.index.values * scenario_mean.values).sum()
            axis.plot(scenario_mean.index**-1, scenario_mean.values, color=COLOURS[scenario]['strong'], linewidth=2, label=f'Exceedance: {scenario}')
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

            n_colours = 0
            observations_event = observations[ix_event]

            if plot_var == "impact":
                obs_var = 'value'
            if plot_var == "impact_fraction":
                obs_var = 'value_fraction'

            for i, obs_exposure_type in enumerate(observations_event['exposure_type'].unique()):
                observations_event_subset = observations_event[observations_event['exposure_type'] == obs_exposure_type]
                n_colours = n_colours + 1
                for _, row in observations_event_subset.iterrows():
                    rp_lower = row['rp_lower']
                    rp_mid = row['rp_mid']
                    rp_upper = row['rp_upper']
                    value = row[obs_var]
                    axis.plot([rp_lower, rp_upper], [value, value], color=COLOURS['observations']['strong'][n_colours], linestyle='--', linewidth=1)
                    axis.plot(rp_mid, value, marker='o', color=COLOURS['observations']['strong'][n_colours], markersize=4, label=f"Observation: {obs_exposure_type}")
            
            observations_aal = observations[ix_aal]
            for _, row in observations_aal.iterrows():
                value = row[obs_var]
                axis.hlines(value, xmin=1, xmax=rp_max, color=COLOURS['observations']['normal'], linestyle='--', linewidth=1)
                axis.plot(1, value, marker='s', color=COLOURS['observations']['normal'], markersize=4, label=f"AAL observation: {float('%.3g' % value)}")
        
            observations_rp = observations[ix_rp]
            for i, obs_exposure_type in enumerate(observations_rp['exposure_type'].unique()):
                n_colours = n_colours + 1
                observations_rp_subset = observations_rp[observations_rp['exposure_type'] == obs_exposure_type]
                axis.plot(observations_rp['rp_mid'], observations_rp[obs_var], color=COLOURS['observations']['strong'][n_colours], linestyle='--', linewidth=1, label=f'RP curve prior')

        axis.legend(loc="lower right")

    plt.savefig(exceedance_plot_path)
    plt.close(axis.figure)
    
    # if observations_path is not None:
    #     pd.concat(observations_collection).to_csv(observations_path)



def compare_obs(impf_dict, curves_all, observations_all):
    assert np.all(observations_all['exposure_type'] == impf_dict['exposure_type']), 'Filter for exposure type first'
    if observations_all.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame()

    impact_subtypes = [impf_dict["impact_type"]] + list(impf_dict["thresholds"].keys())
    all_obs = []
    
    for impact_subtype in impact_subtypes:
        observations = observations_all[observations_all['impact_type'] == impact_subtype]
        observations = observations[observations['weight'] > 0]
        if observations.shape[0] == 0:
            continue
        
        curves = curves_all[curves_all['impact_type'] == impact_subtype]
        scenario_df = curves[curves['scenario'] == "present"]
        if scenario_df.shape[0] == 0:
            raise ValueError(f"Couldn't find exceedance curves for {impf_dict['exposure_type']} and {impact_subtype} – is there some unexpected combination of observations that wasn't modelled?")
        
        # We compare on impact fraction rather than impact because we want to be able to compare economic USD values and thresholded affected/damaged/destroyed values
        scenario_mean = scenario_df.groupby('frequency')['impact_fraction'].agg('mean')
        scenario_aal = (scenario_mean.index.values * scenario_mean.values).sum()

        ix_aal = observations['impact_statistic'] == 'aal'
        ix_event = observations['impact_statistic'] == 'event'
        ix_rp = observations['impact_statistic'] == 'rp'
        ix_event = ix_event + ix_rp
        observations_event = observations[ix_event]
        if observations_event.shape[0] > 0:
            observations_event['model_lower'] = np.interp(observations_event['rp_lower'], (scenario_mean.index**-1)[::-1], scenario_mean.values[::-1])
            observations_event['model_mid'] = np.interp(observations_event['rp_mid'], (scenario_mean.index**-1)[::-1], scenario_mean.values[::-1])
            observations_event['model_upper'] = np.interp(observations_event['rp_upper'], (scenario_mean.index**-1)[::-1], scenario_mean.values[::-1])

        observations_aal = observations[ix_aal]
        if observations_aal.shape[0] > 0:
            observations_aal['model_lower'] = scenario_aal
            observations_aal['model_mid'] = scenario_aal
            observations_aal['model_upper'] = scenario_aal

        observations = pd.concat([observations_aal, observations_event])
        observations['cost_lower'] = [cost_function(x1, x2) for x1, x2 in zip(observations['model_lower'], observations['value_fraction'])]
        observations['cost_mid'] = [cost_function(x1, x2) for x1, x2 in zip(observations['model_mid'], observations['value_fraction'])]
        observations['cost_upper'] = [cost_function(x1, x2) for x1, x2 in zip(observations['model_upper'], observations['value_fraction'])]

        all_obs.append(observations)

    all_obs = pd.concat(all_obs)
    all_obs['weight'] = all_obs['weight'] / all_obs['weight'].sum()
    all_obs['weighted_cost_lower'] = all_obs['weight'] * all_obs['cost_lower']
    all_obs['weighted_cost_mid'] = all_obs['weight'] * all_obs['cost_mid']
    all_obs['weighted_cost_upper'] = all_obs['weight'] * all_obs['cost_upper']
    all_obs.to_csv(Path(impf_dict["impact_dir"], 'deleteme_testing_analysis_obs.csv'))
    
    trimmed_obs = all_obs[['impact_type', 'weighted_cost_lower', 'weighted_cost_mid', 'weighted_cost_upper']]
    scores = trimmed_obs.groupby('impact_type')[['weighted_cost_lower', 'weighted_cost_mid', 'weighted_cost_upper']].agg(lambda x: np.sqrt(sum(x)))
    # scores_total = all_obs[['weighted_cost_lower', 'weighted_cost_mid', 'weighted_cost_upper']].agg('sum')
    scores.loc["TOTAL"] = trimmed_obs.set_index('impact_type').agg(lambda x: np.sqrt(sum(x)), axis=0)
    return all_obs, scores


def plot_sectoral_exceedances(all_rp_data, sector_comparison_plot_path, figsize, rp_max=None):
    plot_rp_data = all_rp_data[
        (all_rp_data["scenario"] == "present") &
        (all_rp_data["hazard_source"] == "aqueduct") &
        (all_rp_data["impact_type"] == "economic_loss")
    ]
    exp_type_list = plot_rp_data["exposure_type"].unique()
    exp_type_list = set(exp_type_list) - set(['economic_assets'])

    _, axis = plt.subplots(1, 1, figsize=figsize)
    axis.set_xlabel("Return period (year)")
    imp_unit = "%"
    axis.set_ylabel(f"{imp_unit}")

    plt.suptitle(f"The Gambia flood exceedance curves by economic sector")

    for i, exp_type in enumerate(exp_type_list):
        curves = plot_rp_data[plot_rp_data['exposure_type'] == exp_type]
        curves['rp'] = 1/curves['frequency']
        if rp_max is None:
            rp_max = max(curves['rp'])
        axis.set_xlim(1, rp_max)
        scenario = "present"
        
        scenario_mean = curves.groupby('frequency')['impact_fraction'].agg('mean')
        scenario_aal = (scenario_mean.index.values * scenario_mean.values).sum()
        axis.plot(scenario_mean.index**-1, scenario_mean.values, color=COLOURS2[i], linewidth=1.5, label=exp_type.title())
        # axis.hlines(scenario_aal, xmin=1, xmax=rp_max, color=COLOURS[scenario]['normal'], linestyle='--', linewidth=1)
        # axis.plot(1, scenario_aal, marker='s', color=COLOURS[scenario]['normal'], markersize=4, label=f"AAL modelled: {float('%.3g' % scenario_aal)}")

        axis.legend(loc="lower right")

    plt.savefig(sector_comparison_plot_path)
    plt.close(axis.figure)


def main(overwrite=False):
    conf = CONFIG
    data_dir = Path(conf.get("data_dir"))
    output_base_dir = Path(conf.get("output_dir"))
    if not os.path.exists(output_base_dir):
        raise FileNotFoundError(f'Please create an output directory at {output_base_dir}')

    analysis_name = CONFIG["default_analysis_name"]
    print("======================================================")
    print(f"Working on {analysis_name} data")
    
    impf_list = utils_config.gather_impact_calculation_metadata()

    os.makedirs(csv_dir, exist_ok=True)


    # Gather all impact calculations:
    all_rp_data = []

    for impf_dict in impf_list:
        print("-----------------------------------------------------")
        print(f"Visualising impacts for {impf_dict['exposure_type']}: {impf_dict['exposure_source']} - {impf_dict['hazard_type']}: {impf_dict['hazard_source']}")

        if not impf_dict["exposure_node"]:
            print(' MISSING: No exposure configuration found as specified in impact functions. Skipping')
            continue

        if not impf_dict["hazard_node"]:
            print(' MISSING: No hazard configuration found as specified in impact functions. Skipping')
            continue

        csv_path = impf_dict.exceedance_csv_path  # same for all impact types actually

        try:
            this_rp_data, _ = analyse_exceedance(
                impf_dict,
                data_dir=data_dir,
                plot_dir=plot_dir,
                scenario=None,
                write_extras=True,
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
        all_rp_data.to_csv(csv_path, index=False)

        print("Generating sectoral comparison plots")
        sectoral_plot_path = Path(plot_dir, f"exceedance_sectoral_percentage.png")
        sectoral_plot_path_zoom = Path(plot_dir, f"exceedance_sectoral_percentage_zoom.png")
        plot_sectoral_exceedances(all_rp_data, sectoral_plot_path, (18, 8), rp_max=None)
        plot_sectoral_exceedances(all_rp_data, sectoral_plot_path_zoom, (18, 8), rp_max=100)
    else:
        print(f"No data found for analysis {analysis_name}")


if __name__ == "__main__":
    main(overwrite=overwrite)