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

from check_inputs import check_node, check_enabled_node
import utils_config
import utils_observations
from config import CONFIG

HAZARD_MAP = {
    "aqueduct": "FL",
    "jrc": "FL"
}

figsize = (16, 9)

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
        'strong': 'blue',
        'normal': 'lightblue'
    }
}

def analyse_exceedance(impf_dict, data_dir, plot_dir, observations=None, overwrite=True):
    calibrated_string = "calibrated" if impf_dict["calibrated"] else "uncalibrated"
    exposure_type = impf_dict["exposure_type"]
    exposure_source = impf_dict["exposure_source"]
    hazard_source = impf_dict["hazard_source"]
    impact_type = impf_dict["impact_type"]
    impact_dir = impf_dict["impact_dir"]

    if not os.path.exists(plot_dir.parent.parent):
        raise FileNotFoundError(f'Something is wrong with the directory structure. Not found: {plot_dir.parent.parent}')
    os.makedirs(plot_dir, exist_ok=True)

    exposure_files = impf_dict['exposure_node']['files']
    exposure_files = exposure_files if isinstance(exposure_files, list) else [exposure_files]
    exposure_files = [Path(impf_dict['exposure_dir'], fn) for fn in exposure_files]
    exp = Exposures.concat([Exposures.from_hdf5(epath) for epath in exposure_files])
    total_exposed_value = exp.gdf['value'].sum()

    curves = []
    for scenario, haz_node in impf_dict["hazard_node"].items():
        # print(f"... reading scenario {scenario}")
        haz_filepath_list = haz_node['files']
        if not isinstance(haz_filepath_list, list):
            haz_filepath_list = [haz_filepath_list]
        haz_filepath_list = [fp for fp in haz_filepath_list if not '_ALL_' in fp]  # don't analyse combined impacts from all events
        
        for i, haz_filepath in enumerate(haz_filepath_list):
            impact_path = Path(impact_dir, f'impact_{impact_type}_{exposure_type}_{exposure_source}_{hazard_source}_{Path(haz_filepath).stem}.hdf5')
            if not os.path.exists(impact_path):
                raise FileNotFoundError(f'Impact data is missing: {impact_path}')

            imp = Impact.from_hdf5(impact_path)
            rp_data = [
                {
                    "scenario": scenario,
                    "hazard_type": impf_dict["hazard_type"],
                    "hazard_source": hazard_source,
                    "hazard_filepath": Path(haz_filepath).stem,
                    "exposure_type": exposure_type,
                    "exposure_source": exposure_source,
                    "impact_type": impact_type,
                    "total_exposed_value": total_exposed_value,
                    "unit": imp.unit,
                    "frequency": freq,
                    "impact": i,
                    "impact_fraction": i / total_exposed_value
                } for freq, i in zip(
                    imp.frequency,
                    imp.at_event
                )
            ]
            curves = curves + rp_data
    
    curves = pd.DataFrame(curves)

    exceedance_plot_path = Path(plot_dir, f"exceedance_{impact_type}_{impf_dict['hazard_source']}_{exposure_source}_{exposure_type}.png")
    exceedance_plot_path_zoom = Path(plot_dir, f"exceedance_{impact_type}_{impf_dict['hazard_source']}_{exposure_source}_{exposure_type}_zoom.png")

    if os.path.exists(exceedance_plot_path) and not overwrite:
        print('... plot already exists, just extracting exceedance values')
    else:
        plot_exceedance_curves(impf_dict, curves=curves, impact=imp, total_exposed_value=total_exposed_value, exceedance_plot_path=exceedance_plot_path, rp_max = None, plot_observations=False)
        plot_exceedance_curves(impf_dict, curves=curves, impact=imp, total_exposed_value=total_exposed_value, exceedance_plot_path=exceedance_plot_path_zoom, rp_max = 100, plot_observations=True)
    
    return curves


def plot_exceedance_curves(impf_dict, curves, impact, total_exposed_value, exceedance_plot_path, rp_max, plot_observations):
    _, axis = plt.subplots(1, 1, figsize=figsize)
    plt.suptitle(f"Exceedance curve for {impf_dict['exposure_type']} {impf_dict['impact_type']}: {impf_dict['exposure_source']}")
    axis.set_ylabel(f"{impf_dict['impact_type']} ({impact.unit})")
    rp = [1/f for f in impact.frequency]
    axis.set_xlabel("Return period (year)")
    if rp_max is None:
        rp_max = max(rp)
    axis.set_xlim(1, rp_max)
    
    # Plot exceedance curves
    for scenario in impf_dict["hazard_node"].keys():
        scenario_df = curves[curves['scenario'] == scenario]
        for haz_filepath in set(scenario_df['hazard_filepath']):
            sub_df = scenario_df[scenario_df['hazard_filepath'] == haz_filepath]
            axis.plot(rp, sub_df['impact'], color=COLOURS[scenario]['normal'])
        
        scenario_mean = scenario_df.groupby('frequency')['impact'].agg('mean')
        scenario_aal = (scenario_mean.index.values * scenario_mean.values).sum()
        axis.plot(scenario_mean.index**-1, scenario_mean.values, color=COLOURS[scenario]['strong'], linewidth=2)
        axis.hlines(scenario_aal, xmin=1, xmax=rp_max, color=COLOURS[scenario]['normal'], linestyle='--', linewidth=1)
        axis.plot(1, scenario_aal, marker='s', color=COLOURS[scenario]['normal'], markersize=4, label=f'AAL modelled: {scenario_aal}')

    # plot observations
    if plot_observations:
        observations = utils_observations.load_observations(
            exposure_type=impf_dict['exposure_type'],
            impact_type=impf_dict['impact_type'],
            get_dependents=True,
            get_uncalibrated=impf_dict["calibrated"]
            )
        
        if observations.shape[0] == 0:
            print('... no observations found for this exposure and impact type')
        else:
            observations = observations[observations['value'].notna()]
            observations = utils_observations.calculate_observation_fractions(
                observations,
                total_exposed_value=total_exposed_value
            )
            ix_aal = observations['impact_statistic'] == 'aal'
            ix_event = observations['impact_statistic'] == 'event'

            observations_event = observations[ix_event]
            for _, row in observations_event.iterrows():
                rp_lower = row['rp_lower']
                rp_mid = row['rp_mid']
                rp_upper = row['rp_upper']
                value = row['value']
                axis.plot([rp_lower, rp_upper], [value, value], color=COLOURS['observations']['strong'], linestyle='--', linewidth=1)
                axis.plot(rp_mid, value, marker='o', color=COLOURS['observations']['strong'], markersize=4)
            
            if ix_aal.sum() > 0:
                observations_aal = observations[ix_aal]
                for _, row in observations_aal.iterrows():
                    value = row['value']
                    axis.hlines(value, xmin=1, xmax=rp_max, color=COLOURS['observations']['normal'], linestyle='--', linewidth=1)
                    axis.plot(1, value, marker='s', color=COLOURS['observations']['normal'], markersize=4, label=f'AAL observation: {value}')

    axis.figure.savefig(exceedance_plot_path)
    plt.close(axis.figure)



def main(overwrite=False):
    conf = CONFIG
    data_dir = Path(conf.get("data_dir"))
    output_base_dir = Path(conf.get("output_dir"))
    if not os.path.exists(output_base_dir):
        raise FileNotFoundError(f'Please create an output directory at {output_base_dir}')

    for calibrated_string in ["calibrated", "uncalibrated"]:
        print("======================================================")
        print(f"Working on {calibrated_string} data")
        output_dir = Path(output_base_dir, calibrated_string, 'exceedance')
        plot_dir = Path(output_dir, 'plots')
        csv_dir = Path(output_dir, 'exceedance')
        csv_path = Path(csv_dir, "exceedance.csv")
        os.makedirs(csv_dir, exist_ok=True)


        # Gather all impact calculations:
        impf_list = utils_config.gather_impact_function_metadata(filter={"calibrated_string": calibrated_string})
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

            try:
                this_rp_data = analyse_exceedance(
                    impf_dict,
                    data_dir=data_dir,
                    plot_dir=plot_dir,
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
    main(overwrite=overwrite)