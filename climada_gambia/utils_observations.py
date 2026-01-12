import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from climada_gambia.config import CONFIG
from climada_gambia.utils_total_exposed_value import get_total_exposed_value
from climada_gambia.paths import PathBuilder

observations_path = Path('/Users/chrisfairless/Library/CloudStorage/OneDrive-Personal/Projects/UNU/gambia2025/data/observations_curated.xlsx')

OBSERVATIONS_FULL = pd.read_excel(observations_path, sheet_name='event_observations')
HIERARCHY_FULL = pd.read_excel(observations_path, sheet_name='hierarchy')

valid_impact_types = ['economic_loss', 'displaced', 'affected', 'damaged', 'destroyed']


def load_observations(
        exposure_type: str,
        impact_type: Optional[str] = None,
        load_exceedance: bool = True,
        load_supplementary_sources: bool =True
    ):
    """Load observations for a given exposure type.
    Args:
        exposure_type: Exposure type to load observations for
        impact_type: Impact type to filter observations for
        load_exceedance: Whether to load uncalibrated exceedance curves as additional obervations (when specified in hierarchy).
            Name of analysis to load exceedance data from is taken from CONFIG["uncalibrated_analysis_name"] but specified in the 
            hierarchy file as simply 'uncalibrated'.
        load_supplementary_sources: Whether to load supplementary observation sources (when specified in hierarchy)
    Returns:
        DataFrame of observations
    """

    hazard_type = 'flood'  # for now
    hierarchy = load_hierarchy(exposure_type)
    hierarchy = hierarchy[hierarchy['weight'] != 0]

    exposure_observations_list = []

    if 'observations' in hierarchy['data_source'].values:
        hierarchy_subset = hierarchy[hierarchy['data_source'] == 'observations']
        assert hierarchy_subset.shape[0] == 1, f'Multiple observation entries found in hierarchy for exposure_type={exposure_type}'
        total_weight = hierarchy_subset['weight'].values[0]

        observations_subset = copy.deepcopy(OBSERVATIONS_FULL)
        observations_subset = observations_subset[observations_subset['value'].notna()]
        observations_subset = observations_subset[observations_subset['weight'] != 0]
        observations_subset = observations_subset[observations_subset['exposure_type'] == exposure_type]
        observations_subset = observations_subset[observations_subset['impact_type'].isin(valid_impact_types)]
        columns_observations = [
            "impact_statistic", "obs_impact_type", "impact_unit_type", "exposure_unit",
            "exposure_type", "impact_type", "value",
            "rp_lower",	"rp_mid", "rp_upper", "weight"
        ]
        observations_subset = observations_subset[columns_observations]
        observations_subset = observations_subset.reset_index(drop=True)

        if impact_type is None:
            impact_type_list_obs = observations_subset['impact_type'].dropna().unique()
        else:
            impact_type_list_obs = [impact_type]

        for i_type in impact_type_list_obs:
            impact_subset = observations_subset[observations_subset['impact_type'] == i_type]
            if impact_subset.shape[0] == 0:
                continue
            impact_subset['weight'] = impact_subset['weight'] * total_weight / impact_subset['weight'].sum()
            impact_subset['observation_type'] = 'observations'
            assert len(impact_subset['exposure_unit'].unique()) == 1, f"Unexpected set of exposure units for exposure_type {exposure_type} impact_type {i_type}: found {impact_subset['exposure_unit'].unique()}"
            exposure_unit = impact_subset['exposure_unit'].values[0]
            total_exposed_value = get_total_exposed_value(exposure_type, usd=(exposure_unit=='USD'))
            impact_subset = calculate_observation_fractions(impact_subset, total_exposed_value=total_exposed_value)
            exposure_observations_list.append(impact_subset)

    if load_exceedance and 'uncalibrated' in hierarchy['data_source'].values:
        if impact_type is not None:
            i_type = impact_type
        else:
            i_type = "people" if exposure_type == "population" else "economic_loss"   # TODO make this less fragile for now non-financial exposures
        
        if i_type not in ['affected', 'damaged', 'destroyed']:          # Not implemented for priors for threshold step functions yet
            row = hierarchy[hierarchy['data_source'] == 'uncalibrated']
            assert row.shape[0] == 1, f'Multiple uncalibrated entries found in hierarchy for exposure_type={exposure_type}, impact_type={i_type}'
            uncalibrated_exposure_source = row['uncalibrated_exposure_source'].values[0]
            
            # Build path using MetadataCalibration
            from climada_gambia.paths import MetadataCalibration
            path_builder = MetadataCalibration(
                config=CONFIG,
                analysis_name=CONFIG["uncalibrated_analysis_name"]
            )
            exceedance_dir = path_builder.exceedance_output_dir()
            exceedance_path = Path(exceedance_dir, "exceedance.csv")
            
            if not os.path.exists(exceedance_path):
                print(f"WARNING: Exceedance data not found at {exceedance_path} for analysis: you can ignore this if this is the first time you have run calculate_impacts.")
                return pd.DataFrame()

            exceedance_df = pd.read_csv(exceedance_path)
            uncalibrated_df = exceedance_df[
                (exceedance_df['scenario'] == 'present') &
                (exceedance_df['exposure_type'] == exposure_type) &
                (exceedance_df['exposure_source'] == uncalibrated_exposure_source) &
                # (exceedance_df['hazard_type'] == hazard_type) &
                (exceedance_df['impact_type'] == i_type)
            ]
            if uncalibrated_df.shape[0] == 0:
                raise ValueError(f'No uncalibrated exceedance data found for filter exposure_type={exposure_type}, exposure_source={uncalibrated_exposure_source}, impact_type={i_type}')
            if len(uncalibrated_df['hazard_source'].unique()) > 1:
                raise ValueError(f'Multiple hazard sources found in uncalibrated exceedance data for filter exposure_type={exposure_type}, exposure_source={uncalibrated_exposure_source}. Not ready for this.')

            exposure_unit = uncalibrated_df['unit'].values[0]
            uncalibrated_df = uncalibrated_df[['frequency', 'impact', 'impact_fraction']]
            uncalibrated_df = uncalibrated_df.reset_index(drop=True).groupby('frequency').agg('mean').reset_index()
            uncalibrated_df['rp'] = 1 / uncalibrated_df['frequency']
            total_weight = row['weight'].values[0]
            weight = total_weight / uncalibrated_df.shape[0]
            uncalibrated_obs = pd.DataFrame([
                {
                    "observation_type": "prior",
                    "impact_statistic": "rp",
                    "impact_type": i_type,
                    "impact_unit_type": "fraction",
                    "exposure_unit": exposure_unit,
                    "exposure_type": exposure_type,
                    "value": irow['impact_fraction'],
                    "rp_lower": irow['rp'],
                    "rp_mid": irow['rp'],
                    "rp_upper": irow['rp'],
                    "weight": weight
                } for i, irow in uncalibrated_df.iterrows()
            ])
            total_exposed_value = get_total_exposed_value(exposure_type, usd=(exposure_unit=='USD'))
            uncalibrated_obs = calculate_observation_fractions(uncalibrated_obs, total_exposed_value=total_exposed_value)
            exposure_observations_list.append(uncalibrated_obs)
        
    if get_supplementary_sources:
        supplementary_source_list = list(set(hierarchy['data_source']).difference({'observations', 'uncalibrated'}))
        for supplementary_source in supplementary_source_list:
            assert supplementary_source in HIERARCHY_FULL['exposure_type'].values, f"Data source {supplementary_source} not found in hierarchy exposure types: {HIERARCHY_FULL['exposure_type'].values}"
            new_hierarchy = hierarchy[hierarchy['data_source'] == supplementary_source]
            assert new_hierarchy.shape[0] == 1, f'Multiple {supplementary_source} entries found in hierarchy for exposure_type={exposure_type}, impact_type={impact_type}'
            total_weight = new_hierarchy['weight'].values[0]
            if supplementary_source == 'population':
                exposure_unit = 'people'
            else:
                exposure_unit = 'USD'

            total_exposed_value = get_total_exposed_value(supplementary_source, usd=(exposure_unit=='USD'))
            supplement = load_observations(exposure_type=supplementary_source, impact_type=None, load_exceedance=load_exceedance, load_supplementary_sources=True)
            if supplement.shape[0] == 0:
                continue
            
            supplement['weight'] = supplement['weight'] * total_weight / supplement['weight'].sum()
            supplement = supplement.rename(columns={'value': 'original_value', 'exposure_type': 'original_exposure_type'})
            supplement['exposure_type'] = exposure_type
            supplement['value'] = supplement['value_fraction'] * total_exposed_value
            exposure_observations_list.append(supplement)

    for i, df in enumerate(exposure_observations_list):
        # Align the data.
        # TODO: standardise the columns I want here (e.g. original_value, original_exposure_type) to keep more information 
        exposure_observations_list[i] = df[["observation_type", "impact_statistic", "impact_type", "impact_unit_type", "exposure_unit", "exposure_type", "value", "value_fraction", "rp_lower", "rp_mid", "rp_upper", "weight"]]

    if len(exposure_observations_list) > 0:
        exposure_observations = pd.concat(exposure_observations_list, axis=0, ignore_index=True)
    else:
        exposure_observations = pd.DataFrame()

    return exposure_observations


def calculate_observation_fractions(observations, total_exposed_value):
    if 'value_fraction' not in observations.columns:
        observations['value_fraction'] = np.nan
    ix_fraction = observations['impact_unit_type'] == 'fraction'
    observations.loc[ix_fraction, 'value_fraction'] = observations.loc[ix_fraction, 'value']
    observations.loc[~ix_fraction, 'value_fraction'] = observations.loc[~ix_fraction, 'value'] / total_exposed_value
    observations['value'] = observations['value_fraction'] * total_exposed_value
    observations['impact_unit_type'] = 'absolute'
    return observations


def load_hierarchy(exposure_type=None):
    hierarchy = copy.deepcopy(HIERARCHY_FULL)
    if exposure_type:
        hierarchy = hierarchy[hierarchy['exposure_type'] == exposure_type]
    if hierarchy.shape[0] == 0:
        print(f'No exposure type {exposure_type} found in hierarchy')
    return hierarchy
