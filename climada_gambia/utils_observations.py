import os
import copy
import pandas as pd
from pathlib import Path

from climada_gambia.config import CONFIG

observations_path = Path('/Users/chrisfairless/Library/CloudStorage/OneDrive-Personal/Projects/UNU/gambia2025/data/observations_curated.xlsx')
rp_data_path = Path(CONFIG["output_dir"], "uncalibrated", "exceedance", "exceedance.csv")

OBSERVATIONS_FULL = pd.read_excel(observations_path, sheet_name='event_observations')
HIERARCHY_FULL = pd.read_excel(observations_path, sheet_name='hierarchy')
if os.path.exists(rp_data_path):
    EXCEEDANCE_FULL = pd.read_csv(rp_data_path)
else:
    EXCEEDANCE_FULL = None


def load_observations(
        exposure_type,
        impact_type,
        get_uncalibrated=True,
        get_dependents=True,
        get_supplementary_sources=True
    ):
    hazard_type = 'flood'  # for now
    uncalibrated_exceedance = None
    hierarchy = load_hierarchy(exposure_type, impact_type)
    hierarchy = hierarchy[hierarchy['weight'] != 0]
    if EXCEEDANCE_FULL is None:
        get_uncalibrated = False


    exposure_observations_list = []

    if 'observations' in hierarchy['data_source'].values:
        assert hierarchy[hierarchy['data_source'] == 'observations'].shape[0] == 1, f'Multiple observation entries found in hierarchy for exposure_type={exposure_type}, impact_type={impact_type}'
        weight = hierarchy[hierarchy['data_source'] == 'observations']['weight'].values[0]

        observations_subset = copy.deepcopy(OBSERVATIONS_FULL)
        observations_subset = observations_subset[observations_subset['model_exposure_type'] == exposure_type]
        observations_subset = observations_subset[observations_subset['model_impact_type'] == impact_type]
        columns_observations = [
            "impact_statistic", "impact_type", "impact_unit_type", "exposure_unit",
            "model_exposure_type", "model_impact_type", "value",
            "rp_lower",	"rp_mid", "rp_upper", "weight"
        ]
        observations_subset = observations_subset[columns_observations]
        observations_subset = observations_subset.rename(columns={"model_exposure_type": "exposure_type"})
        observations_subset = observations_subset.reset_index(drop=True)

        if observations_subset.shape[0] > 0:
            observations_subset['weight'] = observations_subset['weight'] * weight / observations_subset['weight'].sum()
            observations_subset['observation_type'] = 'observations'
        exposure_observations_list.append(observations_subset)

    if get_uncalibrated and 'uncalibrated' in hierarchy['data_source'].values:
        assert hierarchy[hierarchy['data_source'] == 'uncalibrated'].shape[0] == 1, f'Multiple uncalibrated entries found in hierarchy for exposure_type={exposure_type}, impact_type={impact_type}'
        weight = hierarchy[hierarchy['data_source'] == 'uncalibrated']['weight'].values[0]
        uncalibrated_exceedance = pd.read_csv(rp_data_path)
        uncalibrated_df = uncalibrated_exceedance[
            (uncalibrated_exceedance['scenario'] == 'present') &
            (uncalibrated_exceedance['exposure_type'] == exposure_type) &
            (uncalibrated_exceedance['exposure_source'] == row['uncalibrated_exposure_source']) &
            # (uncalibrated_exceedance['hazard_type'] == hazard_type) &
            (uncalibrated_exceedance['impact_type'] == impact_type)
        ]
        if uncalibrated_df.shape[0] == 0:
            raise ValueError(f'No uncalibrated exceedance data found for filter exposure_type={exposure_type}, exposure_source={row["uncalibrated_exposure_source"]}')
        if len(uncalibrated_df['hazard_source'].unique()) > 1:
            raise ValueError(f'Multiple hazard sources found in uncalibrated exceedance data for filter exposure_type={exposure_type}, exposure_source={row["uncalibrated_exposure_source"]}. Not ready for this.')

        uncalibrated_df = uncalibrated_df.reset_index(drop=True).groupby('frequency').agg('mean').reset_index()
        uncalibrated_df['rp'] = 1 / uncalibrated_df['frequency']
        weight = row['weight'] / uncalibrated_df.shape[0]
        observations = pd.DataFrame([
            {
                "observation_type": "prior",
                "impact_statistic": "rp",
                "impact_type": impact_type,
                "impact_unit_type": "fraction",
                "exposure_unit": uncalibrated_df['unit'].values[0],
                "exposure_type": exposure_type,
                "value": irow['impact'],
                "rp_lower": irow['rp'],
                "rp_mid": irow['rp'],
                "rp_upper": irow['rp'],
                "weight": weight
            } for i, irow in uncalibrated_df.iterrows()
        ])
        exposure_observations_list.append(observations)
        
    if get_supplementary_sources:
        supplementary_source_list = list(set(hierarchy['data_source']).difference({'observations', 'uncalibrated'}))
        for supplementary_source in supplementary_source_list:
            assert supplementary_source in HIERARCHY_FULL['exposure_type'].values, f"Data source {supplementary_source} not found in hierarchy exposure types: {HIERARCHY_FULL['exposure_type'].values}"
            assert hierarchy[hierarchy['data_source'] == supplementary_source].shape[0] == 1, f'Multiple {supplementary_source} entries found in hierarchy for exposure_type={exposure_type}, impact_type={impact_type}'
            weight = hierarchy[hierarchy['data_source'] == supplementary_source]['weight'].values[0]
            supplement = load_observations(exposure_type=supplementary_source, impact_type=impact_type, get_dependents=True, get_uncalibrated=get_uncalibrated, get_supplementary_sources=True)
            supplement['weight'] = supplement['weight'] * weight / supplement['weight'].sum()
            exposure_observations_list.append(supplement)

    if get_dependents:
        dependent_impact_type_list = list_hierarchical_dependencies(exposure_type, impact_type)
        if impact_type in dependent_impact_type_list:
            raise ValueError(f'Circular dependency found for exposure_type={exposure_type}, impact_type={impact_type}')

        for dependent_impact_type in dependent_impact_type_list:
            exposure_observations_list.append(
                load_observations(exposure_type=exposure_type, impact_type=dependent_impact_type)
            )

    if len(exposure_observations_list) > 0:
        exposure_observations = pd.concat(exposure_observations_list, ignore_index=True)
    else:
        exposure_observations = pd.DataFrame()

    return exposure_observations


def calculate_observation_fractions(observations, total_exposed_value):
    ix_fraction = observations['impact_unit_type'] == 'fraction'
    observations.loc[ix_fraction, 'value_fraction'] = observations.loc[ix_fraction, 'value']
    observations.loc[~ix_fraction, 'value_fraction'] = observations.loc[~ix_fraction, 'value'] / total_exposed_value
    observations.loc[ix_fraction, 'value'] = observations.loc[ix_fraction, 'value'] * total_exposed_value
    return observations


def load_hierarchy(exposure_type=None, impact_type=None):
    hierarchy = copy.deepcopy(HIERARCHY_FULL)
    if exposure_type:
        hierarchy = hierarchy[hierarchy['exposure_type'] == exposure_type]
    if impact_type:
        hierarchy = hierarchy[hierarchy['impact_type'] == impact_type]
    if hierarchy.shape[0] == 0:
        print(f'No exposure type {exposure_type} and impact type {impact_type} found in hierarchy')
    return hierarchy


def list_hierarchical_dependencies(exposure_type, impact_type):
    hierarchy = load_hierarchy(exposure_type, impact_type=None)
    hierarchy = hierarchy[hierarchy['weight'] != 0]
    hierarchy = hierarchy[hierarchy['exposure_type'] == exposure_type]
    hierarchy = hierarchy[hierarchy['parent'] == impact_type]
    return hierarchy['impact_type'].unique().tolist()