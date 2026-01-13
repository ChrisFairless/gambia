import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from climada_gambia.config import CONFIG
from climada_gambia.utils_total_exposed_value import get_total_exposed_value
from climada_gambia.metadata_calibration import MetadataCalibration

observations_path = Path('/Users/chrisfairless/Library/CloudStorage/OneDrive-Personal/Projects/UNU/gambia2025/data/observations_curated.xlsx')

OBSERVATIONS_FULL = pd.read_excel(observations_path, sheet_name='event_observations')
HIERARCHY_FULL = pd.read_excel(observations_path, sheet_name='hierarchy')

valid_impact_types = ['economic_loss', 'displaced', 'affected', 'damaged', 'destroyed']


class ObservationLoader:
    """Loads and processes observation data for calibration."""
    
    def __init__(self, observations_df: pd.DataFrame = None, hierarchy_df: pd.DataFrame = None):
        """
        Args:
            observations_df: Full observations from Excel
            hierarchy_df: Hierarchy defining data sources and weights
        """
        if observations_df is None:
            observations_df = OBSERVATIONS_FULL
        if hierarchy_df is None:
            hierarchy_df = HIERARCHY_FULL
        self.observations = observations_df
        self.hierarchy = hierarchy_df
    
    def load_for_exposure(
        self,
        exposure_type: str,
        impact_type: Optional[str] = None,
        load_exceedance: bool = True,
        load_supplementary_sources: bool = True
    ) -> pd.DataFrame:
        """Main entry point - load all observations for exposure type.
        
        Args:
            exposure_type: Exposure type to load observations for
            impact_type: Impact type to filter observations for (if None, loads all types)
            load_exceedance: Whether to load related uncalibrated exceedance curves as prior observations
                (This is specified as 'uncalibrated' data in the hierarchy and uses the analysis under CONFIG["uncalibrated_analysis_name"] as data)
            load_supplementary_sources: Whether to load related observations from related exposure types (specified in hierarchy)
            
        Returns:
            DataFrame with all observations for the exposure type
        """
        exposure_observations_list = []
        hierarchy = self._get_hierarchy_for_exposure(exposure_type)
        
        # Load direct observations from Excel
        direct_obs = self._load_direct_observations(exposure_type, impact_type)
        exposure_observations_list.append(direct_obs)
        
        # Load uncalibrated priors if requested
        if load_exceedance and 'uncalibrated' in hierarchy['data_source'].values:
            uncalibrated_obs = self._load_uncalibrated_priors(exposure_type, impact_type)
            exposure_observations_list.append(uncalibrated_obs)
        
        # Load supplementary sources if requested
        supplementary_source_list = self._supplementary_sources(exposure_type=exposure_type)

        if load_supplementary_sources and len(supplementary_source_list) > 0:
            supplementary_obs = self._load_supplementary_sources(exposure_type, impact_type, load_exceedance)
            exposure_observations_list.append(supplementary_obs)
        
        # Standardize columns and concatenate all observations
        exposure_observations_list = [df for df in exposure_observations_list if df.shape[0] > 0]
        for i, df in enumerate(exposure_observations_list):
            exposure_observations_list[i] = df[[
                "observation_type", "impact_statistic", "impact_type", "impact_unit_type",
                "exposure_unit", "exposure_type", "value", "value_fraction",
                "rp_lower", "rp_mid", "rp_upper", "weight"
            ]]
        
        if len(exposure_observations_list) > 0:
            return pd.concat(exposure_observations_list, axis=0, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _get_hierarchy_for_exposure(self, exposure_type: str) -> pd.DataFrame:
        """Get hierarchy rows for exposure type, filtering by weight > 0.
        
        Args:
            exposure_type: Exposure type to filter hierarchy for
            
        Returns:
            Filtered hierarchy DataFrame
        """
        hierarchy = self.hierarchy[self.hierarchy['exposure_type'] == exposure_type]
        hierarchy = hierarchy[hierarchy['weight'] != 0]
        return hierarchy
    
    def _calculate_fractions(
        self,
        obs_df: pd.DataFrame,
        total_exposed_value: float
    ) -> pd.DataFrame:
        """Convert observations to fractions of total exposed value.
        
        Args:
            obs_df: Observations DataFrame
            total_exposed_value: Total exposed value for the exposure type
            
        Returns:
            DataFrame with value_fraction column added and values converted to absolute
        """
        if 'value_fraction' not in obs_df.columns:
            obs_df['value_fraction'] = np.nan
        
        ix_fraction = obs_df['impact_unit_type'] == 'fraction'
        obs_df.loc[ix_fraction, 'value_fraction'] = obs_df.loc[ix_fraction, 'value']
        obs_df.loc[~ix_fraction, 'value_fraction'] = obs_df.loc[~ix_fraction, 'value'] / total_exposed_value
        obs_df['value'] = obs_df['value_fraction'] * total_exposed_value
        obs_df['impact_unit_type'] = 'absolute'
        
        return obs_df
    
    def _normalise_weights(
        self,
        obs_df: pd.DataFrame,
        total_weight: float
    ) -> pd.DataFrame:
        """Apply and normalize observation weights.
        
        Args:
            obs_df: Observations DataFrame
            total_weight: Total weight to normalize to
            
        Returns:
            DataFrame with normalized weights
        """
        obs_df['weight'] = obs_df['weight'] * total_weight / obs_df['weight'].sum()
        return obs_df
    
    def _supplementary_sources(self, exposure_type: str) -> list:
        """Get list of supplementary data sources from hierarchy.
        
        Returns:
            List of supplementary data source names
        """
        hierarchy = self._get_hierarchy_for_exposure(exposure_type)
        supplementary_sources = self.hierarchy['data_source'].unique().tolist()
        supplementary_sources = [src for src in supplementary_sources if src not in ['observations', 'uncalibrated']]
        return supplementary_sources

    def _load_direct_observations(
        self,
        exposure_type: str,
        impact_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Load observations from Excel event_observations sheet.
        
        Args:
            exposure_type: Exposure type to load observations for
            impact_type: Impact type to filter observations for (if None, loads all types)
            
        Returns:
            DataFrame with direct observations
        """
        hierarchy = self._get_hierarchy_for_exposure(exposure_type)
        
        if 'observations' not in hierarchy['data_source'].values:
            return pd.DataFrame()
        
        hierarchy_subset = hierarchy[hierarchy['data_source'] == 'observations']
        assert hierarchy_subset.shape[0] == 1, \
            f'Multiple observation entries found in hierarchy for exposure_type={exposure_type}'
        total_weight = hierarchy_subset['weight'].values[0]
        
        # Filter observations
        observations_subset = copy.deepcopy(self.observations)
        observations_subset = observations_subset[observations_subset['value'].notna()]
        observations_subset = observations_subset[observations_subset['weight'] != 0]
        observations_subset = observations_subset[observations_subset['exposure_type'] == exposure_type]
        observations_subset = observations_subset[observations_subset['impact_type'].isin(valid_impact_types)]
        
        columns_observations = [
            "impact_statistic", "obs_impact_type", "impact_unit_type", "exposure_unit",
            "exposure_type", "impact_type", "value",
            "rp_lower", "rp_mid", "rp_upper", "weight"
        ]
        observations_subset = observations_subset[columns_observations]
        observations_subset = observations_subset.reset_index(drop=True)
        
        # Determine impact types to process
        if impact_type is None:
            impact_type_list_obs = observations_subset['impact_type'].dropna().unique()
        else:
            impact_type_list_obs = [impact_type]
        
        # Process each impact type separately
        exposure_observations_list = []
        for i_type in impact_type_list_obs:
            impact_subset = observations_subset[observations_subset['impact_type'] == i_type]
            if impact_subset.shape[0] == 0:
                continue
            
            # Apply weights
            impact_subset = self._normalise_weights(impact_subset, total_weight)
            impact_subset['observation_type'] = 'observations'
            
            # Calculate impacts as fractions
            assert len(impact_subset['exposure_unit'].unique()) == 1, \
                f"Unexpected set of exposure units for exposure_type {exposure_type} impact_type {i_type}: " \
                f"found {impact_subset['exposure_unit'].unique()}"
            exposure_unit = impact_subset['exposure_unit'].values[0]
            total_exposed_value = get_total_exposed_value(exposure_type, usd=(exposure_unit == 'USD'))
            impact_subset = self._calculate_fractions(impact_subset, total_exposed_value=total_exposed_value)
            
            exposure_observations_list.append(impact_subset)
        
        if len(exposure_observations_list) > 0:
            return pd.concat(exposure_observations_list, axis=0, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _load_uncalibrated_priors(
        self,
        exposure_type: str,
        impact_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Load uncalibrated impact curves as prior observations.
        
        Args:
            exposure_type: Exposure type to load priors for
            impact_type: Impact type to filter priors for (if None, uses default)
            
        Returns:
            DataFrame with prior observations from uncalibrated curves
        """

        # Determine impact type
        if impact_type is not None:
            i_type = impact_type
        else:
            i_type = "people" if exposure_type == "population" else "economic_loss"
        
        # Not implemented for threshold step functions yet
        if i_type in ['affected', 'damaged', 'destroyed']:
            return pd.DataFrame()
        
        row = hierarchy[hierarchy['data_source'] == 'uncalibrated']
        assert row.shape[0] == 1, \
            f'Multiple uncalibrated entries found in hierarchy for exposure_type={exposure_type}, impact_type={i_type}'
        uncalibrated_exposure_source = row['uncalibrated_exposure_source'].values[0]
        
        # Build path using MetadataCalibration
        path_builder = MetadataCalibration(
            config=CONFIG,
            analysis_name=CONFIG["uncalibrated_analysis_name"]
        )
        exceedance_dir = path_builder.exceedance_output_dir()
        exceedance_path = Path(exceedance_dir, "exceedance.csv")
        
        if not os.path.exists(exceedance_path):
            print(f"WARNING: Exceedance data not found at {exceedance_path} for analysis: "
                  f"you can ignore this if this is the first time you have run calculate_impacts.")
            return pd.DataFrame()
        
        # Load and filter exceedance data
        exceedance_df = pd.read_csv(exceedance_path)
        uncalibrated_df = exceedance_df[
            (exceedance_df['scenario'] == 'present') &
            (exceedance_df['exposure_type'] == exposure_type) &
            (exceedance_df['exposure_source'] == uncalibrated_exposure_source) &
            (exceedance_df['impact_type'] == i_type)
        ]
        
        if uncalibrated_df.shape[0] == 0:
            raise ValueError(
                f'No uncalibrated exceedance data found for filter exposure_type={exposure_type}, '
                f'exposure_source={uncalibrated_exposure_source}, impact_type={i_type}'
            )
        
        if len(uncalibrated_df['hazard_source'].unique()) > 1:
            raise ValueError(
                f'Multiple hazard sources found in uncalibrated exceedance data for filter '
                f'exposure_type={exposure_type}, exposure_source={uncalibrated_exposure_source}. '
                f'Not ready for this.'
            )
        
        # Average across exceedance curves from different hazard sources if multiple present
        exposure_unit = uncalibrated_df['unit'].values[0]
        uncalibrated_df = uncalibrated_df[['frequency', 'impact', 'impact_fraction']]
        uncalibrated_df = uncalibrated_df.reset_index(drop=True).groupby('frequency').agg('mean').reset_index()
        uncalibrated_df['rp'] = 1 / uncalibrated_df['frequency']
        
        # Calculate weights
        total_weight = row['weight'].values[0]
        weight = total_weight / uncalibrated_df.shape[0]
        
        # Convert to observation format
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
        
        # Calculate fractions
        total_exposed_value = get_total_exposed_value(exposure_type, usd=(exposure_unit == 'USD'))
        uncalibrated_obs = self._calculate_fractions(uncalibrated_obs, total_exposed_value=total_exposed_value)
        
        return uncalibrated_obs
    
    def _load_supplementary_sources(
        self,
        exposure_type: str,
        impact_type: Optional[str] = None,
        load_exceedance: bool = True,
        _recursion_list: list = []
    ) -> pd.DataFrame:
        """Recursively load observations from related exposure types.
        
        Args:
            exposure_type: Exposure type to load supplementary sources for
            impact_type: Impact type to filter observations for
            load_exceedance: Whether to load uncalibrated priors for supplementary sources
            _recursion_depth: Internal parameter to prevent infinite recursion
            
        Returns:
            DataFrame with supplementary observations
        """
        # Prevent infinite recursion
        if exposure_type in _recursion_list:
            raise RecursionError(f"Circular reference detected for exposure type {exposure_type}")
        
        supplementary_source_list = self._supplementary_sources(exposure_type=exposure_type)
        if len(supplementary_source_list) == 0:
            return pd.DataFrame()
        
        supplementary_observations_list = []
        
        for supplementary_source in supplementary_source_list:
            assert supplementary_source in self.hierarchy['exposure_type'].values, \
                f"Data source {supplementary_source} not found in hierarchy exposure types: " \
                f"{self.hierarchy['exposure_type'].values}"
            
            new_hierarchy = hierarchy[hierarchy['data_source'] == supplementary_source]
            assert new_hierarchy.shape[0] == 1, \
                f'Multiple {supplementary_source} entries found in hierarchy for ' \
                f'exposure_type={exposure_type}, impact_type={impact_type}'
            total_weight = new_hierarchy['weight'].values[0]
            
            # Determine exposure unit
            if supplementary_source == 'population':
                exposure_unit = 'people'
            else:
                exposure_unit = 'USD'
            
            # Recursively load observations from supplementary source
            total_exposed_value = get_total_exposed_value(supplementary_source, usd=(exposure_unit == 'USD'))
            supplement = self.load_for_exposure(
                exposure_type=supplementary_source,
                impact_type=None,
                load_exceedance=load_exceedance,
                load_supplementary_sources=True
            )
            
            if supplement.shape[0] == 0:
                continue
            
            # Apply weights and scale values
            supplement = self._normalise_weights(supplement, total_weight)
            supplement = supplement.rename(columns={'value': 'original_value', 'exposure_type': 'original_exposure_type'})
            supplement['exposure_type'] = exposure_type
            supplement['value'] = supplement['value_fraction'] * get_total_exposed_value(exposure_type, usd=(exposure_unit == 'USD'))
            
            supplementary_observations_list.append(supplement)
        
        if len(supplementary_observations_list) > 0:
            return pd.concat(supplementary_observations_list, axis=0, ignore_index=True)
        else:
            return pd.DataFrame()


def load_observations(
        exposure_type: str,
        impact_type: Optional[str] = None,
        load_exceedance: bool = True,
        load_supplementary_sources: bool = True
    ):
    """Load observations for a given exposure type.
    
    This function is a wrapper around ObservationLoader for backward compatibility.
    
    Args:
        exposure_type: Exposure type to load observations for
        impact_type: Impact type to filter observations for
        load_exceedance: Whether to load uncalibrated exceedance curves as additional observations 
            (when specified in hierarchy). Name of analysis to load exceedance data from is taken 
            from CONFIG["uncalibrated_analysis_name"] but specified in the hierarchy file as 
            simply 'uncalibrated'.
        load_supplementary_sources: Whether to load supplementary observation sources 
            (when specified in hierarchy)
            
    Returns:
        DataFrame of observations
    """
    loader = ObservationLoader(
        observations_df=OBSERVATIONS_FULL,
        hierarchy_df=HIERARCHY_FULL
    )
    return loader.load_for_exposure(
        exposure_type=exposure_type,
        impact_type=impact_type,
        load_exceedance=load_exceedance,
        load_supplementary_sources=load_supplementary_sources
    )


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
