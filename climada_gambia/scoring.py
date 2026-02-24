#!/usr/bin/env python3
"""
ScoringEngine module for comparing modelled impacts to observations.

This module provides a flexible framework for calculating cost scores when comparing
modelled impact curves against observed data. It supports pluggable cost functions
and handles the complexity of interpolating curves to return periods, calculating
costs, and aggregating scores.
"""

from typing import Callable, Tuple, Dict
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================================
# Cost Functions
# ============================================================================

def squared_error(modelled: float, target: float) -> float:
    """
    Simple squared error cost function.
    
    This is the default cost function. It penalizes deviations quadratically,
    making larger errors much more costly than small ones.
    
    Args:
        modelled: The modelled impact value
        target: The observed/target value
        
    Returns:
        Cost as (modelled - target)^2
    """
    return (modelled - target) ** 2


def normalized_squared_error(modelled: float, target: float) -> float:
    """
    Normalized squared error cost function.
    
    WARNING: This function is problematic with small target values, as it can
    produce very large costs. It's included for completeness but not recommended
    for general use.
    
    Args:
        modelled: The modelled impact value
        target: The observed/target value
        
    Returns:
        Cost as ((modelled - target) / target)^2
    """
    if target == 0:
        return modelled ** 2
    return ((modelled - target) / target) ** 2


def absolute_error(modelled: float, target: float) -> float:
    """
    Absolute error cost function.
    
    Linear penalty for deviations. Less sensitive to outliers than squared error.
    Good when you want to treat all deviations equally regardless of magnitude.
    
    Args:
        modelled: The modelled impact value
        target: The observed/target value
        
    Returns:
        Cost as |modelled - target|
    """
    return abs(modelled - target)


def relative_error(modelled: float, target: float) -> float:
    """
    Relative error cost function.
    
    Normalizes errors by the target value, useful when comparing impacts of
    very different magnitudes. Better behaved than normalized_squared_error.
    
    Args:
        modelled: The modelled impact value
        target: The observed/target value
        
    Returns:
        Cost as |modelled - target| / target
    """
    if target == 0:
        return abs(modelled)
    return abs(modelled - target) / target


# Cost function lookup dictionary
COST_FUNCTIONS = {
    'squared_error': squared_error,
    'normalized': normalized_squared_error,
    'absolute': absolute_error,
    'relative': relative_error
}


def calc_aal(impact, event_frequency):
    """Calculate Average Annual Loss/Impact from event-based data.

    Uses: AAL = Σ impact_i × event_frequency_i

    IMPORTANT: event_frequency is NOT exceedance frequency (1/return_period).
    For return-period-based hazards, event frequencies are the differences
    between consecutive exceedance frequencies:
        event_freq_i = exceedance_freq_i - exceedance_freq_(i+1)
    They represent the probability of each discrete severity bin.

    Args:
        impact: Array of impact values per event (or per RP after grouping)
        event_frequency: Array of event frequencies (NOT exceedance frequencies)
    """
    impact = np.asarray(impact, dtype=float)
    event_frequency = np.asarray(event_frequency, dtype=float)
    assert np.all(event_frequency > 0), "Event frequencies must be positive"
    aal = np.sum(impact * event_frequency)
    assert aal >= 0, "Calculated AAL should be non-negative"
    return aal


def calc_aal_trapezoidal(impact, exceedance_frequency):
    """Calculate AAL using trapezoidal integration over exceedance frequencies.

    Mathematically equivalent to calc_aal when event frequencies are consistent,
    but works directly with exceedance frequencies (1/return_period).
    Useful as a cross-check validation.

    Assumes impact drops to 0 beyond the rarest modelled event.

    Args:
        impact: Array of impact values, one per return period level
        exceedance_frequency: Array of exceedance frequencies (= 1/return_period)
    """
    impact = np.asarray(impact, dtype=float)
    exceedance_frequency = np.asarray(exceedance_frequency, dtype=float)
    assert np.all(exceedance_frequency > 0), "Exceedance frequencies must be positive"

    # Sort by exceedance frequency descending (most frequent / lowest RP first)
    sort_idx = np.argsort(exceedance_frequency)[::-1]
    exc_sorted = exceedance_frequency[sort_idx]
    imp_sorted = impact[sort_idx]

    # Assumption: impacts are constant beyond the rarest modelled event, so we can append a final point at (exceedance_freq=0, impact=imp_sorted[-1])
    exc_sorted = np.append(exc_sorted, 0)
    imp_sorted = np.append(imp_sorted, imp_sorted[-1])
    # Assumption: impacts are zero at frequency = 1 (RP=1), so we can prepend a point at (exceedance_freq=1, impact=0)
    # For Aqueduct where the 2-year RP events are usually zero, this is a reasonable assumption.
    exc_sorted = np.insert(exc_sorted, 0, 1)
    imp_sorted = np.insert(imp_sorted, 0, 0)

    # Trapezoidal: AAL = Σ 0.5 * (L_i + L_{i+1}) * |Δν|
    aal = np.sum(0.5 * (imp_sorted[:-1] + imp_sorted[1:]) * np.abs(np.diff(exc_sorted)))
    assert aal >= 0, "Calculated AAL should be non-negative"
    return aal

class ScoringEngine:
    """
    Calculates cost scores comparing modelled impacts to observations.
    
    This class handles the complete workflow of:
    1. Interpolating exceedance curves to observation return periods
    2. Calculating costs using a pluggable cost function
    3. Weighting and aggregating costs across observations
    4. Producing summary statistics
    
    Example:
        >>> engine = ScoringEngine(cost_function=squared_error)
        >>> observations_with_costs, scores = engine.compare_to_observations(
        ...     curves, observations, impf_dict
        ... )
    """
    
    def __init__(self, cost_function: Callable[[float, float], float] = None):
        """
        Initialize the ScoringEngine with a cost function.
        
        Args:
            cost_function: Function with signature (modelled, target) -> cost.
                          If None, defaults to squared_error.
        """
        if cost_function is None:
            cost_function = squared_error
        
        if not callable(cost_function):
            raise TypeError("cost_function must be callable")
        
        self.cost_function = cost_function
    
    def compare_to_observations(
        self, 
        curves: pd.DataFrame, 
        observations: pd.DataFrame,
        impf_dict: dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main entry point: compare modelled curves to observations.
        
        This method:
        - Filters observations and exceedance curves by impact subtype
        - Interpolates curves to observation return periods
        - Calculates costs for lower/mid/upper RP bounds
        - Applies observation weights
        - Aggregates into summary scores
        
        Args:
            curves: DataFrame with exceedance curves. Must include columns:
                   scenario, event_frequency, exceedance_frequency, impact_fraction, impact_type
            observations: DataFrame with observation data. Must include columns:
                         exposure_type, impact_type, impact_statistic, 
                         value_fraction, rp_lower, rp_mid, rp_upper, weight
            impf_dict: Metadata dictionary containing:
                      - exposure_type: Type of exposure
                      - impact_type: Main impact type
                      - thresholds: Dict of threshold impact types
        
        Returns:
            Tuple of (observations_with_costs, summary_scores):
            - observations_with_costs: Original observations with added columns:
                model_lower, model_mid, model_upper (interpolated values)
                cost_lower, cost_mid, cost_upper (raw costs)
                weighted_cost_lower, weighted_cost_mid, weighted_cost_upper
            - summary_scores: DataFrame indexed by impact_type with columns
                weighted_cost_lower, weighted_cost_mid, weighted_cost_upper
                containing sqrt(sum(costs)) for each group, plus 'TOTAL' row
        """
        # Validate inputs
        assert np.all(observations['exposure_type'] == impf_dict['exposure_type']), \
            'Please filter observations to exposure type'
        
        if observations.shape[0] == 0:
            print('No observations found for this exposure type; skipping scoring.')
            return pd.DataFrame(), pd.DataFrame()
        
        # Get all impact subtypes (main type + threshold types)
        impact_subtypes = impf_dict.impact_type_list(observations=observations)
        all_obs = []
        
        # Process each impact subtype
        for impact_subtype in impact_subtypes:
            observations_subset = observations[observations['impact_type'] == impact_subtype]
            observations_subset = observations_subset[observations_subset['weight'] > 0]
            
            if observations_subset.shape[0] == 0:
                continue
            
            curves_subset = curves[curves['impact_type'] == impact_subtype]
            scenario_df = curves_subset[curves_subset['scenario'] == "present"]
            
            if scenario_df.shape[0] == 0:
                raise ValueError(
                    f"{impf_dict['analysis_name']} - couldn't find exceedance curves for {impf_dict['exposure_type']} "
                    f"and {impact_subtype} – is there some unexpected combination of "
                    f"observations that wasn't modelled?"
                )
            
            scenario_mean = {}
            scenario_aal = {}
            for rp_level in ['lower', 'mid', 'upper']:
                if 'rp_level' in scenario_df.columns and np.all(scenario_df['rp_level'].notna()):
                    filtered = scenario_df[scenario_df['rp_level'] == rp_level]
                else:
                    if 'rp_level' in scenario_df.columns:
                        assert np.all(scenario_df['rp_level'].isna()), "Inconsistent rp_level data in curves"
                    filtered = scenario_df

                grouped = filtered.groupby('exceedance_frequency').agg(
                    impact_fraction=('impact_fraction', 'mean'),
                    event_frequency=('event_frequency', 'sum')
                )
                # Normalize event frequency by number of hazard files to avoid double-counting
                n_haz_files = filtered['hazard_filepath'].nunique() if 'hazard_filepath' in filtered.columns else 1
                if n_haz_files > 1:
                    grouped['event_frequency'] /= n_haz_files

                # Validate: event frequencies should sum to the max exceedance frequency
                assert np.isclose(grouped['event_frequency'].sum(), grouped.index.max(), rtol=1e-4), \
                    f"Event frequencies ({grouped['event_frequency'].sum()}) should sum to max exceedance frequency ({grouped.index.max()})"

                scenario_mean[rp_level] = grouped['impact_fraction']
                scenario_aal[rp_level] = calc_aal(
                    impact=grouped['impact_fraction'].values,
                    event_frequency=grouped['event_frequency'].values
                )
            
            # Split observations by statistic type
            ix_aal = observations_subset['impact_statistic'] == 'aal'
            ix_event = observations_subset['impact_statistic'] == 'event'
            ix_rp = observations_subset['impact_statistic'] == 'rp'
            ix_event = ix_event + ix_rp
            assert np.all(ix_aal + ix_event == True), "Unexpected impact_statistic values in observations"
            
            # Process event/RP observations (need interpolation)
            observations_event = observations_subset[ix_event]
            if observations_event.shape[0] > 0:
                # Interpolate for each return period bound
                observations_event = observations_event.copy()
                observations_event['model_lower'] = self._interpolate_curve_values(
                    scenario_mean['lower'], observations_event['rp_lower']
                )
                observations_event['model_mid'] = self._interpolate_curve_values(
                    scenario_mean['mid'], observations_event['rp_mid']
                )
                observations_event['model_upper'] = self._interpolate_curve_values(
                    scenario_mean['upper'], observations_event['rp_upper']
                )
            
            # Process AAL observations (no interpolation needed)
            observations_aal = observations_subset[ix_aal]
            if observations_aal.shape[0] > 0:
                observations_aal = observations_aal.copy()
                observations_aal['model_lower'] = scenario_aal['lower']
                observations_aal['model_mid'] = scenario_aal['mid']
                observations_aal['model_upper'] = scenario_aal['upper']
            
            # Combine AAL and event observations
            observations_combined = pd.concat([observations_aal, observations_event])
            
            # Calculate costs for each bound
            observations_combined['cost_lower'] = [
                self._calculate_cost(modelled, target) 
                for modelled, target in zip(
                    observations_combined['model_lower'], 
                    observations_combined['value_fraction']
                )
            ]
            observations_combined['cost_mid'] = [
                self._calculate_cost(modelled, target)
                for modelled, target in zip(
                    observations_combined['model_mid'],
                    observations_combined['value_fraction']
                )
            ]
            observations_combined['cost_upper'] = [
                self._calculate_cost(modelled, target)
                for modelled, target in zip(
                    observations_combined['model_upper'],
                    observations_combined['value_fraction']
                )
            ]
            
            all_obs.append(observations_combined)
        
        # Combine all observations
        all_obs = pd.concat(all_obs)
        
        # Normalize weights to sum to 1
        all_obs['weight'] = all_obs['weight'] / all_obs['weight'].sum()
        
        # Calculate weighted costs
        all_obs['weighted_cost_lower'] = all_obs['weight'] * all_obs['cost_lower']
        all_obs['weighted_cost_mid'] = all_obs['weight'] * all_obs['cost_mid']
        all_obs['weighted_cost_upper'] = all_obs['weight'] * all_obs['cost_upper']
        
        # Optional: save detailed observations for debugging
        # all_obs.to_csv(Path(impf_dict.impact_output_dir(), 'deleteme_testing_analysis_obs.csv'))
        
        # Aggregate scores
        scores = self._aggregate_scores(all_obs)
        
        return all_obs, scores
    
    def _interpolate_curve_values(
        self, 
        scenario_mean: pd.Series, 
        return_periods: pd.Series
    ) -> np.ndarray:
        """
        Interpolate exceedance curve to specific return period values.
        
        Args:
            scenario_mean: Series with exceedance_frequency as index, impact_fraction as values.
                          Return periods are computed as 1/exceedance_frequency.
            return_periods: Series of return period values to interpolate to
            
        Returns:
            Array of interpolated impact fraction values
        """
        # Convert exceedance frequency to return period (rp = 1/exceedance_freq)
        exceedance_frequencies = scenario_mean.index.values
        rp_curve = exceedance_frequencies ** -1
        impact_curve = scenario_mean.values
        
        # Reverse arrays for interpolation (np.interp requires increasing x values)
        return np.interp(return_periods, rp_curve[::-1], impact_curve[::-1])
    
    def _calculate_cost(self, modelled_value: float, observed_value: float) -> float:
        """
        Calculate cost using configured cost function.
        
        Handles edge cases like NaN and inf values.
        
        Args:
            modelled_value: The modelled impact value
            observed_value: The observed target value
            
        Returns:
            Cost value as float
        """
        # Handle edge cases
        if np.isnan(modelled_value) or np.isnan(observed_value):
            return np.nan
        if np.isinf(modelled_value) or np.isinf(observed_value):
            return np.inf
        
        return self.cost_function(modelled_value, observed_value)
    
    def _aggregate_scores(self, observations_with_costs: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate individual observation costs into summary scores.
        
        Calculates sqrt(sum(weighted_costs)) for each impact_type group,
        plus a TOTAL row with overall scores.
        
        Args:
            observations_with_costs: DataFrame with weighted_cost_* columns
            
        Returns:
            DataFrame indexed by impact_type with cost summary columns
        """
        trimmed_obs = observations_with_costs[[
            'impact_type', 
            'weighted_cost_lower', 
            'weighted_cost_mid', 
            'weighted_cost_upper'
        ]]
        
        # Aggregate by impact type: sqrt(sum(costs))
        scores = trimmed_obs.groupby('impact_type')[[
            'weighted_cost_lower', 
            'weighted_cost_mid', 
            'weighted_cost_upper'
        ]].agg(lambda x: np.sqrt(sum(x)))
        
        # Add TOTAL row: sqrt(sum(all costs))
        scores.loc["TOTAL"] = trimmed_obs.set_index('impact_type').agg(
            lambda x: np.sqrt(sum(x)), axis=0
        )
        
        return scores
