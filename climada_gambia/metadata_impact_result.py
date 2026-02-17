"""Impact calculation metadata container for CLIMADA Gambia project."""

import warnings
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any, Union, List

from climada_gambia.metadata_impact import MetadataImpact


class MetadataImpactResult(MetadataImpact):
    """Container for impact calculation metadata, configuration, and path building.
    
    This class encapsulates all metadata related to an impact calculation,
    including hazard, exposure, impact function details, and path construction.
    """
    
    # Required fields that must be present in every impact calculation
    REQUIRED_FIELDS = {
        'scores',
        'curves',
        'fitted_thresholds'
    } | MetadataImpact.REQUIRED_FIELDS
    
    OPTIONAL_FIELDS = {} | MetadataImpact.OPTIONAL_FIELDS

    # All allowed fields
    ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS

    
    @classmethod
    def from_metadata_impact(cls, impf_dict: dict) -> "MetadataImpactResult":
        """Create MetadataImpactResult from MetadataImpact.
        
        Args:
            impf_dict: MetadataImpact instance containing impact function metadata
            curves: DataFrame of exceedance curves from impact calculation analysis
            scores: DataFrame of scores from impact calculation analysis

        Returns:
            MetadataImpactResult instance with all computed fields
        """
        return cls(impf_dict)
    
    
    def __init__(self, impf_dict: dict):
        """Initialize MetadataImpactResult

        Args:
            impf_dict: MetadataImpact instance containing impact function metadata
            curves: DataFrame of exceedance curves from impact calculation analysis
            scores: DataFrame of scores from impact calculation analysis
        """
        if isinstance(impf_dict, MetadataImpact):
            impf_dict = impf_dict.to_dict()  # convert to dict
        self["curves"] = self.read_curves()
        self["scores"] = self.read_scores()
        self["fitted_thresholds"] = self.get_fitted_thresholds()
        super().__init__(data=impf_dict, analysis_name=impf_dict["analysis_name"])

    def read_curves(self) -> pd.DataFrame:
        """Read exceedance curves from CSV file.

        Returns:
            DataFrame of exceedance curves
        """
        path = self.curves_csv_path()
        return pd.read_csv(path)
    
    def read_scores(self) -> pd.DataFrame:
        """Read scores from CSV file.

        Returns:
            DataFrame of scores
        """
        path = self.scores_csv_path()
        return pd.read_csv(path)
