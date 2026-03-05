"""Impact calculation metadata container for CLIMADA Gambia project."""

import warnings
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any, Union, List

from climada_gambia.config import CONFIG
from climada_gambia.metadata_config import MetadataConfig
from climada_gambia.utils_total_exposed_value import get_total_exposed_value

VALID_MAIN_IMPACT_TYPES = {'economic_loss', 'displaced'}
VALID_THRESHOLD_IMPACT_TYPES = {'affected', 'damaged', 'destroyed'}
VALID_IMPACT_TYPES = VALID_MAIN_IMPACT_TYPES | VALID_THRESHOLD_IMPACT_TYPES

class MetadataImpact(MetadataConfig):
    """Container for impact calculation metadata, configuration, and path building.
    
    This class encapsulates all metadata related to an impact calculation,
    including hazard, exposure, impact function details, and path construction.
    """
    
    # Required fields that must be present in every impact calculation
    REQUIRED_FIELDS = {
        'hazard_type', 'hazard_source', 'exposure_type', 'exposure_source', 'impact_type'
    }
    
    # Optional fields from config or added programmatically
    OPTIONAL_FIELDS = {
        # From config.py impact_functions
        'dir', 'files', 'thresholds', 'scale_x', 'scale_y', 'enabled',
        # Added programmatically
        'analysis_name', 'hazard_abbr', 'exposure_node', 'hazard_node', 'scenarios',
        # Added during calculation/analysis
        'scores'
    }
    
    # All allowed fields
    ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS

    # Hazard abbreviations
    HAZARD_MAP = {"flood": "FL"}
    
    @classmethod
    def from_config_impf(cls, hazard_type: str, hazard_source: str, impf_raw: dict, analysis_name: Optional[str] = None) -> "MetadataImpact":
        """Create MetadataImpact from impact function dict as specified in config.py.
        
        This enriches the raw impact function dictionary with computed paths and metadata.
        
        Args:
            hazard_type: Type of hazard (from parent config level)
            hazard_source: Source of hazard data (from parent config level)
            impf_raw: Raw impact function dictionary from config
            analysis_name: Optional analysis name to include; if None, uses default from CONFIG
            
        Returns:
            MetadataImpact instance with all computed fields
        """
        
        # Start with copy of raw data
        if isinstance(impf_raw, MetadataImpact):
            impf = impf_raw.to_dict()
        else:
            impf = dict(impf_raw)
        
        # Add analysis_name from config
        if analysis_name is not None:
            impf['analysis_name'] = analysis_name
        if 'analysis_name' not in impf:
            raise ValueError("Must provide 'analysis_name' either as an argument or in the input dictionary.")
        
        # Add hazard metadata
        impf['hazard_type'] = hazard_type
        impf['hazard_source'] = hazard_source
        impf['hazard_abbr'] = cls.HAZARD_MAP.get(hazard_type, None)
        
        # Add config nodes
        impf['exposure_node'] = CONFIG.get("exposures", {}).get(
            impf['exposure_type'], {}).get(impf['exposure_source'], {}).get("present", {})
        impf['hazard_node'] = CONFIG.get("hazard", {}).get(hazard_type, {}).get(hazard_source, {})
        impf["scenarios"] = list(impf['hazard_node'].keys()) if impf['hazard_node'] else []
        
        # Add total exposed values to exposure node
        if 'exposure_node' in impf and impf['exposure_node']:
            impf['exposure_node']['total_exposed_value'] = get_total_exposed_value(
                impf['exposure_type'], usd=False)
            impf['exposure_node']['total_exposed_USD'] = get_total_exposed_value(
                impf['exposure_type'], usd=True)
        
        # Ensure thresholds dict exists
        if 'thresholds' not in impf:
            impf['thresholds'] = {}
        
        return cls(impf, analysis_name=analysis_name)
    
    def __init__(self, impf_dict: dict, analysis_name: Optional[str] = None):
        """Initialize MetadataImpact with validation.
        
        Args:
            impf_dict: Dictionary containing impact function metadata
            
        Raises:
            ValueError: If required fields are missing
            
        Warns:
            UserWarning: If unexpected fields are present
        """
        # Validate required fields
        if isinstance(impf_dict, MetadataImpact):
            impf_dict = impf_dict.to_dict()

        self.check_required_fields(impf_dict)
        self.check_allowed_fields(impf_dict)
        
        super().__init__(analysis_name=analysis_name, data=impf_dict)
        if self.analysis_name == "deleteme":
            raise ValueError("MetadataImpact initialized with placeholder analysis_name 'deleteme'.")
    
    @classmethod
    def check_required_fields(cls, impf_dict):
        missing = cls.REQUIRED_FIELDS - set(impf_dict.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

    @classmethod
    def check_allowed_fields(cls, impf_dict):
        unexpected = set(impf_dict.keys()) - cls.ALLOWED_FIELDS
        if unexpected:
            warnings.warn(
                f"Unexpected fields in impact metadata: {unexpected}. "
                f"Allowed fields are: {cls.ALLOWED_FIELDS}",
                UserWarning
            )

    def __setitem__(self, key, value):
        """Allow dictionary-style assignment with validation.
        
        Warns:
            UserWarning: If key is not in allowed fields
        """
        if key not in self.ALLOWED_FIELDS:
            warnings.warn(
                f"Setting unexpected field '{key}'. "
                f"Allowed fields are: {self.ALLOWED_FIELDS}",
                UserWarning
            )
        self._data[key] = value
    
    @property
    def identifier(self) -> str:
        return f"{self.exposure_type}_{self.exposure_source}_{self.hazard_abbr}_{self.hazard_source}"

    @property
    def hazard_type(self) -> str:
        """Type of hazard (e.g., 'flood')."""
        return self._data["hazard_type"]
    
    @property
    def hazard_source(self) -> str:
        """Source of hazard data (e.g., 'aqueduct')."""
        return self._data["hazard_source"]
    
    @property
    def hazard_abbr(self) -> str:
        """Hazard abbreviation (e.g., 'FL' for flood).
        
        If not stored in data, derives from hazard_type.
        """
        if "hazard_abbr" in self._data:
            return self._data["hazard_abbr"]
        
        # Derive from hazard_type
        if self.hazard_type in self.HAZARD_MAP:
            return self.HAZARD_MAP[self.hazard_type]
        raise ValueError(f"No abbreviation found for hazard_type '{self.hazard_type}'")        
    
    @property
    def exposure_type(self) -> str:
        """Type of exposure (e.g., 'housing', 'agriculture')."""
        return self._data["exposure_type"]
    
    @property
    def exposure_source(self) -> str:
        """Source of exposure data (e.g., 'GHS', 'IUCN')."""
        return self._data["exposure_source"]
    
    @property
    def impact_type(self) -> str:
        """Type of impact (e.g., 'economic_loss', 'displaced')."""
        return self._data["impact_type"]

    # All impact types associated with this function, including thresholds and observations    
    def impact_type_list(self, observations: pd.DataFrame) -> list[str]:
        if observations is None:
            observations = pd.DataFrame()
        if observations.shape[0] > 0:
            return list(set([self._data["impact_type"]]) | set(observations["impact_type"]) | set(self._data["thresholds"].keys()))
        return list(set([self._data["impact_type"]]) | set(self._data["thresholds"].keys()))
        
    @property
    def impf_dir(self) -> Optional[str]:
        """Directory containing impact function file."""
        return self._data.get("dir")
    
    @property
    def impf_files(self) -> Optional[Union[str, List[str]]]:
        """Impact function filename(s)."""
        return self._data.get("files")
    
    @property
    def thresholds(self) -> Optional[Dict[str, float]]:
        """Impact thresholds by type (e.g., {'affected': 0.1, 'damaged': 0.5})."""
        return self._data.get("thresholds", {})
    
    @property
    def scale_x(self) -> Optional[float]:
        """Scaling factor for impact function in the x dimension."""
        return self._data.get("scale_x")
    
    @property
    def scale_y(self) -> Optional[float]:
        """Scaling factor for impact function in the y dimension."""
        return self._data.get("scale_y")
    
    @property
    def enabled(self) -> bool:
        """Whether this impact function is enabled."""
        return self._data.get("enabled")
    
    @property
    def hazard_node(self) -> Optional[Dict]:
        """Configuration node for this hazard from CONFIG."""
        return self._data.get("hazard_node")

    @property
    def scenarios(self) -> List[str]:
        """List of scenarios available for this impact function, derived from hazard node."""
        return self._data.get("scenarios")
    
    @property
    def exposure_node(self) -> Optional[Dict]:
        """Configuration node for this exposure from CONFIG."""
        return self._data.get("exposure_node")
    

    # Path construction methods:
    def hazard_dir(self, create: bool = False) -> Path:
        """Directory containing hazard files.
        Returns:
            Path to hazard directory (computed from CONFIG)
        """
        base_data_dir = self.data_dir()
        path = Path(base_data_dir, "hazard", 
                    f'{self.hazard_type}_{self.hazard_source}', 'haz')
        return path
    
    def exposure_dir(self, create: bool = False) -> Path:
        """Directory containing exposure files.
        Returns:
            Path to exposure directory (computed from CONFIG)
        """
        base_data_dir = self.data_dir()
        path = Path(base_data_dir, "exposures", 
                    f'{self.exposure_type}_{self.exposure_source}', 'exp')
        return path
    
    def impact_function_dir(self, create: bool = False) -> Path:
        """Directory containing impact function files.
        
        Args:
            create: If True, ensure directory exists (checks parent and creates with exist_ok=True)
            
        Returns:
            Path to impact function directory
        """
        base_data_dir = self.data_dir()
        impf_dir = self._data.get("dir")
        if not impf_dir:
            raise ValueError(
                "Cannot construct impact_function_dir: missing 'dir' parameter in metadata"
            )
        path = Path(base_data_dir, impf_dir)
        
        if create:
            if "/" in impf_dir:
                self._ensure_directory(path.parent, depth=2)
            self._ensure_directory(path)
        return path

    def impact_function_path(self, create: bool = False) -> Path:
        """Path to impact function CSV file.
        
        Args:
            create: If True, ensure parent directory exists (checks grandparent and creates with exist_ok=True)
            
        Returns:
            Path to impact function file
        """
        # Construct from impf_dir and impf_files
        impf_dir = self.impact_function_dir(create=create)
        impf_files = self._data.get("files")
        if not impf_files:
            raise ValueError(
                "Cannot construct impact_function_path: missing 'files' in metadata"
            )
        
        # Handle case where files might be a list (take first)
        if isinstance(impf_files, list):
            print("Multiple impact function files found; using the first one.")
            impf_file = impf_files[0]
        else:
            impf_file = impf_files
        path = Path(impf_dir, impf_file)
        
        return path

    def impact_output_dir(self, create: bool = False) -> Path:
        """Directory for impact calculation outputs.
        
        Args:
            create: If True, ensure directory exists (checks parent and creates with exist_ok=True)
            
        Returns:
            Path to impact output directory
        """
        impact_parent_dir = Path(self.analysis_output_dir(create=create), "impacts")
        if create:
            self._ensure_directory(impact_parent_dir)
        
        path = Path(impact_parent_dir, f"{self.exposure_type}_{self.exposure_source}")
        if create:
            self._ensure_directory(path)
        return path


    def impact_file_path(self, hazard_file_stem: str, impact_type: str, create: bool = False) -> Path:
        """Get full path for single impact HDF5 file.
        
        Args:
            hazard_file_stem: Stem of the hazard filename (without extension)
            impact_type: Type of impact being calculated
            create: If True, ensure parent directory exists (checks grandparent and creates with exist_ok=True)
            
        Returns:
            Full path to impact HDF5 file
        """
        assert impact_type in VALID_IMPACT_TYPES
        output_dir = self.impact_output_dir(create=create)
        filename = (f'impact_{impact_type}_{self.identifier}_{hazard_file_stem}.hdf5')
        return Path(output_dir, filename)


    def impact_rp_level_file_path(self, hazard_file_stem: str, impact_type: str, rp_level: str, create: bool = False) -> Path:
        """Get full path for single impact HDF5 file.
        
        Args:
            hazard_file_stem: Stem of the hazard filename (without extension)
            impact_type: Type of impact being calculated
            rp_level: Return period level (optional), one of "lower", "mid", "upper"
            create: If True, ensure parent directory exists (checks grandparent and creates with exist_ok=True)
            
        Returns:
            Full path to impact HDF5 file
        """
        assert impact_type in VALID_THRESHOLD_IMPACT_TYPES, f"impact_type '{impact_type}' not valid for creating this file"
        output_dir = self.impact_output_dir(create=create)
        filename = (f'impact_{impact_type}_{self.identifier}_{hazard_file_stem}_{rp_level}.hdf5')
        return Path(output_dir, filename)
    
    def fitted_thresholds_dir(self, create: bool = False) -> Path:
        """Get directory for fitted thresholds CSV file.
        
        Args:
            create: If True, ensure directory exists (checks parent and creates with exist_ok=True)
        Returns:
            Path to fitted thresholds directory
        """
        output_dir = self.impact_output_dir(create=create)
        if create:
            self._ensure_directory(output_dir)
        return output_dir

    def fitted_thresholds_file_path(self, create: bool = False) -> Path:
        """Get path for fitted thresholds CSV file.
        
        Args:
            create: If True, ensure parent directory exists (checks grandparent and creates with exist_ok=True)
            
        Returns:
            Path to fitted thresholds CSV file
        """
        dir_path = self.fitted_thresholds_dir(create=create)
        filename = (f"fitted_thresholds_{self.identifier}.csv")
        return Path(dir_path, filename)

    def exceedance_type_csv_path(self, create: bool = False) -> Path:
        """Get path for exceedance curve CSV file for a specific impact type.
        
        Args:
            create: If True, ensure parent directory exists (checks grandparent and creates with exist_ok=True)
        Returns:
            Path to exceedance CSV file
        """
        output_dir = self.exceedance_csv_dir(create=create)
        base_name = f"exceedance_{self.identifier}.csv"
        return Path(output_dir, base_name)

    def exceedance_plot_path(self, impact_type: str, zoom: str = None, create: bool = False) -> Path:
        """Get path for exceedance curve plot PNG.
        
        Args:
            impact_type: Type of impact (e.g., 'economic_loss', 'affected')
            zoom: Optional zoom suffix ('zoom', 'zoom_obs', 'zoom_obs_fraction')
            create: If True, ensure parent directory exists (checks grandparent and creates with exist_ok=True)
        
        Returns:
            Path to exceedance plot PNG file
        """
        plot_dir = self.exceedance_plot_dir(create=create)
        base_name = (f"exceedance_{impact_type}_{self.identifier}")
        if zoom:
            base_name = f"{base_name}_{zoom}"
        return Path(plot_dir, f"{base_name}.png")
    
    def scored_observations_csv_path(self, create: bool = False) -> Path:
        """Get path for scored observations CSV file.
        
        Args:
            create: If True, ensure parent directory exists (checks grandparent and creates with exist_ok=True)
        Returns:
            Path to scored observations CSV file
        """
        plot_dir = self.exceedance_output_dir(create=create)
        filename = (f"scored_observations_{self.impact_type}_{self.identifier}.csv")
        return Path(plot_dir, filename)
    
    def scores_csv_path(self, create: bool = False) -> Path:
        """Get path for scores CSV file.
        
        Args:
            create: If True, ensure parent directory exists (checks grandparent and creates with exist_ok=True)
        
        Returns:
            Path to scores CSV file
        """
        plot_dir = self.exceedance_output_dir(create=create)
        filename = (f"scores_{self.impact_type}_{self.identifier}.csv")
        return Path(plot_dir, filename) 

    def uncertainty_results_dir(self, create: bool = False) -> Path:
        """Get directory for uncertainty analysis results.
        
        Args:
            scenario: Scenario name
            create: If True, ensure directory exists (checks parent and creates with exist_ok=True)
        Returns:
            Path to uncertainty results directory
        """
        path = Path(self.analysis_output_dir(create=create), "uncertainty")
        if create:
            self._ensure_directory(path)
        return path

    def uncertainty_results_paths(self, scenario: str, create: bool = False) -> Path:
        """Get paths for uncertainty analysis output files.
        
        Args:
            scenario: Scenario name
            create: If True, ensure parent directory exists (checks grandparent and creates with exist_ok=True)
        Returns:
            Dictionary of paths to uncertainty results files
        """
        output_dir = self.uncertainty_results_dir(create=create)
        return {
            "csv": Path(output_dir, f"uncertainty_{scenario}.csv"),
            "plot": Path(output_dir, f"plot_uncertainty_{scenario}.png"),
            "rps": Path(output_dir, f"plot_uncertainty_rps_{scenario}.png")
        }

    def insurance_results_dir(self, create: bool = False) -> Path:
        """Get directory for insurance analysis results.
        
        Args:
            scenario: Scenario name
            create: If True, ensure directory exists (checks parent and creates with exist_ok=True)
        Returns:
            Path to insurance results directory
        """
        path = Path(self.analysis_output_dir(create=create), "insurance")
        if create:
            self._ensure_directory(path)
        return path

    def insurance_results_paths(self, scenario: str, create: bool = False) -> Path:
        """Get paths for insurance analysis output files.
        
        Args:
            scenario: Scenario name
            create: If True, ensure parent directory exists (checks grandparent and creates with exist_ok=True)
        Returns:
            Dictionary of paths to insurance results files
        """
        output_dir = self.insurance_results_dir(create=create)
        return {
            "csv": Path(output_dir, f"insurance_{scenario}.csv"),
            "plot_curve": Path(output_dir, f"plot_insurance_curve_{scenario}.png"),
            "plot_policy_space": Path(output_dir, f"plot_policy_space_{scenario}.png")
        }
    

    def to_dict(self) -> dict:
        """Return underlying dictionary."""
        return self._data
