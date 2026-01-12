"""Impact function metadata container for CLIMADA Gambia project."""

import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

from climada_gambia.config import CONFIG
from climada_gambia.utils_total_exposed_value import get_total_exposed_value


class MetadataImpact:
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
        'dir', 'files', 'thresholds', 'scale_impf', 'enabled',
        # Added programmatically
        'analysis_name', 'hazard_abbr', 'hazard_dir', 'exposure_dir', 'exposure_node',
        'hazard_node', 'impact_dir', 'impf_file_path',
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
        impf = dict(impf_raw)
        
        # Add analysis_name from config
        if analysis_name is not None:
            impf['analysis_name'] = analysis_name
        else:
            impf['analysis_name'] = CONFIG["default_analysis_name"]
        
        # Add hazard metadata
        impf['hazard_type'] = hazard_type
        impf['hazard_source'] = hazard_source
        
        # Create instance temporarily to access path methods
        temp_instance = cls(impf)
        
        # Add computed paths using the instance properties
        impf['hazard_dir'] = temp_instance.hazard_dir
        impf['exposure_dir'] = temp_instance.exposure_dir
        impf['impact_dir'] = temp_instance.impact_output_dir
        impf['impf_file_path'] = temp_instance.impact_function_path
        
        # Add config nodes
        impf['exposure_node'] = CONFIG.get("exposures", {}).get(
            impf['exposure_type'], {}).get(impf['exposure_source'], {}).get("present", {})
        impf['hazard_node'] = CONFIG.get("hazard", {}).get(hazard_type, {}).get(hazard_source, {})
        
        # Add total exposed values to exposure node
        if 'exposure_node' in impf and impf['exposure_node']:
            impf['exposure_node']['total_exposed_value'] = get_total_exposed_value(
                impf['exposure_type'], usd=False)
            impf['exposure_node']['total_exposed_USD'] = get_total_exposed_value(
                impf['exposure_type'], usd=True)
        
        # Ensure thresholds dict exists
        if 'thresholds' not in impf:
            impf['thresholds'] = {}
        
        return cls(impf)
    
    def __init__(self, impf_dict: dict):
        """Initialize MetadataImpact with validation.
        
        Args:
            impf_dict: Dictionary containing impact function metadata
            
        Raises:
            ValueError: If required fields are missing
            
        Warns:
            UserWarning: If unexpected fields are present
        """
        # Validate required fields
        missing = self.REQUIRED_FIELDS - set(impf_dict.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        # Warn about unexpected fields
        unexpected = set(impf_dict.keys()) - self.ALLOWED_FIELDS
        if unexpected:
            warnings.warn(
                f"Unexpected fields in impact metadata: {unexpected}. "
                f"Allowed fields are: {self.ALLOWED_FIELDS}",
                UserWarning
            )
        
        self._data = impf_dict
    
    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return self._data[key]
    
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
    
    def get(self, key, default=None):
        """Get value with default."""
        return self._data.get(key, default)
    
    def keys(self):
        """Return keys."""
        return self._data.keys()
    
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
    
    @property
    def analysis_name(self) -> str:
        """Name of the analysis (e.g., 'uncalibrated', 'calibrated')."""
        return self._data["analysis_name"]
    
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
    def scale_impf(self) -> Optional[float]:
        """Scaling factor for impact function."""
        return self._data.get("scale_impf")
    
    @property
    def enabled(self) -> bool:
        """Whether this impact function is enabled."""
        return self._data.get("enabled")
    
    @property
    def hazard_node(self) -> Optional[Dict]:
        """Configuration node for this hazard from CONFIG."""
        return self._data.get("hazard_node")
    
    @property
    def exposure_node(self) -> Optional[Dict]:
        """Configuration node for this exposure from CONFIG."""
        return self._data.get("exposure_node")
    
    @property
    def data_dir(self) -> Path:
        """Base data directory from CONFIG."""
        return Path(CONFIG["data_dir"])

    @property
    def hazard_dir(self) -> Path:
        """Directory containing hazard files.
        
        Returns stored path if available, otherwise computes it from CONFIG.
        """
        stored = self._data.get("hazard_dir")
        if stored:
            return stored
        
        base_data_dir = Path(CONFIG["data_dir"])
        return Path(base_data_dir, "hazard", 
                   f'{self.hazard_type}_{self.hazard_source}', 'haz')
    
    @property
    def exposure_dir(self) -> Path:
        """Directory containing exposure files.
        
        Returns stored path if available, otherwise computes it from CONFIG.
        """
        stored = self._data.get("exposure_dir")
        if stored:
            return stored
        
        base_data_dir = Path(CONFIG["data_dir"])
        return Path(base_data_dir, "exposures", 
                   f'{self.exposure_type}_{self.exposure_source}', 'exp')

    @property
    def impact_file_path(self, hazard_file_stem: str, impact_type: str) -> Path:
        """Get full path for single impact HDF5 file.
        
        Args:
            hazard_file_stem: Stem of the hazard filename (without extension)
            impact_type: Type of impact being calculated
            
        Returns:
            Full path to impact HDF5 file
        """
        output_dir = self.impact_output_dir
        filename = (f'impact_{impact_type}_{self.exposure_type}_'
                   f'{self.exposure_source}_{self.hazard_source}_{hazard_file_stem}.hdf5')
        return Path(output_dir, filename)
    
    @property
    def exceedance_output_dir(self) -> Path:
        """Get directory for exceedance curve outputs."""
        return Path(self.base_output_dir, self.analysis_name, "exceedance")

    @property
    def exceedance_csv_path(self) -> Path:
        """Get path for exceedance curve CSV file."""
        base_name = "exceedance.csv"
        return Path(self.exceedance_output_dir, base_name)
    
    @property
    def exceedance_plot_dir(self) -> Path:
        """Get directory for exceedance curve plots."""
        return Path(self.exceedance_output_dir(), "plots")

    @property
    def exceedance_plot_path(self, impact_type: str, zoom: str = None) -> Path:
        """Get path for exceedance curve plot PNG.
        
        Args:
            impact_type: Type of impact (e.g., 'economic_loss', 'affected')
            zoom: Optional zoom suffix ('zoom', 'zoom_obs', 'zoom_obs_fraction')
        
        Returns:
            Path to exceedance plot PNG file (filename only, not full path)
        """
        base_name = (f"exceedance_{impact_type}_{self.hazard_source}_"
                    f"{self.exposure_source}_{self.exposure_type}")
        if zoom:
            base_name = f"{base_name}_{zoom}"
        return Path(self.exceedance_plot_dir, f"{base_name}.png")
        
    def to_dict(self) -> dict:
        """Return underlying dictionary."""
        return self._data
