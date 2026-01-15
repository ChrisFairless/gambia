"""Config metadata container for CLIMADA Gambia project."""

from pathlib import Path
from typing import Optional
from climada_gambia.config import CONFIG

class MetadataConfig:
    """Base class for configuration metadata containers."""
    
    def __init__(self, analysis_name: Optional[str] = None, data: Optional[dict] = None):
        if not analysis_name:
            analysis_name = CONFIG["default_analysis_name"]
        if data:
            self._data = data
        else:
            self._data = {}
        self["analysis_name"] = analysis_name

    
    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return self._data[key]

    def __setitem__(self, key, value):
        """Allow dictionary-style assignment with validation."""
        if key == "analysis_name":
            raise KeyError("It's too dangerous to modify 'analysis_name' after initialization.")
        self._data[key] = value
    
    def get(self, key, default=None):
        """Get value with default."""
        return self._data.get(key, default)
    
    def keys(self):
        """Return keys."""
        return self._data.keys()
    
    def _ensure_directory(self, path: Path) -> Path:
        """Ensure directory exists, checking parent directory first.
        
        Args:
            path: Path to directory to create
            
        Returns:
            The path that was created
            
        Raises:
            FileNotFoundError: If parent directory does not exist
        """
        if not path.parent.exists():
            raise FileNotFoundError(
                f"Parent directory does not exist: {path.parent}. "
                f"Cannot create {path}"
            )
        path.mkdir(exist_ok=True)
        return path

    @property
    def analysis_name(self) -> str:
        """Name of the analysis (e.g., 'uncalibrated', 'calibrated')."""
        return self._data["analysis_name"]


    # Path construction methods:
    def data_dir(self, create: bool = False) -> Path:
        """Base data directory from CONFIG.
        Returns:
            Path to data directory
        """
        return Path(CONFIG["data_dir"])

    def base_output_dir(self) -> Path:
        """Base output directory from CONFIG.
        Returns:
            Path to output directory
        """
        return Path(CONFIG["output_dir"])

    def analysis_output_dir(self, create: bool = False) -> Path:
        """Get output directory for the analysis.
        
        Args:
            create: If True, ensure directory exists (checks parent and creates with exist_ok=True)
        Returns:
            Path to analysis output directory
        """
        path = Path(self.base_output_dir(), self.analysis_name.split("/"))
        if not self.base_output_dir().exists():
            raise FileNotFoundError(
                f"Base output directory does not exist: {self.base_output_dir()}. "
                f"Cannot create calibration working directory {path}"
            )
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def exceedance_output_dir(self, create: bool = False) -> Path:
        """Get directory for exceedance curve outputs.
        
        Args:
            create: If True, ensure directory exists (checks parent and creates with exist_ok=True)
            
        Returns:
            Path to exceedance output directory
        """
        path = Path(self.analysis_output_dir(create), "exceedance")
        if create:
            self._ensure_directory(path)
        return path

    def exceedance_csv_path(self, create: bool = False) -> Path:
        """Get path for exceedance curve CSV file.
        
        Args:
            create: If True, ensure parent directory exists (checks grandparent and creates with exist_ok=True)
            
        Returns:
            Path to exceedance CSV file
        """
        output_dir = self.exceedance_output_dir(create=create)
        base_name = "exceedance.csv"
        return Path(output_dir, base_name)

    
