"""Centralized path construction for CLIMADA Gambia project."""

from pathlib import Path


class MetadataCalibration:
    """Path construction for calibration and analysis that doesn't require impact metadata."""
    
    def __init__(self, config: dict, analysis_name: str):
        """
        Args:
            config: Configuration dictionary (typically CONFIG)
            analysis_name: Name of the analysis for output directory structure
        """
        self.config = config
        self.base_data_dir = Path(config["data_dir"])
        self.base_output_dir = Path(config["output_dir"])
        self.analysis_name = analysis_name
            
    def calibration_working_dir(self, calibration_name: str) -> Path:
        """Get working directory for calibration run."""
        return Path(self.base_output_dir, calibration_name.split("/"))
        
    def calibration_temp_impf_path(self, working_dir: Path) -> Path:
        """Get path for temporary impact function CSV during calibration."""
        return Path(working_dir, "temp_impf.csv")
        
    def calibration_output_subdir(self, working_dir: Path, rp_level: str) -> Path:
        """Get subdirectory for specific RP level (lower/mid/upper)."""
        return Path(working_dir, rp_level)
