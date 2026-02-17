"""Centralized path construction for CLIMADA Gambia project."""

from pathlib import Path
from typing import Optional
from climada_gambia.config import CONFIG
from climada_gambia.metadata_config import MetadataConfig
class MetadataCalibration(MetadataConfig):
    """Path construction for calibration and analysis that doesn't require impact metadata."""
    
    def __init__(self, analysis_name: Optional[str] = None):
        """
        Args:
            config: Configuration dictionary (typically CONFIG)
            analysis_name: Name of the analysis for output directory structure
        """
        super().__init__(analysis_name=analysis_name)
            
    def calibration_working_dir(self, create: bool = False) -> Path:
        """Get working directory for calibration analysis."""
        return self.analysis_output_dir(create=create)

    def calibration_temp_impf_path(self, temp_str: str = None, create: bool = False) -> Path:
        """Get path for temporary impact function CSV during calibration."""
        working_dir = self.calibration_working_dir(create=create)
        if not temp_str:
            path = Path(working_dir, f"temp_impf.csv")
        else:
            path = Path(working_dir, f"temp_impf_{temp_str}.csv")
        return path
        
    # def calibration_output_subdir(self, rp_level: str, create: bool = False) -> Path:
    #     """Get subdirectory for specific RP level (lower/mid/upper)."""
    #     working_dir = self.calibration_working_dir(create=create)
    #     path = Path(working_dir, rp_level)
    #     if create:
    #         self._ensure_directory(path)
    #     return path
    
    def calibration_search_csv_path(self, create: bool = False) -> Path:
        """Get path for calibration search results CSV."""
        working_dir = self.calibration_working_dir(create=create)
        path = Path(working_dir, "calibration_search.csv")
        return path

    def calibration_search_plot_path(self, rp_level: str, create: bool = False) -> Path:
        """Get path for calibration search results plot PNG."""
        working_dir = self.calibration_working_dir(create=create)
        path = Path(working_dir, f'calibration_search_score_{rp_level}.png')
        return path

    def calibration_plotting_dir(self, create: bool = False) -> Path:
        """Get directory for calibration plots."""
        working_dir = self.calibration_working_dir(create=create)
        path = Path(working_dir, "plots")
        if create:
            self._ensure_directory(path)
        return path

    def calibration_output_paths(self, create: bool = False) -> dict[str, Path]:
        """Get a dict containing paths for calibration outputs"""
        working_dir = self.calibration_plotting_dir(create=create)
        return {
            'uncertainty_stats_csv': Path(working_dir, "uncertainty_stats.csv"),
            'present_comparison': Path(working_dir, "plot_present_comparison.png"),
            'future_comparison': Path(working_dir, "plot_future_comparison.png"),
            'waterfall': Path(working_dir, "plot_waterfall.png"),
        }
        path = Path(working_dir, f'calibrated_impf_{rp_level}.csv')
        return path