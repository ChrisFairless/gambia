#!/usr/bin/env python3
"""
Manager for loading, scaling, and transforming CLIMADA impact functions.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from climada.entity import ImpactFunc, ImpactFuncSet


class ImpactFunctionManager:
    """Manages loading, scaling, and transformation of CLIMADA impact functions."""
    
    def __init__(self, filepath: Path, hazard_type: str):
        """
        Args:
            filepath: Path to CSV file with columns [id, intensity, mdd, paa]
            hazard_type: CLIMADA hazard abbreviation (e.g., 'FL' for flood)
        """
        self.filepath = Path(filepath)
        self.hazard_type = hazard_type
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Impact function file not found: {self.filepath}")
    
    def load_impfset(self, scale_x: float = 1.0, scale_y: float = 1.0) -> ImpactFuncSet:
        """Load impact function set from CSV with optional scaling.
        
        Args:
            scale_x: Scaling factor to apply to intensity (x-axis) values.
                     Default is 1.0 (no scaling).
            scale_y: Scaling factor to apply to mdd (mean damage degree) values.
                     Default is 1.0 (no scaling).
        
        Returns:
            ImpactFuncSet containing the loaded impact functions.
            
        Raises:
            ValueError: If required columns are missing from the CSV.
        """
        df = pd.read_csv(self.filepath)
        
        # Validate required columns
        required_cols = {'id', 'intensity', 'mdd', 'paa'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        
        impf_list = []
        for impf_id in df['id'].unique():
            df_id = df[df['id'] == impf_id]
            impf = ImpactFunc(
                haz_type=self.hazard_type,
                id=impf_id,
                intensity=df_id['intensity'].values * scale_x,
                mdd=df_id['mdd'].values * scale_y,
                paa=df_id['paa'].values
            )
            impf_list.append(impf)
        
        return ImpactFuncSet(impf_list)
    
    def load_impf(self, scale_x: float = 1.0, scale_y: float = 1.0) -> ImpactFunc:
        """Load impact function set from CSV with optional scaling.
        Fails if CSV contains multiple impact functions.
        
        Args:
            scale_x: Scaling factor to apply to intensity (x-axis) values.
                      Default is 1.0 (no scaling).
            scale_y: Scaling factor to apply to mdd (mean damage degree) values.
                      Default is 1.0 (no scaling). (Impacts are not capped at 1)
        
        Returns:
            Single ImpactFunc object.
            
        Raises:
            ValueError: If the CSV contains multiple impact functions.
        """
        impf_set = self.load_impfset(scale_x=scale_x, scale_y=scale_y)
        impf_ids = impf_set.get_ids(haz_type=self.hazard_type)
        
        if len(impf_ids) > 1:
            raise ValueError(
                f"Multiple impact function IDs found in impact function set: {impf_ids}. "
                "Expected only one."
            )
        
        if len(impf_ids) == 0:
            raise ValueError("No impact functions found in the set.")
        
        impf_id = impf_ids[0]
        return impf_set.get_func(haz_type=self.hazard_type, fun_id=impf_id)
    
    @staticmethod
    def apply_scaling(impf: ImpactFunc, scale_x: float, scale_y: float) -> ImpactFunc:
        """Scale intensity (x-axis) and mdd (y-axis) of impact function.
        
        Args:
            impf: Original impact function to scale.
            scale_x: Scaling factor for intensity values (x-axis).
            scale_y: Scaling factor for mdd values (y-axis).
        
        Returns:
            New ImpactFunc with scaled values.
        """
        scaled_impf = ImpactFunc(
            haz_type=impf.haz_type,
            id=impf.id,
            intensity=impf.intensity * scale_x,
            mdd=impf.mdd * scale_y,
            paa=impf.paa
        )
        return scaled_impf
    
    @staticmethod
    def create_step_function(impf: ImpactFunc, threshold: float, impf_id: int = 1) -> ImpactFuncSet:
        """Create a step function from an impact function at a given impact threshold.
        
        This creates a binary step function where damage jumps from 0 to 100% at the chosen 
        threshold intensity. The threshold intensity is calculated by interpolating 
        the original impact function at the specified damage threshold.
        
        Args:
            impf: Original impact function to derive threshold from.
            threshold: Damage threshold (0-1) at which to create the step.
            impf_id: ID to assign to the step function. Default is 1.
        
        Returns:
            ImpactFuncSet containing the step function.
        """
        # Calculate the intensity at which the given damage threshold occurs
        impacts = impf.calc_mdr(impf.intensity)
        assert np.array_equal(impacts, np.sort(impacts)), "This code only works with monotonically increasing impact functions"
        threshold_intensity = np.interp(
            threshold, 
            impacts, 
            impf.intensity
        )
        
        impf_step = ImpactFunc.from_step_impf(
            intensity=(0, threshold_intensity, impf.intensity.max()),
            haz_type=impf.haz_type,
            mdd=(0, 1),
            paa=(1, 1),
            impf_id=impf_id
        )

        if threshold <= impf.calc_mdr(threshold_intensity).max():
            assert np.abs(impf.calc_mdr(threshold_intensity) - threshold) < 1e-3, "Step function threshold intensity does not match expected damage threshold"
        
        return ImpactFuncSet([impf_step])
    
    @staticmethod
    def to_csv(impf: ImpactFunc, filepath: [Path, str], impf_id: int = 1):
        """Save impact function to CSV format compatible with CLIMADA.
        
        Args:
            impf: Impact function to save.
            filepath: Path where CSV file will be written.
            impf_id: ID to assign in the CSV file. Default is 1.
        """
        df = pd.DataFrame({
            'intensity': impf.intensity,
            'paa': impf.paa,
            'mdd': impf.mdd,
            'id': impf_id
        })
        
        filepath = Path(filepath)
        if not filepath.parent.exists():
            raise FileNotFoundError(f"Directory does not exist: {filepath.parent}")
        df.to_csv(filepath, index=False)
