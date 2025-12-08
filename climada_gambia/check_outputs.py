#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import os
from climada.engine import Impact

from config import CONFIG


def check_impacts(conf, output_dir, verbose=False):
    for i, impf_dict in enumerate(conf.get("impact_functions")):
        haz_type = impf_dict.get("hazard_type")
        haz_source = impf_dict.get("hazard_source")
        if verbose:
            print(f'Checking impact outputs for hazard {haz_type} - {haz_source}')

        haz_node = conf.get("hazard", {}).get(haz_type, {}).get(haz_source, {})
        
        for j, impf in enumerate(impf_dict.get("impfs")):
            if not impf['enabled']:
                continue
            exposure_type = impf.get("exposure_type")
            exposure_source = impf.get("exposure_source")
            calibration = "calibrated" if impf.get("calibrated") else "uncalibrated"
            if verbose:
                print(f'... exposure {exposure_type} - {exposure_source}')

            imp_dir = Path(output_dir, calibration, "impacts", f"{exposure_type}_{exposure_source}")
            if not os.path.exists(imp_dir):
                print(f"    MISSING: no impact output directory at {imp_dir}")
                continue

            for scenario, haz_data in haz_node.items():
                print(f'    ... scenario {scenario}')
                haz_filename_list = haz_data['files']
                if not isinstance(haz_filename_list, list):
                    haz_filename_list = [haz_filename_list]
                imp_filename_list = [f'impact_{exposure_type}_{exposure_source}_{haz_source}_{fn}' for fn in haz_filename_list]
                for fn in imp_filename_list:
                    imp_path = Path(imp_dir, fn)
                    if not os.path.exists(imp_path):
                        print(f"    MISSING: no impact file at {imp_path}")
                        continue

                    imp = Impact.from_hdf5(imp_path)                    
                    if imp.imp_mat is None:
                        print(f"    WARNING: impact has no impact matrix saved: {fn}")
                    if imp.aai_agg == 0:
                        print(f"    WARNING: impact has no AAI of 0: {fn}")



def main(verbose=False):
    output_dir = Path(CONFIG.get("output_dir"))
    print("-------- Checking impact calculations ---------")
    check_impacts(CONFIG, output_dir, verbose=verbose)


if __name__ == "__main__":
    main(verbose=True)