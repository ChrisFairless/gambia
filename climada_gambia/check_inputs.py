#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import os
import copy

from config import CONFIG

def check_node(node, data_dir, key_path):
    check_node = extract_using_path(key_path)
    if node != check_node:
        raise ValueError(f"This node is not accessed by the path {' - '.join(key_path)}")

    if "enabled" not in node:
        raise ValueError(f"Node at {' - '.join(key_path)} missing 'enabled' key")
    
    if bool(node.get("enabled")):
        check_enabled_node(node, data_dir, key_path or ["root"])


def check_enabled_node(node, data_dir, key_path):
    dir_val = node.get("dir") or node.get("directory")
    if not dir_val:
        hazard_or_exposure = key_path[-4]
        node_type = key_path[-3]
        node_source = key_path[-2]
        node_scenario = key_path[-1]
        node_dir = hazard_or_exposure[0:3]
        dir_val = f"{hazard_or_exposure}/{node_type}_{node_source}/{node_dir}/"
    
    file_vals = None
    if "files" in node:
        file_vals = node["files"]
    # Exposures fall back on 'present' scenario parameters when they're missing their own
    elif "exposures" in key_path:
        fallback_key_path = copy.deepcopy(key_path)
        fallback_key_path[-1] = "present"
        try:
            fallback_node = extract_using_path(fallback_key_path)
            file_vals = fallback_node["files"]
        except Exception as e:
            print(f"ERROR: could not fall back on file info at {' - '.join(fallback_key_path)}")
    if not file_vals:
        print(f"ERROR: Enabled node at {' - '.join(key_path)} missing 'files' key")
        return

    base_dir = Path(data_dir, dir_val)

    exists_dir = os.path.exists(base_dir)
    if not exists_dir:
        print(f"MISSING: Directory not found at {base_dir} (path: {' - '.join(key_path)})")
        return

    # Normalize file_vals into list
    if isinstance(file_vals, str):
        files = [file_vals]
    elif isinstance(file_vals, (list, tuple)):
        files = list(file_vals)
    else:
        print(f"ERROR: Unknown 'files' format at {' - '.join(key_path)} -> {type(file_vals)}")
        return

    for f in files:
        file_path = Path(base_dir, f)
        exists = os.path.exists(file_path)
        if not exists:
            print(f"MISSING: file not found at {file_path} (path: {' - '.join(key_path)})")


def recurse(conf, data_dir, key_path=None, verbose=False):
    if key_path is None:
        key_path = []

    if isinstance(conf, dict):
        active_keys = ("file", "files", "dir", "directory", "enabled")
        if any(k in conf for k in active_keys):
            if verbose:
                print(f'...{" - ".join(key_path)}')
            check_node(conf, data_dir, key_path)

        # Recurse into children
        for k, v in conf.items():
            # skip keys already handled
            if k in active_keys:
                continue
            new_key_path = key_path + [str(k)]
            recurse(v, data_dir, new_key_path, verbose)

    elif isinstance(conf, list):
        for i, item in enumerate(conf):
            new_key_path = key_path + [f"[{i}]"]
            recurse(item, data_dir, new_key_path, verbose)


def check_impfs(conf, data_dir, verbose=False):
    for i, impf_dict in enumerate(conf.get("impact_functions")):
        haz_type = impf_dict.get("hazard_type")
        haz_source = impf_dict.get("hazard_source")
        if verbose:
            print(f'Checking impact functions for hazard {haz_type} / {haz_source}')

        haz_node = conf.get("hazard", {}).get(haz_type, {}).get(haz_source, {})
        if not haz_node:
            print(f' MISSING: No hazard configuration found for {haz_type} / {haz_source} as specified in impact functions')
        else:
            for scenario, node in haz_node.items():
                check_node(node, data_dir, key_path=["hazard", haz_type, haz_source, scenario])

        for j, impf in enumerate(impf_dict.get("impfs")):
            if not impf['enabled']:
                continue
            exposure_type = impf.get("exposure_type")
            exposure_source = impf.get("exposure_source")
            if verbose:
                print(f'... exposure {exposure_type} / {exposure_source}')

            exposure_node = conf.get("exposures", {}).get(exposure_type, {}).get(exposure_source, {})
            if not exposure_node:
                print(f' MISSING: No exposure configuration found for {exposure_type} / {exposure_source} as specified in impact functions')
            else:
                for scenario, node in exposure_node.items():
                    check_node(node, data_dir, key_path=["exposures", exposure_type, exposure_source, scenario])

            impf_path = ["impact_functions", f"[{i}]", "impfs", f"[{j}]"]
            check_enabled_node(impf, data_dir, impf_path)


def extract_using_path(key_path):
    node = CONFIG
    for k in key_path:
        if k[0] == '[':
            node = node[int(k[1:-1])]
        else:
            node = node[k]
    return node


def main(verbose=False):
    data_dir = Path(CONFIG.get("data_dir"))
    print("-------- Checking files exist ---------")
    recurse(CONFIG, data_dir, key_path=[], verbose=verbose)
    print("-------- Checking impact functions work ---------")
    check_impfs(CONFIG, data_dir, verbose=verbose)


if __name__ == "__main__":
    main(verbose=True)