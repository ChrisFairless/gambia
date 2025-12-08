from pathlib import Path
from climada_gambia.config import CONFIG

def gather_impact_function_metadata(filter={}):
    impf_list = []
    for impf_dict in CONFIG.get("impact_functions"):
        hazard_type = impf_dict.get("hazard_type")
        hazard_source = impf_dict.get("hazard_source")
        for impf in impf_dict.get("impfs"):
            if not impf['enabled']:
                continue
            impf['hazard_type'] = hazard_type
            impf['hazard_source'] = hazard_source
            impf['hazard_dir'] = Path(CONFIG["data_dir"], "hazard", f'{impf["hazard_type"]}_{impf["hazard_source"]}', 'haz')
            impf['exposure_node'] = CONFIG.get("exposures", {}).get(impf['exposure_type'], {}).get(impf['exposure_source'], {}).get("present", {})  # for now
            impf['exposure_dir'] = Path(CONFIG["data_dir"], "exposures", f'{impf["exposure_type"]}_{impf["exposure_source"]}', 'exp')
            impf['hazard_node'] = CONFIG.get("hazard", {}).get(hazard_type, {}).get(hazard_source, {})
            impf['calibrated_string'] = "calibrated" if impf['calibrated'] else "uncalibrated"
            impf['impact_dir'] = Path(CONFIG["output_dir"], impf["calibrated_string"], "impacts", f"{impf['exposure_type']}_{impf['exposure_source']}")
            impf['impf_file_path'] = Path(CONFIG["data_dir"], impf.get("dir"), impf.get("files"))

            append = True
            for key in filter.keys():
                if key not in impf.keys():
                    raise ValueError(f"Filter key {key} not found in impf_dict keys: {list(impf.keys())}")
                if impf[key] != filter[key]:
                    append = False     
               
            if append:
                impf_list.append(impf)
    return impf_list


def gather_hazard_metadata(hazard_type, hazard_source, flatten=False):
    haz_conf = CONFIG.get("hazard", {}).get(hazard_type, {}).get(hazard_source, {})
    if not haz_conf:
        raise ValueError(f"Could not find hazard config information at hazard - {hazard_type} - {hazard_source}")

    haz_filepaths = []
    haz_dicts = {}
    for scenario, node in haz_conf.items():
        haz_dir = node.get("dir") or node.get("directory")
        if not haz_dir:
            haz_dir = f"hazard/{hazard_type}_{hazard_source}/haz/"

        haz_files = None
        if "files" in node:
            haz_files = node["files"]
        else:
            raise ValueError("Hazard configuration missing 'files' key")

        if not isinstance(haz_files, list):
            haz_files = [haz_files]

        haz_files = [Path(CONFIG["data_dir"], haz_dir, fn) for fn in haz_files]
        haz_filepaths.extend(haz_files)
        haz_dicts[scenario] = haz_files

    if flatten:
        return haz_filepaths
    return haz_dicts