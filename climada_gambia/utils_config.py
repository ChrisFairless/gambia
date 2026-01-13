from pathlib import Path
from climada_gambia.config import CONFIG
from climada_gambia.metadata_impact import MetadataImpact

def gather_impact_calculation_metadata(filter={}):
    """Gather impact calculation metadata and return as MetadataImpact instances.
    
    Args:
        filter: Dictionary of key-value pairs to filter impact calculations to matching values
        
    Returns:
        List of MetadataImpact instances
    """
    impf_list = []
    for impf_dict in CONFIG.get("impact_functions"):
        hazard_type = impf_dict.get("hazard_type")
        hazard_source = impf_dict.get("hazard_source")
        for impf_raw in impf_dict.get("impfs"):
            if not impf_raw.get('enabled', True):
                continue
            
            # Create enriched MetadataImpact instance from config
            impf = MetadataImpact.from_config_impf(CONFIG, hazard_type, hazard_source, impf_raw)
            
            # Apply filters
            append = True
            for key, value in filter.items():
                if key not in impf.keys():
                    raise ValueError(f"Filter key {key} not found in impf keys: {list(impf.keys())}")
                if impf[key] != value:
                    append = False
                    break
               
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