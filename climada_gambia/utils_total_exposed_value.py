from pathlib import Path
import pandas as pd

total_exposed_value_path = Path('/Users/chrisfairless/Library/CloudStorage/OneDrive-Personal/Projects/UNU/gambia2025/data/total_exposed_value.csv')
TOTAL_EXPOSED_VALUE = pd.read_csv(total_exposed_value_path)

def get_total_exposed_value_df(exposure_type: str = None):
    output_cols = ["exposure_type", "units", "total_exposed_value", "total_exposed_USD"]
    if not exposure_type:
        return TOTAL_EXPOSED_VALUE[output_cols]
    total_exposed = TOTAL_EXPOSED_VALUE[TOTAL_EXPOSED_VALUE["exposure_type"] == exposure_type]
    assert total_exposed.shape[0] == 1, f"Expected one match for total exposed value of {exposure_type}"
    return total_exposed[output_cols]


def get_total_exposed_value(exposure_type: str, usd: bool):
    df = get_total_exposed_value_df(exposure_type)
    if usd:
        return df['total_exposed_USD'].values[0]
    return df['total_exposed_value'].values[0]


def get_total_exposed_units(exposure_type: str, usd: bool):
    if usd:
        return 'USD'
    return get_total_exposed_value_df(exposure_type)['units'].values[0]
