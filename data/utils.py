# utils.py
from IPython.display import display
import yaml
import pandas as pd
from training.utils import ROOT_PATH

def log(msg, verbose: bool = True):
    if verbose:
        print(msg)


def describe_cols(dframe, title: str, verbose: bool = True):
    if verbose:
        print(f"\n[{title}] describe(include='all'):")
        display(dframe.describe(include='all').T)  # Proper formatting
        print(f"[{title}] dtypes:")
        display(dframe.dtypes.to_frame("dtype").T)  # Show types as a DataFrame

def load_column_mapping_by_group(filename: str = "column_groups.yaml", verbose: bool = False) -> dict:
    """
    Load column mapping from grouped YAML format.

    Returns:
        flat_mapping: dict with all original → anonymized mappings.
        grouped_mapping: nested group-wise mapping.
    """
    path = ROOT_PATH / "config" / filename
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    grouped_mapping = data.get("groups", {})
    flat_mapping = {
        original: anonymized
        for group in grouped_mapping.values()
        for original, anonymized in group.items()
    }

    log(f"🔐 Loaded grouped column mapping from {path}", verbose)
    log(f"📦 Groups: {list(grouped_mapping.keys())}", verbose)
    return flat_mapping, grouped_mapping


def apply_column_mapping(df: pd.DataFrame, mapping: dict, verbose: bool = False) -> pd.DataFrame:
    """
    Renames DataFrame columns using a flat mapping dictionary.
    """
    intersecting = {k: v for k, v in mapping.items() if k in df.columns}
    df_renamed = df.rename(columns=intersecting)

    log(f"🔁 Renamed {len(intersecting)} columns using anonymization mapping.", verbose)
    log(f"🔎 Preview: {list(intersecting.items())[:3]}", verbose)
    return df_renamed

def load_type_mapping(filename: str = "type_mapping.yaml", verbose: bool = False) -> dict:
    path = ROOT_PATH / "config" / filename
    with open(path, "r") as f:
        mapping = yaml.safe_load(f)
    if verbose:
        print(f"✅ Loaded type-to-index mapping: {mapping['type_to_index']}")
    return mapping["type_to_index"]