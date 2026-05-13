# utils.py
import json
from pathlib import Path
from typing import Any

from IPython.display import display
import yaml
import pandas as pd

# Resolve the repo root locally so `data/` can operate without depending on
# training-only utilities or external path helpers.
ROOT_PATH = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT_PATH / "config"
LOCAL_CONFIG_ROOT = CONFIG_ROOT / "local"


def _resolve_repo_config_path(filename: str) -> Path:
    requested = Path(filename)
    if requested.is_absolute():
        return requested

    candidates = [
        LOCAL_CONFIG_ROOT / requested.name,
        CONFIG_ROOT / requested,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _config_visibility_label(path: Path) -> str:
    try:
        path.resolve().relative_to(LOCAL_CONFIG_ROOT.resolve())
        return "local/private overlay"
    except ValueError:
        return "tracked repo copy"

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
    path = _resolve_repo_config_path(filename)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    grouped_mapping = data.get("groups", {})
    flat_mapping = {
        original: anonymized
        for group in grouped_mapping.values()
        for original, anonymized in group.items()
    }

    log(
        f"🔐 Loaded grouped column mapping from {path} "
        f"({_config_visibility_label(path)})",
        verbose,
    )
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
    path = _resolve_repo_config_path(filename)
    with open(path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    if verbose:
        print(
            "✅ Loaded type-to-index mapping from "
            f"{path} ({_config_visibility_label(path)}): "
            f"{mapping['type_to_index']}"
        )
    return mapping["type_to_index"]


def _require_nested_config_value(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]

    if current is None:
        raise ValueError(dotted_key)
    if isinstance(current, str) and not current.strip():
        raise ValueError(dotted_key)
    if isinstance(current, (list, dict)) and len(current) == 0:
        raise ValueError(dotted_key)
    return current


def load_cleaning_contract(
    filename: str = "cleaning_contract.yaml",
    verbose: bool = False,
) -> tuple[dict[str, Any], Path]:
    path = _resolve_repo_config_path(filename)
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict YAML in {path}")

    required_keys = [
        "source.raw_filename",
        "source.read_csv.sep",
        "source.read_csv.engine",
        "source.columns.type",
        "source.columns.process",
        "source.filters.allowed_types",
        "source.filters.allowed_processes",
        "source.remove_columns_after_mapping",
    ]

    missing_or_empty: list[str] = []
    for dotted_key in required_keys:
        try:
            _require_nested_config_value(payload, dotted_key)
        except (KeyError, ValueError):
            missing_or_empty.append(dotted_key)

    if missing_or_empty:
        joined = ", ".join(missing_or_empty)
        raise ValueError(
            f"Cleaning contract at {path} is missing required non-empty keys: {joined}"
        )

    log(
        f"🧩 Loaded cleaning contract from {path} "
        f"({_config_visibility_label(path)})",
        verbose,
    )
    return payload, path


def path_relative_to_root(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path(ROOT_PATH).resolve()))
    except ValueError:
        return str(path)


def dump_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
