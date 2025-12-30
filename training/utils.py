from from_root import from_root

ROOT_PATH = from_root()

import yaml
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd

def load_yaml_config(filename_or_path) -> dict:
    """
    Load a YAML config file. Supports full paths or just filenames.

    Args:
        filename_or_path (str or Path): Path or name of the config file. If only a name is provided,
                                        it is assumed to be in ROOT_PATH/config.

    Returns:
        dict: Parsed YAML config.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is not a valid YAML.
    """
    # Handle string or Path input
    path = Path(filename_or_path)

    # If only a filename is passed (no parent), assume it's inside config/
    if not path.parent or path.parent == Path("."):
        path = ROOT_PATH / "config" / path

    # Try to add .yaml if missing
    if not path.suffix:
        path = path.with_suffix(".yaml")

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path, "r",    encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {path}\n{e}")

def flowpre_log(msg, log_training: bool = True, filename_or_path: str = None, verbose: bool = True):
    if verbose:
        print(msg)
    if log_training and filename_or_path:
        with open(filename_or_path, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")

def setup_training_logs_and_dirs(
    base_name: str,
    config_filename: str,
    config: dict,
    verbose: bool,
    should_save_states: bool = False,
    log_training: bool = True,
    subdir: str = "flow_pre",  # 🆕 Default to "flow_pre" for backward compatibility
):
    base_model_dir = ROOT_PATH / "outputs" / "models" / subdir
    base_model_dir.mkdir(parents=True, exist_ok=True)

    # Determine next version
    existing_versions = [
        d for d in base_model_dir.iterdir()
        if d.is_dir() and d.name.startswith(base_name + "_v")
    ]
    if existing_versions:
        version_numbers = [
            int(d.name.split("_v")[-1]) for d in existing_versions if d.name.split("_v")[-1].isdigit()
        ]
        next_version = max(version_numbers) + 1
    else:
        next_version = 1

    versioned_name = f"{base_name}_v{next_version}"
    versioned_dir = base_model_dir / versioned_name
    versioned_dir.mkdir(parents=True)

    # ───── Log file setup ─────
    log_file_path = None
    if log_training:
        logs_dir = ROOT_PATH / "outputs" / "logs" / subdir
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = logs_dir / f"{versioned_name}.log"
        log_file_path.write_text("")  # Clear previous content

        flowpre_log(f"Logging to {log_file_path}", log_training=True, filename_or_path=log_file_path, verbose=verbose)
        flowpre_log("📋 Configuration used for training:", log_training=True, filename_or_path=log_file_path, verbose=verbose)
        config_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
        for line in config_str.splitlines():
            flowpre_log(f"   {line}", log_training=True, filename_or_path=log_file_path, verbose=verbose)

    # ───── Snapshots ─────
    snapshots_dir = None
    if should_save_states:
        snapshots_dir = versioned_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        flowpre_log(f"Saving model states to {snapshots_dir}", log_training=True, filename_or_path=log_file_path, verbose=verbose)

    return versioned_dir, versioned_name, log_file_path, snapshots_dir


def log_epoch_diagnostics(epoch: int, diagnostics: dict, log_file_path, verbose: bool = True):
    """
    Logs averaged diagnostic values for a given epoch.

    Args:
        epoch (int): The current epoch.
        diagnostics (dict): A dict of lists of diagnostic values.
        log_file_path (Path or str): File to write logs to.
        verbose (bool): Whether to also print to stdout.
    """
    flowpre_log(f"📊 Epoch {epoch + 1} diagnostics (averaged):", filename_or_path=log_file_path, verbose=verbose)
    for k, values in diagnostics.items():
        if not values:
            continue
        if "min" in k:
            value = min(values)
        elif "max" in k:
            value = max(values)
        else:
            value = sum(values) / len(values)
        flowpre_log(f"   {k}: {value:.6f}", filename_or_path=log_file_path, verbose=verbose)

def list_scaled_sets(root: Path | None = None) -> List[str]:
    """
    List available dataset folders under data/sets/scaled_sets.
    """
    from training.utils import ROOT_PATH  # avoid circular import issues elsewhere
    base = (root or Path(ROOT_PATH)) / "data" / "sets" / "scaled_sets"
    if not base.exists():
        return []
    return sorted([d.name for d in base.iterdir() if d.is_dir()])

def _find_one_csv(dirpath: Path, pattern: str) -> Path:
    """
    Find exactly ONE CSV matching `pattern` under `dirpath`.
    Raises if none or ambiguous.
    """
    cands = sorted(dirpath.glob(pattern))
    if len(cands) == 0:
        raise FileNotFoundError(f"No CSV matching '{pattern}' under {dirpath}")
    if len(cands) > 1:
        names = ", ".join(p.name for p in cands[:6])
        raise RuntimeError(f"Ambiguous CSVs for '{pattern}' under {dirpath}: {names} ...")
    return cands[0]

def _read_csv_enforce(df_path: Path, *, name: str) -> pd.DataFrame:
    """
    Read CSV with safe defaults; ensure 'post_cleaning_index' exists.
    """
    df = pd.read_csv(df_path)
    if "post_cleaning_index" not in df.columns:
        raise ValueError(f"[{name}] missing required column 'post_cleaning_index' in {df_path.name}")
    # modest dtype nudges (don’t fail if casting is impossible)
    for col in ("post_cleaning_index",):
        try:
            df[col] = df[col].astype(int)
        except Exception:
            pass
    return df

def load_scaled_sets(
    folder_name: str,
    *,
    root: Path | None = None,
    require_condition_col: Optional[str] = "type",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load X/y train/val/test from:
        ROOT_PATH/data/sets/scaled_sets/<folder_name>/{X,y}/*_{X|y}_{split}.csv

    Returns (in this exact order):
        X_train, X_val, X_test, y_train, y_val, y_test

    Notes:
    - We match files by suffix: '*X_train.csv', '*X_val.csv', '*X_test.csv'
      and the analogous for y.
    - Validates presence of 'post_cleaning_index' in all files.
    - Optionally validates the context/condition column (default 'type') in X*.
    """
    from training.utils import ROOT_PATH  # keep local to avoid import cycles
    base = (root or Path(ROOT_PATH)) / "data" / "sets" / "scaled_sets" / folder_name
    x_dir = base / "X"
    y_dir = base / "y"

    if not base.exists():
        raise FileNotFoundError(f"Dataset folder not found: {base}")
    if not x_dir.exists() or not y_dir.exists():
        raise FileNotFoundError(f"Expected subfolders 'X' and 'y' under: {base}")

    # Locate CSVs by split
    x_tr_p = _find_one_csv(x_dir, "*_X_train.csv")
    x_va_p = _find_one_csv(x_dir, "*_X_val.csv")
    x_te_p = _find_one_csv(x_dir, "*_X_test.csv")

    y_tr_p = _find_one_csv(y_dir, "*_y_train.csv")
    y_va_p = _find_one_csv(y_dir, "*_y_val.csv")
    y_te_p = _find_one_csv(y_dir, "*_y_test.csv")

    # Read
    X_train = _read_csv_enforce(x_tr_p, name="X_train")
    X_val   = _read_csv_enforce(x_va_p, name="X_val")
    X_test  = _read_csv_enforce(x_te_p, name="X_test")

    y_train = _read_csv_enforce(y_tr_p, name="y_train")
    y_val   = _read_csv_enforce(y_va_p, name="y_val")
    y_test  = _read_csv_enforce(y_te_p, name="y_test")

    # Optional schema check for the condition/context column in X*
    if require_condition_col:
        for df_name, df in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test)):
            if require_condition_col not in df.columns:
                raise ValueError(
                    f"[{df_name}] missing required condition column '{require_condition_col}'."
                    f" Present columns: {list(df.columns)[:8]}..."
                )
            # nudge dtype to int when possible
            try:
                df[require_condition_col] = df[require_condition_col].astype(int)
            except Exception:
                pass

    # Sort by index column for clean merges downstream
    for df in (X_train, X_val, X_test, y_train, y_val, y_test):
        df.sort_values("post_cleaning_index", inplace=True)
        df.reset_index(drop=True, inplace=True)

    if verbose:
        print(f"[load_scaled_sets] Loaded '{folder_name}' from {base}")
        for nm, df in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
                       ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)):
            print(f"  - {nm:7s}: shape={df.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

# training/utils.py (or same module where load_scaled_sets lives)

def load_augmented_scaled_sets(
    folder_name: str,
    *,
    root: Path | None = None,
    require_condition_col: Optional[str] = "type",
    verbose: bool = True,
):
    """
    Same API as before, but loads from augmented_scaled_sets and
    drops 'is_synth' column if present.
    """

    from training.utils import ROOT_PATH

    base = (root or Path(ROOT_PATH)) / "data" / "sets" / "augmented_scaled_sets" / f"{folder_name}"
    x_dir = base / "X"
    y_dir = base / "y"

    if not base.exists():
        raise FileNotFoundError(f"Dataset folder not found: {base}")

    # locate csvs
    x_tr_p = _find_one_csv(x_dir, "*_X_train.csv")
    x_va_p = _find_one_csv(x_dir, "*_X_val.csv")
    x_te_p = _find_one_csv(x_dir, "*_X_test.csv")

    y_tr_p = _find_one_csv(y_dir, "*_y_train.csv")
    y_va_p = _find_one_csv(y_dir, "*_y_val.csv")
    y_te_p = _find_one_csv(y_dir, "*_y_test.csv")

    # read
    X_train = _read_csv_enforce(x_tr_p, name="X_train")
    X_val = _read_csv_enforce(x_va_p, name="X_val")
    X_test = _read_csv_enforce(x_te_p, name="X_test")

    y_train = _read_csv_enforce(y_tr_p, name="y_train")
    y_val = _read_csv_enforce(y_va_p, name="y_val")
    y_test = _read_csv_enforce(y_te_p, name="y_test")

    # --------------------------------------------------
    # DROP is_synth (if exists)
    # --------------------------------------------------
    for df in (X_train, X_val, X_test, y_train, y_val, y_test):
        if "is_synth" in df.columns:
            df.drop(columns=["is_synth"], inplace=True)

    # --------------------------------------------------
    # Schema checks (same as before)
    # --------------------------------------------------
    if require_condition_col:
        for df_name, df in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test)):
            if require_condition_col not in df.columns:
                raise ValueError(f"{df_name} missing '{require_condition_col}' column")
            df[require_condition_col] = df[require_condition_col].astype(int, errors="ignore")

    # --------------------------------------------------
    # Sort for clean merges
    # --------------------------------------------------
    for df in (X_train, X_val, X_test, y_train, y_val, y_test):
        df.sort_values("post_cleaning_index", inplace=True)
        df.reset_index(drop=True, inplace=True)

    if verbose:
        print(f"[load_scaled_sets] Loaded AUGMENTED set: augmented_{folder_name}")
        for name, df in (
            ("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
            ("y_train", y_train), ("y_val", y_val), ("y_test", y_test),
        ):
            print(f"  - {name:7s}: shape={df.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def list_augmented_scaled_sets(root: Path | None = None) -> List[str]:
    """
    List available dataset folders under data/sets/scaled_sets.
    """
    from training.utils import ROOT_PATH  # avoid circular import issues elsewhere
    base = (root or Path(ROOT_PATH)) / "data" / "sets" / "augmented_scaled_sets"
    if not base.exists():
        return []
    return sorted([d.name for d in base.iterdir() if d.is_dir()])

