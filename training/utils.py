import json
import yaml
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import torch
from data.dataset_contract import DATASET_LEVEL_AXES, SUPPORTED_SPACE_STATUSES
from data.splits import DEFAULT_OFFICIAL_SPLIT_ID

try:
    from from_root import from_root as _from_root
except Exception:
    _from_root = None


def _resolve_root_path() -> Path:
    if _from_root is not None:
        try:
            return Path(_from_root())
        except Exception:
            pass
    return Path(__file__).resolve().parents[1]


ROOT_PATH = _resolve_root_path()

CANONICAL_DERIVED_MANIFEST_FIELDS = {
    "contract_id",
    "dataset_name",
    "dataset_role",
    "split_id",
    "cleaning_policy_id",
    "source_dataset_manifest",
    "dataset_level_axes",
    "supported_space_status",
    "counts_by_split",
    "counts_by_class",
}
LEGACY_POLICY_STATUSES = {"legacy", "legacy_policy"}


def _canonical_scaled_root(root: Path | None = None) -> Path:
    base_root = Path(root or ROOT_PATH)
    return base_root / "data" / "sets" / "official" / DEFAULT_OFFICIAL_SPLIT_ID / "scaled"


def _canonical_augmented_scaled_root(root: Path | None = None) -> Path:
    base_root = Path(root or ROOT_PATH)
    return base_root / "data" / "sets" / "official" / DEFAULT_OFFICIAL_SPLIT_ID / "augmented_scaled"


def _legacy_scaled_root(root: Path | None = None) -> Path:
    base_root = Path(root or ROOT_PATH)
    return base_root / "data" / "sets" / "scaled_sets"


def _legacy_augmented_scaled_root(root: Path | None = None) -> Path:
    base_root = Path(root or ROOT_PATH)
    return base_root / "data" / "sets" / "augmented_scaled_sets"


def resolve_scaled_set_base(
    folder_name: str,
    *,
    root: Path | None = None,
    allow_legacy: bool = False,
) -> tuple[Path, str]:
    canonical_base = _canonical_scaled_root(root) / folder_name
    legacy_base = _legacy_scaled_root(root) / folder_name

    if canonical_base.exists():
        return canonical_base, "canonical"
    if legacy_base.exists():
        if allow_legacy:
            return legacy_base, "legacy"
        raise FileNotFoundError(
            f"Scaled dataset '{folder_name}' exists only under the legacy namespace: {legacy_base}. "
            "Canonical mode reads exclusively from data/sets/official/<split_id>/scaled/. "
            "Pass allow_legacy=True only for historical workflows."
        )
    raise FileNotFoundError(
        f"Scaled dataset '{folder_name}' not found under canonical namespace: {canonical_base}"
    )


def resolve_augmented_scaled_set_base(
    folder_name: str,
    *,
    root: Path | None = None,
    allow_legacy: bool = False,
) -> tuple[Path, str]:
    canonical_base = _canonical_augmented_scaled_root(root) / folder_name
    legacy_base = _legacy_augmented_scaled_root(root) / folder_name

    if canonical_base.exists():
        return canonical_base, "canonical"
    if legacy_base.exists():
        if allow_legacy:
            return legacy_base, "legacy"
        raise FileNotFoundError(
            f"Augmented scaled dataset '{folder_name}' exists only under the legacy namespace: {legacy_base}. "
            "Canonical mode reads exclusively from data/sets/official/<split_id>/augmented_scaled/. "
            "Pass allow_legacy=True only for historical workflows."
        )
    raise FileNotFoundError(
        f"Augmented scaled dataset '{folder_name}' not found under canonical namespace: {canonical_base}"
    )

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


def select_training_device(prefer: str | torch.device | None = None) -> torch.device:
    """
    Resolve device with best-effort preference:
    explicit request first, otherwise MPS → CUDA → CPU.
    """
    if isinstance(prefer, torch.device):
        return prefer

    def has_mps() -> bool:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    def has_cuda() -> bool:
        return bool(torch.cuda.is_available())

    if isinstance(prefer, str):
        requested = prefer.lower().strip()
        if requested == "auto":
            prefer = None
        elif requested == "mps" and has_mps():
            return torch.device("mps")
        elif requested == "cuda" and has_cuda():
            return torch.device("cuda")
        elif requested == "cpu":
            return torch.device("cpu")

    if has_mps():
        return torch.device("mps")
    if has_cuda():
        return torch.device("cuda")
    return torch.device("cpu")

def setup_training_logs_and_dirs(
    base_name: str,
    config_filename: str,
    config: dict,
    verbose: bool,
    should_save_states: bool = False,
    log_training: bool = True,
    subdir: str = "flow_pre",  # 🆕 Default to "flow_pre" for backward compatibility
    namespace: str | None = None,
    relative_subdir: str | None = None,
    fixed_run_id: str | None = None,
    log_in_run_dir: bool = False,
):
    base_model_dir = ROOT_PATH / "outputs" / "models"
    base_logs_dir = ROOT_PATH / "outputs" / "logs"
    if namespace:
        base_model_dir = base_model_dir / namespace / subdir
        base_logs_dir = base_logs_dir / namespace / subdir
    else:
        base_model_dir = base_model_dir / subdir
        base_logs_dir = base_logs_dir / subdir
    if relative_subdir:
        base_model_dir = base_model_dir / relative_subdir
        base_logs_dir = base_logs_dir / relative_subdir
    base_model_dir.mkdir(parents=True, exist_ok=True)

    if fixed_run_id is not None:
        versioned_name = str(fixed_run_id)
        versioned_dir = base_model_dir / versioned_name
        if versioned_dir.exists():
            raise FileExistsError(f"Run directory already exists: {versioned_dir}")
    else:
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
        if log_in_run_dir:
            log_file_path = versioned_dir / f"{versioned_name}.log"
        else:
            base_logs_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = base_logs_dir / f"{versioned_name}.log"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
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

def _dataset_manifest_path(base: Path) -> Path:
    return base / "meta" / "manifest.json"


def _load_dataset_manifest(base: Path) -> Optional[dict]:
    manifest_path = _dataset_manifest_path(base)
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest if isinstance(manifest, dict) else None


def load_dataset_manifest(base: Path | str) -> Optional[dict]:
    return _load_dataset_manifest(Path(base))


def _canonical_manifest_error(base: Path) -> Optional[str]:
    manifest = _load_dataset_manifest(base)
    if manifest is None:
        return "missing manifest"

    missing = sorted(CANONICAL_DERIVED_MANIFEST_FIELDS.difference(manifest.keys()))
    if missing:
        return f"manifest missing canonical fields: {missing}"

    axes = manifest.get("dataset_level_axes")
    if not isinstance(axes, dict):
        return "manifest missing dataset_level_axes dict"

    missing_axes = [axis for axis in DATASET_LEVEL_AXES if axis not in axes]
    if missing_axes:
        return f"manifest missing dataset-level axes: {missing_axes}"

    support_status = manifest.get("supported_space_status")
    if support_status not in SUPPORTED_SPACE_STATUSES:
        return f"manifest has invalid supported_space_status: {support_status!r}"
    if support_status != "materialized_now":
        return f"manifest not yet canonical-materialized: supported_space_status={support_status!r}"

    if manifest.get("policy_status") in LEGACY_POLICY_STATUSES:
        return "manifest explicitly marked as legacy"
    return None


def _ensure_canonical_dataset(base: Path, *, allow_legacy: bool, dataset_kind: str) -> None:
    error = _canonical_manifest_error(base)
    if error is None or allow_legacy:
        return
    raise ValueError(
        f"{dataset_kind} dataset '{base.name}' is legacy and not available in canonical mode: {error}. "
        "Pass allow_legacy=True only for historical workflows."
    )


def list_scaled_sets(root: Path | None = None, *, allow_legacy: bool = False) -> List[str]:
    """
    List available dataset folders under the canonical official scaled namespace.
    """
    canonical_base = _canonical_scaled_root(root)
    canonical_names = []
    if canonical_base.exists():
        canonical_names = sorted(
            [d.name for d in canonical_base.iterdir() if d.is_dir() and _canonical_manifest_error(d) is None]
        )
    if not allow_legacy:
        return canonical_names

    legacy_base = _legacy_scaled_root(root)
    legacy_names = []
    if legacy_base.exists():
        legacy_names = sorted([d.name for d in legacy_base.iterdir() if d.is_dir()])
    return sorted(set(canonical_names).union(legacy_names))

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
    allow_legacy: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load X/y train/val/test from:
        ROOT_PATH/data/sets/official/<split_id>/scaled/<folder_name>/{X,y}/*_{X|y}_{split}.csv

    Returns (in this exact order):
        X_train, X_val, X_test, y_train, y_val, y_test

    Notes:
    - We match files by suffix: '*X_train.csv', '*X_val.csv', '*X_test.csv'
      and the analogous for y.
    - Validates presence of 'post_cleaning_index' in all files.
    - Optionally validates the context/condition column (default 'type') in X*.
    - In allow_legacy=True mode, folders that only exist under data/sets/scaled_sets/
      remain loadable for historical workflows, but canonical mode never falls back.
    """
    base, resolved_mode = resolve_scaled_set_base(
        folder_name,
        root=root,
        allow_legacy=allow_legacy,
    )
    x_dir = base / "X"
    y_dir = base / "y"

    _ensure_canonical_dataset(base, allow_legacy=allow_legacy, dataset_kind="scaled")
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
        mode = resolved_mode
        print(f"[load_scaled_sets] Loaded '{folder_name}' from {base} ({mode})")
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
    allow_legacy: bool = False,
    preserve_is_synth: bool = True,
    verbose: bool = True,
):
    """
    Same API as before, but prefers the canonical official augmented namespace.
    In canonical mode we preserve 'is_synth' by default because it is part of
    synthetic dataset provenance and training-time handling.
    """

    base, resolved_mode = resolve_augmented_scaled_set_base(
        folder_name,
        root=root,
        allow_legacy=allow_legacy,
    )
    x_dir = base / "X"
    y_dir = base / "y"

    if not base.exists():
        raise FileNotFoundError(f"Dataset folder not found: {base}")
    _ensure_canonical_dataset(base, allow_legacy=allow_legacy, dataset_kind="augmented")

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

    if not preserve_is_synth:
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
        mode = resolved_mode
        print(f"[load_scaled_sets] Loaded AUGMENTED set: augmented_{folder_name} ({mode})")
        for name, df in (
            ("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
            ("y_train", y_train), ("y_val", y_val), ("y_test", y_test),
        ):
            print(f"  - {name:7s}: shape={df.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def list_augmented_scaled_sets(root: Path | None = None, *, allow_legacy: bool = False) -> List[str]:
    """
    List available dataset folders under the canonical augmented namespace.
    """
    canonical_base = _canonical_augmented_scaled_root(root)
    canonical_names = []
    if canonical_base.exists():
        canonical_names = sorted(
            [d.name for d in canonical_base.iterdir() if d.is_dir() and _canonical_manifest_error(d) is None]
        )
    if not allow_legacy:
        return canonical_names

    legacy_base = _legacy_augmented_scaled_root(root)
    legacy_names = []
    if legacy_base.exists():
        legacy_names = sorted([d.name for d in legacy_base.iterdir() if d.is_dir()])
    return sorted(set(canonical_names).union(legacy_names))

