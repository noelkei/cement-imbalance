from data.utils import ROOT_PATH, dump_json, log, load_column_mapping_by_group, load_type_mapping, path_relative_to_root
from data.cleaning import (
    DEFAULT_LEGACY_CLEANING_POLICY_ID,
    DEFAULT_OFFICIAL_CLEANING_POLICY_ID,
    load_official_modeling_source_frame,
)
from data.dataset_contract import (
    DEFAULT_DATASET_CONTRACT_VERSION,
    build_canonical_dataset_name,
    build_canonical_derived_manifest,
    classify_supported_dataset_space,
    counts_from_source_manifest,
)
from data.kmeans_smote_joint import (
    augment_canonical_bundle_with_kmeans_smote,
    load_kmeans_smote_joint_config,
)
from data.splits import DEFAULT_OFFICIAL_SPLIT_ID, prepare_splits
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler

from pathlib import Path
import json
import shutil
from typing import Any, Dict, List, Tuple, Optional, Mapping

_OFFICIAL_SETS_ROOT = Path(ROOT_PATH) / "data" / "sets" / "official"
_OFFICIAL_OUTPUTS_ROOT = Path(ROOT_PATH) / "outputs" / "models" / "official"
_SCALED_ROOT = Path(ROOT_PATH) / "data" / "sets" / "scaled_sets"
_LEGACY_FLOWPRE_MODELS_DIR = Path(ROOT_PATH) / "outputs" / "models" / "flow_pre"
_OFFICIAL_FLOWPRE_MODELS_DIR = _OFFICIAL_OUTPUTS_ROOT / "flow_pre"
_LEGACY_FLOWGEN_MODELS_DIR = Path(ROOT_PATH) / "outputs" / "models" / "flowgen"
_OFFICIAL_FLOWGEN_MODELS_DIR = _OFFICIAL_OUTPUTS_ROOT / "flowgen"
DEFAULT_LEGACY_DATASET_NAME = "df_input"
DEFAULT_OFFICIAL_DATASET_NAME = "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"
FLOWGEN_WORK_BASE_IDS = {"candidate_1", "candidate_2"}


def _official_raw_bundle_dir(
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    dataset_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
) -> Path:
    return _OFFICIAL_SETS_ROOT / split_id / "raw" / dataset_name


def _official_scaled_root(
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
) -> Path:
    return _OFFICIAL_SETS_ROOT / split_id / "scaled"


def _official_augmented_scaled_root(
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
) -> Path:
    return _OFFICIAL_SETS_ROOT / split_id / "augmented_scaled"


def _official_scaled_bundle_dir(
    storage_name: str,
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
) -> Path:
    return _official_scaled_root(split_id=split_id) / storage_name


def _official_augmented_bundle_dir(
    storage_name: str,
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
) -> Path:
    return _official_augmented_scaled_root(split_id=split_id) / storage_name


def official_raw_bundle_manifest_path(
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    dataset_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
) -> Path:
    return _official_raw_bundle_dir(split_id=split_id, dataset_name=dataset_name) / "manifest.json"


def _load_json_dict(path: str | Path | None) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    with open(candidate, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def _model_search_roots(model_family: str) -> list[Path]:
    family = str(model_family).lower()
    if family == "flow_pre":
        return [_OFFICIAL_FLOWPRE_MODELS_DIR, _LEGACY_FLOWPRE_MODELS_DIR]
    if family == "flowgen":
        return [_OFFICIAL_FLOWGEN_MODELS_DIR, _LEGACY_FLOWGEN_MODELS_DIR]
    raise ValueError(f"Unsupported model family: {model_family}")


def _locate_model_run_dir(model_name: str, model_family: str = "flow_pre") -> Path:
    for root in _model_search_roots(model_family):
        candidate = root / model_name
        if candidate.exists():
            return candidate
    family = str(model_family).lower()
    raise FileNotFoundError(f"Model folder not found for family '{family}': {model_name}")


def _bundle_filepaths(base_dir: Path, dataset_name: str) -> dict[str, dict[str, Path]]:
    return {
        split: {
            "X": base_dir / "X" / f"{dataset_name}_X_{split}.csv",
            "y": base_dir / "y" / f"{dataset_name}_y_{split}.csv",
            "r": base_dir / "removed" / f"{dataset_name}_r_{split}.csv",
        }
        for split in ("train", "val", "test")
    }


def build_raw_modeling_frames(
    df_cleaned: pd.DataFrame,
    target: str = "init",
    condition_col: str = "type",
    assign_post_cleaning_index: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Canonical projection from cleaned modeling rows into X/y/removed while
    freezing the legacy X/y feature surface exactly.
    """
    df_cleaned = df_cleaned.copy()
    if assign_post_cleaning_index or "post_cleaning_index" not in df_cleaned.columns:
        df_cleaned = df_cleaned.reset_index(drop=True)
        df_cleaned.insert(0, "post_cleaning_index", np.arange(len(df_cleaned), dtype=int))

    initial_cols = df_cleaned.columns.tolist()
    log(f"\n📊 Initial shape: {df_cleaned.shape}", verbose)
    log(f"📑 Initial columns ({len(initial_cols)}):", verbose)
    log(initial_cols, verbose)

    type_to_index = load_type_mapping(verbose=verbose)
    df_cleaned[condition_col] = df_cleaned[condition_col].map(type_to_index)
    if df_cleaned[condition_col].isna().any():
        raise ValueError("🚨 Some 'type' values couldn't be mapped. Check your YAML mapping.")
    df_cleaned[condition_col] = df_cleaned[condition_col].astype(int)

    target_col = target
    provenance_cols = [
        "split_id",
        "split",
        "split_row_id",
        "source_row_number",
        "kept_for_model",
        "drop_for_model_flag",
    ]
    cols_to_remove = [
        "date",
        "process",
        "uni_removal_flag",
        "Outlier_IForest",
        "overlap_flag",
        "or_removal_flag",
        *[col for col in df_cleaned.columns if col.startswith("chem_")],
        *[col for col in df_cleaned.columns if col.startswith("phase_")],
        "sum",
        "sum_chem",
        "sum_phase",
        *[col for col in df_cleaned.columns if col.startswith("norm_")],
    ]

    removed = set(cols_to_remove + [target_col] + provenance_cols)
    excluded = removed.union({"post_cleaning_index"})
    flowpre_cols = [col for col in df_cleaned.columns if col not in excluded and col != condition_col]
    ordered_removed_cols = [col for col in df_cleaned.columns if col in removed and col not in provenance_cols]
    ordered_removed_cols.extend([col for col in provenance_cols if col in df_cleaned.columns])

    df_X_input = df_cleaned[["post_cleaning_index", condition_col] + flowpre_cols].copy()
    df_y_input = df_cleaned[["post_cleaning_index", target_col]].copy()
    df_removed_input = df_cleaned[["post_cleaning_index"] + ordered_removed_cols].copy()

    rename_map = {
        "90": "90um_mesh",
        "902": "90um",
        "753": "75um",
        "454": "45um",
        "305": "30um",
    }

    for df in [df_X_input, df_y_input, df_removed_input]:
        rename_cols = {old: new for old, new in rename_map.items() if old in df.columns}
        df.rename(columns=rename_cols, inplace=True)

    for df in [df_X_input, df_y_input, df_removed_input]:
        for col in ["blaine", "blaines"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") / 100
        if "init" in df.columns:
            df["init"] = pd.to_numeric(df["init"], errors="coerce") / 60

    log(f"\n✅ FlowPre input columns ({len(flowpre_cols)}):", verbose)
    log(flowpre_cols, verbose)
    log("\n🔹 Final shapes:", verbose)
    log(f"FlowPre: {df_X_input.shape}", verbose)
    log(f"Target: {df_y_input.shape}", verbose)
    log(f"Removed: {df_removed_input.shape}", verbose)

    return df_X_input, df_y_input, df_removed_input


def materialize_official_raw_bundle(
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    dataset_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
    target: str = "init",
    condition_col: str = "type",
    cleaning_policy_id: str = DEFAULT_OFFICIAL_CLEANING_POLICY_ID,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    base_dir = _official_raw_bundle_dir(split_id=split_id, dataset_name=dataset_name)
    manifest_path = base_dir / "manifest.json"
    filepaths = _bundle_filepaths(base_dir, dataset_name)

    if manifest_path.exists() and all(
        path.exists()
        for split_paths in filepaths.values()
        for path in split_paths.values()
    ) and not force:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    source_manifest, df_modeling = load_official_modeling_source_frame(
        split_id=split_id,
        target=target,
        cleaning_policy_id=cleaning_policy_id,
        force=force,
        verbose=verbose,
    )
    df_modeling = df_modeling.copy()
    df_modeling["split_id"] = split_id
    df_modeling = df_modeling.sort_values(["date", "source_row_number"]).reset_index(drop=True)

    class_counts_by_split = {
        split_name: {
            str(type_name): int(count)
            for type_name, count in (
                df_modeling.loc[df_modeling["split"] == split_name, "type"]
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            )
        }
        for split_name in ("train", "val", "test")
    }

    df_X, df_y, df_r = build_raw_modeling_frames(
        df_modeling,
        target=target,
        condition_col=condition_col,
        assign_post_cleaning_index=True,
        verbose=verbose,
    )

    if "split" not in df_r.columns:
        raise ValueError("Removed bundle is missing split provenance required for official partitioning.")

    for dir_name in ("X", "y", "removed"):
        (base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    summary_by_split = {}
    for split_name in ("train", "val", "test"):
        split_indices = df_r.loc[df_r["split"] == split_name, "post_cleaning_index"]
        split_index_set = set(split_indices.tolist())

        X_split = df_X[df_X["post_cleaning_index"].isin(split_index_set)].copy()
        y_split = df_y[df_y["post_cleaning_index"].isin(split_index_set)].copy()
        r_split = df_r[df_r["post_cleaning_index"].isin(split_index_set)].copy()

        X_split = X_split.sort_values("post_cleaning_index").reset_index(drop=True)
        y_split = y_split.sort_values("post_cleaning_index").reset_index(drop=True)
        r_split = r_split.sort_values("post_cleaning_index").reset_index(drop=True)

        X_split.to_csv(filepaths[split_name]["X"], index=False)
        y_split.to_csv(filepaths[split_name]["y"], index=False)
        r_split.to_csv(filepaths[split_name]["r"], index=False)

        summary_by_split[split_name] = {
            "X_rows": int(len(X_split)),
            "y_rows": int(len(y_split)),
            "removed_rows": int(len(r_split)),
            "class_counts": class_counts_by_split.get(split_name, {}),
        }

    policy_status = (
        "legacy_policy"
        if cleaning_policy_id == DEFAULT_LEGACY_CLEANING_POLICY_ID or dataset_name == DEFAULT_LEGACY_DATASET_NAME
        else "canonical"
    )
    manifest = {
        "split_id": split_id,
        "dataset_name": dataset_name,
        "target": target,
        "status": "official",
        "policy_status": policy_status,
        "cleaning_policy_id": cleaning_policy_id,
        "source_cleaning_manifest": source_manifest["source_cleaning_manifest"],
        "source_cleaning_policy_id": source_manifest["cleaning_policy_id"],
        "source_split_manifest": source_manifest["source_split_manifest"],
        "source_file": source_manifest["source_file"],
        "source_file_sha256": source_manifest["source_file_sha256"],
        "feature_surface_freeze": {
            "X_columns": df_X.columns.tolist(),
            "y_columns": df_y.columns.tolist(),
            "condition_col": condition_col,
            "index_col": "post_cleaning_index",
        },
        "artifacts": {
            "X": {split: path_relative_to_root(paths["X"]) for split, paths in filepaths.items()},
            "y": {split: path_relative_to_root(paths["y"]) for split, paths in filepaths.items()},
            "removed": {split: path_relative_to_root(paths["r"]) for split, paths in filepaths.items()},
        },
        "summary_by_split": summary_by_split,
    }
    dump_json(manifest, manifest_path)

    if verbose:
        log(f"✅ Official raw bundle materialized at: {base_dir}", verbose)
    return manifest


def load_official_raw_bundle(
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    dataset_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
    target: str = "init",
    condition_col: str = "type",
    cleaning_policy_id: str = DEFAULT_OFFICIAL_CLEANING_POLICY_ID,
    force: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_dir = _official_raw_bundle_dir(split_id=split_id, dataset_name=dataset_name)
    manifest_path = base_dir / "manifest.json"
    filepaths = _bundle_filepaths(base_dir, dataset_name)

    if force or not manifest_path.exists() or not all(
        path.exists()
        for split_paths in filepaths.values()
        for path in split_paths.values()
    ):
        materialize_official_raw_bundle(
            split_id=split_id,
            dataset_name=dataset_name,
            target=target,
            condition_col=condition_col,
            cleaning_policy_id=cleaning_policy_id,
            force=force,
            verbose=verbose,
        )

    return (
        pd.read_csv(filepaths["train"]["X"]),
        pd.read_csv(filepaths["val"]["X"]),
        pd.read_csv(filepaths["test"]["X"]),
        pd.read_csv(filepaths["train"]["y"]),
        pd.read_csv(filepaths["val"]["y"]),
        pd.read_csv(filepaths["test"]["y"]),
        pd.read_csv(filepaths["train"]["r"]),
        pd.read_csv(filepaths["val"]["r"]),
        pd.read_csv(filepaths["test"]["r"]),
    )


def load_or_create_raw_splits_legacy(
    df_name: str = "df_input",
    condition_col: str = "type",
    val_size: int = 150,
    test_size: int = 100,
    target: str = "init",
    verbose: bool = True,
    force: bool = False
):
    """
    Legacy random/stratified split helper preserved for historical consumers.

    Args:
        df_name (str): Folder name under data/sets/
        condition_col (str): Column used for stratified splitting
        val_size (int): Max val samples per class
        test_size (int): Max test samples per class
        target (str): Target passed to prepare_splits
        verbose (bool): Log steps
        force (bool): Recreate sets even if they exist

    Returns:
        9 DataFrames: X_train, X_val, X_test, y_train, y_val, y_test, r_train, r_val, r_test
    """

    sets_base = ROOT_PATH / "data" / "sets" / df_name
    sets_X_dir = sets_base / "X"
    sets_y_dir = sets_base / "y"
    sets_r_dir = sets_base / "removed"

    sets_X_dir.mkdir(parents=True, exist_ok=True)
    sets_y_dir.mkdir(parents=True, exist_ok=True)
    sets_r_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        "train": {
            "X": f"{df_name}_X_train.csv",
            "y": f"{df_name}_y_train.csv",
            "r": f"{df_name}_r_train.csv"
        },
        "val": {
            "X": f"{df_name}_X_val.csv",
            "y": f"{df_name}_y_val.csv",
            "r": f"{df_name}_r_val.csv"
        },
        "test": {
            "X": f"{df_name}_X_test.csv",
            "y": f"{df_name}_y_test.csv",
            "r": f"{df_name}_r_test.csv"
        },
    }

    filepaths = {
        split: {
            "X": sets_X_dir / filenames[split]["X"],
            "y": sets_y_dir / filenames[split]["y"],
            "r": sets_r_dir / filenames[split]["r"]
        } for split in ["train", "val", "test"]
    }

    if all(p["X"].exists() and p["y"].exists() and p["r"].exists() for p in filepaths.values()) and not force:
        log(f"📁 All sets for '{df_name}' already exist. Loading from disk...", verbose)
        X_train = pd.read_csv(filepaths["train"]["X"])
        X_val   = pd.read_csv(filepaths["val"]["X"])
        X_test  = pd.read_csv(filepaths["test"]["X"])
        y_train = pd.read_csv(filepaths["train"]["y"])
        y_val   = pd.read_csv(filepaths["val"]["y"])
        y_test  = pd.read_csv(filepaths["test"]["y"])
        r_train = pd.read_csv(filepaths["train"]["r"])
        r_val   = pd.read_csv(filepaths["val"]["r"])
        r_test  = pd.read_csv(filepaths["test"]["r"])
        return X_train, X_val, X_test, y_train, y_val, y_test, r_train, r_val, r_test

    log(f"🔄 Creating sets for {df_name} (val={val_size}, test={test_size})...", verbose)

    df_X, df_y, df_r = prepare_splits(target=target, verbose=verbose, force=force)

    df_X = df_X.sort_values("post_cleaning_index").reset_index(drop=True)
    df_y = df_y.sort_values("post_cleaning_index").reset_index(drop=True)
    df_r = df_r.sort_values("post_cleaning_index").reset_index(drop=True)

    used_indices = set()
    val_indices = []
    test_indices = []

    for cond in df_X[condition_col].unique():
        cond_indices = df_X[df_X[condition_col] == cond]["post_cleaning_index"].tolist()
        cond_indices = [i for i in cond_indices if i not in used_indices]

        n_val = min(val_size, len(cond_indices))
        n_test = min(test_size, len(cond_indices) - n_val)

        selected = shuffle(cond_indices, random_state=42)
        val_indices.extend(selected[:n_val])
        test_indices.extend(selected[n_val:n_val + n_test])
        used_indices.update(selected[:n_val + n_test])

    val_mask = df_X["post_cleaning_index"].isin(val_indices)
    test_mask = df_X["post_cleaning_index"].isin(test_indices)
    train_mask = ~(val_mask | test_mask)

    sets_X = {
        "train": df_X[train_mask],
        "val": df_X[val_mask],
        "test": df_X[test_mask]
    }

    sets_y = {
        "train": df_y[df_y["post_cleaning_index"].isin(sets_X["train"]["post_cleaning_index"])],
        "val": df_y[df_y["post_cleaning_index"].isin(sets_X["val"]["post_cleaning_index"])],
        "test": df_y[df_y["post_cleaning_index"].isin(sets_X["test"]["post_cleaning_index"])]
    }

    sets_r = {
        "train": df_r[df_r["post_cleaning_index"].isin(sets_X["train"]["post_cleaning_index"])],
        "val": df_r[df_r["post_cleaning_index"].isin(sets_X["val"]["post_cleaning_index"])],
        "test": df_r[df_r["post_cleaning_index"].isin(sets_X["test"]["post_cleaning_index"])]
    }

    # Sort and save
    for split in ["train", "val", "test"]:
        for d in [sets_X, sets_y, sets_r]:
            d[split] = d[split].sort_values("post_cleaning_index").reset_index(drop=True)

        sets_X[split].to_csv(filepaths[split]["X"], index=False)
        sets_y[split].to_csv(filepaths[split]["y"], index=False)
        sets_r[split].to_csv(filepaths[split]["r"], index=False)

        log(f"✅ Saved {split} set → X: {sets_X[split].shape}, y: {sets_y[split].shape}, removed: {sets_r[split].shape}", verbose)

    return (
        sets_X["train"], sets_X["val"], sets_X["test"],
        sets_y["train"], sets_y["val"], sets_y["test"],
        sets_r["train"], sets_r["val"], sets_r["test"]
    )


def load_or_create_raw_splits(
    df_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
    condition_col: str = "type",
    val_size: int = 150,
    test_size: int = 100,
    target: str = "init",
    verbose: bool = True,
    force: bool = False,
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    split_mode: str = "official",
):
    """
    Canonical raw bundle loader for downstream training code.

    Public contract preserved:
      returns exactly
      X_train, X_val, X_test, y_train, y_val, y_test, r_train, r_val, r_test

    `split_mode="official"` is the F3 canonical path.
    `split_mode="legacy"` preserves the historical random-split behavior.
    """
    if split_mode == "legacy":
        return load_or_create_raw_splits_legacy(
            df_name=df_name,
            condition_col=condition_col,
            val_size=val_size,
            test_size=test_size,
            target=target,
            verbose=verbose,
            force=force,
        )

    if (val_size, test_size) != (150, 100) and verbose:
        log(
            "ℹ️ Official raw bundle ignores legacy val/test size knobs because the official split is fixed upstream.",
            verbose,
        )

    return load_official_raw_bundle(
        split_id=split_id,
        dataset_name=df_name,
        target=target,
        condition_col=condition_col,
        cleaning_policy_id=DEFAULT_OFFICIAL_CLEANING_POLICY_ID,
        force=force,
        verbose=verbose,
    )

def plot_umap_and_histograms_per_set(X_train, X_val, X_test, y_train, y_val, y_test, n_neighbors=15, min_dist=0.1, random_state=42):
    import umap

    def clean_input(X):
        return X.drop(columns=["post_cleaning_index", "type"], errors="ignore")

    # Step 1: Fit UMAP on all combined sets (same projection space)
    X_all = pd.concat([clean_input(X_train), clean_input(X_val), clean_input(X_test)], axis=0)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    reducer.fit(X_all)

    # Step 2: Transform each set
    sets = {
        "Train": (X_train, reducer.transform(clean_input(X_train))),
        "Validation": (X_val, reducer.transform(clean_input(X_val))),
        "Test": (X_test, reducer.transform(clean_input(X_test)))
    }

    # Step 3: Plot UMAP per set
    for name, (X_orig, embedding) in sets.items():
        plt.figure(figsize=(7, 5))
        scatter = plt.scatter(
            embedding[:, 0], embedding[:, 1],
            c=X_orig["type"], cmap="tab10", s=20, alpha=0.7
        )
        plt.colorbar(scatter, label="Type")
        plt.title(f"UMAP - {name} Set")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Step 4: Plot histograms of 'init' target column
    plt.figure(figsize=(10, 4))
    sns.histplot(y_train["init"], kde=False, bins=50, label="Train", color="blue", stat="density", alpha=0.5)
    sns.histplot(y_val["init"], kde=False, bins=50, label="Val", color="green", stat="density", alpha=0.5)
    sns.histplot(y_test["init"], kde=False, bins=50, label="Test", color="red", stat="density", alpha=0.5)
    plt.title("Histogram of Target Column: 'init'")
    plt.legend()
    plt.grid(True)
    plt.xticks([])
    plt.tight_layout()
    plt.show()

def show_type_distribution(X_train, X_val, X_test):
    print("🔹 Number of samples per 'type' in each set:\n")

    print("📦 Train set:")
    print(X_train["type"].value_counts().sort_index(), "\n")

    print("📦 Validation set:")
    print(X_val["type"].value_counts().sort_index(), "\n")

    print("📦 Test set:")
    print(X_test["type"].value_counts().sort_index(), "\n")

    # Optionally return the counts if you want to reuse them
    return (
        X_train["type"].value_counts(),
        X_val["type"].value_counts(),
        X_test["type"].value_counts()
    )

def check_index_alignment(X_train, y_train, removed_train,
                          X_val, y_val, removed_val,
                          X_test, y_test, removed_test):
    def check_set(name, x_df, y_df, r_df):
        x_idx = x_df["post_cleaning_index"]
        y_idx = y_df["post_cleaning_index"]
        r_idx = r_df["post_cleaning_index"]

        x_match_y = x_idx.equals(y_idx)
        x_match_r = x_idx.equals(r_idx)

        print(f"\n🔎 Checking set: {name}")
        print(f"   📐 X vs y index match: {'✅' if x_match_y else '❌'}")
        print(f"   📐 X vs removed index match: {'✅' if x_match_r else '❌'}")

        if not x_match_y:
            mismatches = x_idx.compare(y_idx)
            print("   ❌ X vs y mismatched indices:\n", mismatches.head())
        if not x_match_r:
            mismatches = x_idx.compare(r_idx)
            print("   ❌ X vs removed mismatched indices:\n", mismatches.head())

    check_set("Train", X_train, y_train, removed_train)
    check_set("Validation", X_val, y_val, removed_val)
    check_set("Test", X_test, y_test, removed_test)

def compute_rrmse_mean_std(df1: pd.DataFrame, df2: pd.DataFrame, exclude_cols: list = ["post_cleaning_index", "type", "datum", "date", "process"], scaled: bool = False) -> dict:
    """
    Compute the RRMSE between the means and standard deviations of matching columns from two DataFrames.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        exclude_cols (list): Columns to exclude from comparison.

    Returns:
        dict: {
            "mean_rrmse": float,
            "std_rrmse": float
        }
    """
    common_cols = [col for col in df1.columns if col in df2.columns and col not in exclude_cols]

    df1_means = df1[common_cols].mean()
    df2_means = df2[common_cols].mean()
    df1_stds = df1[common_cols].std()
    df2_stds = df2[common_cols].std()

    # Mean RRMSE
    mean_rmse = np.sqrt(np.mean((df1_means - df2_means) ** 2))
    mean_denom = np.mean(np.abs(df1_means)) if not scaled else 1
    mean_rrmse = mean_rmse / mean_denom if mean_denom != 0 else np.nan

    # Std RRMSE
    std_rmse = np.sqrt(np.mean((df1_stds - df2_stds) ** 2))
    std_denom = np.mean(np.abs(df1_stds))
    std_rrmse = std_rmse / std_denom if std_denom != 0 else np.nan

    return {
        "mean_rrmse": mean_rrmse,
        "std_rrmse": std_rrmse
    }

def plot_scaled_histograms(X, bins=100, exclude_cols=["post_cleaning_index", "type"]):
    features = [col for col in X.columns if col not in exclude_cols]

    # Fit scalers on the entire dataset
    scaler_standard = StandardScaler()
    scaler_robust = RobustScaler()

    X_standard = pd.DataFrame(scaler_standard.fit_transform(X[features]), columns=features)
    X_robust = pd.DataFrame(scaler_robust.fit_transform(X[features]), columns=features)

    for feature in features:
        plt.figure(figsize=(15, 4))

        # Original
        plt.subplot(1, 3, 1)
        plt.hist(X[feature], bins=bins, color="skyblue", edgecolor="black", alpha=0.7, density=True)
        plt.title(f"Original: {feature}")
        plt.grid(True)
        plt.xticks([])

        # Standard scaled
        plt.subplot(1, 3, 2)
        plt.hist(X_standard[feature], bins=bins, color="orange", edgecolor="black", alpha=0.7, density=True)
        plt.title("Standard Scaler")
        plt.grid(True)

        # Robust scaled
        plt.subplot(1, 3, 3)
        plt.hist(X_robust[feature], bins=bins, color="green", edgecolor="black", alpha=0.7, density=True)
        plt.title("Robust Scaler")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

def get_scaler(name: str):
    name = name.lower()
    if name == "standard":
        return StandardScaler()
    elif name == "robust":
        return RobustScaler()
    elif name == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=42)
    elif name == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(f"❌ Unsupported scaler: {name}. Choose from 'standard', 'robust', 'quantile', 'minmax'.")

def load_or_create_scaled_sets(
    raw_df_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
    scaled_df_name: str = "df_scaled",
    condition_col: str = "type",
    val_size: int = 150,
    test_size: int = 100,
    target: str = "init",
    force: bool = False,
    verbose: bool = True,
    x_scaler_type: str = "standard",
    y_scaler_type: str = "standard",
    exclude_cols: list = None,
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    synthetic_policy: str = "none",
    manifest_version: str = DEFAULT_DATASET_CONTRACT_VERSION,
):
    if exclude_cols is None:
        exclude_cols = ["post_cleaning_index", "type"]

    x_scaler_type = x_scaler_type.lower()
    y_scaler_type = y_scaler_type.lower()
    synthetic_policy = str(synthetic_policy).lower()

    allowed_x_scalers = ["standard", "robust", "minmax", "quantile"]
    allowed_y_scalers = list(allowed_x_scalers)

    if x_scaler_type not in allowed_x_scalers:
        raise ValueError(f"❌ Invalid X scaler type: {x_scaler_type}. Allowed: {allowed_x_scalers}")
    if y_scaler_type not in allowed_y_scalers:
        raise ValueError(f"❌ Invalid Y scaler type: {y_scaler_type}. Allowed: {allowed_y_scalers}")
    if synthetic_policy != "none":
        raise ValueError(
            "load_or_create_scaled_sets only materializes non-synthetic classical scaled datasets. "
            "Synthetic dataset-level variants are modeled in the F5 contract, but they are not materialized here."
        )

    scaled_suffix = f"{scaled_df_name}_x{x_scaler_type}_y{y_scaler_type}"
    sets_base = _official_scaled_bundle_dir(storage_name=scaled_suffix, split_id=split_id)

    dirs = {
        "X": sets_base / "X",
        "y": sets_base / "y",
        "r": sets_base / "removed",
        "scalers": sets_base / "scalers",
        "meta": sets_base / "meta",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    filenames = {
        split: {
            "X": dirs["X"] / f"{scaled_suffix}_X_{split}.csv",
            "y": dirs["y"] / f"{scaled_suffix}_y_{split}.csv",
            "r": dirs["r"] / f"{scaled_suffix}_r_{split}.csv"
        } for split in ["train", "val", "test"]
    }

    scaler_paths = {
        "X": dirs["scalers"] / f"{scaled_suffix}_X_scaler.pkl",
        "y": dirs["scalers"] / f"{scaled_suffix}_y_scaler.pkl"
    }
    manifest_path = dirs["meta"] / "manifest.json"

    if all(
        filenames[split]["X"].exists() and
        filenames[split]["y"].exists() and
        filenames[split]["r"].exists()
        for split in ["train", "val", "test"]
    ) and all(p.exists() for p in scaler_paths.values()) and not force:
        log(f"📦 Scaled sets already exist for '{scaled_suffix}'. Loading from disk...", verbose)

        return (
            pd.read_csv(filenames["train"]["X"]),
            pd.read_csv(filenames["val"]["X"]),
            pd.read_csv(filenames["test"]["X"]),
            pd.read_csv(filenames["train"]["y"]),
            pd.read_csv(filenames["val"]["y"]),
            pd.read_csv(filenames["test"]["y"]),
            pd.read_csv(filenames["train"]["r"]),
            pd.read_csv(filenames["val"]["r"]),
            pd.read_csv(filenames["test"]["r"]),
            joblib.load(scaler_paths["X"]),
            joblib.load(scaler_paths["y"])
        )

    # Load raw sets
    X_train, X_val, X_test, y_train, y_val, y_test, r_train, r_val, r_test = load_or_create_raw_splits(
        df_name=raw_df_name,
        condition_col=condition_col,
        val_size=val_size,
        test_size=test_size,
        target=target,
        verbose=verbose,
        force=force,
        split_id=split_id,
    )

    X_cols = [col for col in X_train.columns if col not in exclude_cols]
    y_cols = [col for col in y_train.columns if col not in exclude_cols]

    # Fit scalers
    x_scaler = get_scaler(x_scaler_type).fit(X_train[X_cols])
    y_scaler = get_scaler(y_scaler_type).fit(y_train[y_cols])

    for split, X_df, y_df, r_df in zip(
        ["train", "val", "test"],
        [X_train, X_val, X_test],
        [y_train, y_val, y_test],
        [r_train, r_val, r_test]
    ):
        X_scaled = X_df.copy()
        y_scaled = y_df.copy()

        X_scaled[X_cols] = x_scaler.transform(X_df[X_cols])
        y_scaled[y_cols] = y_scaler.transform(y_df[y_cols])

        X_scaled.to_csv(filenames[split]["X"], index=False)
        y_scaled.to_csv(filenames[split]["y"], index=False)
        r_df.to_csv(filenames[split]["r"], index=False)

        log(f"✅ Saved {split} → X: {X_scaled.shape}, y: {y_scaled.shape}, r: {r_df.shape}", verbose)

    joblib.dump(x_scaler, scaler_paths["X"])
    joblib.dump(y_scaler, scaler_paths["y"])
    log(f"💾 Scalers saved to {scaler_paths['X']} and {scaler_paths['y']}", verbose)

    source_manifest_path = official_raw_bundle_manifest_path(split_id=split_id, dataset_name=raw_df_name)
    source_manifest = _load_json_dict(source_manifest_path)
    if raw_df_name == DEFAULT_OFFICIAL_DATASET_NAME and source_manifest and source_manifest.get("policy_status") == "canonical":
        dataset_axes = {
            "x_transform": x_scaler_type,
            "y_transform": y_scaler_type,
            "synthetic_policy": synthetic_policy,
        }
        support = classify_supported_dataset_space(**dataset_axes)
        manifest = build_canonical_derived_manifest(
            dataset_name=build_canonical_dataset_name(
                x_transform=x_scaler_type,
                y_transform=y_scaler_type,
                synthetic_policy=synthetic_policy,
                version=manifest_version,
            ),
            dataset_level_axes=dataset_axes,
            split_id=split_id,
            cleaning_policy_id=str(
                source_manifest.get("cleaning_policy_id", DEFAULT_OFFICIAL_CLEANING_POLICY_ID)
            ),
            source_dataset_manifest_path=source_manifest_path,
            source_split_manifest_path=source_manifest.get("source_split_manifest"),
            source_cleaning_manifest_path=source_manifest.get("source_cleaning_manifest"),
            source_manifest=source_manifest,
            support_status=str(support["support_status"]),
            artifacts={
                "storage_name": scaled_suffix,
                "X": {split: filenames[split]["X"] for split in ["train", "val", "test"]},
                "y": {split: filenames[split]["y"] for split in ["train", "val", "test"]},
                "removed": {split: filenames[split]["r"] for split in ["train", "val", "test"]},
            },
            scaler_artifacts={
                "X": scaler_paths["X"],
                "y": scaler_paths["y"],
            },
            extra_manifest_fields={
                "condition_col": condition_col,
                "target": target,
                "exclude_cols": list(exclude_cols),
                "x_scaler": {
                    "name": x_scaler_type,
                    "fit_cols": X_cols,
                },
                "y_scaler": {
                    "name": y_scaler_type,
                    "fit_cols": y_cols,
                },
            },
        )
        dump_json(manifest, manifest_path)

    return (
        pd.read_csv(filenames["train"]["X"]),
        pd.read_csv(filenames["val"]["X"]),
        pd.read_csv(filenames["test"]["X"]),
        pd.read_csv(filenames["train"]["y"]),
        pd.read_csv(filenames["val"]["y"]),
        pd.read_csv(filenames["test"]["y"]),
        pd.read_csv(filenames["train"]["r"]),
        pd.read_csv(filenames["val"]["r"]),
        pd.read_csv(filenames["test"]["r"]),
        x_scaler,
        y_scaler
    )

def inverse_transform_and_compare(original_df, scaled_df, scaler, exclude_cols, name=""):
    original = original_df.copy()
    scaled = scaled_df.copy()

    cols_to_scale = [col for col in original.columns if col not in exclude_cols]
    original = original[cols_to_scale].reset_index(drop=True)
    recovered = pd.DataFrame(
        scaler.inverse_transform(scaled[cols_to_scale]),
        columns=cols_to_scale
    )

    recovered = recovered[original.columns]
    metrics = compute_rrmse_mean_std(original, recovered)
    print(f"🔁 Recovery check: {name}")
    print(f"   RRMSE of Mean: {metrics['mean_rrmse']:.6f}")
    print(f"   RRMSE of Std:  {metrics['std_rrmse']:.6f}")
    print()


def _official_flowpre_suffix(meta_tag: str, y_scaler_name: str) -> str:
    return f"df_scaled_xflowpre_{meta_tag}_y{y_scaler_name}"


def _official_flowpre_dirs(meta_tag: str, y_scaler_name: str, split_id: str = DEFAULT_OFFICIAL_SPLIT_ID) -> Dict[str, Path]:
    base = _official_scaled_bundle_dir(_official_flowpre_suffix(meta_tag, y_scaler_name), split_id=split_id)
    return {
        "base": base,
        "X": base / "X",
        "y": base / "y",
        "removed": base / "removed",
        "scalers": base / "scalers",
        "meta": base / "meta",
    }


def _official_flowgen_suffix(meta_tag: str, y_scaler_name: str, synthetic_policy_id: str) -> str:
    return f"df_augmented_xflowpre_{meta_tag}_y{y_scaler_name}__syn-{synthetic_policy_id}"


def _official_flowgen_dirs(
    meta_tag: str,
    y_scaler_name: str,
    synthetic_policy_id: str,
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
) -> Dict[str, Path]:
    base = _official_augmented_bundle_dir(
        _official_flowgen_suffix(meta_tag, y_scaler_name, synthetic_policy_id),
        split_id=split_id,
    )
    return {
        "base": base,
        "X": base / "X",
        "y": base / "y",
        "removed": base / "removed",
        "scalers": base / "scalers",
        "meta": base / "meta",
    }


def _official_kmeans_smote_suffix(source_storage_name: str, synthetic_policy_id: str) -> str:
    return f"{source_storage_name}__syn-{synthetic_policy_id}"


def _official_kmeans_smote_dirs(
    source_storage_name: str,
    synthetic_policy_id: str,
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
) -> Dict[str, Path]:
    base = _official_augmented_bundle_dir(
        _official_kmeans_smote_suffix(source_storage_name, synthetic_policy_id),
        split_id=split_id,
    )
    return {
        "base": base,
        "X": base / "X",
        "y": base / "y",
        "removed": base / "removed",
        "scalers": base / "scalers",
        "meta": base / "meta",
    }


def _load_promotion_manifest(path: str | Path) -> dict[str, Any]:
    payload = _load_json_dict(path)
    if not payload:
        raise FileNotFoundError(f"Promotion manifest not found or invalid: {path}")
    return payload


def _resolve_run_manifest_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    repo_candidate = Path(ROOT_PATH) / candidate
    if repo_candidate.exists():
        return repo_candidate
    raise FileNotFoundError(f"Run manifest not found: {path}")


def _run_manifest_to_run_dir(run_manifest_path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = _resolve_run_manifest_path(run_manifest_path)
    manifest = _load_json_dict(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"Invalid run manifest: {manifest_path}")
    return manifest_path.parent, manifest


def _load_bundle_manifest(base_dir: Path) -> dict[str, Any]:
    manifest = _load_json_dict(base_dir / "meta" / "manifest.json")
    if not manifest:
        raise FileNotFoundError(f"Bundle manifest not found under: {base_dir}")
    return manifest


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    repo_candidate = Path(ROOT_PATH) / candidate
    if repo_candidate.exists():
        return repo_candidate
    raise FileNotFoundError(f"Path not found: {path}")


def _read_bundle_csv(path: str | Path, *, name: str) -> pd.DataFrame:
    resolved = _resolve_repo_path(path)
    df = pd.read_csv(resolved)
    if "post_cleaning_index" not in df.columns:
        raise ValueError(f"[{name}] missing required column 'post_cleaning_index' in {resolved}")
    try:
        df["post_cleaning_index"] = df["post_cleaning_index"].astype(int)
    except Exception:
        pass
    df.sort_values("post_cleaning_index", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _load_bundle_frames_from_manifest(bundle_manifest: Mapping[str, Any]) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
]:
    artifacts = dict(bundle_manifest.get("artifacts") or {})
    x_artifacts = dict(artifacts.get("X") or {})
    y_artifacts = dict(artifacts.get("y") or {})
    removed_artifacts = dict(artifacts.get("removed") or {})
    if not x_artifacts or not y_artifacts or not removed_artifacts:
        raise ValueError("Bundle manifest is missing one or more artifact groups: X, y, removed.")
    return (
        _read_bundle_csv(x_artifacts["train"], name="X_train"),
        _read_bundle_csv(x_artifacts["val"], name="X_val"),
        _read_bundle_csv(x_artifacts["test"], name="X_test"),
        _read_bundle_csv(y_artifacts["train"], name="y_train"),
        _read_bundle_csv(y_artifacts["val"], name="y_val"),
        _read_bundle_csv(y_artifacts["test"], name="y_test"),
        _read_bundle_csv(removed_artifacts["train"], name="r_train"),
        _read_bundle_csv(removed_artifacts["val"], name="r_val"),
        _read_bundle_csv(removed_artifacts["test"], name="r_test"),
    )


def _copy_scaler_artifacts_to_dir(
    scaler_artifacts: Mapping[str, Any],
    *,
    dst_dir: Path,
) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for key, path in dict(scaler_artifacts or {}).items():
        src_path = _resolve_repo_path(path)
        dst_path = dst_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        copied[str(key)] = dst_path
    return copied


def _build_condition_value_to_label_map(
    *,
    condition_values: list[int],
    verbose: bool = False,
) -> dict[int, str]:
    try:
        raw_mapping = load_type_mapping(verbose=verbose)
    except Exception:
        raw_mapping = {}
    inverse = {int(idx): str(label) for label, idx in dict(raw_mapping or {}).items()}
    return {int(value): inverse.get(int(value), f"type_{int(value)}") for value in condition_values}


def materialize_kmeans_smote_joint_set(
    source_bundle_manifest_path: str | Path,
    *,
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> Tuple[Path, dict]:
    source_manifest = _load_promotion_manifest(source_bundle_manifest_path)
    if str(source_manifest.get("policy_status", "")).lower() != "canonical":
        raise ValueError("kmeans_smote requires a canonical source bundle manifest.")
    if str(source_manifest.get("dataset_role", "")).lower() != "derived_modeling_bundle":
        raise ValueError(
            "kmeans_smote v1 expects a canonical derived bundle source, not the raw official bundle."
        )

    source_axes = dict(source_manifest.get("dataset_level_axes") or {})
    if not source_axes.get("x_transform") or not source_axes.get("y_transform"):
        raise ValueError(
            "kmeans_smote v1 requires a source bundle with explicit dataset_level_axes "
            "(x_transform, y_transform)."
        )
    if str(source_axes.get("synthetic_policy", "none")).lower() != "none":
        raise ValueError("kmeans_smote v1 requires a non-synthetic source bundle (synthetic_policy='none').")

    split_id = str(source_manifest.get("split_id") or DEFAULT_OFFICIAL_SPLIT_ID)
    source_storage_name = str(
        (source_manifest.get("artifacts") or {}).get("storage_name")
        or source_manifest.get("dataset_name")
        or "bundle"
    )

    default_config_path = Path(ROOT_PATH) / "config" / "kmeans_smote_joint_base_v1.yaml"
    if config_path is None and config is None and default_config_path.exists():
        config_path = default_config_path

    cfg, raw_cfg_payload, resolved_config_path = load_kmeans_smote_joint_config(
        config_path=config_path,
        config=config,
    )
    condition_col = str(cfg.condition_col)
    synthetic_policy_id = (
        f"kmeans_smote__{cfg.synthetic_policy_config_id}__seed{int(cfg.synthetic_seed)}__v1"
    )

    dirs = _official_kmeans_smote_dirs(
        source_storage_name,
        synthetic_policy_id,
        split_id=split_id,
    )
    _ensure_dirs(dirs)
    suffix = _official_kmeans_smote_suffix(source_storage_name, synthetic_policy_id)
    manifest_path = dirs["meta"] / "manifest.json"

    if not force and manifest_path.exists():
        manifest = _load_json_dict(manifest_path)
        if manifest:
            if verbose:
                rel = dirs["base"].relative_to(Path(ROOT_PATH))
                log(f"📦 Exists → {rel}", verbose)
            return dirs["base"], manifest

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        r_train,
        r_val,
        r_test,
    ) = _load_bundle_frames_from_manifest(source_manifest)

    condition_values = sorted(X_train[condition_col].astype(int).unique().tolist())
    condition_value_to_label_map = _build_condition_value_to_label_map(
        condition_values=condition_values,
        verbose=verbose,
    )

    augmented, report = augment_canonical_bundle_with_kmeans_smote(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        r_train=r_train,
        r_val=r_val,
        r_test=r_test,
        config=cfg,
        condition_col=condition_col,
        condition_value_to_label_map=condition_value_to_label_map,
    )

    X_train_aug = augmented["X_train"]
    X_val_aug = augmented["X_val"]
    X_test_aug = augmented["X_test"]
    y_train_aug = augmented["y_train"]
    y_val_aug = augmented["y_val"]
    y_test_aug = augmented["y_test"]
    r_train_aug = augmented["r_train"]
    r_val_aug = augmented["r_val"]
    r_test_aug = augmented["r_test"]

    X_train_aug.to_csv(dirs["X"] / f"{suffix}_X_train.csv", index=False)
    X_val_aug.to_csv(dirs["X"] / f"{suffix}_X_val.csv", index=False)
    X_test_aug.to_csv(dirs["X"] / f"{suffix}_X_test.csv", index=False)

    y_train_aug.to_csv(dirs["y"] / f"{suffix}_y_train.csv", index=False)
    y_val_aug.to_csv(dirs["y"] / f"{suffix}_y_val.csv", index=False)
    y_test_aug.to_csv(dirs["y"] / f"{suffix}_y_test.csv", index=False)

    r_train_aug.to_csv(dirs["removed"] / f"{suffix}_r_train.csv", index=False)
    r_val_aug.to_csv(dirs["removed"] / f"{suffix}_r_val.csv", index=False)
    r_test_aug.to_csv(dirs["removed"] / f"{suffix}_r_test.csv", index=False)

    copied_scalers = _copy_scaler_artifacts_to_dir(
        source_manifest.get("scaler_artifacts", {}),
        dst_dir=dirs["scalers"],
    )

    counts_by_split, counts_by_class = counts_from_source_manifest(source_manifest)
    counts_by_split = dict(counts_by_split)
    counts_by_split["train"] = int(len(X_train_aug))
    counts_by_split["val"] = int(len(X_val_aug))
    counts_by_split["test"] = int(len(X_test_aug))

    counts_by_class = dict(counts_by_class)
    train_class_counts = X_train_aug[condition_col].astype(int).value_counts().sort_index()
    counts_by_class["train"] = {
        str(condition_value_to_label_map.get(int(class_id), f"type_{int(class_id)}")): int(count)
        for class_id, count in train_class_counts.items()
    }

    dataset_axes = {
        "x_transform": str(source_axes.get("x_transform")),
        "y_transform": str(source_axes.get("y_transform")),
        "synthetic_policy": "kmeans_smote",
    }
    source_requirements = ["official_raw_bundle"]
    if str(dataset_axes["x_transform"]).startswith("flowpre_"):
        source_requirements.append("flowpre_upstream")

    report_payload = {
        **report,
        "source_bundle_manifest_path": path_relative_to_root(_resolve_repo_path(source_bundle_manifest_path)),
        "source_bundle_name": str(source_manifest.get("dataset_name")),
        "synthetic_policy_id": synthetic_policy_id,
        "synthetic_policy_external_name": "kmeans_smote",
        "synthetic_policy_variant": "kmeans_smote_joint",
        "dataset_level_axes": dataset_axes,
        "source_requirements": source_requirements,
        "config_path": None if resolved_config_path is None else path_relative_to_root(resolved_config_path),
        "config_payload": raw_cfg_payload,
    }
    dump_json(report_payload, dirs["meta"] / "kmeans_smote_joint_report.json")

    manifest = build_canonical_derived_manifest(
        dataset_name=build_canonical_dataset_name(
            x_transform=dataset_axes["x_transform"],
            y_transform=dataset_axes["y_transform"],
            synthetic_policy="kmeans_smote",
            version=DEFAULT_DATASET_CONTRACT_VERSION,
        ),
        dataset_level_axes=dataset_axes,
        split_id=split_id,
        cleaning_policy_id=str(source_manifest.get("cleaning_policy_id")),
        source_dataset_manifest_path=source_manifest["source_dataset_manifest"],
        source_split_manifest_path=source_manifest.get("source_split_manifest"),
        source_cleaning_manifest_path=source_manifest.get("source_cleaning_manifest"),
        source_manifest=source_manifest,
        support_status="materialized_now",
        synthetic_policy_id=synthetic_policy_id,
        train_only_mutations=["synthetic_policy", "is_synth"],
        artifacts={
            "storage_name": suffix,
            "X": {
                "train": dirs["X"] / f"{suffix}_X_train.csv",
                "val": dirs["X"] / f"{suffix}_X_val.csv",
                "test": dirs["X"] / f"{suffix}_X_test.csv",
            },
            "y": {
                "train": dirs["y"] / f"{suffix}_y_train.csv",
                "val": dirs["y"] / f"{suffix}_y_val.csv",
                "test": dirs["y"] / f"{suffix}_y_test.csv",
            },
            "removed": {
                "train": dirs["removed"] / f"{suffix}_r_train.csv",
                "val": dirs["removed"] / f"{suffix}_r_val.csv",
                "test": dirs["removed"] / f"{suffix}_r_test.csv",
            },
        },
        scaler_artifacts=copied_scalers,
        upstream_model_manifests=list(source_manifest.get("upstream_model_manifests", [])),
        extra_manifest_fields={
            "source_bundle_manifest_path": path_relative_to_root(_resolve_repo_path(source_bundle_manifest_path)),
            "source_bundle_name": str(source_manifest.get("dataset_name")),
            "condition_col": condition_col,
            "condition_value_to_label_map": {
                str(int(key)): str(value) for key, value in condition_value_to_label_map.items()
            },
            "synthetic_policy_family": "kmeans_smote_joint",
            "synthetic_policy_config_id": cfg.synthetic_policy_config_id,
            "synthetic_seed": int(cfg.synthetic_seed),
            "target_policy": {
                "mode": cfg.target_mode,
                "value": cfg.target_value,
            },
            "resolved_target_by_class": report["resolved_target_by_class"],
            "resolved_cluster_k_by_class": report["resolved_cluster_k_by_class"],
            "silhouette_by_class": report["silhouette_by_class"],
            "resolved_neighbor_k_by_class": report["resolved_neighbor_k_by_class"],
            "metric_space_mode": cfg.metric_space_mode,
            "cluster_min_size": int(cfg.min_cluster_size),
            "added_by_class": report["added_by_class"],
            "counts_by_split": counts_by_split,
            "counts_by_class": counts_by_class,
            "train_source_rows": int(len(X_train)),
            "train_augmented_rows": int(len(X_train_aug)),
            "generated_rows_total": int(report.get("generated_rows_total", 0)),
            "kmeans_smote_joint_report": path_relative_to_root(dirs["meta"] / "kmeans_smote_joint_report.json"),
            "source_requirements": source_requirements,
        },
    )
    dump_json(manifest, manifest_path)

    if verbose:
        rel = dirs["base"].relative_to(Path(ROOT_PATH))
        log(f"✅ Saved kmeans_smote joint augmented set → {rel}", verbose)
    return dirs["base"], manifest


def materialize_kmeans_smote_joint_sets(
    source_bundle_manifest_paths: List[str | Path],
    *,
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> List[str]:
    created: List[str] = []
    for source_bundle_manifest_path in source_bundle_manifest_paths:
        base_dir, _ = materialize_kmeans_smote_joint_set(
            source_bundle_manifest_path,
            config_path=config_path,
            config=config,
            force=force,
            verbose=verbose,
        )
        created.append(path_relative_to_root(base_dir))
    return created

def _flowpre_suffix(meta_tag: str, y_scaler_name: str) -> str:
    # e.g. "df_scaled_x_flowpre_rrmse_yminmax"
    return f"df_scaled_x_flowpre_{meta_tag}_y{y_scaler_name}"

def _flowpre_dirs(meta_tag: str, y_scaler_name: str) -> Dict[str, Path]:
    base = _SCALED_ROOT / _flowpre_suffix(meta_tag, y_scaler_name)
    return {
        "base":    base,
        "X":       base / "X",
        "y":       base / "y",
        "removed": base / "removed",
        "scalers": base / "scalers",
        "meta":    base / "meta",
        "model":   base / "model",     # ← store copied YAML + .pt here
    }

def _ensure_dirs(dirs: Dict[str, Path]) -> None:
    for k, p in dirs.items():
        if k == "base":
            continue
        p.mkdir(parents=True, exist_ok=True)

def _locate_model_artifacts(model_name: str, model_family: str = "flow_pre") -> Tuple[Path, Path]:
    """
    Return (config_yaml, weights_pt) from official-first model roots.
    Tries exact filenames first, then falls back to first match.
    """
    model_dir = _locate_model_run_dir(model_name, model_family=model_family)

    cfg = model_dir / f"{model_name}.yaml"
    if not cfg.exists():
        # fallback: first .yaml that doesn't look like results/influence
        cands = [p for p in model_dir.glob("*.yaml") if "_results" not in p.stem]
        if not cands:
            raise FileNotFoundError(f"No YAML config found under {model_dir}")
        cfg = sorted(cands)[0]

    pt = model_dir / f"{model_name}.pt"
    if not pt.exists():
        pts = list(model_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No .pt weights found under {model_dir}")
        pt = sorted(pts)[0]

    return cfg, pt

def _copy_model_artifacts(model_name: str, dest_dir: Path) -> Dict[str, str]:
    """
    Copy YAML + .pt into dest_dir. Returns dict with their relative paths.
    If files already exist with same name, they are left as-is.
    """
    cfg_src, pt_src = _locate_model_artifacts(model_name, model_family="flow_pre")
    dest_dir.mkdir(parents=True, exist_ok=True)

    cfg_dst = dest_dir / cfg_src.name
    pt_dst  = dest_dir / pt_src.name

    if not cfg_dst.exists():
        shutil.copy2(cfg_src, cfg_dst)
    if not pt_dst.exists():
        shutil.copy2(pt_src, pt_dst)

    return {
        "config": path_relative_to_root(cfg_dst),
        "weights": path_relative_to_root(pt_dst),
        "source_dir": path_relative_to_root(_locate_model_run_dir(model_name, model_family="flow_pre")),
    }

def _set_has_all_core_files(dirs: Dict[str, Path], suffix: str) -> bool:
    """Check X/Y/removed CSVs, scaler, and presence of model artifacts (yaml + .pt)."""
    required_csvs = [
        dirs["X"] / f"{suffix}_X_train.csv",
        dirs["X"] / f"{suffix}_X_val.csv",
        dirs["X"] / f"{suffix}_X_test.csv",
        dirs["y"] / f"{suffix}_y_train.csv",
        dirs["y"] / f"{suffix}_y_val.csv",
        dirs["y"] / f"{suffix}_y_test.csv",
        dirs["removed"] / f"{suffix}_r_train.csv",
        dirs["removed"] / f"{suffix}_r_val.csv",
        dirs["removed"] / f"{suffix}_r_test.csv",
        dirs["scalers"] / f"{suffix}_y_scaler.pkl",
    ]
    csv_ok = all(p.exists() for p in required_csvs)
    # model dir must contain at least one yaml and one pt
    has_yaml = any(dirs["model"].glob("*.yaml"))
    has_pt   = any(dirs["model"].glob("*.pt"))
    return csv_ok and has_yaml and has_pt

def transform_X_with_flowpre_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    condition_col: str = "type",
    cols_to_exclude: Optional[List[str]] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Uses training.transform_to_latent_with_flowpre to encode X into latent z.
    Returns latent DataFrames for train/val/test (with condition_col and optional post_cleaning_index preserved).
    """
    if cols_to_exclude is None:
        cols_to_exclude = ["post_cleaning_index"]

    # ⬇️ Lazy import to break the cycle
    from training.train_flow_pre import transform_to_latent_with_flowpre

    X_lat_train, X_lat_val, X_lat_test = transform_to_latent_with_flowpre(
        condition_col=condition_col,
        cols_to_exclude=cols_to_exclude,
        model_name=model_name,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        device=device,
        verbose=verbose
    )
    return X_lat_train, X_lat_val, X_lat_test


def encode_dataframe_with_flowpre_model(
    model_name: str,
    df: pd.DataFrame,
    *,
    reference_df: pd.DataFrame,
    condition_col: str = "type",
    cols_to_exclude: Optional[List[str]] = None,
    device: str = "auto",
    verbose: bool = True,
) -> pd.DataFrame:
    if cols_to_exclude is None:
        cols_to_exclude = ["post_cleaning_index"]

    import torch

    from training.train_flow_pre import (
        build_flow_pre_model,
        encode_with_flowpre_model,
        filter_flowpre_columns,
    )
    from training.utils import load_yaml_config, select_training_device

    config_path, model_path = _locate_model_artifacts(model_name, model_family="flow_pre")
    config = load_yaml_config(config_path)
    device_obj = select_training_device(device)

    ref_filtered = filter_flowpre_columns(reference_df, cols_to_exclude, condition_col)
    input_dim = ref_filtered.drop(columns=[condition_col]).shape[1]
    num_classes = int(ref_filtered[condition_col].nunique())

    model = build_flow_pre_model(
        config["model"],
        input_dim=input_dim,
        num_classes=num_classes,
        device=device_obj,
    )
    state_dict = torch.load(model_path, map_location=device_obj)
    model.load_state_dict(state_dict)
    model.eval()

    if verbose:
        log(f"📦 Encoding dataframe with FlowPre model: {model_name}", verbose)
    return encode_with_flowpre_model(df, model, device_obj, condition_col, cols_to_exclude)


def _load_flowgen_model_from_promotion(
    promotion_manifest_path: str | Path,
    *,
    x_reference: pd.DataFrame,
    y_reference: pd.DataFrame,
    condition_col: str = "type",
    device: str = "auto",
) -> tuple[Any, Any, dict[str, Any], Path]:
    import torch

    from models.flowgen import FlowGen
    from training.utils import load_yaml_config, select_training_device

    promotion_manifest = _load_promotion_manifest(promotion_manifest_path)
    run_dir, run_manifest = _run_manifest_to_run_dir(promotion_manifest["source_run_manifest"])
    run_id = str(run_manifest.get("run_id", run_dir.name))

    cfg_path = run_dir / f"{run_id}.yaml"
    if not cfg_path.exists():
        ymls = sorted(p for p in run_dir.glob("*.yaml") if "_results" not in p.stem)
        if not ymls:
            raise FileNotFoundError(f"No FlowGen YAML found under: {run_dir}")
        cfg_path = ymls[0]

    pt_path = run_dir / f"{run_id}.pt"
    if not pt_path.exists():
        finetuned = run_dir / f"{run_id}_finetuned.pt"
        if finetuned.exists():
            pt_path = finetuned
        else:
            pts = sorted(run_dir.glob("*.pt"))
            if not pts:
                raise FileNotFoundError(f"No FlowGen weights found under: {run_dir}")
            pt_path = pts[0]

    config = load_yaml_config(cfg_path)
    model_cfg = config["model"]
    device_obj = select_training_device(device)

    x_cols = [c for c in x_reference.columns if c not in ("post_cleaning_index", condition_col)]
    y_cols = [c for c in y_reference.columns if c != "post_cleaning_index"]
    num_classes = int(x_reference[condition_col].nunique())

    model = FlowGen(
        x_dim=len(x_cols),
        y_dim=len(y_cols),
        num_classes=num_classes,
        embedding_dim=model_cfg.get("embedding_dim", 8),
        hidden_features=model_cfg.get("hidden_features", 64),
        num_layers=model_cfg.get("num_layers", 2),
        use_actnorm=model_cfg.get("use_actnorm", True),
        use_learnable_permutations=model_cfg.get("use_learnable_permutations", True),
        num_bins=model_cfg.get("num_bins", 8),
        tail_bound=model_cfg.get("tail_bound", 3.0),
        initial_affine_layers=model_cfg.get("initial_affine_layers", 2),
        affine_rq_ratio=tuple(model_cfg.get("affine_rq_ratio", [1, 2])),
        n_repeat_blocks=model_cfg.get("n_repeat_blocks", 4),
        final_rq_layers=model_cfg.get("final_rq_layers", 3),
        lulinear_finisher=model_cfg.get("lulinear_finisher", True),
        device=device_obj,
    )
    state_dict = torch.load(pt_path, map_location=device_obj)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, device_obj, promotion_manifest, pt_path

def save_flowpre_scaled_bundle(
    X_lat_train: pd.DataFrame, X_lat_val: pd.DataFrame, X_lat_test: pd.DataFrame,
    y_train: pd.DataFrame, y_val: pd.DataFrame, y_test: pd.DataFrame,
    r_train: pd.DataFrame, r_val: pd.DataFrame, r_test: pd.DataFrame,
    y_scaler_name: str,
    meta_tag: str,           # "rrmse" | "mvn" | "fair"
    model_name: str,         # subfolder in outputs/models/flow_pre
    condition_col: str = "type",
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    source_dataset_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
    manifest_version: str = DEFAULT_DATASET_CONTRACT_VERSION,
    verbose: bool = True,
) -> Tuple[Path, dict]:
    """
    Saves latent-X + scaled-Y + removed to:
      data/sets/scaled_sets/df_scaled_x_flowpre_{meta_tag}_y{y_scaler_name}/
    Also saves the Y scaler AND copies the model's YAML + .pt into ./model/.
    """
    dirs = _flowpre_dirs(meta_tag, y_scaler_name)
    _ensure_dirs(dirs)
    suffix = _flowpre_suffix(meta_tag, y_scaler_name)

    # --- Fit Y scaler on TRAIN (exclude ids and condition)
    exclude_cols = ["post_cleaning_index", condition_col]
    y_cols = [c for c in y_train.columns if c not in exclude_cols]
    scaler = get_scaler(y_scaler_name).fit(y_train[y_cols])

    # Transform Y splits
    y_train_sc = y_train.copy(); y_train_sc[y_cols] = scaler.transform(y_train[y_cols])
    y_val_sc   = y_val.copy();   y_val_sc[y_cols]   = scaler.transform(y_val[y_cols])
    y_test_sc  = y_test.copy();  y_test_sc[y_cols]  = scaler.transform(y_test[y_cols])

    # Save CSVs
    X_lat_train.to_csv(dirs["X"] / f"{suffix}_X_train.csv", index=False)
    X_lat_val.to_csv(  dirs["X"] / f"{suffix}_X_val.csv",   index=False)
    X_lat_test.to_csv( dirs["X"] / f"{suffix}_X_test.csv",  index=False)

    y_train_sc.to_csv(dirs["y"] / f"{suffix}_y_train.csv", index=False)
    y_val_sc.to_csv(  dirs["y"] / f"{suffix}_y_val.csv",   index=False)
    y_test_sc.to_csv( dirs["y"] / f"{suffix}_y_test.csv",  index=False)

    r_train.to_csv(dirs["removed"] / f"{suffix}_r_train.csv", index=False)
    r_val.to_csv(  dirs["removed"] / f"{suffix}_r_val.csv",   index=False)
    r_test.to_csv( dirs["removed"] / f"{suffix}_r_test.csv",  index=False)

    # Save Y scaler
    scaler_path = dirs["scalers"] / f"{suffix}_y_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Copy model YAML + .pt into ./model/
    copied = _copy_model_artifacts(model_name, dirs["model"])

    # Manifest
    source_manifest_path = official_raw_bundle_manifest_path(
        split_id=split_id,
        dataset_name=source_dataset_name,
    )
    source_manifest = _load_json_dict(source_manifest_path)
    dataset_axes = {
        "x_transform": f"flowpre_{meta_tag}",
        "y_transform": y_scaler_name,
        "synthetic_policy": "none",
    }
    support = classify_supported_dataset_space(**dataset_axes)
    manifest = build_canonical_derived_manifest(
        dataset_name=build_canonical_dataset_name(
            x_transform=dataset_axes["x_transform"],
            y_transform=y_scaler_name,
            synthetic_policy="none",
            version=manifest_version,
        ),
        dataset_level_axes=dataset_axes,
        split_id=split_id,
        cleaning_policy_id=str(
            (source_manifest or {}).get("cleaning_policy_id", DEFAULT_OFFICIAL_CLEANING_POLICY_ID)
        ),
        source_dataset_manifest_path=source_manifest_path,
        source_split_manifest_path=(source_manifest or {}).get("source_split_manifest"),
        source_cleaning_manifest_path=(source_manifest or {}).get("source_cleaning_manifest"),
        source_manifest=source_manifest,
        support_status=str(support["support_status"]),
        artifacts={
            "storage_name": suffix,
            "X": {
                "train": dirs["X"] / f"{suffix}_X_train.csv",
                "val": dirs["X"] / f"{suffix}_X_val.csv",
                "test": dirs["X"] / f"{suffix}_X_test.csv",
            },
            "y": {
                "train": dirs["y"] / f"{suffix}_y_train.csv",
                "val": dirs["y"] / f"{suffix}_y_val.csv",
                "test": dirs["y"] / f"{suffix}_y_test.csv",
            },
            "removed": {
                "train": dirs["removed"] / f"{suffix}_r_train.csv",
                "val": dirs["removed"] / f"{suffix}_r_val.csv",
                "test": dirs["removed"] / f"{suffix}_r_test.csv",
            },
        },
        scaler_artifacts={
            "y": scaler_path,
        },
        extra_manifest_fields={
            "condition_col": condition_col,
            "suffix": suffix,
            "x_transform": {
                "type": "flowpre_latent",
                "model_name": model_name,
                "source_dir": copied["source_dir"],
                "copied_config": copied["config"],
                "copied_weights": copied["weights"],
            },
            "y_scaler": {
                "name": y_scaler_name,
                "path": path_relative_to_root(scaler_path),
                "fit_cols": y_cols,
            },
        },
    )
    (dirs["meta"]).mkdir(parents=True, exist_ok=True)
    dump_json(manifest, dirs["meta"] / "manifest.json")

    if verbose:
        rel = dirs["base"].relative_to(Path(ROOT_PATH))
        log(f"✅ Saved FlowPre set → {rel}\n    ↳ + config/weights copied to {dirs['model'].relative_to(Path(ROOT_PATH))}")
    return dirs["base"], manifest

def load_flowpre_scaled_set(
    meta_tag: str,
    y_scaler_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame,
           object]:
    """
    Loads the nine CSVs + the saved Y scaler for a given FlowPre set.
    Returns:
      X_train, X_val, X_test,
      y_train, y_val, y_test,
      r_train, r_val, r_test,
      y_scaler
    """
    dirs = _flowpre_dirs(meta_tag, y_scaler_name)
    suffix = _flowpre_suffix(meta_tag, y_scaler_name)

    X_train = pd.read_csv(dirs["X"] / f"{suffix}_X_train.csv")
    X_val   = pd.read_csv(dirs["X"] / f"{suffix}_X_val.csv")
    X_test  = pd.read_csv(dirs["X"] / f"{suffix}_X_test.csv")

    y_train = pd.read_csv(dirs["y"] / f"{suffix}_y_train.csv")
    y_val   = pd.read_csv(dirs["y"] / f"{suffix}_y_val.csv")
    y_test  = pd.read_csv(dirs["y"] / f"{suffix}_y_test.csv")

    r_train = pd.read_csv(dirs["removed"] / f"{suffix}_r_train.csv")
    r_val   = pd.read_csv(dirs["removed"] / f"{suffix}_r_val.csv")
    r_test  = pd.read_csv(dirs["removed"] / f"{suffix}_r_test.csv")

    y_scaler = joblib.load(dirs["scalers"] / f"{suffix}_y_scaler.pkl")
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            r_train, r_val, r_test,
            y_scaler)

def load_or_create_flowpre_sets(
    model_map: Dict[str, str],              # {"rrmse": "<model_folder>", "mvn": "...", "fair": "..."}
    y_scalers: List[str],                   # e.g., ["minmax","quantile","standard"]
    df_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
    condition_col: str = "type",
    val_size: int = 150,
    test_size: int = 100,
    target: str = "init",
    device: str = "cuda",
    force: bool = False,
    verbose: bool = True,
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
) -> List[str]:
    """
    For each FlowPre model (tag → folder), encodes X into latent z, then for each y_scaler,
    creates a new set under:
      data/sets/scaled_sets/df_scaled_x_flowpre_{tag}_y{scaler}
    Also ensures the model's YAML + .pt are copied into each set's ./model/ dir.
    Returns a list of created/loaded set suffixes.
    """
    # 1) raw splits once
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     r_train, r_val, r_test) = load_or_create_raw_splits(
        df_name=df_name,
        condition_col=condition_col,
        val_size=val_size,
        test_size=test_size,
        target=target,
        force=False,
        verbose=verbose,
        split_id=split_id,
    )

    created = []
    for tag, model_name in model_map.items():
        if verbose:
            log(f"🌀 Encoding X with FlowPre model [{tag}] → {model_name}")
        # 2) X → latent with chosen FlowPre
        X_lat_train, X_lat_val, X_lat_test = transform_X_with_flowpre_model(
            model_name=model_name,
            X_train=X_train, X_val=X_val, X_test=X_test,
            condition_col=condition_col,
            cols_to_exclude=["post_cleaning_index"],
            device=device,
            verbose=verbose
        )

        # 3) per-Y-scaler save bundle (+ copy model artifacts)
        for y_s in y_scalers:
            dirs = _flowpre_dirs(tag, y_s)
            suffix = _flowpre_suffix(tag, y_s)

            if not force and _set_has_all_core_files(dirs, suffix):
                # Even if skipping CSV/scaler creation, ensure artifacts exist (idempotent)
                _copy_model_artifacts(model_name, dirs["model"])
                if verbose:
                    rel = dirs["base"].relative_to(Path(ROOT_PATH))
                    log(f"📦 Exists → {rel} (verified config/weights present)")
                created.append(suffix)
                continue

            base_dir, _ = save_flowpre_scaled_bundle(
                X_lat_train, X_lat_val, X_lat_test,
                y_train, y_val, y_test,
                r_train, r_val, r_test,
                y_scaler_name=y_s,
                meta_tag=tag,
                model_name=model_name,
                condition_col=condition_col,
                split_id=split_id,
                source_dataset_name=df_name,
                verbose=verbose
            )
            created.append(base_dir.name)

    return created


def _official_flowpre_bundle_has_core_files(meta_tag: str, y_scaler_name: str, split_id: str = DEFAULT_OFFICIAL_SPLIT_ID) -> bool:
    dirs = _official_flowpre_dirs(meta_tag, y_scaler_name, split_id=split_id)
    suffix = _official_flowpre_suffix(meta_tag, y_scaler_name)
    required = [
        dirs["X"] / f"{suffix}_X_train.csv",
        dirs["X"] / f"{suffix}_X_val.csv",
        dirs["X"] / f"{suffix}_X_test.csv",
        dirs["y"] / f"{suffix}_y_train.csv",
        dirs["y"] / f"{suffix}_y_val.csv",
        dirs["y"] / f"{suffix}_y_test.csv",
        dirs["removed"] / f"{suffix}_r_train.csv",
        dirs["removed"] / f"{suffix}_r_val.csv",
        dirs["removed"] / f"{suffix}_r_test.csv",
        dirs["scalers"] / f"{suffix}_y_scaler.pkl",
        dirs["meta"] / "manifest.json",
    ]
    return all(path.exists() for path in required)


def save_official_flowpre_scaled_bundle(
    X_lat_train: pd.DataFrame,
    X_lat_val: pd.DataFrame,
    X_lat_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    y_test: pd.DataFrame,
    r_train: pd.DataFrame,
    r_val: pd.DataFrame,
    r_test: pd.DataFrame,
    *,
    y_scaler_name: str,
    meta_tag: str,
    model_name: str,
    promotion_manifest_path: str | Path,
    flowpre_source_id: str,
    condition_col: str = "type",
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    source_dataset_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
    manifest_version: str = DEFAULT_DATASET_CONTRACT_VERSION,
    verbose: bool = True,
) -> Tuple[Path, dict]:
    dirs = _official_flowpre_dirs(meta_tag, y_scaler_name, split_id=split_id)
    _ensure_dirs(dirs)
    suffix = _official_flowpre_suffix(meta_tag, y_scaler_name)

    exclude_cols = ["post_cleaning_index", condition_col]
    y_cols = [c for c in y_train.columns if c not in exclude_cols]
    scaler = get_scaler(y_scaler_name).fit(y_train[y_cols])

    y_train_sc = y_train.copy()
    y_val_sc = y_val.copy()
    y_test_sc = y_test.copy()
    y_train_sc[y_cols] = scaler.transform(y_train[y_cols])
    y_val_sc[y_cols] = scaler.transform(y_val[y_cols])
    y_test_sc[y_cols] = scaler.transform(y_test[y_cols])

    X_lat_train.to_csv(dirs["X"] / f"{suffix}_X_train.csv", index=False)
    X_lat_val.to_csv(dirs["X"] / f"{suffix}_X_val.csv", index=False)
    X_lat_test.to_csv(dirs["X"] / f"{suffix}_X_test.csv", index=False)

    y_train_sc.to_csv(dirs["y"] / f"{suffix}_y_train.csv", index=False)
    y_val_sc.to_csv(dirs["y"] / f"{suffix}_y_val.csv", index=False)
    y_test_sc.to_csv(dirs["y"] / f"{suffix}_y_test.csv", index=False)

    r_train.to_csv(dirs["removed"] / f"{suffix}_r_train.csv", index=False)
    r_val.to_csv(dirs["removed"] / f"{suffix}_r_val.csv", index=False)
    r_test.to_csv(dirs["removed"] / f"{suffix}_r_test.csv", index=False)

    scaler_path = dirs["scalers"] / f"{suffix}_y_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    source_manifest_path = official_raw_bundle_manifest_path(split_id=split_id, dataset_name=source_dataset_name)
    source_manifest = _load_json_dict(source_manifest_path)
    dataset_axes = {
        "x_transform": f"flowpre_{meta_tag}",
        "y_transform": y_scaler_name,
        "synthetic_policy": "none",
    }
    manifest = build_canonical_derived_manifest(
        dataset_name=build_canonical_dataset_name(
            x_transform=dataset_axes["x_transform"],
            y_transform=y_scaler_name,
            synthetic_policy="none",
            version=manifest_version,
        ),
        dataset_level_axes=dataset_axes,
        split_id=split_id,
        cleaning_policy_id=str(
            (source_manifest or {}).get("cleaning_policy_id", DEFAULT_OFFICIAL_CLEANING_POLICY_ID)
        ),
        source_dataset_manifest_path=source_manifest_path,
        source_split_manifest_path=(source_manifest or {}).get("source_split_manifest"),
        source_cleaning_manifest_path=(source_manifest or {}).get("source_cleaning_manifest"),
        source_manifest=source_manifest,
        support_status="materialized_now",
        artifacts={
            "storage_name": suffix,
            "X": {
                "train": dirs["X"] / f"{suffix}_X_train.csv",
                "val": dirs["X"] / f"{suffix}_X_val.csv",
                "test": dirs["X"] / f"{suffix}_X_test.csv",
            },
            "y": {
                "train": dirs["y"] / f"{suffix}_y_train.csv",
                "val": dirs["y"] / f"{suffix}_y_val.csv",
                "test": dirs["y"] / f"{suffix}_y_test.csv",
            },
            "removed": {
                "train": dirs["removed"] / f"{suffix}_r_train.csv",
                "val": dirs["removed"] / f"{suffix}_r_val.csv",
                "test": dirs["removed"] / f"{suffix}_r_test.csv",
            },
        },
        scaler_artifacts={"y": scaler_path},
        upstream_model_manifests=[promotion_manifest_path],
        extra_manifest_fields={
            "condition_col": condition_col,
            "suffix": suffix,
            "branch_id": meta_tag,
            "flowpre_source_id": flowpre_source_id,
            "x_transform": {
                "type": "flowpre_latent",
                "model_name": model_name,
                "flowpre_source_id": flowpre_source_id,
            },
            "y_scaler": {
                "name": y_scaler_name,
                "path": path_relative_to_root(scaler_path),
                "fit_cols": y_cols,
            },
        },
    )
    dump_json(manifest, dirs["meta"] / "manifest.json")

    if verbose:
        rel = dirs["base"].relative_to(Path(ROOT_PATH))
        log(f"✅ Saved official FlowPre set → {rel}", verbose)
    return dirs["base"], manifest


def materialize_official_flowpre_sets(
    promoted_upstreams: Dict[str, str | Path],
    y_scalers: List[str],
    *,
    df_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
    condition_col: str = "type",
    target: str = "init",
    device: str = "auto",
    force: bool = False,
    verbose: bool = True,
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
) -> List[str]:
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     r_train, r_val, r_test) = load_or_create_raw_splits(
        df_name=df_name,
        condition_col=condition_col,
        target=target,
        force=False,
        verbose=verbose,
        split_id=split_id,
    )

    created: List[str] = []
    for tag, promotion_manifest_path in promoted_upstreams.items():
        promotion_manifest = _load_promotion_manifest(promotion_manifest_path)
        run_dir, run_manifest = _run_manifest_to_run_dir(promotion_manifest["source_run_manifest"])
        model_name = str(run_manifest.get("run_id", run_dir.name))

        if verbose:
            log(f"🌀 Encoding X with official FlowPre upstream [{tag}] → {model_name}", verbose)

        X_lat_train, X_lat_val, X_lat_test = transform_X_with_flowpre_model(
            model_name=model_name,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            condition_col=condition_col,
            cols_to_exclude=["post_cleaning_index"],
            device=device,
            verbose=verbose,
        )

        for y_scaler_name in y_scalers:
            suffix = _official_flowpre_suffix(tag, y_scaler_name)
            if not force and _official_flowpre_bundle_has_core_files(tag, y_scaler_name, split_id=split_id):
                if verbose:
                    log(f"📦 Exists → data/sets/official/{split_id}/scaled/{suffix}", verbose)
                created.append(suffix)
                continue

            base_dir, _ = save_official_flowpre_scaled_bundle(
                X_lat_train,
                X_lat_val,
                X_lat_test,
                y_train,
                y_val,
                y_test,
                r_train,
                r_val,
                r_test,
                y_scaler_name=y_scaler_name,
                meta_tag=tag,
                model_name=model_name,
                promotion_manifest_path=promotion_manifest_path,
                flowpre_source_id=str(promotion_manifest["source_id"]),
                condition_col=condition_col,
                split_id=split_id,
                source_dataset_name=df_name,
                verbose=verbose,
            )
            created.append(base_dir.name)
    return created


def _sample_flowgen_train_only_raw(
    flowgen_promotion_manifest_path: str | Path,
    *,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    condition_col: str = "type",
    target_majority: str = "train_majority",
    device: str = "auto",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], int]:
    import torch

    if target_majority != "train_majority":
        raise ValueError("Only target_majority='train_majority' is supported in canonical F6 materialization.")

    model, device_obj, _, _ = _load_flowgen_model_from_promotion(
        flowgen_promotion_manifest_path,
        x_reference=X_train,
        y_reference=y_train,
        condition_col=condition_col,
        device=device,
    )

    x_cols = [c for c in X_train.columns if c not in ("post_cleaning_index", condition_col)]
    y_cols = [c for c in y_train.columns if c != "post_cleaning_index"]
    counts = X_train[condition_col].astype(int).value_counts().sort_index()
    majority = int(counts.max())
    next_index = int(X_train["post_cleaning_index"].max()) + 1

    x_chunks: list[pd.DataFrame] = []
    y_chunks: list[pd.DataFrame] = []
    added_by_class: dict[str, int] = {}

    for cls, cls_count in counts.items():
        n_to_generate = max(0, majority - int(cls_count))
        added_by_class[str(int(cls))] = int(n_to_generate)
        if n_to_generate <= 0:
            continue

        if verbose:
            log(f"🧪 FlowGen sampling class={int(cls)} count={n_to_generate}", verbose)

        remaining = n_to_generate
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        attempts = 0
        while remaining > 0 and attempts < 8:
            attempts += 1
            class_labels = torch.full((remaining,), int(cls), dtype=torch.long, device=device_obj)
            with torch.no_grad():
                xs_c, ys_c = model.sample_xy(remaining, class_labels)
            xs_np = xs_c.detach().cpu().numpy()
            ys_np = ys_c.detach().cpu().numpy()
            finite_mask = np.isfinite(xs_np).all(axis=1) & np.isfinite(ys_np).all(axis=1)
            if finite_mask.any():
                x_rows.append(xs_np[finite_mask])
                y_rows.append(ys_np[finite_mask])
                remaining -= int(finite_mask.sum())

        if remaining > 0:
            raise RuntimeError(
                f"FlowGen could not produce enough finite synthetic rows for class={int(cls)}. "
                f"Missing {remaining} rows after {attempts} attempts."
            )

        x_block = np.concatenate(x_rows, axis=0)[:n_to_generate]
        y_block = np.concatenate(y_rows, axis=0)[:n_to_generate]
        indices = np.arange(next_index, next_index + n_to_generate, dtype=int)
        next_index += n_to_generate

        X_syn = pd.DataFrame(x_block, columns=x_cols)
        X_syn.insert(0, condition_col, int(cls))
        X_syn.insert(0, "post_cleaning_index", indices)

        y_syn = pd.DataFrame(y_block, columns=y_cols)
        y_syn.insert(0, "post_cleaning_index", indices)

        x_chunks.append(X_syn)
        y_chunks.append(y_syn)

    if not x_chunks:
        X_empty = X_train.iloc[0:0].copy()
        y_empty = y_train.iloc[0:0].copy()
        return X_empty, y_empty, added_by_class, majority

    X_synth = pd.concat(x_chunks, axis=0).sort_values("post_cleaning_index").reset_index(drop=True)
    y_synth = pd.concat(y_chunks, axis=0).sort_values("post_cleaning_index").reset_index(drop=True)
    return X_synth, y_synth, added_by_class, majority


def materialize_official_flowgen_augmented_set(
    *,
    flowgen_work_base_manifest_path: str | Path,
    flowgen_promotion_manifest_path: str | Path,
    y_scaler_name: str,
    condition_col: str = "type",
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    source_dataset_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
    target: str = "init",
    force: bool = False,
    device: str = "auto",
    verbose: bool = True,
) -> Tuple[Path, dict]:
    from training.utils import load_scaled_sets

    flowgen_work_base_promotion = _load_promotion_manifest(flowgen_work_base_manifest_path)
    flowgen_promotion = _load_promotion_manifest(flowgen_promotion_manifest_path)

    meta_tag = str(flowgen_work_base_promotion.get("branch_id") or "")
    if meta_tag not in FLOWGEN_WORK_BASE_IDS:
        raise ValueError(
            "Canonical FlowGen synthetic materialization requires a promoted work base "
            f"in {sorted(FLOWGEN_WORK_BASE_IDS)}, got '{meta_tag}'."
        )

    flowgen_work_base_source_id = str(flowgen_work_base_promotion["source_id"])
    flowgen_source_id = str(flowgen_promotion["source_id"])
    synthetic_policy_id = f"flowgen__{flowgen_source_id}__train_only__v1"

    dirs = _official_flowgen_dirs(meta_tag, y_scaler_name, synthetic_policy_id, split_id=split_id)
    _ensure_dirs(dirs)
    suffix = _official_flowgen_suffix(meta_tag, y_scaler_name, synthetic_policy_id)

    if not force and (dirs["meta"] / "manifest.json").exists():
        manifest = _load_json_dict(dirs["meta"] / "manifest.json")
        if manifest:
            if verbose:
                rel = dirs["base"].relative_to(Path(ROOT_PATH))
                log(f"📦 Exists → {rel}", verbose)
            return dirs["base"], manifest

    source_flowpre_folder = _official_flowpre_suffix(meta_tag, y_scaler_name)
    source_flowpre_base = _official_scaled_bundle_dir(source_flowpre_folder, split_id=split_id)
    if force or not (source_flowpre_base / "meta" / "manifest.json").exists():
        materialize_official_flowpre_sets(
            {meta_tag: flowgen_work_base_manifest_path},
            [y_scaler_name],
            df_name=source_dataset_name,
            condition_col=condition_col,
            target=target,
            device=device,
            force=force,
            verbose=verbose,
            split_id=split_id,
        )

    flowpre_manifest = _load_bundle_manifest(source_flowpre_base)
    X_train_fp, X_val_fp, X_test_fp, y_train_fp, y_val_fp, y_test_fp = load_scaled_sets(
        source_flowpre_folder,
        allow_legacy=False,
        verbose=False,
    )

    source_y_scaler_path = _resolve_repo_path(flowpre_manifest["scaler_artifacts"]["y"])
    y_scaler = joblib.load(source_y_scaler_path)

    (X_train_raw, X_val_raw, X_test_raw,
     y_train_raw, y_val_raw, y_test_raw,
     r_train_raw, r_val_raw, r_test_raw) = load_or_create_raw_splits(
        df_name=source_dataset_name,
        condition_col=condition_col,
        target=target,
        force=False,
        verbose=verbose,
        split_id=split_id,
    )

    X_synth_raw, y_synth_raw, added_by_class, majority = _sample_flowgen_train_only_raw(
        flowgen_promotion_manifest_path,
        X_train=X_train_raw,
        y_train=y_train_raw,
        condition_col=condition_col,
        device=device,
        verbose=verbose,
    )

    run_dir, run_manifest = _run_manifest_to_run_dir(flowgen_work_base_promotion["source_run_manifest"])
    flowpre_model_name = str(run_manifest.get("run_id", run_dir.name))
    if len(X_synth_raw) > 0:
        X_synth_lat = encode_dataframe_with_flowpre_model(
            flowpre_model_name,
            X_synth_raw,
            reference_df=X_train_raw,
            condition_col=condition_col,
            cols_to_exclude=["post_cleaning_index"],
            device=device,
            verbose=verbose,
        )
    else:
        X_synth_lat = X_train_fp.iloc[0:0].copy()

    y_cols = [c for c in y_train_raw.columns if c != "post_cleaning_index"]
    y_synth_scaled = y_synth_raw.copy()
    if len(y_synth_scaled) > 0:
        y_synth_scaled[y_cols] = y_scaler.transform(y_synth_scaled[y_cols])

    X_train_aug = X_train_fp.copy()
    X_train_aug.insert(2, "is_synth", False)
    y_train_aug = y_train_fp.copy()
    y_train_aug.insert(1, "is_synth", False)

    if len(X_synth_lat) > 0:
        X_synth_lat = X_synth_lat.copy()
        X_synth_lat.insert(2, "is_synth", True)
        y_synth_scaled = y_synth_scaled.copy()
        y_synth_scaled.insert(1, "is_synth", True)
        X_train_aug = pd.concat([X_train_aug, X_synth_lat], axis=0, ignore_index=True)
        y_train_aug = pd.concat([y_train_aug, y_synth_scaled], axis=0, ignore_index=True)

    X_train_aug = X_train_aug.sort_values("post_cleaning_index").reset_index(drop=True)
    y_train_aug = y_train_aug.sort_values("post_cleaning_index").reset_index(drop=True)

    X_train_aug.to_csv(dirs["X"] / f"{suffix}_X_train.csv", index=False)
    X_val_fp.to_csv(dirs["X"] / f"{suffix}_X_val.csv", index=False)
    X_test_fp.to_csv(dirs["X"] / f"{suffix}_X_test.csv", index=False)

    y_train_aug.to_csv(dirs["y"] / f"{suffix}_y_train.csv", index=False)
    y_val_fp.to_csv(dirs["y"] / f"{suffix}_y_val.csv", index=False)
    y_test_fp.to_csv(dirs["y"] / f"{suffix}_y_test.csv", index=False)

    r_train_aug = r_train_raw.copy()
    r_train_aug["is_synth"] = False
    if len(X_synth_raw) > 0:
        synth_removed = pd.DataFrame(columns=r_train_aug.columns)
        synth_removed["post_cleaning_index"] = X_synth_raw["post_cleaning_index"].values
        if "split" in synth_removed.columns:
            synth_removed["split"] = "train"
        if "split_id" in synth_removed.columns:
            synth_removed["split_id"] = split_id
        if "split_row_id" in synth_removed.columns:
            synth_removed["split_row_id"] = [f"synth_train_{i}" for i in range(len(synth_removed))]
        synth_removed["is_synth"] = True
        r_train_aug = pd.concat([r_train_aug, synth_removed], axis=0, ignore_index=True, sort=False)
    r_val_aug = r_val_raw.copy()
    r_test_aug = r_test_raw.copy()

    r_train_aug.to_csv(dirs["removed"] / f"{suffix}_r_train.csv", index=False)
    r_val_aug.to_csv(dirs["removed"] / f"{suffix}_r_val.csv", index=False)
    r_test_aug.to_csv(dirs["removed"] / f"{suffix}_r_test.csv", index=False)

    copied_scaler_path = dirs["scalers"] / source_y_scaler_path.name
    shutil.copy2(source_y_scaler_path, copied_scaler_path)

    counts_by_split = dict(flowpre_manifest.get("counts_by_split", {}))
    counts_by_class = json.loads(json.dumps(flowpre_manifest.get("counts_by_class", {}), ensure_ascii=True))
    counts_by_split["train"] = int(len(X_train_aug))
    counts_by_class["train"] = {
        str(cls): int(count)
        for cls, count in X_train_aug[condition_col].astype(int).value_counts().sort_index().items()
    }

    manifest = build_canonical_derived_manifest(
        dataset_name=build_canonical_dataset_name(
            x_transform=f"flowpre_{meta_tag}",
            y_transform=y_scaler_name,
            synthetic_policy="flowgen",
        ),
        dataset_level_axes={
            "x_transform": f"flowpre_{meta_tag}",
            "y_transform": y_scaler_name,
            "synthetic_policy": "flowgen",
        },
        split_id=split_id,
        cleaning_policy_id=str(flowpre_manifest["cleaning_policy_id"]),
        source_dataset_manifest_path=flowpre_manifest["source_dataset_manifest"],
        source_split_manifest_path=flowpre_manifest.get("source_split_manifest"),
        source_cleaning_manifest_path=flowpre_manifest.get("source_cleaning_manifest"),
        source_manifest=flowpre_manifest,
        support_status="materialized_now",
        synthetic_policy_id=synthetic_policy_id,
        train_only_mutations=["synthetic_policy", "is_synth"],
        artifacts={
            "storage_name": suffix,
            "X": {
                "train": dirs["X"] / f"{suffix}_X_train.csv",
                "val": dirs["X"] / f"{suffix}_X_val.csv",
                "test": dirs["X"] / f"{suffix}_X_test.csv",
            },
            "y": {
                "train": dirs["y"] / f"{suffix}_y_train.csv",
                "val": dirs["y"] / f"{suffix}_y_val.csv",
                "test": dirs["y"] / f"{suffix}_y_test.csv",
            },
            "removed": {
                "train": dirs["removed"] / f"{suffix}_r_train.csv",
                "val": dirs["removed"] / f"{suffix}_r_val.csv",
                "test": dirs["removed"] / f"{suffix}_r_test.csv",
            },
        },
        scaler_artifacts={"y": copied_scaler_path},
        upstream_model_manifests=[flowgen_work_base_manifest_path, flowgen_promotion_manifest_path],
        extra_manifest_fields={
            "source_dataset_name": flowpre_manifest.get("dataset_name"),
            "counts_by_split": counts_by_split,
            "counts_by_class": counts_by_class,
            "branch_id": meta_tag,
            "flowgen_work_base_id": meta_tag,
            "flowgen_work_base_source_id": flowgen_work_base_source_id,
            "flowpre_source_id": flowgen_work_base_source_id,
            "flowgen_source_id": flowgen_source_id,
            "paired_flowpre_source_id": flowgen_promotion.get("paired_flowpre_source_id"),
            "source_flowpre_bundle_name": source_flowpre_folder,
            "synthetic_target_policy": {
                "scope": "train_only",
                "target": "equalize_to_train_majority",
                "target_count_per_class": majority,
            },
            "added_train_rows_by_class": added_by_class,
            "condition_col": condition_col,
            "is_synth_scope": "train_only",
        },
    )
    dump_json(manifest, dirs["meta"] / "manifest.json")

    if verbose:
        rel = dirs["base"].relative_to(Path(ROOT_PATH))
        log(f"✅ Saved official FlowGen augmented set → {rel}", verbose)
    return dirs["base"], manifest


def materialize_official_flowgen_augmented_sets(
    *,
    flowgen_work_base_manifest_path: str | Path,
    flowgen_promotion_manifest_path: str | Path,
    y_scalers: List[str],
    condition_col: str = "type",
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    source_dataset_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
    target: str = "init",
    force: bool = False,
    device: str = "auto",
    verbose: bool = True,
) -> List[str]:
    created: List[str] = []
    for y_scaler_name in y_scalers:
        base_dir, _ = materialize_official_flowgen_augmented_set(
            flowgen_work_base_manifest_path=flowgen_work_base_manifest_path,
            flowgen_promotion_manifest_path=flowgen_promotion_manifest_path,
            y_scaler_name=y_scaler_name,
            condition_col=condition_col,
            split_id=split_id,
            source_dataset_name=source_dataset_name,
            target=target,
            force=force,
            device=device,
            verbose=verbose,
        )
        created.append(base_dir.name)
    return created

