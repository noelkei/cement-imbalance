import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data.cleaning import (
    clean_univariate_bootstrap_density_weighted,
    clean_multivariate_outliers_iforest_set,
    load_processed_data
)
from data.utils import ROOT_PATH, log, load_column_mapping_by_group, load_type_mapping
from scipy.stats import ks_2samp

OFFICIAL_SPLITS_ROOT = ROOT_PATH / "data" / "splits" / "official"
DEFAULT_OFFICIAL_SPLIT_ID = "init_temporal_processed_v1"
DEFAULT_SPLIT_TARGET = "init"
DEFAULT_DESIRED_CLASS_FLOOR = 100
DEFAULT_MINIMUM_CLASS_FLOOR = 50
DEFAULT_TARGET_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
LEGACY_SPLIT_ARTIFACTS = [
    "data/splits/df_X_input_init.csv",
    "data/splits/df_y_input_init.csv",
    "data/splits/df_removed_input_init.csv",
    "data.sets.load_or_create_raw_splits",
]

def load_cleaned_data(target: str = "init", verbose: bool = True, force: bool = False) -> pd.DataFrame:
    """
    Load cleaned dataset with outliers removed. If it doesn't exist or force=True, recomputes and saves it.

    Args:
        target (str): Target variable to exclude from predictors.
        verbose (bool): Whether to print logs.
        force (bool): Whether to regenerate cleaned file even if it exists.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for downstream modeling.
    """
    cleaned_path = ROOT_PATH / "data" / "cleaned"
    cleaned_path.mkdir(parents=True, exist_ok=True)
    cleaned_file = cleaned_path / f"df_cleaned_{target}.csv"

    if cleaned_file.exists() and not force:
        log(f"📦 Loading cleaned data from: {cleaned_file}", verbose)
        return pd.read_csv(cleaned_file)

    log("🔄 Cleaned data not found or force=True. Running cleaning pipeline...", verbose)

    # Column groups
    _, grouped_mapping = load_column_mapping_by_group(verbose=verbose)

    groups = {
        "chem": list(grouped_mapping.get("chem", {}).values()),
        "phase": list(grouped_mapping.get("phase", {}).values())
    }

    ratio_cols = list(grouped_mapping.get("chem_ratios", {}).values()) + \
                 list(grouped_mapping.get("phase_ratios", {}).values())

    # Load processed
    df_normalized = load_processed_data(verbose=verbose, force=force)

    if target == "init":
        df_normalized = df_normalized.drop(columns=["end", "6h"], errors="ignore")

    # Define columns
    compositional_cols = [col for cols in groups.values() for col in cols]
    phase_cols = [col for col in df_normalized.columns if col.startswith("ilr_phase")]
    visually_considered_unnecessary = ["blaine", "blaines", "water"]
    target_col = [target]

    excluded_cols = set(
        compositional_cols +
        ratio_cols +
        ["date", "type", "process"] +
        visually_considered_unnecessary +
        target_col
    )

    filter_cols = [
        col for col in df_normalized.columns
        if not (
            col.startswith("ilr_phase") or
            col.startswith("norm_") or
            "sum" in col or
            col in excluded_cols
        )
    ]

    # Thresholds for univariate outlier detection
    all_filter_cols = filter_cols + phase_cols
    n_total = len(all_filter_cols)
    alpha = (1 - 0.95) / n_total
    thresholds = np.linspace(1 - alpha, 1, 5).tolist()
    only_threshold = min(thresholds)

    # Univariate filter
    df_by_type = {
        t: df_normalized[df_normalized["type"] == t].copy()
        for t in df_normalized["type"].unique()
    }

    filtered_dfs_masked = clean_univariate_bootstrap_density_weighted(
        dataframes_dict=df_by_type,
        filter_columns=filter_cols,
        phase_columns=phase_cols,
        thresholds=thresholds,
        n_bins=200,
        dtype="float64",
        verbose=False,
        bootstrap="auto",
        n_bootstrap="auto",
        only_threshold=only_threshold,
        pect_prominence=0.05,
        weight=None,
        filtering=True
    )

    df_with_clusters = pd.concat(filtered_dfs_masked.values()).sort_index()
    df_with_clusters.index.name = "original_index"

    df_uni = df_normalized.copy()
    df_uni.index.name = "original_index"
    df_uni["uni_removal_flag"] = (~df_uni.index.isin(df_with_clusters.index)).astype(int)

    # Multivariate outlier detection
    contamination_values = np.linspace(0.0005, 0.01, 20).round(6).tolist()
    df_by_type_iforest = {
        t: df_uni[df_uni["type"] == t].copy()
        for t in df_uni["type"].unique()
    }

    iforest_results = clean_multivariate_outliers_iforest_set(
        dataframes_dict=df_by_type_iforest,
        columns_to_check=all_filter_cols,
        contamination_values=contamination_values,
        filtering=False,
        verbose=False
    )

    df_iforest_flagged = pd.concat(iforest_results.values()).sort_index()

    df_final = df_uni.copy()
    df_final["Outlier_IForest"] = df_iforest_flagged["Outlier_IForest"].reindex(df_final.index).fillna(False)

    # Verify consistency with original
    assert df_final.drop(columns=["uni_removal_flag", "Outlier_IForest"]).equals(
        df_normalized.loc[df_final.index]
    ), "⚠️ Mismatch detected between df_final and df_normalized in original data."

    # OR removal
    df_final["or_removal_flag"] = (
        (df_final["uni_removal_flag"] == 1) |
        (df_final["Outlier_IForest"] == True)
    ).astype(int)

    df_or_filtered = df_final[df_final["or_removal_flag"] == 0].copy()

    # Save
    df_or_filtered.to_csv(cleaned_file, index=False)
    log(f"\n✅ Cleaned dataset saved to: {cleaned_file}", verbose)
    log(f"📐 Final shape: {df_or_filtered.shape}", verbose)

    return df_or_filtered

def prepare_splits(target: str = "init", verbose: bool = True, force: bool = False):
    """
    Legacy helper that materializes input bundles under `data/splits/`.

    Important:
        This function does NOT define the canonical official temporal split.
        It survives only as a legacy input-bundle builder for downstream code
        that still depends on top-level CSV artifacts inside `data/splits/`.

    Load cleaned data, reset and rename index, split into FlowPre input, target, and removed columns.

    Returns:
        df_flowpre_input (pd.DataFrame): FlowPre model inputs
        df_target_input (pd.DataFrame): Target variable mapping
        df_removed_input (pd.DataFrame): Removed metadata and unused features
    """

    # Setup paths
    splits_path = ROOT_PATH / "data" / "splits"
    splits_path.mkdir(parents=True, exist_ok=True)

    input_file = splits_path / f"df_X_input_{target}.csv"
    target_file = splits_path / f"df_y_input_{target}.csv"
    removed_file = splits_path / f"df_removed_input_{target}.csv"

    if all(f.exists() for f in [input_file, target_file, removed_file]) and not force:
        log(f"📦 FlowPre files already exist for target='{target}'. Loading from disk...", verbose)
        df_X_input = pd.read_csv(input_file)
        df_y_input = pd.read_csv(target_file)
        df_removed_input = pd.read_csv(removed_file)
        log(f"Files loaded!", verbose)
        return df_X_input, df_y_input, df_removed_input

    log("🔄 Files not found or force=True. Preparing from cleaned data...", verbose)

    # Load cleaned data
    df_cleaned = load_cleaned_data(target=target, verbose=verbose, force=force)

    # --- Initial info ---
    initial_cols = df_cleaned.columns.tolist()
    initial_shape = df_cleaned.shape
    log(f"\n📊 Initial shape: {initial_shape}", verbose)
    log(f"📑 Initial columns ({len(initial_cols)}):", verbose)
    log(initial_cols, verbose)

    # Load mapping
    type_to_index = load_type_mapping(verbose=verbose)

    # Apply mapping to "type"
    df_cleaned["type"] = df_cleaned["type"].map(type_to_index)
    if df_cleaned["type"].isna().any():
        raise ValueError("🚨 Some 'type' values couldn't be mapped. Check your YAML mapping.")

    # Optional: sort by type
    df_cleaned = df_cleaned.sort_values("type").reset_index(drop=True)

    # Reset index and name it
    df_cleaned = df_cleaned.reset_index().rename(columns={"index": "post_cleaning_index"})

    # --- Define columns ---
    target_col = target
    condition_col = "type"

    # Columns to remove
    cols_to_remove = [
        "date", "process", "uni_removal_flag", "Outlier_IForest", "or_removal_flag",
        *[col for col in df_cleaned.columns if col.startswith("chem_")],
        *[col for col in df_cleaned.columns if col.startswith("phase_")],
        "sum", "sum_chem", "sum_phase",
        *[col for col in df_cleaned.columns if col.startswith("norm_")],
    ]

    # Final FlowPre features (excluding target, condition, and removed cols)
    removed = set(cols_to_remove + [target_col])
    excluded = removed.union({"post_cleaning_index"})
    flowpre_cols = [col for col in df_cleaned.columns if col not in excluded and col != condition_col]

    # Ensure ordered consistency with df_cleaned for removed input
    ordered_removed_cols = [col for col in df_cleaned.columns if col in removed]

    # --- Create aligned DataFrames ---
    df_X_input = df_cleaned[["post_cleaning_index", condition_col] + flowpre_cols].copy()
    df_y_input = df_cleaned[["post_cleaning_index", target_col]].copy()
    df_removed_input = df_cleaned[["post_cleaning_index"] + ordered_removed_cols].copy()

    # --- Rename selected columns in all DataFrames if they exist ---
    rename_map = {
        "90": "90um_mesh",
        "902": "90um",
        "753": "75um",
        "454": "45um",
        "305": "30um"
    }

    for df in [df_X_input, df_y_input, df_removed_input]:
        rename_cols = {old: new for old, new in rename_map.items() if old in df.columns}
        df.rename(columns=rename_cols, inplace=True)

    # --- Rescale 'blaine' and 'blaines' in all DataFrames if they exist ---
    for df in [df_X_input, df_y_input, df_removed_input]:
        for col in ["blaine", "blaines"]:
            if col in df.columns:
                df[col] = df[col] / 100
    # --- Rescale 'init' to hours in all DataFrames if they exist ---
    for df in [df_X_input, df_y_input, df_removed_input]:
        for col in ["init"]:
            if col in df.columns:
                df[col] = df[col] / 60
    # Save outputs
    df_X_input.to_csv(input_file, index=False)
    df_y_input.to_csv(target_file, index=False)
    df_removed_input.to_csv(removed_file, index=False)

    # --- Final info ---
    log(f"\n🧹 Columns removed ({len(cols_to_remove)}):", verbose)
    log(sorted(cols_to_remove), verbose)

    log(f"\n✅ FlowPre input columns ({len(flowpre_cols)}):", verbose)
    log(flowpre_cols, verbose)

    # Log renamed columns
    renamed_cols_logged = {old: new for old, new in rename_map.items() if
                           any(old in df.columns for df in [df_X_input, df_y_input, df_removed_input])}
    if renamed_cols_logged:
        log(f"\n🔁 Renamed columns ({len(renamed_cols_logged)}):", verbose)
        for old, new in renamed_cols_logged.items():
            log(f"   {old} → {new}", verbose)

    # Log rescaled columns
    rescaled_cols = [col for col in ["blaine", "blaines", "init"] if
                     any(col in df.columns for df in [df_X_input, df_y_input, df_removed_input])]
    if rescaled_cols:
        blaine_cols = [col for col in ["blaine", "blaines"] if
                     any(col in df.columns for df in [df_X_input, df_y_input, df_removed_input])]
        init_col = [col for col in ["init"] if
                       any(col in df.columns for df in [df_X_input, df_y_input, df_removed_input])]

        log(f"\n⚖️ Rescaled columns (÷100): {blaine_cols}", verbose)
        log(f"\n⚖️ Rescaled columns (÷60): {init_col}", verbose)

    log("\n🔹 Final shapes:", verbose)
    log(f"FlowPre: {df_X_input.shape}", verbose)
    log(f"Target: {df_y_input.shape}", verbose)
    log(f"Removed: {df_removed_input.shape}", verbose)

    return df_X_input, df_y_input, df_removed_input


def _validate_official_split_source_hash(
    manifest: Dict[str, Any],
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
) -> str:
    source_file_rel = manifest.get("source_file", "data/processed/df_processed.csv")
    source_file = ROOT_PATH / source_file_rel
    if not source_file.exists():
        raise FileNotFoundError(
            f"Official split source for '{split_id}' is missing: {source_file}"
        )

    current_sha = _sha256_file(source_file)
    expected_sha = manifest.get("source_file_sha256")
    if expected_sha and current_sha != expected_sha:
        raise ValueError(
            "The processed source backing the official split has changed since F2. "
            f"Expected sha256={expected_sha}, got sha256={current_sha}."
        )
    return current_sha


def load_official_assigned_source_frame(
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    target: str = DEFAULT_SPLIT_TARGET,
    verbose: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Load the canonical processed source and attach the official split assignment
    row by row.

    This is the canonical F3 input surface:
      - processed_pre_statistical_cleaning source,
      - frozen by the F2 manifest hash,
      - with split metadata attached,
      - before any learned cleaning is fit.
    """
    manifest, assignments = load_official_split(split_id=split_id, verbose=verbose)
    current_sha = _validate_official_split_source_hash(manifest, split_id=split_id)

    source_df = load_split_source_frame(target=target, verbose=verbose, force=False)
    source_df = source_df.copy()

    assignments_cols = ["split_row_id", "source_row_number", "split"]
    merged = source_df.merge(
        assignments[assignments_cols],
        on=["split_row_id", "source_row_number"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(source_df):
        raise ValueError(
            f"Official split '{split_id}' assignments do not cover the full source frame: "
            f"{len(merged)} merged rows vs {len(source_df)} source rows."
        )

    merged = merged.sort_values(["date", "source_row_number"]).reset_index(drop=True)
    manifest = dict(manifest)
    manifest["source_file_sha256_verified"] = current_sha
    return manifest, merged


def partition_official_source_frame(
    assigned_source_df: pd.DataFrame,
    split_col: str = "split",
) -> dict[str, pd.DataFrame]:
    required_cols = {split_col, "split_row_id", "source_row_number"}
    missing = sorted(required_cols.difference(assigned_source_df.columns))
    if missing:
        raise ValueError(f"Assigned source frame is missing required columns: {missing}")

    partitions: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        split_df = (
            assigned_source_df[assigned_source_df[split_col] == split_name]
            .copy()
            .sort_values(["date", "source_row_number"])
            .reset_index(drop=True)
        )
        partitions[split_name] = split_df
    return partitions


def _official_split_dir(split_id: str = DEFAULT_OFFICIAL_SPLIT_ID) -> Path:
    return OFFICIAL_SPLITS_ROOT / split_id


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _smd_abs(train: pd.Series, comp: pd.Series) -> float:
    train_vals = pd.to_numeric(train, errors="coerce").dropna().to_numpy()
    comp_vals = pd.to_numeric(comp, errors="coerce").dropna().to_numpy()
    if train_vals.size == 0 or comp_vals.size == 0:
        return float("nan")

    train_mean = float(train_vals.mean())
    comp_mean = float(comp_vals.mean())
    train_std = float(train_vals.std(ddof=1)) if train_vals.size > 1 else 0.0
    comp_std = float(comp_vals.std(ddof=1)) if comp_vals.size > 1 else 0.0
    pooled = np.sqrt((train_std ** 2 + comp_std ** 2) / 2.0)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float(abs(train_mean - comp_mean) / pooled)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = _safe_divide(p, p.sum())
    q = _safe_divide(q, q.sum())
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def load_split_source_frame(
    target: str = DEFAULT_SPLIT_TARGET,
    verbose: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """
    Load the earliest practical source available for the official split contract.

    For F2 this intentionally uses the processed stage, not the cleaned stage,
    because `load_cleaned_data()` already applies statistical cleaning fit on the
    full dataset and is therefore not acceptable as the canonical source of truth
    for the temporal split contract.
    """
    df = load_processed_data(verbose=verbose, force=force).copy()
    required_cols = {"date", "type", target}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(f"Official split source is missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        n_bad = int(df["date"].isna().sum())
        raise ValueError(f"Official split source contains {n_bad} rows with invalid dates.")

    df = df.reset_index(drop=True)
    df["source_row_number"] = np.arange(len(df), dtype=int)
    df["split_row_id"] = df["source_row_number"].map(lambda i: f"processed_row_{i:05d}")
    return df


def _build_daily_class_matrix(
    source_df: pd.DataFrame,
    date_col: str = "date",
    type_col: str = "type",
) -> pd.DataFrame:
    return (
        source_df.groupby([date_col, type_col], observed=False)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )


def select_official_temporal_boundaries(
    source_df: pd.DataFrame,
    date_col: str = "date",
    type_col: str = "type",
    desired_class_floor: int = DEFAULT_DESIRED_CLASS_FLOOR,
    minimum_class_floor: int = DEFAULT_MINIMUM_CLASS_FLOOR,
    target_fractions: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Select a chronological contiguous split over whole dates with:
      1. no randomness,
      2. `test` as the most recent block,
      3. `val` as the block immediately before `test`,
      4. desired class floor 100 in val/test when feasible,
      5. fallback class floor 50 only if needed.
    """
    target_fractions = target_fractions or DEFAULT_TARGET_FRACTIONS

    by_date = _build_daily_class_matrix(source_df, date_col=date_col, type_col=type_col)
    classes = by_date.columns.tolist()
    if len(classes) < 2:
        raise ValueError("Official split selection requires at least two classes.")

    counts = by_date.to_numpy(dtype=np.int64)
    cum_counts = counts.cumsum(axis=0)
    total_counts = cum_counts[-1]
    total_n = int(total_counts.sum())
    n_dates = len(by_date.index)

    if n_dates < 3:
        raise ValueError("Official split selection requires at least three unique dates.")

    overall_props = total_counts / total_n
    desired_floor_met = False
    best_candidate = None

    for active_floor in (desired_class_floor, minimum_class_floor):
        floor_best = None
        feasible_candidates = 0

        for train_end_idx in range(n_dates - 2):
            train_counts = cum_counts[train_end_idx]
            candidate_val_end = np.arange(train_end_idx + 1, n_dates - 1)

            val_counts = cum_counts[candidate_val_end] - train_counts
            test_counts = total_counts - cum_counts[candidate_val_end]

            feasible = (val_counts.min(axis=1) >= active_floor) & (test_counts.min(axis=1) >= active_floor)
            if not np.any(feasible):
                continue

            feasible_candidates += int(feasible.sum())
            candidate_val_end = candidate_val_end[feasible]
            val_counts = val_counts[feasible]
            test_counts = test_counts[feasible]

            train_n = int(train_counts.sum())
            val_n = val_counts.sum(axis=1).astype(float)
            test_n = test_counts.sum(axis=1).astype(float)

            size_penalty = (
                np.abs(train_n / total_n - target_fractions["train"])
                + np.abs(val_n / total_n - target_fractions["val"])
                + np.abs(test_n / total_n - target_fractions["test"])
            )
            symmetry_penalty = 0.20 * np.abs(val_n / total_n - test_n / total_n)

            val_props = val_counts / val_n[:, None]
            test_props = test_counts / test_n[:, None]
            mix_penalty = (
                np.max(np.abs(val_props - overall_props), axis=1)
                + np.max(np.abs(test_props - overall_props), axis=1)
            )

            total_score = size_penalty + symmetry_penalty + 0.05 * mix_penalty
            best_idx = int(np.argmin(total_score))
            score = float(total_score[best_idx])

            if floor_best is None or score < floor_best["selection_score"]:
                val_end_idx = int(candidate_val_end[best_idx])
                floor_best = {
                    "selection_score": score,
                    "active_class_floor": int(active_floor),
                    "desired_class_floor": int(desired_class_floor),
                    "minimum_class_floor": int(minimum_class_floor),
                    "fallback_activated": bool(active_floor != desired_class_floor),
                    "meets_desired_class_floor": bool(active_floor == desired_class_floor),
                    "num_feasible_candidates": int(feasible_candidates),
                    "type_labels": [str(c) for c in classes],
                    "train_end_date": by_date.index[train_end_idx],
                    "val_start_date": by_date.index[train_end_idx + 1],
                    "val_end_date": by_date.index[val_end_idx],
                    "test_start_date": by_date.index[val_end_idx + 1],
                    "test_end_date": by_date.index[-1],
                    "train_counts_by_class": {
                        str(cls): int(cnt) for cls, cnt in zip(classes, train_counts)
                    },
                    "val_counts_by_class": {
                        str(cls): int(cnt) for cls, cnt in zip(classes, val_counts[best_idx])
                    },
                    "test_counts_by_class": {
                        str(cls): int(cnt) for cls, cnt in zip(classes, test_counts[best_idx])
                    },
                    "train_n": train_n,
                    "val_n": int(val_n[best_idx]),
                    "test_n": int(test_n[best_idx]),
                    "train_fraction": train_n / total_n,
                    "val_fraction": float(val_n[best_idx] / total_n),
                    "test_fraction": float(test_n[best_idx] / total_n),
                }

        if floor_best is not None:
            best_candidate = floor_best
            desired_floor_met = active_floor == desired_class_floor
            break

    if best_candidate is None:
        raise RuntimeError(
            "No feasible chronological contiguous split was found even after "
            f"falling back from class floor {desired_class_floor} to {minimum_class_floor}."
        )

    best_candidate["desired_floor_met"] = desired_floor_met
    return best_candidate


def build_official_temporal_split(
    source_df: pd.DataFrame,
    target: str = DEFAULT_SPLIT_TARGET,
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    desired_class_floor: int = DEFAULT_DESIRED_CLASS_FLOOR,
    minimum_class_floor: int = DEFAULT_MINIMUM_CLASS_FLOOR,
    target_fractions: Dict[str, float] | None = None,
    date_col: str = "date",
    type_col: str = "type",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    selection = select_official_temporal_boundaries(
        source_df=source_df,
        date_col=date_col,
        type_col=type_col,
        desired_class_floor=desired_class_floor,
        minimum_class_floor=minimum_class_floor,
        target_fractions=target_fractions,
    )

    train_end = pd.Timestamp(selection["train_end_date"])
    val_end = pd.Timestamp(selection["val_end_date"])

    assignments = source_df.copy()
    assignments["split"] = np.where(
        assignments[date_col] <= train_end,
        "train",
        np.where(assignments[date_col] <= val_end, "val", "test"),
    )
    assignments = assignments.sort_values([date_col, "source_row_number"]).reset_index(drop=True)

    manifest = {
        "split_id": split_id,
        "target": target,
        "status": "official",
        "policy_family": "temporal_contiguous_by_date",
        "date_col": date_col,
        "type_col": type_col,
        "source_stage": "processed_pre_statistical_cleaning",
        "source_file": "data/processed/df_processed.csv",
        "selection_policy": {
            "randomness": "none",
            "assign_whole_dates": True,
            "test_block": "most_recent",
            "val_block": "immediately_before_test",
            "desired_class_floor": int(desired_class_floor),
            "minimum_class_floor": int(minimum_class_floor),
            "active_class_floor": int(selection["active_class_floor"]),
            "fallback_activated": bool(selection["fallback_activated"]),
            "meets_desired_class_floor": bool(selection["meets_desired_class_floor"]),
            "target_fractions": target_fractions or DEFAULT_TARGET_FRACTIONS,
            "val_test_rebalancing": "forbidden",
            "class_comparison_policy": "use_per_class_and_macro_metrics_later_not_split_rebalancing",
        },
        "boundary_dates": {
            "train_end_date": str(pd.Timestamp(selection["train_end_date"]).date()),
            "val_start_date": str(pd.Timestamp(selection["val_start_date"]).date()),
            "val_end_date": str(pd.Timestamp(selection["val_end_date"]).date()),
            "test_start_date": str(pd.Timestamp(selection["test_start_date"]).date()),
            "test_end_date": str(pd.Timestamp(selection["test_end_date"]).date()),
        },
        "counts": {
            "n_rows_total": int(len(assignments)),
            "by_split": {
                "train": int(selection["train_n"]),
                "val": int(selection["val_n"]),
                "test": int(selection["test_n"]),
            },
            "by_class_and_split": {
                "train": selection["train_counts_by_class"],
                "val": selection["val_counts_by_class"],
                "test": selection["test_counts_by_class"],
            },
        },
        "legacy_coexistence": {
            "top_level_data_splits_csvs_are_legacy_input_bundles": True,
            "legacy_artifacts": LEGACY_SPLIT_ARTIFACTS,
        },
        "known_limitations": [
            "F2 fixes the official split contract only.",
            "The source stage is processed_pre_statistical_cleaning by design.",
            "Statistical cleaning and dataset regeneration remain pending for F3.",
        ],
    }
    return assignments, manifest


def validate_official_split(
    assignments: pd.DataFrame,
    manifest: Dict[str, Any],
    date_col: str = "date",
    type_col: str = "type",
) -> Dict[str, Any]:
    required_cols = {"split_row_id", "source_row_number", date_col, type_col, "split"}
    missing = sorted(required_cols.difference(assignments.columns))
    if missing:
        raise ValueError(f"Official split assignments are missing required columns: {missing}")

    split_values = set(assignments["split"].unique())
    if split_values != {"train", "val", "test"}:
        raise ValueError(f"Official split assignments must contain exactly train/val/test, got {split_values}")

    if not assignments["split_row_id"].is_unique:
        raise ValueError("split_row_id is not unique in official split assignments.")
    if not assignments["source_row_number"].is_unique:
        raise ValueError("source_row_number is not unique in official split assignments.")

    by_split = {name: df.copy() for name, df in assignments.groupby("split", sort=False)}
    train_dates = by_split["train"][date_col]
    val_dates = by_split["val"][date_col]
    test_dates = by_split["test"][date_col]

    order_ok = (
        train_dates.max() < val_dates.min()
        and val_dates.max() < test_dates.min()
    )
    if not order_ok:
        raise ValueError("Official split is not strictly ordered in time.")

    date_split_uniqueness = assignments.groupby(date_col)["split"].nunique().max()
    if int(date_split_uniqueness) != 1:
        raise ValueError("At least one date has been split across multiple partitions.")

    class_counts = (
        assignments.groupby(["split", type_col], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    min_counts = class_counts.groupby("split")["count"].min().to_dict()
    active_floor = int(manifest["selection_policy"]["active_class_floor"])
    if min_counts.get("val", 0) < active_floor or min_counts.get("test", 0) < active_floor:
        raise ValueError("Official split does not satisfy the active class floor.")

    return {
        "no_overlap": True,
        "ordered_temporally": True,
        "no_split_dates": True,
        "min_class_count_by_split": {k: int(v) for k, v in min_counts.items()},
        "active_class_floor": active_floor,
        "meets_desired_class_floor": bool(manifest["selection_policy"]["meets_desired_class_floor"]),
    }


def summarize_official_split(
    assignments: pd.DataFrame,
    manifest: Dict[str, Any],
    date_col: str = "date",
    type_col: str = "type",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts_total = len(assignments)

    split_summary_rows = []
    class_count_rows = []
    desired_floor = int(manifest["selection_policy"]["desired_class_floor"])
    active_floor = int(manifest["selection_policy"]["active_class_floor"])

    for split_name, split_df in assignments.groupby("split", sort=False):
        split_n = len(split_df)
        class_counts = split_df[type_col].value_counts(dropna=False)
        split_summary_rows.append(
            {
                "split": split_name,
                "n_rows": int(split_n),
                "fraction": float(split_n / counts_total),
                "start_date": str(split_df[date_col].min().date()),
                "end_date": str(split_df[date_col].max().date()),
                "n_unique_dates": int(split_df[date_col].nunique()),
                "n_classes": int(split_df[type_col].nunique()),
                "min_class_count": int(class_counts.min()),
                "desired_class_floor": desired_floor,
                "active_class_floor": active_floor,
                "meets_desired_class_floor": bool(class_counts.min() >= desired_floor),
            }
        )

        for cls, count in class_counts.sort_index().items():
            class_count_rows.append(
                {
                    "split": split_name,
                    "type": str(cls),
                    "count": int(count),
                    "proportion": float(count / split_n),
                }
            )

    return pd.DataFrame(split_summary_rows), pd.DataFrame(class_count_rows)


def compute_minimal_split_drift(
    assignments: pd.DataFrame,
    target: str = DEFAULT_SPLIT_TARGET,
    date_col: str = "date",
    type_col: str = "type",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = assignments[assignments["split"] == "train"].copy()
    drift_numeric_rows = []

    numeric_cols = assignments.select_dtypes(include=[np.number]).columns.tolist()
    excluded_numeric = {target, "source_row_number"}
    numeric_feature_cols = [c for c in numeric_cols if c not in excluded_numeric]

    for comparator in ("val", "test"):
        comp_df = assignments[assignments["split"] == comparator].copy()
        for col in numeric_feature_cols:
            train_series = pd.to_numeric(train_df[col], errors="coerce").dropna()
            comp_series = pd.to_numeric(comp_df[col], errors="coerce").dropna()
            if train_series.empty or comp_series.empty:
                continue

            ks_stat, ks_p = ks_2samp(train_series, comp_series)
            drift_numeric_rows.append(
                {
                    "comparator": comparator,
                    "feature": col,
                    "train_mean": float(train_series.mean()),
                    "comp_mean": float(comp_series.mean()),
                    "train_std": float(train_series.std(ddof=1)),
                    "comp_std": float(comp_series.std(ddof=1)),
                    "ks_stat": float(ks_stat),
                    "ks_pvalue": float(ks_p),
                    "smd_abs": _smd_abs(train_series, comp_series),
                }
            )

    drift_numeric = pd.DataFrame(drift_numeric_rows).sort_values(
        ["comparator", "smd_abs", "ks_stat"],
        ascending=[True, False, False],
        ignore_index=True,
    )

    drift_target_rows = []
    for comparator in ("val", "test"):
        comp_df = assignments[assignments["split"] == comparator].copy()
        train_series = pd.to_numeric(train_df[target], errors="coerce").dropna()
        comp_series = pd.to_numeric(comp_df[target], errors="coerce").dropna()
        ks_stat, ks_p = ks_2samp(train_series, comp_series)
        drift_target_rows.append(
            {
                "comparator": comparator,
                "feature": target,
                "train_mean": float(train_series.mean()),
                "comp_mean": float(comp_series.mean()),
                "train_std": float(train_series.std(ddof=1)),
                "comp_std": float(comp_series.std(ddof=1)),
                "train_q10": float(train_series.quantile(0.10)),
                "train_q50": float(train_series.quantile(0.50)),
                "train_q90": float(train_series.quantile(0.90)),
                "comp_q10": float(comp_series.quantile(0.10)),
                "comp_q50": float(comp_series.quantile(0.50)),
                "comp_q90": float(comp_series.quantile(0.90)),
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_p),
                "smd_abs": _smd_abs(train_series, comp_series),
            }
        )

    drift_target = pd.DataFrame(drift_target_rows)

    train_props = train_df[type_col].value_counts(normalize=True).sort_index()
    drift_type_rows = []
    train_counts = train_df[type_col].value_counts().sort_index()

    for comparator in ("val", "test"):
        comp_df = assignments[assignments["split"] == comparator].copy()
        comp_props = comp_df[type_col].value_counts(normalize=True).sort_index()
        comp_counts = comp_df[type_col].value_counts().sort_index()
        all_types = sorted(set(train_props.index).union(comp_props.index))
        train_vector = np.array([train_props.get(t, 0.0) for t in all_types], dtype=float)
        comp_vector = np.array([comp_props.get(t, 0.0) for t in all_types], dtype=float)
        js_div = _js_divergence(train_vector, comp_vector)

        for cls in all_types:
            drift_type_rows.append(
                {
                    "comparator": comparator,
                    "type": str(cls),
                    "train_count": int(train_counts.get(cls, 0)),
                    "comp_count": int(comp_counts.get(cls, 0)),
                    "train_proportion": float(train_props.get(cls, 0.0)),
                    "comp_proportion": float(comp_props.get(cls, 0.0)),
                    "delta_abs_proportion": float(abs(train_props.get(cls, 0.0) - comp_props.get(cls, 0.0))),
                    "js_divergence_vs_train": float(js_div),
                }
            )

    drift_type = pd.DataFrame(drift_type_rows).sort_values(
        ["comparator", "delta_abs_proportion"],
        ascending=[True, False],
        ignore_index=True,
    )
    return drift_numeric, drift_target, drift_type


def _plot_temporal_coverage(assignments: pd.DataFrame, plots_dir: Path, date_col: str = "date") -> None:
    monthly = (
        assignments.assign(month=assignments[date_col].dt.to_period("M").astype(str))
        .groupby(["month", "split"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    monthly.plot(kind="bar", stacked=True, figsize=(14, 5))
    plt.title("Temporal Coverage by Split (Monthly)")
    plt.xlabel("Month")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(plots_dir / "01_temporal_coverage.png", dpi=180)
    plt.close()


def _plot_type_mix(assignments: pd.DataFrame, plots_dir: Path, type_col: str = "type") -> None:
    class_counts = (
        assignments.groupby(["split", type_col], observed=False)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    class_props = class_counts.div(class_counts.sum(axis=1), axis=0)
    class_props.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.title("Type Mix by Split")
    plt.xlabel("Split")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(plots_dir / "02_type_mix_by_split.png", dpi=180)
    plt.close()


def _plot_target_distribution(
    assignments: pd.DataFrame,
    plots_dir: Path,
    target: str = DEFAULT_SPLIT_TARGET,
) -> None:
    plt.figure(figsize=(10, 5))
    for split_name, color in (("train", "#1f77b4"), ("val", "#ff7f0e"), ("test", "#2ca02c")):
        vals = pd.to_numeric(assignments.loc[assignments["split"] == split_name, target], errors="coerce").dropna()
        plt.hist(vals, bins=35, alpha=0.45, density=True, label=split_name, color=color)
    plt.title(f"{target} Distribution by Split")
    plt.xlabel(target)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "03_init_distribution_by_split.png", dpi=180)
    plt.close()


def _plot_top_numeric_drift(drift_numeric: pd.DataFrame, plots_dir: Path) -> None:
    test_df = drift_numeric[drift_numeric["comparator"] == "test"].copy().head(10)
    if test_df.empty:
        return
    test_df = test_df.sort_values("smd_abs", ascending=True)
    plt.figure(figsize=(9, 5))
    plt.barh(test_df["feature"], test_df["smd_abs"], color="#d62728")
    plt.title("Top Numeric Drift vs Train (Test, by |SMD|)")
    plt.xlabel("|SMD|")
    plt.tight_layout()
    plt.savefig(plots_dir / "04_top_numeric_drift_train_vs_test.png", dpi=180)
    plt.close()


def save_official_split_artifacts(
    assignments: pd.DataFrame,
    manifest: Dict[str, Any],
    split_summary: pd.DataFrame,
    class_counts: pd.DataFrame,
    drift_numeric: pd.DataFrame,
    drift_target: pd.DataFrame,
    drift_type: pd.DataFrame,
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
) -> Path:
    split_dir = _official_split_dir(split_id)
    plots_dir = split_dir / "plots"
    split_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    assignments_to_save = assignments[["split_row_id", "source_row_number", "date", "type", "split"]].copy()
    assignments_to_save["date"] = assignments_to_save["date"].dt.strftime("%Y-%m-%d")
    assignments_to_save.to_csv(split_dir / "assignments.csv", index=False)
    split_summary.to_csv(split_dir / "split_summary.csv", index=False)
    class_counts.to_csv(split_dir / "class_counts.csv", index=False)
    drift_numeric.to_csv(split_dir / "drift_numeric.csv", index=False)
    drift_target.to_csv(split_dir / "drift_target_init.csv", index=False)
    drift_type.to_csv(split_dir / "drift_type.csv", index=False)

    with open(split_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    _plot_temporal_coverage(assignments, plots_dir)
    _plot_type_mix(assignments, plots_dir)
    _plot_target_distribution(assignments, plots_dir)
    _plot_top_numeric_drift(drift_numeric, plots_dir)
    return split_dir


def materialize_official_temporal_split(
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    target: str = DEFAULT_SPLIT_TARGET,
    desired_class_floor: int = DEFAULT_DESIRED_CLASS_FLOOR,
    minimum_class_floor: int = DEFAULT_MINIMUM_CLASS_FLOOR,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    split_dir = _official_split_dir(split_id)
    manifest_path = split_dir / "manifest.json"

    if manifest_path.exists() and not force:
        if verbose:
            log(f"📦 Official split '{split_id}' already exists. Loading manifest...", verbose)
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    source_df = load_split_source_frame(target=target, verbose=verbose, force=False)
    assignments, manifest = build_official_temporal_split(
        source_df=source_df,
        target=target,
        split_id=split_id,
        desired_class_floor=desired_class_floor,
        minimum_class_floor=minimum_class_floor,
    )

    source_file = ROOT_PATH / "data" / "processed" / "df_processed.csv"
    manifest["source_file_sha256"] = _sha256_file(source_file)
    manifest["source_n_unique_dates"] = int(source_df["date"].nunique())

    validation = validate_official_split(assignments, manifest)
    split_summary, class_counts = summarize_official_split(assignments, manifest)
    drift_numeric, drift_target, drift_type = compute_minimal_split_drift(assignments, target=target)

    manifest["validation"] = validation
    manifest["generated_artifacts"] = [
        "manifest.json",
        "assignments.csv",
        "split_summary.csv",
        "class_counts.csv",
        "drift_numeric.csv",
        "drift_target_init.csv",
        "drift_type.csv",
        "plots/01_temporal_coverage.png",
        "plots/02_type_mix_by_split.png",
        "plots/03_init_distribution_by_split.png",
        "plots/04_top_numeric_drift_train_vs_test.png",
    ]

    split_dir = save_official_split_artifacts(
        assignments=assignments,
        manifest=manifest,
        split_summary=split_summary,
        class_counts=class_counts,
        drift_numeric=drift_numeric,
        drift_target=drift_target,
        drift_type=drift_type,
        split_id=split_id,
    )

    if verbose:
        log(f"✅ Official temporal split materialized at: {split_dir}", verbose)
    return manifest


def load_official_split(
    split_id: str = DEFAULT_OFFICIAL_SPLIT_ID,
    verbose: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    split_dir = _official_split_dir(split_id)
    manifest_path = split_dir / "manifest.json"
    assignments_path = split_dir / "assignments.csv"

    if not manifest_path.exists() or not assignments_path.exists():
        raise FileNotFoundError(
            f"Official split '{split_id}' does not exist yet under {split_dir}. "
            "Materialize it first with materialize_official_temporal_split()."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assignments = pd.read_csv(assignments_path, parse_dates=["date"])

    if verbose:
        log(f"📦 Loaded official split '{split_id}' from {split_dir}", verbose)
    return manifest, assignments

