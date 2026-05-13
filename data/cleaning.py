import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from data.utils import ROOT_PATH, dump_json, log, path_relative_to_root
import time
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, grey_closing
import matplotlib.pyplot as plt
import math
from data.preprocess import (
    clean_cement_dataframe,
    apply_deterministic_preprocessing,
    filter_by_type_and_summarize,
    ilr_transform_groups,
    normalize_groups
)
from data.utils import load_column_mapping_by_group, load_cleaning_contract
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeRegressor
import threading
import traceback


OFFICIAL_CLEANED_ROOT = ROOT_PATH / "data" / "cleaned" / "official"
DEFAULT_LEGACY_CLEANING_POLICY_ID = "or_drop_holdout_v1"
DEFAULT_OFFICIAL_CLEANING_POLICY_ID = "trainfit_overlap_cap1pct_holdoutflag_v1"
DEFAULT_TRAIN_DROP_CAP_FRACTION = 0.01
DEFAULT_UNIVARIATE_N_BINS = 200
DEFAULT_IFOREST_CONTAMINATIONS = np.linspace(0.0005, 0.01, 20).round(6).tolist()


def load_processed_data(filename: str = "df_processed.csv", verbose: bool = True, force = False) -> pd.DataFrame:
    """
    Load processed cement dataset, or create it if it doesn't exist.

    Args:
        filename (str): CSV file name for the processed data.
        verbose (bool): Whether to print log messages.

    Returns:
        pd.DataFrame: Preprocessed cement data.
    """
    processed_path = ROOT_PATH / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)
    processed_file = processed_path / filename

    if processed_file.exists() and not force:
        log(f"📦 Loading processed data from: {processed_file}", verbose)
        return pd.read_csv(processed_file)

    log("🔄 Processed data not found. Running full preprocessing pipeline...", verbose)

    cleaning_contract, contract_path = load_cleaning_contract(verbose=verbose)
    source_cfg = cleaning_contract["source"]
    type_col = str(source_cfg["columns"]["type"])
    process_col = str(source_cfg["columns"]["process"])
    allowed_types = [str(value) for value in source_cfg["filters"]["allowed_types"]]
    allowed_processes = [str(value).lower() for value in source_cfg["filters"]["allowed_processes"]]
    cols_to_remove = [str(value) for value in source_cfg["remove_columns_after_mapping"]]

    log(
        f"🧩 Using cleaning contract: {path_relative_to_root(contract_path)}",
        verbose,
    )

    # Step 1: Load raw CSV
    raw_path = ROOT_PATH / "data" / "raw"
    raw_file = raw_path / str(source_cfg["raw_filename"])
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Configured raw dataset not found: {path_relative_to_root(raw_file)} "
            f"(contract: {path_relative_to_root(contract_path)})"
        )

    df_raw_full = pd.read_csv(raw_file, **source_cfg["read_csv"])

    missing_raw_columns = [col for col in [type_col, process_col] if col not in df_raw_full.columns]
    if missing_raw_columns:
        raise ValueError(
            "Raw dataset is missing configured columns "
            f"{missing_raw_columns} required by cleaning contract "
            f"{path_relative_to_root(contract_path)}"
        )

    # Step 2: Filter top types and accepted processes immediately
    df_raw = df_raw_full[
        df_raw_full[type_col].astype(str).isin(allowed_types) &
        df_raw_full[process_col].astype(str).str.lower().isin(allowed_processes)
        ].copy()

    if df_raw.empty:
        raise ValueError(
            "Configured type/process filters produced zero rows in load_processed_data(). "
            f"Contract={path_relative_to_root(contract_path)}, raw_file={path_relative_to_root(raw_file)}. "
            "This usually means the active cleaning contract does not match the local raw dataset "
            "or the private overlay is missing."
        )

    shape_raw = df_raw.shape
    cols_raw = set(df_raw.columns)

    print(shape_raw)

    # Step 3: Clean raw data
    df_clean = clean_cement_dataframe(df_raw, verbose=verbose, col_threshold=0.9, remove_cols=cols_to_remove)

    # Step 4: Load encrypted column groups
    _, grouped_mapping = load_column_mapping_by_group(verbose=verbose)

    chem_cols = list(grouped_mapping.get("chem", {}).values())
    phase_cols = list(grouped_mapping.get("phase", {}).values())

    # Optional log
    log(f"[🔐] Using anonymized columns — chem: {chem_cols}", verbose)
    log(f"[🔐] Using anonymized columns — phase: {phase_cols}", verbose)

    ranges = {
        "chem": {"cols": chem_cols, "lower": 0, "upper": 100},
        "phase": {"cols": phase_cols, "lower": 0, "upper": 100}
    }
    exclude_cols = ["a", "b"]

    # Step 5: Run cleaning pipeline
    df_pre_outliers = apply_deterministic_preprocessing(
        df_clean,
        ranges=ranges,
        drop_invalid=True,
        verbose=verbose,
        threshold=0.99,
        excluded_cols=exclude_cols
    )

    # Step 6: Filter by type again for summary logging
    df_filtered = filter_by_type_and_summarize(df_pre_outliers, allowed_types, verbose=verbose)

    # Step 7: ILR + normalization
    groups = {
        "chem": chem_cols,
        "phase": phase_cols
    }
    df_ilr = ilr_transform_groups(df_filtered, groups=groups, verbose=verbose)
    df_normalized = normalize_groups(df_ilr, groups=groups, verbose=verbose)

    # Save and log
    df_normalized.to_csv(processed_file, index=False)
    log(f"✅ Processed data saved to: {processed_file}", verbose)

    # Logging final stats
    shape_final = df_normalized.shape
    cols_final = set(df_normalized.columns)

    removed_cols = sorted(cols_raw - cols_final)
    added_cols = sorted(cols_final - cols_raw)

    log("\n📊 Preprocessing Summary:", verbose)
    log(f"• Initial shape (top types only): {shape_raw}", verbose)
    log(f"• Final shape:                   {shape_final}", verbose)

    pct_lost = 100 * (shape_raw[0] - shape_final[0]) / shape_raw[0]
    log(f"• Rows removed: {shape_raw[0] - shape_final[0]} ({pct_lost:.2f}%)", verbose)

    log(f"• Columns removed ({len(removed_cols)}): {removed_cols}", verbose)
    log(f"• Columns added   ({len(added_cols)}): {added_cols}", verbose)

    # Per-type breakdown
    original_counts = df_raw[type_col].astype(str).value_counts()
    final_counts = df_normalized["type"].value_counts()
    log("\n📎 Per-type row retention:", verbose)
    for t in final_counts.index:
        original = original_counts.get(t, 0)
        final = final_counts[t]
        pct = 100 * (original - final) / original if original > 0 else 0.0
        log(f"  → {t}: {original} → {final} ({pct:.2f}% removed)", verbose)

    return df_normalized


def _legacy_cleaning_dir(split_id: str) -> Path:
    return OFFICIAL_CLEANED_ROOT / split_id


def _official_cleaning_dir(
    split_id: str,
    cleaning_policy_id: str = DEFAULT_OFFICIAL_CLEANING_POLICY_ID,
) -> Path:
    base_dir = OFFICIAL_CLEANED_ROOT / split_id
    if cleaning_policy_id == DEFAULT_LEGACY_CLEANING_POLICY_ID:
        legacy_dir = _legacy_cleaning_dir(split_id)
        if (legacy_dir / "manifest.json").exists():
            return legacy_dir
    return base_dir / cleaning_policy_id


def _cleaning_policy_status(cleaning_policy_id: str) -> str:
    return "legacy_policy" if cleaning_policy_id == DEFAULT_LEGACY_CLEANING_POLICY_ID else "canonical"


def _counts_by_class(series: pd.Series) -> dict[str, int]:
    counts = series.astype(str).value_counts().sort_index()
    return {str(key): int(val) for key, val in counts.items()}


def _train_drop_warnings_by_class(
    split_df: pd.DataFrame,
    drop_mask: pd.Series,
    *,
    type_col: str = "type",
) -> tuple[dict[str, dict[str, float | int]], list[str]]:
    total_rows = int(len(split_df))
    global_drop_fraction = float(drop_mask.mean()) if total_rows else 0.0
    warnings: list[str] = []
    summary: dict[str, dict[str, float | int]] = {}

    for class_value, class_df in split_df.groupby(type_col, sort=False):
        class_mask = drop_mask.loc[class_df.index]
        class_rows = int(len(class_df))
        drop_rows = int(class_mask.sum())
        drop_fraction = float(drop_rows / class_rows) if class_rows else 0.0
        warning = False

        if class_rows > 0 and drop_rows > 0:
            if global_drop_fraction == 0.0:
                warning = True
            else:
                relative_ratio = drop_fraction / global_drop_fraction if global_drop_fraction > 0 else float("inf")
                warning = drop_fraction > (global_drop_fraction + 0.005) and relative_ratio > 2.0

        summary[str(class_value)] = {
            "n_rows": class_rows,
            "drop_rows": drop_rows,
            "drop_fraction": round(drop_fraction, 6),
            "warning": bool(warning),
        }
        if warning:
            warnings.append(
                f"class={class_value}: drop_fraction={drop_fraction:.4f} vs global={global_drop_fraction:.4f}"
            )

    return summary, warnings


def prepare_statistical_cleaning_frame(
    df: pd.DataFrame,
    target: str = "init",
) -> pd.DataFrame:
    """
    Prepare the processed source frame for learned cleaning without changing the
    model-facing feature surface later built in `data.sets`.
    """
    df_out = df.copy()
    if target == "init":
        df_out = df_out.drop(columns=["end", "6h"], errors="ignore")
    return df_out


def build_statistical_cleaning_spec(
    df: pd.DataFrame,
    target: str = "init",
    verbose: bool = False,
) -> dict:
    _, grouped_mapping = load_column_mapping_by_group(verbose=verbose)

    groups = {
        "chem": list(grouped_mapping.get("chem", {}).values()),
        "phase": list(grouped_mapping.get("phase", {}).values()),
    }
    ratio_cols = list(grouped_mapping.get("chem_ratios", {}).values()) + list(
        grouped_mapping.get("phase_ratios", {}).values()
    )

    compositional_cols = [col for cols in groups.values() for col in cols]
    phase_cols = [col for col in df.columns if col.startswith("ilr_phase")]
    visually_considered_unnecessary = ["blaine", "blaines", "water"]
    provenance_cols = ["split", "split_row_id", "source_row_number"]

    excluded_cols = set(
        compositional_cols
        + ratio_cols
        + ["date", "type", "process"]
        + provenance_cols
        + visually_considered_unnecessary
        + [target]
    )
    filter_cols = [
        col
        for col in df.columns
        if not (
            col.startswith("ilr_phase")
            or col.startswith("norm_")
            or "sum" in col
            or col in excluded_cols
        )
    ]
    all_filter_cols = filter_cols + phase_cols
    n_total = len(all_filter_cols)
    if n_total == 0:
        raise ValueError("No statistical cleaning columns were selected from the processed source.")

    alpha = (1 - 0.95) / n_total
    thresholds = np.linspace(1 - alpha, 1, 5).tolist()
    return {
        "groups": groups,
        "ratio_cols": ratio_cols,
        "filter_cols": filter_cols,
        "phase_cols": phase_cols,
        "all_filter_cols": all_filter_cols,
        "thresholds": thresholds,
        "only_threshold": float(min(thresholds)),
        "n_bins": DEFAULT_UNIVARIATE_N_BINS,
        "contamination_values": DEFAULT_IFOREST_CONTAMINATIONS,
    }


def _call_with_numpy_seed(random_state: int | None, func, *args, **kwargs):
    if random_state is None:
        return func(*args, **kwargs)

    previous_state = np.random.get_state()
    try:
        np.random.seed(random_state)
        return func(*args, **kwargs)
    finally:
        np.random.set_state(previous_state)


def fit_univariate_density_rules(
    train_df: pd.DataFrame,
    filter_columns: list[str],
    phase_columns: list[str] | None = None,
    thresholds: list[float] | None = None,
    n_bins: int = DEFAULT_UNIVARIATE_N_BINS,
    only_threshold: float = 0.99,
    type_col: str = "type",
    random_state: int | None = 42,
    verbose: bool = False,
) -> dict:
    phase_columns = phase_columns or []
    columns_to_check = [col for col in filter_columns + phase_columns if col in train_df.columns]
    df_by_type = {
        str(t): train_df[train_df[type_col] == t].copy()
        for t in train_df[type_col].dropna().unique().tolist()
    }

    fitted = _call_with_numpy_seed(
        random_state,
        clean_univariate_bootstrap_density_weighted,
        dataframes_dict=df_by_type,
        filter_columns=filter_columns,
        phase_columns=phase_columns,
        thresholds=thresholds,
        n_bins=n_bins,
        dtype="float64",
        verbose=verbose,
        bootstrap="auto",
        n_bootstrap="auto",
        only_threshold=only_threshold,
        pect_prominence=0.05,
        weight=None,
        filtering=False,
    )

    rules: dict[str, dict] = {
        "type_col": type_col,
        "n_bins": int(n_bins),
        "only_threshold": float(only_threshold),
        "columns_to_check": columns_to_check,
        "per_type": {},
    }

    for type_value, df_marked in fitted.items():
        type_rules: dict[str, dict] = {}
        for col in columns_to_check:
            cluster_col = f"{col}_cluster"
            if cluster_col not in df_marked.columns:
                continue

            values = pd.to_numeric(df_marked[col], errors="coerce").to_numpy(dtype=float)
            keep_mask = (pd.to_numeric(df_marked[cluster_col], errors="coerce").fillna(-1) != -1).to_numpy()
            finite_mask = np.isfinite(values)
            values = values[finite_mask]
            keep_mask = keep_mask[finite_mask]
            if values.size == 0:
                continue

            if np.nanmin(values) == np.nanmax(values):
                bin_edges = np.array([values.min() - 0.5, values.max() + 0.5], dtype=float)
            else:
                bin_edges = np.histogram_bin_edges(values, bins=n_bins)

            bin_indices = np.digitize(values, bin_edges[1:-1], right=False)
            bin_count = len(bin_edges) - 1
            counts = np.bincount(bin_indices, minlength=bin_count)
            kept_counts = np.bincount(bin_indices[keep_mask], minlength=bin_count)
            keep_bins = ((counts > 0) & (kept_counts > 0)).astype(int)

            type_rules[col] = {
                "bin_edges": bin_edges.astype(float).tolist(),
                "keep_bins": keep_bins.astype(int).tolist(),
                "train_rows": int(values.size),
                "train_kept_rows": int(keep_mask.sum()),
            }

        rules["per_type"][str(type_value)] = type_rules
    return rules


def apply_univariate_density_rules(
    df: pd.DataFrame,
    rules: dict,
    type_col: str = "type",
) -> pd.Series:
    keep_series = pd.Series(True, index=df.index, dtype=bool)
    type_rules = rules.get("per_type", {})

    for type_value, df_type in df.groupby(type_col, sort=False):
        current_rules = type_rules.get(str(type_value), {})
        if not current_rules:
            continue

        type_keep = np.ones(len(df_type), dtype=bool)
        for col, rule in current_rules.items():
            if col not in df_type.columns:
                continue

            values = pd.to_numeric(df_type[col], errors="coerce").to_numpy(dtype=float)
            bin_edges = np.asarray(rule["bin_edges"], dtype=float)
            keep_bins = np.asarray(rule["keep_bins"], dtype=bool)
            bin_indices = np.digitize(values, bin_edges[1:-1], right=False)
            valid = (
                np.isfinite(values)
                & (values >= bin_edges[0])
                & (values <= bin_edges[-1])
                & (bin_indices >= 0)
                & (bin_indices < len(keep_bins))
            )
            col_keep = np.zeros(len(df_type), dtype=bool)
            col_keep[valid] = keep_bins[bin_indices[valid]]
            type_keep &= col_keep

        keep_series.loc[df_type.index] = type_keep
    return keep_series


def _select_iforest_model_for_frame(
    df: pd.DataFrame,
    columns_to_check: list[str],
    contamination_values: list[float] | None = None,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[IsolationForest | None, dict]:
    contamination_values = contamination_values or DEFAULT_IFOREST_CONTAMINATIONS
    X = df[columns_to_check].dropna().astype(np.float64)
    if X.empty or len(X) < 10:
        return None, {"status": "skipped_not_enough_rows", "n_rows": int(len(X))}

    best_score = None
    best_model = None
    best_preds = None
    best_contamination = None

    for contamination in contamination_values:
        model = ProgressIsolationForest(
            contamination=contamination,
            n_estimators=50,
            max_samples=min(len(X), 5000),
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X)
        preds = model.predict(X)
        if len(np.unique(preds)) < 2:
            continue

        try:
            score = silhouette_score(
                X,
                preds,
                sample_size=min(10000, X.shape[0]),
                random_state=random_state,
            )
        except Exception:
            continue

        if verbose:
            print(f"[IForest] contamination={contamination:.6f} silhouette={score:.6f}")

        if best_score is None or score > best_score:
            best_score = float(score)
            best_model = model
            best_preds = preds
            best_contamination = float(contamination)

    if best_model is None:
        return None, {"status": "skipped_no_valid_model", "n_rows": int(len(X))}

    return best_model, {
        "status": "ok",
        "n_rows": int(len(X)),
        "best_contamination": best_contamination,
        "best_silhouette": float(best_score),
        "train_outliers": int(np.sum(best_preds == -1)),
    }


def fit_iforest_models_by_type(
    train_df: pd.DataFrame,
    columns_to_check: list[str],
    contamination_values: list[float] | None = None,
    type_col: str = "type",
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[dict[str, IsolationForest | None], dict[str, dict]]:
    fitted_models: dict[str, IsolationForest | None] = {}
    fitted_meta: dict[str, dict] = {}

    for type_value, df_type in train_df.groupby(type_col, sort=False):
        model, meta = _select_iforest_model_for_frame(
            df_type,
            columns_to_check=columns_to_check,
            contamination_values=contamination_values,
            random_state=random_state,
            verbose=verbose,
        )
        fitted_models[str(type_value)] = model
        fitted_meta[str(type_value)] = meta
    return fitted_models, fitted_meta


def apply_iforest_models(
    df: pd.DataFrame,
    models_by_type: dict[str, IsolationForest | None],
    columns_to_check: list[str],
    type_col: str = "type",
) -> pd.Series:
    flags = pd.Series(False, index=df.index, dtype=bool)

    for type_value, df_type in df.groupby(type_col, sort=False):
        model = models_by_type.get(str(type_value))
        if model is None:
            continue

        X = df_type[columns_to_check].dropna().astype(np.float64)
        if X.empty:
            continue
        preds = model.predict(X)
        flags.loc[X.index] = preds == -1
    return flags


def materialize_official_cleaning_audit(
    split_id: str = "init_temporal_processed_v1",
    target: str = "init",
    cleaning_policy_id: str = DEFAULT_OFFICIAL_CLEANING_POLICY_ID,
    force: bool = False,
    verbose: bool = True,
    random_state: int = 42,
    train_drop_cap_fraction: float = DEFAULT_TRAIN_DROP_CAP_FRACTION,
) -> dict:
    """
    Fit learned cleaning on the official TRAIN split only and materialize the
    resulting row-level audit for train/val/test.
    """
    from data.splits import load_official_assigned_source_frame, partition_official_source_frame

    cleaning_dir = _official_cleaning_dir(split_id, cleaning_policy_id=cleaning_policy_id)
    manifest_path = cleaning_dir / "manifest.json"
    row_status_path = cleaning_dir / "row_status.csv"
    univariate_rules_path = cleaning_dir / "artifacts" / "univariate_rules.json"
    iforest_models_path = cleaning_dir / "artifacts" / "iforest_models.joblib"

    if (
        manifest_path.exists()
        and row_status_path.exists()
        and univariate_rules_path.exists()
        and iforest_models_path.exists()
        and not force
    ):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    split_manifest, assigned_source = load_official_assigned_source_frame(
        split_id=split_id,
        target=target,
        verbose=verbose,
    )
    working_df = prepare_statistical_cleaning_frame(assigned_source, target=target)
    spec = build_statistical_cleaning_spec(working_df, target=target, verbose=verbose)
    partitions = partition_official_source_frame(working_df)

    train_df = partitions["train"]
    univariate_rules = fit_univariate_density_rules(
        train_df=train_df,
        filter_columns=spec["filter_cols"],
        phase_columns=spec["phase_cols"],
        thresholds=spec["thresholds"],
        n_bins=spec["n_bins"],
        only_threshold=spec["only_threshold"],
        type_col="type",
        random_state=random_state,
        verbose=False,
    )
    iforest_models, iforest_meta = fit_iforest_models_by_type(
        train_df=train_df,
        columns_to_check=spec["all_filter_cols"],
        contamination_values=spec["contamination_values"],
        type_col="type",
        random_state=random_state,
        verbose=False,
    )

    row_status_frames = []
    summary_by_split: dict[str, dict] = {}
    class_counts_post_cleaning_by_split: dict[str, dict[str, int]] = {}
    class_counts_pre_cleaning_by_split: dict[str, dict[str, int]] = {}
    train_drop_guardrail: dict[str, object] = {
        "global_cap_fraction": float(train_drop_cap_fraction),
        "global_cap_rows": None,
        "candidate_overlap_rows": None,
        "applied_drop_rows": None,
        "cap_exceeded": False,
        "fallback_mode": None,
        "by_class": {},
        "warnings": [],
    }
    for split_name, split_df in partitions.items():
        uni_keep = apply_univariate_density_rules(split_df, univariate_rules, type_col="type")
        iforest_flags = apply_iforest_models(
            split_df,
            models_by_type=iforest_models,
            columns_to_check=spec["all_filter_cols"],
            type_col="type",
        )

        overlap_flags = (~uni_keep) & iforest_flags

        status_df = split_df[["split_row_id", "source_row_number", "split"]].copy()
        status_df["uni_removal_flag"] = (~uni_keep).astype(int)
        status_df["Outlier_IForest"] = iforest_flags.astype(bool)
        status_df["overlap_flag"] = overlap_flags.astype(int)
        status_df["or_removal_flag"] = (
            (status_df["uni_removal_flag"] == 1) | status_df["Outlier_IForest"]
        ).astype(int)

        if split_name == "train":
            max_drop_rows = int(np.floor(train_drop_cap_fraction * len(status_df)))
            drop_for_model = overlap_flags.copy()
            candidate_overlap_rows = int(overlap_flags.sum())
            cap_exceeded = candidate_overlap_rows > max_drop_rows
            fallback_mode = "drop_overlap"

            if cap_exceeded:
                drop_for_model = pd.Series(False, index=split_df.index, dtype=bool)
                fallback_mode = "flag_only_due_to_global_cap"

            class_guardrail, class_warnings = _train_drop_warnings_by_class(
                split_df=split_df,
                drop_mask=drop_for_model,
                type_col="type",
            )
            train_drop_guardrail = {
                "global_cap_fraction": float(train_drop_cap_fraction),
                "global_cap_rows": int(max_drop_rows),
                "candidate_overlap_rows": candidate_overlap_rows,
                "applied_drop_rows": int(drop_for_model.sum()),
                "cap_exceeded": bool(cap_exceeded),
                "fallback_mode": fallback_mode,
                "by_class": class_guardrail,
                "warnings": class_warnings,
            }
        else:
            drop_for_model = pd.Series(False, index=split_df.index, dtype=bool)

        status_df["drop_for_model_flag"] = drop_for_model.astype(int)
        status_df["kept_for_model"] = status_df["drop_for_model_flag"] == 0
        row_status_frames.append(status_df)

        class_counts_pre_cleaning = _counts_by_class(split_df["type"])
        kept_mask = status_df["kept_for_model"].to_numpy(dtype=bool)
        class_counts_post_cleaning = _counts_by_class(split_df.loc[kept_mask, "type"])
        class_counts_pre_cleaning_by_split[split_name] = class_counts_pre_cleaning
        class_counts_post_cleaning_by_split[split_name] = class_counts_post_cleaning
        summary_by_split[split_name] = {
            "n_rows": int(len(status_df)),
            "kept_rows": int(status_df["kept_for_model"].sum()),
            "dropped_rows": int(status_df["drop_for_model_flag"].sum()),
            "drop_fraction": round(float(status_df["drop_for_model_flag"].mean()), 6),
            "univariate_flagged_rows": int(status_df["uni_removal_flag"].sum()),
            "iforest_flagged_rows": int(status_df["Outlier_IForest"].sum()),
            "overlap_flagged_rows": int(status_df["overlap_flag"].sum()),
            "union_flagged_rows": int(status_df["or_removal_flag"].sum()),
            "holdout_policy": "flag_only" if split_name in {"val", "test"} else "train_drop_overlap_cap",
        }

    row_status = (
        pd.concat(row_status_frames, ignore_index=True)
        .sort_values(["source_row_number"])
        .reset_index(drop=True)
    )

    if not row_status["split_row_id"].is_unique:
        raise ValueError("Official cleaning audit produced duplicated split_row_id values.")
    if not row_status["source_row_number"].is_unique:
        raise ValueError("Official cleaning audit produced duplicated source_row_number values.")

    cleaning_dir.mkdir(parents=True, exist_ok=True)
    (cleaning_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    row_status.to_csv(row_status_path, index=False)
    dump_json(univariate_rules, univariate_rules_path)
    joblib.dump(
        {
            "models_by_type": iforest_models,
            "metadata_by_type": iforest_meta,
            "columns_to_check": spec["all_filter_cols"],
            "type_col": "type",
            "random_state": random_state,
        },
        iforest_models_path,
    )

    manifest = {
        "split_id": split_id,
        "target": target,
        "status": "official",
        "policy_status": _cleaning_policy_status(cleaning_policy_id),
        "cleaning_policy_id": cleaning_policy_id,
        "fit_split": "train",
        "drop_rule": "train_overlap_only_with_global_cap",
        "holdout_policy": "flag_only",
        "train_drop_cap_fraction_global": float(train_drop_cap_fraction),
        "source_split_manifest": path_relative_to_root(
            ROOT_PATH / "data" / "splits" / "official" / split_id / "manifest.json"
        ),
        "source_stage": split_manifest.get("source_stage"),
        "source_file": split_manifest.get("source_file"),
        "source_file_sha256": split_manifest.get("source_file_sha256_verified", split_manifest.get("source_file_sha256")),
        "random_state": int(random_state),
        "artifacts": {
            "row_status": path_relative_to_root(row_status_path),
            "univariate_rules": path_relative_to_root(univariate_rules_path),
            "iforest_models": path_relative_to_root(iforest_models_path),
        },
        "row_status_columns": row_status.columns.tolist(),
        "columns_used_for_cleaning": {
            "univariate": spec["filter_cols"],
            "phase": spec["phase_cols"],
            "iforest": spec["all_filter_cols"],
        },
        "quality_filter_policy": {
            "sum_columns": "treated_as_domain_quality_filter_upstream",
        },
        "summary_by_split": summary_by_split,
        "class_counts_pre_cleaning_by_split": class_counts_pre_cleaning_by_split,
        "class_counts_post_cleaning_by_split": class_counts_post_cleaning_by_split,
        "train_drop_guardrail": train_drop_guardrail,
        "iforest_selection_by_type": iforest_meta,
    }
    dump_json(manifest, manifest_path)

    if verbose:
        log(f"✅ Official cleaning audit materialized at: {cleaning_dir}", verbose)
    return manifest


def load_official_cleaning_audit(
    split_id: str = "init_temporal_processed_v1",
    target: str = "init",
    cleaning_policy_id: str = DEFAULT_OFFICIAL_CLEANING_POLICY_ID,
    force: bool = False,
    verbose: bool = True,
) -> tuple[dict, pd.DataFrame]:
    cleaning_dir = _official_cleaning_dir(split_id, cleaning_policy_id=cleaning_policy_id)
    manifest_path = cleaning_dir / "manifest.json"
    row_status_path = cleaning_dir / "row_status.csv"

    if force or not manifest_path.exists() or not row_status_path.exists():
        materialize_official_cleaning_audit(
            split_id=split_id,
            target=target,
            cleaning_policy_id=cleaning_policy_id,
            force=force,
            verbose=verbose,
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    row_status = pd.read_csv(row_status_path)
    return manifest, row_status


def load_official_modeling_source_frame(
    split_id: str = "init_temporal_processed_v1",
    target: str = "init",
    cleaning_policy_id: str = DEFAULT_OFFICIAL_CLEANING_POLICY_ID,
    force: bool = False,
    verbose: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """
    Return the processed source rows kept for modeling after applying the
    official split-aware learned cleaning audit.
    """
    from data.splits import load_official_assigned_source_frame

    split_manifest, assigned_source = load_official_assigned_source_frame(
        split_id=split_id,
        target=target,
        verbose=verbose,
    )
    cleaning_manifest, row_status = load_official_cleaning_audit(
        split_id=split_id,
        target=target,
        cleaning_policy_id=cleaning_policy_id,
        force=force,
        verbose=verbose,
    )

    working_df = prepare_statistical_cleaning_frame(assigned_source, target=target)
    merged = working_df.merge(
        row_status,
        on=["split_row_id", "source_row_number", "split"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(working_df):
        raise ValueError(
            f"Official cleaning audit does not cover the full working source: {len(merged)} rows vs {len(working_df)}."
        )

    kept_df = (
        merged[merged["kept_for_model"]]
        .copy()
        .sort_values(["date", "source_row_number"])
        .reset_index(drop=True)
    )
    manifest = {
        "split_id": split_id,
        "target": target,
        "cleaning_policy_id": cleaning_policy_id,
        "source_split_manifest": cleaning_manifest["source_split_manifest"],
        "source_cleaning_manifest": path_relative_to_root(
            _official_cleaning_dir(split_id, cleaning_policy_id=cleaning_policy_id) / "manifest.json"
        ),
        "source_file": split_manifest.get("source_file"),
        "source_file_sha256": split_manifest.get("source_file_sha256_verified", split_manifest.get("source_file_sha256")),
    }
    return manifest, kept_df



def clean_univariate_bootstrap_density_weighted(
        dataframes_dict,
        filter_columns,
        phase_columns=None,
        thresholds=None,
        n_bins=100,
        dtype="float64",
        verbose=True,
        bootstrap="auto",
        n_bootstrap="auto",
        only_threshold=0.99,
        pect_prominence=0.05,
        weight=None,
        filtering=True
):
    if phase_columns is None:
        phase_columns = []

    ilr_columns = filter_columns
    cleaned_dfs = {}

    for name, df in dataframes_dict.items():
        start_time = time.time()
        if verbose:
            print(f"\n🚀 Cleaning '{name}' using univariate mountain-based density filtering...")

        columns_to_check = ilr_columns + phase_columns
        columns_to_check = [col for col in columns_to_check if col in df.columns]
        valid_mask = df[columns_to_check].notna().all(axis=1)

        # If using weights, also check for NaNs in weight column
        if weight:
            valid_mask &= df[weight].notna()

        df_cleaned = df.loc[valid_mask].copy()

        # ──────────────────────────────────────────────────────────────
        # NEW → start with “keep everything” for all cleaned rows
        n_rows = df_cleaned.shape[0]
        global_keep_mask = np.ones(n_rows, dtype=bool)

        # optional: to preserve cluster labels until final filtering
        cluster_labels_dict = {}  # {col_name: full-length array}
        # ──────────────────────────────────────────────────────────────

        # ✅ Dynamically resolve bootstrap after filtering NaNs
        n_samples_cleaned = df_cleaned.shape[0]

        if bootstrap == "auto":
            bootstrap = n_samples_cleaned < 1000
            if verbose:
                print(f"   ➤ Bootstrap auto-evaluated: {bootstrap}  (cleaned samples = {n_samples_cleaned})")

        for col in columns_to_check:
            col_start_time = time.time()

            weights = df_cleaned[weight].values.astype(dtype) if weight else None
            original_values = df_cleaned[col].values.astype(dtype)
            original_plot_weights = df_cleaned[weight].values.astype(dtype) if weight else None

            n_samples = len(original_values)

            if verbose:
                print(f"\n🔍 Processing column: {col}")
                print(f"   ➤ Original sample count: {n_samples}")
                print(f"   ➤ Bootstrap set to: {bootstrap}")
                print(f"   ➤ Weight set to: {weight}")

            # Use bootstrapped values only for histogram/peak detection
            if bootstrap:
                if n_bootstrap == "auto":
                    if n_samples < 500:
                        n_boot = 5000
                    elif n_samples < 800:
                        n_boot = 3000
                    else:
                        n_boot = 2000
                else:
                    n_boot = n_bootstrap

                if verbose:
                    print(f"📦 Using n_boot = {n_boot} resamples")

                # Sampling indices with or without weight
                bootstrap_indices = np.concatenate([
                    np.random.choice(n_samples, size=n_samples, replace=True)
                    for _ in range(n_boot)
                ])
                bootstrap_values = original_values[bootstrap_indices]

            else:
                bootstrap_values = original_values  # fallback if no bootstrapping

            # Histogram (weighted or not)
            # Step: Generate both histograms if bootstrap is used
            # ------------------------------------

            # Histogram for original values
            if weight:
                hist_original, bin_edges = np.histogram(
                    original_values,
                    bins=n_bins,
                    weights=weights,
                    density=True
                )
            else:
                hist_original, bin_edges = np.histogram(
                    original_values,
                    bins=n_bins,
                    density=True
                )

            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Histogram for bootstrapped values (used for peak detection)
            if bootstrap:
                if weight:
                    sample_weights = np.tile(weights, n_boot)
                    hist_bootstrap, _ = np.histogram(
                        bootstrap_values,
                        bins=bin_edges,
                        weights=sample_weights,
                        density=True
                    )
                else:
                    hist_bootstrap, _ = np.histogram(
                        bootstrap_values,
                        bins=bin_edges,
                        density=True
                    )
            else:
                hist_bootstrap = hist_original  # fallback to original
                bootstrap_values = original_values  # safety if plotting later

            # === Define smoothing constants ===
            gradient_threshold = 0.05
            window_ratio = 1 / 40
            min_window_bins = 3
            min_sigma = 1
            max_sigma = 5

            base_sigma_bootstrap = 1.25 if col in phase_columns else 1.0
            base_sigma_original = 1.5 if col in phase_columns else 1.25

            # === Adaptive smoothing function ===
            def adaptive_smoothing(hist, base_sigma, label=""):
                window_raw = int(len(hist) * window_ratio)
                window = max(min_window_bins, window_raw)
                half_window = window  # window // 2

                if verbose:
                    print(
                        f"   ➤ Window ({label}): {'MIN' if window == min_window_bins else f'{window} (from {window_raw})'} bins")

                # === Detect not-isolated zero bins ===
                n_not_isolated_zeros = 0
                for i in range(half_window, len(hist) - half_window):
                    if hist[i] == 0:
                        left = hist[i - half_window:i]
                        right = hist[i + 1:i + 1 + half_window]
                        if np.any(left > 0) or np.any(right > 0):
                            n_not_isolated_zeros += 1

                gap_pct = 100 * n_not_isolated_zeros / len(hist)
                dyn_percentile = min(98, 90 + gap_pct)

                if verbose:
                    print(f"   ➤ Not-isolated 0 bins: {n_not_isolated_zeros} / {len(hist)} ({gap_pct:.2f}%)")
                    print(f"   ➤ Using dynamic percentile: {dyn_percentile:.2f}%")

                # === Gradient stats over window range ===
                gradients = np.array([
                    np.abs(hist[i + half_window] - hist[i - half_window])
                    for i in range(half_window, len(hist) - half_window)
                ])
                grad_stat = np.percentile(gradients, dyn_percentile)

                if grad_stat > gradient_threshold:
                    inflation = (1 / math.sqrt(grad_stat)) * (grad_stat - gradient_threshold)
                    smoothing_sigma = base_sigma * (1 + inflation ** 2)

                    if smoothing_sigma < min_sigma:
                        if verbose:
                            print(f"   ➤ σ clamped to MIN ({label}): {smoothing_sigma:.4f} → {min_sigma}")
                        smoothing_sigma = min_sigma
                    elif smoothing_sigma > max_sigma:
                        if verbose:
                            print(f"   ➤ σ clamped to MAX ({label}): {smoothing_sigma:.4f} → {max_sigma}")
                        smoothing_sigma = max_sigma
                    else:
                        if verbose:
                            print(f"   ➤ σ adjusted ({label}): {smoothing_sigma:.4f}")

                    hist_proc = grey_closing(hist.astype(float), size=3)
                    closing_applied = True
                else:
                    smoothing_sigma = base_sigma
                    hist_proc = hist
                    closing_applied = False
                    if verbose:
                        print(f"   ➤ σ kept base ({label}): {smoothing_sigma:.2f}")

                if verbose:
                    print(
                        f"   ➤ Gradient ({dyn_percentile:.2f}th %ile) ({label}): {grad_stat:.4f} → σ = {smoothing_sigma:.2f} | Closing: {'Yes' if closing_applied else 'No'}")

                smoothed = gaussian_filter1d(hist_proc, sigma=smoothing_sigma)
                return smoothed

            # === Apply adaptive smoothing ===
            smoothed_hist_original = adaptive_smoothing(hist_original, base_sigma_original, label="Original")
            if bootstrap:
                smoothed_hist_bootstrap = adaptive_smoothing(hist_bootstrap, base_sigma_bootstrap, label="Bootstrap")
            else:
                smoothed_hist_bootstrap = smoothed_hist_original.copy()

            # Use the bootstrap version for peak detection logic
            max_density = hist_bootstrap.max()
            prominence_threshold = max_density * 0.01  # for peak detection
            bin_width = bin_edges[1] - bin_edges[0]
            total_mass = np.sum(smoothed_hist_bootstrap) * bin_width
            total_mass_original = np.sum(hist_original) * bin_width

            # --- Peak Detection ---
            peaks, properties = find_peaks(
                smoothed_hist_bootstrap,
                prominence=prominence_threshold,
                distance=max(1, (n_bins * 3) // 20)
            )

            # Keep peaks based on prominence of total mass
            peak_mask = []
            discarded_peaks = []

            # Define dynamic window size (1/10 of n_bins)
            if n_bins < 100:
                window_size = int(np.ceil(n_bins / 20))
            else:
                window_size = int(np.floor(n_bins / 20))

            for peak in peaks:
                start = max(0, peak - window_size)
                end = min(len(smoothed_hist_bootstrap), peak + window_size + 1)  # +1 because slice is exclusive

                # Area in smoothed bootstrap hist
                peak_area_smoothed = np.sum(smoothed_hist_bootstrap[start:end]) * bin_width
                peak_fraction_smoothed = peak_area_smoothed / total_mass

                # Area in original hist
                peak_area_original = np.sum(hist_original[start:end]) * bin_width
                peak_fraction_original = peak_area_original / total_mass_original

                if (peak_fraction_smoothed >= pect_prominence) and (peak_fraction_original >= pect_prominence):
                    peak_mask.append(True)
                else:
                    peak_mask.append(False)
                    discarded_peaks.append(peak)

            # Apply mask
            peaks = peaks[np.asarray(peak_mask, dtype=bool)]

            # Merge close peaks if valley not deep enough
            valley_threshold = 0.5
            i = 0
            while i < len(peaks) - 1:
                current = peaks[i]
                j = i + 1
                merged = False
                while j < len(peaks):
                    left = current
                    right = peaks[j]
                    h_left = smoothed_hist_bootstrap[left]
                    h_right = smoothed_hist_bootstrap[right]

                    valley_means = [
                        np.mean(smoothed_hist_bootstrap[k - 1: k + 2])
                        for k in range(left + 1, right - 1)
                    ]
                    valley = min(valley_means) if valley_means else np.min(smoothed_hist_bootstrap[left:right + 1])
                    peak_ratio = min(h_left, h_right) / max(h_left, h_right)
                    divisor = min(h_left, h_right) if peak_ratio < 0.5 else 0.5 * (h_left + h_right)
                    valley_ratio = valley / divisor

                    if valley_ratio > valley_threshold:
                        drop_peak = left if h_left < h_right else right
                        discarded_peaks.append(drop_peak)
                        peaks = peaks[peaks != drop_peak]
                        merged = True
                        break
                    j += 1
                if not merged:
                    i += 1

            # Step 4: Print info and plot
            # Step 4: Print info and plot
            if verbose:
                print(
                    f"{col} — Pect_Prominence = {pect_prominence} | Prominence Threshold = {prominence_threshold:.4f} ({0.01 * 100:.2f}% of peak height)")

                # Estimate bin width
                bin_width = bin_edges[1] - bin_edges[0]

                # Recalculate heights without plotting, for consistency
                if weight:
                    hist_heights_original, _ = np.histogram(
                        original_values,
                        bins=bin_edges,
                        weights=original_plot_weights,
                        density=True
                    )
                else:
                    hist_heights_original, _ = np.histogram(
                        original_values,
                        bins=bin_edges,
                        density=True
                    )

                max_density_original = hist_heights_original.max()

                max_density_pct = max_density_original * bin_width * 100
                prominence_density_pct = prominence_threshold * bin_width * 100

                print(f"   ➤ Max bin density ≈ {max_density_pct:.2f}% of total samples")
                print(f"   ➤ Prominence threshold ≈ {prominence_density_pct:.2f}% of total samples")
                print(f"   ➤ Density threshold ≈ {pect_prominence * 100:.2f}% of total samples")

                # Common Y-axis config for original
                temp_pect = pect_prominence
                y_max_orig = max_density_original * (1 + temp_pect)
                percent_levels = np.arange(0, 100 + (temp_pect * 100), temp_pect * 100)
                ticks_orig = (percent_levels / 100) * max_density_original
                tick_labels_orig = [f"{p:>5.2f}% - {v:.3f}" for p, v in zip(percent_levels, ticks_orig)]

                # ----- Plot 1: Original Histogram -----
                plt.figure(figsize=(10, 6))
                plt.hist(
                    original_values,
                    bins=n_bins,
                    density=True,
                    weights=original_plot_weights,
                    color="steelblue",
                    alpha=0.6,
                    edgecolor="black",
                    label="Original Histogram"
                )

                # Overlay smoothed histogram from ORIGINAL
                plt.plot(
                    bin_centers,
                    smoothed_hist_original,
                    color="darkorange",
                    linewidth=2,
                    label="Smoothed (Original)"
                )

                # Overlay smoothed histogram from BOOTSTRAP
                if bootstrap:
                    plt.plot(
                        bin_centers,
                        smoothed_hist_bootstrap,
                        color="mediumvioletred",
                        linewidth=2,
                        linestyle="--",
                        label="Smoothed (Bootstrap)"
                    )

                # Always plot on original smoothed line
                if len(peaks) > 0:
                    plt.scatter(
                        bin_centers[peaks],
                        smoothed_hist_original[peaks],
                        color="red",
                        s=60,
                        zorder=5,
                        label="Detected Peaks (Original)"
                    )

                if len(discarded_peaks) > 0:
                    plt.scatter(
                        bin_centers[discarded_peaks],
                        smoothed_hist_original[discarded_peaks],
                        color="green",
                        s=60,
                        zorder=5,
                        label="Discarded Peaks (Original)"
                    )

                # If bootstrap is on, also overlay peaks on bootstrap line
                if bootstrap:
                    if len(peaks) > 0:
                        plt.scatter(
                            bin_centers[peaks],
                            smoothed_hist_bootstrap[peaks],
                            color="red",
                            s=60,
                            zorder=5,
                            marker="x",
                            label="Detected Peaks (Bootstrap)"
                        )

                    if len(discarded_peaks) > 0:
                        plt.scatter(
                            bin_centers[discarded_peaks],
                            smoothed_hist_bootstrap[discarded_peaks],
                            color="green",
                            s=60,
                            zorder=5,
                            marker="x",
                            label="Discarded Peaks (Bootstrap)"
                        )

                plt.ylim(0, y_max_orig)
                plt.yticks(ticks_orig, tick_labels_orig)
                plt.title(f"{col} — Original Distribution")
                plt.xlabel(col)
                plt.ylabel("Density")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()

                # ----- Plot 2: Bootstrapped Histogram + Smoothed (if enabled) -----
                if bootstrap:
                    if weight and weight in df_cleaned.columns:
                        bootstrap_plot_weights = original_plot_weights[bootstrap_indices]
                    else:
                        bootstrap_plot_weights = None

                    # Compute bar heights for bootstrapped histogram (without plotting)
                    if weight:
                        hist_heights, _ = np.histogram(
                            bootstrap_values,
                            bins=bin_edges,
                            weights=bootstrap_plot_weights,
                            density=True
                        )
                    else:
                        hist_heights, _ = np.histogram(
                            bootstrap_values,
                            bins=bin_edges,
                            density=True
                        )

                    # Use this for Y-axis scaling
                    max_bar_height = hist_heights.max()
                    y_max_bootstrap = max_bar_height * (1 + temp_pect)
                    ticks_bootstrap = (percent_levels / 100) * max_bar_height
                    tick_labels_bootstrap = [f"{p:>5.2f}% - {v:.3f}" for p, v in zip(percent_levels, ticks_bootstrap)]

                    plt.figure(figsize=(10, 6))
                    plt.hist(
                        bootstrap_values,
                        bins=bin_edges,
                        density=True,
                        weights=bootstrap_plot_weights,
                        color="skyblue",
                        alpha=0.6,
                        edgecolor="black",
                        label="Bootstrapped Histogram"
                    )

                    # Overlay smoothed histogram from BOOTSTRAP
                    plt.plot(
                        bin_centers,
                        smoothed_hist_bootstrap,
                        color="mediumvioletred",
                        linewidth=2,
                        label="Smoothed (Bootstrap)"
                    )

                    # Overlay smoothed histogram from ORIGINAL
                    plt.plot(
                        bin_centers,
                        smoothed_hist_original,
                        color="darkorange",
                        linewidth=2,
                        linestyle="--",
                        label="Smoothed (Original)"
                    )

                    # Bootstrap peaks on bootstrap line
                    if len(peaks) > 0:
                        plt.scatter(
                            bin_centers[peaks],
                            smoothed_hist_bootstrap[peaks],
                            color="red",
                            s=60,
                            zorder=5,
                            label="Detected Peaks (Bootstrap)"
                        )

                    if len(discarded_peaks) > 0:
                        plt.scatter(
                            bin_centers[discarded_peaks],
                            smoothed_hist_bootstrap[discarded_peaks],
                            color="green",
                            s=60,
                            zorder=5,
                            label="Discarded Peaks (Bootstrap)"
                        )

                    # Overlay same peaks on original smoothed line for reference
                    if len(peaks) > 0:
                        plt.scatter(
                            bin_centers[peaks],
                            smoothed_hist_original[peaks],
                            color="red",
                            s=60,
                            zorder=5,
                            marker="x",
                            label="Detected Peaks (Original)"
                        )

                    if len(discarded_peaks) > 0:
                        plt.scatter(
                            bin_centers[discarded_peaks],
                            smoothed_hist_original[discarded_peaks],
                            color="green",
                            s=60,
                            zorder=5,
                            marker="x",
                            label="Discarded Peaks (Original)"
                        )

                    plt.ylim(0, y_max_bootstrap)
                    plt.yticks(ticks_bootstrap, tick_labels_bootstrap)
                    plt.title(f"{col} — Bootstrapped + Smoothed (Peak Detection)")
                    plt.xlabel(col)
                    plt.ylabel("Density")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

            hist = hist_original

            if len(peaks) < 1:
                if verbose:
                    print(f"⚠️ No peaks detected for {col}, applying low-density filtering...")

                if only_threshold is None:
                    only_threshold = 0.99

                main_peak_idx = np.argmax(hist)
                main_peak_density = hist[main_peak_idx]
                total_mass = hist.sum()

                if verbose:
                    print(f"🔹 Main peak at bin {main_peak_idx} with density {main_peak_density}")
                    print(f"🔸 Total mass (sum of all bin frequencies): {total_mass}")

                # Penalty: lower density and farther from the main peak
                penalties = []
                for i in range(len(hist)):
                    density = hist[i]
                    distance = abs(i - main_peak_idx)
                    score = (1 / (density + 1e-8)) * (1 + distance)
                    penalties.append((score, i))

                sorted_bins = [i for _, i in sorted(penalties, key=lambda x: x[0])]

                keep_mask = np.zeros_like(hist, dtype=bool)
                accumulated = 0
                for idx in sorted_bins:
                    if accumulated / total_mass >= only_threshold:
                        break
                    keep_mask[idx] = True
                    accumulated += hist[idx]

                bin_indices = np.clip(np.digitize(original_values, bin_edges) - 1, 0, len(hist) - 1)
                value_keep_mask = keep_mask[bin_indices]

                kept_n = value_keep_mask.sum()
                removed_n = len(original_values) - kept_n
                retained_ratio = kept_n / len(original_values)

                if verbose:
                    print(
                        f"📌 {col}: Retained {retained_ratio * 100:.2f}% of samples → {kept_n}/{len(original_values)} kept, {removed_n} removed.")

                    # Plot filtered histogram of original values
                    plt.figure(figsize=(10, 6))

                    plt.hist(
                        original_values[value_keep_mask],
                        bins=n_bins,
                        density=True,
                        weights=original_plot_weights[value_keep_mask] if weight else None,
                        color="orange",
                        edgecolor="black",
                        alpha=0.8,
                        label="Filtered"
                    )

                    # Recalculate histogram heights for filtered values
                    if weight:
                        hist_filtered, _ = np.histogram(
                            original_values[value_keep_mask],
                            bins=n_bins,
                            weights=original_plot_weights[value_keep_mask],
                            density=True
                        )
                    else:
                        hist_filtered, _ = np.histogram(
                            original_values[value_keep_mask],
                            bins=n_bins,
                            density=True
                        )

                    max_density_filtered = hist_filtered.max()

                    # Y-axis config
                    temp_pect = pect_prominence
                    y_max = max_density_filtered * (1 + temp_pect)
                    percent_levels = np.arange(0, 100 + (temp_pect * 100), temp_pect * 100)
                    ticks = (percent_levels / 100) * max_density_filtered
                    tick_labels = [f"{p:>5.2f}% - {v:.3f}" for p, v in zip(percent_levels, ticks)]

                    plt.ylim(0, y_max)
                    plt.yticks(ticks, tick_labels)

                    plt.title(f"{col} — Filtered (No Peaks) with threshold = {only_threshold}")
                    plt.xlabel(col)
                    plt.ylabel("Density")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

                # ── OLD ─────────────────────────────────────────────────────────
                # df_cleaned = df_cleaned.loc[value_keep_mask]
                # df_cleaned[f"{col}_cluster"] = 0
                # continue
                # ── NEW ─────────────────────────────────────────────────────────
                global_keep_mask &= value_keep_mask  # ❶ accumulate

                # store cluster labels (0 for kept rows, NaN for dropped)
                cluster_full = np.full(n_rows, np.nan)
                cluster_full[value_keep_mask] = 0
                cluster_labels_dict[col] = cluster_full

                continue


            elif len(peaks) == 1:
                if verbose:
                    print(f"⚠️ Only one peak found for {col}, applying centered filtering...")

                if only_threshold is None:
                    only_threshold = 0.99

                main_peak_idx = peaks[0]
                main_peak_density = hist[main_peak_idx]
                total_mass = hist.sum()

                if verbose:
                    print(f"🔹 Single peak at bin {main_peak_idx} with density {main_peak_density}")
                    print(f"🔸 Total mass (sum of all bin frequencies): {total_mass:.2f}")

                # Penalty: prioritize bins with lower density and farther from the peak
                penalties = []
                for i in range(len(hist)):
                    density = hist[i]
                    distance = abs(i - main_peak_idx)
                    score = (1 / (density + 1e-8)) * (1 + distance)
                    penalties.append((score, i))

                sorted_bins = [i for _, i in sorted(penalties, key=lambda x: x[0])]

                keep_mask = np.zeros_like(hist, dtype=bool)
                accumulated = 0
                for idx in sorted_bins:
                    if accumulated / total_mass >= only_threshold:
                        break
                    keep_mask[idx] = True
                    accumulated += hist[idx]

                # Apply to original values
                bin_indices = np.clip(np.digitize(original_values, bin_edges) - 1, 0, len(hist) - 1)
                value_keep_mask = keep_mask[bin_indices]

                kept_n = value_keep_mask.sum()
                removed_n = len(original_values) - kept_n
                retained_ratio = kept_n / len(original_values)

                if verbose:
                    print(
                        f"📌 {col}: Retained {retained_ratio * 100:.2f}% of samples → {kept_n}/{len(original_values)} kept, {removed_n} removed.")

                    # Plot filtered histogram (original values after filtering)
                    plt.figure(figsize=(10, 6))

                    plt.hist(
                        original_values[value_keep_mask],
                        bins=n_bins,
                        density=True,
                        weights=original_plot_weights[value_keep_mask] if weight else None,
                        color="orange",
                        edgecolor="black",
                        alpha=0.8,
                        label="Filtered"
                    )

                    # Recalculate histogram height for filtered values
                    if weight:
                        hist_filtered, _ = np.histogram(
                            original_values[value_keep_mask],
                            bins=n_bins,
                            weights=original_plot_weights[value_keep_mask],
                            density=True
                        )
                    else:
                        hist_filtered, _ = np.histogram(
                            original_values[value_keep_mask],
                            bins=n_bins,
                            density=True
                        )

                    max_density_filtered = hist_filtered.max()

                    # Use filtered max for correct scaling
                    temp_pect = pect_prominence
                    y_max = max_density_filtered * (1 + temp_pect)
                    percent_levels = np.arange(0, 100 + (temp_pect * 100), temp_pect * 100)
                    ticks = (percent_levels / 100) * max_density_filtered
                    tick_labels = [f"{p:>5.2f}% - {v:.3f}" for p, v in zip(percent_levels, ticks)]

                    plt.ylim(0, y_max)
                    plt.yticks(ticks, tick_labels)

                    plt.title(f"{col} — Filtered (1 Peak) with threshold = {only_threshold}")
                    plt.xlabel(col)
                    plt.ylabel("Density")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

                # ── OLD ─────────────────────────────────────────────────────────
                # df_cleaned = df_cleaned.loc[value_keep_mask]
                # df_cleaned[f"{col}_cluster"] = 0
                # continue
                # ── NEW ─────────────────────────────────────────────────────────
                global_keep_mask &= value_keep_mask  # ❶ accumulate

                cluster_full = np.full(n_rows, np.nan)
                cluster_full[value_keep_mask] = 0
                cluster_labels_dict[col] = cluster_full

                continue


            elif len(peaks) > 1:
                if verbose:
                    print(
                        f"🔍 Found {len(peaks)} peaks for {col}, applying threshold-based filtering with silhouette optimization.")

                if thresholds is None:
                    thresholds = [0.9825, 0.9875, 0.99, 0.9925, 0.995, 0.9975, 1]

                best_score = -np.inf
                best_keep_mask = None
                best_labels_hist = None
                best_threshold = None

                for threshold in thresholds:
                    total_mass = hist.sum()

                    # Penalty score based on density and distance to closest peak
                    penalties = []
                    for i in range(len(hist)):
                        distances = [abs(i - p) for p in peaks]
                        min_distance = min(distances)
                        penalty = (1 / (hist[i] + 1e-8)) * (1 + min_distance)
                        penalties.append((penalty, i))

                    sorted_bins = [i for _, i in sorted(penalties, key=lambda x: x[0])]

                    # Create keep mask
                    keep_mask = np.zeros_like(hist, dtype=bool)
                    accumulated = 0
                    for idx in sorted_bins:
                        if accumulated / total_mass >= threshold:
                            break
                        keep_mask[idx] = True
                        accumulated += hist[idx]

                    # Assign bin labels by peak proximity
                    labels = np.zeros_like(hist, dtype=int)
                    for i in range(len(hist)):
                        distances = [abs(i - p) for p in peaks]
                        labels[i] = np.argmin(distances)

                    filtered_labels = labels.copy()
                    filtered_labels[~keep_mask] = -1

                    # Apply filtering to original values (not bootstrap)
                    bin_indices = np.clip(np.digitize(original_values, bin_edges) - 1, 0, len(hist) - 1)
                    value_labels = filtered_labels[bin_indices]
                    mask_valid = value_labels != -1

                    if len(np.unique(value_labels[mask_valid])) < 2:
                        continue

                    # Determine sample size for silhouette
                    n_samples = len(original_values[mask_valid])
                    use_subsample = n_samples > 10000
                    sample_size = 10000 if use_subsample else None

                    if verbose:
                        print(
                            f"📏 Silhouette: using {'ALL' if sample_size is None else sample_size} out of {n_samples} samples")

                    score = silhouette_score(
                        original_values[mask_valid].reshape(-1, 1),
                        value_labels[mask_valid],
                        sample_size=sample_size,  # Only used if subsampling
                        random_state=42 if use_subsample else None  # Only matters if subsampling
                    )

                    alpha = 0.65
                    kept_ratio = len(original_values[mask_valid]) / len(original_values)
                    adjusted_score = score * (kept_ratio ** alpha)

                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_keep_mask = mask_valid
                        best_labels_hist = filtered_labels
                        best_threshold = threshold

                if best_keep_mask is None:
                    if verbose:
                        print(f"⚠️ No valid segmentation found for {col}.")
                    continue

                # --- OLD lines to remove / replace ---------------------------------
                # filtered_values = original_values[best_keep_mask]
                # before_n = df_cleaned.shape[0]
                # df_cleaned = df_cleaned.loc[best_keep_mask]
                # removed_pct = 100 * (before_n - df_cleaned.shape[0]) / before_n
                #
                # bin_indices_final = ...
                # cluster_labels = ...
                # df_cleaned[f"{col}_cluster"] = cluster_labels
                # -------------------------------------------------------------------

                # ❶ Keep filtered values just for plots / summaries
                filtered_values = original_values[best_keep_mask]
                filtered_weights = original_plot_weights[best_keep_mask] if weight else None

                # ❷ Accumulate the mask instead of shrinking df_cleaned now
                global_keep_mask &= best_keep_mask

                # ❸ Prepare full-length cluster label array (NaN for rows that will be dropped)
                cluster_full = np.full(n_rows, np.nan)
                bin_indices_final = np.clip(np.digitize(original_values, bin_edges) - 1, 0, len(hist) - 1)
                cluster_labels = best_labels_hist[bin_indices_final]
                cluster_full[best_keep_mask] = cluster_labels[best_keep_mask]
                cluster_labels_dict[col] = cluster_full

                # ➍ For verbose “removed %”, compute using the global mask so far
                before_n = n_rows
                after_n = global_keep_mask.sum()
                removed_pct = 100 * (before_n - after_n) / before_n

                if verbose:
                    print(
                        f"✅ {col}: Best threshold={best_threshold}, Silhouette={best_score:.4f}, Alpha={alpha}, Removed {removed_pct:.2f}%")
                    print(f"⏱️ Column time: {time.time() - col_start_time:.2f} seconds.")

                    # Plot filtered bins from original data
                    plt.figure(figsize=(10, 6))
                    for label in np.unique(best_labels_hist[best_labels_hist != -1]):
                        mask = best_labels_hist == label
                        plt.bar(
                            bin_centers[mask],
                            hist[mask],
                            width=bin_edges[1] - bin_edges[0],
                            alpha=0.6,
                            edgecolor="black",
                            label=f"Mountain {label + 1}"
                        )

                    # Use temp_pect for consistent y-axis setup
                    temp_pect = pect_prominence
                    y_max = hist.max() * (1 + temp_pect)
                    percent_levels = np.arange(0, 100 + (temp_pect * 100), temp_pect * 100)
                    ticks = (percent_levels / 100) * hist.max()
                    tick_labels = [f"{p:>5.2f}% - {v:.3f}" for p, v in zip(percent_levels, ticks)]

                    plt.ylim(0, y_max)
                    plt.yticks(ticks, tick_labels)

                    plt.title(f"{col} — Filtered with Best Threshold = {best_threshold}")
                    plt.xlabel(col)
                    plt.ylabel("Density")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

                    # Filtered values and aligned weights
                    filtered_values = original_values[best_keep_mask]
                    filtered_weights = original_plot_weights[best_keep_mask] if weight else None

        # ───────── FINAL UNION / ATTACH CLUSTERS ───────────────────────
        if filtering:
            # really drop rows that failed any column mask
            df_cleaned = df_cleaned.loc[global_keep_mask].copy()

        # add a <col>_cluster column for every processed feature
        for col, full_labels in cluster_labels_dict.items():
            if filtering:
                # keep only the rows that survived
                df_cleaned[f"{col}_cluster"] = full_labels[global_keep_mask]
            else:
                # keep ALL rows; rows that would have been dropped get label –1
                labels_out = full_labels.copy()
                labels_out[np.isnan(labels_out)] = -1
                df_cleaned[f"{col}_cluster"] = labels_out
        # ───────────────────────────────────────────────────────────────

        print(f"✅ {name}: {df.shape[0]} → {df_cleaned.shape[0]} rows remaining after filtering.")

        print(f"⏱️ Total time elapsed: {time.time() - start_time:.2f} seconds.")

        cleaned_dfs[name] = df_cleaned

    return cleaned_dfs

# Thread-safe tqdm update lock
_tqdm_lock = threading.Lock()


class ProgressTree(DecisionTreeRegressor):
    def fit(self, *args, **kwargs):
        with _tqdm_lock:
            if hasattr(ProgressTree, 'bar') and ProgressTree.bar is not None:
                ProgressTree.bar.update()
        return super().fit(*args, **kwargs)


class ProgressIsolationForest(IsolationForest):
    def _make_estimator(self, append=True, random_state=None):
        estimator = ProgressTree(max_features=1.0)
        if append:
            self.estimators_.append(estimator)
        return estimator


def clean_multivariate_outliers_iforest_set(dataframes_dict, columns_to_check, contamination_values=None,
                                            outlier_threshold=15, filtering=True, verbose=False):
    if contamination_values is None:
        contamination_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1]

    cleaned_dfs = {}

    for name, df in dataframes_dict.items():

        print(f"\n🔹 Analyzing '{name}' dataset using Isolation Forest 🔹")
        try:
            df_copy = df.copy()
            X = df_copy[columns_to_check].dropna().astype(np.float64)

            if X.empty or len(X) < 10:
                if verbose:
                    print(f"⚠️ Not enough valid samples to apply Isolation Forest in '{name}'. Skipping.")
                cleaned_dfs[name] = df_copy
                continue
        except Exception as e:
            print(f"❌ Error preparing dataset '{name}': {e}")
            traceback.print_exc()
            cleaned_dfs[name] = df
            continue

        best_score = 0
        best_model = None
        best_preds = None
        best_contamination = None

        for c in contamination_values:
            if verbose:
                print(f"\n🔍 Trying contamination = {c:.4f}")
                bar = tqdm(total=50, desc=f"   → Training Forest ({c:.4f})", leave=False)
                ProgressTree.bar = bar

            try:
                model = ProgressIsolationForest(
                    contamination=c,
                    n_estimators=50,
                    max_samples=min(len(X), 5000),
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X)
                if verbose:
                    bar.close()
                    ProgressTree.bar = None
            except Exception as e:
                if verbose:
                    bar.close()
                    ProgressTree.bar = None
                print(f"❌ Error training model for contamination={c:.4f}: {e}")
                traceback.print_exc()
                continue

            try:
                preds = model.predict(X)
            except Exception as e:
                print(f"❌ Error during prediction with contamination={c:.4f}: {e}")
                traceback.print_exc()
                continue

            if len(np.unique(preds)) < 2:
                if verbose:
                    print(f"⚠️ Only one cluster found for contamination={c:.4f}, skipping.")
                continue

            try:
                score = silhouette_score(X, preds, sample_size=min(10000, X.shape[0]), random_state=42)
                if verbose:
                    print(f"   - Contamination={c:.4f} → Silhouette Score={score:.4f}")
            except Exception as e:
                print(f"❌ Silhouette score failed for contamination={c:.4f}: {e}")
                traceback.print_exc()
                continue

            if score > best_score:
                best_score = score
                best_model = model
                best_preds = preds
                best_contamination = c

        if best_preds is None:
            if verbose:
                print(f"❌ No valid clustering found for '{name}'. Defaulting to no outliers.")
            df_copy["Outlier_IForest"] = False
            cleaned_dfs[name] = df_copy
            continue

        try:
            outlier_flags = pd.Series(best_preds == -1, index=X.index)
            df_copy["Outlier_IForest"] = False
            df_copy.loc[outlier_flags.index, "Outlier_IForest"] = outlier_flags.values

            total_outliers = df_copy["Outlier_IForest"].sum()
            total_samples = df_copy.shape[0]
            pct_outliers = (total_outliers / total_samples) * 100

            if verbose:
                print(f"📊 Best contamination: {best_contamination:.4f}")
                print(f"📊 Total outliers: {total_outliers} / {total_samples} ({pct_outliers:.2f}%)")

            if filtering:
                if pct_outliers > outlier_threshold:
                    if verbose:
                        print(
                            f"⚠️ Too many outliers detected (> {outlier_threshold}%). Skipping filtering for '{name}'.")
                    cleaned_dfs[name] = df_copy
                else:
                    cleaned_df = df_copy[~df_copy["Outlier_IForest"]]
                    if verbose:
                        print(f"✅ {name}: After removing outliers, {cleaned_df.shape[0]} samples remain.")
                    cleaned_dfs[name] = cleaned_df
            else:
                if verbose:
                    print(f"🧪 Filtering disabled — returning data with 'Outlier_IForest' flags intact.")
                cleaned_dfs[name] = df_copy
        except Exception as e:
            print(f"❌ Error finalizing output for '{name}': {e}")
            traceback.print_exc()
            cleaned_dfs[name] = df_copy

    return cleaned_dfs
