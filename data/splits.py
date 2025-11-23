import numpy as np
import pandas as pd
from data.cleaning import (
    clean_univariate_bootstrap_density_weighted,
    clean_multivariate_outliers_iforest_set,
    load_processed_data
)
from data.utils import log, load_column_mapping_by_group, load_type_mapping
from training.utils import ROOT_PATH

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

