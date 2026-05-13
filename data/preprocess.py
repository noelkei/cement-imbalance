import pandas as pd
from data.utils import describe_cols
from typing import List
import numpy as np
import pandas as pd
import math
from data.utils import log, load_column_mapping_by_group, apply_column_mapping

def clean_cement_dataframe(
    df_raw: pd.DataFrame,
    verbose: bool = False,
    col_threshold: float = 0.9,
    remove_cols: list[str] = None
) -> pd.DataFrame:
    df = df_raw.copy()
    original_shape = df.shape
    original_columns = df.columns.tolist()

    log("=== Cleaning Cement DataFrame ===", verbose)

    # Step 1: Clean column names
    old_columns = df.columns.tolist()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    renamed_columns = [(o, n) for o, n in zip(old_columns, df.columns.tolist()) if o != n]
    if renamed_columns:
        log(f"[1] Renamed columns:\n" + "\n".join([f" - '{o}' → '{n}'" for o, n in renamed_columns]), verbose)

    # 🆕 Step 1B: Apply anonymization mapping
    flat_mapping, grouped_mapping = load_column_mapping_by_group(verbose=verbose)
    df = apply_column_mapping(df, flat_mapping, verbose=verbose)

    # Step 1.5: Drop explicitly specified columns
    if remove_cols:
        matched_remove_cols = [col for col in remove_cols if col in df.columns]
        df.drop(columns=matched_remove_cols, inplace=True)
        if matched_remove_cols:
            log(f"[1.5] Dropped specified columns ({len(matched_remove_cols)}): {matched_remove_cols}", verbose)

    # Step 2: Convert 'date' to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        log(f"[2] Parsed 'date' to datetime. Null dates: {df['date'].isna().sum()}", verbose)

    # Step 3: Replace placeholder values with NaN
    placeholder_vals = ["", "N/A", "NA", "-", "null", "NULL", "nan", "#DIV/0!"]
    df.replace(placeholder_vals, np.nan, inplace=True)
    log(f"[3] Replaced placeholder values with NaN", verbose)

    # Step 4: Convert numeric-like object columns (with commas)
    converted_cols = []
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='raise')
            converted_cols.append(col)
        except Exception:
            continue
    if converted_cols:
        log(f"[4] Converted to numeric ({len(converted_cols)}): {converted_cols}", verbose)

    # Step 5: Convert remaining 'object' columns to string
    remaining_obj = df.select_dtypes(include='object').columns.tolist()
    df[remaining_obj] = df[remaining_obj].astype(str)
    if remaining_obj:
        log(f"[5] Converted to string ({len(remaining_obj)}): {remaining_obj}", verbose)

    # Step 6: Drop constant columns
    constant_cols = df.columns[df.nunique(dropna=False) <= 1].tolist()
    df.drop(columns=constant_cols, inplace=True)
    if constant_cols:
        log(f"[6] Dropped constant columns ({len(constant_cols)}): {constant_cols}", verbose)

    # Step 7: Drop columns below col_threshold non-NA
    col_threshold_count = int(col_threshold * len(df))
    below_thresh_cols = df.columns[df.isna().sum() > (len(df) - col_threshold_count)].tolist()
    df.drop(columns=below_thresh_cols, inplace=True)
    if below_thresh_cols:
        log(f"[7] Dropped cols with < {int(col_threshold*100)}% non-NA ({len(below_thresh_cols)}): {below_thresh_cols}", verbose)

    # Step 8: Drop rows with any NaN
    nan_cols = df.columns[df.isna().any()].tolist()
    rows_before = len(df)
    df.dropna(axis=0, how="any", inplace=True)
    rows_after = len(df)
    if nan_cols:
        log(f"[8] Dropped rows with NaN in cols: {nan_cols}", verbose)
        log(f"[8] Rows dropped: {rows_before - rows_after}", verbose)

    # Step 9: Reset index
    df.reset_index(drop=True, inplace=True)

    # Step 10: Summary
    log("\n=== Cleaning Summary ===", verbose)
    log(f"Original shape: {original_shape} → Cleaned shape: {df.shape}", verbose)
    log(f"Columns before: {len(original_columns)} → after: {df.shape[1]}", verbose)
    describe_cols(df, "FINAL", verbose)

    return df

def compute_loss_of_ignition(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    df_out = df.copy()

    loi_col = 'loi'
    loi_calc_col = 'loi_calc'

    # Check presence of required columns
    missing = [col for col in [loi_col, loi_calc_col] if col not in df_out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert to numeric safely
    loi = pd.to_numeric(df_out[loi_col], errors='coerce')
    loi_calc = pd.to_numeric(df_out[loi_calc_col], errors='coerce')

    # Masks
    use_loi = loi > 0
    use_loi_calc = ~use_loi & (loi_calc > 0)
    fallback = ~(use_loi | use_loi_calc)

    # Build new column
    df_out['loss_of_ignition'] = np.zeros(len(df_out), dtype='float64')
    df_out.loc[use_loi, 'loss_of_ignition'] = loi[use_loi]
    df_out.loc[use_loi_calc, 'loss_of_ignition'] = loi_calc[use_loi_calc]

    # Drop original columns
    df_out.drop(columns=[loi_col, loi_calc_col], inplace=True)

    # Logging
    log(f"[🔥] Created 'loss_of_ignition' from '{loi_col}' and '{loi_calc_col}'", verbose)
    log(f"[🔥] Used 'loi' for {use_loi.sum()} rows", verbose)
    log(f"[🔥] Used 'loi_calc' for {use_loi_calc.sum()} rows", verbose)
    log(f"[🔥] Defaulted to 0 for {fallback.sum()} rows", verbose)
    log(f"[🔥] Final stats for 'loss_of_ignition':\n{df_out['loss_of_ignition'].describe()}", verbose)

    return df_out

def check_all_float_positive(
    df: pd.DataFrame,
    drop_invalid: bool = False,
    verbose: bool = False,
    excluded_cols: list[str] = None
) -> pd.DataFrame:
    df_out = df.copy()
    excluded_cols = excluded_cols or []

    # Identify float64 columns, excluding any specified
    float_cols = [
        col for col in df_out.select_dtypes(include='float64').columns
        if col not in excluded_cols
    ]
    non_float_cols = [col for col in df_out.columns if col not in float_cols]

    # Basic logging
    log(f"[🔍] Checking float64 columns for ≥ 0 (excluding: {excluded_cols}): {float_cols}", verbose)
    log(f"[ℹ️] Skipped non-float64 or excluded columns: {non_float_cols}", verbose)

    rows_before = len(df_out)

    # Mask for invalid rows (any float column < 0)
    invalid_mask = (df_out[float_cols] < 0).any(axis=1)
    n_invalid = invalid_mask.sum()

    if n_invalid > 0:
        log(f"[🚫] {n_invalid} rows have at least one negative float value", verbose)

        if drop_invalid:
            df_out = df_out.loc[~invalid_mask].reset_index(drop=True)
            rows_after = len(df_out)
            log(f"[🧹] Dropped {n_invalid} rows", verbose)
            log(f"[📊] Rows before: {rows_before} → after: {rows_after}", verbose)
    else:
        log(f"[✅] All float64 values are ≥ 0", verbose)
        log(f"[📊] Rows unchanged: {rows_before}", verbose)

    return df_out


def filter_by_range(
    df: pd.DataFrame,
    cols: list[str],
    lower: float = float('-inf'),
    upper: float = float('inf'),
    drop_invalid: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    df_out = df.copy()
    rows_before = len(df_out)

    # Validate columns
    valid_cols = [col for col in cols if col in df_out.columns]
    invalid_cols = [col for col in cols if col not in df_out.columns]

    if invalid_cols:
        log(f"[⚠️] Skipped missing columns: {invalid_cols}", verbose)
    if not valid_cols:
        log(f"[❌] No valid columns to filter. Skipping.", verbose)
        return df_out

    log(f"[🔍] Filtering rows with values outside [{lower}, {upper}] in columns: {valid_cols}", verbose)

    # Create mask for out-of-bound values
    out_of_bounds_mask = (df_out[valid_cols] < lower) | (df_out[valid_cols] > upper)
    invalid_rows = out_of_bounds_mask.any(axis=1)
    n_invalid = invalid_rows.sum()

    if n_invalid > 0:
        pct_lost = (n_invalid / rows_before) * 100
        pct_kept = 100 - pct_lost

        log(f"[🚫] {n_invalid} rows have out-of-bound values", verbose)
        log(f"[📉] % lost: {pct_lost:.2f}% — % kept: {pct_kept:.2f}%", verbose)

        if drop_invalid:
            df_out = df_out.loc[~invalid_rows].reset_index(drop=True)
            rows_after = len(df_out)
            log(f"[🧹] Dropped {n_invalid} rows", verbose)
            log(f"[📊] Rows before: {rows_before} → after: {rows_after}", verbose)
    else:
        log(f"[✅] All values in specified columns are within bounds", verbose)
        log(f"[📊] Rows unchanged: {rows_before}", verbose)

    return df_out


def sum_columns(
    df: pd.DataFrame,
    cols: list[str],
    name: str,
    verbose: bool = True
) -> pd.DataFrame:
    df_out = df.copy()
    new_col = name.lower()

    # Check for missing columns
    missing = [col for col in cols if col not in df_out.columns]
    if missing:
        log(f"[⚠️] Skipped missing columns: {missing}", verbose)
        cols = [col for col in cols if col in df_out.columns]

    if not cols:
        log(f"[❌] No valid columns to sum. Skipping.", verbose)
        return df_out

    # Sum the selected columns
    df_out[new_col] = df_out[cols].sum(axis=1)

    log(f"[➕] Added column '{new_col}' as the sum of: {cols}", verbose)
    log(f"[📊] '{new_col}' summary:\n{df_out[new_col].describe()}", verbose)

    return df_out

def apply_deterministic_preprocessing(
    df: pd.DataFrame,
    ranges: dict[str, dict],
    drop_invalid: bool = True,
    verbose: bool = True,
    threshold: float = 0.995,
    excluded_cols: list[str] = None,
) -> pd.DataFrame:
    df_out = df.copy()

    log("🚀 Starting cleaning pipeline...", verbose)

    # Step 1: Compute loss_of_ignition
    df_out = compute_loss_of_ignition(df_out, verbose)

    # Step 2: Check all float64 values are ≥ 0 (excluding some)
    df_out = check_all_float_positive(
        df_out,
        drop_invalid=drop_invalid,
        verbose=verbose,
        excluded_cols=excluded_cols
    )

    # Step 3: Process feature groups
    for key, settings in ranges.items():
        cols = settings.get("cols", [])
        lower = settings.get("lower", float("-inf"))
        upper = settings.get("upper", float("inf"))
        sum_col_name = f"sum_{key}"

        log(f"\n🔎 Processing group: '{key}'", verbose)

        # 3a: Filter by range per column
        df_out = filter_by_range(
            df_out, cols=cols, lower=lower, upper=upper,
            drop_invalid=drop_invalid, verbose=verbose
        )

        # 3b: Sum columns
        df_out = sum_columns(
            df_out, cols=cols, name=sum_col_name, verbose=verbose
        )

    # Step 4: Domain quality filter on compositional sums.
    # This stays in the global deterministic preprocessing stage because it is
    # treated as an accepted physical/chemical consistency check, not as the
    # learned statistical cleaning that is fit later on the official TRAIN split.
    log("\n📉 Applying domain quality filtering for all 'sum_' columns...", verbose)
    combined_mask = pd.Series([False] * len(df_out), index=df_out.index)

    for col in df_out.columns:
        if col.startswith("sum_") and pd.api.types.is_numeric_dtype(df_out[col]):
            alpha = (1 - threshold) / 2
            lower = df_out[col].quantile(alpha)
            upper = df_out[col].quantile(1 - alpha)

            mask = (df_out[col] < lower) | (df_out[col] > upper)
            n_out = mask.sum()

            log(f"[📈] {col}: [{lower:.4f}, {upper:.4f}] → outliers: {n_out}", verbose)

            combined_mask |= mask

    total_outliers = combined_mask.sum()
    if total_outliers > 0:
        log(f"[🧹] Dropping {total_outliers} rows based on sum_* bounds", verbose)
        df_out = df_out.loc[~combined_mask].reset_index(drop=True)
    else:
        log("[✅] No rows removed by sum_* bounds", verbose)

    log(f"\n✅ Cleaning pipeline completed. Final shape: {df_out.shape}", verbose)
    return df_out


def apply_cleaning_pipeline(
    df: pd.DataFrame,
    ranges: dict[str, dict],
    drop_invalid: bool = True,
    verbose: bool = True,
    threshold: float = 0.995,
    excluded_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Legacy alias kept for backward compatibility.

    F3 canonical usage should prefer `apply_deterministic_preprocessing()` to
    make it explicit that this stage does not learn from the official split.
    """
    return apply_deterministic_preprocessing(
        df=df,
        ranges=ranges,
        drop_invalid=drop_invalid,
        verbose=verbose,
        threshold=threshold,
        excluded_cols=excluded_cols,
    )

def filter_by_type_and_summarize(
    df: pd.DataFrame,
    selected_types: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filters the DataFrame to keep only selected types.
    Logs value counts (absolute & percentage) and date ranges for each type.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'type' and 'date' columns.
        selected_types (List[str]): List of type values to retain.
        verbose (bool): Whether to log details.

    Returns:
        pd.DataFrame: Filtered DataFrame with only the selected types.
    """
    df_filtered = df[df['type'].isin(selected_types)].reset_index(drop=True)

    # Log absolute and normalized value counts
    log("\n🔢 Value counts for selected 'type's:", verbose)
    log(df_filtered['type'].value_counts().to_string(), verbose)
    log("\n📊 Normalized value counts (percentages):", verbose)
    log((df_filtered['type'].value_counts(normalize=True) * 100).round(2).astype(str) + "%", verbose)

    # Log date ranges per type
    log("\n🗓️ Date range per selected type:", verbose)
    for t in selected_types:
        subset = df_filtered[df_filtered['type'] == t]
        first_date = subset['date'].min()
        last_date = subset['date'].max()
        log(f" - {t}: {first_date.date()} → {last_date.date()}", verbose)

    return df_filtered

def ilr_transform_groups(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    verbose: bool = True,
    pseudocount: float = 1e-6,
    normalize: bool = True
) -> pd.DataFrame:
    df_out = df.copy()

    for key, cols in groups.items():
        X = df_out[cols].astype(float).values

        # Optional normalization to ensure compositional context
        if normalize:
            X = X / X.sum(axis=1, keepdims=True)

        X[X == 0] = pseudocount

        n_components = X.shape[1]
        ilr_data = np.zeros((X.shape[0], n_components - 1))

        for i in range(1, n_components):
            num = X[:, i - 1]
            denom = np.exp(np.mean(np.log(X[:, i:]), axis=1))
            coef = math.sqrt(i / (i + 1))
            ilr_data[:, i - 1] = coef * np.log(num / denom)

        ilr_cols = [f"ilr_{key}_{i+1}" for i in range(ilr_data.shape[1])]
        for col_name, col_values in zip(ilr_cols, ilr_data.T):
            df_out[col_name] = col_values

        log(f"[🔁] Transformed '{key}': {len(cols)} → {len(ilr_cols)} ILR features", verbose)

    return df_out

def ilr_inverse_groups(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    scale_factors: dict[str, np.ndarray] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Invert ILR transforms for multiple groups.
    Returns the original DataFrame with inv_{group}_{i} columns appended.
    """
    df_out = df.copy()

    for key in groups:
        ilr_cols = [col for col in df.columns if col.startswith(f"ilr_{key}_")]
        ilr_array = df[ilr_cols].values

        n_samples, D_minus_1 = ilr_array.shape
        D = D_minus_1 + 1
        comp_array = np.zeros((n_samples, D))

        # If no external scale factors, use sum of original components for this group
        if scale_factors is not None and key in scale_factors:
            scales = scale_factors[key]
        else:
            scales = df[groups[key]].sum(axis=1).values  # <-- updated

        for i in range(n_samples):
            y = ilr_array[i]
            z = np.zeros(D)
            z[-1] = 1.0  # last component

            for j in reversed(range(D - 1)):
                coef = math.sqrt((j + 1) / (j + 2))
                gmean = np.exp(np.mean(np.log(z[j + 1:])))
                z[j] = gmean * np.exp(y[j] / coef)

            comp_array[i] = scales[i] * z / np.sum(z)

        inv_cols = [f"inv_{key}_{i+1}" for i in range(D)]
        for col_name, col_values in zip(inv_cols, comp_array.T):
            df_out[col_name] = col_values

        log(f"[🔁] Inverted ILR for '{key}': {len(inv_cols)} components", verbose)

    return df_out

def normalize_groups(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Normalize values in each compositional group so that their row-wise sum is 1.
    Appends new columns prefixed with 'norm_' for each input column.

    Parameters:
        df: Input DataFrame
        groups: Dict of {group_name: [columns]}
        verbose: Whether to log progress

    Returns:
        DataFrame with normalized columns added.
    """
    df_out = df.copy()

    for key, cols in groups.items():
        missing = [col for col in cols if col not in df_out.columns]
        if missing:
            log(f"[⚠️] Skipping group '{key}' — missing columns: {missing}", verbose)
            continue

        X = df_out[cols].astype(float).values
        row_sums = X.sum(axis=1, keepdims=True)

        # Avoid division by zero: only normalize rows with positive sum
        with np.errstate(divide='ignore', invalid='ignore'):
            X_normalized = np.divide(X, row_sums, where=row_sums != 0)

        for i, col in enumerate(cols):
            df_out[f"norm_{col}"] = X_normalized[:, i]

        log(f"[✅] Normalized group '{key}' — {len(cols)} columns → 'norm_' prefixed", verbose)

    return df_out







