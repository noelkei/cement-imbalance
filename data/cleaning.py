import pandas as pd
import numpy as np
from data.utils import log
import time
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, grey_closing
import matplotlib.pyplot as plt
import math
from training.utils import ROOT_PATH
from data.preprocess import (
    clean_cement_dataframe,
    apply_cleaning_pipeline,
    filter_by_type_and_summarize,
    ilr_transform_groups,
    normalize_groups
)
from data.utils import load_column_mapping_by_group
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeRegressor
import threading
import traceback


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

    # Step 1: Load raw CSV
    raw_path = ROOT_PATH / "data" / "raw"
    raw_file = raw_path / "bd_CALIDAD_CALUCEM_KM_MB(all_data_for_modeling)_13_07_2025.csv"
    df_raw_full = pd.read_csv(raw_file, sep=";", engine="python")

    # Step 2: Filter top types and accepted processes immediately
    top_types = ["ISTRA 40", "ISTRA 50", "Lumnite SG"]
    valid_processes = ["otprema", "meljava"]

    df_raw = df_raw_full[
        df_raw_full["Type"].isin(top_types) &
        df_raw_full["Process"].str.lower().isin(valid_processes)
        ].copy()

    shape_raw = df_raw.shape
    cols_raw = set(df_raw.columns)

    print(shape_raw)

    # Step 3: Clean raw data
    cols_to_remove = ["customer", "mill", "fe2o3uk"]
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
    df_pre_outliers = apply_cleaning_pipeline(
        df_clean,
        ranges=ranges,
        drop_invalid=True,
        verbose=verbose,
        threshold=0.99,
        excluded_cols=exclude_cols
    )

    # Step 6: Filter by type again for summary logging
    df_filtered = filter_by_type_and_summarize(df_pre_outliers, top_types, verbose=verbose)

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
    original_counts = df_raw["Type"].value_counts()
    final_counts = df_normalized["type"].value_counts()
    log("\n📎 Per-type row retention:", verbose)
    for t in final_counts.index:
        original = original_counts.get(t, 0)
        final = final_counts[t]
        pct = 100 * (original - final) / original if original > 0 else 0.0
        log(f"  → {t}: {original} → {final} ({pct:.2f}% removed)", verbose)

    return df_normalized



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
