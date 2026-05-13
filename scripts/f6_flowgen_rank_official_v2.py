#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import f6_flowgen_rank_official as hist


OFFICIAL_FLOWGEN_ROOT = hist.OFFICIAL_FLOWGEN_ROOT
DEFAULT_OUTPUT_DIR = OFFICIAL_FLOWGEN_ROOT / "campaign_summaries" / "final_rankings"

DEFAULT_TOP_K = 5
DEFAULT_ELIGIBLE_TOP_FRACTION = 1.0 / 3.0
DEFAULT_SELECTION_CLOSE_MARGIN = 0.05
EPS = 1.0e-9

# Temporal sub-scores are mapped with the same continuous-points idea as the
# historical banded lens, then compressed to a banded-compatible scale.
TEMPORAL_POINT_DIVISOR = 10.0
TEMPORAL_BAND_CUTS = {
    "ratio25": [0.08, 0.15, 0.22, 0.30, 0.42, 0.50],
    "ratio50": [0.44, 0.46, 0.48, 0.50, 0.55, 0.60],
    "q3_peak": [1.22, 1.24, 1.26, 1.28, 1.30, 1.33],
}

TEMPORAL_SUFFIX_WEIGHTS = {"t50": 0.70, "t25": 0.30}
VAL_TEMPORAL_GUARD_WEIGHTS = {
    "banded_val": 0.50,
    "temporal_suffix": 0.30,
    "tq3": 0.20,
}
TRAIN_STRONG_WEIGHTS = {"banded_train": 0.60, "banded_val": 0.40}

FINAL_LENS_SPECS = {
    "TrainStrong60": {"score_col": "trainstrong60_score", "ascending": False},
    "ValTemporalGuard": {"score_col": "val_temporal_guard_score", "ascending": False},
    "banded": {"score_col": "banded_score", "ascending": False},
    "classic": {"score_col": "classic_score", "ascending": True},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final closure ranking for official FlowGen runs. "
            "This keeps the historical v6-style ranking intact and adds a new "
            "train-strong plus val+temporal-aware selection layer."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=OFFICIAL_FLOWGEN_ROOT,
        help="Official FlowGen root to scan.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the final ranking artifacts will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k rows to export for each final lens.",
    )
    parser.add_argument(
        "--eligible-top-fraction",
        type=float,
        default=DEFAULT_ELIGIBLE_TOP_FRACTION,
        help="Fraction of runs kept in the ValTemporalGuard-eligible set.",
    )
    parser.add_argument(
        "--selection-close-margin",
        type=float,
        default=DEFAULT_SELECTION_CLOSE_MARGIN,
        help=(
            "TrainStrong60 margin used to define near-tied finalists before "
            "ValTemporalGuard/banded/classic tie-breaking."
        ),
    )
    return parser.parse_args()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_version(run_id: str) -> str | None:
    match = re.search(r"_v(\d+)$", str(run_id))
    return None if match is None else f"v{match.group(1)}"


def _base_token(run_id: str) -> str | None:
    for token in ("c1", "c2"):
        if f"_{token}_" in str(run_id):
            return token
    return None


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values:
        if not np.isfinite(value):
            continue
        numerator += float(weight) * float(value)
        denominator += float(weight)
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _continuous_points_from_cuts(
    *,
    value: float,
    cuts: list[float],
    obs_max: float,
) -> tuple[str, float]:
    if not np.isfinite(value):
        return "nan", float("nan")

    label = "unusable"
    for idx, upper_bound in enumerate(cuts):
        if value <= upper_bound:
            label = hist.LABELS_BEST_FIRST[idx]
            break

    bounds = [0.0] + list(cuts) + [float("inf")]
    bounds_index = hist.LABEL_TO_BOUNDS_INDEX[label]
    lower_bound = bounds[bounds_index]
    upper_bound = bounds[bounds_index + 1]

    if label == "unusable":
        if np.isfinite(obs_max) and obs_max > lower_bound:
            upper_bound = obs_max
        elif lower_bound > 0:
            upper_bound = lower_bound * 1.25
        else:
            upper_bound = 1.0

    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound) or upper_bound <= lower_bound:
        current_idx = hist.RANK_TO_IDX[label]
        return label, hist._points_for_rank_idx(current_idx)

    current_idx = hist.RANK_TO_IDX[label]
    worse_idx = max(0, current_idx - 1)
    current_pts = hist._points_for_rank_idx(current_idx)
    worse_pts = hist._points_for_rank_idx(worse_idx)
    interp = float(np.clip((upper_bound - value) / (upper_bound - lower_bound), 0.0, 1.0))
    pts = worse_pts + interp * (current_pts - worse_pts)
    return label, pts


def _extract_temporal_metrics(results: dict[str, Any]) -> dict[str, float]:
    prefix25 = hist._to_float(
        hist._nested_get(
            results,
            "val.temporal_realism.prefix_suffix.summary.generated_vs_slice.overall_w1_suffix25_minus_prefix25",
        )
    )
    prefix50 = hist._to_float(
        hist._nested_get(
            results,
            "val.temporal_realism.prefix_suffix.summary.generated_vs_slice.overall_w1_suffix50_minus_prefix50",
        )
    )
    baseline25 = hist._to_float(
        hist._nested_get(
            results,
            "val.temporal_realism.prefix_suffix.summary.train_ref_vs_slice_real.overall_w1_suffix25_minus_prefix25",
        )
    )
    baseline50 = hist._to_float(
        hist._nested_get(
            results,
            "val.temporal_realism.prefix_suffix.summary.train_ref_vs_slice_real.overall_w1_suffix50_minus_prefix50",
        )
    )

    ratio25 = float("nan")
    ratio50 = float("nan")
    if np.isfinite(prefix25) and np.isfinite(baseline25) and baseline25 > EPS:
        ratio25 = float(prefix25 / baseline25)
    if np.isfinite(prefix50) and np.isfinite(baseline50) and baseline50 > EPS:
        ratio50 = float(prefix50 / baseline50)

    q1 = hist._to_float(
        hist._nested_get(
            results,
            "val.temporal_realism.quartiles.slices.q1.generated_vs_slice.overall.w1_mean",
        )
    )
    q2 = hist._to_float(
        hist._nested_get(
            results,
            "val.temporal_realism.quartiles.slices.q2.generated_vs_slice.overall.w1_mean",
        )
    )
    q3 = hist._to_float(
        hist._nested_get(
            results,
            "val.temporal_realism.quartiles.slices.q3.generated_vs_slice.overall.w1_mean",
        )
    )
    q4 = hist._to_float(
        hist._nested_get(
            results,
            "val.temporal_realism.quartiles.slices.q4.generated_vs_slice.overall.w1_mean",
        )
    )

    q3_peak = float("nan")
    others = np.array([q1, q2, q4], dtype=float)
    if np.isfinite(q3) and np.isfinite(others).all():
        baseline = float(np.mean(others))
        if baseline > EPS:
            q3_peak = float(q3 / baseline)

    return {
        "temporal_delta25_w1": prefix25,
        "temporal_delta50_w1": prefix50,
        "temporal_baseline25_w1": baseline25,
        "temporal_baseline50_w1": baseline50,
        "temporal_ratio25": ratio25,
        "temporal_ratio50": ratio50,
        "temporal_q1_w1": q1,
        "temporal_q2_w1": q2,
        "temporal_q3_w1": q3,
        "temporal_q4_w1": q4,
        "temporal_q3_peak": q3_peak,
    }


def _banded_score_for_splits(
    results: dict[str, Any],
    *,
    obs_max: dict[str, float],
    split_weights: dict[str, float],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for split, split_weight in split_weights.items():
        split_weight = float(split_weight)
        if split_weight <= 0:
            continue

        overall_weight = float(hist.W_SCOPE["overall"])
        for component in hist.COMPONENTS:
            component_weight = float(hist.W_COMPONENT[component])
            base_weight = split_weight * overall_weight * component_weight
            for metric in hist.METRICS6:
                value = hist._get_results_value(results, split, "overall", component, metric, None)
                _, points = hist._continuous_points(metric, value, obs_max=obs_max[metric])
                if np.isfinite(points):
                    weight = base_weight * hist.METRIC_WEIGHT[metric]
                    numerator += weight * points
                    denominator += weight

        perclass_weight = float(hist.W_SCOPE["perclass"])
        for cls in hist.CLASSES:
            class_weight = float(hist.W_CLASS_NORM[int(cls)])
            for component in hist.COMPONENTS:
                component_weight = float(hist.W_COMPONENT[component])
                base_weight = split_weight * perclass_weight * class_weight * component_weight
                for metric in hist.METRICS6:
                    value = hist._get_results_value(results, split, "perclass", component, metric, int(cls))
                    _, points = hist._continuous_points(metric, value, obs_max=obs_max[metric])
                    if np.isfinite(points):
                        weight = base_weight * hist.METRIC_WEIGHT[metric]
                        numerator += weight * points
                        denominator += weight

    return round(float(numerator / denominator), 3) if denominator > 0 else float("nan")


def _rank_tables(
    df: pd.DataFrame,
    *,
    lens_specs: dict[str, dict[str, Any]],
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    ranked = df.copy()
    top_tables: dict[str, pd.DataFrame] = {}

    for lens_name, spec in lens_specs.items():
        ordered = hist._ordered_by_lens(
            ranked,
            score_col=str(spec["score_col"]),
            ascending=bool(spec["ascending"]),
        )
        rank_map = {run_id: idx + 1 for idx, run_id in enumerate(ordered["run_id"].tolist())}
        ranked[f"rank_{lens_name}"] = ranked["run_id"].map(rank_map)
        top_tables[lens_name] = ordered.head(top_k).copy()

    return ranked, top_tables


def _selection_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values(
        by=["trainstrong60_score", "val_temporal_guard_score", "banded_score", "classic_score", "run_id"],
        ascending=[False, False, False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    top_k = int(args.top_k)
    eligible_top_fraction = float(args.eligible_top_fraction)
    selection_close_margin = float(args.selection_close_margin)

    if not root.exists():
        raise FileNotFoundError(f"Official FlowGen root does not exist: {root}")
    if not (0.0 < eligible_top_fraction <= 1.0):
        raise ValueError("--eligible-top-fraction must be in (0, 1].")
    if selection_close_margin < 0.0:
        raise ValueError("--selection-close-margin must be >= 0.")

    run_dirs, skipped_rows, non_run_roots = hist._discover_candidate_run_dirs(root)

    valid_records: list[hist.RunRecord] = []
    for run_dir in run_dirs:
        record, invalid_row = hist._load_run_record(run_dir)
        if invalid_row is not None:
            skipped_rows.append(invalid_row)
            continue
        if record is None:
            continue
        if not hist._has_any_finite_metric(record.results):
            skipped_rows.append(
                {
                    "status": "invalid_run",
                    "run_dir": str(run_dir),
                    "run_id": record.run_id,
                    "reason": "results.yaml does not expose any finite realism metric",
                }
            )
            continue
        valid_records.append(record)

    stamp = _utc_stamp()
    prefix = f"flowgen_official_final_rankings_v2_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped_csv = output_dir / f"{prefix}_skipped_items.csv"
    skipped_json = output_dir / f"{prefix}_skipped_items.json"
    hist._write_csv(skipped_csv, skipped_rows)
    hist._write_json(skipped_json, {"rows": skipped_rows})

    if not valid_records:
        summary_payload = {
            "root": str(root),
            "output_dir": str(output_dir),
            "timestamp_utc": stamp,
            "top_k": top_k,
            "eligible_top_fraction": eligible_top_fraction,
            "selection_close_margin": selection_close_margin,
            "valid_run_count": 0,
            "skipped_count": len(skipped_rows),
            "generated_files": {
                "skipped_csv": str(skipped_csv),
                "skipped_json": str(skipped_json),
            },
        }
        summary_path = output_dir / f"{prefix}_summary.json"
        hist._write_json(summary_path, summary_payload)
        print(f"No valid official FlowGen runs found under {root}.")
        print(f"Skipped inventory: {skipped_csv}")
        print(f"Summary: {summary_path}")
        return 0

    obs_max = hist._compute_obs_max(valid_records)
    rows = [hist._build_row(record, obs_max) for record in valid_records]
    df = pd.DataFrame(rows)

    extra_rows: list[dict[str, Any]] = []
    for record in valid_records:
        temporal = _extract_temporal_metrics(record.results)
        extra_rows.append(
            {
                "run_id": record.run_id,
                "base_token": _base_token(record.run_id),
                "run_version": _run_version(record.run_id),
                "banded_train": _banded_score_for_splits(
                    record.results,
                    obs_max=obs_max,
                    split_weights={"train": 1.0},
                ),
                "banded_val": _banded_score_for_splits(
                    record.results,
                    obs_max=obs_max,
                    split_weights={"val": 1.0},
                ),
                **temporal,
            }
        )
    extra_df = pd.DataFrame(extra_rows)
    merged = df.merge(extra_df, on="run_id", how="left")

    temporal_obs_max = {
        metric: float(np.nanmax(pd.to_numeric(merged[f"temporal_{metric}"], errors="coerce")))
        for metric in TEMPORAL_BAND_CUTS
    }
    for metric, cuts in TEMPORAL_BAND_CUTS.items():
        labels: list[str] = []
        points: list[float] = []
        for value in pd.to_numeric(merged[f"temporal_{metric}"], errors="coerce").tolist():
            label, raw_points = _continuous_points_from_cuts(
                value=float(value),
                cuts=cuts,
                obs_max=temporal_obs_max[metric],
            )
            labels.append(label)
            points.append(
                float(raw_points / TEMPORAL_POINT_DIVISOR) if np.isfinite(raw_points) else float("nan")
            )
        merged[f"temporal_{metric}_label"] = labels
        merged[f"temporal_{metric}_points"] = points

    merged["temporal_suffix_score"] = [
        _weighted_mean(
            [
                (row["temporal_ratio50_points"], TEMPORAL_SUFFIX_WEIGHTS["t50"]),
                (row["temporal_ratio25_points"], TEMPORAL_SUFFIX_WEIGHTS["t25"]),
            ]
        )
        for _, row in merged.iterrows()
    ]
    merged["trainstrong60_score"] = [
        _weighted_mean(
            [
                (row["banded_train"], TRAIN_STRONG_WEIGHTS["banded_train"]),
                (row["banded_val"], TRAIN_STRONG_WEIGHTS["banded_val"]),
            ]
        )
        for _, row in merged.iterrows()
    ]
    merged["val_temporal_guard_score"] = [
        _weighted_mean(
            [
                (row["banded_val"], VAL_TEMPORAL_GUARD_WEIGHTS["banded_val"]),
                (row["temporal_suffix_score"], VAL_TEMPORAL_GUARD_WEIGHTS["temporal_suffix"]),
                (row["temporal_q3_peak_points"], VAL_TEMPORAL_GUARD_WEIGHTS["tq3"]),
            ]
        )
        for _, row in merged.iterrows()
    ]

    ranked_df, top_tables = _rank_tables(merged, lens_specs=FINAL_LENS_SPECS, top_k=top_k)

    eligible_rank_cutoff = max(1, int(math.ceil(len(ranked_df) * eligible_top_fraction)))
    ranked_df["eligible_by_val_temporal_guard"] = (
        pd.to_numeric(ranked_df["rank_ValTemporalGuard"], errors="coerce") <= eligible_rank_cutoff
    )

    eligible_df = _selection_sort(ranked_df[ranked_df["eligible_by_val_temporal_guard"]].copy())
    if eligible_df.empty:
        raise RuntimeError("Eligible set is empty after applying ValTemporalGuard cutoff.")

    val_guard_cutoff_score = float(eligible_df["val_temporal_guard_score"].min())
    best_trainstrong60 = float(eligible_df["trainstrong60_score"].max())
    finalists_df = eligible_df[
        pd.to_numeric(eligible_df["trainstrong60_score"], errors="coerce")
        >= (best_trainstrong60 - selection_close_margin)
    ].copy()
    finalists_df = _selection_sort(finalists_df)
    winner_df = finalists_df.head(1).copy()
    winner_run_id = str(winner_df.iloc[0]["run_id"])

    ranked_df["selection_within_close_margin"] = ranked_df["run_id"].isin(finalists_df["run_id"].tolist())
    ranked_df["selection_final_winner"] = ranked_df["run_id"] == winner_run_id

    all_runs_csv, all_runs_json = hist._write_table_pair(output_dir / f"{prefix}_all_runs", ranked_df)
    eligible_csv, eligible_json = hist._write_table_pair(output_dir / f"{prefix}_eligible_set", eligible_df)
    finalists_csv, finalists_json = hist._write_table_pair(
        output_dir / f"{prefix}_selection_finalists",
        finalists_df,
    )
    winner_csv, winner_json = hist._write_table_pair(output_dir / f"{prefix}_winner", winner_df)

    lens_output_paths: dict[str, dict[str, str]] = {}
    for lens_name, table in top_tables.items():
        csv_path, json_path = hist._write_table_pair(output_dir / f"{prefix}_top{top_k}_{lens_name}", table)
        lens_output_paths[lens_name] = {"csv": str(csv_path), "json": str(json_path)}

    summary_payload = {
        "root": str(root),
        "output_dir": str(output_dir),
        "timestamp_utc": stamp,
        "top_k": top_k,
        "eligible_top_fraction": eligible_top_fraction,
        "eligible_rank_cutoff": eligible_rank_cutoff,
        "eligible_val_temporal_guard_min_score": val_guard_cutoff_score,
        "selection_close_margin": selection_close_margin,
        "valid_run_count": len(valid_records),
        "skipped_count": len(skipped_rows),
        "detected_non_run_root_dirs": sorted(non_run_roots),
        "historical_ranking_script": str(Path(hist.__file__).resolve()),
        "final_lens_specs": FINAL_LENS_SPECS,
        "temporal_band_cuts": TEMPORAL_BAND_CUTS,
        "temporal_point_divisor": TEMPORAL_POINT_DIVISOR,
        "trainstrong60_formula": {
            "formula": "0.60 * B(train) + 0.40 * B(val)",
            "weights": TRAIN_STRONG_WEIGHTS,
            "banded_metric_paths": {
                "overall": "{split}.realism.overall.{ks_mean,ks_median,w1_mean,w1_median,mmd2_rvs,mmd2_rvr_med}",
                "x": "{split}.realism.x.{ks_mean,ks_median,w1_mean,w1_median,mmd2_rvs,mmd2_rvr_med}",
                "y": "{split}.realism.y.{ks_mean,ks_median,w1_mean,w1_median,mmd2_rvs,mmd2_rvr_med}",
                "per_class": "{split}.realism.per_class.{class}.{overall,x,y}.{ks_mean,ks_median,w1_mean,w1_median,mmd2_rvs,mmd2_rvr_med}",
            },
            "mmd2_gap_definition": "max(mmd2_rvs - mmd2_rvr_med, 0)",
        },
        "val_temporal_guard_formula": {
            "formula": "0.50 * B(val) + 0.30 * TemporalSuffix + 0.20 * TQ3",
            "weights": VAL_TEMPORAL_GUARD_WEIGHTS,
            "temporal_suffix_formula": "0.70 * T50 + 0.30 * T25",
            "temporal_suffix_weights": TEMPORAL_SUFFIX_WEIGHTS,
            "ratio25_path": (
                "val.temporal_realism.prefix_suffix.summary.generated_vs_slice.overall_w1_suffix25_minus_prefix25 / "
                "val.temporal_realism.prefix_suffix.summary.train_ref_vs_slice_real.overall_w1_suffix25_minus_prefix25"
            ),
            "ratio50_path": (
                "val.temporal_realism.prefix_suffix.summary.generated_vs_slice.overall_w1_suffix50_minus_prefix50 / "
                "val.temporal_realism.prefix_suffix.summary.train_ref_vs_slice_real.overall_w1_suffix50_minus_prefix50"
            ),
            "q3_peak_path": (
                "val.temporal_realism.quartiles.slices.q3.generated_vs_slice.overall.w1_mean / "
                "mean(q1,q2,q4 generated_vs_slice overall w1_mean)"
            ),
            "ratio_denominator_guard": f"baseline must be finite and > {EPS}",
        },
        "selection_policy": {
            "eligible_rule": "rank_ValTemporalGuard <= eligible_rank_cutoff",
            "winner_rule": (
                "top TrainStrong60 within eligible set; if within close margin, "
                "tie-break by ValTemporalGuard, then banded, then classic, then run_id"
            ),
        },
        "winner": {
            "run_id": winner_run_id,
            "trainstrong60_score": float(winner_df.iloc[0]["trainstrong60_score"]),
            "val_temporal_guard_score": float(winner_df.iloc[0]["val_temporal_guard_score"]),
            "banded_score": float(winner_df.iloc[0]["banded_score"]),
            "classic_score": float(winner_df.iloc[0]["classic_score"]),
        },
        "generated_files": {
            "all_runs_csv": str(all_runs_csv),
            "all_runs_json": str(all_runs_json),
            "eligible_csv": str(eligible_csv),
            "eligible_json": str(eligible_json),
            "selection_finalists_csv": str(finalists_csv),
            "selection_finalists_json": str(finalists_json),
            "winner_csv": str(winner_csv),
            "winner_json": str(winner_json),
            "skipped_csv": str(skipped_csv),
            "skipped_json": str(skipped_json),
            "lens_top_tables": lens_output_paths,
        },
    }
    summary_path = output_dir / f"{prefix}_summary.json"
    hist._write_json(summary_path, summary_payload)

    print(f"Official FlowGen root: {root}")
    print(f"Valid runs ranked: {len(valid_records)}")
    print(f"Skipped items: {len(skipped_rows)}")
    print(f"Final ranking CSV: {all_runs_csv}")
    print(f"Eligible set CSV: {eligible_csv}")
    print(f"Selection finalists CSV: {finalists_csv}")
    print(f"Winner JSON: {winner_json}")
    print(f"Summary JSON: {summary_path}")
    print(f"Final winner: {winner_run_id}")
    for lens_name, table in top_tables.items():
        print(f"Top {top_k} [{lens_name}]: {', '.join(table['run_id'].tolist())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
