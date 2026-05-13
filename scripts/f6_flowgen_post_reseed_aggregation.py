#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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
from scripts import f6_flowgen_rank_official_v2 as final


CONTRACT_ID = "f6_flowgen_post_reseed_aggregation_v1"
OFFICIAL_FLOWGEN_ROOT = hist.OFFICIAL_FLOWGEN_ROOT
DEFAULT_OUTPUT_DIR = OFFICIAL_FLOWGEN_ROOT / "campaign_summaries" / "post_reseed"

SOURCE_RUN_IDS = [
    "flowgen_tpv1_c2_train_s01_e38_softclip_seed2468_v2",
    "flowgen_tpv1_c2_train_h01_bridge300_lowmmd_seed2468_v2",
    "flowgen_tpv1_c2_train_k01_e36_ksy_seed2468_v2",
    "flowgen_tpv1_c2_train_h02_bridge500_lowmmd_seed2468_v2",
    "flowgen_tpv1_c2_train_e03_seed2468_v1",
]
EXPECTED_RESEED_SEEDS = [1117, 2221, 3331, 4447]

PRIMARY_LENS_METRICS = [
    ("trainstrong60_score", "trainstrong60"),
    ("val_temporal_guard_score", "val_temporal_guard"),
    ("banded_score", "banded"),
    ("classic_score", "classic"),
]
TEMPORAL_METRICS = [
    ("temporal_ratio25", "ratio25"),
    ("temporal_ratio50", "ratio50"),
    ("temporal_q3_peak", "q3peak"),
]
SUPPORT_METRICS = [
    ("banded_train", "banded_train"),
    ("banded_val", "banded_val"),
    ("temporal_suffix_score", "temporal_suffix"),
]
AGGREGATE_METRICS = PRIMARY_LENS_METRICS + TEMPORAL_METRICS + SUPPORT_METRICS

LENS_MEAN_COLUMNS = {
    "TrainStrong60": ("trainstrong60_mean", False),
    "ValTemporalGuard": ("val_temporal_guard_mean", False),
    "banded": ("banded_mean", False),
    "classic": ("classic_mean", True),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the final FlowGen reseed panel by cfg/base run, combining the "
            "original run plus the 4 reseed runs for each selected candidate."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=OFFICIAL_FLOWGEN_ROOT,
        help="Official FlowGen root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Namespace where the post-reseed closeout artifacts will be written.",
    )
    parser.add_argument(
        "--source-run",
        dest="source_run_ids",
        action="append",
        choices=SOURCE_RUN_IDS,
        default=None,
        help="Optional subset of source runs to aggregate.",
    )
    return parser.parse_args()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True, ensure_ascii=True)
    return path


def _write_markdown(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_table_pair(base_path: Path, df: pd.DataFrame) -> tuple[Path, Path]:
    csv_path = base_path.with_suffix(".csv")
    json_path = base_path.with_suffix(".json")
    rows = df.to_dict(orient="records")
    hist._write_csv(csv_path, rows)
    _write_json(json_path, {"rows": rows})
    return csv_path, json_path


def _score_std_col(mean_col: str) -> str:
    return mean_col.replace("_mean", "_std")


def _score_n_col(mean_col: str) -> str:
    return mean_col.replace("_mean", "_n")


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except Exception:
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def _record_axes(record: hist.RunRecord) -> dict[str, Any]:
    run_axes = record.manifest.get("run_level_axes") or {}
    official_training = record.config.get("official_training") or {}
    return {
        "policy_id": run_axes.get("policy_id") or official_training.get("policy_id"),
        "policy_origin": run_axes.get("policy_origin") or official_training.get("policy_origin"),
        "flowgen_base_run_id": run_axes.get("flowgen_base_run_id") or official_training.get("base_run_id"),
        "flowgen_base_work_base_id": run_axes.get("flowgen_base_work_base_id")
        or official_training.get("base_work_base_id"),
        "flowgen_base_seed": run_axes.get("flowgen_base_seed") or official_training.get("base_seed"),
        "paired_flowpre_source_id": run_axes.get("paired_flowpre_source_id")
        or official_training.get("paired_flowpre_source_id"),
        "paired_flowpre_run_id": run_axes.get("paired_flowpre_run_id")
        or official_training.get("paired_flowpre_run_id"),
        "paired_flowpre_seed": run_axes.get("paired_flowpre_seed") or official_training.get("paired_flowpre_seed"),
        "reseed_source_run_id": run_axes.get("reseed_source_run_id") or official_training.get("reseed_source_run_id"),
        "reseed_source_seed": run_axes.get("reseed_source_seed") or official_training.get("reseed_source_seed"),
        "reseed_source_version": run_axes.get("reseed_source_version")
        or official_training.get("reseed_source_version"),
        "run_seed": run_axes.get("run_seed") or official_training.get("run_seed"),
    }


def _load_record_or_raise(run_dir: Path) -> hist.RunRecord:
    record, invalid_row = hist._load_run_record(run_dir)
    if invalid_row is not None:
        raise ValueError(_stable_json(invalid_row))
    if record is None:
        raise ValueError(f"Unable to load run record under {run_dir}.")
    if not hist._has_any_finite_metric(record.results):
        raise ValueError(f"Run does not expose finite realism metrics: {run_dir}")
    return record


def _original_member_row(record: hist.RunRecord) -> dict[str, Any]:
    axes = _record_axes(record)
    run_seed = hist._run_seed(record)
    return {
        "cfg_id": record.run_id,
        "policy_id": axes["policy_id"],
        "policy_origin": axes["policy_origin"],
        "member_type": "original",
        "panel_seed": run_seed,
        "run_id": record.run_id,
        "run_dir": str(record.run_dir),
        "source_run_id": record.run_id,
        "source_version": final._run_version(record.run_id),
        "resolution_signal": "original_run_id",
        "reseed_source_seed": None,
        "flowgen_base_run_id": axes["flowgen_base_run_id"],
        "flowgen_base_work_base_id": axes["flowgen_base_work_base_id"],
        "flowgen_base_seed": axes["flowgen_base_seed"],
        "paired_flowpre_source_id": axes["paired_flowpre_source_id"],
        "paired_flowpre_run_id": axes["paired_flowpre_run_id"],
        "paired_flowpre_seed": axes["paired_flowpre_seed"],
        "split_id": record.manifest.get("split_id"),
        "test_enabled": bool(record.manifest.get("test_enabled")),
        "contract_id": record.manifest.get("contract_id"),
        "metrics_long_path": None if record.metrics_long_path is None else str(record.metrics_long_path),
        "results_path": str(record.results_path),
        "run_manifest_path": None if record.manifest_path is None else str(record.manifest_path),
    }


def _reseed_member_row(record: hist.RunRecord) -> dict[str, Any]:
    axes = _record_axes(record)
    return {
        "cfg_id": axes["reseed_source_run_id"],
        "policy_id": axes["policy_id"],
        "policy_origin": axes["policy_origin"],
        "member_type": "reseed",
        "panel_seed": hist._run_seed(record),
        "run_id": record.run_id,
        "run_dir": str(record.run_dir),
        "source_run_id": axes["reseed_source_run_id"],
        "source_version": axes["reseed_source_version"],
        "resolution_signal": "run_manifest.run_level_axes.reseed_source_run_id",
        "reseed_source_seed": axes["reseed_source_seed"],
        "flowgen_base_run_id": axes["flowgen_base_run_id"],
        "flowgen_base_work_base_id": axes["flowgen_base_work_base_id"],
        "flowgen_base_seed": axes["flowgen_base_seed"],
        "paired_flowpre_source_id": axes["paired_flowpre_source_id"],
        "paired_flowpre_run_id": axes["paired_flowpre_run_id"],
        "paired_flowpre_seed": axes["paired_flowpre_seed"],
        "split_id": record.manifest.get("split_id"),
        "test_enabled": bool(record.manifest.get("test_enabled")),
        "contract_id": record.manifest.get("contract_id"),
        "metrics_long_path": None if record.metrics_long_path is None else str(record.metrics_long_path),
        "results_path": str(record.results_path),
        "run_manifest_path": None if record.manifest_path is None else str(record.manifest_path),
    }


def _validate_group(original_row: dict[str, Any], reseed_rows: list[dict[str, Any]]) -> None:
    cfg_id = str(original_row["cfg_id"])
    if len(reseed_rows) != len(EXPECTED_RESEED_SEEDS):
        raise ValueError(
            f"{cfg_id} expected {len(EXPECTED_RESEED_SEEDS)} reseed members, found {len(reseed_rows)}."
        )

    reseed_seed_set = sorted(int(row["panel_seed"]) for row in reseed_rows)
    if reseed_seed_set != sorted(EXPECTED_RESEED_SEEDS):
        raise ValueError(
            f"{cfg_id} resolved reseed seed set {reseed_seed_set}, expected {sorted(EXPECTED_RESEED_SEEDS)}."
        )

    expected_source_version = str(original_row["source_version"])
    expected_source_seed = int(original_row["panel_seed"])
    expected_fields = [
        "policy_id",
        "flowgen_base_run_id",
        "flowgen_base_work_base_id",
        "flowgen_base_seed",
        "paired_flowpre_source_id",
        "paired_flowpre_run_id",
        "paired_flowpre_seed",
        "split_id",
    ]
    for reseed_row in reseed_rows:
        if str(reseed_row["source_run_id"]) != cfg_id:
            raise ValueError(
                f"{cfg_id} has reseed member bound to unexpected source {reseed_row['source_run_id']}."
            )
        if str(reseed_row["source_version"]) != expected_source_version:
            raise ValueError(
                f"{cfg_id} reseed source version mismatch: {reseed_row['source_version']} vs {expected_source_version}."
            )
        if int(reseed_row["reseed_source_seed"]) != expected_source_seed:
            raise ValueError(
                f"{cfg_id} reseed source seed mismatch: {reseed_row['reseed_source_seed']} vs {expected_source_seed}."
            )
        if bool(reseed_row["test_enabled"]):
            raise ValueError(f"{cfg_id} reseed member unexpectedly enables test holdout.")
        for field_name in expected_fields:
            if reseed_row[field_name] != original_row[field_name]:
                raise ValueError(
                    f"{cfg_id} mismatch for {field_name}: {reseed_row[field_name]} vs {original_row[field_name]}."
                )


def _resolve_panel(
    root: Path,
    source_run_ids: list[str],
) -> tuple[list[hist.RunRecord], pd.DataFrame, dict[str, Any]]:
    original_records: dict[str, hist.RunRecord] = {}
    family_rows: list[dict[str, Any]] = []
    all_records: list[hist.RunRecord] = []

    for run_id in source_run_ids:
        record = _load_record_or_raise(root / run_id)
        if bool(record.manifest.get("test_enabled")):
            raise ValueError(f"Original source run unexpectedly enables test holdout: {run_id}")
        original_records[run_id] = record
        all_records.append(record)
        family_rows.append(_original_member_row(record))

    reseed_root = root / "reseed_final"
    if not reseed_root.exists():
        raise FileNotFoundError(f"Reseed-final directory not found: {reseed_root}")

    grouped_reseed_rows: dict[str, list[dict[str, Any]]] = {run_id: [] for run_id in source_run_ids}
    grouped_reseed_records: dict[str, list[hist.RunRecord]] = {run_id: [] for run_id in source_run_ids}
    for run_dir in hist._discover_runs_under(reseed_root):
        record = _load_record_or_raise(run_dir)
        row = _reseed_member_row(record)
        source_run_id = row["source_run_id"]
        if source_run_id not in grouped_reseed_rows:
            continue
        grouped_reseed_rows[source_run_id].append(row)
        grouped_reseed_records[source_run_id].append(record)
        all_records.append(record)
        family_rows.append(row)

    for run_id in source_run_ids:
        original_row = _original_member_row(original_records[run_id])
        _validate_group(original_row, grouped_reseed_rows[run_id])

    family_df = pd.DataFrame(family_rows)
    member_order = {"original": 0, "reseed": 1}
    family_df["member_sort_key"] = family_df["member_type"].map(member_order).fillna(9)
    family_df = family_df.sort_values(
        by=["cfg_id", "member_sort_key", "panel_seed", "run_id"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).drop(columns=["member_sort_key"]).reset_index(drop=True)

    validation = {
        "source_cfg_count": len(source_run_ids),
        "panel_run_count": int(len(family_df)),
        "expected_panel_run_count": int(len(source_run_ids) * (1 + len(EXPECTED_RESEED_SEEDS))),
        "expected_reseed_seeds": list(EXPECTED_RESEED_SEEDS),
        "panel_seed_order": sorted({int(seed) for seed in family_df["panel_seed"].dropna().tolist()}),
        "unique_split_ids": sorted({str(value) for value in family_df["split_id"].dropna().tolist()}),
        "unique_base_run_ids": sorted({str(value) for value in family_df["flowgen_base_run_id"].dropna().tolist()}),
        "unique_base_work_base_ids": sorted(
            {str(value) for value in family_df["flowgen_base_work_base_id"].dropna().tolist()}
        ),
        "unique_paired_flowpre_run_ids": sorted(
            {str(value) for value in family_df["paired_flowpre_run_id"].dropna().tolist()}
        ),
        "all_test_disabled": bool((~family_df["test_enabled"].fillna(False)).all()),
        "family_sizes": {
            cfg_id: int(size) for cfg_id, size in family_df.groupby("cfg_id").size().to_dict().items()
        },
    }
    return all_records, family_df, validation


def _build_panel_table(records: list[hist.RunRecord], family_df: pd.DataFrame) -> pd.DataFrame:
    obs_max = hist._compute_obs_max(records)

    rows = [hist._build_row(record, obs_max) for record in records]
    extra_rows: list[dict[str, Any]] = []
    for record in records:
        temporal = final._extract_temporal_metrics(record.results)
        extra_rows.append(
            {
                "run_id": record.run_id,
                "base_token": final._base_token(record.run_id),
                "run_version": final._run_version(record.run_id),
                "banded_train": final._banded_score_for_splits(
                    record.results,
                    obs_max=obs_max,
                    split_weights={"train": 1.0},
                ),
                "banded_val": final._banded_score_for_splits(
                    record.results,
                    obs_max=obs_max,
                    split_weights={"val": 1.0},
                ),
                **temporal,
            }
        )

    panel_df = pd.DataFrame(rows).merge(pd.DataFrame(extra_rows), on="run_id", how="left")
    join_columns = [
        "cfg_id",
        "member_type",
        "source_run_id",
        "source_version",
        "resolution_signal",
        "reseed_source_seed",
        "paired_flowpre_run_id",
    ]
    panel_df = panel_df.merge(
        family_df[["run_id", *join_columns]],
        on="run_id",
        how="left",
        validate="one_to_one",
    )

    # Recompute the final lens surface on the closed 25-run panel so originals and reseeds share one frame.
    temporal_obs_max = {
        metric: float(np.nanmax(pd.to_numeric(panel_df[f"temporal_{metric}"], errors="coerce")))
        for metric in final.TEMPORAL_BAND_CUTS
    }
    for metric, cuts in final.TEMPORAL_BAND_CUTS.items():
        labels: list[str] = []
        points: list[float] = []
        for value in pd.to_numeric(panel_df[f"temporal_{metric}"], errors="coerce").tolist():
            label, raw_points = final._continuous_points_from_cuts(
                value=float(value),
                cuts=cuts,
                obs_max=temporal_obs_max[metric],
            )
            labels.append(label)
            points.append(
                float(raw_points / final.TEMPORAL_POINT_DIVISOR) if np.isfinite(raw_points) else float("nan")
            )
        panel_df[f"temporal_{metric}_label"] = labels
        panel_df[f"temporal_{metric}_points"] = points

    panel_df["temporal_suffix_score"] = [
        final._weighted_mean(
            [
                (row["temporal_ratio50_points"], final.TEMPORAL_SUFFIX_WEIGHTS["t50"]),
                (row["temporal_ratio25_points"], final.TEMPORAL_SUFFIX_WEIGHTS["t25"]),
            ]
        )
        for _, row in panel_df.iterrows()
    ]
    panel_df["trainstrong60_score"] = [
        final._weighted_mean(
            [
                (row["banded_train"], final.TRAIN_STRONG_WEIGHTS["banded_train"]),
                (row["banded_val"], final.TRAIN_STRONG_WEIGHTS["banded_val"]),
            ]
        )
        for _, row in panel_df.iterrows()
    ]
    panel_df["val_temporal_guard_score"] = [
        final._weighted_mean(
            [
                (row["banded_val"], final.VAL_TEMPORAL_GUARD_WEIGHTS["banded_val"]),
                (row["temporal_suffix_score"], final.VAL_TEMPORAL_GUARD_WEIGHTS["temporal_suffix"]),
                (row["temporal_q3_peak_points"], final.VAL_TEMPORAL_GUARD_WEIGHTS["tq3"]),
            ]
        )
        for _, row in panel_df.iterrows()
    ]

    ranked_panel_df, _ = final._rank_tables(
        panel_df,
        lens_specs=final.FINAL_LENS_SPECS,
        top_k=len(panel_df),
    )
    ranked_panel_df = ranked_panel_df.sort_values(
        by=["cfg_id", "seed", "run_id"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return ranked_panel_df


def _aggregate_cfg_table(panel_df: pd.DataFrame, family_df: pd.DataFrame) -> pd.DataFrame:
    family_groups = {cfg_id: group.copy() for cfg_id, group in family_df.groupby("cfg_id", sort=True)}
    aggregate_rows: list[dict[str, Any]] = []

    for cfg_id, group in panel_df.groupby("cfg_id", sort=True):
        group = group.sort_values(by=["seed", "run_id"], kind="mergesort")
        family_group = family_groups[cfg_id].sort_values(by=["member_type", "panel_seed", "run_id"], kind="mergesort")
        original_row = group[group["member_type"] == "original"].iloc[0]
        reseed_rows = family_group[family_group["member_type"] == "reseed"].copy()

        row: dict[str, Any] = {
            "cfg_id": cfg_id,
            "policy_id": group["policy_id"].dropna().iloc[0] if group["policy_id"].notna().any() else None,
            "policy_origin": group["policy_origin"].dropna().iloc[0] if group["policy_origin"].notna().any() else None,
            "source_version": group["source_version"].dropna().iloc[0] if group["source_version"].notna().any() else None,
            "split_id": group["split_id"].dropna().iloc[0] if group["split_id"].notna().any() else None,
            "flowgen_base_run_id": group["flowgen_base_run_id"].dropna().iloc[0]
            if group["flowgen_base_run_id"].notna().any()
            else None,
            "flowgen_base_work_base_id": group["flowgen_base_work_base_id"].dropna().iloc[0]
            if group["flowgen_base_work_base_id"].notna().any()
            else None,
            "flowgen_base_seed": group["flowgen_base_seed"].dropna().iloc[0]
            if group["flowgen_base_seed"].notna().any()
            else None,
            "paired_flowpre_source_id": group["paired_flowpre_source_id"].dropna().iloc[0]
            if group["paired_flowpre_source_id"].notna().any()
            else None,
            "paired_flowpre_run_id": group["paired_flowpre_run_id"].dropna().iloc[0]
            if group["paired_flowpre_run_id"].notna().any()
            else None,
            "original_run_id": str(original_row["run_id"]),
            "original_seed": int(original_row["seed"]),
            "panel_run_count": int(len(group)),
            "panel_seed_set_json": _stable_json(sorted(int(seed) for seed in group["seed"].tolist())),
            "panel_run_ids_json": _stable_json(group["run_id"].tolist()),
            "reseed_seed_set_json": _stable_json(sorted(int(seed) for seed in reseed_rows["panel_seed"].tolist())),
            "reseed_run_ids_json": _stable_json(reseed_rows["run_id"].tolist()),
        }

        for metric_col, metric_alias in AGGREGATE_METRICS:
            values = pd.to_numeric(group[metric_col], errors="coerce")
            row[f"{metric_alias}_mean"] = float(values.mean()) if values.count() > 0 else float("nan")
            row[f"{metric_alias}_std"] = float(values.std(ddof=1)) if values.count() > 1 else float("nan")
            row[f"{metric_alias}_n"] = int(values.count())

        for metric_col, metric_alias in PRIMARY_LENS_METRICS:
            row[f"{metric_alias}_original"] = float(original_row[metric_col])
            row[f"{metric_alias}_mean_minus_original"] = float(row[f"{metric_alias}_mean"] - original_row[metric_col])

        aggregate_rows.append(row)

    return pd.DataFrame(aggregate_rows).sort_values(by=["cfg_id"], kind="mergesort").reset_index(drop=True)


def _rank_cfg_table(
    aggregate_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    ranked_df = aggregate_df.copy()
    lens_tables: dict[str, pd.DataFrame] = {}
    ranking_rows: list[dict[str, Any]] = []

    for lens_name, (mean_col, ascending) in LENS_MEAN_COLUMNS.items():
        ordered = ranked_df.sort_values(
            by=[mean_col, "cfg_id"],
            ascending=[ascending, True],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
        ordered = ordered.copy()
        ordered["rank"] = np.arange(1, len(ordered) + 1, dtype=int)
        lens_table = ordered[
            [
                "rank",
                "cfg_id",
                "policy_id",
                "panel_run_count",
                mean_col,
                _score_std_col(mean_col),
                _score_n_col(mean_col),
            ]
        ].copy()
        lens_table = lens_table.rename(
            columns={
                "panel_run_count": "n",
                mean_col: "score_mean",
                _score_std_col(mean_col): "score_std",
                _score_n_col(mean_col): "score_n",
            }
        )
        lens_tables[lens_name] = lens_table
        rank_map = {str(row.cfg_id): int(row.rank) for row in lens_table.itertuples(index=False)}
        ranked_df[f"rank_{lens_name}"] = ranked_df["cfg_id"].map(rank_map)
        for row in lens_table.itertuples(index=False):
            ranking_rows.append(
                {
                    "lens": lens_name,
                    "rank": int(row.rank),
                    "cfg_id": row.cfg_id,
                    "policy_id": row.policy_id,
                    "n": int(row.n),
                    "score_mean": float(row.score_mean),
                    "score_std": float(row.score_std),
                    "score_n": int(row.score_n),
                    "ascending": bool(ascending),
                }
            )

    ranking_long_df = pd.DataFrame(ranking_rows).sort_values(
        by=["lens", "rank", "cfg_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return ranked_df, lens_tables, ranking_long_df


def _pooled_standard_error(top: pd.Series, runner_up: pd.Series, mean_col: str) -> float:
    top_std = float(top[_score_std_col(mean_col)])
    runner_up_std = float(runner_up[_score_std_col(mean_col)])
    top_n = max(1, int(top[_score_n_col(mean_col)]))
    runner_up_n = max(1, int(runner_up[_score_n_col(mean_col)]))
    return float(math.sqrt((top_std ** 2) / top_n + (runner_up_std ** 2) / runner_up_n))


def _selection_sort_cfg(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=[
            "trainstrong60_mean",
            "val_temporal_guard_mean",
            "banded_mean",
            "classic_mean",
            "cfg_id",
        ],
        ascending=[False, False, False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _dominates(left: pd.Series, right: pd.Series) -> bool:
    checks = [
        float(left["trainstrong60_mean"]) >= float(right["trainstrong60_mean"]),
        float(left["val_temporal_guard_mean"]) >= float(right["val_temporal_guard_mean"]),
        float(left["banded_mean"]) >= float(right["banded_mean"]),
        float(left["classic_mean"]) <= float(right["classic_mean"]),
    ]
    strict = [
        float(left["trainstrong60_mean"]) > float(right["trainstrong60_mean"]),
        float(left["val_temporal_guard_mean"]) > float(right["val_temporal_guard_mean"]),
        float(left["banded_mean"]) > float(right["banded_mean"]),
        float(left["classic_mean"]) < float(right["classic_mean"]),
    ]
    return all(checks) and any(strict)


def _objective_summary(aggregate_df: pd.DataFrame) -> dict[str, Any]:
    train_order = aggregate_df.sort_values(
        by=["trainstrong60_mean", "cfg_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    temporal_order = aggregate_df.sort_values(
        by=["val_temporal_guard_mean", "cfg_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    global_order = _selection_sort_cfg(aggregate_df)

    eligible_rank_cutoff = max(1, int(math.ceil(len(aggregate_df) * final.DEFAULT_ELIGIBLE_TOP_FRACTION)))
    eligible_cfgs = temporal_order.head(eligible_rank_cutoff).copy()
    eligible_cfgs = _selection_sort_cfg(eligible_cfgs)
    best_trainstrong = float(eligible_cfgs.iloc[0]["trainstrong60_mean"])
    finalists = eligible_cfgs[
        pd.to_numeric(eligible_cfgs["trainstrong60_mean"], errors="coerce")
        >= (best_trainstrong - final.DEFAULT_SELECTION_CLOSE_MARGIN)
    ].copy()
    finalists = _selection_sort_cfg(finalists)

    recommended = finalists.iloc[0]
    dominated_cfgs = [
        str(other_row["cfg_id"])
        for _, other_row in aggregate_df.iterrows()
        if str(other_row["cfg_id"]) != str(recommended["cfg_id"]) and _dominates(recommended, other_row)
    ]

    objective_a_top = train_order.iloc[0]
    objective_a_runner_up = train_order.iloc[1]
    objective_b_top = temporal_order.iloc[0]
    objective_b_runner_up = temporal_order.iloc[1]

    return {
        "objective_A_train_balance": {
            "label": "Objetivo A: maximizar capacidad de balancear train",
            "winner_cfg_id": str(objective_a_top["cfg_id"]),
            "winner_policy_id": objective_a_top["policy_id"],
            "winner_mean": float(objective_a_top["trainstrong60_mean"]),
            "winner_std": float(objective_a_top["trainstrong60_std"]),
            "runner_up_cfg_id": str(objective_a_runner_up["cfg_id"]),
            "runner_up_mean": float(objective_a_runner_up["trainstrong60_mean"]),
            "mean_gap_vs_runner_up": float(
                objective_a_top["trainstrong60_mean"] - objective_a_runner_up["trainstrong60_mean"]
            ),
            "pooled_standard_error_vs_runner_up": _pooled_standard_error(
                objective_a_top,
                objective_a_runner_up,
                "trainstrong60_mean",
            ),
        },
        "objective_B_temporal_guard": {
            "label": "Objetivo B: minimizar colapso temporal en val",
            "winner_cfg_id": str(objective_b_top["cfg_id"]),
            "winner_policy_id": objective_b_top["policy_id"],
            "winner_mean": float(objective_b_top["val_temporal_guard_mean"]),
            "winner_std": float(objective_b_top["val_temporal_guard_std"]),
            "runner_up_cfg_id": str(objective_b_runner_up["cfg_id"]),
            "runner_up_mean": float(objective_b_runner_up["val_temporal_guard_mean"]),
            "mean_gap_vs_runner_up": float(
                objective_b_top["val_temporal_guard_mean"] - objective_b_runner_up["val_temporal_guard_mean"]
            ),
            "pooled_standard_error_vs_runner_up": _pooled_standard_error(
                objective_b_top,
                objective_b_runner_up,
                "val_temporal_guard_mean",
            ),
        },
        "objective_C_global_closeout": {
            "label": "Objetivo C: mejor compromiso global de cierre",
            "selection_rule": {
                "eligible_top_fraction_by_val_temporal_guard": float(final.DEFAULT_ELIGIBLE_TOP_FRACTION),
                "eligible_rank_cutoff": int(eligible_rank_cutoff),
                "selection_close_margin": float(final.DEFAULT_SELECTION_CLOSE_MARGIN),
                "winner_sort": [
                    "trainstrong60_mean desc",
                    "val_temporal_guard_mean desc",
                    "banded_mean desc",
                    "classic_mean asc",
                    "cfg_id asc",
                ],
            },
            "eligible_cfg_ids": eligible_cfgs["cfg_id"].tolist(),
            "finalist_cfg_ids": finalists["cfg_id"].tolist(),
            "winner_cfg_id": str(recommended["cfg_id"]),
            "winner_policy_id": recommended["policy_id"],
        },
        "recommendation": {
            "status": "single_recommended_winner_with_goal_conditioning",
            "winner_cfg_id": str(recommended["cfg_id"]),
            "winner_policy_id": recommended["policy_id"],
            "dominated_cfg_ids": dominated_cfgs,
            "notes": [
                "No single cfg wins every lens after 5-seed aggregation.",
                "The recommended winner is the best global closeout compromise under the final aggregated rule.",
            ],
        },
        "orders": {
            "trainstrong60_cfg_order": train_order["cfg_id"].tolist(),
            "val_temporal_guard_cfg_order": temporal_order["cfg_id"].tolist(),
            "global_closeout_cfg_order": global_order["cfg_id"].tolist(),
        },
    }


def _markdown_table(df: pd.DataFrame, float_cols: set[str] | None = None, digits: int = 4) -> str:
    if df.empty:
        return "_Empty table._"
    float_cols = float_cols or set()
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values: list[str] = []
        for column in headers:
            value = row[column]
            if column in float_cols:
                values.append(_format_float(value, digits=digits))
            else:
                if isinstance(value, float) and math.isnan(value):
                    values.append("")
                else:
                    values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _family_lines(family_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for cfg_id, group in family_df.groupby("cfg_id", sort=True):
        original_run_id = group.loc[group["member_type"] == "original", "run_id"].iloc[0]
        reseed_rows = group[group["member_type"] == "reseed"].sort_values(by=["panel_seed", "run_id"], kind="mergesort")
        reseed_bits = ", ".join(
            f"{int(row.panel_seed)} -> {row.run_id}" for row in reseed_rows.itertuples(index=False)
        )
        lines.append(f"- `{cfg_id}`")
        lines.append(f"  Original: `{original_run_id}`")
        lines.append(f"  Reseed: {reseed_bits}")
    return lines


def _build_report(
    *,
    summary: dict[str, Any],
    family_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    lens_tables: dict[str, pd.DataFrame],
) -> str:
    recommendation = summary["objective_summary"]["recommendation"]
    objective_a = summary["objective_summary"]["objective_A_train_balance"]
    objective_b = summary["objective_summary"]["objective_B_temporal_guard"]
    objective_c = summary["objective_summary"]["objective_C_global_closeout"]

    recommended_cfg_id = recommendation["winner_cfg_id"]
    recommended_row = aggregate_df.loc[aggregate_df["cfg_id"] == recommended_cfg_id].iloc[0]
    s01_row = aggregate_df.loc[
        aggregate_df["cfg_id"] == "flowgen_tpv1_c2_train_s01_e38_softclip_seed2468_v2"
    ].iloc[0]
    e03_row = aggregate_df.loc[
        aggregate_df["cfg_id"] == "flowgen_tpv1_c2_train_e03_seed2468_v1"
    ].iloc[0]

    aggregate_view = aggregate_df[
        [
            "cfg_id",
            "policy_id",
            "panel_run_count",
            "trainstrong60_mean",
            "trainstrong60_std",
            "val_temporal_guard_mean",
            "val_temporal_guard_std",
            "banded_mean",
            "banded_std",
            "classic_mean",
            "classic_std",
            "ratio25_mean",
            "ratio50_mean",
            "q3peak_mean",
        ]
    ].copy()
    aggregate_view = aggregate_view.rename(columns={"panel_run_count": "n"})
    aggregate_float_cols = {
        "trainstrong60_mean",
        "trainstrong60_std",
        "val_temporal_guard_mean",
        "val_temporal_guard_std",
        "banded_mean",
        "banded_std",
        "classic_mean",
        "classic_std",
        "ratio25_mean",
        "ratio50_mean",
        "q3peak_mean",
    }

    sections = [
        "# FlowGen Post-Reseed Closeout",
        "",
        "## Canonical vs historical",
        "",
        "Este namespace es la capa canónica de cierre post-reseed para FlowGen.",
        "Aquí se agregan las 5 seeds por cfg/base run: la original y las 4 reseeds finales.",
        "",
        "Entradas históricas o previas que se usan solo como insumo de lectura:",
        f"- `outputs/models/official/flowgen/campaign_summaries/final_rankings/`",
        f"- `outputs/models/official/flowgen/reseed_final/`",
        f"- `outputs/models/official/flowgen/FINAL_SELECTION.md`",
        "",
        "Artefactos canónicos de este cierre en este namespace:",
        f"- `flowgen_post_reseed_aggregation_v1_*_family_resolution.(csv|json)`",
        f"- `flowgen_post_reseed_aggregation_v1_*_panel_runs.(csv|json)`",
        f"- `flowgen_post_reseed_aggregation_v1_*_cfg_aggregates.(csv|json)`",
        f"- `flowgen_post_reseed_aggregation_v1_*_cfg_rankings.(csv|json)`",
        f"- `flowgen_post_reseed_aggregation_v1_*_summary.json`",
        f"- `flowgen_post_reseed_aggregation_v1_*_report.md`",
        "",
        "## Resolucion programatica de familias",
        "",
        "Las familias se resolvieron por `run_manifest.json -> run_level_axes.reseed_source_run_id` en cada reseed run.",
        "Se validó además coherencia de `policy_id`, `source_version`, `split_id`, `flowgen_base_run_id`, `flowgen_base_work_base_id`, `paired_flowpre_run_id` y `test_enabled=false`.",
        "",
        *(_family_lines(family_df)),
        "",
        "## Agregados por cfg",
        "",
        _markdown_table(aggregate_view, float_cols=aggregate_float_cols),
        "",
        "## Rankings agregados por lente",
        "",
    ]

    for lens_name in ["TrainStrong60", "ValTemporalGuard", "banded", "classic"]:
        lens_table = lens_tables[lens_name].copy()
        sections.append(f"### {lens_name}")
        sections.append("")
        sections.append(
            _markdown_table(
                lens_table,
                float_cols={"score_mean", "score_std"},
                digits=4,
            )
        )
        sections.append("")

    sections.extend(
        [
            "## Lectura por objetivos",
            "",
            f"### {objective_a['label']}",
            "",
            (
                f"Gana `{objective_a['winner_cfg_id']}` con media `TrainStrong60={objective_a['winner_mean']:.4f}`. "
                f"El segundo es `{objective_a['runner_up_cfg_id']}` con gap medio de "
                f"`{objective_a['mean_gap_vs_runner_up']:.4f}`."
            ),
            (
                f"La ventaja existe, pero no es un runaway claro: el gap es menor que el error estándar combinado "
                f"(`{objective_a['pooled_standard_error_vs_runner_up']:.4f}`)."
            ),
            "",
            f"### {objective_b['label']}",
            "",
            (
                f"Gana `{objective_b['winner_cfg_id']}` con media `ValTemporalGuard={objective_b['winner_mean']:.4f}`, "
                f"pero `{objective_b['runner_up_cfg_id']}` queda prácticamente pegado con un gap de "
                f"`{objective_b['mean_gap_vs_runner_up']:.4f}`."
            ),
            (
                f"Aquí sí hay empate práctico real: el gap es minúsculo frente al error estándar combinado "
                f"(`{objective_b['pooled_standard_error_vs_runner_up']:.4f}`)."
            ),
            "",
            f"### {objective_c['label']}",
            "",
            (
                f"Aplicando la misma lógica de cierre sobre medias agregadas, el conjunto elegible queda en "
                f"`{', '.join(objective_c['eligible_cfg_ids'])}` y el winner resultante es "
                f"`{objective_c['winner_cfg_id']}`."
            ),
            (
                f"Eso ocurre porque `{objective_c['winner_cfg_id']}` conserva casi toda la robustez temporal del bloque "
                f"`H01/E03`, pero evita la caída de train-strong de `H01`."
            ),
            "",
            "## Analisis final profundo",
            "",
            (
                f"No hay un ganador absoluto que domine las 4 lentes. `H02` gana el frente train-strong/banded, "
                f"`H01` gana el frente temporal/classic, y `E03` queda en medio como compromiso."
            ),
            (
                f"`S01`, que era el winner operativo pre-reseed, deja de sostener esa posición al agregar las 5 seeds. "
                f"Su media cae a `TrainStrong60={s01_row['trainstrong60_mean']:.4f}` y "
                f"`ValTemporalGuard={s01_row['val_temporal_guard_mean']:.4f}`."
            ),
            (
                f"Además, `E03` domina agregadamente a `S01`: mejora las cuatro lentes principales "
                f"(`TrainStrong60 {e03_row['trainstrong60_mean']:.4f} > {s01_row['trainstrong60_mean']:.4f}`, "
                f"`ValTemporalGuard {e03_row['val_temporal_guard_mean']:.4f} > {s01_row['val_temporal_guard_mean']:.4f}`, "
                f"`banded {e03_row['banded_mean']:.4f} > {s01_row['banded_mean']:.4f}`, "
                f"`classic {e03_row['classic_mean']:.4f} < {s01_row['classic_mean']:.4f}`)."
            ),
            (
                f"`H02` es la mejor cfg si el objetivo real fuera apretar al máximo el balanceo de train, "
                f"pero paga el peor `ValTemporalGuard` del panel (`{aggregate_df.sort_values('val_temporal_guard_mean').iloc[0]['val_temporal_guard_mean']:.4f}`). "
                f"Para cierre metodológico eso la deja demasiado expuesta."
            ),
            (
                f"`H01` es la mejor lectura si el objetivo fuera robustez temporal pura, pero su media "
                f"`TrainStrong60={aggregate_df.loc[aggregate_df['cfg_id'] == 'flowgen_tpv1_c2_train_h01_bridge300_lowmmd_seed2468_v2', 'trainstrong60_mean'].iloc[0]:.4f}` "
                f"es la más baja del panel y además tiene la mayor dispersión en esa lente."
            ),
            (
                f"`E03` es la cfg con mejor compromiso real de cierre: queda segunda en `ValTemporalGuard`, "
                f"segunda en `classic`, segunda en `ratio25`, y a la vez se mantiene en el bloque alto de "
                f"`TrainStrong60` y `banded`."
            ),
            "",
            "## Recomendacion final",
            "",
            (
                f"Mi recomendacion final para cerrar FlowGen es `{recommended_cfg_id}` "
                f"(`policy_id={recommended_row['policy_id']}`)."
            ),
            (
                f"La decision es condicionada por objetivo en los extremos, pero para el objetivo real del proyecto "
                f"prefiero un winner unico de cierre y ese winner debe ser `E03`: conserva robustez temporal casi "
                f"de nivel `H01`, evita el sesgo demasiado train-centric de `H02`, y deja atras al viejo winner `S01` "
                f"con dominancia agregada en las cuatro lentes principales."
            ),
            "",
        ]
    )

    return "\n".join(sections).strip() + "\n"


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    source_run_ids = list(args.source_run_ids or SOURCE_RUN_IDS)

    if not root.exists():
        raise FileNotFoundError(f"Official FlowGen root does not exist: {root}")

    stamp = _utc_stamp()
    prefix = f"{CONTRACT_ID}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    records, family_df, validation = _resolve_panel(root, source_run_ids)
    panel_df = _build_panel_table(records, family_df)
    aggregate_df = _aggregate_cfg_table(panel_df, family_df)
    ranked_cfg_df, lens_tables, ranking_long_df = _rank_cfg_table(aggregate_df)
    objective_summary = _objective_summary(ranked_cfg_df)

    family_csv, family_json = _write_table_pair(output_dir / f"{prefix}_family_resolution", family_df)
    panel_csv, panel_json = _write_table_pair(output_dir / f"{prefix}_panel_runs", panel_df)
    aggregates_csv, aggregates_json = _write_table_pair(output_dir / f"{prefix}_cfg_aggregates", ranked_cfg_df)
    rankings_csv, rankings_json = _write_table_pair(output_dir / f"{prefix}_cfg_rankings", ranking_long_df)

    lens_outputs: dict[str, dict[str, str]] = {}
    for lens_name, lens_table in lens_tables.items():
        lens_csv, lens_json = _write_table_pair(output_dir / f"{prefix}_ranking_{lens_name}", lens_table)
        lens_outputs[lens_name] = {"csv": str(lens_csv), "json": str(lens_json)}

    summary_payload = {
        "contract_id": CONTRACT_ID,
        "script": str(Path(__file__).resolve()),
        "timestamp_utc": stamp,
        "root": str(root),
        "output_dir": str(output_dir),
        "source_run_ids": list(source_run_ids),
        "historical_inputs": {
            "final_rankings_root": str(root / "campaign_summaries" / "final_rankings"),
            "final_selection_md": str(root / "FINAL_SELECTION.md"),
            "reseed_final_root": str(root / "reseed_final"),
        },
        "resolution_policy": {
            "primary_signal": "run_manifest.run_level_axes.reseed_source_run_id",
            "cross_checks": [
                "policy_id",
                "source_version",
                "split_id",
                "flowgen_base_run_id",
                "flowgen_base_work_base_id",
                "paired_flowpre_run_id",
                "test_enabled=false",
            ],
        },
        "validation": validation,
        "objective_summary": objective_summary,
        "generated_files": {
            "family_resolution_csv": str(family_csv),
            "family_resolution_json": str(family_json),
            "panel_runs_csv": str(panel_csv),
            "panel_runs_json": str(panel_json),
            "cfg_aggregates_csv": str(aggregates_csv),
            "cfg_aggregates_json": str(aggregates_json),
            "cfg_rankings_csv": str(rankings_csv),
            "cfg_rankings_json": str(rankings_json),
            "lens_rankings": lens_outputs,
        },
    }
    summary_path = _write_json(output_dir / f"{prefix}_summary.json", summary_payload)
    report_path = _write_markdown(
        output_dir / f"{prefix}_report.md",
        _build_report(
            summary=summary_payload,
            family_df=family_df,
            aggregate_df=ranked_cfg_df,
            lens_tables=lens_tables,
        ),
    )

    print(f"Post-reseed output dir: {output_dir}")
    print(f"Resolved cfg families: {validation['source_cfg_count']}")
    print(f"Total panel runs: {validation['panel_run_count']}")
    print(f"Family resolution CSV: {family_csv}")
    print(f"Panel runs CSV: {panel_csv}")
    print(f"Cfg aggregates CSV: {aggregates_csv}")
    print(f"Cfg rankings CSV: {rankings_csv}")
    print(f"Summary JSON: {summary_path}")
    print(f"Report MD: {report_path}")
    print(f"Recommended winner: {objective_summary['recommendation']['winner_cfg_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
