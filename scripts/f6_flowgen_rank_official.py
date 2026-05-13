#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.f6_common import load_json, load_yaml


OFFICIAL_FLOWGEN_ROOT = ROOT / "outputs" / "models" / "official" / "flowgen"
DEFAULT_OUTPUT_DIR = OFFICIAL_FLOWGEN_ROOT / "campaign_summaries" / "rankings"
EXCLUDED_ROOT_SUBTREES = {"bases", "campaign_summaries"}

CLASSES = [0, 1, 2]
SPLITS = ["train", "val"]
COMPONENTS = ["xy", "x", "y"]

W_SPLIT = {"train": 0.5, "val": 0.5}
W_SCOPE = {"overall": 0.2, "perclass": 0.8}
W_COMPONENT = {"xy": 1.0, "x": 1.0, "y": 1.0}
W_CLASS = {k: 1.0 for k in CLASSES}

METRICS6 = ["ks_mean", "ks_median", "w1_mean", "w1_median", "mmd2_rvs", "mmd2_gap"]
METRICS4 = ["ks_mean", "ks_median", "w1_mean", "w1_median"]
ROOT_METRICS = ["ks_mean", "ks_median", "w1_mean", "w1_median", "mmd2_rvs", "mmd2_rvr_med", "mmd2_gap"]

BAND_CUTS = {
    "ks_mean": [0.020, 0.060, 0.10, 0.150, 0.20, 0.250],
    "ks_median": [0.01, 0.040, 0.080, 0.10, 0.150, 0.20],
    "w1_mean": [0.025, 0.040, 0.050, 0.060, 0.070, 0.090],
    "w1_median": [0.020, 0.032, 0.040, 0.050, 0.060, 0.080],
    "mmd2_rvs": [0.0030, 0.0050, 0.0060, 0.0080, 0.0100, 0.0150],
    "mmd2_gap": [0.004, 0.007, 0.010, 0.015, 0.020, 0.030],
}

RANKS = ["unusable", "very bad", "bad", "mediocre", "good", "very good", "perfect"]
RANK_TO_IDX = {rank: idx for idx, rank in enumerate(RANKS)}
LABELS_BEST_FIRST = ["perfect", "very good", "good", "mediocre", "bad", "very bad", "unusable"]
LABEL_TO_BOUNDS_INDEX = {label: idx for idx, label in enumerate(LABELS_BEST_FIRST)}
EXPONENT_P = 1.35

METRIC_GROUPS = {
    "ks": ["ks_mean", "ks_median"],
    "w1": ["w1_mean", "w1_median"],
    "mmd": ["mmd2_rvs", "mmd2_gap"],
}
GROUP_WEIGHTS = {"ks": 0.30, "w1": 0.45, "mmd": 0.25}
METRIC_TO_GROUP = {metric: group for group, metrics in METRIC_GROUPS.items() for metric in metrics}
METRIC_WEIGHT = {
    metric: GROUP_WEIGHTS[METRIC_TO_GROUP[metric]] / len(METRIC_GROUPS[METRIC_TO_GROUP[metric]])
    for metric in METRICS6
}

LENS_SPECS = {
    "classic": {"score_col": "classic_score", "ascending": True},
    "banded": {"score_col": "banded_score", "ascending": False},
    "blended_ks_mean": {"score_col": "blended_ks_mean", "ascending": True},
    "blended_ks_median": {"score_col": "blended_ks_median", "ascending": True},
    "blended_w1_mean": {"score_col": "blended_w1_mean", "ascending": True},
    "blended_w1_median": {"score_col": "blended_w1_median", "ascending": True},
}


@dataclass
class RunRecord:
    run_id: str
    run_dir: Path
    results_path: Path
    results: dict[str, Any]
    manifest_path: Path | None
    manifest: dict[str, Any]
    config_path: Path | None
    config: dict[str, Any]
    metrics_long_path: Path | None
    missing_optional_artifacts: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank official FlowGen runs using the historical v6 lenses."
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
        help="Directory where ranking artifacts will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k rows to export for each lens.",
    )
    return parser.parse_args()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _value_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return _stable_json(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame(columns=["status"]).to_csv(path, index=False)
        return path
    normalized_rows = [{key: _value_for_csv(val) for key, val in row.items()} for row in rows]
    pd.DataFrame(normalized_rows).to_csv(path, index=False)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _nested_get(payload: dict[str, Any], dotted: str) -> Any:
    cur: Any = payload
    for token in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        if token in cur:
            cur = cur[token]
            continue
        try:
            token_int = int(token)
        except Exception:
            return None
        if token_int in cur:
            cur = cur[token_int]
            continue
        token_str = str(token_int)
        if token_str in cur:
            cur = cur[token_str]
            continue
        return None
    return cur


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _minmax01(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    lo = series.min(skipna=True)
    hi = series.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - lo) / (hi - lo)


def _norm_class_weights() -> dict[int, float]:
    total = sum(max(0.0, float(value)) for value in W_CLASS.values()) or 1.0
    return {int(key): max(0.0, float(value)) / total for key, value in W_CLASS.items()}


W_CLASS_NORM = _norm_class_weights()


def _points_for_rank_idx(idx: int, p: float = EXPONENT_P) -> float:
    return 100.0 * ((idx / (len(RANKS) - 1)) ** p)


def _band_label(metric: str, value: float) -> str:
    if metric not in BAND_CUTS or not np.isfinite(value):
        return "unusable"
    cuts = BAND_CUTS[metric]
    for idx, upper_bound in enumerate(cuts):
        if value <= upper_bound:
            return LABELS_BEST_FIRST[idx]
    return "unusable"


def _continuous_points(metric: str, value: float, obs_max: float) -> tuple[str, float]:
    if not np.isfinite(value):
        return "nan", float("nan")
    label = _band_label(metric, value)
    cuts = BAND_CUTS[metric]
    bounds = [0.0] + list(cuts) + [float("inf")]
    bounds_index = LABEL_TO_BOUNDS_INDEX[label]
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
        current_idx = RANK_TO_IDX[label]
        pts = _points_for_rank_idx(current_idx)
        return label, pts
    current_idx = RANK_TO_IDX[label]
    worse_idx = max(0, current_idx - 1)
    current_pts = _points_for_rank_idx(current_idx)
    worse_pts = _points_for_rank_idx(worse_idx)
    interp = float(np.clip((upper_bound - value) / (upper_bound - lower_bound), 0.0, 1.0))
    pts = worse_pts + interp * (current_pts - worse_pts)
    return label, pts


def _get_results_value(
    results: dict[str, Any],
    split: str,
    scope: str,
    component: str,
    metric: str,
    cls: int | None = None,
) -> float:
    component_key = "overall" if component == "xy" else component
    if scope == "overall":
        base = f"{split}.realism.{component_key}"
    else:
        if cls is None:
            return float("nan")
        base = f"{split}.realism.per_class.{int(cls)}.{component_key}"
    if metric == "mmd2_gap":
        rvs = _to_float(_nested_get(results, f"{base}.mmd2_rvs"))
        rvr = _to_float(_nested_get(results, f"{base}.mmd2_rvr_med"))
        return float(max(rvs - rvr, 0.0)) if (np.isfinite(rvs) and np.isfinite(rvr)) else float("nan")
    return _to_float(_nested_get(results, f"{base}.{metric}"))


def _cell_sum_metrics(
    results: dict[str, Any],
    split: str,
    scope: str,
    component: str,
    cls: int | None,
) -> float:
    values = np.array(
        [_get_results_value(results, split, scope, component, metric, cls) for metric in METRICS6],
        dtype=float,
    )
    if not np.isfinite(values).any():
        return float("nan")
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.nanmean(finite_values) * len(METRICS6))


def _classic_score(results: dict[str, Any]) -> float:
    numerator = 0.0
    denominator = 0.0
    for split in SPLITS:
        split_weight = float(W_SPLIT.get(split, 0.0))
        overall_weight = float(W_SCOPE["overall"])
        for component in COMPONENTS:
            component_weight = float(W_COMPONENT[component])
            cell_sum = _cell_sum_metrics(results, split, "overall", component, None)
            if np.isfinite(cell_sum):
                weight = split_weight * overall_weight * component_weight
                numerator += weight * cell_sum
                denominator += weight
        perclass_weight = float(W_SCOPE["perclass"])
        for cls in CLASSES:
            class_weight = float(W_CLASS_NORM[int(cls)])
            for component in COMPONENTS:
                component_weight = float(W_COMPONENT[component])
                cell_sum = _cell_sum_metrics(results, split, "perclass", component, int(cls))
                if np.isfinite(cell_sum):
                    weight = split_weight * perclass_weight * class_weight * component_weight
                    numerator += weight * cell_sum
                    denominator += weight
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _aggregate_metric(results: dict[str, Any], metric: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for split in SPLITS:
        split_weight = float(W_SPLIT.get(split, 0.0))
        overall_weight = float(W_SCOPE["overall"])
        for component in COMPONENTS:
            component_weight = float(W_COMPONENT[component])
            value = _get_results_value(results, split, "overall", component, metric, None)
            if np.isfinite(value):
                weight = split_weight * overall_weight * component_weight
                numerator += weight * value
                denominator += weight
        perclass_weight = float(W_SCOPE["perclass"])
        for cls in CLASSES:
            class_weight = float(W_CLASS_NORM[int(cls)])
            for component in COMPONENTS:
                component_weight = float(W_COMPONENT[component])
                value = _get_results_value(results, split, "perclass", component, metric, int(cls))
                if np.isfinite(value):
                    weight = split_weight * perclass_weight * class_weight * component_weight
                    numerator += weight * value
                    denominator += weight
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _classic_part(results: dict[str, Any], split: str, scope: str) -> float:
    numerator = 0.0
    denominator = 0.0
    if scope == "overall":
        for component in COMPONENTS:
            component_weight = float(W_COMPONENT[component])
            cell_sum = _cell_sum_metrics(results, split, "overall", component, None)
            if np.isfinite(cell_sum):
                numerator += component_weight * cell_sum
                denominator += component_weight
        return float(numerator / denominator) if denominator > 0 else float("nan")

    for cls in CLASSES:
        class_weight = float(W_CLASS_NORM[int(cls)])
        for component in COMPONENTS:
            component_weight = float(W_COMPONENT[component])
            cell_sum = _cell_sum_metrics(results, split, "perclass", component, int(cls))
            if np.isfinite(cell_sum):
                weight = class_weight * component_weight
                numerator += weight * cell_sum
                denominator += weight
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _banded_score(results: dict[str, Any], obs_max: dict[str, float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for split in SPLITS:
        split_weight = float(W_SPLIT.get(split, 0.0))
        overall_weight = float(W_SCOPE["overall"])
        for component in COMPONENTS:
            component_weight = float(W_COMPONENT[component])
            base_weight = split_weight * overall_weight * component_weight
            for metric in METRICS6:
                value = _get_results_value(results, split, "overall", component, metric, None)
                _, points = _continuous_points(metric, value, obs_max=obs_max[metric])
                if np.isfinite(points):
                    weight = base_weight * METRIC_WEIGHT[metric]
                    numerator += weight * points
                    denominator += weight
        perclass_weight = float(W_SCOPE["perclass"])
        for cls in CLASSES:
            class_weight = float(W_CLASS_NORM[int(cls)])
            for component in COMPONENTS:
                component_weight = float(W_COMPONENT[component])
                base_weight = split_weight * perclass_weight * class_weight * component_weight
                for metric in METRICS6:
                    value = _get_results_value(results, split, "perclass", component, metric, int(cls))
                    _, points = _continuous_points(metric, value, obs_max=obs_max[metric])
                    if np.isfinite(points):
                        weight = base_weight * METRIC_WEIGHT[metric]
                        numerator += weight * points
                        denominator += weight
    return round(float(numerator / denominator), 3) if denominator > 0 else float("nan")


def _banded_group_points(results: dict[str, Any], obs_max: dict[str, float], group: str) -> float:
    metrics = METRIC_GROUPS[group]
    group_weight = float(GROUP_WEIGHTS[group])
    numerator = 0.0
    denominator = 0.0
    for split in SPLITS:
        split_weight = float(W_SPLIT.get(split, 0.0))
        overall_weight = float(W_SCOPE["overall"])
        for component in COMPONENTS:
            component_weight = float(W_COMPONENT[component])
            base_weight = split_weight * overall_weight * component_weight
            for metric in metrics:
                value = _get_results_value(results, split, "overall", component, metric, None)
                _, points = _continuous_points(metric, value, obs_max=obs_max[metric])
                if np.isfinite(points):
                    within_group_weight = METRIC_WEIGHT[metric] / group_weight
                    weight = base_weight * within_group_weight
                    numerator += weight * points
                    denominator += weight
        perclass_weight = float(W_SCOPE["perclass"])
        for cls in CLASSES:
            class_weight = float(W_CLASS_NORM[int(cls)])
            for component in COMPONENTS:
                component_weight = float(W_COMPONENT[component])
                base_weight = split_weight * perclass_weight * class_weight * component_weight
                for metric in metrics:
                    value = _get_results_value(results, split, "perclass", component, metric, int(cls))
                    _, points = _continuous_points(metric, value, obs_max=obs_max[metric])
                    if np.isfinite(points):
                        within_group_weight = METRIC_WEIGHT[metric] / group_weight
                        weight = base_weight * within_group_weight
                        numerator += weight * points
                        denominator += weight
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _resolve_artifact(
    run_dir: Path,
    *,
    canonical_name: str,
    suffix: str,
    fallback_stem: str | None = None,
) -> Path | None:
    candidates = [run_dir / canonical_name]
    if fallback_stem:
        candidates.append(run_dir / f"{fallback_stem}{suffix}")
    matches = sorted(run_dir.glob(f"*{suffix}"))
    for match in matches:
        if match not in candidates:
            candidates.append(match)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_config_artifact(run_dir: Path, run_id: str) -> Path | None:
    candidates = [
        run_dir / "config.yaml",
        run_dir / f"{run_id}.yaml",
        run_dir / f"{run_dir.name}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    yaml_matches = sorted(
        path
        for path in run_dir.glob("*.yaml")
        if path.name not in {"results.yaml", "config.yaml"} and not path.name.endswith("_results.yaml")
    )
    if len(yaml_matches) == 1:
        return yaml_matches[0]
    return None


def _discover_runs_under(path: Path) -> list[Path]:
    results_path = _resolve_artifact(
        path,
        canonical_name="results.yaml",
        suffix="_results.yaml",
        fallback_stem=path.name,
    )
    if results_path is not None:
        return [path]
    run_dirs: list[Path] = []
    for child in sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")):
        run_dirs.extend(_discover_runs_under(child))
    return run_dirs


def _discover_candidate_run_dirs(root: Path) -> tuple[list[Path], list[dict[str, Any]], list[str]]:
    run_dirs: list[Path] = []
    skipped_rows: list[dict[str, Any]] = []
    discovered_non_run_roots: list[str] = []

    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if child.name in EXCLUDED_ROOT_SUBTREES:
            skipped_rows.append(
                {
                    "status": "excluded_subtree",
                    "path": str(child),
                    "reason": f"explicitly excluded root subtree '{child.name}'",
                }
            )
            continue
        if child.name.startswith("."):
            skipped_rows.append(
                {
                    "status": "excluded_hidden_dir",
                    "path": str(child),
                    "reason": "hidden root directory",
                }
            )
            continue
        child_runs = _discover_runs_under(child)
        if child_runs:
            run_dirs.extend(child_runs)
            continue
        discovered_non_run_roots.append(child.name)
        skipped_rows.append(
            {
                "status": "excluded_non_run_root_dir",
                "path": str(child),
                "reason": "root child without any discoverable run artifacts",
            }
        )

    return sorted(run_dirs), skipped_rows, discovered_non_run_roots


def _load_run_record(run_dir: Path) -> tuple[RunRecord | None, dict[str, Any] | None]:
    run_id_guess = run_dir.name
    results_path = _resolve_artifact(
        run_dir,
        canonical_name="results.yaml",
        suffix="_results.yaml",
        fallback_stem=run_id_guess,
    )
    if results_path is None:
        return None, {
            "status": "invalid_run",
            "run_dir": str(run_dir),
            "run_id": run_id_guess,
            "reason": "missing results artifact",
        }

    try:
        results = load_yaml(results_path)
    except Exception as exc:
        return None, {
            "status": "invalid_run",
            "run_dir": str(run_dir),
            "run_id": run_id_guess,
            "reason": f"failed to load results yaml: {type(exc).__name__}: {exc}",
        }
    if not isinstance(results, dict) or not results:
        return None, {
            "status": "invalid_run",
            "run_dir": str(run_dir),
            "run_id": run_id_guess,
            "reason": "empty or invalid results payload",
        }

    manifest_path = _resolve_artifact(
        run_dir,
        canonical_name="run_manifest.json",
        suffix="_run_manifest.json",
        fallback_stem=run_id_guess,
    )
    manifest = load_json(manifest_path) if manifest_path is not None else {}
    run_id = str(manifest.get("run_id") or run_id_guess)

    config_path = _resolve_config_artifact(run_dir, run_id)
    config = load_yaml(config_path) if config_path is not None else {}

    metrics_long_path = _resolve_artifact(
        run_dir,
        canonical_name="metrics_long.csv",
        suffix="_metrics_long.csv",
        fallback_stem=run_id,
    )

    missing_optional_artifacts: list[str] = []
    if manifest_path is None:
        missing_optional_artifacts.append("run_manifest")
    if config_path is None:
        missing_optional_artifacts.append("config")
    if metrics_long_path is None:
        missing_optional_artifacts.append("metrics_long")

    run_record = RunRecord(
        run_id=run_id,
        run_dir=run_dir,
        results_path=results_path,
        results=results,
        manifest_path=manifest_path,
        manifest=manifest,
        config_path=config_path,
        config=config,
        metrics_long_path=metrics_long_path,
        missing_optional_artifacts=missing_optional_artifacts,
    )
    return run_record, None


def _has_any_finite_metric(results: dict[str, Any]) -> bool:
    for split in SPLITS:
        for component in COMPONENTS:
            for metric in METRICS6:
                value = _get_results_value(results, split, "overall", component, metric, None)
                if np.isfinite(value):
                    return True
            for cls in CLASSES:
                for metric in METRICS6:
                    value = _get_results_value(results, split, "perclass", component, metric, int(cls))
                    if np.isfinite(value):
                        return True
    return False


def _compute_obs_max(run_records: list[RunRecord]) -> dict[str, float]:
    obs_max = {metric: 0.0 for metric in METRICS6}
    for record in run_records:
        results = record.results
        for split in SPLITS:
            for component in COMPONENTS:
                for metric in METRICS6:
                    value = _get_results_value(results, split, "overall", component, metric, None)
                    if np.isfinite(value):
                        obs_max[metric] = max(obs_max[metric], float(value))
                for cls in CLASSES:
                    for metric in METRICS6:
                        value = _get_results_value(results, split, "perclass", component, metric, int(cls))
                        if np.isfinite(value):
                            obs_max[metric] = max(obs_max[metric], float(value))
    return obs_max


def _run_seed(record: RunRecord) -> int | None:
    candidates = [
        record.manifest.get("seed"),
        record.results.get("seed"),
        record.config.get("seed"),
        (record.config.get("training") or {}).get("seed") if isinstance(record.config, dict) else None,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return int(candidate)
        except Exception:
            continue
    return None


def _base_axes(record: RunRecord) -> dict[str, Any]:
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
    }


def _build_row(record: RunRecord, obs_max: dict[str, float]) -> dict[str, Any]:
    classic_score = _classic_score(record.results)
    banded_score = _banded_score(record.results, obs_max)
    metric_aggs = {f"metric_agg_{metric}": _aggregate_metric(record.results, metric) for metric in ROOT_METRICS}

    classic_parts = {
        "classic_part_train_overall": _classic_part(record.results, "train", "overall"),
        "classic_part_train_perclass": _classic_part(record.results, "train", "perclass"),
        "classic_part_val_overall": _classic_part(record.results, "val", "overall"),
        "classic_part_val_perclass": _classic_part(record.results, "val", "perclass"),
    }
    banded_groups = {
        "banded_ks_points": _banded_group_points(record.results, obs_max, "ks"),
        "banded_w1_points": _banded_group_points(record.results, obs_max, "w1"),
        "banded_mmd_points": _banded_group_points(record.results, obs_max, "mmd"),
    }

    phase1 = record.results.get("phase1") or {}
    finetune = record.results.get("finetune") or {}
    axes = _base_axes(record)

    row = {
        "run_id": record.run_id,
        "run_dir": str(record.run_dir),
        "results_path": str(record.results_path),
        "run_manifest_path": None if record.manifest_path is None else str(record.manifest_path),
        "config_path": None if record.config_path is None else str(record.config_path),
        "metrics_long_path": None if record.metrics_long_path is None else str(record.metrics_long_path),
        "seed": _run_seed(record),
        "split_id": record.manifest.get("split_id"),
        "contract_id": record.manifest.get("contract_id"),
        "test_enabled": record.manifest.get("test_enabled"),
        "policy_id": axes["policy_id"],
        "policy_origin": axes["policy_origin"],
        "flowgen_base_run_id": axes["flowgen_base_run_id"],
        "flowgen_base_work_base_id": axes["flowgen_base_work_base_id"],
        "flowgen_base_seed": axes["flowgen_base_seed"],
        "paired_flowpre_source_id": axes["paired_flowpre_source_id"],
        "phase1_best_epoch": phase1.get("best_epoch"),
        "finetune_best_epoch": finetune.get("best_epoch"),
        "classic_score": classic_score,
        "banded_score": banded_score,
        "missing_optional_artifacts": list(record.missing_optional_artifacts),
    }
    row.update(classic_parts)
    row.update(banded_groups)
    row.update(metric_aggs)
    return row


def _ordered_by_lens(df: pd.DataFrame, *, score_col: str, ascending: bool) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ordered = df.sort_values(
        by=[score_col, "run_id"],
        ascending=[ascending, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    return ordered


def _add_blended_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mm_banded_inv = _minmax01(-pd.to_numeric(out["banded_score"], errors="coerce"))
    out["mm_banded_inv"] = mm_banded_inv
    for metric in METRICS4:
        metric_col = f"metric_agg_{metric}"
        metric_mm = _minmax01(pd.to_numeric(out[metric_col], errors="coerce"))
        out[f"metric_mm_{metric}"] = metric_mm
        out[f"blended_{metric}"] = 0.5 * mm_banded_inv + 0.5 * metric_mm
    return out


def _rank_dataframe(df: pd.DataFrame, *, top_k: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    ranked = _add_blended_scores(df)
    top_tables: dict[str, pd.DataFrame] = {}

    for lens_name, spec in LENS_SPECS.items():
        ordered = _ordered_by_lens(ranked, score_col=spec["score_col"], ascending=bool(spec["ascending"]))
        rank_map = {run_id: idx + 1 for idx, run_id in enumerate(ordered["run_id"].tolist())}
        ranked[f"rank_{lens_name}"] = ranked["run_id"].map(rank_map)
        top_tables[lens_name] = ordered.head(top_k).copy()

    return ranked, top_tables


def _union_top_rows(ranked_df: pd.DataFrame, top_tables: dict[str, pd.DataFrame], *, top_k: int) -> pd.DataFrame:
    union_run_ids: set[str] = set()
    for lens_name in LENS_SPECS:
        union_run_ids.update(top_tables[lens_name]["run_id"].tolist())

    union_df = ranked_df[ranked_df["run_id"].isin(union_run_ids)].copy()
    for lens_name in LENS_SPECS:
        top_ids = set(top_tables[lens_name]["run_id"].tolist())
        union_df[f"in_top{top_k}_{lens_name}"] = union_df["run_id"].isin(top_ids)

    union_flags = [f"in_top{top_k}_{lens_name}" for lens_name in LENS_SPECS]
    rank_cols = [f"rank_{lens_name}" for lens_name in LENS_SPECS]
    union_df["union_appearance_count"] = union_df[union_flags].sum(axis=1)
    union_df["best_union_rank"] = union_df[rank_cols].min(axis=1)
    union_df = union_df.sort_values(
        by=["union_appearance_count", "best_union_rank", "run_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return union_df


def _write_table_pair(base_path: Path, df: pd.DataFrame) -> tuple[Path, Path]:
    csv_path = base_path.with_suffix(".csv")
    json_path = base_path.with_suffix(".json")
    rows = df.to_dict(orient="records")
    _write_csv(csv_path, rows)
    _write_json(json_path, {"rows": rows})
    return csv_path, json_path


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    top_k = int(args.top_k)

    if not root.exists():
        raise FileNotFoundError(f"Official FlowGen root does not exist: {root}")

    run_dirs, skipped_rows, non_run_roots = _discover_candidate_run_dirs(root)

    valid_records: list[RunRecord] = []
    for run_dir in run_dirs:
        record, invalid_row = _load_run_record(run_dir)
        if invalid_row is not None:
            skipped_rows.append(invalid_row)
            continue
        if record is None:
            continue
        if not _has_any_finite_metric(record.results):
            skipped_rows.append(
                {
                    "status": "invalid_run",
                    "run_dir": str(run_dir),
                    "run_id": record.run_id,
                    "reason": "results.yaml does not expose any finite v6 realism metric",
                }
            )
            continue
        valid_records.append(record)

    stamp = _utc_stamp()
    prefix = f"flowgen_official_v6_rankings_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped_csv = output_dir / f"{prefix}_skipped_items.csv"
    skipped_json = output_dir / f"{prefix}_skipped_items.json"
    _write_csv(skipped_csv, skipped_rows)
    _write_json(skipped_json, {"rows": skipped_rows})

    if not valid_records:
        summary_payload = {
            "root": str(root),
            "output_dir": str(output_dir),
            "timestamp_utc": stamp,
            "top_k": top_k,
            "explicitly_excluded_root_subtrees": sorted(EXCLUDED_ROOT_SUBTREES),
            "detected_non_run_root_dirs": sorted(non_run_roots),
            "valid_run_count": 0,
            "skipped_count": len(skipped_rows),
            "generated_files": {
                "skipped_csv": str(skipped_csv),
                "skipped_json": str(skipped_json),
            },
        }
        summary_path = output_dir / f"{prefix}_summary.json"
        _write_json(summary_path, summary_payload)
        print(f"No valid official FlowGen runs found under {root}.")
        print(f"Skipped inventory: {skipped_csv}")
        print(f"Summary: {summary_path}")
        return 0

    obs_max = _compute_obs_max(valid_records)
    rows = [_build_row(record, obs_max) for record in valid_records]
    ranked_df, top_tables = _rank_dataframe(pd.DataFrame(rows), top_k=top_k)
    ranked_df = _ordered_by_lens(ranked_df, score_col="classic_score", ascending=True)
    union_df = _union_top_rows(ranked_df, top_tables, top_k=top_k)

    global_csv, global_json = _write_table_pair(output_dir / f"{prefix}_all_runs", ranked_df)
    union_csv, union_json = _write_table_pair(output_dir / f"{prefix}_union_top{top_k}", union_df)

    lens_output_paths: dict[str, dict[str, str]] = {}
    for lens_name, top_df in top_tables.items():
        csv_path, json_path = _write_table_pair(output_dir / f"{prefix}_top{top_k}_{lens_name}", top_df)
        lens_output_paths[lens_name] = {"csv": str(csv_path), "json": str(json_path)}

    summary_payload = {
        "root": str(root),
        "output_dir": str(output_dir),
        "timestamp_utc": stamp,
        "top_k": top_k,
        "explicitly_excluded_root_subtrees": sorted(EXCLUDED_ROOT_SUBTREES),
        "detected_non_run_root_dirs": sorted(non_run_roots),
        "valid_run_count": len(valid_records),
        "skipped_count": len(skipped_rows),
        "obs_max": obs_max,
        "generated_files": {
            "all_runs_csv": str(global_csv),
            "all_runs_json": str(global_json),
            "union_csv": str(union_csv),
            "union_json": str(union_json),
            "skipped_csv": str(skipped_csv),
            "skipped_json": str(skipped_json),
            "lens_top_tables": lens_output_paths,
        },
    }
    summary_path = output_dir / f"{prefix}_summary.json"
    _write_json(summary_path, summary_payload)

    print(f"Official FlowGen root: {root}")
    print(f"Valid runs ranked: {len(valid_records)}")
    print(f"Skipped items: {len(skipped_rows)}")
    if non_run_roots:
        print(f"Additional non-run root dirs excluded: {', '.join(sorted(non_run_roots))}")
    print(f"Global ranking CSV: {global_csv}")
    print(f"Union CSV: {union_csv}")
    print(f"Summary JSON: {summary_path}")
    for lens_name, top_df in top_tables.items():
        print(f"Top {top_k} [{lens_name}]: {', '.join(top_df['run_id'].tolist())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
