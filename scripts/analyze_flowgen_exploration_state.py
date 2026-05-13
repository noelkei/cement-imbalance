#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sets import load_or_create_raw_splits
from losses.flowgen_loss import _iqr, _perdim_w1_normed
from scripts.f6_common import load_json, load_yaml, write_json
from training.train_flowgen import (
    _loss_kwargs_from_train_cfg,
    build_flowgen_model,
    prepare_flowgen_dataloader,
    select_device,
)


MODEL_FAMILY = "flowgen"
OFFICIAL_ROOT = ROOT / "outputs" / "models" / "official" / MODEL_FAMILY
TRAINONLY_ROOT = ROOT / "outputs" / "models" / "experimental" / "train_only" / MODEL_FAMILY
REPORT_ROOT = ROOT / "outputs" / "reports" / "flowgen_exploration_state"
OFFICIAL_EXCLUDE = {"campaign_summaries", "shortlist"}
TRAINONLY_EXCLUDE = {"campaign_summaries"}
TRAIN_ONLY_POLICY = "train_only"
DEFAULT_CONDITION_COL = "type"
DEFAULT_SPLIT_ID = "init_temporal_processed_v1"
DEFAULT_DATASET_NAME = "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"

REALISM_METRICS = [
    "ks_mean",
    "ks_median",
    "w1_mean",
    "w1_median",
    "w1_mean_trainaligned",
    "w1_median_trainaligned",
    "pearson_fro_rel",
    "spearman_fro_rel",
    "xy_pearson_fro_rel",
    "xy_spearman_fro_rel",
    "mmd2_rvs",
    "mmd2_rvr_med",
    "mmd2_ratio",
]
COMPONENTS = ["overall", "x", "y"]
PERCLASS_AGG_METRICS = [
    "ks_mean",
    "w1_mean",
    "w1_mean_trainaligned",
    "mmd2_rvs",
    "xy_pearson_fro_rel",
    "xy_spearman_fro_rel",
]
TEMPORAL_METRICS = ["ks_mean", "w1_mean", "w1_mean_trainaligned", "mmd2_rvs"]
TEMPORAL_COMPONENTS = ["overall", "x", "y"]
TEMPORAL_PAIR_SPECS = {
    "quartiles": [("q1", "q4", "q4_minus_q1")],
    "prefix_suffix": [
        ("prefix_25", "suffix_25", "suffix25_minus_prefix25"),
        ("prefix_50", "suffix_50", "suffix50_minus_prefix50"),
    ],
}
NUMERIC_POLICY_AXES = [
    "mmd_xy_weight",
    "mmd_x_weight",
    "w1_x_weight",
    "mmd_y_weight",
    "w1_y_weight",
    "ks_y_weight",
    "corr_xy_pearson_weight",
    "corr_xy_spearman_weight",
    "w1_x_softclip_s",
    "w1_y_softclip_s",
    "w1_x_clip_perdim",
    "w1_y_clip_perdim",
    "w1_x_agg_softcap",
    "w1_y_agg_softcap",
    "realism_warmup_epochs",
    "realism_ramp_epochs",
]
BOOL_POLICY_AXES = [
    "use_mmd_xy",
    "use_mmd_x",
    "use_w1_x",
    "use_mmd_y",
    "use_w1_y",
    "use_ks_x",
    "use_ks_y",
    "use_corr_xy_pearson",
    "use_corr_xy_spearman",
    "enforce_realism",
]
KEY_OUTCOME_COLUMNS = [
    "monitor_realism_overall_w1_mean",
    "monitor_realism_overall_w1_mean_trainaligned",
    "monitor_realism_x_w1_mean",
    "monitor_realism_x_w1_mean_trainaligned",
    "monitor_realism_y_w1_mean",
    "monitor_realism_y_w1_mean_trainaligned",
    "monitor_realism_overall_ks_mean",
    "monitor_realism_overall_mmd2_rvs",
    "monitor_realism_overall_xy_pearson_fro_rel",
    "monitor_realism_overall_xy_spearman_fro_rel",
    "gap_val_minus_train_realism_x_w1_mean",
    "gap_val_minus_train_realism_x_w1_mean_trainaligned",
    "val_temporal_quartiles_generated_vs_slice_overall_w1_mean_q4_minus_q1",
    "val_temporal_quartiles_generated_vs_slice_overall_w1_mean_trainaligned_q4_minus_q1",
    "val_temporal_quartiles_excess_overall_w1_mean_q4_minus_q1",
    "val_temporal_quartiles_excess_overall_w1_mean_trainaligned_q4_minus_q1",
]
KEY_DELTA_COLUMNS = [
    "monitor_realism_overall_w1_mean",
    "monitor_realism_overall_w1_mean_trainaligned",
    "monitor_realism_x_w1_mean",
    "monitor_realism_x_w1_mean_trainaligned",
    "monitor_realism_y_w1_mean",
    "monitor_realism_y_w1_mean_trainaligned",
    "monitor_realism_overall_ks_mean",
    "monitor_realism_overall_mmd2_rvs",
    "monitor_realism_overall_xy_pearson_fro_rel",
]
_RAW_SPLIT_CACHE: dict[tuple[str, str, str], tuple[pd.DataFrame, ...]] = {}


@dataclass
class RunArtifact:
    branch: str
    lineage: str
    run_id: str
    run_dir: Path
    manifest_path: Path
    config_path: Path
    results_path: Path
    metrics_long_path: Path | None
    checkpoint_path: Path | None
    manifest: dict[str, Any]
    config: dict[str, Any]
    results: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Comprehensive FlowGen analysis across official and train_only branches: bases, policies, "
            "deltas vs base, temporal drift lenses, policy-space coverage, and optional per-dimension audits."
        )
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--condition-col", default=DEFAULT_CONDITION_COL)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--skip-perdim-audit", action="store_true")
    ap.add_argument("--limit-runs", type=int, default=None, help="Optional debug limit after discovery.")
    return ap.parse_args()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _print(msg: str) -> None:
    print(msg)


def _release_memory() -> None:
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _nested_get(payload: Any, *path: Any) -> Any:
    cur = payload
    for token in path:
        if not isinstance(cur, dict):
            return None
        if token in cur:
            cur = cur[token]
            continue
        token_str = str(token)
        if token_str in cur:
            cur = cur[token_str]
            continue
        try:
            token_int = int(token)
        except Exception:
            return None
        if token_int in cur:
            cur = cur[token_int]
            continue
        return None
    return cur


def _resolve_artifact(run_dir: Path, run_id: str, kind: str, *, required: bool = True) -> Path | None:
    if kind == "manifest":
        candidates = [run_dir / "run_manifest.json", run_dir / f"{run_id}_run_manifest.json"]
    elif kind == "config":
        candidates = [run_dir / "config.yaml", run_dir / f"{run_id}.yaml"]
    elif kind == "results":
        candidates = [run_dir / "results.yaml", run_dir / f"{run_id}_results.yaml"]
    elif kind == "metrics_long":
        candidates = [run_dir / "metrics_long.csv", run_dir / f"{run_id}_metrics_long.csv"]
    elif kind == "checkpoint":
        candidates = [run_dir / "checkpoint.pt", run_dir / f"{run_id}.pt"]
    else:
        raise ValueError(f"Unsupported artifact kind: {kind}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(f"Missing {kind} artifact under {run_dir}")
    return None


def _artifact_candidates(run_dir: Path, run_id: str, kind: str) -> list[Path]:
    if kind == "manifest":
        return [run_dir / "run_manifest.json", run_dir / f"{run_id}_run_manifest.json"]
    if kind == "config":
        return [run_dir / "config.yaml", run_dir / f"{run_id}.yaml"]
    if kind == "results":
        return [run_dir / "results.yaml", run_dir / f"{run_id}_results.yaml"]
    raise ValueError(f"Unsupported loadable artifact kind: {kind}")


def _load_first_valid_yaml(candidates: list[Path]) -> tuple[Path, dict[str, Any]]:
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return candidate, load_yaml(candidate)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"No YAML artifact found in candidates: {[str(p) for p in candidates]}")


def _load_first_valid_json(candidates: list[Path]) -> tuple[Path, dict[str, Any]]:
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return candidate, load_json(candidate)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"No JSON artifact found in candidates: {[str(p) for p in candidates]}")


def _monitoring_policy(manifest: dict[str, Any], config: dict[str, Any]) -> str:
    monitoring = manifest.get("monitoring") or {}
    if isinstance(monitoring, dict) and monitoring.get("policy"):
        return str(monitoring["policy"])
    axes = manifest.get("run_level_axes") or {}
    if isinstance(axes, dict) and axes.get("monitoring_policy"):
        return str(axes["monitoring_policy"])
    train_cfg = config.get("training") or {}
    if train_cfg.get("monitoring_policy"):
        return str(train_cfg["monitoring_policy"])
    return "official"


def _classify_official_lineage(rel: Path) -> str | None:
    if not rel.parts:
        return None
    head = rel.parts[0]
    if head in OFFICIAL_EXCLUDE:
        return None
    if head == "bases":
        return "base"
    if head == "reseed_final":
        return "reseed"
    return "train"


def _classify_trainonly_lineage(rel: Path) -> str | None:
    if not rel.parts:
        return None
    head = rel.parts[0]
    if head in TRAINONLY_EXCLUDE:
        return None
    if head == "bases":
        return "base"
    if head == "round1":
        return "train"
    return None


def _discover_runs() -> list[RunArtifact]:
    runs: list[RunArtifact] = []
    for manifest_path in sorted(OFFICIAL_ROOT.rglob("run_manifest.json")):
        rel = manifest_path.parent.relative_to(OFFICIAL_ROOT)
        lineage = _classify_official_lineage(rel)
        if lineage is None:
            continue
        run_dir = manifest_path.parent
        manifest_path, manifest = _load_first_valid_json(_artifact_candidates(run_dir, run_dir.name, "manifest"))
        run_id = str(manifest.get("run_id") or run_dir.name)
        config_path, config = _load_first_valid_yaml(_artifact_candidates(run_dir, run_id, "config"))
        results_path, results = _load_first_valid_yaml(_artifact_candidates(run_dir, run_id, "results"))
        metrics_long_path = _resolve_artifact(run_dir, run_id, "metrics_long", required=False)
        checkpoint_path = _resolve_artifact(run_dir, run_id, "checkpoint", required=False)
        runs.append(
            RunArtifact(
                branch="official",
                lineage=lineage,
                run_id=run_id,
                run_dir=run_dir,
                manifest_path=manifest_path,
                config_path=config_path,
                results_path=results_path,
                metrics_long_path=metrics_long_path,
                checkpoint_path=checkpoint_path,
                manifest=manifest,
                config=config,
                results=results,
            )
        )

    for manifest_path in sorted(TRAINONLY_ROOT.rglob("run_manifest.json")):
        rel = manifest_path.parent.relative_to(TRAINONLY_ROOT)
        lineage = _classify_trainonly_lineage(rel)
        if lineage is None:
            continue
        run_dir = manifest_path.parent
        manifest_path, manifest = _load_first_valid_json(_artifact_candidates(run_dir, run_dir.name, "manifest"))
        run_id = str(manifest.get("run_id") or run_dir.name)
        config_path, config = _load_first_valid_yaml(_artifact_candidates(run_dir, run_id, "config"))
        results_path, results = _load_first_valid_yaml(_artifact_candidates(run_dir, run_id, "results"))
        metrics_long_path = _resolve_artifact(run_dir, run_id, "metrics_long", required=False)
        checkpoint_path = _resolve_artifact(run_dir, run_id, "checkpoint", required=False)
        runs.append(
            RunArtifact(
                branch="train_only",
                lineage=lineage,
                run_id=run_id,
                run_dir=run_dir,
                manifest_path=manifest_path,
                config_path=config_path,
                results_path=results_path,
                metrics_long_path=metrics_long_path,
                checkpoint_path=checkpoint_path,
                manifest=manifest,
                config=config,
                results=results,
            )
        )
    return sorted(runs, key=lambda item: (item.branch, item.lineage, item.run_id))


def _policy_axes(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = dict(config.get("training") or {})
    model_cfg = dict(config.get("model") or {})
    payload: dict[str, Any] = {
        "hidden_features": model_cfg.get("hidden_features"),
        "num_layers": model_cfg.get("num_layers"),
        "num_bins": model_cfg.get("num_bins"),
        "tail_bound": model_cfg.get("tail_bound"),
        "n_repeat_blocks": model_cfg.get("n_repeat_blocks"),
        "final_rq_layers": model_cfg.get("final_rq_layers"),
        "embedding_dim": model_cfg.get("embedding_dim"),
        "affine_rq_ratio": None,
    }
    ratio = model_cfg.get("affine_rq_ratio")
    if isinstance(ratio, (list, tuple)) and len(ratio) == 2:
        payload["affine_rq_ratio"] = f"{ratio[0]}x{ratio[1]}"
    for key in NUMERIC_POLICY_AXES:
        payload[key] = train_cfg.get(key)
    for key in BOOL_POLICY_AXES:
        payload[key] = train_cfg.get(key)
    return payload


def _policy_regime_label(row: dict[str, Any]) -> str:
    mmd_x = _safe_float(row.get("mmd_x_weight"))
    mmd_y = _safe_float(row.get("mmd_y_weight"))
    w1_x = _safe_float(row.get("w1_x_weight"))
    if mmd_x is None or mmd_y is None or w1_x is None:
        base = "unknown"
    elif mmd_x <= 0.48 and mmd_y <= 1.25:
        base = "lowmmd"
    elif mmd_x <= 0.5 and mmd_y <= 1.5:
        base = "stdmmd"
    else:
        base = "altmmd"
    if w1_x < 150:
        w1_band = "w1low"
    elif w1_x < 650:
        w1_band = "w1bridge"
    elif w1_x < 900:
        w1_band = "w1strong"
    else:
        w1_band = "w1extreme"
    flags: list[str] = []
    if row.get("use_ks_y") is True:
        flags.append("ksy")
    if row.get("use_mmd_xy") is True:
        flags.append("mmdxy")
    if row.get("use_corr_xy_pearson") is True or row.get("use_corr_xy_spearman") is True:
        flags.append("xycorr")
    if (_safe_float(row.get("realism_warmup_epochs")) or 0.0) > 0 or (_safe_float(row.get("realism_ramp_epochs")) or 0.0) > 0:
        flags.append("ramp")
    default_softclip = {
        "w1_x_softclip_s": 1.25,
        "w1_y_softclip_s": 1.25,
        "w1_x_clip_perdim": 2.0,
        "w1_y_clip_perdim": 2.0,
        "w1_x_agg_softcap": 2.0,
        "w1_y_agg_softcap": 2.0,
    }
    softclip_changed = False
    for key, default in default_softclip.items():
        value = _safe_float(row.get(key))
        if value is not None and abs(value - default) > 1e-9:
            softclip_changed = True
            break
    if softclip_changed:
        flags.append("softclip")
    label = f"{base}|{w1_band}"
    if flags:
        label += "|" + "+".join(flags)
    return label


def _extract_temporal_deltas(results: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    block = _nested_get(results, "val", "temporal_realism")
    if not isinstance(block, dict):
        return out
    for family, pair_specs in TEMPORAL_PAIR_SPECS.items():
        family_block = _nested_get(block, family, "slices")
        if not isinstance(family_block, dict):
            continue
        for left_id, right_id, pair_label in pair_specs:
            left = family_block.get(left_id) or {}
            right = family_block.get(right_id) or {}
            for suite_name in ("generated_vs_slice", "train_ref_vs_slice_real"):
                for component in TEMPORAL_COMPONENTS:
                    for metric_name in TEMPORAL_METRICS:
                        left_value = _safe_float(_nested_get(left, suite_name, component, metric_name))
                        right_value = _safe_float(_nested_get(right, suite_name, component, metric_name))
                        if left_value is None or right_value is None:
                            continue
                        key = f"val_temporal_{family}_{suite_name}_{component}_{metric_name}_{pair_label}"
                        out[key] = float(right_value - left_value)
                for component in TEMPORAL_COMPONENTS:
                    for metric_name in TEMPORAL_METRICS:
                        gen_key = f"val_temporal_{family}_generated_vs_slice_{component}_{metric_name}_{pair_label}"
                        ref_key = f"val_temporal_{family}_train_ref_vs_slice_real_{component}_{metric_name}_{pair_label}"
                        if gen_key in out and ref_key in out:
                            out[f"val_temporal_{family}_excess_{component}_{metric_name}_{pair_label}"] = (
                                float(out[gen_key]) - float(out[ref_key])
                            )
    return out


def _build_run_row(record: RunArtifact) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_axes = record.manifest.get("run_level_axes") or {}
    official_training = record.config.get("official_training") or {}
    monitoring_policy = _monitoring_policy(record.manifest, record.config)
    monitor_split = "train" if monitoring_policy == TRAIN_ONLY_POLICY else "val"
    base_run_id = (
        run_axes.get("flowgen_base_run_id")
        or official_training.get("base_run_id")
        or (record.run_id if record.lineage == "base" else None)
    )
    policy_id = run_axes.get("policy_id") or official_training.get("policy_id")
    policy_signature = run_axes.get("policy_signature") or official_training.get("policy_signature")
    row: dict[str, Any] = {
        "branch": record.branch,
        "lineage": record.lineage,
        "run_id": record.run_id,
        "run_dir": str(record.run_dir),
        "manifest_path": str(record.manifest_path),
        "config_path": str(record.config_path),
        "results_path": str(record.results_path),
        "metrics_long_path": None if record.metrics_long_path is None else str(record.metrics_long_path),
        "checkpoint_path": None if record.checkpoint_path is None else str(record.checkpoint_path),
        "dataset_name": record.manifest.get("dataset_name"),
        "split_id": record.manifest.get("split_id"),
        "seed": record.manifest.get("seed"),
        "contract_id": record.manifest.get("contract_id"),
        "comparison_group_id": record.manifest.get("comparison_group_id"),
        "monitoring_policy": monitoring_policy,
        "monitor_split": monitor_split,
        "phase": run_axes.get("phase"),
        "policy_id": policy_id,
        "policy_signature": policy_signature,
        "policy_origin": run_axes.get("policy_origin") or official_training.get("policy_origin"),
        "base_run_id": base_run_id,
        "base_work_base_id": run_axes.get("flowgen_base_work_base_id") or official_training.get("base_work_base_id"),
        "base_seed": run_axes.get("flowgen_base_seed") or official_training.get("base_seed"),
        "reseed_source_run_id": run_axes.get("reseed_source_run_id"),
        "reseed_source_seed": run_axes.get("reseed_source_seed"),
        "paired_flowpre_run_id": run_axes.get("paired_flowpre_run_id") or official_training.get("paired_flowpre_run_id"),
        "paired_flowpre_seed": run_axes.get("paired_flowpre_seed") or official_training.get("paired_flowpre_seed"),
        "paired_flowpre_source_id": run_axes.get("paired_flowpre_source_id") or official_training.get("paired_flowpre_source_id"),
        "historical_source_run_ids": _stable_json(run_axes.get("historical_source_run_ids") or official_training.get("historical_source_run_ids")),
        "phase1_best_epoch": _nested_get(record.results, "phase1", "best_epoch"),
        "phase1_total_epochs": _nested_get(record.results, "phase1", "total_epochs"),
        "finetune_enabled": _nested_get(record.results, "finetune", "enabled"),
        "finetune_best_epoch": _nested_get(record.results, "finetune", "best_epoch"),
        "finetune_total_epochs": _nested_get(record.results, "finetune", "total_epochs"),
    }
    row.update(_policy_axes(record.config))
    row["policy_regime"] = _policy_regime_label(row)

    perclass_rows: list[dict[str, Any]] = []
    for split in ("train", "val"):
        split_block = record.results.get(split) or {}
        if not isinstance(split_block, dict):
            continue
        for recon_key in (
            "rrmse_x_recon",
            "rrmse_y_recon",
            "r2_x_recon",
            "r2_y_recon",
            "loss_rrmse_x_mean_whole",
            "loss_rrmse_x_std_whole",
            "loss_rrmse_y_mean_whole",
            "loss_rrmse_y_std_whole",
        ):
            value = _safe_float(split_block.get(recon_key))
            if value is not None:
                row[f"{split}_{recon_key}"] = value
        realism = split_block.get("realism") or {}
        for component in COMPONENTS:
            suite = realism.get(component) or {}
            if not isinstance(suite, dict):
                continue
            for metric_name in REALISM_METRICS:
                value = _safe_float(suite.get(metric_name))
                if value is not None:
                    row[f"{split}_realism_{component}_{metric_name}"] = value

        per_class = realism.get("per_class") or {}
        for component in COMPONENTS:
            for metric_name in PERCLASS_AGG_METRICS:
                values: list[float] = []
                for cls_id, suites in per_class.items():
                    suite = (suites or {}).get(component) or {}
                    value = _safe_float(suite.get(metric_name))
                    if value is None:
                        continue
                    values.append(value)
                    perclass_rows.append(
                        {
                            "branch": record.branch,
                            "lineage": record.lineage,
                            "run_id": record.run_id,
                            "policy_id": policy_id,
                            "base_run_id": base_run_id,
                            "split": split,
                            "component": component,
                            "class_id": _safe_int(cls_id),
                            "metric_name": metric_name,
                            "metric_value": value,
                        }
                    )
                if values:
                    row[f"{split}_realism_{component}_{metric_name}_pc_macro"] = float(np.mean(values))
                    row[f"{split}_realism_{component}_{metric_name}_pc_worst"] = float(np.max(values))

    row.update(_extract_temporal_deltas(record.results))

    for component in COMPONENTS:
        for metric_name in ("w1_mean", "w1_median"):
            raw = _safe_float(row.get(f"{monitor_split}_realism_{component}_{metric_name}"))
            aligned = _safe_float(row.get(f"{monitor_split}_realism_{component}_{metric_name}_trainaligned"))
            if raw is not None:
                row[f"monitor_realism_{component}_{metric_name}"] = raw
            if aligned is not None:
                row[f"monitor_realism_{component}_{metric_name}_trainaligned"] = aligned
            if raw is not None and aligned is not None:
                row[f"monitor_realism_{component}_{metric_name}_alignment_gap_abs"] = float(raw - aligned)
                row[f"monitor_realism_{component}_{metric_name}_alignment_gap_ratio"] = float(aligned / raw) if raw != 0 else np.nan
        for metric_name in ("ks_mean", "ks_median", "mmd2_rvs", "mmd2_rvr_med", "mmd2_ratio", "pearson_fro_rel", "spearman_fro_rel", "xy_pearson_fro_rel", "xy_spearman_fro_rel"):
            value = _safe_float(row.get(f"{monitor_split}_realism_{component}_{metric_name}"))
            if value is not None:
                row[f"monitor_realism_{component}_{metric_name}"] = value

    if _safe_float(row.get("val_realism_x_w1_mean")) is not None and _safe_float(row.get("train_realism_x_w1_mean")) is not None:
        row["gap_val_minus_train_realism_x_w1_mean"] = float(row["val_realism_x_w1_mean"] - row["train_realism_x_w1_mean"])
    if _safe_float(row.get("val_realism_x_w1_mean_trainaligned")) is not None and _safe_float(row.get("train_realism_x_w1_mean_trainaligned")) is not None:
        row["gap_val_minus_train_realism_x_w1_mean_trainaligned"] = float(
            row["val_realism_x_w1_mean_trainaligned"] - row["train_realism_x_w1_mean_trainaligned"]
        )
    if _safe_float(row.get("val_realism_overall_ks_mean")) is not None and _safe_float(row.get("train_realism_overall_ks_mean")) is not None:
        row["gap_val_minus_train_realism_overall_ks_mean"] = float(
            row["val_realism_overall_ks_mean"] - row["train_realism_overall_ks_mean"]
        )

    return row, perclass_rows


def _finite_pair(series_a: pd.Series, series_b: pd.Series) -> pd.DataFrame:
    pair = pd.DataFrame({"a": pd.to_numeric(series_a, errors="coerce"), "b": pd.to_numeric(series_b, errors="coerce")})
    return pair.replace([np.inf, -np.inf], np.nan).dropna()


def _build_delta_long(runs_df: pd.DataFrame) -> pd.DataFrame:
    base_lookup = runs_df[runs_df["lineage"] == "base"].set_index("run_id", drop=False)
    metric_cols = [
        col
        for col in runs_df.columns
        if (
            col.startswith("train_realism_")
            or col.startswith("val_realism_")
            or col.startswith("monitor_realism_")
            or col.startswith("train_rrmse_")
            or col.startswith("val_rrmse_")
            or col.startswith("train_r2_")
            or col.startswith("val_r2_")
            or col.startswith("train_loss_rrmse_")
            or col.startswith("val_loss_rrmse_")
        )
    ]
    rows: list[dict[str, Any]] = []
    for _, run in runs_df.iterrows():
        if run["lineage"] == "base":
            continue
        base_run_id = run.get("base_run_id")
        if base_run_id not in base_lookup.index:
            continue
        base_row = base_lookup.loc[base_run_id]
        for metric_name in metric_cols:
            run_value = _safe_float(run.get(metric_name))
            base_value = _safe_float(base_row.get(metric_name))
            if run_value is None or base_value is None:
                continue
            rows.append(
                {
                    "branch": run["branch"],
                    "lineage": run["lineage"],
                    "run_id": run["run_id"],
                    "policy_id": run.get("policy_id"),
                    "policy_signature": run.get("policy_signature"),
                    "base_run_id": base_run_id,
                    "metric_name": metric_name,
                    "run_value": run_value,
                    "base_value": base_value,
                    "delta": float(run_value - base_value),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_numeric(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg_spec: dict[str, tuple[str, str]] = {}
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        agg_spec[f"{metric}__mean"] = (metric, "mean")
        agg_spec[f"{metric}__std"] = (metric, "std")
        agg_spec[f"{metric}__min"] = (metric, "min")
        agg_spec[f"{metric}__max"] = (metric, "max")
        agg_spec[f"{metric}__median"] = (metric, "median")
    return df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()


def _build_policy_summaries(runs_df: pd.DataFrame, delta_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    finetune_df = runs_df[runs_df["lineage"] != "base"].copy()
    key_metrics = [col for col in KEY_OUTCOME_COLUMNS if col in finetune_df.columns]
    policy_summary = _aggregate_numeric(
        finetune_df,
        ["branch", "policy_id", "policy_signature", "policy_regime"],
        key_metrics,
    )
    if not policy_summary.empty:
        counts = finetune_df.groupby(["branch", "policy_id", "policy_signature", "policy_regime"], dropna=False).agg(
            n_runs=("run_id", "count"),
            n_bases=("base_run_id", "nunique"),
            n_seeds=("seed", "nunique"),
            lineages=("lineage", lambda s: ",".join(sorted({str(v) for v in s if pd.notna(v)}))),
        ).reset_index()
        policy_summary = policy_summary.merge(
            counts,
            on=["branch", "policy_id", "policy_signature", "policy_regime"],
            how="left",
        )
    delta_wide = None
    if not delta_long.empty:
        delta_key = delta_long[delta_long["metric_name"].isin(KEY_DELTA_COLUMNS)].copy()
        if not delta_key.empty:
            delta_wide = delta_key.pivot_table(
                index=["branch", "policy_id", "policy_signature"],
                columns="metric_name",
                values="delta",
                aggfunc="mean",
            ).reset_index()
            delta_wide.columns = [
                col if isinstance(col, str) else col[1] if isinstance(col, tuple) else str(col)
                for col in delta_wide.columns
            ]
            rename_map = {
                metric: f"delta_base_{metric}__mean"
                for metric in KEY_DELTA_COLUMNS
                if metric in delta_wide.columns
            }
            delta_wide = delta_wide.rename(columns=rename_map)
    if delta_wide is not None and not policy_summary.empty:
        policy_summary = policy_summary.merge(
            delta_wide,
            on=["branch", "policy_id", "policy_signature"],
            how="left",
        )

    base_policy_summary = _aggregate_numeric(
        finetune_df,
        ["branch", "base_run_id", "policy_id", "policy_signature", "policy_regime"],
        key_metrics,
    )
    if not base_policy_summary.empty:
        counts = finetune_df.groupby(
            ["branch", "base_run_id", "policy_id", "policy_signature", "policy_regime"], dropna=False
        ).agg(
            n_runs=("run_id", "count"),
            n_seeds=("seed", "nunique"),
            lineages=("lineage", lambda s: ",".join(sorted({str(v) for v in s if pd.notna(v)}))),
        ).reset_index()
        base_policy_summary = base_policy_summary.merge(
            counts,
            on=["branch", "base_run_id", "policy_id", "policy_signature", "policy_regime"],
            how="left",
        )
    return policy_summary, base_policy_summary


def _build_base_summary(runs_df: pd.DataFrame, delta_long: pd.DataFrame) -> pd.DataFrame:
    base_df = runs_df[runs_df["lineage"] == "base"].copy()
    finetune_df = runs_df[runs_df["lineage"] != "base"].copy()
    rows: list[dict[str, Any]] = []
    for _, base in base_df.iterrows():
        descendants = finetune_df[finetune_df["base_run_id"] == base["run_id"]].copy()
        row = {
            "branch": base["branch"],
            "base_run_id": base["run_id"],
            "base_work_base_id": base.get("base_work_base_id"),
            "paired_flowpre_run_id": base.get("paired_flowpre_run_id"),
            "seed": base.get("seed"),
            "n_descendants": int(len(descendants)),
            "n_descendant_policies": int(descendants["policy_signature"].nunique()) if not descendants.empty else 0,
        }
        for metric in [
            "monitor_realism_overall_w1_mean",
            "monitor_realism_x_w1_mean",
            "monitor_realism_y_w1_mean",
            "monitor_realism_overall_ks_mean",
            "train_rrmse_x_recon",
            "train_rrmse_y_recon",
            "val_rrmse_x_recon",
            "val_rrmse_y_recon",
        ]:
            if metric in base.index:
                row[f"base_{metric}"] = base.get(metric)
        if not descendants.empty:
            for metric in [
                "monitor_realism_overall_w1_mean",
                "monitor_realism_overall_w1_mean_trainaligned",
                "monitor_realism_x_w1_mean",
                "monitor_realism_x_w1_mean_trainaligned",
                "monitor_realism_y_w1_mean",
                "monitor_realism_y_w1_mean_trainaligned",
                "monitor_realism_overall_ks_mean",
                "monitor_realism_overall_xy_pearson_fro_rel",
            ]:
                vals = pd.to_numeric(descendants.get(metric), errors="coerce")
                if vals.notna().any():
                    row[f"desc_{metric}__mean"] = float(vals.mean())
                    row[f"desc_{metric}__min"] = float(vals.min())
        if not delta_long.empty:
            subset = delta_long[delta_long["base_run_id"] == base["run_id"]]
            for metric in KEY_DELTA_COLUMNS:
                vals = pd.to_numeric(subset.loc[subset["metric_name"] == metric, "delta"], errors="coerce")
                if vals.notna().any():
                    row[f"delta_{metric}__mean"] = float(vals.mean())
                    row[f"delta_{metric}__min"] = float(vals.min())
        rows.append(row)
    return pd.DataFrame(rows)


def _build_reseed_family_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    reseed_df = runs_df[(runs_df["branch"] == "official") & (runs_df["lineage"] == "reseed")].copy()
    source_df = runs_df[(runs_df["branch"] == "official") & (runs_df["lineage"] == "train")].copy()
    if reseed_df.empty:
        return pd.DataFrame()
    key_metrics = [col for col in KEY_OUTCOME_COLUMNS if col in reseed_df.columns]
    summary = _aggregate_numeric(
        reseed_df,
        ["reseed_source_run_id", "policy_id", "policy_signature", "policy_regime", "base_run_id"],
        key_metrics,
    )
    counts = reseed_df.groupby(
        ["reseed_source_run_id", "policy_id", "policy_signature", "policy_regime", "base_run_id"], dropna=False
    ).agg(n_runs=("run_id", "count"), n_seeds=("seed", "nunique")).reset_index()
    summary = summary.merge(
        counts,
        on=["reseed_source_run_id", "policy_id", "policy_signature", "policy_regime", "base_run_id"],
        how="left",
    )
    source_small = source_df[["run_id"] + [col for col in key_metrics if col in source_df.columns]].rename(
        columns={"run_id": "reseed_source_run_id"}
    )
    summary = summary.merge(source_small, on="reseed_source_run_id", how="left", suffixes=("", "__source"))
    for metric in key_metrics:
        mean_col = f"{metric}__mean"
        src_col = metric
        if mean_col in summary.columns and src_col in summary.columns:
            summary[f"delta_vs_source_{metric}__mean"] = summary[mean_col] - summary[src_col]
    return summary


def _build_policy_axis_coverage(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    finetune_df = runs_df[runs_df["lineage"] != "base"].copy()
    for scope_name, scoped_df in [("all", finetune_df), ("official", finetune_df[finetune_df["branch"] == "official"]), ("train_only", finetune_df[finetune_df["branch"] == "train_only"])]:
        for axis in NUMERIC_POLICY_AXES + BOOL_POLICY_AXES:
            if axis not in scoped_df.columns:
                continue
            counts = scoped_df[axis].fillna("NA").astype(str).value_counts(dropna=False).sort_index()
            for value, count in counts.items():
                rows.append(
                    {
                        "scope": scope_name,
                        "axis_name": axis,
                        "axis_value": value,
                        "count": int(count),
                    }
                )
    return pd.DataFrame(rows)


def _nearest_policy_neighbors(runs_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    finetune_df = runs_df[runs_df["lineage"] != "base"].copy()
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for scope_name, scoped_df in [("all", finetune_df), ("official", finetune_df[finetune_df["branch"] == "official"]), ("train_only", finetune_df[finetune_df["branch"] == "train_only"])]:
        uniq = (
            scoped_df[
                ["policy_signature", "policy_id", "policy_regime"] + NUMERIC_POLICY_AXES + BOOL_POLICY_AXES
            ]
            .drop_duplicates(subset=["policy_signature"])
            .reset_index(drop=True)
        )
        if len(uniq) <= 1:
            summaries.append(
                {
                    "scope": scope_name,
                    "n_unique_policies": int(len(uniq)),
                    "mean_nearest_distance": np.nan,
                    "median_nearest_distance": np.nan,
                }
            )
            continue
        num = uniq[NUMERIC_POLICY_AXES].apply(pd.to_numeric, errors="coerce")
        norm = num.copy()
        for col in NUMERIC_POLICY_AXES:
            series = norm[col]
            lo = series.min(skipna=True)
            hi = series.max(skipna=True)
            if pd.isna(lo) or pd.isna(hi) or hi == lo:
                norm[col] = 0.0
            else:
                norm[col] = (series - lo) / (hi - lo)
        bool_df = uniq[BOOL_POLICY_AXES].astype("float").fillna(0.0)
        vec = pd.concat([norm.fillna(0.0), bool_df], axis=1).to_numpy(dtype=float)
        nearest: list[float] = []
        for i in range(len(uniq)):
            best_j = None
            best_d = None
            for j in range(len(uniq)):
                if i == j:
                    continue
                d = float(np.mean(np.abs(vec[i] - vec[j])))
                if best_d is None or d < best_d:
                    best_d = d
                    best_j = j
            nearest.append(best_d if best_d is not None else np.nan)
            rows.append(
                {
                    "scope": scope_name,
                    "policy_signature": uniq.loc[i, "policy_signature"],
                    "policy_id": uniq.loc[i, "policy_id"],
                    "policy_regime": uniq.loc[i, "policy_regime"],
                    "nearest_policy_signature": None if best_j is None else uniq.loc[best_j, "policy_signature"],
                    "nearest_policy_id": None if best_j is None else uniq.loc[best_j, "policy_id"],
                    "nearest_policy_regime": None if best_j is None else uniq.loc[best_j, "policy_regime"],
                    "nearest_distance": best_d,
                }
            )
        summaries.append(
            {
                "scope": scope_name,
                "n_unique_policies": int(len(uniq)),
                "mean_nearest_distance": float(np.nanmean(nearest)),
                "median_nearest_distance": float(np.nanmedian(nearest)),
                **{f"cardinality_{axis}": int(scoped_df[axis].dropna().astype(str).nunique()) for axis in NUMERIC_POLICY_AXES + BOOL_POLICY_AXES if axis in scoped_df.columns},
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def _policy_axis_correlations(runs_df: pd.DataFrame, delta_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    finetune_df = runs_df[runs_df["lineage"] != "base"].copy()
    delta_pivot = pd.DataFrame()
    if not delta_long.empty:
        delta_key = delta_long[delta_long["metric_name"].isin(KEY_DELTA_COLUMNS)].copy()
        if not delta_key.empty:
            delta_pivot = delta_key.pivot_table(
                index="run_id",
                columns="metric_name",
                values="delta",
                aggfunc="mean",
            ).reset_index()
            delta_pivot.columns = [
                col if isinstance(col, str) else col[1] if isinstance(col, tuple) else str(col)
                for col in delta_pivot.columns
            ]
            delta_pivot = delta_pivot.rename(columns={metric: f"delta_base_{metric}" for metric in KEY_DELTA_COLUMNS if metric in delta_pivot.columns})
            finetune_df = finetune_df.merge(delta_pivot, on="run_id", how="left")
    outcome_cols = [col for col in KEY_OUTCOME_COLUMNS if col in finetune_df.columns]
    outcome_cols += [col for col in finetune_df.columns if col.startswith("delta_base_")]
    corr_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for scope_name, scoped_df in [("all", finetune_df), ("official", finetune_df[finetune_df["branch"] == "official"]), ("train_only", finetune_df[finetune_df["branch"] == "train_only"])]:
        for axis in NUMERIC_POLICY_AXES:
            if axis not in scoped_df.columns:
                continue
            axis_values = pd.to_numeric(scoped_df[axis], errors="coerce")
            for outcome in outcome_cols:
                pair = _finite_pair(axis_values, pd.to_numeric(scoped_df[outcome], errors="coerce"))
                if len(pair) < 5 or pair["a"].nunique() < 3 or pair["b"].nunique() < 3:
                    continue
                rho = float(pair["a"].corr(pair["b"], method="spearman"))
                corr_rows.append({"scope": scope_name, "axis_name": axis, "outcome_name": outcome, "n": int(len(pair)), "spearman_rho": rho})
        for axis in BOOL_POLICY_AXES:
            if axis not in scoped_df.columns:
                continue
            mask = scoped_df[axis].astype("boolean")
            if mask.dropna().nunique() < 2:
                continue
            for outcome in outcome_cols:
                vals = pd.to_numeric(scoped_df[outcome], errors="coerce")
                true_vals = vals[mask == True].dropna()
                false_vals = vals[mask == False].dropna()
                if len(true_vals) < 2 or len(false_vals) < 2:
                    continue
                effect_rows.append(
                    {
                        "scope": scope_name,
                        "axis_name": axis,
                        "outcome_name": outcome,
                        "n_true": int(len(true_vals)),
                        "n_false": int(len(false_vals)),
                        "mean_true": float(true_vals.mean()),
                        "mean_false": float(false_vals.mean()),
                        "mean_true_minus_false": float(true_vals.mean() - false_vals.mean()),
                    }
                )
    return pd.DataFrame(corr_rows), pd.DataFrame(effect_rows)


def _load_raw_splits_cached(dataset_name: str, condition_col: str, split_id: str) -> tuple[pd.DataFrame, ...]:
    key = (dataset_name, condition_col, split_id)
    if key not in _RAW_SPLIT_CACHE:
        _RAW_SPLIT_CACHE[key] = load_or_create_raw_splits(
            df_name=dataset_name,
            condition_col=condition_col,
            verbose=False,
            split_id=split_id,
            split_mode="official",
        )
    return _RAW_SPLIT_CACHE[key]


def _build_cxy(X_df: pd.DataFrame, y_df: pd.DataFrame, *, condition_col: str) -> pd.DataFrame:
    X_df = X_df.sort_values("post_cleaning_index").reset_index(drop=True).copy()
    y_df = y_df.sort_values("post_cleaning_index").reset_index(drop=True).copy()
    df = X_df.merge(y_df, on="post_cleaning_index", how="inner", validate="one_to_one")
    x_cols = [col for col in X_df.columns if col not in ("post_cleaning_index", condition_col)]
    y_cols = [col for col in y_df.columns if col != "post_cleaning_index"]
    return df[["post_cleaning_index", condition_col] + x_cols + y_cols].copy()


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = np.sort(a.astype(float, copy=False))
    b = np.sort(b.astype(float, copy=False))
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / float(len(a))
    cdf_b = np.searchsorted(b, grid, side="right") / float(len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _load_run_model_bundle(record: RunArtifact, *, device_name: str, condition_col: str) -> dict[str, Any]:
    if record.checkpoint_path is None:
        raise FileNotFoundError(f"{record.run_id}: checkpoint missing")
    device = select_device(device_name)
    dataset_name = str(record.manifest.get("dataset_name") or DEFAULT_DATASET_NAME)
    split_id = str(record.manifest.get("split_id") or DEFAULT_SPLIT_ID)
    monitoring_policy = _monitoring_policy(record.manifest, record.config)
    (
        X_train,
        X_val,
        _X_test,
        y_train_df,
        y_val_df,
        _y_test,
        _r_train,
        _r_val,
        _r_test,
    ) = _load_raw_splits_cached(dataset_name, condition_col, split_id)
    cxy_train = _build_cxy(X_train, y_train_df, condition_col=condition_col)
    cxy_val = cxy_train.copy() if monitoring_policy == TRAIN_ONLY_POLICY else _build_cxy(X_val, y_val_df, condition_col=condition_col)

    (
        x_train,
        y_train,
        x_val,
        y_val,
        _x_test,
        _y_test,
        c_train,
        c_val,
        _c_test,
        feature_names_x,
        _target_names_y,
        _train_dataset,
        _train_dataloader,
    ) = prepare_flowgen_dataloader(
        df_train=cxy_train,
        df_val=cxy_val,
        condition_col=condition_col,
        batch_size=int((record.config.get("training") or {}).get("batch_size", 256)),
        device=device,
        df_test=None,
        seed=_safe_int(record.manifest.get("seed")),
    )
    model = build_flowgen_model(
        model_cfg=dict(record.config.get("model") or {}),
        x_dim=x_train.shape[1],
        y_dim=y_train.shape[1],
        num_classes=int(c_train.max().item()) + 1,
        device=device,
    )
    state_dict = torch.load(record.checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        model.load_state_dict(state_dict["state_dict"], strict=False)
    elif isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
    elif hasattr(state_dict, "state_dict"):
        model.load_state_dict(state_dict.state_dict(), strict=False)
    else:
        raise ValueError(f"{record.run_id}: unsupported checkpoint format")
    model.eval()
    return {
        "device": device,
        "monitoring_policy": monitoring_policy,
        "feature_names_x": list(feature_names_x),
        "loss_kwargs": _loss_kwargs_from_train_cfg(dict(record.config.get("training") or {})),
        "run_seed": _safe_int(record.manifest.get("seed")),
        "model": model,
        "train": (x_train, y_train, c_train),
        "monitor": ((x_train, y_train, c_train) if monitoring_policy == TRAIN_ONLY_POLICY else (x_val, y_val, c_val)),
    }


def _finite_and_clamp_to_real(xr_c: torch.Tensor, yr_c: torch.Tensor, xs_c: torch.Tensor, ys_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(xs_c).all(dim=1) & torch.isfinite(ys_c).all(dim=1)
    xs_c = xs_c[finite]
    ys_c = ys_c[finite]
    q25x = xr_c.quantile(0.25, dim=0)
    q75x = xr_c.quantile(0.75, dim=0)
    iqr_x = (q75x - q25x).clamp_min(1e-6)
    lo_x = q25x - 5.0 * iqr_x
    hi_x = q75x + 5.0 * iqr_x
    q25y = yr_c.quantile(0.25, dim=0)
    q75y = yr_c.quantile(0.75, dim=0)
    iqr_y = (q75y - q25y).clamp_min(1e-6)
    lo_y = q25y - 5.0 * iqr_y
    hi_y = q75y + 5.0 * iqr_y
    xs_c = xs_c.clamp(min=lo_x, max=hi_x)
    ys_c = ys_c.clamp(min=lo_y, max=hi_y)
    return xs_c, ys_c


def _perdim_audit_for_run(record: RunArtifact, *, device_name: str, condition_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = _load_run_model_bundle(record, device_name=device_name, condition_col=condition_col)
    model = bundle["model"]
    device = bundle["device"]
    loss_kwargs = bundle["loss_kwargs"]
    feature_names = bundle["feature_names_x"]
    run_seed = int(bundle["run_seed"] or 0)
    dx = len(feature_names)
    dy = int(bundle["train"][1].shape[1])
    dxy = dx + dy
    gen_cpu = torch.Generator(device="cpu")
    gen_cpu.manual_seed(int(run_seed) + int(loss_kwargs.get("realism_seed_offset", 0)))
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for surface_name in ("train", "monitor"):
        x_ref, y_ref, c_ref = bundle[surface_name]
        classes = [int(ci) for ci in torch.unique(c_ref).tolist()]
        synth_x_by_cls: list[torch.Tensor] = []
        synth_y_by_cls: list[torch.Tensor] = []
        real_x_by_cls = {cls: x_ref[c_ref == cls] for cls in classes}
        real_y_by_cls = {cls: y_ref[c_ref == cls] for cls in classes}
        with torch.no_grad():
            for cls in classes:
                ns = int(real_x_by_cls[cls].shape[0])
                z_s = torch.randn(ns, dxy, generator=gen_cpu, device="cpu").to(device)
                c_s = torch.full((ns,), cls, dtype=torch.long, device=device)
                (xs_c, ys_c), _ = model.inverse_xy(z_s, c_s)
                xs_c, ys_c = _finite_and_clamp_to_real(real_x_by_cls[cls], real_y_by_cls[cls], xs_c, ys_c)
                synth_x_by_cls.append(xs_c)
                synth_y_by_cls.append(ys_c)

        xs_all = torch.cat(synth_x_by_cls, dim=0)
        ys_all = torch.cat(synth_y_by_cls, dim=0)
        denom_x = _iqr(x_ref).to(torch.float32).clamp_min(1e-4)
        w1_perdim, _ = _perdim_w1_normed(x_ref, xs_all, norm=str(loss_kwargs.get("w1_x_norm", "iqr")), denom_override=denom_x)
        w1_values = w1_perdim.detach().cpu().numpy().astype(float)
        x_real = x_ref.detach().cpu().numpy()
        x_synth = xs_all.detach().cpu().numpy()
        ks_values = np.array([_ks_statistic(x_real[:, j], x_synth[:, j]) for j in range(dx)], dtype=float)
        order_w1 = np.argsort(-w1_values)
        total_w1 = float(np.sum(np.clip(w1_values, 0.0, None)))
        shares = np.clip(w1_values[order_w1], 0.0, None) / total_w1 if total_w1 > 0 else np.zeros_like(order_w1, dtype=float)
        cumulative = np.cumsum(shares)

        def _n_to_threshold(threshold: float) -> int:
            idx = np.searchsorted(cumulative, threshold, side="left")
            return int(idx + 1) if len(cumulative) else 0

        for rank, idx in enumerate(order_w1, start=1):
            rows.append(
                {
                    "branch": record.branch,
                    "lineage": record.lineage,
                    "run_id": record.run_id,
                    "policy_id": _nested_get(record.manifest, "run_level_axes", "policy_id"),
                    "base_run_id": _nested_get(record.manifest, "run_level_axes", "flowgen_base_run_id") or (record.run_id if record.lineage == "base" else None),
                    "surface": surface_name,
                    "feature_name": feature_names[int(idx)],
                    "feature_index": int(idx),
                    "w1_norm": float(w1_values[int(idx)]),
                    "ks_stat": float(ks_values[int(idx)]),
                    "rank_w1": int(rank),
                    "share_w1": float(shares[rank - 1]) if total_w1 > 0 else np.nan,
                    "cum_share_w1": float(cumulative[rank - 1]) if total_w1 > 0 else np.nan,
                }
            )
        top_features = [feature_names[int(i)] for i in order_w1[:5]]
        summaries.append(
            {
                "branch": record.branch,
                "lineage": record.lineage,
                "run_id": record.run_id,
                "surface": surface_name,
                "w1_mean_recomputed": float(np.nanmean(w1_values)),
                "w1_median_recomputed": float(np.nanmedian(w1_values)),
                "ks_mean_recomputed": float(np.nanmean(ks_values)),
                "ks_median_recomputed": float(np.nanmedian(ks_values)),
                "top1_feature": top_features[0] if len(top_features) >= 1 else None,
                "top2_feature": top_features[1] if len(top_features) >= 2 else None,
                "top3_feature": top_features[2] if len(top_features) >= 3 else None,
                "top5_signature": ",".join(top_features),
                "top1_share_w1": float(shares[0]) if len(shares) >= 1 else np.nan,
                "top3_share_w1": float(np.sum(shares[:3])) if len(shares) >= 3 else np.nan,
                "top5_share_w1": float(np.sum(shares[:5])) if len(shares) >= 5 else np.nan,
                "n_dims_50pct_w1": _n_to_threshold(0.50),
                "n_dims_80pct_w1": _n_to_threshold(0.80),
                "n_dims_90pct_w1": _n_to_threshold(0.90),
            }
        )

    del model
    _release_memory()
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def _build_perdim_outputs(runs: list[RunArtifact], *, device_name: str, condition_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    for idx, record in enumerate(runs, start=1):
        _print(f"[perdim {idx}/{len(runs)}] {record.run_id}")
        try:
            detail_df, summary_df = _perdim_audit_for_run(record, device_name=device_name, condition_col=condition_col)
            long_parts.append(detail_df)
            summary_parts.append(summary_df)
        except Exception as exc:
            _print(f"  failed {record.run_id}: {type(exc).__name__}: {exc}")
        finally:
            _release_memory()
    long_df = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
    summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    freq_df = pd.DataFrame()
    if not long_df.empty:
        top5 = long_df[long_df["rank_w1"] <= 5].copy()
        freq_df = (
            top5.groupby(["branch", "surface", "feature_name"], dropna=False)
            .agg(
                top5_hits=("run_id", "count"),
                mean_w1_norm=("w1_norm", "mean"),
                top1_hits=("rank_w1", lambda s: int((pd.to_numeric(s, errors="coerce") == 1).sum())),
                top3_hits=("rank_w1", lambda s: int((pd.to_numeric(s, errors="coerce") <= 3).sum())),
            )
            .reset_index()
            .sort_values(["branch", "surface", "top5_hits", "mean_w1_norm", "feature_name"], ascending=[True, True, False, False, True], kind="mergesort")
        )
    return long_df, summary_df, freq_df


def _build_summary_markdown(
    *,
    analysis_dir: Path,
    inventory_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    base_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
    policy_diversity: pd.DataFrame,
    reseed_summary: pd.DataFrame,
    perdim_summary: pd.DataFrame,
    perdim_freq: pd.DataFrame,
) -> Path:
    def _df_to_md(df: pd.DataFrame) -> str:
        if df.empty:
            return "_empty_"
        safe = df.copy()
        safe = safe.replace({np.nan: ""})
        headers = [str(col) for col in safe.columns]
        lines_local = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row_vals in safe.itertuples(index=False, name=None):
            lines_local.append("| " + " | ".join(str(val) for val in row_vals) + " |")
        return "\n".join(lines_local)

    lines: list[str] = []
    lines.append("# FlowGen Exploration State")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Total runs discovered: `{len(inventory_df)}`")
    lines.append(f"- Official runs: `{int((inventory_df['branch'] == 'official').sum())}`")
    lines.append(f"- Train-only runs: `{int((inventory_df['branch'] == 'train_only').sum())}`")
    lines.append(f"- Official bases: `{int(((inventory_df['branch'] == 'official') & (inventory_df['lineage'] == 'base')).sum())}`")
    lines.append(f"- Official finetunes/reseeds: `{int(((inventory_df['branch'] == 'official') & (inventory_df['lineage'] != 'base')).sum())}`")
    lines.append(f"- Train-only bases: `{int(((inventory_df['branch'] == 'train_only') & (inventory_df['lineage'] == 'base')).sum())}`")
    lines.append(f"- Train-only finetunes: `{int(((inventory_df['branch'] == 'train_only') & (inventory_df['lineage'] != 'base')).sum())}`")
    lines.append("")
    lines.append("## Bases")
    if not base_summary.empty:
        cols = [col for col in ["branch", "base_run_id", "n_descendants", "base_monitor_realism_overall_w1_mean", "base_monitor_realism_x_w1_mean", "base_train_rrmse_x_recon", "base_train_rrmse_y_recon"] if col in base_summary.columns]
        lines.append(_df_to_md(base_summary[cols]))
    else:
        lines.append("No base summary rows were produced.")
    lines.append("")
    lines.append("## Policy Diversity")
    if not policy_diversity.empty:
        keep = ["scope", "n_unique_policies", "mean_nearest_distance", "median_nearest_distance"]
        extra = [col for col in policy_diversity.columns if col.startswith("cardinality_")]
        lines.append(_df_to_md(policy_diversity[keep + extra[:8]]))
    else:
        lines.append("No policy diversity summary rows were produced.")
    lines.append("")
    lines.append("## Policy Outcomes")
    if not policy_summary.empty:
        cols = [col for col in [
            "branch",
            "policy_id",
            "policy_regime",
            "n_runs",
            "n_bases",
            "monitor_realism_x_w1_mean_trainaligned__mean",
            "monitor_realism_overall_ks_mean__mean",
            "monitor_realism_overall_xy_pearson_fro_rel__mean",
            "delta_base_monitor_realism_x_w1_mean__mean",
        ] if col in policy_summary.columns]
        lines.append(
            _df_to_md(
                policy_summary[cols]
                .sort_values(["branch", "monitor_realism_x_w1_mean_trainaligned__mean", "policy_id"], kind="mergesort")
                .head(12)
            )
        )
    else:
        lines.append("No policy summary rows were produced.")
    lines.append("")
    lines.append("## Official Reseed Families")
    if not reseed_summary.empty:
        cols = [col for col in [
            "reseed_source_run_id",
            "policy_id",
            "n_runs",
            "monitor_realism_x_w1_mean_trainaligned__mean",
            "monitor_realism_x_w1_mean_trainaligned__std",
            "delta_vs_source_monitor_realism_x_w1_mean__mean",
        ] if col in reseed_summary.columns]
        lines.append(
            _df_to_md(
                reseed_summary[cols]
                .sort_values(["monitor_realism_x_w1_mean_trainaligned__mean", "policy_id"], kind="mergesort")
            )
        )
    else:
        lines.append("No reseed-family summary rows were produced.")
    lines.append("")
    lines.append("## Per-Dimension Stress")
    if not perdim_summary.empty:
        cols = [col for col in [
            "branch",
            "run_id",
            "surface",
            "w1_mean_recomputed",
            "ks_mean_recomputed",
            "top1_feature",
            "top3_feature",
            "top1_share_w1",
            "top3_share_w1",
            "n_dims_50pct_w1",
            "n_dims_80pct_w1",
        ] if col in perdim_summary.columns]
        lines.append(
            _df_to_md(
                perdim_summary[cols]
                .sort_values(["branch", "surface", "w1_mean_recomputed"], kind="mergesort")
                .head(16)
            )
        )
        lines.append("")
        lines.append("### Recurrent Offending Features")
        if not perdim_freq.empty:
            lines.append(_df_to_md(perdim_freq.head(20)))
    else:
        lines.append("Per-dimension audit was skipped or produced no rows.")
    lines.append("")
    lines.append("## Reading Guide")
    lines.append("- `run_metrics_wide.csv`: main cross-branch table with branch, base, policy axes, raw realism, trainaligned realism, temporal deltas, and monitor-gap features.")
    lines.append("- `delta_vs_base_long.csv`: direct run-vs-base deltas to see whether finetune actually improves or degrades each base.")
    lines.append("- `policy_summary.csv`: family view by branch/policy signature.")
    lines.append("- `base_policy_summary.csv`: interaction view `base x policy`.")
    lines.append("- `policy_axis_correlations.csv` and `policy_flag_effects.csv`: crude sensitivity map for the explored policy space.")
    lines.append("- `perdim_*`: where the pain is concentrated by feature, instead of relying only on aggregate W1/KS.")
    out_path = analysis_dir / "summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    args = _parse_args()
    analysis_id = _utc_stamp()
    analysis_dir = REPORT_ROOT / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs()
    if args.limit_runs is not None:
        runs = runs[: int(args.limit_runs)]
    if not runs:
        raise RuntimeError("No FlowGen runs discovered for analysis.")

    inventory_rows = []
    run_rows: list[dict[str, Any]] = []
    perclass_rows: list[dict[str, Any]] = []
    for record in runs:
        inventory_rows.append(
            {
                "branch": record.branch,
                "lineage": record.lineage,
                "run_id": record.run_id,
                "run_dir": str(record.run_dir),
                "manifest_path": str(record.manifest_path),
                "config_path": str(record.config_path),
                "results_path": str(record.results_path),
                "metrics_long_path": None if record.metrics_long_path is None else str(record.metrics_long_path),
                "checkpoint_path": None if record.checkpoint_path is None else str(record.checkpoint_path),
                "monitoring_policy": _monitoring_policy(record.manifest, record.config),
            }
        )
        row, pc_rows = _build_run_row(record)
        run_rows.append(row)
        perclass_rows.extend(pc_rows)

    inventory_df = pd.DataFrame(inventory_rows).sort_values(["branch", "lineage", "run_id"], kind="mergesort")
    runs_df = pd.DataFrame(run_rows).sort_values(["branch", "lineage", "run_id"], kind="mergesort")
    perclass_df = pd.DataFrame(perclass_rows)
    delta_long = _build_delta_long(runs_df)
    policy_summary, base_policy_summary = _build_policy_summaries(runs_df, delta_long)
    base_summary = _build_base_summary(runs_df, delta_long)
    reseed_summary = _build_reseed_family_summary(runs_df)
    axis_coverage = _build_policy_axis_coverage(runs_df)
    nearest_neighbors, diversity_summary = _nearest_policy_neighbors(runs_df)
    axis_corr, flag_effects = _policy_axis_correlations(runs_df, delta_long)

    perdim_long = pd.DataFrame()
    perdim_summary = pd.DataFrame()
    perdim_freq = pd.DataFrame()
    if not args.summary_only and not args.skip_perdim_audit:
        perdim_long, perdim_summary, perdim_freq = _build_perdim_outputs(
            runs,
            device_name=str(args.device),
            condition_col=str(args.condition_col),
        )

    inventory_path = analysis_dir / "run_inventory.csv"
    runs_wide_path = analysis_dir / "run_metrics_wide.csv"
    perclass_path = analysis_dir / "perclass_realism_long.csv"
    delta_path = analysis_dir / "delta_vs_base_long.csv"
    base_summary_path = analysis_dir / "base_summary.csv"
    policy_summary_path = analysis_dir / "policy_summary.csv"
    base_policy_path = analysis_dir / "base_policy_summary.csv"
    reseed_summary_path = analysis_dir / "official_reseed_family_summary.csv"
    axis_coverage_path = analysis_dir / "policy_axis_coverage.csv"
    nearest_neighbors_path = analysis_dir / "policy_nearest_neighbors.csv"
    diversity_path = analysis_dir / "policy_diversity_summary.csv"
    axis_corr_path = analysis_dir / "policy_axis_correlations.csv"
    flag_effects_path = analysis_dir / "policy_flag_effects.csv"
    perdim_long_path = analysis_dir / "perdim_feature_realism_long.csv"
    perdim_summary_path = analysis_dir / "perdim_run_summary.csv"
    perdim_freq_path = analysis_dir / "perdim_feature_frequency.csv"

    inventory_df.to_csv(inventory_path, index=False)
    runs_df.to_csv(runs_wide_path, index=False)
    perclass_df.to_csv(perclass_path, index=False)
    delta_long.to_csv(delta_path, index=False)
    base_summary.to_csv(base_summary_path, index=False)
    policy_summary.to_csv(policy_summary_path, index=False)
    base_policy_summary.to_csv(base_policy_path, index=False)
    reseed_summary.to_csv(reseed_summary_path, index=False)
    axis_coverage.to_csv(axis_coverage_path, index=False)
    nearest_neighbors.to_csv(nearest_neighbors_path, index=False)
    diversity_summary.to_csv(diversity_path, index=False)
    axis_corr.to_csv(axis_corr_path, index=False)
    flag_effects.to_csv(flag_effects_path, index=False)
    if not perdim_long.empty:
        perdim_long.to_csv(perdim_long_path, index=False)
    if not perdim_summary.empty:
        perdim_summary.to_csv(perdim_summary_path, index=False)
    if not perdim_freq.empty:
        perdim_freq.to_csv(perdim_freq_path, index=False)

    summary_path = _build_summary_markdown(
        analysis_dir=analysis_dir,
        inventory_df=inventory_df,
        runs_df=runs_df,
        base_summary=base_summary,
        policy_summary=policy_summary,
        policy_diversity=diversity_summary,
        reseed_summary=reseed_summary,
        perdim_summary=perdim_summary,
        perdim_freq=perdim_freq,
    )
    manifest = {
        "analysis_id": analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(args.device),
        "condition_col": str(args.condition_col),
        "summary_only": bool(args.summary_only),
        "skip_perdim_audit": bool(args.skip_perdim_audit),
        "n_runs": int(len(runs_df)),
        "n_official": int((runs_df["branch"] == "official").sum()),
        "n_train_only": int((runs_df["branch"] == "train_only").sum()),
        "outputs": {
            "inventory": str(inventory_path),
            "runs_wide": str(runs_wide_path),
            "perclass_long": str(perclass_path),
            "delta_long": str(delta_path),
            "base_summary": str(base_summary_path),
            "policy_summary": str(policy_summary_path),
            "base_policy_summary": str(base_policy_path),
            "reseed_family_summary": str(reseed_summary_path),
            "axis_coverage": str(axis_coverage_path),
            "nearest_neighbors": str(nearest_neighbors_path),
            "diversity_summary": str(diversity_path),
            "axis_correlations": str(axis_corr_path),
            "flag_effects": str(flag_effects_path),
            "summary_md": str(summary_path),
            "perdim_long": str(perdim_long_path) if not perdim_long.empty else None,
            "perdim_summary": str(perdim_summary_path) if not perdim_summary.empty else None,
            "perdim_frequency": str(perdim_freq_path) if not perdim_freq.empty else None,
        },
    }
    write_json(analysis_dir / "analysis_manifest.json", manifest)

    _print(f"Analysis written to {analysis_dir}")
    _print(f"Runs discovered: total={len(runs_df)} official={int((runs_df['branch'] == 'official').sum())} train_only={int((runs_df['branch'] == 'train_only').sum())}")
    if args.summary_only:
        _print("summary-only mode: skipped per-dimension audit")
    elif args.skip_perdim_audit:
        _print("per-dimension audit skipped by flag")
    else:
        _print(f"per-dimension audit rows: {len(perdim_long)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
