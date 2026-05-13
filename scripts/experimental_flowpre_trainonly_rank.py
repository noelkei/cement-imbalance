from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / "outputs" / "models" / "experimental" / "train_only" / "flow_pre"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "reports" / "experimental" / "train_only" / "flowpre_rankings"

SCORE_WEIGHTS = {
    "per_class_prior_fit": 0.32,
    "global_prior_fit": 0.28,
    "artifact_robustness": 0.25,
    "overfit_risk": 0.15,
}

GLOBAL_BLOCK_METRICS = {
    "train_sum_rrmse": 0.35,
    "train_mvn_score": 0.35,
    "train_rrmse_recon": 0.15,
    "train_eigstd": 0.10,
    "train_r2_recon_error": 0.05,
}

PER_CLASS_BLOCK_METRICS = {
    "pc_worst_sum": 0.35,
    "pc_macro_sum": 0.20,
    "minority_pc_sum": 0.20,
    "pc_dispersion": 0.10,
    "pc_worst_mvn_score": 0.15,
}

ROBUSTNESS_BLOCK_METRICS = {
    "worst_to_macro_ratio": 0.20,
    "minority_to_macro_ratio": 0.15,
    "pc_eigstd_worst": 0.20,
    "pc_kurt_excess_worst": 0.15,
    "influence_top1_share_mean": 0.10,
    "influence_top3_share_mean": 0.10,
    "influence_entropy_inv_mean": 0.10,
}

OVERFIT_BLOCK_METRICS = {
    "best_epoch": 0.20,
    "best_epoch_ratio": 0.10,
    "tail_monitor_improvement": 0.20,
    "logabsdet_abs_final": 0.15,
    "logabsdet_range": 0.15,
    "capacity_proxy": 0.10,
    "influence_top1_share_max": 0.10,
}

RECON_REJECT_THRESHOLD = 0.15
RECON_CAUTION_THRESHOLD = 0.05
WORST_CLASS_SEVERE_RZ = 2.5
WORST_CLASS_CAUTION_RZ = 1.5
OVERFIT_HIGH_RZ = 1.5
OVERFIT_MEDIUM_RZ = 0.5
ROBUST_Z_CLIP = 5.0
EPS = 1e-12
SOFT_RISK_FLAGS = {"late_best_epoch", "tail_still_improving"}


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_div(num: float | None, den: float | None, default: float | None = None) -> float | None:
    if num is None or den is None or abs(den) <= EPS:
        return default
    out = num / den
    return out if math.isfinite(out) else default


def _get(mapping: dict[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return loaded


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping JSON at {path}")
    return loaded


def _normalize_class_mapping(raw: Any) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        cls = _as_int(key)
        if cls is None or not isinstance(value, dict):
            continue
        out[cls] = value
    return out


def _finite_values(values: list[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _robust_z_values(values: list[float | None]) -> list[float | None]:
    finite = _finite_values(values)
    if not finite:
        return [None for _ in values]
    med = median(finite)
    deviations = [abs(v - med) for v in finite]
    mad = median(deviations)
    if mad <= EPS:
        return [0.0 if v is not None else None for v in values]
    out: list[float | None] = []
    for value in values:
        if value is None or not math.isfinite(float(value)):
            out.append(None)
        else:
            z = 0.6745 * (float(value) - med) / mad
            out.append(max(-ROBUST_Z_CLIP, min(ROBUST_Z_CLIP, z)))
    return out


def _weighted_average(row: dict[str, Any], metrics: dict[str, float], suffix: str = "_rz") -> float | None:
    total_weight = 0.0
    total = 0.0
    for metric, weight in metrics.items():
        value = _as_float(row.get(f"{metric}{suffix}"))
        if value is None:
            continue
        total += weight * value
        total_weight += weight
    if total_weight <= EPS:
        return None
    return total / total_weight


def _mvn_score(stats: dict[str, Any], latent_dim: int) -> float | None:
    skew_abs = abs(_as_float(stats.get("skewness_mean"), 0.0) or 0.0)
    kurt = _as_float(stats.get("kurtosis_mean"))
    eigstd = _as_float(stats.get("eigval_std"))
    mahal_mu = _as_float(stats.get("mahalanobis_mean"))
    mahal_md = _as_float(stats.get("mahalanobis_median"))
    if kurt is None or eigstd is None or mahal_mu is None or mahal_md is None:
        return None
    target_mu = math.sqrt(max(float(latent_dim) - 0.5, 1.0))
    return (
        1.0 * skew_abs
        + 1.0 * abs(kurt - 3.0)
        + 1.2 * eigstd
        + 0.3 * abs(mahal_mu - target_mu)
        + 0.3 * abs(mahal_md - target_mu)
    ) / 3.8


def _discover_runs(runs_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for results_path in sorted(runs_root.rglob("*_results.yaml")):
        if "configs" in results_path.parts:
            continue
        run_dir = results_path.parent
        run_id = results_path.name[: -len("_results.yaml")]
        records.append(
            {
                "run_id": run_id,
                "run_dir": run_dir,
                "results_path": results_path,
                "manifest_path": run_dir / f"{run_id}_run_manifest.json",
                "metrics_long_path": run_dir / f"{run_id}_metrics_long.csv",
                "influence_path": run_dir / f"{run_id}_influence.json",
                "log_path": run_dir / f"{run_id}.log",
                "config_path": run_dir / f"{run_id}.yaml",
                "model_path": run_dir / f"{run_id}.pt",
                "snapshot_count": len(list((run_dir / "snapshots").glob("*.pt"))) if (run_dir / "snapshots").exists() else 0,
            }
        )
    return records


def _parse_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "log_parse_status": "missing",
            "monitor_loss_count": 0,
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    epoch_re = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+.+?Train Loss:\s+([\-0-9.eE]+)\s+\|\s+Train-monitor Loss:\s+([\-0-9.eE]+)"
    )
    train_rrmse_re = re.compile(r"RRMSE \(Train\): mean=([\-0-9.eE]+), std=([\-0-9.eE]+)")
    monitor_rrmse_re = re.compile(r"RRMSE \(Train-monitor\):\s+mean=([\-0-9.eE]+), std=([\-0-9.eE]+)")
    logdet_re = re.compile(r"logabsdet_mean:\s*([\-0-9.eE]+)")
    best_re = re.compile(r"Best monitor loss:\s*([\-0-9.eE]+)\s*\(epoch\s+(\d+)\)")
    early_re = re.compile(r"Early stopping at epoch\s+(\d+)")

    epoch_rows: list[tuple[int, int, float, float]] = []
    for match in epoch_re.finditer(text):
        epoch = _as_int(match.group(1))
        max_epoch = _as_int(match.group(2))
        train_loss = _as_float(match.group(3))
        monitor_loss = _as_float(match.group(4))
        if epoch is not None and max_epoch is not None and train_loss is not None and monitor_loss is not None:
            epoch_rows.append((epoch, max_epoch, train_loss, monitor_loss))

    train_rrmse_rows = [(_as_float(a), _as_float(b)) for a, b in train_rrmse_re.findall(text)]
    monitor_rrmse_rows = [(_as_float(a), _as_float(b)) for a, b in monitor_rrmse_re.findall(text)]
    logdets = [_as_float(match.group(1)) for match in logdet_re.finditer(text)]
    logdets_f = _finite_values(logdets)

    explicit_best_loss = None
    explicit_best_epoch = None
    best_match = best_re.search(text)
    if best_match:
        explicit_best_loss = _as_float(best_match.group(1))
        explicit_best_epoch = _as_int(best_match.group(2))

    early_epoch = None
    early_match = early_re.search(text)
    if early_match:
        early_epoch = _as_int(early_match.group(1))

    monitor_losses = [(epoch, loss) for epoch, _, _, loss in epoch_rows]
    final_monitor_loss = monitor_losses[-1][1] if monitor_losses else None
    final_epoch = monitor_losses[-1][0] if monitor_losses else None
    if explicit_best_loss is not None and explicit_best_epoch is not None:
        best_monitor_loss = explicit_best_loss
        best_monitor_epoch = explicit_best_epoch
    elif monitor_losses:
        best_monitor_epoch, best_monitor_loss = min(monitor_losses, key=lambda item: item[1])
    else:
        best_monitor_loss = None
        best_monitor_epoch = None

    tail_monitor_improvement = None
    tail_monitor_slope = None
    if len(monitor_losses) >= 5:
        tail_n = max(5, int(math.ceil(len(monitor_losses) * 0.2)))
        tail = monitor_losses[-tail_n:]
        first_loss = tail[0][1]
        last_loss = tail[-1][1]
        tail_monitor_improvement = first_loss - last_loss
        tail_monitor_slope = (last_loss - first_loss) / max(len(tail) - 1, 1)

    out: dict[str, Any] = {
        "log_parse_status": "parsed",
        "monitor_loss_count": len(monitor_losses),
        "final_epoch_from_log": final_epoch,
        "best_monitor_epoch_from_log": best_monitor_epoch,
        "best_monitor_loss": best_monitor_loss,
        "final_monitor_loss": final_monitor_loss,
        "final_minus_best_monitor_loss": (
            final_monitor_loss - best_monitor_loss
            if final_monitor_loss is not None and best_monitor_loss is not None
            else None
        ),
        "tail_monitor_improvement": tail_monitor_improvement,
        "tail_monitor_slope": tail_monitor_slope,
        "early_stopping_epoch": early_epoch,
        "logabsdet_mean_min": min(logdets_f) if logdets_f else None,
        "logabsdet_mean_max": max(logdets_f) if logdets_f else None,
        "logabsdet_mean_final": logdets_f[-1] if logdets_f else None,
        "logabsdet_range": (max(logdets_f) - min(logdets_f)) if logdets_f else None,
        "logabsdet_abs_final": abs(logdets_f[-1]) if logdets_f else None,
        "train_rrmse_tail_mean": train_rrmse_rows[-1][0] if train_rrmse_rows else None,
        "train_rrmse_tail_std": train_rrmse_rows[-1][1] if train_rrmse_rows else None,
        "monitor_rrmse_tail_mean": monitor_rrmse_rows[-1][0] if monitor_rrmse_rows else None,
        "monitor_rrmse_tail_std": monitor_rrmse_rows[-1][1] if monitor_rrmse_rows else None,
    }
    return out


def _summarize_influence(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"influence_status": "missing", "latent_dim": None}
    data = _load_json(path)
    top1: list[float] = []
    top3: list[float] = []
    entropies: list[float] = []
    for value in data.values():
        if not isinstance(value, dict):
            continue
        shares: list[float] = []
        for pair in value.values():
            if isinstance(pair, list) and len(pair) >= 2:
                share = _as_float(pair[1])
                if share is not None and share >= 0:
                    shares.append(share)
        total = sum(shares)
        if total <= EPS:
            continue
        probs = [share / total for share in shares if share > 0]
        probs_sorted = sorted(probs, reverse=True)
        top1.append(probs_sorted[0])
        top3.append(sum(probs_sorted[:3]))
        if len(probs) > 1:
            entropy = -sum(p * math.log(p) for p in probs) / math.log(len(probs))
        else:
            entropy = 0.0
        entropies.append(entropy)

    return {
        "influence_status": "parsed",
        "latent_dim": len(data),
        "influence_top1_share_mean": sum(top1) / len(top1) if top1 else None,
        "influence_top1_share_max": max(top1) if top1 else None,
        "influence_top3_share_mean": sum(top3) / len(top3) if top3 else None,
        "influence_top3_share_max": max(top3) if top3 else None,
        "influence_entropy_norm_mean": sum(entropies) / len(entropies) if entropies else None,
        "influence_entropy_norm_min": min(entropies) if entropies else None,
        "influence_entropy_inv_mean": (1.0 - (sum(entropies) / len(entropies))) if entropies else None,
    }


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _load_yaml(path)
    except Exception:
        return {}


def _capacity_proxy(config: dict[str, Any]) -> float | None:
    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    hidden = _as_float(model.get("hidden_features"))
    layers = _as_float(model.get("num_layers"))
    initial = _as_float(model.get("initial_affine_layers"), 0.0)
    repeat_blocks = _as_float(model.get("n_repeat_blocks"), 0.0)
    final_rq = _as_float(model.get("final_rq_layers"), 0.0)
    ratio = model.get("affine_rq_ratio", [1, 3])
    ratio_sum = None
    if isinstance(ratio, list) and len(ratio) >= 2:
        left = _as_float(ratio[0])
        right = _as_float(ratio[1])
        if left is not None and right is not None:
            ratio_sum = left + right
    if hidden is None or layers is None or ratio_sum is None:
        return None
    transform_count = (initial or 0.0) + (repeat_blocks or 0.0) * ratio_sum + (final_rq or 0.0)
    return hidden * max(layers, 1.0) * max(transform_count, 1.0)


def _extract_row(discovered: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    run_id = str(discovered["run_id"])
    results_path = Path(discovered["results_path"])
    result = _load_yaml(results_path)
    manifest = _load_json(discovered["manifest_path"]) if Path(discovered["manifest_path"]).exists() else {}
    config = _read_config(discovered["config_path"])
    influence = _summarize_influence(Path(discovered["influence_path"]))
    log_summary = _parse_log(Path(discovered["log_path"]))

    train = result.get("train", {}) if isinstance(result.get("train"), dict) else {}
    val = result.get("val", {}) if isinstance(result.get("val"), dict) else {}
    monitoring = result.get("monitoring", {}) if isinstance(result.get("monitoring"), dict) else {}
    run_axes = manifest.get("run_level_axes", {}) if isinstance(manifest.get("run_level_axes"), dict) else {}

    latent_dim = _as_int(influence.get("latent_dim"), 43) or 43
    train_iso = train.get("isotropy_stats", {}) if isinstance(train.get("isotropy_stats"), dict) else {}
    val_iso = val.get("isotropy_stats", {}) if isinstance(val.get("isotropy_stats"), dict) else {}
    pc = _normalize_class_mapping(train.get("per_class_iso_rrmse"))
    pc_stats = _normalize_class_mapping(train.get("isotropy_stats_per_class"))

    pc_rows: dict[int, dict[str, float | int | None]] = {}
    for cls, metrics in pc.items():
        mean = _as_float(metrics.get("rrmse_mean"))
        std = _as_float(metrics.get("rrmse_std"))
        n = _as_int(metrics.get("n"))
        cls_stats = pc_stats.get(cls, {})
        cls_mvn = _mvn_score(cls_stats, latent_dim) if cls_stats else None
        pc_rows[cls] = {
            "rrmse_mean": mean,
            "rrmse_std": std,
            "rrmse_sum": (mean + std) if mean is not None and std is not None else None,
            "n": n,
            "mvn_score": cls_mvn,
            "eigstd": _as_float(cls_stats.get("eigval_std")),
            "skew_abs": abs(_as_float(cls_stats.get("skewness_mean"), 0.0) or 0.0),
            "kurt_excess_abs": abs((_as_float(cls_stats.get("kurtosis_mean")) or 3.0) - 3.0),
        }

    pc_sums = {cls: _as_float(row.get("rrmse_sum")) for cls, row in pc_rows.items()}
    finite_pc_sums = {cls: value for cls, value in pc_sums.items() if value is not None}
    worst_class_id = max(finite_pc_sums, key=finite_pc_sums.get) if finite_pc_sums else None
    minority_class_id = None
    finite_ns = {cls: _as_int(row.get("n")) for cls, row in pc_rows.items() if _as_int(row.get("n")) is not None}
    if finite_ns:
        minority_class_id = min(finite_ns, key=finite_ns.get)

    pc_sum_values = list(finite_pc_sums.values())
    pc_macro_sum = sum(pc_sum_values) / len(pc_sum_values) if pc_sum_values else None
    pc_weighted_sum = None
    if finite_ns and pc_sum_values:
        weighted_total = 0.0
        n_total = 0
        for cls, value in finite_pc_sums.items():
            n = finite_ns.get(cls)
            if n is None:
                continue
            weighted_total += value * n
            n_total += n
        pc_weighted_sum = weighted_total / n_total if n_total else None

    train_rrmse_mean = _as_float(train.get("rrmse_mean_whole"))
    train_rrmse_std = _as_float(train.get("rrmse_std_whole"))
    val_rrmse_mean = _as_float(val.get("rrmse_mean_whole"))
    val_rrmse_std = _as_float(val.get("rrmse_std_whole"))
    train_sum = train_rrmse_mean + train_rrmse_std if train_rrmse_mean is not None and train_rrmse_std is not None else None
    val_sum = val_rrmse_mean + val_rrmse_std if val_rrmse_mean is not None and val_rrmse_std is not None else None

    best_epoch = _as_int(result.get("best_epoch"))
    total_epochs = _as_int(result.get("total_epochs"))
    row: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(discovered["run_dir"]),
        "results_path": str(discovered["results_path"]),
        "manifest_path": str(discovered["manifest_path"]),
        "metrics_long_path": str(discovered["metrics_long_path"]),
        "influence_path": str(discovered["influence_path"]),
        "log_path": str(discovered["log_path"]),
        "config_path": str(discovered["config_path"]),
        "model_path": str(discovered["model_path"]),
        "snapshot_count": discovered.get("snapshot_count"),
        "has_results": Path(discovered["results_path"]).exists(),
        "has_manifest": Path(discovered["manifest_path"]).exists(),
        "has_metrics_long": Path(discovered["metrics_long_path"]).exists(),
        "has_influence": Path(discovered["influence_path"]).exists(),
        "has_log": Path(discovered["log_path"]).exists(),
        "has_config": Path(discovered["config_path"]).exists(),
        "has_model": Path(discovered["model_path"]).exists(),
        "contract_id": manifest.get("contract_id"),
        "comparison_group_id": manifest.get("comparison_group_id"),
        "seed_set_id": manifest.get("seed_set_id"),
        "objective_metric_id": manifest.get("objective_metric_id"),
        "dataset_name": manifest.get("dataset_name"),
        "split_id": manifest.get("split_id"),
        "seed": result.get("seed", manifest.get("seed")),
        "test_enabled": manifest.get("test_enabled"),
        "monitoring_policy": monitoring.get("policy"),
        "monitor_source_split": monitoring.get("monitor_source_split"),
        "monitor_role": monitoring.get("monitor_role"),
        "monitor_is_holdout": monitoring.get("monitor_is_holdout"),
        "canonical_selection_eligible": monitoring.get("canonical_selection_eligible"),
        "cfg_signature": run_axes.get("cfg_signature"),
        "phase": run_axes.get("phase"),
        "source_view": run_axes.get("source_view"),
        "source_run_id": run_axes.get("source_run_id"),
        "source_rank": run_axes.get("source_rank"),
        "shortlist_rank": run_axes.get("shortlist_rank"),
        "source_seed": run_axes.get("source_seed"),
        "best_epoch": best_epoch,
        "total_epochs": total_epochs,
        "best_epoch_ratio": _safe_div(float(best_epoch) if best_epoch is not None else None, float(total_epochs) if total_epochs is not None else None),
        "train_n": train.get("n"),
        "val_n": val.get("n"),
        "train_rrmse_recon": _as_float(train.get("rrmse_recon")),
        "train_r2_recon": _as_float(train.get("r2_recon")),
        "train_r2_recon_error": 1.0 - (_as_float(train.get("r2_recon")) or 0.0),
        "val_rrmse_recon": _as_float(val.get("rrmse_recon")),
        "train_rrmse_mean": train_rrmse_mean,
        "train_rrmse_std": train_rrmse_std,
        "train_sum_rrmse": train_sum,
        "val_rrmse_mean": val_rrmse_mean,
        "val_rrmse_std": val_rrmse_std,
        "val_sum_rrmse": val_sum,
        "train_skew_abs": abs(_as_float(train_iso.get("skewness_mean"), 0.0) or 0.0),
        "train_kurt_excess_abs": abs((_as_float(train_iso.get("kurtosis_mean")) or 3.0) - 3.0),
        "train_eigstd": _as_float(train_iso.get("eigval_std")),
        "train_mahal_mu": _as_float(train_iso.get("mahalanobis_mean")),
        "train_mahal_md": _as_float(train_iso.get("mahalanobis_median")),
        "train_mvn_score": _mvn_score(train_iso, latent_dim),
        "val_mvn_score": _mvn_score(val_iso, latent_dim),
        "latent_dim": latent_dim,
        "class_ids": ",".join(str(cls) for cls in sorted(pc_rows)),
        "class_count": len(pc_rows),
        "minority_class_id": minority_class_id,
        "worst_class_id": worst_class_id,
        "pc_worst_sum": finite_pc_sums.get(worst_class_id) if worst_class_id is not None else None,
        "pc_macro_sum": pc_macro_sum,
        "pc_weighted_sum": pc_weighted_sum,
        "minority_pc_sum": finite_pc_sums.get(minority_class_id) if minority_class_id is not None else None,
        "pc_dispersion": (max(pc_sum_values) - min(pc_sum_values)) if pc_sum_values else None,
        "worst_to_macro_ratio": _safe_div(finite_pc_sums.get(worst_class_id) if worst_class_id is not None else None, pc_macro_sum),
        "minority_to_macro_ratio": _safe_div(finite_pc_sums.get(minority_class_id) if minority_class_id is not None else None, pc_macro_sum),
        "pc_worst_mvn_score": max(_finite_values([_as_float(row.get("mvn_score")) for row in pc_rows.values()]), default=None),
        "pc_macro_mvn_score": (
            sum(_finite_values([_as_float(row.get("mvn_score")) for row in pc_rows.values()]))
            / len(_finite_values([_as_float(row.get("mvn_score")) for row in pc_rows.values()]))
            if _finite_values([_as_float(row.get("mvn_score")) for row in pc_rows.values()])
            else None
        ),
        "pc_eigstd_worst": max(_finite_values([_as_float(row.get("eigstd")) for row in pc_rows.values()]), default=None),
        "pc_skew_abs_worst": max(_finite_values([_as_float(row.get("skew_abs")) for row in pc_rows.values()]), default=None),
        "pc_kurt_excess_worst": max(_finite_values([_as_float(row.get("kurt_excess_abs")) for row in pc_rows.values()]), default=None),
        "capacity_proxy": _capacity_proxy(config),
        "semantic_status": "not_computed_v1_artifact_only",
    }
    row.update(log_summary)
    row.update(influence)
    inventory = {
        key: row.get(key)
        for key in [
            "run_id",
            "run_dir",
            "has_results",
            "has_manifest",
            "has_metrics_long",
            "has_influence",
            "has_log",
            "has_config",
            "has_model",
            "snapshot_count",
            "contract_id",
            "phase",
            "cfg_signature",
            "source_view",
            "source_run_id",
            "source_rank",
            "seed",
        ]
    }
    return row, inventory


def _add_gate_reasons(rows: list[dict[str, Any]]) -> None:
    expected_classes = set()
    for row in rows:
        class_ids = str(row.get("class_ids") or "")
        for item in class_ids.split(","):
            cls = _as_int(item)
            if cls is not None:
                expected_classes.add(cls)

    for row in rows:
        reject: list[str] = []
        caution: list[str] = []
        existing_reject = str(row.get("reject_reasons") or "")
        if existing_reject and existing_reject != "none":
            reject.extend(item for item in existing_reject.split("; ") if item)
        if not row.get("has_results"):
            reject.append("missing_results")
        if not row.get("has_manifest"):
            reject.append("missing_manifest")
        if not row.get("has_metrics_long"):
            reject.append("missing_metrics_long")
        if not row.get("has_config"):
            reject.append("missing_config")
        if not row.get("has_log"):
            caution.append("missing_log")
        if not row.get("has_influence"):
            caution.append("missing_influence")
        if row.get("monitoring_policy") != "train_only":
            reject.append("not_train_only_policy")
        if row.get("monitor_source_split") != "train":
            reject.append("monitor_not_train_sourced")
        if bool(row.get("monitor_is_holdout")):
            reject.append("monitor_is_holdout")
        if bool(row.get("test_enabled")):
            reject.append("test_enabled")
        if row.get("semantic_status") != "not_computed_v1_artifact_only":
            caution.append("unexpected_semantic_status")

        required_metrics = [
            "train_rrmse_recon",
            "train_sum_rrmse",
            "train_mvn_score",
            "pc_worst_sum",
            "pc_macro_sum",
            "minority_pc_sum",
        ]
        for metric in required_metrics:
            if _as_float(row.get(metric)) is None:
                reject.append(f"missing_metric:{metric}")
        train_recon = _as_float(row.get("train_rrmse_recon"))
        if train_recon is not None:
            if train_recon > RECON_REJECT_THRESHOLD:
                reject.append(f"train_rrmse_recon>{RECON_REJECT_THRESHOLD}")
            elif train_recon > RECON_CAUTION_THRESHOLD:
                caution.append(f"train_rrmse_recon>{RECON_CAUTION_THRESHOLD}")

        observed_classes = set()
        for item in str(row.get("class_ids") or "").split(","):
            cls = _as_int(item)
            if cls is not None:
                observed_classes.add(cls)
        missing_classes = sorted(expected_classes - observed_classes)
        if missing_classes:
            reject.append("missing_classes:" + ",".join(str(cls) for cls in missing_classes))

        row["reject_reasons"] = "; ".join(reject) if reject else "none"
        row["caution_reasons"] = "; ".join(caution) if caution else "none"
        row["gate_status"] = "reject" if reject else ("caution" if caution else "pass")


def _add_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks = {
        "global_prior_fit": GLOBAL_BLOCK_METRICS,
        "per_class_prior_fit": PER_CLASS_BLOCK_METRICS,
        "artifact_robustness": ROBUSTNESS_BLOCK_METRICS,
        "overfit_risk": OVERFIT_BLOCK_METRICS,
    }
    all_score_metrics = sorted({metric for metrics in blocks.values() for metric in metrics})
    for metric in all_score_metrics:
        z_values = _robust_z_values([_as_float(row.get(metric)) for row in rows])
        for row, z in zip(rows, z_values):
            row[f"{metric}_rz"] = z

    blocks_long: list[dict[str, Any]] = []
    source_by_metric = {
        **{metric: "results.yaml" for metric in GLOBAL_BLOCK_METRICS},
        **{metric: "results.yaml" for metric in PER_CLASS_BLOCK_METRICS},
        **{metric: "results.yaml/influence.json" for metric in ROBUSTNESS_BLOCK_METRICS},
        **{metric: "results.yaml/log/influence.json/config" for metric in OVERFIT_BLOCK_METRICS},
    }
    for row in rows:
        for block_name, metrics in blocks.items():
            for metric, weight in metrics.items():
                rz = _as_float(row.get(f"{metric}_rz"))
                contribution = rz * weight if rz is not None else None
                blocks_long.append(
                    {
                        "run_id": row.get("run_id"),
                        "block": block_name,
                        "metric": metric,
                        "raw_value": row.get(metric),
                        "robust_z": rz,
                        "weight_within_block": weight,
                        "contribution": contribution,
                        "source": source_by_metric.get(metric, "artifact"),
                    }
                )

        row["global_prior_fit_score"] = _weighted_average(row, GLOBAL_BLOCK_METRICS)
        row["per_class_prior_fit_score"] = _weighted_average(row, PER_CLASS_BLOCK_METRICS)
        row["artifact_robustness_score"] = _weighted_average(row, ROBUSTNESS_BLOCK_METRICS)
        row["overfit_risk_score"] = _weighted_average(row, OVERFIT_BLOCK_METRICS)

        final = 0.0
        total_weight = 0.0
        for block_name, weight in SCORE_WEIGHTS.items():
            value = _as_float(row.get(f"{block_name}_score"))
            if value is None:
                continue
            final += weight * value
            total_weight += weight
        row["final_score"] = final / total_weight if total_weight > EPS else None

    for row in rows:
        flags: list[str] = []
        severe_flags: list[str] = []
        pc_worst_rz = _as_float(row.get("pc_worst_sum_rz"))
        if pc_worst_rz is not None:
            if pc_worst_rz >= WORST_CLASS_SEVERE_RZ:
                severe_flags.append("worst_class_severe")
            elif pc_worst_rz >= WORST_CLASS_CAUTION_RZ:
                flags.append("worst_class_caution")

        overfit_score = _as_float(row.get("overfit_risk_score"))
        if overfit_score is not None:
            if overfit_score >= OVERFIT_HIGH_RZ:
                severe_flags.append("overfit_risk_high")
            elif overfit_score >= OVERFIT_MEDIUM_RZ:
                flags.append("overfit_risk_medium")

        best_epoch = _as_float(row.get("best_epoch"))
        best_ratio = _as_float(row.get("best_epoch_ratio"))
        if best_epoch is not None and best_ratio is not None and best_epoch >= 180 and best_ratio >= 0.90:
            flags.append("late_best_epoch")
        tail_improvement = _as_float(row.get("tail_monitor_improvement"))
        if tail_improvement is not None and tail_improvement > 1.0 and best_ratio is not None and best_ratio >= 0.85:
            flags.append("tail_still_improving")
        if (_as_float(row.get("logabsdet_abs_final_rz")) or 0.0) >= 2.0:
            flags.append("logdet_abs_high")
        if (_as_float(row.get("logabsdet_range_rz")) or 0.0) >= 2.0:
            flags.append("logdet_range_high")
        if (_as_float(row.get("influence_top1_share_max_rz")) or 0.0) >= 2.0:
            flags.append("influence_concentrated")
        if (_as_float(row.get("global_prior_fit_score")) or 0.0) <= -1.5 and (
            "late_best_epoch" in flags or "tail_still_improving" in flags
        ):
            flags.append("internal_beauty_outlier")

        row["risk_flags"] = "; ".join(flags) if flags else "none"
        row["severe_flags"] = "; ".join(severe_flags) if severe_flags else "none"
        row["overfit_risk_level"] = "high" if "overfit_risk_high" in severe_flags else ("medium" if "overfit_risk_medium" in flags else "low")
        if severe_flags:
            existing = [] if row.get("caution_reasons") == "none" else str(row.get("caution_reasons")).split("; ")
            row["caution_reasons"] = "; ".join([*existing, *severe_flags])
            if row.get("gate_status") == "pass":
                row["gate_status"] = "caution"

    non_reject = [row for row in rows if row.get("gate_status") != "reject" and _as_float(row.get("final_score")) is not None]
    non_reject_sorted = sorted(
        non_reject,
        key=lambda row: (
            _as_float(row.get("final_score"), 999.0),
            str(row.get("cfg_signature") or ""),
            _as_float(row.get("source_rank"), 999.0),
            str(row.get("run_id")),
        ),
    )
    eligible_count = len(non_reject_sorted)
    tier_a_limit = max(3, int(math.ceil(eligible_count * 0.15))) if eligible_count else 0
    tier_b_limit = max(8, int(math.ceil(eligible_count * 0.40))) if eligible_count else 0

    for idx, row in enumerate(non_reject_sorted, start=1):
        row["rank"] = idx
        if idx <= tier_a_limit:
            tier = "A"
        elif idx <= tier_b_limit:
            tier = "B"
        else:
            tier = "C"
        if tier == "A" and (row.get("gate_status") == "caution" or row.get("overfit_risk_level") == "high"):
            tier = "B"
        row["tier"] = tier

    reject_rows = [row for row in rows if row.get("gate_status") == "reject" or _as_float(row.get("final_score")) is None]
    for offset, row in enumerate(sorted(reject_rows, key=lambda r: str(r.get("run_id"))), start=len(non_reject_sorted) + 1):
        row["rank"] = offset
        row["tier"] = "Reject"

    for row in rows:
        reason_bits = []
        if row.get("tier") == "Reject":
            reason_bits.append(f"reject={row.get('reject_reasons')}")
        else:
            reason_bits.append(f"pc={_fmt(row.get('per_class_prior_fit_score'))}")
            reason_bits.append(f"global={_fmt(row.get('global_prior_fit_score'))}")
            reason_bits.append(f"robust={_fmt(row.get('artifact_robustness_score'))}")
            reason_bits.append(f"overfit={_fmt(row.get('overfit_risk_score'))}/{row.get('overfit_risk_level')}")
            if row.get("risk_flags") != "none":
                reason_bits.append(f"flags={row.get('risk_flags')}")
        row["ranking_reason"] = "; ".join(reason_bits)

    return blocks_long


def _fmt(value: Any, digits: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str] | None = None) -> None:
    _write_csv_with_options(path, rows, preferred_fields, include_extra=True)


def _write_csv_with_options(
    path: Path,
    rows: list[dict[str, Any]],
    preferred_fields: list[str] | None = None,
    *,
    include_extra: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_set = set()
    for row in rows:
        field_set.update(row.keys())
    fields = list(preferred_fields or [])
    if include_extra:
        fields.extend(sorted(field_set - set(fields)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    selected = rows[:limit] if limit is not None else rows
    if not selected:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in selected:
        cells = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                cells.append(_fmt(value))
            else:
                cells.append(str(value) if value is not None else "NA")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _tier_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out = {"A": 0, "B": 0, "C": 0, "Reject": 0}
    for row in rows:
        tier = str(row.get("tier") or "Reject")
        out[tier] = out.get(tier, 0) + 1
    return out


def _distinct_by_cfg(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        cfg = str(row.get("cfg_signature") or row.get("run_id"))
        if cfg in seen:
            continue
        seen.add(cfg)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _split_flags(value: Any) -> set[str]:
    text = str(value or "")
    if not text or text == "none":
        return set()
    return {item.strip() for item in text.split(";") if item.strip()}


def _has_material_review_flag(row: dict[str, Any]) -> bool:
    if row.get("tier") == "Reject":
        return True
    if row.get("gate_status") == "caution":
        return True
    if row.get("severe_flags") != "none":
        return True
    if row.get("overfit_risk_level") in {"medium", "high"}:
        return True
    risk_flags = _split_flags(row.get("risk_flags"))
    return bool(risk_flags - SOFT_RISK_FLAGS)


def _tier_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: _as_float(row.get("rank"), 9999.0) or 9999.0)
    out: list[dict[str, Any]] = []
    labels = {
        "A": "candidatos fuertes",
        "B": "backups/reseed",
        "C": "mantener como contexto",
        "Reject": "no usar como prior v1",
    }
    for tier in ["A", "B", "C", "Reject"]:
        tier_rows = [row for row in sorted_rows if row.get("tier") == tier]
        best = tier_rows[0] if tier_rows else {}
        out.append(
            {
                "tier": tier,
                "count": len(tier_rows),
                "best_rank": best.get("rank", "NA"),
                "best_run_id": best.get("run_id", "NA"),
                "lectura": labels[tier],
            }
        )
    return out


def _build_recommendation(rows: list[dict[str, Any]]) -> tuple[str, str]:
    sorted_rows = sorted(rows, key=lambda row: _as_float(row.get("rank"), 9999.0) or 9999.0)
    tier_a = [row for row in sorted_rows if row.get("tier") == "A"]
    tier_b = [row for row in sorted_rows if row.get("tier") == "B"]
    high_risk_top = [
        row
        for row in sorted_rows[:5]
        if row.get("overfit_risk_level") == "high" or row.get("severe_flags") != "none"
    ]
    if not tier_a:
        return (
            "seguir_iterando_flowpre",
            "No hay Tier A limpio. Conviene seguir iterando FlowPre o revisar flags antes de usar un prior para FlowGen.",
        )
    if high_risk_top:
        return (
            "hacer_reseed_o_micro_ablation",
            "Hay candidato(s) alto(s) con flags severos. Mejor reseed/micro-ablation antes de fijar prior.",
        )
    tier_a_seeds = {str(row.get("seed")) for row in tier_a if row.get("seed") is not None}
    tier_ab = [row for row in sorted_rows if row.get("tier") in {"A", "B"}]
    tier_ab_seeds = {str(row.get("seed")) for row in tier_ab if row.get("seed") is not None}
    if len(tier_a_seeds) <= 1 or len(tier_ab_seeds) <= 1:
        return (
            "pasar_a_flowgen_smoke_y_reseed_top_priors",
            "Hay Tier A usable, pero la shortlist efectiva viene de una sola seed. Conviene lanzar FlowGen smoke y reseed de los priors top antes de fijar winner.",
        )
    if len(tier_a) == 1 and tier_b:
        return (
            "pasar_a_flowgen_smoke_y_reseed_si_cierra",
            "Hay un Tier A claro. Puede arrancar FlowGen train-only smoke; para cierre, reseed del candidato y backups.",
        )
    return (
        "pasar_a_flowgen_shortlist",
        "Hay varios Tier A sin flags severos. Tiene sentido pasar a FlowGen train-only con shortlist.",
    )


def _write_summary(path: Path, rows: list[dict[str, Any]], inventories: list[dict[str, Any]], shortlist: dict[str, Any]) -> None:
    sorted_rows = sorted(rows, key=lambda row: _as_float(row.get("rank"), 9999.0) or 9999.0)
    counts = _tier_counts(rows)
    gate_counts: dict[str, int] = {}
    for row in rows:
        gate = str(row.get("gate_status") or "unknown")
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
    direct = [row for row in sorted_rows if row.get("tier") == "A"]
    reseed = _distinct_by_cfg([row for row in sorted_rows if row.get("tier") in {"A", "B"}], limit=5)
    doubtful = [row for row in sorted_rows if row.get("tier") != "Reject" and _has_material_review_flag(row)]
    reject = [row for row in sorted_rows if row.get("tier") == "Reject"]
    tier_a_seeds = sorted({str(row.get("seed")) for row in direct if row.get("seed") is not None})
    tier_ab = [row for row in sorted_rows if row.get("tier") in {"A", "B"}]
    tier_ab_cfg_count = len({str(row.get("cfg_signature") or row.get("run_id")) for row in tier_ab})
    top_has_severe = any(row.get("severe_flags") != "none" for row in sorted_rows[:5])

    lines = [
        "# FlowPre Train-Only Ranking v1",
        "",
        "## Estado general",
        f"- Runs descubiertas: `{len(inventories)}`",
        f"- Gate counts: `{gate_counts}`",
        f"- Tier counts: `{counts}`",
        "- Modo: `artifact-only`; no se cargan modelos, no se recalculan latentes y no se usa test.",
        "- Provenance no pesa en el score; solo se usa como trazabilidad y desempate.",
        "- Menor `final_score` es mejor; las scores son z-scores robustas compuestas, no métricas absolutas.",
        "",
        "## Recomendacion operativa",
        f"- Decision: `{shortlist['recommendation']['decision']}`",
        f"- Lectura: {shortlist['recommendation']['rationale']}",
        f"- Semillas en Tier A: `{', '.join(tier_a_seeds) if tier_a_seeds else 'none'}`",
        f"- Configs distintas en Tier A/B: `{tier_ab_cfg_count}`",
        f"- Flags severos en top-5: `{'si' if top_has_severe else 'no'}`",
        "",
        "## Tiers",
        _markdown_table(_tier_summary(sorted_rows), ["tier", "count", "best_rank", "best_run_id", "lectura"]),
        "",
        "## Candidatos directos a FlowGen train-only",
        _markdown_table(
            direct,
            [
                "rank",
                "tier",
                "run_id",
                "final_score",
                "pc_worst_sum",
                "minority_pc_sum",
                "overfit_risk_level",
                "risk_flags",
            ],
            limit=5,
        ),
        "",
        "## Candidatos para reseed / backup",
        _markdown_table(
            reseed,
            [
                "rank",
                "tier",
                "run_id",
                "cfg_signature",
                "final_score",
                "source_view",
                "overfit_risk_level",
            ],
            limit=8,
        ),
        "",
        "## Runs dudosas / revisar",
        _markdown_table(
            doubtful,
            [
                "rank",
                "tier",
                "run_id",
                "gate_status",
                "overfit_risk_level",
                "risk_flags",
                "severe_flags",
                "caution_reasons",
            ],
            limit=12,
        ),
        "",
        "## Ranking compacto",
        _markdown_table(
            sorted_rows,
            [
                "rank",
                "tier",
                "run_id",
                "final_score",
                "per_class_prior_fit_score",
                "global_prior_fit_score",
                "artifact_robustness_score",
                "overfit_risk_score",
                "worst_class_id",
                "minority_class_id",
            ],
            limit=15,
        ),
        "",
        "## Rejects",
        _markdown_table(reject, ["rank", "run_id", "reject_reasons"], limit=20),
        "",
        "## Nota metodologica",
        "Este ranking selecciona priors de FlowPre para FlowGen train-only. La clave `val` de esta rama es una superficie de monitorizacion derivada de train, no un holdout temporal. Por eso el score prioriza train, per-class y riesgo de over-optimization.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _build_shortlist(rows: list[dict[str, Any]], analysis_id: str) -> dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda row: _as_float(row.get("rank"), 9999.0) or 9999.0)
    direct = [row for row in sorted_rows if row.get("tier") == "A"]
    reseed = _distinct_by_cfg([row for row in sorted_rows if row.get("tier") in {"A", "B"}], limit=5)
    watchlist = [row for row in sorted_rows if row.get("tier") != "Reject" and _has_material_review_flag(row)][:10]
    rejects = [row for row in sorted_rows if row.get("tier") == "Reject"]
    decision, rationale = _build_recommendation(rows)

    def slim(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "rank": row.get("rank"),
            "tier": row.get("tier"),
            "run_id": row.get("run_id"),
            "cfg_signature": row.get("cfg_signature"),
            "final_score": row.get("final_score"),
            "overfit_risk_level": row.get("overfit_risk_level"),
            "risk_flags": row.get("risk_flags"),
            "severe_flags": row.get("severe_flags"),
            "ranking_reason": row.get("ranking_reason"),
            "run_dir": row.get("run_dir"),
        }

    return {
        "analysis_id": analysis_id,
        "semantic_status": "not_computed_v1_artifact_only",
        "recommendation": {
            "decision": decision,
            "rationale": rationale,
        },
        "direct_flowgen_candidates": [slim(row) for row in direct],
        "reseed_candidates": [slim(row) for row in reseed],
        "watchlist": [slim(row) for row in watchlist],
        "rejects": [slim(row) for row in rejects],
    }


def _analysis_id() -> str:
    return "flowpre_trainonly_rank_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank FlowPre train-only runs using existing artifacts only.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--analysis-id", default=None)
    parser.add_argument("--mode", choices=["artifacts"], default="artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_root = args.runs_root.resolve()
    output_root = args.output_root.resolve()
    analysis_id = args.analysis_id or _analysis_id()
    out_dir = output_root / analysis_id
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered = _discover_runs(runs_root)
    rows: list[dict[str, Any]] = []
    inventories: list[dict[str, Any]] = []
    for record in discovered:
        try:
            row, inventory = _extract_row(record)
        except Exception as exc:
            run_id = str(record.get("run_id"))
            row = {
                "run_id": run_id,
                "run_dir": str(record.get("run_dir")),
                "gate_status": "reject",
                "reject_reasons": f"artifact_parse_error:{type(exc).__name__}",
                "caution_reasons": "none",
                "tier": "Reject",
                "ranking_reason": str(exc),
            }
            inventory = {
                "run_id": run_id,
                "run_dir": str(record.get("run_dir")),
                "artifact_parse_error": str(exc),
            }
        rows.append(row)
        inventories.append(inventory)

    _add_gate_reasons(rows)
    blocks_long = _add_scores(rows)
    sorted_rows = sorted(rows, key=lambda row: _as_float(row.get("rank"), 9999.0) or 9999.0)
    shortlist = _build_shortlist(sorted_rows, analysis_id)

    ranking_fields = [
        "rank",
        "tier",
        "run_id",
        "final_score",
        "per_class_prior_fit_score",
        "global_prior_fit_score",
        "artifact_robustness_score",
        "overfit_risk_score",
        "gate_status",
        "reject_reasons",
        "caution_reasons",
        "risk_flags",
        "severe_flags",
        "overfit_risk_level",
        "worst_class_id",
        "minority_class_id",
        "pc_worst_sum",
        "minority_pc_sum",
        "train_sum_rrmse",
        "train_mvn_score",
        "train_rrmse_recon",
        "cfg_signature",
        "phase",
        "source_view",
        "source_rank",
        "seed",
        "ranking_reason",
        "run_dir",
    ]
    gates_fields = [
        "run_id",
        "gate_status",
        "reject_reasons",
        "caution_reasons",
        "risk_flags",
        "severe_flags",
        "overfit_risk_level",
        "monitoring_policy",
        "monitor_source_split",
        "monitor_is_holdout",
        "test_enabled",
        "class_ids",
        "semantic_status",
    ]
    overfit_fields = [
        "run_id",
        "overfit_risk_score",
        "overfit_risk_level",
        "best_epoch",
        "total_epochs",
        "best_epoch_ratio",
        "best_monitor_loss",
        "final_monitor_loss",
        "tail_monitor_improvement",
        "tail_monitor_slope",
        "logabsdet_abs_final",
        "logabsdet_range",
        "capacity_proxy",
        "influence_top1_share_mean",
        "influence_top1_share_max",
        "influence_top3_share_mean",
        "influence_entropy_norm_mean",
        "risk_flags",
        "severe_flags",
    ]

    _write_csv(out_dir / "flowpre_trainonly_ranking.csv", sorted_rows, ranking_fields)
    _write_csv_with_options(out_dir / "flowpre_trainonly_blocks_long.csv", blocks_long, include_extra=True)
    _write_csv_with_options(out_dir / "flowpre_trainonly_gates_flags.csv", sorted_rows, gates_fields, include_extra=False)
    _write_csv_with_options(out_dir / "flowpre_trainonly_overfit_signals.csv", sorted_rows, overfit_fields, include_extra=False)
    _write_csv_with_options(out_dir / "flowpre_trainonly_inventory.csv", inventories, include_extra=True)
    with (out_dir / "flowpre_trainonly_shortlist.json").open("w", encoding="utf-8") as handle:
        json.dump(shortlist, handle, indent=2)
        handle.write("\n")

    manifest = {
        "analysis_id": analysis_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "runs_root": str(runs_root),
        "output_dir": str(out_dir),
        "run_count": len(discovered),
        "ranked_count": len([row for row in rows if row.get("tier") != "Reject"]),
        "reject_count": len([row for row in rows if row.get("tier") == "Reject"]),
        "score_weights": SCORE_WEIGHTS,
        "block_metrics": {
            "global_prior_fit": GLOBAL_BLOCK_METRICS,
            "per_class_prior_fit": PER_CLASS_BLOCK_METRICS,
            "artifact_robustness": ROBUSTNESS_BLOCK_METRICS,
            "overfit_risk": OVERFIT_BLOCK_METRICS,
        },
        "semantic_status": "not_computed_v1_artifact_only",
        "git_commit": _git_commit(),
        "argv": sys.argv if argv is None else argv,
    }
    with (out_dir / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    _write_summary(out_dir / "summary.md", sorted_rows, inventories, shortlist)

    print(f"Wrote FlowPre train-only ranking artifacts to: {out_dir}")
    print(f"Recommendation: {shortlist['recommendation']['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
