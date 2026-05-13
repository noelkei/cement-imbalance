from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import yaml

from data.sets import DEFAULT_OFFICIAL_DATASET_NAME, official_raw_bundle_manifest_path


ROOT_PATH = Path(__file__).resolve().parents[1]


OFFICIAL_SPLIT_ID = "init_temporal_processed_v1"
OFFICIAL_SPLIT_MANIFEST = ROOT_PATH / "data" / "splits" / "official" / OFFICIAL_SPLIT_ID / "manifest.json"
OFFICIAL_DATASET_MANIFEST = official_raw_bundle_manifest_path(
    split_id=OFFICIAL_SPLIT_ID,
    dataset_name=DEFAULT_OFFICIAL_DATASET_NAME,
)
DEFAULT_SPLIT_ROLE_MAP = {
    "train": "train_diagnostic",
    "val": "val_selection",
    "test": "test_holdout",
}


def _as_path_str(path: str | Path | None) -> Optional[str]:
    if path is None:
        return None
    return str(Path(path))


def _load_json_dict(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    with open(candidate, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _json_blob(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=_json_default)


def _strip_seed_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_seed_keys(val)
            for key, val in value.items()
            if key not in {"seed", "random_seed", "sampler_seed"}
        }
    if isinstance(value, list):
        return [_strip_seed_keys(item) for item in value]
    return value


def _load_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def build_run_context(
    *,
    model_family: str,
    run_id: str,
    seed: int | None,
    config: Optional[dict[str, Any]] = None,
    config_path: str | Path | None = None,
    dataset_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
    dataset_manifest_path: str | Path | None = OFFICIAL_DATASET_MANIFEST,
    split_id: str = OFFICIAL_SPLIT_ID,
    split_manifest_path: str | Path | None = OFFICIAL_SPLIT_MANIFEST,
    upstream_variant_fingerprint: str | None = None,
    split_role_map: Optional[dict[str, str]] = None,
    contract_id: str | None = None,
    comparison_group_id: str | None = None,
    seed_set_id: str | None = None,
    base_config_id: str | None = None,
    objective_metric_id: str | None = None,
    dataset_level_axes: Optional[Mapping[str, Any]] = None,
    run_level_axes: Optional[Mapping[str, Any]] = None,
    monitoring: Optional[Mapping[str, Any]] = None,
    test_enabled: bool | None = None,
) -> dict[str, Any]:
    config_dict = config or _load_config(config_path)
    config_path_str = _as_path_str(config_path)
    dataset_manifest_str = _as_path_str(dataset_manifest_path or OFFICIAL_DATASET_MANIFEST)
    split_manifest_str = _as_path_str(split_manifest_path or OFFICIAL_SPLIT_MANIFEST)
    dataset_manifest = _load_json_dict(dataset_manifest_str)
    resolved_dataset_name = dataset_name
    if dataset_manifest.get("dataset_name") and (
        dataset_name == DEFAULT_OFFICIAL_DATASET_NAME or dataset_name is None
    ):
        resolved_dataset_name = str(dataset_manifest["dataset_name"])
    resolved_split_id = str(dataset_manifest.get("split_id", split_id))
    resolved_contract_id = contract_id or dataset_manifest.get("contract_id")
    resolved_dataset_axes = dict(dataset_level_axes or dataset_manifest.get("dataset_level_axes") or {})
    resolved_run_axes = dict(run_level_axes or {})
    resolved_monitoring = dict(monitoring or {})
    resolved_comparison_group_id = comparison_group_id
    if (
        resolved_comparison_group_id is None
        and seed_set_id is not None
        and base_config_id is not None
        and objective_metric_id is not None
    ):
        metric_token = str(objective_metric_id).replace(".", "_")
        resolved_comparison_group_id = (
            f"{resolved_dataset_name}"
            f"__{resolved_split_id}"
            f"__{base_config_id}"
            f"__{seed_set_id}"
            f"__{metric_token}"
        )

    if config_path_str and Path(config_path_str).exists():
        config_sha256 = hashlib.sha256(Path(config_path_str).read_bytes()).hexdigest()
    else:
        config_sha256 = _stable_hash(config_dict)

    fingerprint_payload = {
        "model_family": model_family,
        "dataset_name": resolved_dataset_name,
        "dataset_manifest_path": dataset_manifest_str,
        "split_id": resolved_split_id,
        "split_manifest_path": split_manifest_str,
        "upstream_variant_fingerprint": upstream_variant_fingerprint,
        "contract_id": resolved_contract_id,
        "comparison_group_id": resolved_comparison_group_id,
        "seed_set_id": seed_set_id,
        "base_config_id": base_config_id,
        "objective_metric_id": objective_metric_id,
        "dataset_level_axes": resolved_dataset_axes,
        "run_level_axes": _strip_seed_keys(resolved_run_axes),
        "test_enabled": test_enabled,
        "config": _strip_seed_keys(config_dict),
    }
    if resolved_monitoring:
        fingerprint_payload["monitoring"] = resolved_monitoring

    variant_fingerprint = _stable_hash(fingerprint_payload)

    return {
        "run_id": run_id,
        "variant_fingerprint": variant_fingerprint,
        "model_family": model_family,
        "contract_id": resolved_contract_id,
        "comparison_group_id": resolved_comparison_group_id,
        "seed_set_id": seed_set_id,
        "base_config_id": base_config_id,
        "objective_metric_id": objective_metric_id,
        "upstream_variant_fingerprint": upstream_variant_fingerprint,
        "dataset_name": resolved_dataset_name,
        "dataset_manifest_path": dataset_manifest_str,
        "dataset_level_axes": resolved_dataset_axes or None,
        "run_level_axes": resolved_run_axes or None,
        "monitoring": resolved_monitoring or None,
        "split_id": resolved_split_id,
        "split_manifest_path": split_manifest_str,
        "seed": None if seed is None else int(seed),
        "config_path": config_path_str,
        "config_sha256": config_sha256,
        "test_enabled": None if test_enabled is None else bool(test_enabled),
        "split_role_map": dict(split_role_map or DEFAULT_SPLIT_ROLE_MAP),
    }


def _base_row(
    context: Mapping[str, Any],
    *,
    split: str,
    metric_group: str,
    metric_name: str,
    metric_scope: str,
    value_space: str,
    metric_value: Any,
    component: str | None = None,
    class_id: int | None = None,
    target_name: str | None = None,
    n_obs: int | None = None,
) -> dict[str, Any]:
    return {
        "run_id": context["run_id"],
        "variant_fingerprint": context["variant_fingerprint"],
        "model_family": context["model_family"],
        "contract_id": context.get("contract_id"),
        "comparison_group_id": context.get("comparison_group_id"),
        "seed_set_id": context.get("seed_set_id"),
        "base_config_id": context.get("base_config_id"),
        "objective_metric_id": context.get("objective_metric_id"),
        "upstream_variant_fingerprint": context.get("upstream_variant_fingerprint"),
        "dataset_name": context.get("dataset_name"),
        "dataset_manifest_path": context.get("dataset_manifest_path"),
        "dataset_level_axes": _json_blob(context.get("dataset_level_axes")),
        "run_level_axes": _json_blob(context.get("run_level_axes")),
        "split_id": context.get("split_id"),
        "split_manifest_path": context.get("split_manifest_path"),
        "seed": context.get("seed"),
        "config_path": context.get("config_path"),
        "config_sha256": context.get("config_sha256"),
        "test_enabled": context.get("test_enabled"),
        "split": split,
        "split_role": context.get("split_role_map", DEFAULT_SPLIT_ROLE_MAP).get(split, split),
        "metric_group": metric_group,
        "metric_name": metric_name,
        "metric_scope": metric_scope,
        "component": component,
        "class_id": class_id,
        "target_name": target_name,
        "value_space": value_space,
        "metric_value": None if metric_value is None else float(metric_value),
        "n_obs": None if n_obs is None else int(n_obs),
    }


def _safe_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    return int(str(value))


def _flatten_predictive_metrics(
    rows: list[dict[str, Any]],
    *,
    split: str,
    metrics: Mapping[str, Any],
    context: Mapping[str, Any],
    value_space: str,
) -> None:
    overall = metrics.get("overall", {})
    for metric_name, metric_value in overall.items():
        if metric_name == "quantiles":
            for qname, qmetrics in metric_value.items():
                n_obs = qmetrics.get("n")
                for q_metric_name, q_metric_value in qmetrics.items():
                    if q_metric_name == "n":
                        continue
                    rows.append(
                        _base_row(
                            context,
                            split=split,
                            metric_group="predictive",
                            metric_name=q_metric_name,
                            metric_scope="overall_quantile",
                            component=qname,
                            value_space=value_space,
                            metric_value=q_metric_value,
                            n_obs=n_obs,
                        )
                    )
            continue
        if metric_name == "n":
            continue
        rows.append(
            _base_row(
                context,
                split=split,
                metric_group="predictive",
                metric_name=metric_name,
                metric_scope="overall",
                value_space=value_space,
                metric_value=metric_value,
                n_obs=overall.get("n"),
            )
        )

    for scope_name in ("macro", "worst_class"):
        scope_metrics = metrics.get(scope_name, {})
        for metric_name, metric_value in scope_metrics.items():
            if metric_name == "n":
                continue
            rows.append(
                _base_row(
                    context,
                    split=split,
                    metric_group="predictive",
                    metric_name=metric_name,
                    metric_scope=scope_name,
                    value_space=value_space,
                    metric_value=metric_value,
                    n_obs=scope_metrics.get("n") or overall.get("n"),
                )
            )

    for cls_id, cls_metrics in metrics.get("per_class", {}).items():
        class_id = _safe_int(cls_id)
        n_obs = cls_metrics.get("n")
        for metric_name, metric_value in cls_metrics.items():
            if metric_name == "quantiles":
                for qname, qmetrics in metric_value.items():
                    q_n_obs = qmetrics.get("n") or n_obs
                    for q_metric_name, q_metric_value in qmetrics.items():
                        if q_metric_name == "n":
                            continue
                        rows.append(
                            _base_row(
                                context,
                                split=split,
                                metric_group="predictive",
                                metric_name=q_metric_name,
                                metric_scope="per_class_quantile",
                                component=qname,
                                class_id=class_id,
                                value_space=value_space,
                                metric_value=q_metric_value,
                                n_obs=q_n_obs,
                            )
                        )
                continue
            if metric_name == "n":
                continue
            rows.append(
                _base_row(
                    context,
                    split=split,
                    metric_group="predictive",
                    metric_name=metric_name,
                    metric_scope="per_class",
                    class_id=class_id,
                    value_space=value_space,
                    metric_value=metric_value,
                    n_obs=n_obs,
                )
            )


def _flatten_flowpre_metrics(
    rows: list[dict[str, Any]],
    *,
    split: str,
    metrics: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    for metric_name in ("rrmse_recon", "r2_recon"):
        if metric_name in metrics:
            rows.append(
                _base_row(
                    context,
                    split=split,
                    metric_group="reconstruction",
                    metric_name=metric_name,
                    metric_scope="overall",
                    component="x",
                    value_space="native",
                    metric_value=metrics[metric_name],
                    n_obs=metrics.get("n"),
                )
            )

    for metric_name in ("rrmse_mean_whole", "rrmse_std_whole"):
        if metric_name in metrics:
            rows.append(
                _base_row(
                    context,
                    split=split,
                    metric_group="isotropy",
                    metric_name=metric_name,
                    metric_scope="overall",
                    component="z",
                    value_space="latent",
                    metric_value=metrics[metric_name],
                    n_obs=metrics.get("n"),
                )
            )

    for cls_id, cls_metrics in metrics.get("per_class_iso_rrmse", {}).items():
        class_id = _safe_int(cls_id)
        n_obs = cls_metrics.get("n")
        for metric_name, metric_value in cls_metrics.items():
            if metric_name == "n":
                continue
            rows.append(
                _base_row(
                    context,
                    split=split,
                    metric_group="isotropy",
                    metric_name=metric_name,
                    metric_scope="per_class",
                    component="z",
                    class_id=class_id,
                    value_space="latent",
                    metric_value=metric_value,
                    n_obs=n_obs,
                )
            )

    for metric_name, metric_value in metrics.get("isotropy_stats", {}).items():
        rows.append(
            _base_row(
                context,
                split=split,
                metric_group="latent_stats",
                metric_name=metric_name,
                metric_scope="overall",
                component="z",
                value_space="latent",
                metric_value=metric_value,
                n_obs=metrics.get("n"),
            )
        )

    for cls_id, cls_metrics in metrics.get("isotropy_stats_per_class", {}).items():
        class_id = _safe_int(cls_id)
        n_obs = cls_metrics.get("n")
        for metric_name, metric_value in cls_metrics.items():
            if metric_name == "n":
                continue
            rows.append(
                _base_row(
                    context,
                    split=split,
                    metric_group="latent_stats",
                    metric_name=metric_name,
                    metric_scope="per_class",
                    component="z",
                    class_id=class_id,
                    value_space="latent",
                    metric_value=metric_value,
                    n_obs=n_obs,
                )
            )


def _flatten_flowgen_metrics(
    rows: list[dict[str, Any]],
    *,
    split: str,
    metrics: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    recon_map = {
        "rrmse_x_recon": ("rrmse_recon", "x"),
        "rrmse_y_recon": ("rrmse_recon", "y"),
        "r2_x_recon": ("r2_recon", "x"),
        "r2_y_recon": ("r2_recon", "y"),
    }
    for key, (metric_name, component) in recon_map.items():
        if key in metrics:
            rows.append(
                _base_row(
                    context,
                    split=split,
                    metric_group="reconstruction",
                    metric_name=metric_name,
                    metric_scope="overall",
                    component=component,
                    value_space="native",
                    metric_value=metrics[key],
                    n_obs=metrics.get("n"),
                )
            )

    iso_map = {
        "loss_rrmse_x_mean_whole": ("rrmse_mean_whole", "x"),
        "loss_rrmse_x_std_whole": ("rrmse_std_whole", "x"),
        "loss_rrmse_y_mean_whole": ("rrmse_mean_whole", "y"),
        "loss_rrmse_y_std_whole": ("rrmse_std_whole", "y"),
    }
    for key, (metric_name, component) in iso_map.items():
        if key in metrics:
            rows.append(
                _base_row(
                    context,
                    split=split,
                    metric_group="isotropy",
                    metric_name=metric_name,
                    metric_scope="overall",
                    component=component,
                    value_space="latent",
                    metric_value=metrics[key],
                    n_obs=metrics.get("n"),
                )
            )

    for cls_id, cls_metrics in metrics.get("per_class_iso_rrmse", {}).items():
        class_id = _safe_int(cls_id)
        n_obs = cls_metrics.get("n")
        for metric_name, metric_value in cls_metrics.items():
            if metric_name == "n":
                continue
            component = "x" if "_x_" in metric_name else "y" if "_y_" in metric_name else "z"
            rows.append(
                _base_row(
                    context,
                    split=split,
                    metric_group="isotropy",
                    metric_name=metric_name,
                    metric_scope="per_class",
                    component=component,
                    class_id=class_id,
                    value_space="latent",
                    metric_value=metric_value,
                    n_obs=n_obs,
                )
            )

    for metric_name, metric_value in metrics.get("isotropy_stats", {}).items():
        rows.append(
            _base_row(
                context,
                split=split,
                metric_group="latent_stats",
                metric_name=metric_name,
                metric_scope="overall",
                component="z",
                value_space="latent",
                metric_value=metric_value,
                n_obs=metrics.get("n"),
            )
        )

    for cls_id, cls_metrics in metrics.get("isotropy_stats_per_class", {}).items():
        class_id = _safe_int(cls_id)
        n_obs = cls_metrics.get("n")
        for metric_name, metric_value in cls_metrics.items():
            if metric_name == "n":
                continue
            rows.append(
                _base_row(
                    context,
                    split=split,
                    metric_group="latent_stats",
                    metric_name=metric_name,
                    metric_scope="per_class",
                    component="z",
                    class_id=class_id,
                    value_space="latent",
                    metric_value=metric_value,
                    n_obs=n_obs,
                )
            )

    realism = metrics.get("realism", {})
    for component in ("overall", "x", "y"):
        for metric_name, metric_value in realism.get(component, {}).items():
            rows.append(
                _base_row(
                    context,
                    split=split,
                    metric_group="realism",
                    metric_name=metric_name,
                    metric_scope="overall",
                    component=component,
                    value_space="distribution",
                    metric_value=metric_value,
                    n_obs=metrics.get("n"),
                )
            )
    for cls_id, suites in realism.get("per_class", {}).items():
        class_id = _safe_int(cls_id)
        for component in ("overall", "x", "y"):
            for metric_name, metric_value in suites.get(component, {}).items():
                rows.append(
                    _base_row(
                        context,
                        split=split,
                        metric_group="realism",
                        metric_name=metric_name,
                        metric_scope="per_class",
                        component=component,
                        class_id=class_id,
                        value_space="distribution",
                        metric_value=metric_value,
                        n_obs=metrics.get("n"),
                    )
                )


def flatten_run_results(results: Mapping[str, Any], context: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    family = str(context["model_family"]).lower()

    if family == "mlp":
        for split in ("train", "val", "test"):
            metrics = results.get(split)
            if isinstance(metrics, Mapping):
                _flatten_predictive_metrics(rows, split=split, metrics=metrics, context=context, value_space="native")
        raw_metrics = results.get("raw_real", {})
        for split in ("train", "val", "test"):
            metrics = raw_metrics.get(split)
            if isinstance(metrics, Mapping):
                _flatten_predictive_metrics(rows, split=split, metrics=metrics, context=context, value_space="raw_real")
    elif family == "flowpre":
        for split in ("train", "val", "test"):
            metrics = results.get(split)
            if isinstance(metrics, Mapping):
                _flatten_flowpre_metrics(rows, split=split, metrics=metrics, context=context)
    elif family == "flowgen":
        for split in ("train", "val", "test"):
            metrics = results.get(split)
            if isinstance(metrics, Mapping):
                _flatten_flowgen_metrics(rows, split=split, metrics=metrics, context=context)
    else:
        raise ValueError(f"Unsupported model_family '{context['model_family']}'.")

    return pd.DataFrame(rows)


def save_canonical_run_artifacts(
    *,
    results: Mapping[str, Any],
    context: Mapping[str, Any],
    out_dir: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_long = flatten_run_results(results, context)
    manifest_path = out_dir / f"{stem}_run_manifest.json"
    metrics_path = out_dir / f"{stem}_metrics_long.csv"

    manifest = {
        "run_id": context["run_id"],
        "variant_fingerprint": context["variant_fingerprint"],
        "model_family": context["model_family"],
        "contract_id": context.get("contract_id"),
        "comparison_group_id": context.get("comparison_group_id"),
        "seed_set_id": context.get("seed_set_id"),
        "base_config_id": context.get("base_config_id"),
        "objective_metric_id": context.get("objective_metric_id"),
        "upstream_variant_fingerprint": context.get("upstream_variant_fingerprint"),
        "dataset_name": context.get("dataset_name"),
        "dataset_manifest_path": context.get("dataset_manifest_path"),
        "dataset_level_axes": context.get("dataset_level_axes"),
        "run_level_axes": context.get("run_level_axes"),
        "monitoring": context.get("monitoring"),
        "split_id": context.get("split_id"),
        "split_manifest_path": context.get("split_manifest_path"),
        "seed": context.get("seed"),
        "config_path": context.get("config_path"),
        "config_sha256": context.get("config_sha256"),
        "test_enabled": context.get("test_enabled"),
        "split_role_map": context.get("split_role_map"),
        "metrics_long_path": str(metrics_path),
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    metrics_long.to_csv(metrics_path, index=False)
    return manifest_path, metrics_path


def save_promotion_manifest(
    *,
    out_path: str | Path,
    model_family: str,
    source_id: str,
    source_run_manifest_path: str | Path,
    source_metrics_long_path: str | Path,
    split_id: str,
    cleaning_policy_id: str,
    raw_bundle_manifest_path: str | Path,
    branch_id: str | None = None,
    paired_flowpre_source_id: str | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_family": model_family,
        "source_id": source_id,
        "branch_id": branch_id,
        "paired_flowpre_source_id": paired_flowpre_source_id,
        "source_run_manifest": _as_path_str(source_run_manifest_path),
        "source_metrics_long_path": _as_path_str(source_metrics_long_path),
        "split_id": split_id,
        "cleaning_policy_id": cleaning_policy_id,
        "raw_bundle_manifest_path": _as_path_str(raw_bundle_manifest_path),
    }
    if extra_fields:
        payload.update(json.loads(json.dumps(extra_fields, default=_json_default)))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return out_path


def load_metrics_long(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
