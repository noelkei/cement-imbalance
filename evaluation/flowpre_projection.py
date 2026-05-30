from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from data.utils import ROOT_PATH, dump_json, path_relative_to_root
from training.utils import load_yaml_config, select_training_device


_MLP_INTERPRETABILITY_CONTRACT_PATH = Path(ROOT_PATH) / "config" / "f7_mlp_interpretability_contract_v1.yaml"


@lru_cache(maxsize=1)
def load_f7_mlp_interpretability_contract() -> dict[str, Any]:
    with open(_MLP_INTERPRETABILITY_CONTRACT_PATH, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    contract = payload.get("mlp_interpretability_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            f"Invalid MLP interpretability contract payload in {_MLP_INTERPRETABILITY_CONTRACT_PATH}"
        )
    return contract


def _load_json_dict(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        repo_candidate = Path(ROOT_PATH) / candidate
        if repo_candidate.exists():
            candidate = repo_candidate
    with open(candidate, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping at {candidate}")
    return payload


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    repo_candidate = Path(ROOT_PATH) / candidate
    if repo_candidate.exists():
        return repo_candidate
    raise FileNotFoundError(f"Could not resolve repo path: {path}")


def _resolve_flowpre_run_config_path(run_dir: Path, run_manifest: dict[str, Any], run_id: str) -> Path:
    explicit_candidates = [
        run_manifest.get("config_snapshot_path"),
        run_manifest.get("config_path"),
    ]
    for candidate in explicit_candidates:
        if candidate in (None, ""):
            continue
        resolved = _resolve_repo_path(candidate)
        if resolved.exists():
            return resolved
    exact_versioned = run_dir / f"{run_id}.yaml"
    if exact_versioned.exists():
        return exact_versioned
    exact_stable = run_dir / "config.yaml"
    if exact_stable.exists():
        return exact_stable
    raise FileNotFoundError(
        f"Could not resolve strict FlowPre config for run_id '{run_id}' under {run_dir}. "
        "Expected config_snapshot_path/config_path in run manifest or exact config artifact."
    )


def _resolve_flowpre_run_weights_path(run_dir: Path, run_manifest: dict[str, Any], run_id: str) -> Path:
    explicit_candidates = [
        run_manifest.get("model_path"),
    ]
    for candidate in explicit_candidates:
        if candidate in (None, ""):
            continue
        resolved = _resolve_repo_path(candidate)
        if resolved.exists():
            return resolved
    exact_versioned = run_dir / f"{run_id}.pt"
    if exact_versioned.exists():
        return exact_versioned
    raise FileNotFoundError(
        f"Could not resolve strict FlowPre weights for run_id '{run_id}' under {run_dir}. "
        "Expected model_path in run manifest or exact checkpoint artifact."
    )


def _projection_cache_dir(source_id: str, projection_cache_root: str | Path | None = None) -> Path:
    contract = load_f7_mlp_interpretability_contract()
    projection_cfg = dict(contract.get("flowpre_projection") or {})
    root = projection_cache_root or projection_cfg.get("cache_root", "outputs/reports/f7_flowpre_projection_cache")
    cache_version = int(projection_cfg.get("cache_version", 1))
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = Path(ROOT_PATH) / root_path
    return root_path / f"{source_id}__v{cache_version}"


def resolve_flowpre_promotion_manifest_from_dataset_manifest(dataset_manifest_payload: dict[str, Any]) -> Path | None:
    upstream_manifests = list(dataset_manifest_payload.get("upstream_model_manifests") or [])
    if not upstream_manifests:
        return None
    for ref in upstream_manifests:
        try:
            candidate = _resolve_repo_path(ref)
        except FileNotFoundError:
            continue
        try:
            payload = _load_json_dict(candidate)
        except Exception:
            continue
        source_id = str(payload.get("source_id", ""))
        model_family = str(payload.get("model_family", "")).lower()
        if model_family == "flowpre" or source_id.startswith("flowpre__"):
            return candidate
    return None


def _load_flowpre_model_from_promotion(
    promotion_manifest_path: str | Path,
    *,
    x_reference: pd.DataFrame,
    condition_col: str = "type",
    device: str = "cpu",
) -> tuple[Any, torch.device, dict[str, Any], dict[str, Any], Path]:
    from training.train_flow_pre import build_flow_pre_model, filter_flowpre_columns

    promotion_manifest = _load_json_dict(promotion_manifest_path)
    run_manifest_path = _resolve_repo_path(promotion_manifest["source_run_manifest"])
    run_manifest = _load_json_dict(run_manifest_path)
    run_dir = run_manifest_path.parent
    run_id = str(run_manifest.get("run_id", run_dir.name))

    cfg_path = _resolve_flowpre_run_config_path(run_dir, run_manifest, run_id)
    pt_path = _resolve_flowpre_run_weights_path(run_dir, run_manifest, run_id)

    config = load_yaml_config(cfg_path)
    model_cfg = dict(config["model"])
    device_obj = select_training_device(device)

    ref_filtered = filter_flowpre_columns(
        x_reference,
        cols_to_exclude=["post_cleaning_index"],
        condition_col=condition_col,
    )
    input_dim = ref_filtered.drop(columns=[condition_col]).shape[1]
    num_classes = int(ref_filtered[condition_col].nunique())

    model = build_flow_pre_model(
        model_cfg,
        input_dim=input_dim,
        num_classes=num_classes,
        device=device_obj,
    )
    state_dict = torch.load(pt_path, map_location=device_obj)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device_obj, promotion_manifest, run_manifest, cfg_path


def _load_modeled_raw_train_frame_from_run_manifest(run_manifest: dict[str, Any]) -> pd.DataFrame:
    dataset_manifest_path = run_manifest.get("dataset_manifest_path")
    if not dataset_manifest_path:
        raise ValueError("FlowPre run manifest is missing dataset_manifest_path.")
    dataset_manifest = _load_json_dict(_resolve_repo_path(dataset_manifest_path))
    artifacts = dict(dataset_manifest.get("artifacts") or {})
    x_train_path = (((artifacts.get("X") or {}).get("train")))
    if not x_train_path:
        raise ValueError("Raw bundle manifest is missing artifacts.X.train.")
    return pd.read_csv(_resolve_repo_path(x_train_path))


@lru_cache(maxsize=8)
def _load_flowpre_decoder_runtime_cached(
    promotion_manifest_path_str: str,
    device: str,
    condition_col: str,
) -> dict[str, Any]:
    promotion_manifest_path = _resolve_repo_path(promotion_manifest_path_str)
    promotion_manifest = _load_json_dict(promotion_manifest_path)
    run_manifest_path = _resolve_repo_path(promotion_manifest["source_run_manifest"])
    run_manifest = _load_json_dict(run_manifest_path)
    x_train_raw = _load_modeled_raw_train_frame_from_run_manifest(run_manifest)
    model, device_obj, _, _, _ = _load_flowpre_model_from_promotion(
        promotion_manifest_path,
        x_reference=x_train_raw,
        condition_col=condition_col,
        device=device,
    )
    semantic_feature_names = [
        col for col in x_train_raw.columns if col not in {"post_cleaning_index", condition_col, "is_synth"}
    ]
    return {
        "promotion_manifest_path": promotion_manifest_path,
        "promotion_manifest": promotion_manifest,
        "run_manifest_path": run_manifest_path,
        "run_manifest": run_manifest,
        "x_train_raw": x_train_raw,
        "decoder_model": model,
        "device": device_obj,
        "semantic_feature_names": semantic_feature_names,
    }


def load_flowpre_decoder_runtime(
    promotion_manifest_path: str | Path,
    *,
    device: str = "cpu",
    condition_col: str = "type",
) -> dict[str, Any]:
    resolved_path = _resolve_repo_path(promotion_manifest_path)
    return _load_flowpre_decoder_runtime_cached(str(resolved_path), str(device), str(condition_col))


def _build_projection_rows(
    *,
    model: Any,
    device: torch.device,
    x_train_raw: pd.DataFrame,
    condition_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from training.train_flow_pre import encode_with_flowpre_model

    contract = load_f7_mlp_interpretability_contract()
    projection_cfg = dict(contract.get("flowpre_projection") or {})
    sweep_min, sweep_max = tuple(projection_cfg.get("sweep_offset_range", [-1.0, 1.0]))
    num_steps = int(projection_cfg.get("sweep_num_steps", 33))
    normalize_rows = bool(projection_cfg.get("normalize_rows", True))

    encoded_train = encode_with_flowpre_model(
        x_train_raw,
        model=model,
        device=device,
        condition_col=condition_col,
        cols_to_exclude=["post_cleaning_index"],
    )
    latent_names = [col for col in encoded_train.columns if col.startswith("z_")]
    semantic_feature_names = [
        col for col in x_train_raw.columns if col not in {"post_cleaning_index", condition_col, "is_synth"}
    ]
    x_numeric = x_train_raw[semantic_feature_names].to_numpy(dtype=np.float32)
    latent_numeric = encoded_train[latent_names].to_numpy(dtype=np.float32)
    class_codes = encoded_train[condition_col].to_numpy(dtype=np.int64)

    sweep_offsets = torch.linspace(
        float(sweep_min),
        float(sweep_max),
        steps=num_steps,
        dtype=torch.float32,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    for class_code in sorted(np.unique(class_codes).tolist()):
        class_mask = class_codes == int(class_code)
        z_class = latent_numeric[class_mask]
        if z_class.size == 0:
            continue
        base_z = torch.as_tensor(z_class.mean(axis=0), dtype=torch.float32, device=device)
        c_repeat = torch.full((num_steps,), int(class_code), dtype=torch.long, device=device)
        for latent_idx, latent_name in enumerate(latent_names):
            z_sweep = base_z.repeat(num_steps, 1)
            z_sweep[:, latent_idx] = base_z[latent_idx] + sweep_offsets
            with torch.no_grad():
                x_rec = model.inverse(z_sweep, c_repeat)[0].detach().cpu().numpy()
            feature_std = np.std(x_rec, axis=0)
            total = float(feature_std.sum())
            if normalize_rows and total > 0:
                feature_norm = feature_std / total
            else:
                feature_norm = np.zeros_like(feature_std)
            for semantic_idx, semantic_feature in enumerate(semantic_feature_names):
                rows.append(
                    {
                        "type": int(class_code),
                        "latent_name": str(latent_name),
                        "semantic_feature": str(semantic_feature),
                        "weight_raw_std": round(float(feature_std[semantic_idx]), 10),
                        "weight_norm": round(float(feature_norm[semantic_idx]), 10),
                    }
                )

    df = pd.DataFrame.from_records(rows)
    stats = {
        "n_classes": int(len(np.unique(class_codes))),
        "n_latent_features": int(len(latent_names)),
        "n_semantic_features": int(len(semantic_feature_names)),
        "sweep_offset_range": [float(sweep_min), float(sweep_max)],
        "sweep_num_steps": int(num_steps),
        "normalize_rows": bool(normalize_rows),
    }
    return df, stats


def resolve_or_build_flowpre_projection_cache(
    *,
    promotion_manifest_path: str | Path,
    projection_cache_root: str | Path | None = None,
    device: str = "cpu",
    force_rebuild: bool = False,
    condition_col: str = "type",
) -> dict[str, Any]:
    promotion_manifest_path = _resolve_repo_path(promotion_manifest_path)
    promotion_manifest = _load_json_dict(promotion_manifest_path)
    source_id = str(promotion_manifest.get("source_id") or promotion_manifest_path.stem)
    cache_dir = _projection_cache_dir(source_id, projection_cache_root=projection_cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "projection_manifest.json"
    table_path = cache_dir / "latent_to_semantic_per_class.csv"

    if not force_rebuild and manifest_path.exists() and table_path.exists():
        manifest = _load_json_dict(manifest_path)
        table = pd.read_csv(table_path)
        return {
            "projection_manifest_path": manifest_path,
            "projection_cache_path": cache_dir,
            "projection_table_path": table_path,
            "projection_manifest": manifest,
            "projection_table": table,
        }

    decoder_runtime = load_flowpre_decoder_runtime(
        promotion_manifest_path,
        device=device,
        condition_col=condition_col,
    )
    run_manifest_path = Path(decoder_runtime["run_manifest_path"])
    run_manifest = dict(decoder_runtime["run_manifest"])
    x_train_raw = decoder_runtime["x_train_raw"]
    model = decoder_runtime["decoder_model"]
    device_obj = decoder_runtime["device"]
    projection_table, projection_stats = _build_projection_rows(
        model=model,
        device=device_obj,
        x_train_raw=x_train_raw,
        condition_col=condition_col,
    )
    projection_table.to_csv(table_path, index=False)
    legacy_reference_path = run_manifest_path.parent / f"{run_manifest_path.parent.name}_influence.json"
    manifest = {
        "projection_contract_id": "f7_flowpre_projection_cache_v1",
        "source_id": source_id,
        "promotion_manifest_path": path_relative_to_root(promotion_manifest_path),
        "source_run_manifest": path_relative_to_root(run_manifest_path),
        "raw_train_dataset_manifest_path": path_relative_to_root(run_manifest["dataset_manifest_path"]),
        "legacy_reference_path": (
            path_relative_to_root(legacy_reference_path) if legacy_reference_path.exists() else None
        ),
        "projection_table_path": path_relative_to_root(table_path),
        "projection_cache_path": path_relative_to_root(cache_dir),
        "projection_stats": projection_stats,
    }
    dump_json(manifest, manifest_path)
    return {
        "projection_manifest_path": manifest_path,
        "projection_cache_path": cache_dir,
        "projection_table_path": table_path,
        "projection_manifest": manifest,
        "projection_table": projection_table,
    }
