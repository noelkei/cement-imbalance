from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from data.utils import ROOT_PATH
from evaluation.metrics import inverse_transform_tensor


_ARTIFACT_CONTRACT_PATH = Path(ROOT_PATH) / "config" / "f7_artifact_persistence_contract_v1.yaml"
_F7_STABLE_FILENAME_POLICY = "stable_single_copy"
_LEGACY_FILENAME_POLICY = "versioned_aliases"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


@lru_cache(maxsize=1)
def load_f7_artifact_persistence_contract() -> dict[str, Any]:
    with open(_ARTIFACT_CONTRACT_PATH, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    contract = payload.get("artifact_persistence_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Invalid artifact persistence contract payload in {_ARTIFACT_CONTRACT_PATH}")
    return contract


def resolve_model_artifact_policy(model_family: str) -> str:
    contract = load_f7_artifact_persistence_contract()
    families = dict(contract.get("family_overrides") or {})
    family_block = dict(families.get(str(model_family).lower()) or {})
    return str(family_block.get("model_artifact_policy", "unspecified"))


def resolve_f7_run_filename_policy(analysis_contracts: Mapping[str, Any] | None = None) -> str:
    contract = load_f7_artifact_persistence_contract()
    filename_policy = dict(contract.get("filename_policy") or {})
    legacy_default = str(filename_policy.get("legacy_default", _LEGACY_FILENAME_POLICY))
    if not analysis_contracts:
        return legacy_default
    artifact_policy_id = str((analysis_contracts or {}).get("artifact_policy_id") or "")
    if artifact_policy_id != str(contract.get("artifact_policy_id") or ""):
        return legacy_default
    if bool(filename_policy.get("single_copy_new_f7_campaign_runs", False)):
        return str(filename_policy.get("new_f7_campaign_runs", _F7_STABLE_FILENAME_POLICY))
    return legacy_default


def uses_f7_stable_single_copy_policy(analysis_contracts: Mapping[str, Any] | None = None) -> bool:
    return resolve_f7_run_filename_policy(analysis_contracts) == _F7_STABLE_FILENAME_POLICY


def build_interpretability_status_block() -> dict[str, Any]:
    contract = load_f7_artifact_persistence_contract()
    policy = dict(contract.get("interpretability_policy") or {})
    return {
        "interpretability_policy_status": str(policy.get("interpretability_policy_status", "pending_family_block")),
        "interpretability_artifacts": {},
        "interpretability_required_now": bool(policy.get("interpretability_summary_required_per_run", False)),
        "interpretability_required_for_shortlist": bool(
            policy.get("interpretability_summary_required_per_shortlist", True)
        ),
        "interpretability_required_for_finalist": bool(
            policy.get("interpretability_summary_required_per_finalist", True)
        ),
    }


def _to_numpy_2d(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Expected non-null array-like value.")
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _inverse_with_scaler(array: np.ndarray, y_scaler: Any | None) -> np.ndarray:
    if y_scaler is None:
        return array.astype(np.float32, copy=False)
    tensor = torch.as_tensor(array, dtype=torch.float32)
    restored = inverse_transform_tensor(tensor, y_scaler, device=torch.device("cpu"))
    return restored.detach().cpu().numpy()


def build_prediction_sidecar_df(
    *,
    split_payloads: Mapping[str, Mapping[str, Any]],
    target_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, payload in split_payloads.items():
        meta_df = payload.get("meta")
        if meta_df is None:
            continue
        meta = pd.DataFrame(meta_df).copy()
        if meta.empty:
            continue
        if "post_cleaning_index" not in meta.columns or "type" not in meta.columns:
            raise ValueError(f"Prediction sidecar meta for split '{split_name}' must include post_cleaning_index and type.")
        if "is_synth" not in meta.columns:
            meta["is_synth"] = 0
        meta["is_synth"] = meta["is_synth"].fillna(0).astype(int)

        y_true_raw = _to_numpy_2d(payload.get("y_true_raw"))
        y_pred_raw = _to_numpy_2d(payload.get("y_pred_raw"))
        y_true_native = payload.get("y_true_native")
        y_pred_native = payload.get("y_pred_native")
        y_true_native_np = _to_numpy_2d(y_true_native) if y_true_native is not None else None
        y_pred_native_np = _to_numpy_2d(y_pred_native) if y_pred_native is not None else None

        if y_true_raw.shape != y_pred_raw.shape:
            raise ValueError(f"Raw prediction shapes do not match for split '{split_name}'.")
        if len(meta) != y_true_raw.shape[0]:
            raise ValueError(f"Prediction sidecar row count mismatch for split '{split_name}'.")
        if y_true_raw.shape[1] != len(target_names):
            raise ValueError(
                f"Target name count mismatch for split '{split_name}': "
                f"{y_true_raw.shape[1]} values vs {len(target_names)} target names."
            )

        for row_idx in range(len(meta)):
            base = {
                "split": str(split_name),
                "post_cleaning_index": int(meta.iloc[row_idx]["post_cleaning_index"]),
                "type": meta.iloc[row_idx]["type"],
                "is_synth": int(meta.iloc[row_idx]["is_synth"]),
            }
            for target_idx, target_name in enumerate(target_names):
                y_true_raw_value = float(y_true_raw[row_idx, target_idx])
                y_pred_raw_value = float(y_pred_raw[row_idx, target_idx])
                record = dict(base)
                record.update(
                    {
                        "target_name": str(target_name),
                        "y_true_raw": y_true_raw_value,
                        "y_pred_raw": y_pred_raw_value,
                        "abs_error_raw": abs(y_pred_raw_value - y_true_raw_value),
                        "squared_error_raw": float((y_pred_raw_value - y_true_raw_value) ** 2),
                        "y_true_native": (
                            None if y_true_native_np is None else float(y_true_native_np[row_idx, target_idx])
                        ),
                        "y_pred_native": (
                            None if y_pred_native_np is None else float(y_pred_native_np[row_idx, target_idx])
                        ),
                    }
                )
                rows.append(record)
    return pd.DataFrame(rows)


def write_prediction_sidecar(
    *,
    out_dir: str | Path,
    split_payloads: Mapping[str, Mapping[str, Any]],
    target_names: list[str],
) -> Path | None:
    if not split_payloads:
        return None
    contract = load_f7_artifact_persistence_contract()
    sidecar_cfg = dict(contract.get("prediction_sidecar_policy") or {})
    filename = str(sidecar_cfg.get("filename", "predictions_eval_raw.csv.gz"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_df = build_prediction_sidecar_df(split_payloads=split_payloads, target_names=target_names)
    if sidecar_df.empty:
        return None
    out_path = out_dir / filename
    sidecar_df.to_csv(out_path, index=False, compression="gzip")
    return out_path


def build_prediction_sidecar_payload_from_native(
    *,
    split_meta: Mapping[str, pd.DataFrame | None],
    predictions_by_split: Mapping[str, Any],
    targets_by_split: Mapping[str, Any],
    target_names: list[str],
    y_scaler: Any | None,
    include_splits: list[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payloads: dict[str, dict[str, Any]] = {}
    included: list[str] = []
    for split_name in include_splits:
        meta_df = split_meta.get(split_name)
        if meta_df is None:
            continue
        if split_name not in predictions_by_split or split_name not in targets_by_split:
            continue
        y_pred_native = _to_numpy_2d(predictions_by_split[split_name])
        y_true_native = _to_numpy_2d(targets_by_split[split_name])
        payloads[split_name] = {
            "meta": pd.DataFrame(meta_df).copy(),
            "y_true_native": y_true_native,
            "y_pred_native": y_pred_native,
            "y_true_raw": _inverse_with_scaler(y_true_native, y_scaler),
            "y_pred_raw": _inverse_with_scaler(y_pred_native, y_scaler),
        }
        included.append(split_name)
    return payloads, included


def build_prediction_sidecar_payload_from_raw(
    *,
    split_meta: Mapping[str, pd.DataFrame | None],
    predictions_by_split: Mapping[str, Any],
    targets_by_split: Mapping[str, Any],
    include_splits: list[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payloads: dict[str, dict[str, Any]] = {}
    included: list[str] = []
    for split_name in include_splits:
        meta_df = split_meta.get(split_name)
        if meta_df is None:
            continue
        if split_name not in predictions_by_split or split_name not in targets_by_split:
            continue
        payloads[split_name] = {
            "meta": pd.DataFrame(meta_df).copy(),
            "y_true_raw": _to_numpy_2d(targets_by_split[split_name]),
            "y_pred_raw": _to_numpy_2d(predictions_by_split[split_name]),
        }
        included.append(split_name)
    return payloads, included


def build_artifact_index_payload(
    *,
    model_family: str,
    results_path: str | Path,
    run_manifest_path: str | Path,
    metrics_long_path: str | Path,
    config_snapshot_path: str | Path,
    prediction_sidecar_path: str | Path | None,
    model_artifact_path: str | Path | None = None,
    extra_artifact_paths: Mapping[str, str | Path | None] | None = None,
    interpretability_artifact_paths: Mapping[str, str | Path | None] | None = None,
    interpretability_status_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = load_f7_artifact_persistence_contract()
    interpretability_block = build_interpretability_status_block()
    artifact_paths: dict[str, str | None] = {
        "results_yaml": str(Path(results_path)),
        "run_manifest_json": str(Path(run_manifest_path)),
        "metrics_long_csv": str(Path(metrics_long_path)),
        "config_snapshot": str(Path(config_snapshot_path)),
        "predictions_eval_raw_csv_gz": None if prediction_sidecar_path is None else str(Path(prediction_sidecar_path)),
        "model_artifact": None if model_artifact_path is None else str(Path(model_artifact_path)),
        "interpretability_summary_json": None,
        "input_feature_influence_global_csv": None,
        "input_feature_influence_per_class_csv": None,
        "feature_influence_global_csv": None,
        "feature_influence_per_class_csv": None,
        "top_features_global_csv": None,
        "top_features_per_class_csv": None,
        "latent_feature_influence_global_csv": None,
        "latent_feature_influence_per_class_csv": None,
        "flowpre_projection_manifest_json": None,
        "flowpre_projection_cache_path": None,
        "xgb_interpretability_summary_json": None,
        "xgb_shap_feature_influence_global_csv": None,
        "xgb_shap_feature_influence_per_class_csv": None,
        "xgb_shap_top_features_global_csv": None,
        "xgb_shap_top_features_per_class_csv": None,
        "xgb_perturbation_feature_influence_global_csv": None,
        "xgb_perturbation_feature_influence_per_class_csv": None,
        "xgb_perturbation_top_features_global_csv": None,
        "xgb_perturbation_top_features_per_class_csv": None,
    }
    if interpretability_artifact_paths:
        for key, value in interpretability_artifact_paths.items():
            artifact_paths[str(key)] = None if value is None else str(Path(value))
    if extra_artifact_paths:
        for key, value in extra_artifact_paths.items():
            artifact_paths[str(key)] = None if value is None else str(Path(value))

    artifact_availability = {key: value is not None for key, value in artifact_paths.items()}
    interpretability_payload = dict(interpretability_block)
    if interpretability_status_override:
        interpretability_payload.update(dict(interpretability_status_override))
    return {
        "artifact_policy_id": str(contract.get("artifact_policy_id")),
        "artifact_tier": "per_run",
        "prediction_sidecar_schema_version": int(contract.get("prediction_sidecar_schema_version", 1)),
        "model_artifact_policy": resolve_model_artifact_policy(model_family),
        "artifact_availability": artifact_availability,
        "artifact_paths": artifact_paths,
        "interpretability_status": {
            **interpretability_payload,
            "family_specific_implementation_pending": bool(
                interpretability_payload.get(
                    "family_specific_implementation_pending",
                    dict(contract.get("interpretability_policy") or {}).get(
                        "family_specific_implementation_pending",
                        True,
                    ),
                )
            ),
        },
        **interpretability_payload,
    }
