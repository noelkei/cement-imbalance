from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from data.utils import ROOT_PATH
from evaluation.mlp_interpretability import (
    aggregate_effect_matrices,
    build_top_k_views,
    compute_class_conditioned_feature_means,
)


_XGB_INTERPRETABILITY_CONTRACT_PATH = Path(ROOT_PATH) / "config" / "f7_xgb_interpretability_contract_v1.yaml"


@lru_cache(maxsize=1)
def load_f7_xgb_interpretability_contract() -> dict[str, Any]:
    with open(_XGB_INTERPRETABILITY_CONTRACT_PATH, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    contract = payload.get("xgb_interpretability_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            f"Invalid XGBoost interpretability contract payload in {_XGB_INTERPRETABILITY_CONTRACT_PATH}"
        )
    return contract


def _contract() -> dict[str, Any]:
    return load_f7_xgb_interpretability_contract()


def _resolve_required_splits(run_mode: str) -> list[str]:
    contract = _contract()
    split_cfg = dict(contract.get("required_splits_by_run_mode") or {})
    resolved = split_cfg.get(str(run_mode))
    if not isinstance(resolved, list) or not resolved:
        raise ValueError(f"Unsupported run mode for XGBoost interpretability contract: {run_mode}")
    return [str(split_name) for split_name in resolved]


def _baseline_matrix_for_samples(
    class_codes: np.ndarray,
    class_feature_means: Mapping[int, np.ndarray],
) -> np.ndarray:
    rows = []
    for class_code in np.asarray(class_codes, dtype=np.int64):
        if int(class_code) not in class_feature_means:
            raise KeyError(f"Missing class-conditioned baseline for class {int(class_code)}")
        rows.append(np.asarray(class_feature_means[int(class_code)], dtype=np.float32))
    return np.stack(rows, axis=0)


def compute_xgb_feature_delta_matrix(
    *,
    model: Any,
    x_eval: np.ndarray,
    class_codes: np.ndarray,
    y_pred_raw: np.ndarray,
    class_feature_means: Mapping[int, np.ndarray],
    best_iteration: int | None,
    chunk_size: int,
) -> np.ndarray:
    x_eval = np.asarray(x_eval, dtype=np.float32)
    class_codes = np.asarray(class_codes, dtype=np.int64)
    y_pred_raw = np.asarray(y_pred_raw, dtype=np.float32)
    if y_pred_raw.ndim == 1:
        y_pred_raw = y_pred_raw.reshape(-1, 1)
    if y_pred_raw.shape[1] != 1:
        raise ValueError("Canonical F7 XGBoost perturbation currently supports single-target output only.")

    baseline_by_sample = _baseline_matrix_for_samples(class_codes, class_feature_means)
    original_raw = y_pred_raw.reshape(-1, 1)

    n_samples, n_features = x_eval.shape
    delta_matrix = np.empty((n_samples, n_features), dtype=np.float32)
    iteration_range = None if best_iteration is None else (0, int(best_iteration))

    for start in range(0, n_features, chunk_size):
        feature_indices = list(range(start, min(start + chunk_size, n_features)))
        feature_count = len(feature_indices)
        perturb = np.repeat(x_eval[None, :, :], feature_count, axis=0)
        for local_idx, feature_idx in enumerate(feature_indices):
            perturb[local_idx, :, feature_idx] = baseline_by_sample[:, feature_idx]
        reshaped = perturb.reshape(-1, n_features)
        if iteration_range is None:
            perturbed_raw = model.predict(reshaped).reshape(feature_count, n_samples, 1)
        else:
            perturbed_raw = model.predict(reshaped, iteration_range=iteration_range).reshape(feature_count, n_samples, 1)
        signed_delta = original_raw.reshape(1, n_samples, 1) - perturbed_raw
        delta_matrix[:, feature_indices] = signed_delta[:, :, 0].transpose(1, 0)

    return delta_matrix


def compute_xgb_shap_values(
    *,
    model: Any,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, float]:
    import shap  # type: ignore

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_eval)
    shap_array = np.asarray(shap_values, dtype=np.float32)
    if shap_array.ndim != 2:
        raise ValueError(f"Expected 2D SHAP array for single-target regression, got shape {shap_array.shape}")
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = np.asarray(expected_value).reshape(-1)[0]
    return shap_array, float(expected_value)


def _validate_layer_tables(
    *,
    global_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    top_global: pd.DataFrame,
    top_per_class: pd.DataFrame,
    metric_prefix: str,
) -> list[str]:
    missing_items: list[str] = []
    if global_df.empty:
        missing_items.append(f"{metric_prefix}.feature_influence_global")
    if per_class_df.empty:
        missing_items.append(f"{metric_prefix}.feature_influence_per_class")
    if top_global.empty:
        missing_items.append(f"{metric_prefix}.top_features_global")
    if top_per_class.empty:
        missing_items.append(f"{metric_prefix}.top_features_per_class")

    numeric_cols = [
        f"mean_abs_{metric_prefix}",
        f"mean_signed_{metric_prefix}",
        f"sum_abs_{metric_prefix}",
        f"std_abs_{metric_prefix}",
        f"median_abs_{metric_prefix}",
        f"p90_abs_{metric_prefix}",
        f"p95_abs_{metric_prefix}",
        f"stderr_abs_{metric_prefix}",
        "share_abs_importance",
    ]
    for col in numeric_cols:
        if col in global_df.columns and global_df[col].isna().any():
            missing_items.append(f"{metric_prefix}.feature_influence_global_nan.{col}")
        if col in per_class_df.columns and per_class_df[col].isna().any():
            missing_items.append(f"{metric_prefix}.feature_influence_per_class_nan.{col}")
        if col in top_global.columns and top_global[col].isna().any():
            missing_items.append(f"{metric_prefix}.top_features_global_nan.{col}")
        if col in top_per_class.columns and top_per_class[col].isna().any():
            missing_items.append(f"{metric_prefix}.top_features_per_class_nan.{col}")

    if "share_abs_importance" in global_df.columns and not global_df.empty:
        for split_name, split_df in global_df.groupby("split", sort=False):
            share_sum = float(split_df["share_abs_importance"].sum())
            if not (np.isclose(share_sum, 1.0, atol=1e-5) or np.isclose(share_sum, 0.0, atol=1e-8)):
                missing_items.append(f"{metric_prefix}.feature_influence_global_share_sum.{split_name}")
    if "share_abs_importance" in per_class_df.columns and not per_class_df.empty:
        for (split_name, class_code), split_df in per_class_df.groupby(["split", "type"], sort=False):
            share_sum = float(split_df["share_abs_importance"].sum())
            if not (np.isclose(share_sum, 1.0, atol=1e-5) or np.isclose(share_sum, 0.0, atol=1e-8)):
                missing_items.append(f"{metric_prefix}.feature_influence_per_class_share_sum.{split_name}.{class_code}")
    return missing_items


def _rename_metric_columns(df: pd.DataFrame, *, layer_prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.rename(
        columns={
            "mean_abs_delta_pred_raw": f"mean_abs_{layer_prefix}",
            "mean_signed_delta_pred_raw": f"mean_signed_{layer_prefix}",
            "sum_abs_delta_pred_raw": f"sum_abs_{layer_prefix}",
            "std_abs_delta_pred_raw": f"std_abs_{layer_prefix}",
            "median_abs_delta_pred_raw": f"median_abs_{layer_prefix}",
            "p90_abs_delta_pred_raw": f"p90_abs_{layer_prefix}",
            "p95_abs_delta_pred_raw": f"p95_abs_{layer_prefix}",
            "stderr_abs_delta_pred_raw": f"stderr_abs_{layer_prefix}",
        },
        inplace=True,
    )
    return df


def compute_and_persist_xgb_interpretability(
    *,
    model: Any,
    out_dir: str | Path,
    feature_names: list[str],
    x_train_matrix: np.ndarray,
    c_train: np.ndarray,
    split_arrays: Mapping[str, tuple[np.ndarray, np.ndarray] | None],
    predictions_by_split: Mapping[str, np.ndarray],
    run_mode: str,
    best_iteration: int | None,
    target_names: list[str],
) -> dict[str, Any]:
    if len(target_names) != 1:
        raise ValueError("Canonical F7 XGBoost interpretability currently supports exactly one target name.")

    contract = _contract()
    required_splits = _resolve_required_splits(run_mode)
    shap_cfg = dict(contract.get("shap_layer") or {})
    perturb_cfg = dict(contract.get("perturbation_layer") or {})
    chunk_size = int(perturb_cfg.get("feature_chunk_size", 16))
    class_feature_means = compute_class_conditioned_feature_means(x_train_matrix, c_train)

    started_total = time.perf_counter()
    shap_started = time.perf_counter()
    shap_global_frames: list[pd.DataFrame] = []
    shap_per_class_frames: list[pd.DataFrame] = []
    shap_expected_value_by_split: dict[str, float] = {}
    available_splits: list[str] = []

    for split_name in required_splits:
        split_payload = split_arrays.get(split_name)
        if split_payload is None:
            raise ValueError(f"Missing split payload required for XGBoost interpretability: {split_name}")
        x_eval, c_eval = split_payload
        shap_values, expected_value = compute_xgb_shap_values(model=model, x_eval=x_eval)
        shap_expected_value_by_split[str(split_name)] = float(expected_value)
        shap_global_df, shap_per_class_df = aggregate_effect_matrices(
            split_name=split_name,
            item_names=feature_names,
            class_codes=c_eval,
            signed_matrix=shap_values,
            abs_matrix=np.abs(shap_values),
            name_col="feature_name",
            feature_space_kind="xgb_model_input_feature",
            projection_status="direct_model_input",
        )
        shap_global_frames.append(_rename_metric_columns(shap_global_df, layer_prefix="shap"))
        shap_per_class_frames.append(_rename_metric_columns(shap_per_class_df, layer_prefix="shap"))
        available_splits.append(str(split_name))
    shap_runtime_s = round(time.perf_counter() - shap_started, 3)

    perturb_started = time.perf_counter()
    perturb_global_frames: list[pd.DataFrame] = []
    perturb_per_class_frames: list[pd.DataFrame] = []
    for split_name in required_splits:
        split_payload = split_arrays.get(split_name)
        if split_payload is None:
            raise ValueError(f"Missing split payload required for XGBoost perturbation: {split_name}")
        x_eval, c_eval = split_payload
        y_pred_raw = predictions_by_split.get(split_name)
        if y_pred_raw is None:
            raise ValueError(f"Missing predictions required for XGBoost perturbation: {split_name}")
        signed_delta_matrix = compute_xgb_feature_delta_matrix(
            model=model,
            x_eval=x_eval,
            class_codes=c_eval,
            y_pred_raw=y_pred_raw,
            class_feature_means=class_feature_means,
            best_iteration=best_iteration,
            chunk_size=chunk_size,
        )
        perturb_global_df, perturb_per_class_df = aggregate_effect_matrices(
            split_name=split_name,
            item_names=feature_names,
            class_codes=c_eval,
            signed_matrix=signed_delta_matrix,
            abs_matrix=np.abs(signed_delta_matrix),
            name_col="feature_name",
            feature_space_kind="xgb_model_input_feature",
            projection_status="direct_model_input",
        )
        perturb_global_frames.append(perturb_global_df)
        perturb_per_class_frames.append(perturb_per_class_df)
    perturb_runtime_s = round(time.perf_counter() - perturb_started, 3)

    shap_global = pd.concat(shap_global_frames, ignore_index=True)
    shap_per_class = pd.concat(shap_per_class_frames, ignore_index=True)
    perturb_global = pd.concat(perturb_global_frames, ignore_index=True)
    perturb_per_class = pd.concat(perturb_per_class_frames, ignore_index=True)

    top_k = 10
    shap_top_global, shap_top_per_class = build_top_k_views(
        global_df=shap_global,
        per_class_df=shap_per_class,
        top_k=top_k,
    )
    perturb_top_global, perturb_top_per_class = build_top_k_views(
        global_df=perturb_global,
        per_class_df=perturb_per_class,
        top_k=top_k,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "xgb_interpretability_summary_json": out_dir / "xgb_interpretability_summary.json",
        "interpretability_summary_json": out_dir / "xgb_interpretability_summary.json",
        "xgb_shap_feature_influence_global_csv": out_dir / "xgb_shap_feature_influence_global.csv",
        "xgb_shap_feature_influence_per_class_csv": out_dir / "xgb_shap_feature_influence_per_class.csv",
        "xgb_shap_top_features_global_csv": out_dir / "xgb_shap_top_features_global.csv",
        "xgb_shap_top_features_per_class_csv": out_dir / "xgb_shap_top_features_per_class.csv",
        "xgb_perturbation_feature_influence_global_csv": out_dir / "xgb_perturbation_feature_influence_global.csv",
        "xgb_perturbation_feature_influence_per_class_csv": out_dir / "xgb_perturbation_feature_influence_per_class.csv",
        "xgb_perturbation_top_features_global_csv": out_dir / "xgb_perturbation_top_features_global.csv",
        "xgb_perturbation_top_features_per_class_csv": out_dir / "xgb_perturbation_top_features_per_class.csv",
    }
    shap_global.to_csv(artifact_paths["xgb_shap_feature_influence_global_csv"], index=False)
    shap_per_class.to_csv(artifact_paths["xgb_shap_feature_influence_per_class_csv"], index=False)
    shap_top_global.to_csv(artifact_paths["xgb_shap_top_features_global_csv"], index=False)
    shap_top_per_class.to_csv(artifact_paths["xgb_shap_top_features_per_class_csv"], index=False)
    perturb_global.to_csv(artifact_paths["xgb_perturbation_feature_influence_global_csv"], index=False)
    perturb_per_class.to_csv(artifact_paths["xgb_perturbation_feature_influence_per_class_csv"], index=False)
    perturb_top_global.to_csv(artifact_paths["xgb_perturbation_top_features_global_csv"], index=False)
    perturb_top_per_class.to_csv(artifact_paths["xgb_perturbation_top_features_per_class_csv"], index=False)

    interpretability_runtime_s_total = round(time.perf_counter() - started_total, 3)
    summary = {
        "xgb_interpretability_contract_id": str(contract.get("contract_id")),
        "required_layers": ["shap", "perturbation"],
        "available_splits": list(available_splits),
        "feature_space_kind_primary": str((contract.get("feature_surface_policy") or {}).get("feature_space_kind_primary")),
        "type_feature_policy": str((contract.get("feature_surface_policy") or {}).get("type_feature_policy")),
        "shap_expected_value_by_split": dict(shap_expected_value_by_split),
        "ranking_metric_primary_by_layer": {
            "shap": str(shap_cfg.get("ranking_metric_primary")),
            "perturbation": str(perturb_cfg.get("ranking_metric_primary")),
        },
        "interpretability_runtime_s_total": interpretability_runtime_s_total,
        "interpretability_runtime_s_by_layer": {
            "shap": shap_runtime_s,
            "perturbation": perturb_runtime_s,
        },
        "target_names": list(target_names),
        "artifact_refs": {key: str(value) for key, value in artifact_paths.items()},
    }
    with open(artifact_paths["xgb_interpretability_summary_json"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    missing_items: list[str] = []
    for split_name in required_splits:
        if split_name not in available_splits:
            missing_items.append(f"split.{split_name}")
        if split_name not in shap_expected_value_by_split:
            missing_items.append(f"shap.expected_value.{split_name}")
    missing_items.extend(
        _validate_layer_tables(
            global_df=shap_global,
            per_class_df=shap_per_class,
            top_global=shap_top_global,
            top_per_class=shap_top_per_class,
            metric_prefix="shap",
        )
    )
    missing_items.extend(
        _validate_layer_tables(
            global_df=perturb_global,
            per_class_df=perturb_per_class,
            top_global=perturb_top_global,
            top_per_class=perturb_top_per_class,
            metric_prefix="delta_pred_raw",
        )
    )

    validation = {
        "xgb_interpretability_contract_id": str(contract.get("contract_id")),
        "run_mode": str(run_mode),
        "required_splits_resolved": required_splits,
        "required_scopes": [str(item) for item in (contract.get("required_scopes") or [])],
        "required_layers": ["shap", "perturbation"],
        "available_splits": list(available_splits),
        "feature_space_kind_primary": str((contract.get("feature_surface_policy") or {}).get("feature_space_kind_primary")),
        "campaign_valid_interpretability": len(missing_items) == 0,
        "missing_items": missing_items,
    }
    return {
        "summary": summary,
        "validation": validation,
        "artifact_paths": artifact_paths,
        "shap_global": shap_global,
        "shap_per_class": shap_per_class,
        "shap_top_global": shap_top_global,
        "shap_top_per_class": shap_top_per_class,
        "perturb_global": perturb_global,
        "perturb_per_class": perturb_per_class,
        "perturb_top_global": perturb_top_global,
        "perturb_top_per_class": perturb_top_per_class,
    }
