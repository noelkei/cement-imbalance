from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from data.utils import ROOT_PATH
from evaluation.compositional_projection import (
    build_final_semantic_surface_spec,
    distribute_event_deltas_to_surfaces,
)
from evaluation.flowpre_projection import (
    load_flowpre_decoder_runtime,
    load_f7_mlp_interpretability_contract as _load_contract_from_projection,
    resolve_flowpre_promotion_manifest_from_dataset_manifest,
    resolve_or_build_flowpre_projection_cache,
)
from evaluation.metrics import inverse_transform_tensor


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


def _contract() -> dict[str, Any]:
    # Reuse the same cached object if the projection module already loaded it.
    try:
        return _load_contract_from_projection()
    except Exception:
        return load_f7_mlp_interpretability_contract()


def _inverse_predictions_to_raw(y_pred_native: np.ndarray, y_scaler: Any | None, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(y_pred_native, dtype=torch.float32, device=device)
    if y_scaler is None:
        restored = tensor
    else:
        restored = inverse_transform_tensor(tensor, y_scaler, device=device)
    restored_np = restored.detach().cpu().numpy()
    if restored_np.ndim == 1:
        restored_np = restored_np.reshape(-1, 1)
    return restored_np


def compute_class_conditioned_feature_means(
    x_train: np.ndarray,
    class_codes: np.ndarray,
) -> dict[int, np.ndarray]:
    x_train = np.asarray(x_train, dtype=np.float32)
    class_codes = np.asarray(class_codes, dtype=np.int64)
    means: dict[int, np.ndarray] = {}
    for class_code in sorted(np.unique(class_codes).tolist()):
        means[int(class_code)] = x_train[class_codes == int(class_code)].mean(axis=0).astype(np.float32, copy=False)
    return means


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


def _build_standard_perturbed_inputs(
    *,
    x_eval: np.ndarray,
    class_codes: np.ndarray,
    class_feature_means: Mapping[int, np.ndarray],
    feature_indices: list[int],
) -> np.ndarray:
    x_eval = np.asarray(x_eval, dtype=np.float32)
    baseline_by_sample = _baseline_matrix_for_samples(class_codes, class_feature_means)
    perturb = np.repeat(x_eval[None, :, :], len(feature_indices), axis=0)
    for local_idx, feature_idx in enumerate(feature_indices):
        perturb[local_idx, :, feature_idx] = baseline_by_sample[:, feature_idx]
    return perturb


def _decode_flowpre_input_space(
    *,
    decoder_model: Any,
    device: torch.device,
    latent_batch: np.ndarray,
    class_codes: np.ndarray,
) -> np.ndarray:
    latent_tensor = torch.as_tensor(latent_batch, dtype=torch.float32, device=device)
    class_tensor = torch.as_tensor(class_codes, dtype=torch.long, device=device)
    with torch.no_grad():
        decoded = decoder_model.inverse(latent_tensor, class_tensor)[0].detach().cpu().numpy()
    return np.asarray(decoded, dtype=np.float32)


def compute_mlp_feature_delta_matrix(
    *,
    model: Any,
    x_eval: np.ndarray,
    class_codes: np.ndarray,
    y_pred_native: np.ndarray,
    class_feature_means: Mapping[int, np.ndarray],
    y_scaler: Any | None,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    x_eval = np.asarray(x_eval, dtype=np.float32)
    class_codes = np.asarray(class_codes, dtype=np.int64)
    y_pred_native = np.asarray(y_pred_native, dtype=np.float32)
    if y_pred_native.ndim == 1:
        y_pred_native = y_pred_native.reshape(-1, 1)
    if y_pred_native.shape[1] != 1:
        raise ValueError(
            "Canonical F7 MLP interpretability currently supports single-target output only."
        )

    baseline_by_sample = _baseline_matrix_for_samples(class_codes, class_feature_means)
    original_raw = _inverse_predictions_to_raw(y_pred_native, y_scaler, device=device).reshape(-1, 1)

    n_samples, n_features = x_eval.shape
    delta_matrix = np.empty((n_samples, n_features), dtype=np.float32)
    class_tensor_base = torch.as_tensor(class_codes, dtype=torch.long, device=device)

    for start in range(0, n_features, chunk_size):
        feature_indices = list(range(start, min(start + chunk_size, n_features)))
        feature_count = len(feature_indices)
        perturb = np.repeat(x_eval[None, :, :], feature_count, axis=0)
        for local_idx, feature_idx in enumerate(feature_indices):
            perturb[local_idx, :, feature_idx] = baseline_by_sample[:, feature_idx]

        perturb_tensor = torch.as_tensor(perturb.reshape(-1, n_features), dtype=torch.float32, device=device)
        class_tensor = class_tensor_base.repeat(feature_count)
        with torch.no_grad():
            perturbed_native = model(perturb_tensor, class_tensor).detach().cpu().numpy()
        if perturbed_native.ndim == 1:
            perturbed_native = perturbed_native.reshape(-1, 1)
        perturbed_raw = _inverse_predictions_to_raw(perturbed_native, y_scaler, device=device)
        perturbed_raw = perturbed_raw.reshape(feature_count, n_samples, 1)

        # Signed delta definition frozen by contract:
        # original prediction minus perturbed prediction.
        signed_delta = original_raw.reshape(1, n_samples, 1) - perturbed_raw
        delta_matrix[:, feature_indices] = signed_delta[:, :, 0].transpose(1, 0)

    return delta_matrix


def _rank_descending(df: pd.DataFrame, metric_col: str, group_cols: list[str], name_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    ranked_frames: list[pd.DataFrame] = []
    for _, group_df in df.groupby(group_cols, sort=False, dropna=False):
        ordered = group_df.sort_values(
            by=[metric_col, name_col],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ordered["rank_abs"] = np.arange(1, len(ordered) + 1, dtype=int)
        ranked_frames.append(ordered)
    return pd.concat(ranked_frames, ignore_index=True)


def aggregate_effect_matrices(
    *,
    split_name: str,
    item_names: list[str],
    class_codes: np.ndarray,
    signed_matrix: np.ndarray,
    abs_matrix: np.ndarray,
    name_col: str,
    feature_space_kind: str | None = None,
    projection_status: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_codes = np.asarray(class_codes, dtype=np.int64)
    signed_matrix = np.asarray(signed_matrix, dtype=np.float32)
    abs_matrix = np.asarray(abs_matrix, dtype=np.float32)
    if signed_matrix.shape != abs_matrix.shape:
        raise ValueError("Signed and absolute effect matrices must have the same shape.")
    if signed_matrix.shape[1] != len(item_names):
        raise ValueError("Item name count does not match effect matrix width.")

    global_rows: list[dict[str, Any]] = []
    for feature_idx, item_name in enumerate(item_names):
        abs_values = abs_matrix[:, feature_idx].astype(np.float64, copy=False)
        mean_abs = float(abs_values.mean())
        sum_abs = float(abs_values.sum())
        row = {
            "split": str(split_name),
            name_col: str(item_name),
            "mean_abs_delta_pred_raw": mean_abs,
            "mean_signed_delta_pred_raw": float(signed_matrix[:, feature_idx].mean()),
            "sum_abs_delta_pred_raw": sum_abs,
            "std_abs_delta_pred_raw": float(abs_values.std(ddof=0)),
            "median_abs_delta_pred_raw": float(np.median(abs_values)),
            "p90_abs_delta_pred_raw": float(np.percentile(abs_values, 90)),
            "p95_abs_delta_pred_raw": float(np.percentile(abs_values, 95)),
            "stderr_abs_delta_pred_raw": float(abs_values.std(ddof=0) / np.sqrt(max(abs_values.size, 1))),
            "n_samples": int(signed_matrix.shape[0]),
        }
        if feature_space_kind is not None:
            row["feature_space_kind"] = str(feature_space_kind)
            row["projection_status"] = str(projection_status or "unknown")
        global_rows.append(row)
    global_df = pd.DataFrame.from_records(global_rows)
    if not global_df.empty:
        global_df["share_abs_importance"] = (
            global_df["sum_abs_delta_pred_raw"]
            / np.clip(float(global_df["sum_abs_delta_pred_raw"].sum()), 1e-12, None)
        )
    global_df = _rank_descending(
        global_df,
        metric_col="mean_abs_delta_pred_raw",
        group_cols=["split"],
        name_col=name_col,
    )

    per_class_rows: list[dict[str, Any]] = []
    for class_code in sorted(np.unique(class_codes).tolist()):
        mask = class_codes == int(class_code)
        for feature_idx, item_name in enumerate(item_names):
            abs_values = abs_matrix[mask, feature_idx].astype(np.float64, copy=False)
            mean_abs = float(abs_values.mean())
            sum_abs = float(abs_values.sum())
            row = {
                "split": str(split_name),
                "type": int(class_code),
                name_col: str(item_name),
                "mean_abs_delta_pred_raw": mean_abs,
                "mean_signed_delta_pred_raw": float(signed_matrix[mask, feature_idx].mean()),
                "sum_abs_delta_pred_raw": sum_abs,
                "std_abs_delta_pred_raw": float(abs_values.std(ddof=0)),
                "median_abs_delta_pred_raw": float(np.median(abs_values)),
                "p90_abs_delta_pred_raw": float(np.percentile(abs_values, 90)),
                "p95_abs_delta_pred_raw": float(np.percentile(abs_values, 95)),
                "stderr_abs_delta_pred_raw": float(abs_values.std(ddof=0) / np.sqrt(max(abs_values.size, 1))),
                "n_samples": int(mask.sum()),
            }
            if feature_space_kind is not None:
                row["feature_space_kind"] = str(feature_space_kind)
                row["projection_status"] = str(projection_status or "unknown")
            per_class_rows.append(row)
    per_class_df = pd.DataFrame.from_records(per_class_rows)
    if not per_class_df.empty:
        per_class_df["share_abs_importance"] = per_class_df.groupby(
            ["split", "type"], sort=False
        )["sum_abs_delta_pred_raw"].transform(
            lambda col: col / np.clip(float(col.sum()), 1e-12, None)
        )
    per_class_df = _rank_descending(
        per_class_df,
        metric_col="mean_abs_delta_pred_raw",
        group_cols=["split", "type"],
        name_col=name_col,
    )
    return global_df, per_class_df


def build_top_k_views(
    *,
    global_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_global = global_df[global_df["rank_abs"] <= int(top_k)].copy()
    top_global = top_global.sort_values(by=["split", "rank_abs"], kind="mergesort").reset_index(drop=True)

    top_per_class = per_class_df[per_class_df["rank_abs"] <= int(top_k)].copy()
    top_per_class = top_per_class.sort_values(by=["split", "type", "rank_abs"], kind="mergesort").reset_index(
        drop=True
    )
    return top_global, top_per_class


def _projection_weights_by_class(
    projection_table: pd.DataFrame,
    *,
    latent_names: list[str],
) -> tuple[dict[int, np.ndarray], list[str]]:
    if projection_table.empty:
        raise ValueError("Projection table is empty.")
    semantic_feature_names = projection_table["semantic_feature"].drop_duplicates().tolist()
    weights: dict[int, np.ndarray] = {}
    for class_code in sorted(projection_table["type"].unique().tolist()):
        class_df = projection_table[projection_table["type"] == int(class_code)].copy()
        pivot = class_df.pivot(index="latent_name", columns="semantic_feature", values="weight_norm")
        pivot = pivot.reindex(index=latent_names, columns=semantic_feature_names).fillna(0.0)
        weights[int(class_code)] = pivot.to_numpy(dtype=np.float32)
    return weights, semantic_feature_names


def project_latent_effects_to_semantic_space(
    *,
    signed_latent_matrix: np.ndarray,
    class_codes: np.ndarray,
    latent_names: list[str],
    projection_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    signed_latent_matrix = np.asarray(signed_latent_matrix, dtype=np.float32)
    class_codes = np.asarray(class_codes, dtype=np.int64)
    weight_matrices, semantic_feature_names = _projection_weights_by_class(
        projection_table,
        latent_names=latent_names,
    )
    n_samples = signed_latent_matrix.shape[0]
    semantic_count = len(semantic_feature_names)
    signed_semantic = np.zeros((n_samples, semantic_count), dtype=np.float32)
    abs_semantic = np.zeros((n_samples, semantic_count), dtype=np.float32)
    for class_code, weight_matrix in weight_matrices.items():
        mask = class_codes == int(class_code)
        if not mask.any():
            continue
        signed_slice = signed_latent_matrix[mask]
        signed_semantic[mask] = signed_slice @ weight_matrix
        abs_semantic[mask] = np.abs(signed_slice) @ weight_matrix
    return signed_semantic, abs_semantic, semantic_feature_names


def _resolve_required_splits(run_mode: str) -> list[str]:
    contract = _contract()
    split_cfg = dict(contract.get("required_splits_by_run_mode") or {})
    resolved = split_cfg.get(str(run_mode))
    if not isinstance(resolved, list) or not resolved:
        raise ValueError(f"Unsupported run mode for MLP interpretability contract: {run_mode}")
    return [str(split_name) for split_name in resolved]


def _validate_final_semantic_surface(
    *,
    semantic_global: pd.DataFrame,
    semantic_per_class: pd.DataFrame,
    top_global: pd.DataFrame,
    top_per_class: pd.DataFrame,
    exclude_prefixes: list[str],
) -> list[str]:
    missing_items: list[str] = []
    if semantic_global.empty:
        missing_items.append("feature_influence_global")
    if semantic_per_class.empty:
        missing_items.append("feature_influence_per_class")
    if top_global.empty:
        missing_items.append("top_features_global")
    if top_per_class.empty:
        missing_items.append("top_features_per_class")

    numeric_cols = [
        "mean_abs_delta_pred_raw",
        "mean_signed_delta_pred_raw",
        "sum_abs_delta_pred_raw",
        "std_abs_delta_pred_raw",
        "median_abs_delta_pred_raw",
        "p90_abs_delta_pred_raw",
        "p95_abs_delta_pred_raw",
        "stderr_abs_delta_pred_raw",
        "share_abs_importance",
    ]
    for col in numeric_cols:
        if col in semantic_global.columns and semantic_global[col].isna().any():
            missing_items.append(f"feature_influence_global_nan.{col}")
        if col in semantic_per_class.columns and semantic_per_class[col].isna().any():
            missing_items.append(f"feature_influence_per_class_nan.{col}")
        if col in top_global.columns and top_global[col].isna().any():
            missing_items.append(f"top_features_global_nan.{col}")
        if col in top_per_class.columns and top_per_class[col].isna().any():
            missing_items.append(f"top_features_per_class_nan.{col}")

    for prefix in exclude_prefixes:
        prefix = str(prefix)
        if not prefix:
            continue
        if not semantic_global.empty and semantic_global["feature_name"].astype(str).str.startswith(prefix).any():
            missing_items.append(f"final_surface_contains_prefix.{prefix}")
        if not semantic_per_class.empty and semantic_per_class["feature_name"].astype(str).str.startswith(prefix).any():
            missing_items.append(f"final_surface_per_class_contains_prefix.{prefix}")
    if "share_abs_importance" in semantic_global.columns and not semantic_global.empty:
        for split_name, split_df in semantic_global.groupby("split", sort=False):
            share_sum = float(split_df["share_abs_importance"].sum())
            if not (np.isclose(share_sum, 1.0, atol=1e-5) or np.isclose(share_sum, 0.0, atol=1e-8)):
                missing_items.append(f"final_surface_share_sum.{split_name}")
    if "share_abs_importance" in semantic_per_class.columns and not semantic_per_class.empty:
        for (split_name, class_code), split_df in semantic_per_class.groupby(["split", "type"], sort=False):
            share_sum = float(split_df["share_abs_importance"].sum())
            if not (np.isclose(share_sum, 1.0, atol=1e-5) or np.isclose(share_sum, 0.0, atol=1e-8)):
                missing_items.append(f"final_surface_per_class_share_sum.{split_name}.{class_code}")
    return missing_items


def compute_and_persist_mlp_interpretability(
    *,
    model: Any,
    device: torch.device,
    out_dir: str | Path,
    dataset_manifest_payload: Mapping[str, Any],
    dataset_level_axes: Mapping[str, Any] | None,
    feature_names: list[str],
    x_train: np.ndarray,
    c_train: np.ndarray,
    split_arrays: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    predictions_by_split: Mapping[str, np.ndarray],
    y_scaler: Any | None,
    target_names: list[str],
    run_mode: str,
    projection_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    contract = _contract()
    if len(target_names) != 1:
        raise ValueError(
            "Canonical F7 MLP interpretability currently supports exactly one target name."
        )

    started_at = time.perf_counter()
    perturb_cfg = dict(contract.get("perturbation") or {})
    chunk_size = int(perturb_cfg.get("feature_chunk_size", 16))
    required_splits = _resolve_required_splits(run_mode)
    class_feature_means = compute_class_conditioned_feature_means(x_train, c_train)
    dataset_axes = dict(dataset_level_axes or {})
    x_transform = str(dataset_axes.get("x_transform", "unknown"))
    flowpre_required = set(
        str(item) for item in (contract.get("feature_space_policy") or {}).get(
            "flowpre_latent_projection_required_for_x_transform", []
        )
    )
    uses_flowpre_projection = x_transform in flowpre_required

    projection_bundle: dict[str, Any] | None = None
    projection_status = "direct_semantic"
    feature_space_kind_primary = "semantic_final_surface"
    latent_global_frames: list[pd.DataFrame] = []
    latent_per_class_frames: list[pd.DataFrame] = []
    input_global_frames: list[pd.DataFrame] = []
    input_per_class_frames: list[pd.DataFrame] = []
    semantic_global_frames: list[pd.DataFrame] = []
    semantic_per_class_frames: list[pd.DataFrame] = []

    input_feature_names = [str(name) for name in feature_names]
    final_surface_spec = build_final_semantic_surface_spec(input_feature_names)
    compositional_groups_present = bool(final_surface_spec.groups)
    compositional_projection_status = "projected_ilr_groups" if compositional_groups_present else "not_needed"
    if uses_flowpre_projection:
        promotion_manifest_path = resolve_flowpre_promotion_manifest_from_dataset_manifest(dict(dataset_manifest_payload))
        if promotion_manifest_path is None:
            raise ValueError(
                "FlowPre-based MLP interpretability requires a dataset manifest with upstream FlowPre promotion manifest."
            )
        projection_bundle = resolve_or_build_flowpre_projection_cache(
            promotion_manifest_path=promotion_manifest_path,
            projection_cache_root=projection_cache_root,
            device="cpu",
        )
        projection_status = "projected_from_flowpre_cache"
        flowpre_runtime = load_flowpre_decoder_runtime(promotion_manifest_path=promotion_manifest_path, device="cpu")
        decoder_model = flowpre_runtime["decoder_model"]
        decoder_device = flowpre_runtime["device"]
        input_feature_names = [str(name) for name in flowpre_runtime["semantic_feature_names"]]
        final_surface_spec = build_final_semantic_surface_spec(input_feature_names)
        compositional_groups_present = bool(final_surface_spec.groups)
        compositional_projection_status = "projected_ilr_groups" if compositional_groups_present else "not_needed"
    else:
        decoder_model = None
        decoder_device = None

    available_splits: list[str] = []
    for split_name in required_splits:
        split_payload = split_arrays.get(split_name)
        if split_payload is None:
            raise ValueError(f"Missing split payload required for MLP interpretability: {split_name}")
        if split_name not in predictions_by_split:
            raise ValueError(f"Missing predictions required for MLP interpretability: {split_name}")
        x_eval, _y_eval_unused, c_eval = split_payload
        y_pred_native = predictions_by_split[split_name]
        signed_delta_matrix = compute_mlp_feature_delta_matrix(
            model=model,
            x_eval=x_eval,
            class_codes=c_eval,
            y_pred_native=y_pred_native,
            class_feature_means=class_feature_means,
            y_scaler=y_scaler,
            device=device,
            chunk_size=chunk_size,
        )
        available_splits.append(split_name)
        if uses_flowpre_projection:
            latent_names = [str(name) for name in feature_names]
            latent_global_df, latent_per_class_df = aggregate_effect_matrices(
                split_name=split_name,
                item_names=latent_names,
                class_codes=c_eval,
                signed_matrix=signed_delta_matrix,
                abs_matrix=np.abs(signed_delta_matrix),
                name_col="latent_name",
            )
            latent_global_frames.append(latent_global_df)
            latent_per_class_frames.append(latent_per_class_df)
            baseline_latent_by_sample = _baseline_matrix_for_samples(c_eval, class_feature_means)
            original_input = _decode_flowpre_input_space(
                decoder_model=decoder_model,
                device=decoder_device,
                latent_batch=x_eval,
                class_codes=c_eval,
            )
            final_bundle = {
                "input_signed": np.zeros((x_eval.shape[0], len(input_feature_names)), dtype=np.float32),
                "input_abs": np.zeros((x_eval.shape[0], len(input_feature_names)), dtype=np.float32),
                "final_signed": np.zeros((x_eval.shape[0], len(final_surface_spec.final_feature_names)), dtype=np.float32),
                "final_abs": np.zeros((x_eval.shape[0], len(final_surface_spec.final_feature_names)), dtype=np.float32),
            }
            for start in range(0, x_eval.shape[1], chunk_size):
                feature_indices = list(range(start, min(start + chunk_size, x_eval.shape[1])))
                feature_count = len(feature_indices)
                perturb = np.repeat(x_eval[None, :, :], feature_count, axis=0)
                for local_idx, feature_idx in enumerate(feature_indices):
                    perturb[local_idx, :, feature_idx] = baseline_latent_by_sample[:, feature_idx]
                decoded = []
                for local_idx in range(feature_count):
                    decoded.append(
                        _decode_flowpre_input_space(
                            decoder_model=decoder_model,
                            device=decoder_device,
                            latent_batch=perturb[local_idx],
                            class_codes=c_eval,
                        )
                    )
                perturbed_inputs = np.stack(decoded, axis=0)
                distributed = distribute_event_deltas_to_surfaces(
                    signed_event_deltas=signed_delta_matrix[:, feature_indices],
                    input_feature_names=input_feature_names,
                    original_input=original_input,
                    perturbed_inputs=perturbed_inputs,
                    fallback_input_indices=[],
                )
                final_bundle["input_signed"] += distributed["input_signed"]
                final_bundle["input_abs"] += distributed["input_abs"]
                final_bundle["final_signed"] += distributed["final_signed"]
                final_bundle["final_abs"] += distributed["final_abs"]

            input_global_df, input_per_class_df = aggregate_effect_matrices(
                split_name=split_name,
                item_names=input_feature_names,
                class_codes=c_eval,
                signed_matrix=final_bundle["input_signed"],
                abs_matrix=final_bundle["input_abs"],
                name_col="feature_name",
                feature_space_kind="model_input_space",
                projection_status=projection_status,
            )
            semantic_global_df, semantic_per_class_df = aggregate_effect_matrices(
                split_name=split_name,
                item_names=final_surface_spec.final_feature_names,
                class_codes=c_eval,
                signed_matrix=final_bundle["final_signed"],
                abs_matrix=final_bundle["final_abs"],
                name_col="feature_name",
                feature_space_kind=feature_space_kind_primary,
                projection_status=compositional_projection_status,
            )
            input_global_frames.append(input_global_df)
            input_per_class_frames.append(input_per_class_df)
            semantic_global_frames.append(semantic_global_df)
            semantic_per_class_frames.append(semantic_per_class_df)
        else:
            input_global_df, input_per_class_df = aggregate_effect_matrices(
                split_name=split_name,
                item_names=input_feature_names,
                class_codes=c_eval,
                signed_matrix=signed_delta_matrix,
                abs_matrix=np.abs(signed_delta_matrix),
                name_col="feature_name",
                feature_space_kind="model_input_space",
                projection_status=projection_status,
            )
            input_global_frames.append(input_global_df)
            input_per_class_frames.append(input_per_class_df)

            final_bundle = {
                "final_signed": np.zeros((x_eval.shape[0], len(final_surface_spec.final_feature_names)), dtype=np.float32),
                "final_abs": np.zeros((x_eval.shape[0], len(final_surface_spec.final_feature_names)), dtype=np.float32),
            }
            for start in range(0, x_eval.shape[1], chunk_size):
                feature_indices = list(range(start, min(start + chunk_size, x_eval.shape[1])))
                perturbed_inputs = _build_standard_perturbed_inputs(
                    x_eval=x_eval,
                    class_codes=c_eval,
                    class_feature_means=class_feature_means,
                    feature_indices=feature_indices,
                )
                distributed = distribute_event_deltas_to_surfaces(
                    signed_event_deltas=signed_delta_matrix[:, feature_indices],
                    input_feature_names=input_feature_names,
                    original_input=x_eval,
                    perturbed_inputs=perturbed_inputs,
                    fallback_input_indices=feature_indices,
                )
                final_bundle["final_signed"] += distributed["final_signed"]
                final_bundle["final_abs"] += distributed["final_abs"]
            semantic_global_df, semantic_per_class_df = aggregate_effect_matrices(
                split_name=split_name,
                item_names=final_surface_spec.final_feature_names,
                class_codes=c_eval,
                signed_matrix=final_bundle["final_signed"],
                abs_matrix=final_bundle["final_abs"],
                name_col="feature_name",
                feature_space_kind=feature_space_kind_primary,
                projection_status=compositional_projection_status,
            )
            semantic_global_frames.append(semantic_global_df)
            semantic_per_class_frames.append(semantic_per_class_df)

    input_global = pd.concat(input_global_frames, ignore_index=True)
    input_per_class = pd.concat(input_per_class_frames, ignore_index=True)
    semantic_global = pd.concat(semantic_global_frames, ignore_index=True)
    semantic_per_class = pd.concat(semantic_per_class_frames, ignore_index=True)
    top_k = int(contract.get("top_k_for_reporting", 10))
    top_global, top_per_class = build_top_k_views(
        global_df=semantic_global,
        per_class_df=semantic_per_class,
        top_k=top_k,
    )

    latent_global = pd.concat(latent_global_frames, ignore_index=True) if latent_global_frames else None
    latent_per_class = pd.concat(latent_per_class_frames, ignore_index=True) if latent_per_class_frames else None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "interpretability_summary_json": out_dir / "interpretability_summary.json",
        "input_feature_influence_global_csv": out_dir / "input_feature_influence_global.csv",
        "input_feature_influence_per_class_csv": out_dir / "input_feature_influence_per_class.csv",
        "feature_influence_global_csv": out_dir / "feature_influence_global.csv",
        "feature_influence_per_class_csv": out_dir / "feature_influence_per_class.csv",
        "top_features_global_csv": out_dir / "top_features_global.csv",
        "top_features_per_class_csv": out_dir / "top_features_per_class.csv",
        "latent_feature_influence_global_csv": None,
        "latent_feature_influence_per_class_csv": None,
        "flowpre_projection_manifest_json": None,
        "flowpre_projection_cache_path": None,
    }
    input_global.to_csv(artifact_paths["input_feature_influence_global_csv"], index=False)
    input_per_class.to_csv(artifact_paths["input_feature_influence_per_class_csv"], index=False)
    semantic_global.to_csv(artifact_paths["feature_influence_global_csv"], index=False)
    semantic_per_class.to_csv(artifact_paths["feature_influence_per_class_csv"], index=False)
    top_global.to_csv(artifact_paths["top_features_global_csv"], index=False)
    top_per_class.to_csv(artifact_paths["top_features_per_class_csv"], index=False)

    if latent_global is not None and latent_per_class is not None and projection_bundle is not None:
        artifact_paths["latent_feature_influence_global_csv"] = out_dir / "latent_feature_influence_global.csv"
        artifact_paths["latent_feature_influence_per_class_csv"] = out_dir / "latent_feature_influence_per_class.csv"
        latent_global.to_csv(artifact_paths["latent_feature_influence_global_csv"], index=False)
        latent_per_class.to_csv(artifact_paths["latent_feature_influence_per_class_csv"], index=False)
        artifact_paths["flowpre_projection_manifest_json"] = Path(projection_bundle["projection_manifest_path"])
        artifact_paths["flowpre_projection_cache_path"] = Path(projection_bundle["projection_cache_path"])

    interpretability_runtime_s = round(time.perf_counter() - started_at, 3)
    summary = {
        "mlp_interpretability_contract_id": str(contract.get("contract_id")),
        "method_id": str(contract.get("method_id")),
        "feature_space_kind_primary": str(feature_space_kind_primary),
        "projection_status": str(projection_status),
        "compositional_projection_status": str(compositional_projection_status),
        "compositional_component_scale": "normalized_share",
        "uses_flowpre_projection": bool(uses_flowpre_projection),
        "available_splits": list(available_splits),
        "top_k_for_reporting": top_k,
        "baseline_policy": str((contract.get("perturbation") or {}).get("baseline_policy")),
        "delta_definition": str((contract.get("perturbation") or {}).get("delta_definition")),
        "ranking_metric_primary": str(contract.get("ranking_metric_primary")),
        "ranking_metric_auxiliary": str(contract.get("ranking_metric_auxiliary")),
        "target_names": list(target_names),
        "n_input_features": int(input_global["feature_name"].nunique()),
        "n_semantic_features": int(semantic_global["feature_name"].nunique()),
        "n_latent_features": None if latent_global is None else int(latent_global["latent_name"].nunique()),
        "n_direct_passthrough_features": int(len(final_surface_spec.direct_feature_names)),
        "n_projected_chem_components": int(len(final_surface_spec.group_output_indices.get("chem", []))),
        "n_projected_phase_components": int(len(final_surface_spec.group_output_indices.get("phase", []))),
        "interpretability_runtime_s": interpretability_runtime_s,
        "artifact_refs": {
            key: (None if value is None else str(value))
            for key, value in artifact_paths.items()
        },
    }
    with open(artifact_paths["interpretability_summary_json"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    missing_items: list[str] = []
    for split_name in required_splits:
        if split_name not in available_splits:
            missing_items.append(f"split.{split_name}")
    if input_global.empty:
        missing_items.append("input_feature_influence_global")
    if input_per_class.empty:
        missing_items.append("input_feature_influence_per_class")
    missing_items.extend(
        _validate_final_semantic_surface(
            semantic_global=semantic_global,
            semantic_per_class=semantic_per_class,
            top_global=top_global,
            top_per_class=top_per_class,
            exclude_prefixes=[str(item) for item in (contract.get("final_artifact_must_exclude_prefixes") or [])],
        )
    )
    if uses_flowpre_projection:
        if latent_global is None or latent_global.empty:
            missing_items.append("latent_feature_influence_global")
        if latent_per_class is None or latent_per_class.empty:
            missing_items.append("latent_feature_influence_per_class")
        if artifact_paths["flowpre_projection_manifest_json"] is None:
            missing_items.append("flowpre_projection_manifest_json")
        if artifact_paths["flowpre_projection_cache_path"] is None:
            missing_items.append("flowpre_projection_cache_path")

    validation = {
        "mlp_interpretability_contract_id": str(contract.get("contract_id")),
        "method_id": str(contract.get("method_id")),
        "run_mode": str(run_mode),
        "required_splits_resolved": required_splits,
        "required_scopes": [str(item) for item in (contract.get("required_scopes") or [])],
        "available_splits": list(available_splits),
        "feature_space_kind_primary": str(feature_space_kind_primary),
        "projection_status": str(projection_status),
        "compositional_projection_status": str(compositional_projection_status),
        "campaign_valid_interpretability": len(missing_items) == 0,
        "missing_items": missing_items,
    }
    return {
        "summary": summary,
        "validation": validation,
        "artifact_paths": artifact_paths,
        "input_global": input_global,
        "input_per_class": input_per_class,
        "semantic_global": semantic_global,
        "semantic_per_class": semantic_per_class,
        "top_global": top_global,
        "top_per_class": top_per_class,
        "latent_global": latent_global,
        "latent_per_class": latent_per_class,
        "projection_bundle": projection_bundle,
    }
