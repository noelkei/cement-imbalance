from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from evaluation.metrics import (
    DEFAULT_QUANTILE_RANGES,
    compute_regression_metrics_from_preds,
    inverse_transform_tensor,
)


def _as_tensor(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    array = np.asarray(value)
    return torch.as_tensor(array, dtype=dtype, device=device)


def compute_predictive_metrics_for_split(
    *,
    y_hat: Any,
    y_true: Any,
    class_codes: Any,
    device: torch.device | None = None,
    y_scaler: Any | None = None,
    quantile_ranges: Mapping[str, tuple[float, float]] = DEFAULT_QUANTILE_RANGES,
) -> dict[str, Any]:
    resolved_device = device or torch.device("cpu")
    y_hat_tensor = _as_tensor(y_hat, dtype=torch.float32, device=resolved_device)
    y_true_tensor = _as_tensor(y_true, dtype=torch.float32, device=resolved_device)
    c_tensor = _as_tensor(class_codes, dtype=torch.long, device=resolved_device)

    if y_hat_tensor.ndim == 1:
        y_hat_tensor = y_hat_tensor.reshape(-1, 1)
    if y_true_tensor.ndim == 1:
        y_true_tensor = y_true_tensor.reshape(-1, 1)

    if y_scaler is not None:
        y_hat_tensor = inverse_transform_tensor(y_hat_tensor, y_scaler, device=resolved_device)
        y_true_tensor = inverse_transform_tensor(y_true_tensor, y_scaler, device=resolved_device)

    return compute_regression_metrics_from_preds(
        y_hat=y_hat_tensor,
        y=y_true_tensor,
        c=c_tensor,
        quantile_ranges=quantile_ranges,
    )


def build_predictive_metric_spaces(
    *,
    predictions_by_split: Mapping[str, Any],
    targets_by_split: Mapping[str, Any],
    class_codes_by_split: Mapping[str, Any],
    native_enabled: bool,
    raw_real_required: bool,
    native_value_space: str = "native",
    raw_real_from_native_when_no_scaler: bool = False,
    y_scaler: Any | None = None,
    quantile_ranges: Mapping[str, tuple[float, float]] = DEFAULT_QUANTILE_RANGES,
    device: torch.device | None = None,
) -> dict[str, Any]:
    spaces: dict[str, dict[str, Any]] = {}
    resolved_device = device or torch.device("cpu")
    split_names = [split for split in ("train", "val", "test") if split in predictions_by_split and split in targets_by_split]

    if native_enabled:
        native_metrics: dict[str, Any] = {}
        for split in split_names:
            native_metrics[split] = compute_predictive_metrics_for_split(
                y_hat=predictions_by_split[split],
                y_true=targets_by_split[split],
                class_codes=class_codes_by_split[split],
                device=resolved_device,
                y_scaler=None,
                quantile_ranges=quantile_ranges,
            )
        spaces[native_value_space] = native_metrics

    raw_real_metrics: dict[str, Any] = {}
    if y_scaler is not None:
        for split in split_names:
            raw_real_metrics[split] = compute_predictive_metrics_for_split(
                y_hat=predictions_by_split[split],
                y_true=targets_by_split[split],
                class_codes=class_codes_by_split[split],
                device=resolved_device,
                y_scaler=y_scaler,
                quantile_ranges=quantile_ranges,
            )
    elif raw_real_required:
        if native_enabled and raw_real_from_native_when_no_scaler:
            raw_real_metrics = {
                split: spaces[native_value_space][split]
                for split in split_names
            }
        else:
            for split in split_names:
                raw_real_metrics[split] = compute_predictive_metrics_for_split(
                    y_hat=predictions_by_split[split],
                    y_true=targets_by_split[split],
                    class_codes=class_codes_by_split[split],
                    device=resolved_device,
                    y_scaler=None,
                    quantile_ranges=quantile_ranges,
                )
    if raw_real_metrics:
        spaces["raw_real"] = raw_real_metrics

    return spaces


def build_predictive_results_payload(
    *,
    metric_spaces: Mapping[str, Mapping[str, Any]],
    raw_metric_contract_validation: Mapping[str, Any],
    raw_inversion_status: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    native_metrics = metric_spaces.get("native")
    if isinstance(native_metrics, Mapping):
        for split_name, split_metrics in native_metrics.items():
            payload[str(split_name)] = split_metrics
    raw_real_metrics = metric_spaces.get("raw_real")
    if isinstance(raw_real_metrics, Mapping):
        payload["raw_real"] = dict(raw_real_metrics)
    payload["metric_spaces"] = {
        "default": "raw_real" if "raw_real" in metric_spaces else "native",
        "available": list(metric_spaces.keys()),
    }
    payload["run_mode"] = raw_metric_contract_validation.get("run_mode")
    payload["raw_metric_contract_id"] = raw_metric_contract_validation.get("raw_metric_contract_id")
    payload["raw_metric_contract_validation"] = dict(raw_metric_contract_validation)
    payload["raw_inversion_status"] = dict(raw_inversion_status)
    payload["campaign_valid"] = bool(raw_metric_contract_validation.get("campaign_valid"))
    return payload
