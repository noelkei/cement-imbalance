from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from data.utils import ROOT_PATH


_CONTRACT_PATH = Path(ROOT_PATH) / "config" / "f7_raw_metric_contract_v1.yaml"


@lru_cache(maxsize=1)
def load_f7_raw_metric_contract() -> dict[str, Any]:
    with open(_CONTRACT_PATH, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    contract = payload.get("raw_metric_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Invalid raw metric contract payload in {_CONTRACT_PATH}")
    return contract


def resolve_run_mode(*, test_enabled: bool) -> str:
    contract = load_f7_raw_metric_contract()
    rule = dict(contract.get("run_mode_resolution_rule") or {})
    return str(rule.get("if_true") if test_enabled else rule.get("if_false"))


def build_raw_inversion_status(
    *,
    y_transform: str | None,
    y_scaler_present: bool,
    target_scaler_artifact: str | None,
    raw_real_available: bool,
) -> dict[str, Any]:
    resolved_y_transform = str(y_transform or "raw")
    requires_inversion = resolved_y_transform != "raw"
    return {
        "y_transform": resolved_y_transform,
        "requires_raw_inversion": requires_inversion,
        "y_scaler_present": bool(y_scaler_present),
        "target_scaler_artifact": target_scaler_artifact,
        "raw_real_available": bool(raw_real_available),
        "status": (
            "not_required"
            if not requires_inversion
            else "ok"
            if y_scaler_present and raw_real_available
            else "missing_scaler"
            if not y_scaler_present
            else "raw_real_missing"
        ),
    }


def _scope_has_required_metrics(scope_metrics: Mapping[str, Any], required_metric_names: list[str]) -> list[str]:
    missing: list[str] = []
    for metric_name in required_metric_names:
        if metric_name not in scope_metrics:
            missing.append(metric_name)
    return missing


def validate_raw_metric_contract(
    *,
    metric_spaces: Mapping[str, Mapping[str, Any]],
    test_enabled: bool,
    raw_inversion_status: Mapping[str, Any],
    value_space_default: str | None,
) -> dict[str, Any]:
    contract = load_f7_raw_metric_contract()
    run_mode = resolve_run_mode(test_enabled=test_enabled)
    required_splits = list((contract.get("required_splits_by_run_mode") or {}).get(run_mode, []))
    required_scopes = [str(item) for item in contract.get("required_metric_scopes", [])]
    required_metric_names = [str(item) for item in contract.get("required_metric_names", [])]
    missing_items: list[str] = []
    raw_real = metric_spaces.get("raw_real")

    if not isinstance(raw_real, Mapping):
        missing_items.append("value_space.raw_real")
        raw_real = {}

    if not value_space_default:
        missing_items.append("metric_spaces.default")

    for split_name in required_splits:
        split_metrics = raw_real.get(split_name)
        if not isinstance(split_metrics, Mapping):
            missing_items.append(f"raw_real.{split_name}")
            continue
        for scope_name in required_scopes:
            scope_metrics = split_metrics.get(scope_name)
            if scope_name == "per_class":
                if not isinstance(scope_metrics, Mapping) or not scope_metrics:
                    missing_items.append(f"raw_real.{split_name}.per_class")
                    continue
                for cls_id, cls_metrics in scope_metrics.items():
                    if not isinstance(cls_metrics, Mapping):
                        missing_items.append(f"raw_real.{split_name}.per_class.{cls_id}")
                        continue
                    for metric_name in _scope_has_required_metrics(cls_metrics, required_metric_names):
                        missing_items.append(f"raw_real.{split_name}.per_class.{cls_id}.{metric_name}")
                continue
            if not isinstance(scope_metrics, Mapping):
                missing_items.append(f"raw_real.{split_name}.{scope_name}")
                continue
            for metric_name in _scope_has_required_metrics(scope_metrics, required_metric_names):
                missing_items.append(f"raw_real.{split_name}.{scope_name}.{metric_name}")

    raw_inversion_ok = bool(raw_inversion_status.get("status") in {"not_required", "ok"})
    if not raw_inversion_ok:
        missing_items.append(f"raw_inversion:{raw_inversion_status.get('status')}")

    campaign_valid = len(missing_items) == 0
    return {
        "raw_metric_contract_id": str(contract.get("raw_metric_contract_id") or contract.get("contract_id")),
        "contract_version": int(contract.get("contract_version", 1)),
        "run_mode": run_mode,
        "required_splits_resolved": required_splits,
        "required_metric_scopes": required_scopes,
        "required_metric_names": required_metric_names,
        "validation_status": "ok" if campaign_valid else "failed",
        "missing_items": missing_items,
        "raw_inversion_status": dict(raw_inversion_status),
        "campaign_valid": campaign_valid,
        "value_space_default": value_space_default,
    }
