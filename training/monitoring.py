from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


OFFICIAL_VAL_POLICY = "official_val"
TRAIN_ONLY_POLICY = "train_only"
SUPPORTED_MONITORING_POLICIES = {OFFICIAL_VAL_POLICY, TRAIN_ONLY_POLICY}

OFFICIAL_SPLIT_ROLE_MAP = {
    "train": "train_diagnostic",
    "val": "val_selection",
    "test": "test_holdout",
}

TRAIN_ONLY_SPLIT_ROLE_MAP = {
    "train": "train_diagnostic",
    "val": "train_monitor_pseudo_val",
    "test": "test_holdout",
}


def normalize_monitoring_policy(policy: str | None) -> str:
    resolved = str(policy or OFFICIAL_VAL_POLICY).strip().lower()
    if resolved not in SUPPORTED_MONITORING_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_MONITORING_POLICIES))
        raise ValueError(f"Unsupported monitoring_policy={policy!r}. Supported values: {supported}.")
    return resolved


def monitoring_metadata(policy: str | None) -> dict[str, Any]:
    resolved = normalize_monitoring_policy(policy)
    if resolved == TRAIN_ONLY_POLICY:
        return {
            "policy": TRAIN_ONLY_POLICY,
            "monitor_result_key": "val",
            "monitor_source_split": "train",
            "monitor_role": "train_monitor_pseudo_val",
            "monitor_is_holdout": False,
            "canonical_selection_eligible": False,
            "notes": (
                "The result key 'val' is kept for compatibility, but it is a train-derived "
                "monitoring surface, not the official temporal validation split."
            ),
        }
    return {
        "policy": OFFICIAL_VAL_POLICY,
        "monitor_result_key": "val",
        "monitor_source_split": "val",
        "monitor_role": "val_selection",
        "monitor_is_holdout": True,
        "canonical_selection_eligible": True,
        "notes": "The result key 'val' refers to the official temporal validation split.",
    }


def monitoring_split_role_map(policy: str | None) -> dict[str, str]:
    resolved = normalize_monitoring_policy(policy)
    if resolved == TRAIN_ONLY_POLICY:
        return dict(TRAIN_ONLY_SPLIT_ROLE_MAP)
    return dict(OFFICIAL_SPLIT_ROLE_MAP)


def experimental_output_namespace(output_namespace: str | None, policy: str | None) -> str | None:
    if output_namespace:
        return output_namespace
    if normalize_monitoring_policy(policy) == TRAIN_ONLY_POLICY:
        return "experimental/train_only"
    return None


def ensure_holdout_policy(policy: str | None, *, allow_test_holdout: bool) -> None:
    if normalize_monitoring_policy(policy) == TRAIN_ONLY_POLICY and allow_test_holdout:
        raise ValueError("monitoring_policy='train_only' does not allow test holdout metrics in this support phase.")


def with_monitoring_context(
    evaluation_context: Mapping[str, Any] | None,
    policy: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = monitoring_metadata(policy)
    ctx = deepcopy(dict(evaluation_context or {}))

    if metadata["policy"] != TRAIN_ONLY_POLICY:
        return ctx, metadata

    ctx["monitoring"] = metadata
    ctx["monitoring_policy"] = metadata["policy"]
    ctx["split_role_map"] = monitoring_split_role_map(metadata["policy"])

    run_level_axes = dict(ctx.get("run_level_axes") or {})
    run_level_axes["monitoring_policy"] = metadata["policy"]
    run_level_axes["monitor_source_split"] = metadata["monitor_source_split"]
    run_level_axes["monitor_role"] = metadata["monitor_role"]
    run_level_axes["monitor_is_holdout"] = metadata["monitor_is_holdout"]
    run_level_axes["canonical_selection_eligible"] = metadata["canonical_selection_eligible"]
    ctx["run_level_axes"] = run_level_axes

    return ctx, metadata
