from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from data.utils import path_relative_to_root


F5_DATASET_CONTRACT_ID = "f5_dataset_contract_v1"
DEFAULT_DATASET_CONTRACT_VERSION = "v1"

DATASET_LEVEL_AXES = (
    "x_transform",
    "y_transform",
    "synthetic_policy",
)

RUN_LEVEL_AXES = (
    "batch_policy",
    "cycling_policy",
    "loss_policy",
    "objective_metric",
    "seed_set",
    "mlp_base_config",
    "comparison_group_id",
)

CLASSICAL_X_TRANSFORMS = (
    "standard",
    "robust",
    "minmax",
    "quantile",
)

FLOWPRE_SCALER_X_TRANSFORMS = (
    "flowpre_rrmse",
    "flowpre_mvn",
    "flowpre_fair",
)

FLOWGEN_WORK_BASE_X_TRANSFORMS = (
    "flowpre_candidate_1",
    "flowpre_candidate_2",
)

FLOWPRE_X_TRANSFORMS = FLOWPRE_SCALER_X_TRANSFORMS + FLOWGEN_WORK_BASE_X_TRANSFORMS

SUPPORTED_X_TRANSFORMS = CLASSICAL_X_TRANSFORMS + FLOWPRE_X_TRANSFORMS
SUPPORTED_Y_TRANSFORMS = (
    "standard",
    "robust",
    "minmax",
    "quantile",
)
SUPPORTED_SYNTHETIC_POLICIES = (
    "none",
    "flowgen",
    "kmeans_smote",
)

SUPPORTED_SPACE_STATUSES = (
    "supported_space",
    "materialized_now",
    "blocked_by_upstream",
    "future_supported",
    "out_of_scope",
)


def _copy_jsonish(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, sort_keys=True))


def _relativize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return path_relative_to_root(value)
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.exists():
            return path_relative_to_root(candidate)
        return value
    if isinstance(value, Mapping):
        return {str(k): _relativize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_relativize(v) for v in value]
    if isinstance(value, tuple):
        return [_relativize(v) for v in value]
    return value


def is_flowpre_x_transform(x_transform: str) -> bool:
    return x_transform in FLOWPRE_X_TRANSFORMS


def is_classical_x_transform(x_transform: str) -> bool:
    return x_transform in CLASSICAL_X_TRANSFORMS


def is_flowgen_work_base_x_transform(x_transform: str) -> bool:
    return x_transform in FLOWGEN_WORK_BASE_X_TRANSFORMS


def validate_dataset_level_axes(
    *,
    x_transform: str,
    y_transform: str,
    synthetic_policy: str,
) -> None:
    if x_transform not in SUPPORTED_X_TRANSFORMS:
        raise ValueError(
            f"Unsupported x_transform '{x_transform}'. "
            f"Allowed: {list(SUPPORTED_X_TRANSFORMS)}"
        )
    if y_transform not in SUPPORTED_Y_TRANSFORMS:
        raise ValueError(
            f"Unsupported y_transform '{y_transform}'. "
            f"Allowed: {list(SUPPORTED_Y_TRANSFORMS)}"
        )
    if synthetic_policy not in SUPPORTED_SYNTHETIC_POLICIES:
        raise ValueError(
            f"Unsupported synthetic_policy '{synthetic_policy}'. "
            f"Allowed: {list(SUPPORTED_SYNTHETIC_POLICIES)}"
        )


def build_canonical_dataset_name(
    *,
    x_transform: str,
    y_transform: str,
    synthetic_policy: str = "none",
    version: str = DEFAULT_DATASET_CONTRACT_VERSION,
) -> str:
    validate_dataset_level_axes(
        x_transform=x_transform,
        y_transform=y_transform,
        synthetic_policy=synthetic_policy,
    )
    return (
        f"dataset__x-{x_transform}"
        f"__y-{y_transform}"
        f"__syn-{synthetic_policy}"
        f"__{version}"
    )


def classify_supported_dataset_space(
    *,
    x_transform: str,
    y_transform: str,
    synthetic_policy: str = "none",
) -> dict[str, Any]:
    try:
        validate_dataset_level_axes(
            x_transform=x_transform,
            y_transform=y_transform,
            synthetic_policy=synthetic_policy,
        )
    except ValueError as exc:
        return {
            "support_status": "out_of_scope",
            "status_reason": str(exc),
            "source_requirements": [],
            "dataset_storage_family": "unknown",
        }

    storage_family = "official_scaled" if synthetic_policy == "none" else "official_augmented_scaled"

    if synthetic_policy == "kmeans_smote":
        if is_classical_x_transform(x_transform):
            return {
                "support_status": "materialized_now",
                "status_reason": (
                    "kmeans_smote is implemented as a dataset-level synthetic_policy over "
                    "canonical non-synthetic bundles, and classical X/Y scaling can be "
                    "materialized directly from the official raw bundle."
                ),
                "source_requirements": ["official_raw_bundle"],
                "dataset_storage_family": storage_family,
            }
        return {
            "support_status": "supported_space",
            "status_reason": (
                "kmeans_smote is implemented as a dataset-level synthetic_policy over "
                "canonical non-synthetic bundles. FlowPre-based X transforms remain "
                "supported once the corresponding promoted upstream is available."
            ),
            "source_requirements": ["official_raw_bundle", "flowpre_upstream"],
            "dataset_storage_family": storage_family,
        }

    if synthetic_policy == "flowgen":
        if not is_flowgen_work_base_x_transform(x_transform):
            return {
                "support_status": "out_of_scope",
                "status_reason": (
                    "flowgen synthetic_policy is only defined for the FlowGen work bases "
                    "flowpre_candidate_1 and flowpre_candidate_2."
                ),
                "source_requirements": [],
                "dataset_storage_family": storage_family,
            }
        return {
            "support_status": "blocked_by_upstream",
            "status_reason": (
                "flowgen synthetic_policy is supported by contract for the work bases "
                "flowpre_candidate_1 / flowpre_candidate_2, but it remains blocked "
                "until a promoted FlowGen upstream exists for the selected work base."
            ),
            "source_requirements": ["official_raw_bundle", "flowpre_upstream", "flowgen_upstream"],
            "dataset_storage_family": storage_family,
        }

    if is_classical_x_transform(x_transform):
        return {
            "support_status": "materialized_now",
            "status_reason": (
                "Classical X/Y scaling derived directly from the official raw bundle is "
                "deterministic and can be materialized in F5a."
            ),
            "source_requirements": ["official_raw_bundle"],
            "dataset_storage_family": storage_family,
        }

    return {
        "support_status": "supported_space",
        "status_reason": (
            "FlowPre-based X transforms are supported by contract and can be materialized "
            "once the corresponding promoted FlowPre upstream is available."
        ),
        "source_requirements": ["official_raw_bundle", "flowpre_upstream"],
        "dataset_storage_family": storage_family,
    }


def build_dataset_spec(
    *,
    x_transform: str,
    y_transform: str,
    synthetic_policy: str = "none",
    version: str = DEFAULT_DATASET_CONTRACT_VERSION,
) -> dict[str, Any]:
    support = classify_supported_dataset_space(
        x_transform=x_transform,
        y_transform=y_transform,
        synthetic_policy=synthetic_policy,
    )
    dataset_level_axes = {
        "x_transform": x_transform,
        "y_transform": y_transform,
        "synthetic_policy": synthetic_policy,
    }
    return {
        "contract_id": F5_DATASET_CONTRACT_ID,
        "dataset_name": build_canonical_dataset_name(
            x_transform=x_transform,
            y_transform=y_transform,
            synthetic_policy=synthetic_policy,
            version=version,
        ),
        "dataset_level_axes": dataset_level_axes,
        "dataset_level_axis_names": list(DATASET_LEVEL_AXES),
        "dataset_storage_family": support["dataset_storage_family"],
        "support_status": support["support_status"],
        "supported_space_status": support["support_status"],
        "status_reason": support["status_reason"],
        "source_requirements": list(support["source_requirements"]),
    }


def build_base_scaling_matrix(
    *,
    version: str = DEFAULT_DATASET_CONTRACT_VERSION,
) -> list[dict[str, Any]]:
    return [
        build_dataset_spec(
            x_transform=x_transform,
            y_transform=y_transform,
            synthetic_policy="none",
            version=version,
        )
        for x_transform in SUPPORTED_X_TRANSFORMS
        for y_transform in SUPPORTED_Y_TRANSFORMS
    ]


def build_supported_dataset_matrix(
    *,
    version: str = DEFAULT_DATASET_CONTRACT_VERSION,
    include_synthetic_policies: bool = True,
    include_out_of_scope: bool = False,
) -> list[dict[str, Any]]:
    policies = SUPPORTED_SYNTHETIC_POLICIES if include_synthetic_policies else ("none",)
    specs = [
        build_dataset_spec(
            x_transform=x_transform,
            y_transform=y_transform,
            synthetic_policy=synthetic_policy,
            version=version,
        )
        for x_transform in SUPPORTED_X_TRANSFORMS
        for y_transform in SUPPORTED_Y_TRANSFORMS
        for synthetic_policy in policies
    ]
    if include_out_of_scope:
        return specs
    return [spec for spec in specs if spec["support_status"] != "out_of_scope"]


def counts_from_source_manifest(source_manifest: Mapping[str, Any] | None) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    if not source_manifest:
        return {}, {}

    if "counts_by_split" in source_manifest and "counts_by_class" in source_manifest:
        counts_by_split = {
            str(split): int(value)
            for split, value in dict(source_manifest.get("counts_by_split", {})).items()
        }
        counts_by_class = {
            str(split): {str(cls): int(count) for cls, count in dict(class_counts).items()}
            for split, class_counts in dict(source_manifest.get("counts_by_class", {})).items()
        }
        return counts_by_split, counts_by_class

    summary_by_split = dict(source_manifest.get("summary_by_split", {}))
    counts_by_split = {
        str(split): int(
            summary.get("X_rows")
            if summary.get("X_rows") is not None
            else summary.get("rows", 0)
        )
        for split, summary in summary_by_split.items()
    }
    counts_by_class = {
        str(split): {
            str(cls): int(count)
            for cls, count in dict(summary.get("class_counts", {})).items()
        }
        for split, summary in summary_by_split.items()
    }
    return counts_by_split, counts_by_class


def build_canonical_derived_manifest(
    *,
    dataset_name: str,
    dataset_level_axes: Mapping[str, Any],
    split_id: str,
    cleaning_policy_id: str,
    source_dataset_manifest_path: str | Path,
    source_split_manifest_path: str | Path | None,
    source_cleaning_manifest_path: str | Path | None,
    support_status: str,
    artifacts: Mapping[str, Any],
    scaler_artifacts: Mapping[str, Any] | None = None,
    source_manifest: Mapping[str, Any] | None = None,
    upstream_model_manifests: list[str | Path] | None = None,
    policy_status: str = "canonical",
    synthetic_policy_id: str | None = None,
    train_only_mutations: list[str] | None = None,
    extra_manifest_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if support_status not in SUPPORTED_SPACE_STATUSES:
        raise ValueError(
            f"Unsupported support_status '{support_status}'. "
            f"Allowed: {list(SUPPORTED_SPACE_STATUSES)}"
        )

    synthetic_policy = str(dataset_level_axes.get("synthetic_policy", "none"))
    counts_by_split, counts_by_class = counts_from_source_manifest(source_manifest)
    manifest = {
        "contract_id": F5_DATASET_CONTRACT_ID,
        "dataset_name": dataset_name,
        "dataset_role": "derived_modeling_bundle",
        "policy_status": policy_status,
        "split_id": split_id,
        "cleaning_policy_id": cleaning_policy_id,
        "source_dataset_manifest": _relativize(source_dataset_manifest_path),
        "source_split_manifest": _relativize(source_split_manifest_path),
        "source_cleaning_manifest": _relativize(source_cleaning_manifest_path),
        "source_dataset_name": None if source_manifest is None else source_manifest.get("dataset_name"),
        "dataset_level_axes": _copy_jsonish(dict(dataset_level_axes)),
        "dataset_level_axis_names": list(DATASET_LEVEL_AXES),
        "synthetic_policy_id": synthetic_policy_id or synthetic_policy,
        "supported_space_status": support_status,
        "train_only_mutations": (
            list(train_only_mutations)
            if train_only_mutations is not None
            else (["synthetic_policy"] if synthetic_policy != "none" else [])
        ),
        "counts_by_split": counts_by_split,
        "counts_by_class": counts_by_class,
        "artifacts": _relativize(artifacts),
        "scaler_artifacts": _relativize(scaler_artifacts or {}),
        "upstream_model_manifests": _relativize(upstream_model_manifests or []),
    }
    if extra_manifest_fields:
        manifest.update(_relativize(dict(extra_manifest_fields)))
    return manifest
