from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from data.utils import ROOT_PATH, path_relative_to_root


_META_GRAMMAR_PATH = Path(ROOT_PATH) / "config" / "f7_meta_grammar_v1.yaml"
_SEED_PANEL_PATH = Path(ROOT_PATH) / "config" / "f7_seed_panel_v1.yaml"
_CLASS_ONTOLOGY_PATH = Path(ROOT_PATH) / "config" / "f7_class_ontology_v1.yaml"
_TARGET_CONTRACT_PATH = Path(ROOT_PATH) / "config" / "f7_target_contract_v1.yaml"
_ARTIFACT_PERSISTENCE_CONTRACT_PATH = Path(ROOT_PATH) / "config" / "f7_artifact_persistence_contract_v1.yaml"
_METRIC_GRAMMAR_CONTRACT_PATH = Path(ROOT_PATH) / "config" / "f7_metric_grammar_v1.yaml"
_METRIC_AVAILABILITY_CONTRACT_PATH = Path(ROOT_PATH) / "config" / "f7_metric_availability_contract_v1.yaml"
_METRIC_AGGREGATION_CONTRACT_PATH = Path(ROOT_PATH) / "config" / "f7_metric_aggregation_contract_v1.yaml"
_EVALUATION_POPULATION_CONTRACT_PATH = Path(ROOT_PATH) / "config" / "f7_evaluation_population_contract_v1.yaml"
_PREDICTION_ROW_JOIN_CONTRACT_PATH = Path(ROOT_PATH) / "config" / "f7_prediction_row_join_contract_v1.yaml"
_FEATURE_SCHEMA_CONTRACT_PATH = Path(ROOT_PATH) / "config" / "f7_feature_schema_contract_v1.yaml"
_FACTOR_PARSER_CONTRACT_PATH = Path(ROOT_PATH) / "config" / "f7_factor_parser_contract_v1.yaml"

LINEAGE_AGGREGATE_BUILD_VERSION = "f7_lineage_aggregate_v2"
PANEL_BUILD_VERSION = "f7_campaign_panel_v2"


def _load_yaml_dict(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def _load_contract_block(path: Path, top_key: str) -> dict[str, Any]:
    payload = _load_yaml_dict(path)
    block = payload.get(top_key)
    if not isinstance(block, dict):
        raise ValueError(f"Invalid contract payload '{top_key}' in {path}")
    return block


def _stable_bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _safe_token(value: Any) -> str:
    return str(value).strip().replace(" ", "_")


def _normalize_manifest_ref(path: str | Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path_relative_to_root(path)
    except Exception:
        return str(path)


@lru_cache(maxsize=1)
def load_f7_meta_grammar() -> dict[str, Any]:
    payload = _load_yaml_dict(_META_GRAMMAR_PATH)
    grammar = payload.get("meta_grammar")
    if not isinstance(grammar, dict):
        raise ValueError(f"Invalid F7 meta grammar payload in {_META_GRAMMAR_PATH}")
    return grammar


@lru_cache(maxsize=1)
def load_f7_class_ontology() -> dict[str, Any]:
    return _load_contract_block(_CLASS_ONTOLOGY_PATH, "class_ontology")


@lru_cache(maxsize=1)
def load_f7_target_contract() -> dict[str, Any]:
    return _load_contract_block(_TARGET_CONTRACT_PATH, "target_contract")


@lru_cache(maxsize=1)
def load_f7_artifact_persistence_contract() -> dict[str, Any]:
    return _load_contract_block(_ARTIFACT_PERSISTENCE_CONTRACT_PATH, "artifact_persistence_contract")


@lru_cache(maxsize=1)
def load_f7_metric_grammar_contract() -> dict[str, Any]:
    return _load_contract_block(_METRIC_GRAMMAR_CONTRACT_PATH, "metric_grammar")


@lru_cache(maxsize=1)
def load_f7_metric_availability_contract() -> dict[str, Any]:
    return _load_contract_block(_METRIC_AVAILABILITY_CONTRACT_PATH, "metric_availability_contract")


@lru_cache(maxsize=1)
def load_f7_metric_aggregation_contract() -> dict[str, Any]:
    return _load_contract_block(_METRIC_AGGREGATION_CONTRACT_PATH, "metric_aggregation_contract")


@lru_cache(maxsize=1)
def load_f7_evaluation_population_contract() -> dict[str, Any]:
    return _load_contract_block(_EVALUATION_POPULATION_CONTRACT_PATH, "evaluation_population_contract")


@lru_cache(maxsize=1)
def load_f7_prediction_row_join_contract() -> dict[str, Any]:
    return _load_contract_block(_PREDICTION_ROW_JOIN_CONTRACT_PATH, "prediction_row_join_contract")


@lru_cache(maxsize=1)
def load_f7_feature_schema_contract() -> dict[str, Any]:
    return _load_contract_block(_FEATURE_SCHEMA_CONTRACT_PATH, "feature_schema_contract")


@lru_cache(maxsize=1)
def load_f7_factor_parser_contract() -> dict[str, Any]:
    return _load_contract_block(_FACTOR_PARSER_CONTRACT_PATH, "factor_parser_contract")


@lru_cache(maxsize=1)
def load_f7_seed_panel() -> dict[str, Any]:
    payload = _load_yaml_dict(_SEED_PANEL_PATH)
    panel = payload.get("seed_panel")
    if not isinstance(panel, dict):
        raise ValueError(f"Invalid F7 seed panel payload in {_SEED_PANEL_PATH}")
    return panel


def _resolve_seed_panel_path(seed_set_id: str | None, seed_panel_path: str | Path | None) -> Path:
    if seed_panel_path is not None:
        candidate = Path(seed_panel_path)
        return candidate if candidate.is_absolute() else Path(ROOT_PATH) / candidate
    if seed_set_id is None:
        return _SEED_PANEL_PATH
    candidate = Path(ROOT_PATH) / "config" / f"{seed_set_id}.yaml"
    return candidate if candidate.exists() else _SEED_PANEL_PATH


@lru_cache(maxsize=16)
def load_seed_panel_by_ref(seed_set_id: str | None = None, seed_panel_path: str | Path | None = None) -> dict[str, Any]:
    resolved_path = _resolve_seed_panel_path(seed_set_id, seed_panel_path)
    payload = _load_yaml_dict(resolved_path)
    panel = payload.get("seed_panel")
    if not isinstance(panel, dict):
        raise ValueError(f"Invalid seed panel payload in {resolved_path}")
    return panel


@lru_cache(maxsize=1)
def load_f7_inventory_rows() -> list[dict[str, str]]:
    grammar = load_f7_meta_grammar()
    csv_path = Path(ROOT_PATH) / str(grammar["dataset_candidate_id"]["canonical_source"])
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def _inventory_manifest_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for row in load_f7_inventory_rows():
        manifest_path = row.get("manifest_path")
        dataset_candidate_id = row.get("dataset_candidate_id") or row.get("inventory_dataset_id")
        if manifest_path and dataset_candidate_id:
            index[_normalize_manifest_ref(manifest_path) or manifest_path] = dataset_candidate_id
    return index


def resolve_dataset_candidate_id(
    *,
    dataset_manifest_path: str | Path | None,
    dataset_candidate_id: str | None = None,
) -> str | None:
    if dataset_candidate_id:
        return str(dataset_candidate_id)
    normalized = _normalize_manifest_ref(dataset_manifest_path)
    if normalized is None:
        return None
    return _inventory_manifest_index().get(normalized)


def resolve_replication_index(
    *,
    seed: int | None,
    seed_set_id: str | None,
    seed_panel_path: str | Path | None = None,
) -> int | None:
    if seed is None or seed_set_id is None:
        return None
    panel = load_seed_panel_by_ref(seed_set_id=seed_set_id, seed_panel_path=seed_panel_path)
    if str(panel.get("seed_set_id")) != str(seed_set_id):
        return None
    for row in panel.get("seeds", []):
        if int(row.get("value")) == int(seed):
            return int(row.get("index"))
    return None


def derive_run_spec_id(
    *,
    model_family: str,
    base_config_id: str | None,
    run_level_axes: Mapping[str, Any] | None,
) -> str | None:
    if base_config_id is None:
        return None
    family = str(model_family).lower()
    if family == "mlp":
        axes = dict(run_level_axes or {})
        batch_id = str(((axes.get("batch_policy") or {}).get("id")) or "unknown").lower()
        batch_policy = {
            "baseline": "plain",
            "plain": "plain",
            "balanced": "imbalance_aware",
            "imbalance_aware": "imbalance_aware",
        }.get(batch_id, _safe_token(batch_id))
        cycling_policy = "cycling" if bool(((axes.get("cycling_policy") or {}).get("cycle_reals"))) else "no_cycling"
        loss_axes = dict((axes.get("loss_policy") or {}))
        loss_policy = _safe_token(
            f"{loss_axes.get('loss_reduction', 'unknown')}_{loss_axes.get('regression_group_metric', 'unknown')}"
        )
        allow_synth = _stable_bool_token(((axes.get("allow_synth") or {}).get("enabled")))
        return (
            f"runspec__mlp__{_safe_token(base_config_id)}__{batch_policy}__"
            f"{cycling_policy}__{loss_policy}__allow_synth-{allow_synth}"
        )
    if family == "xgboost":
        return f"runspec__xgb__{_safe_token(base_config_id)}"
    return None


def _normalize_run_policy_fragment(value: Any, fallback: str) -> str:
    candidate = str(value).strip()
    return fallback if not candidate or candidate.lower() == "none" else candidate


def parse_f7_factor_fields(
    *,
    model_family: str,
    dataset_level_axes: Mapping[str, Any] | None,
    run_level_axes: Mapping[str, Any] | None,
    fallback_dataset_candidate_id: str | None = None,
    fallback_run_spec_id: str | None = None,
) -> dict[str, Any]:
    dataset_axes = dict(dataset_level_axes or {})
    run_axes = dict(run_level_axes or {})
    x_transform = str(dataset_axes.get("x_transform") or "unknown")
    y_transform = str(dataset_axes.get("y_transform") or "unknown")
    synthetic_policy = str(dataset_axes.get("synthetic_policy") or "unknown")
    if str(model_family).lower() == "mlp":
        batch_policy = _normalize_run_policy_fragment(run_axes.get("batch_policy_id"), "unknown_batch")
        cycling_policy = _normalize_run_policy_fragment(run_axes.get("cycling_policy_id"), "unknown_cycling")
        loss_policy = _normalize_run_policy_fragment(run_axes.get("loss_policy_id"), "unknown_loss")
        allow_synth = run_axes.get("allow_synth")
        allow_synth_token = "allow_synth-true" if bool(allow_synth) else "allow_synth-false"
        run_policy = f"{batch_policy}__{cycling_policy}__{loss_policy}__{allow_synth_token}"
    elif str(model_family).lower() == "xgboost":
        run_policy = "xgb_canonical"
    else:
        run_policy = _normalize_run_policy_fragment(fallback_run_spec_id, "unknown_run_policy")
    parser_contract = load_f7_factor_parser_contract()
    flowpre_prefix = str(((parser_contract.get("rules") or {}).get("flowpre_usage") or {}).get("x_transform_prefix", "flowpre_"))
    flowgen_prefix = str(((parser_contract.get("rules") or {}).get("flowgen_usage") or {}).get("synthetic_policy_prefix", "flowgen_"))
    return {
        "x_transform": x_transform,
        "y_transform": y_transform,
        "synthetic_policy": synthetic_policy,
        "run_policy": run_policy,
        "flowpre_usage": x_transform.startswith(flowpre_prefix),
        "flowgen_usage": synthetic_policy.startswith(flowgen_prefix),
        "dataset_candidate_id_fallback": fallback_dataset_candidate_id,
        "run_spec_id_fallback": fallback_run_spec_id,
    }


def resolve_feature_namespace(
    *,
    model_family: str,
    has_input_projection: bool = False,
    has_latent_surface: bool = False,
    surface_hint: str | None = None,
) -> tuple[str | None, str | None]:
    family = str(model_family).lower()
    if family == "xgboost":
        if surface_hint == "xgb_shap":
            return "xgb_shap", "xgb_native_shap"
        return "xgb_perturbation", "semantic_bridge_perturbation"
    if family == "mlp":
        if has_input_projection:
            return "flowpre_projected_semantic_input", "semantic_bridge_perturbation"
        if surface_hint == "latent" or has_latent_surface:
            return "flowpre_latent", "mlp_flowpre_native_latent_perturbation"
        return "semantic_input", "semantic_bridge_perturbation"
    return None, None


def get_f7_analysis_contract_bundle() -> dict[str, Any]:
    class_ontology = load_f7_class_ontology()
    target_contract = load_f7_target_contract()
    artifact_contract = load_f7_artifact_persistence_contract()
    metric_grammar = load_f7_metric_grammar_contract()
    metric_availability = load_f7_metric_availability_contract()
    metric_aggregation = load_f7_metric_aggregation_contract()
    evaluation_population = load_f7_evaluation_population_contract()
    prediction_join = load_f7_prediction_row_join_contract()
    feature_schema = load_f7_feature_schema_contract()
    factor_parser = load_f7_factor_parser_contract()
    return {
        "class_ontology_contract_id": class_ontology["contract_id"],
        "class_ontology_contract_version": int(class_ontology["contract_version"]),
        "target_contract_id": target_contract["contract_id"],
        "target_contract_version": int(target_contract["contract_version"]),
        "artifact_policy_id": artifact_contract["artifact_policy_id"],
        "artifact_policy_version": int(artifact_contract["policy_version"]),
        "metric_grammar_contract_id": metric_grammar["contract_id"],
        "metric_grammar_contract_version": int(metric_grammar["contract_version"]),
        "metric_availability_contract_id": metric_availability["contract_id"],
        "metric_availability_contract_version": int(metric_availability["contract_version"]),
        "metric_aggregation_contract_id": metric_aggregation["contract_id"],
        "metric_aggregation_contract_version": int(metric_aggregation["contract_version"]),
        "evaluation_population_contract_id": evaluation_population["contract_id"],
        "evaluation_population_contract_version": int(evaluation_population["contract_version"]),
        "prediction_row_join_contract_id": prediction_join["contract_id"],
        "prediction_row_join_contract_version": int(prediction_join["contract_version"]),
        "prediction_row_join_key_kind": str(prediction_join["key_kind"]),
        "feature_schema_contract_id": feature_schema["contract_id"],
        "feature_schema_contract_version": int(feature_schema["contract_version"]),
        "factor_parser_contract_id": factor_parser["contract_id"],
        "factor_parser_contract_version": int(factor_parser["contract_version"]),
        "factor_parser_version": f"{factor_parser['contract_id']}::{factor_parser['contract_version']}",
        "metric_grammar_version": f"{metric_grammar['contract_id']}::{metric_grammar['contract_version']}",
        "lineage_aggregate_build_version": LINEAGE_AGGREGATE_BUILD_VERSION,
        "panel_build_version": PANEL_BUILD_VERSION,
        "target_name": str(target_contract["target_name"]),
        "target_space": str(target_contract["target_space_id"]),
        "target_unit_public": str(target_contract["target_unit_public"]),
        "class_ids": [int(row["class_id"]) for row in class_ontology.get("classes", [])],
        "class_labels_public": {
            int(row["class_id"]): str(row["class_label_public"])
            for row in class_ontology.get("classes", [])
        },
        "feature_schema_namespaces": list((feature_schema.get("namespaces") or {}).keys()),
        "metric_scopes": list(metric_grammar.get("metric_scopes") or []),
    }


def get_f7_analysis_contract_bundle_with_paths() -> dict[str, Any]:
    bundle = get_f7_analysis_contract_bundle()
    bundle.update(
        {
            "class_ontology_contract_path": path_relative_to_root(_CLASS_ONTOLOGY_PATH),
            "target_contract_path": path_relative_to_root(_TARGET_CONTRACT_PATH),
            "artifact_policy_contract_path": path_relative_to_root(_ARTIFACT_PERSISTENCE_CONTRACT_PATH),
            "metric_grammar_contract_path": path_relative_to_root(_METRIC_GRAMMAR_CONTRACT_PATH),
            "metric_availability_contract_path": path_relative_to_root(_METRIC_AVAILABILITY_CONTRACT_PATH),
            "metric_aggregation_contract_path": path_relative_to_root(_METRIC_AGGREGATION_CONTRACT_PATH),
            "evaluation_population_contract_path": path_relative_to_root(_EVALUATION_POPULATION_CONTRACT_PATH),
            "prediction_row_join_contract_path": path_relative_to_root(_PREDICTION_ROW_JOIN_CONTRACT_PATH),
            "feature_schema_contract_path": path_relative_to_root(_FEATURE_SCHEMA_CONTRACT_PATH),
            "factor_parser_contract_path": path_relative_to_root(_FACTOR_PARSER_CONTRACT_PATH),
        }
    )
    return bundle


def derive_comparison_group_id(
    *,
    dataset_candidate_id: str | None,
    run_spec_id: str | None,
    explicit_group_id: str | None = None,
    contrast_id: str | None = None,
) -> str | None:
    if explicit_group_id:
        return str(explicit_group_id)
    if dataset_candidate_id is None:
        return None
    resolved_contrast = contrast_id or run_spec_id
    if resolved_contrast is None:
        return None
    return f"cmp__{dataset_candidate_id}__{resolved_contrast}"


def derive_trial_id(
    *,
    campaign_id: str | None,
    dataset_candidate_id: str | None,
    run_spec_id: str | None,
    seed: int | None,
) -> str | None:
    if campaign_id is None or dataset_candidate_id is None or run_spec_id is None or seed is None:
        return None
    return f"trial__{campaign_id}__{dataset_candidate_id}__{run_spec_id}__seed-{int(seed)}"


def derive_meta_context(
    *,
    model_family: str,
    dataset_manifest_path: str | Path | None,
    base_config_id: str | None,
    run_level_axes: Mapping[str, Any] | None,
    seed: int | None,
    seed_set_id: str | None,
    seed_panel_path: str | Path | None = None,
    dataset_candidate_id: str | None = None,
    comparison_group_id: str | None = None,
    campaign_id: str | None = None,
    run_spec_id: str | None = None,
    trial_id: str | None = None,
    contrast_id: str | None = None,
) -> dict[str, Any]:
    grammar = load_f7_meta_grammar()
    resolved_campaign_id = str(campaign_id or grammar.get("campaign_id"))
    resolved_dataset_candidate_id = resolve_dataset_candidate_id(
        dataset_manifest_path=dataset_manifest_path,
        dataset_candidate_id=dataset_candidate_id,
    )
    resolved_run_spec_id = run_spec_id or derive_run_spec_id(
        model_family=model_family,
        base_config_id=base_config_id,
        run_level_axes=run_level_axes,
    )
    resolved_comparison_group_id = derive_comparison_group_id(
        dataset_candidate_id=resolved_dataset_candidate_id,
        run_spec_id=resolved_run_spec_id,
        explicit_group_id=comparison_group_id,
        contrast_id=contrast_id,
    )
    resolved_trial_id = trial_id or derive_trial_id(
        campaign_id=resolved_campaign_id,
        dataset_candidate_id=resolved_dataset_candidate_id,
        run_spec_id=resolved_run_spec_id,
        seed=seed,
    )
    return {
        "campaign_id": resolved_campaign_id,
        "dataset_candidate_id": resolved_dataset_candidate_id,
        "run_spec_id": resolved_run_spec_id,
        "comparison_group_id": resolved_comparison_group_id,
        "trial_id": resolved_trial_id,
        "replication_index": resolve_replication_index(
            seed=seed,
            seed_set_id=seed_set_id,
            seed_panel_path=seed_panel_path,
        ),
        "seed_panel_version": int(
            load_seed_panel_by_ref(seed_set_id=seed_set_id, seed_panel_path=seed_panel_path).get("panel_version", 1)
        ),
        "seed_panel_path": (
            None
            if _resolve_seed_panel_path(seed_set_id, seed_panel_path) is None
            else _normalize_manifest_ref(_resolve_seed_panel_path(seed_set_id, seed_panel_path))
        ),
    }


def json_blob(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, ensure_ascii=True)
