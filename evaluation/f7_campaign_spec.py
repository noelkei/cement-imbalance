from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from data.utils import ROOT_PATH, path_relative_to_root
from evaluation.meta_context import (
    derive_comparison_group_id,
    derive_trial_id,
    get_f7_analysis_contract_bundle_with_paths,
    load_f7_meta_grammar,
    parse_f7_factor_fields,
    load_seed_panel_by_ref,
)


DEFAULT_SPEC_PATH = Path(ROOT_PATH) / "config" / "f7_campaign_spec_v1.yaml"


@dataclass(frozen=True)
class MaterializedCampaignSpec:
    spec: dict[str, Any]
    dataset_candidates: list[dict[str, Any]]
    run_specs: list[dict[str, Any]]
    trials: list[dict[str, Any]]
    expansion_manifest: dict[str, Any]
    output_paths: dict[str, Path]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def load_f7_campaign_spec(path: str | Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    spec_path = Path(path)
    if not spec_path.is_absolute():
        spec_path = Path(ROOT_PATH) / spec_path
    payload = _load_yaml_mapping(spec_path)
    spec = payload.get("campaign_spec")
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid campaign_spec payload in {spec_path}")
    return spec


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_rel_path(path: str | Path) -> str:
    return path_relative_to_root(path)


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(ROOT_PATH) / candidate


def _resolve_output_path(spec: dict[str, Any], key: str, output_root: str | Path | None = None) -> Path:
    rel = Path(str(spec["outputs"][key]))
    if output_root is None:
        return _resolve_repo_path(rel)
    root = Path(output_root)
    return root / rel.name


def _resolve_seed_panel_path(spec: dict[str, Any]) -> Path:
    return _resolve_repo_path(spec.get("seed_panel_path") or Path("config") / f"{spec['seed_set_id']}.yaml")


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV to {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stable_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _load_inventory_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_manifest_payload(path: str | Path) -> dict[str, Any]:
    manifest_path = _resolve_repo_path(path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _validate_required_file_from_id(config_id: str) -> None:
    candidate = Path(ROOT_PATH) / "config" / f"{config_id}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"Required config/contract file not found for id '{config_id}': {candidate}")


def _validate_spec_consistency(spec: dict[str, Any]) -> None:
    grammar = load_f7_meta_grammar()
    seed_panel = load_seed_panel_by_ref(
        seed_set_id=str(spec.get("seed_set_id")),
        seed_panel_path=spec.get("seed_panel_path"),
    )
    campaign_kind = str(spec.get("campaign_kind", "primary"))
    if not spec.get("campaign_id"):
        raise ValueError("campaign_id is required")
    if str(spec.get("meta_grammar_id")) != "f7_meta_grammar_v1":
        raise ValueError("campaign spec must reference meta_grammar_id=f7_meta_grammar_v1")
    if str(spec.get("seed_set_id")) != str(seed_panel.get("seed_set_id")):
        raise ValueError("campaign spec seed_set_id must match the declared seed panel")
    if campaign_kind not in {"primary", "extension"}:
        raise ValueError("campaign_kind must be 'primary' or 'extension'")
    extension_type = spec.get("extension_type")
    if campaign_kind == "primary":
        if extension_type not in {None, "", "null"}:
            raise ValueError("Primary campaigns must not declare extension_type")
        if spec.get("parent_campaign_id") not in {None, "", "null"}:
            raise ValueError("Primary campaigns must not declare parent_campaign_id")
        if str(spec.get("root_campaign_id")) != str(spec.get("campaign_id")):
            raise ValueError("Primary campaigns must set root_campaign_id equal to campaign_id")
    else:
        if str(extension_type) != "seed_extension":
            raise ValueError("Extension campaigns currently support only extension_type=seed_extension")
        if not spec.get("parent_campaign_id"):
            raise ValueError("Extension campaigns must declare parent_campaign_id")
        if not spec.get("root_campaign_id"):
            raise ValueError("Extension campaigns must declare root_campaign_id")
        if str(spec.get("campaign_id")) == str(spec.get("parent_campaign_id")):
            raise ValueError("Extension campaign_id must differ from parent_campaign_id")
    if not spec.get("campaign_lineage_id"):
        raise ValueError("campaign_lineage_id is required")
    if not spec.get("pooling_group_id"):
        raise ValueError("pooling_group_id is required")
    required_global_contracts = dict(spec.get("required_global_contracts") or {})
    for config_id in (
        str(required_global_contracts.get("raw_metric_contract_id")),
        str(required_global_contracts.get("artifact_policy_id")),
        str(spec["mlp"]["mlp_base_config_id"]),
        str(spec["mlp"]["contracts"]["mlp_interpretability_contract_id"]),
        str(spec["xgboost"]["xgb_base_config_id"]),
        str(spec["xgboost"]["contracts"]["xgb_interpretability_contract_id"]),
        str(spec.get("meta_grammar_id")),
    ):
        _validate_required_file_from_id(config_id)
    seed_panel_path = _resolve_seed_panel_path(spec)
    if not seed_panel_path.exists():
        raise FileNotFoundError(f"Seed panel path not found: {seed_panel_path}")


def _row_matches_filter(row: dict[str, Any], row_filter: dict[str, Any]) -> bool:
    for key, expected in row_filter.items():
        if str(row.get(key)) != str(expected):
            return False
    return True


def _normalize_dataset_candidate_allowlist(values: Any) -> set[str] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError("include_dataset_candidate_ids must be a list when provided")
    return {str(value).strip() for value in values if str(value).strip()}


def _normalize_dataset_candidate_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    source_path = _resolve_repo_path(spec["canonical_sources"]["dataset_inventory_path"])
    inventory_rows = _load_inventory_rows(source_path)
    normalized_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for family_key in ("mlp", "xgboost"):
        family_spec = spec[family_key]
        if not bool(family_spec.get("enabled", True)):
            continue
        family_filter = dict(family_spec.get("dataset_filter") or {})
        allowlist = _normalize_dataset_candidate_allowlist(family_spec.get("include_dataset_candidate_ids"))
        family_source = _resolve_repo_path(family_spec["dataset_inventory_source"])
        if family_source != source_path:
            raise ValueError("Block 12 expects a single canonical dataset inventory source across families")
        for row in inventory_rows:
            if not _row_matches_filter(row, family_filter):
                continue
            dataset_candidate_id = str(row.get("inventory_dataset_id") or "").strip()
            manifest_path = str(row.get("manifest_path") or "").strip()
            if not dataset_candidate_id or not manifest_path:
                raise ValueError("Dataset inventory row is missing inventory_dataset_id or manifest_path")
            if allowlist is not None and dataset_candidate_id not in allowlist:
                continue
            if dataset_candidate_id in seen_ids:
                raise ValueError(f"Duplicate dataset_candidate_id in normalized inventory: {dataset_candidate_id}")
            manifest = _load_manifest_payload(manifest_path)
            axes = dict(manifest.get("dataset_level_axes") or {})
            normalized_rows.append(
                {
                    "campaign_id": spec["campaign_id"],
                    "campaign_spec_id": spec["campaign_spec_id"],
                    "model_family": family_spec["model_family"],
                    "dataset_candidate_id": dataset_candidate_id,
                    "inventory_dataset_id": dataset_candidate_id,
                    "dataset_role": row.get("dataset_role"),
                    "dataset_name": manifest.get("dataset_name"),
                    "split_id": manifest.get("split_id"),
                    "dataset_manifest_path": _normalize_rel_path(manifest_path),
                    "x_transform": axes.get("x_transform"),
                    "y_transform": axes.get("y_transform"),
                    "synthetic_policy": axes.get("synthetic_policy") or row.get("synthetic_policy"),
                    "synthetic_policy_id": manifest.get("synthetic_policy_id"),
                    "source_bundle_manifest_path": row.get("source_bundle_manifest_path") or None,
                    "shared_pool_manifest_path": row.get("shared_pool_manifest_path") or None,
                    "row_counts_train": row.get("row_counts_train"),
                    "row_counts_val": row.get("row_counts_val"),
                    "row_counts_test": row.get("row_counts_test"),
                    "class_counts_train": row.get("class_counts_train"),
                    "status": row.get("status"),
                    "hash_train_X": row.get("hash_train_X"),
                    "hash_train_y": row.get("hash_train_y"),
                    "hash_train_removed": row.get("hash_train_removed"),
                }
            )
            seen_ids.add(dataset_candidate_id)
        if allowlist is not None:
            observed = {
                str(row["dataset_candidate_id"])
                for row in normalized_rows
                if str(row["model_family"]) == str(family_spec["model_family"])
            }
            missing = sorted(allowlist - observed)
            if missing:
                raise ValueError(
                    f"Campaign spec requested dataset candidates not found for family {family_spec['model_family']}: {missing}"
                )

    normalized_rows.sort(key=lambda row: (str(row["model_family"]), str(row["dataset_candidate_id"])))
    return normalized_rows


def _build_mlp_run_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    family_spec = spec["mlp"]
    base_config_id = str(family_spec["mlp_base_config_id"])
    allow_synth_values = list(family_spec["axes"]["allow_synth"])
    if allow_synth_values != [True]:
        raise ValueError("F7 block 12 assumes exactly one allow_synth value for MLP: true")
    loss_policy_map = {
        "overall_rmse": ("overall", "rmse"),
        "per_class_equal_rmse": ("per_class_equal", "rmse"),
        "per_class_equal_rrmse": ("per_class_equal", "rrmse"),
    }
    rows: list[dict[str, Any]] = []
    for family in family_spec["training_behavior_families"]:
        for loss_policy_id in family_spec["axes"]["loss_policy"]:
            if loss_policy_id not in loss_policy_map:
                raise ValueError(f"Unsupported MLP loss policy in campaign spec: {loss_policy_id}")
            loss_reduction, regression_group_metric = loss_policy_map[loss_policy_id]
            run_spec_id = (
                f"runspec__mlp__{base_config_id}__{family['batch_policy']}__"
                f"{family['cycling_policy']}__{loss_policy_id}__allow_synth-true"
            )
            rows.append(
                {
                    "campaign_id": spec["campaign_id"],
                    "campaign_spec_id": spec["campaign_spec_id"],
                    "model_family": "mlp",
                    "run_spec_id": run_spec_id,
                    "base_config_id": base_config_id,
                    "objective_metric_id": family_spec["objective_metric_id"],
                    "training_behavior_family_id": family["family_id"],
                    "batch_policy_id": family["batch_policy"],
                    "cycling_policy_id": family["cycling_policy"],
                    "dataloader_mode": family["dataloader_mode"],
                    "cycle_reals": bool(family["cycle_reals"]),
                    "loss_policy_id": loss_policy_id,
                    "loss_reduction": loss_reduction,
                    "regression_group_metric": regression_group_metric,
                    "allow_synth": True,
                    "raw_metric_contract_id": spec["required_global_contracts"]["raw_metric_contract_id"],
                    "artifact_policy_id": spec["required_global_contracts"]["artifact_policy_id"],
                    "family_interpretability_contract_id": family_spec["contracts"]["mlp_interpretability_contract_id"],
                    "native_interpretability_layer": family_spec["interpretability_layers"]["native"],
                    "bridge_interpretability_layer": family_spec["interpretability_layers"]["bridge"],
                }
            )
    return rows


def _build_xgb_run_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    family_spec = spec["xgboost"]
    base_config_id = str(family_spec["xgb_base_config_id"])
    run_spec_id = f"runspec__xgb__{base_config_id}"
    return [
        {
            "campaign_id": spec["campaign_id"],
            "campaign_spec_id": spec["campaign_spec_id"],
            "model_family": "xgboost",
            "run_spec_id": run_spec_id,
            "base_config_id": base_config_id,
            "objective_metric_id": family_spec["objective_metric_id"],
            "training_behavior_family_id": "canonical",
            "batch_policy_id": None,
            "cycling_policy_id": None,
            "dataloader_mode": None,
            "cycle_reals": None,
            "loss_policy_id": None,
            "loss_reduction": None,
            "regression_group_metric": None,
            "allow_synth": None,
            "raw_metric_contract_id": spec["required_global_contracts"]["raw_metric_contract_id"],
            "artifact_policy_id": spec["required_global_contracts"]["artifact_policy_id"],
            "family_interpretability_contract_id": family_spec["contracts"]["xgb_interpretability_contract_id"],
            "native_interpretability_layer": family_spec["interpretability_layers"]["native"],
            "bridge_interpretability_layer": family_spec["interpretability_layers"]["bridge"],
        }
    ]


def _build_run_spec_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _build_mlp_run_specs(spec) + _build_xgb_run_specs(spec)
    contract_bundle = get_f7_analysis_contract_bundle_with_paths()
    seen: set[str] = set()
    for row in rows:
        run_spec_id = str(row["run_spec_id"])
        if run_spec_id in seen:
            raise ValueError(f"Duplicate run_spec_id in campaign spec expansion: {run_spec_id}")
        seen.add(run_spec_id)
        row.update(
            {
                "target_contract_id": contract_bundle["target_contract_id"],
                "target_contract_version": contract_bundle["target_contract_version"],
                "class_ontology_contract_id": contract_bundle["class_ontology_contract_id"],
                "class_ontology_contract_version": contract_bundle["class_ontology_contract_version"],
                "metric_grammar_contract_id": contract_bundle["metric_grammar_contract_id"],
                "metric_grammar_contract_version": contract_bundle["metric_grammar_contract_version"],
                "metric_availability_contract_id": contract_bundle["metric_availability_contract_id"],
                "metric_availability_contract_version": contract_bundle["metric_availability_contract_version"],
                "metric_aggregation_contract_id": contract_bundle["metric_aggregation_contract_id"],
                "metric_aggregation_contract_version": contract_bundle["metric_aggregation_contract_version"],
                "evaluation_population_contract_id": contract_bundle["evaluation_population_contract_id"],
                "evaluation_population_contract_version": contract_bundle["evaluation_population_contract_version"],
                "prediction_row_join_contract_id": contract_bundle["prediction_row_join_contract_id"],
                "prediction_row_join_contract_version": contract_bundle["prediction_row_join_contract_version"],
                "feature_schema_contract_id": contract_bundle["feature_schema_contract_id"],
                "feature_schema_contract_version": contract_bundle["feature_schema_contract_version"],
                "factor_parser_contract_id": contract_bundle["factor_parser_contract_id"],
                "factor_parser_contract_version": contract_bundle["factor_parser_contract_version"],
            }
        )
    rows.sort(key=lambda row: (str(row["model_family"]), str(row["run_spec_id"])))
    return rows


def _build_trial_rows(
    spec: dict[str, Any],
    dataset_candidates: list[dict[str, Any]],
    run_specs: list[dict[str, Any]],
    *,
    panel_build_timestamp: str,
) -> list[dict[str, Any]]:
    contract_bundle = get_f7_analysis_contract_bundle_with_paths()
    seed_panel = load_seed_panel_by_ref(
        seed_set_id=str(spec.get("seed_set_id")),
        seed_panel_path=spec.get("seed_panel_path"),
    )
    seeds = list(seed_panel.get("seeds", []))
    rows: list[dict[str, Any]] = []
    run_specs_by_family: dict[str, list[dict[str, Any]]] = {}
    for run_spec in run_specs:
        run_specs_by_family.setdefault(str(run_spec["model_family"]), []).append(run_spec)

    for dataset in dataset_candidates:
        family = str(dataset["model_family"])
        for run_spec in run_specs_by_family.get(family, []):
            contrast_id = str(run_spec["run_spec_id"])
            lineage_trial_group_id = (
                f"lineage_group__{spec['campaign_lineage_id']}__{family}__"
                f"{dataset['dataset_candidate_id']}__{run_spec['run_spec_id']}"
            )
            comparison_group_id = derive_comparison_group_id(
                dataset_candidate_id=str(dataset["dataset_candidate_id"]),
                run_spec_id=str(run_spec["run_spec_id"]),
                contrast_id=contrast_id,
            )
            for seed_row in seeds:
                seed_value = int(seed_row["value"])
                replication_index = int(seed_row["index"])
                trial_id = derive_trial_id(
                    campaign_id=str(spec["campaign_id"]),
                    dataset_candidate_id=str(dataset["dataset_candidate_id"]),
                    run_spec_id=str(run_spec["run_spec_id"]),
                    seed=seed_value,
                )
                parsed_factors = parse_f7_factor_fields(
                    model_family=family,
                    dataset_level_axes={
                        "x_transform": dataset["x_transform"],
                        "y_transform": dataset["y_transform"],
                        "synthetic_policy": dataset["synthetic_policy"],
                    },
                    run_level_axes={
                        "batch_policy_id": run_spec["batch_policy_id"],
                        "cycling_policy_id": run_spec["cycling_policy_id"],
                        "loss_policy_id": run_spec["loss_policy_id"],
                        "allow_synth": run_spec["allow_synth"],
                    },
                    fallback_dataset_candidate_id=str(dataset["dataset_candidate_id"]),
                    fallback_run_spec_id=str(run_spec["run_spec_id"]),
                )
                rows.append(
                    {
                        "campaign_id": spec["campaign_id"],
                        "campaign_spec_id": spec["campaign_spec_id"],
                        "campaign_kind": spec["campaign_kind"],
                        "extension_type": spec.get("extension_type"),
                        "campaign_lineage_id": spec["campaign_lineage_id"],
                        "root_campaign_id": spec["root_campaign_id"],
                        "parent_campaign_id": spec.get("parent_campaign_id"),
                        "pooling_group_id": spec["pooling_group_id"],
                        "eligible_for_pooled_seed_analysis": bool(spec["eligible_for_pooled_seed_analysis"]),
                        "is_primary_analysis_campaign": bool(spec["is_primary_analysis_campaign"]),
                        "model_family": family,
                        "dataset_candidate_id": dataset["dataset_candidate_id"],
                        "run_spec_id": run_spec["run_spec_id"],
                        "trial_id": trial_id,
                        "comparison_group_id": comparison_group_id,
                        "lineage_trial_group_id": lineage_trial_group_id,
                        "contrast_id": contrast_id,
                        "seed_set_id": spec["seed_set_id"],
                        "seed_panel_path": _normalize_rel_path(_resolve_seed_panel_path(spec)),
                        "seed": seed_value,
                        "replication_index": replication_index,
                        "expected_seed_count": len(seeds),
                        "run_mode": spec["run_mode"],
                        "allow_test_holdout": bool(spec["allow_test_holdout"]),
                        "test_enabled": bool(spec["test_enabled"]),
                        "dataset_manifest_path": dataset["dataset_manifest_path"],
                        "dataset_name": dataset["dataset_name"],
                        "split_id": dataset["split_id"],
                        "x_transform": dataset["x_transform"],
                        "y_transform": dataset["y_transform"],
                        "synthetic_policy": dataset["synthetic_policy"],
                        "synthetic_policy_id": dataset["synthetic_policy_id"],
                        "base_config_id": run_spec["base_config_id"],
                        "objective_metric_id": run_spec["objective_metric_id"],
                        "raw_metric_contract_id": run_spec["raw_metric_contract_id"],
                        "artifact_policy_id": run_spec["artifact_policy_id"],
                        "family_interpretability_contract_id": run_spec["family_interpretability_contract_id"],
                        "native_interpretability_layer": run_spec["native_interpretability_layer"],
                        "bridge_interpretability_layer": run_spec["bridge_interpretability_layer"],
                        "training_behavior_family_id": run_spec["training_behavior_family_id"],
                        "batch_policy_id": run_spec["batch_policy_id"],
                        "cycling_policy_id": run_spec["cycling_policy_id"],
                        "dataloader_mode": run_spec["dataloader_mode"],
                        "cycle_reals": run_spec["cycle_reals"],
                        "loss_policy_id": run_spec["loss_policy_id"],
                        "loss_reduction": run_spec["loss_reduction"],
                        "regression_group_metric": run_spec["regression_group_metric"],
                        "allow_synth": run_spec["allow_synth"],
                        "x_transform": parsed_factors["x_transform"],
                        "y_transform": parsed_factors["y_transform"],
                        "synthetic_policy": parsed_factors["synthetic_policy"],
                        "run_policy": parsed_factors["run_policy"],
                        "flowpre_usage": bool(parsed_factors["flowpre_usage"]),
                        "flowgen_usage": bool(parsed_factors["flowgen_usage"]),
                        "class_ontology_contract_id": contract_bundle["class_ontology_contract_id"],
                        "class_ontology_contract_version": contract_bundle["class_ontology_contract_version"],
                        "target_contract_id": contract_bundle["target_contract_id"],
                        "target_contract_version": contract_bundle["target_contract_version"],
                        "metric_grammar_contract_id": contract_bundle["metric_grammar_contract_id"],
                        "metric_grammar_contract_version": contract_bundle["metric_grammar_contract_version"],
                        "metric_availability_contract_id": contract_bundle["metric_availability_contract_id"],
                        "metric_availability_contract_version": contract_bundle["metric_availability_contract_version"],
                        "metric_aggregation_contract_id": contract_bundle["metric_aggregation_contract_id"],
                        "metric_aggregation_contract_version": contract_bundle["metric_aggregation_contract_version"],
                        "evaluation_population_contract_id": contract_bundle["evaluation_population_contract_id"],
                        "evaluation_population_contract_version": contract_bundle["evaluation_population_contract_version"],
                        "prediction_row_join_contract_id": contract_bundle["prediction_row_join_contract_id"],
                        "prediction_row_join_contract_version": contract_bundle["prediction_row_join_contract_version"],
                        "prediction_row_join_key_kind": contract_bundle["prediction_row_join_key_kind"],
                        "feature_schema_contract_id": contract_bundle["feature_schema_contract_id"],
                        "feature_schema_contract_version": contract_bundle["feature_schema_contract_version"],
                        "factor_parser_contract_id": contract_bundle["factor_parser_contract_id"],
                        "factor_parser_contract_version": contract_bundle["factor_parser_contract_version"],
                        "factor_parser_version": contract_bundle["factor_parser_version"],
                        "metric_grammar_version": contract_bundle["metric_grammar_version"],
                        "lineage_aggregate_build_version": contract_bundle["lineage_aggregate_build_version"],
                        "panel_build_version": contract_bundle["panel_build_version"],
                        "panel_build_timestamp": panel_build_timestamp,
                        "target_name": contract_bundle["target_name"],
                        "target_space": contract_bundle["target_space"],
                        "target_unit_public": contract_bundle["target_unit_public"],
                    }
                )
    rows.sort(key=lambda row: (str(row["model_family"]), str(row["dataset_candidate_id"]), str(row["run_spec_id"]), int(row["seed"])))
    return rows


def _validate_counts(spec: dict[str, Any], dataset_candidates: list[dict[str, Any]], run_specs: list[dict[str, Any]], trials: list[dict[str, Any]]) -> None:
    expected = dict(spec.get("expected_counts") or {})
    expected_dataset = dict(expected.get("dataset_candidates") or {})
    expected_run_specs = dict(expected.get("run_specs") or {})
    expected_trials = dict(expected.get("trials") or {})

    def _count(rows: list[dict[str, Any]], family: str) -> int:
        return sum(1 for row in rows if str(row["model_family"]) == family)

    observed = {
        "dataset_candidates": {
            "mlp": _count(dataset_candidates, "mlp"),
            "xgboost": _count(dataset_candidates, "xgboost"),
            "total": len(dataset_candidates),
        },
        "run_specs": {
            "mlp": _count(run_specs, "mlp"),
            "xgboost": _count(run_specs, "xgboost"),
            "total": len(run_specs),
        },
        "trials": {
            "mlp": _count(trials, "mlp"),
            "xgboost": _count(trials, "xgboost"),
            "total": len(trials),
        },
    }
    for group_name, expected_group in (
        ("dataset_candidates", expected_dataset),
        ("run_specs", expected_run_specs),
        ("trials", expected_trials),
    ):
        for key, expected_value in expected_group.items():
            if int(observed[group_name][key]) != int(expected_value):
                raise ValueError(
                    f"Unexpected {group_name} count for {key}: "
                    f"observed={observed[group_name][key]} expected={expected_value}"
                )


def _validate_uniqueness(rows: list[dict[str, Any]], key: str) -> None:
    seen: set[str] = set()
    for row in rows:
        value = str(row[key])
        if value in seen:
            raise ValueError(f"Duplicate {key}: {value}")
        seen.add(value)


def materialize_f7_campaign_spec(
    *,
    spec_path: str | Path = DEFAULT_SPEC_PATH,
    output_root: str | Path | None = None,
    write_outputs: bool = True,
) -> MaterializedCampaignSpec:
    spec = load_f7_campaign_spec(spec_path)
    _validate_spec_consistency(spec)

    dataset_candidates = _normalize_dataset_candidate_rows(spec)
    run_specs = _build_run_spec_rows(spec)
    panel_build_timestamp = datetime.now(timezone.utc).isoformat()
    trials = _build_trial_rows(
        spec,
        dataset_candidates,
        run_specs,
        panel_build_timestamp=panel_build_timestamp,
    )

    _validate_uniqueness(dataset_candidates, "dataset_candidate_id")
    _validate_uniqueness(run_specs, "run_spec_id")
    _validate_uniqueness(trials, "trial_id")
    _validate_counts(spec, dataset_candidates, run_specs, trials)

    dataset_output_path = _resolve_output_path(spec, "dataset_candidate_inventory_path", output_root)
    run_spec_output_path = _resolve_output_path(spec, "run_spec_inventory_path", output_root)
    trial_output_path = _resolve_output_path(spec, "trial_inventory_path", output_root)
    manifest_output_path = _resolve_output_path(spec, "expansion_manifest_path", output_root)
    expected_replication_output_path = manifest_output_path.with_name(
        f"{manifest_output_path.stem}_expected_replication.json"
    )

    source_inventory_path = _resolve_repo_path(spec["canonical_sources"]["dataset_inventory_path"])
    spec_source_path = _resolve_repo_path(spec_path)
    contract_bundle = {
        **get_f7_analysis_contract_bundle_with_paths(),
        "panel_build_timestamp": panel_build_timestamp,
    }
    structural_groups = sorted(
        {
            f"lineage_group__{spec['campaign_lineage_id']}__{row['model_family']}__{row['dataset_candidate_id']}__{row['run_spec_id']}"
            for row in trials
        }
    )
    expected_replication_manifest = {
        "campaign_id": spec["campaign_id"],
        "campaign_lineage_id": spec["campaign_lineage_id"],
        "root_campaign_id": spec["root_campaign_id"],
        "parent_campaign_id": spec.get("parent_campaign_id"),
        "seed_set_id": spec["seed_set_id"],
        "seed_panel_path": _normalize_rel_path(_resolve_seed_panel_path(spec)),
        "expected_seed_values": [int(row["value"]) for row in load_seed_panel_by_ref(
            seed_set_id=str(spec.get("seed_set_id")),
            seed_panel_path=spec.get("seed_panel_path"),
        ).get("seeds", [])],
        "expected_seed_count": len(load_seed_panel_by_ref(
            seed_set_id=str(spec.get("seed_set_id")),
            seed_panel_path=spec.get("seed_panel_path"),
        ).get("seeds", [])),
        "expected_structural_group_ids": structural_groups,
        "expected_structural_group_count": len(structural_groups),
        "expected_seed_count_by_structural_group": {
            structural_group_id: len(load_seed_panel_by_ref(
                seed_set_id=str(spec.get("seed_set_id")),
                seed_panel_path=spec.get("seed_panel_path"),
            ).get("seeds", []))
            for structural_group_id in structural_groups
        },
        "panel_completeness_design": "complete" if bool(spec["eligible_for_pooled_seed_analysis"]) else "partial",
    }
    manifest = {
        "campaign_id": spec["campaign_id"],
        "campaign_spec_id": spec["campaign_spec_id"],
        "campaign_kind": spec["campaign_kind"],
        "extension_type": spec.get("extension_type"),
        "campaign_lineage_id": spec["campaign_lineage_id"],
        "root_campaign_id": spec["root_campaign_id"],
        "parent_campaign_id": spec.get("parent_campaign_id"),
        "pooling_group_id": spec["pooling_group_id"],
        "eligible_for_pooled_seed_analysis": bool(spec["eligible_for_pooled_seed_analysis"]),
        "is_primary_analysis_campaign": bool(spec["is_primary_analysis_campaign"]),
        "campaign_scope": spec["campaign_scope"],
        "materialized_at_utc": panel_build_timestamp,
        "counts": {
            "dataset_candidates": {
                "mlp": sum(1 for row in dataset_candidates if row["model_family"] == "mlp"),
                "xgboost": sum(1 for row in dataset_candidates if row["model_family"] == "xgboost"),
                "total": len(dataset_candidates),
            },
            "run_specs": {
                "mlp": sum(1 for row in run_specs if row["model_family"] == "mlp"),
                "xgboost": sum(1 for row in run_specs if row["model_family"] == "xgboost"),
                "total": len(run_specs),
            },
            "trials": {
                "mlp": sum(1 for row in trials if row["model_family"] == "mlp"),
                "xgboost": sum(1 for row in trials if row["model_family"] == "xgboost"),
                "total": len(trials),
            },
            "seed_panel_size": len(
                load_seed_panel_by_ref(
                    seed_set_id=str(spec.get("seed_set_id")),
                    seed_panel_path=spec.get("seed_panel_path"),
                ).get("seeds", [])
            ),
        },
        "input_fingerprints": {
            "campaign_spec": {
                "path": _normalize_rel_path(spec_source_path),
                "sha256": _sha256_file(spec_source_path),
            },
            "meta_grammar": {
                "path": _normalize_rel_path(Path(ROOT_PATH) / "config" / "f7_meta_grammar_v1.yaml"),
                "sha256": _sha256_file(Path(ROOT_PATH) / "config" / "f7_meta_grammar_v1.yaml"),
            },
            "seed_panel": {
                "path": _normalize_rel_path(_resolve_seed_panel_path(spec)),
                "sha256": _sha256_file(_resolve_seed_panel_path(spec)),
            },
            "dataset_inventory": {
                "path": _normalize_rel_path(source_inventory_path),
                "sha256": _sha256_file(source_inventory_path),
            },
        },
        "analysis_contracts": contract_bundle,
        "output_artifacts": {
            "dataset_candidate_inventory_path": _normalize_rel_path(dataset_output_path),
            "run_spec_inventory_path": _normalize_rel_path(run_spec_output_path),
            "trial_inventory_path": _normalize_rel_path(trial_output_path),
            "expansion_manifest_path": _normalize_rel_path(manifest_output_path),
            "expected_replication_manifest_path": _normalize_rel_path(expected_replication_output_path),
        },
        "expected_replication": expected_replication_manifest,
        "family_interpretability_layers": {
            "mlp": {
                "native": spec["mlp"]["interpretability_layers"]["native"],
                "bridge": spec["mlp"]["interpretability_layers"]["bridge"],
            },
            "xgboost": {
                "native": spec["xgboost"]["interpretability_layers"]["native"],
                "bridge": spec["xgboost"]["interpretability_layers"]["bridge"],
            },
        },
    }
    if write_outputs:
        _write_csv_rows(dataset_output_path, dataset_candidates)
        _write_csv_rows(run_spec_output_path, run_specs)
        _write_csv_rows(trial_output_path, trials)
        manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        expected_replication_output_path.write_text(
            json.dumps(expected_replication_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return MaterializedCampaignSpec(
        spec=spec,
        dataset_candidates=dataset_candidates,
        run_specs=run_specs,
        trials=trials,
        expansion_manifest=manifest,
        output_paths={
            "dataset_candidate_inventory_path": dataset_output_path,
            "run_spec_inventory_path": run_spec_output_path,
            "trial_inventory_path": trial_output_path,
            "expansion_manifest_path": manifest_output_path,
            "expected_replication_manifest_path": expected_replication_output_path,
        },
    )
