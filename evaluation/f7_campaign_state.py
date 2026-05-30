from __future__ import annotations

import csv
import json
import os
import shutil
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from data.utils import ROOT_PATH, path_relative_to_root


EXECUTION_STATUS_PENDING = "pending"
EXECUTION_STATUS_RUNNING = "running"
EXECUTION_STATUS_COMPLETED = "completed"
EXECUTION_STATUS_FAILED = "failed"
EXECUTION_STATUS_BLOCKED = "blocked"
EXECUTION_STATUSES = {
    EXECUTION_STATUS_PENDING,
    EXECUTION_STATUS_RUNNING,
    EXECUTION_STATUS_COMPLETED,
    EXECUTION_STATUS_FAILED,
    EXECUTION_STATUS_BLOCKED,
}

VALIDITY_STATUS_VALID_F7 = "valid_f7"
VALIDITY_STATUS_INVALID_F7 = "invalid_f7"
VALIDITY_STATUS_UNKNOWN = "unknown"

CAMPAIGN_STATUS_OPEN = "open"
CAMPAIGN_STATUS_IN_PROGRESS = "in_progress"
CAMPAIGN_STATUS_CLOSED_SUCCESS = "closed_success"
CAMPAIGN_STATUS_CLOSED_WITH_FAILURES = "closed_with_failures"
CAMPAIGN_STATUS_ABORTED = "aborted"

FAILURE_REASON_PRECHECK_MANIFEST = "preflight_missing_manifest"
FAILURE_REASON_PRECHECK_CONFIG = "preflight_missing_config"
FAILURE_REASON_PRECHECK_CONTRACT = "preflight_missing_contract"
FAILURE_REASON_PRECHECK_INVALID_TRIAL = "preflight_invalid_trial"
FAILURE_REASON_LOCK_CONFLICT = "lock_conflict"
FAILURE_REASON_OOM = "oom"
FAILURE_REASON_EXCEPTION_TRAINING = "exception_training"
FAILURE_REASON_EXCEPTION_INTERPRETABILITY = "exception_interpretability"
FAILURE_REASON_EXCEPTION_RESULTS_PERSISTENCE = "exception_results_persistence"
FAILURE_REASON_INVALID_F7 = "invalid_f7"
FAILURE_REASON_MISSING_RUN_ARTIFACTS = "missing_run_artifacts_after_execution"
FAILURE_REASON_MANUAL_ABORT = "manual_abort"
FAILURE_REASON_PARENT_NOT_CLOSED = "parent_campaign_not_closed"
FAILURE_REASON_CODES = {
    FAILURE_REASON_PRECHECK_MANIFEST,
    FAILURE_REASON_PRECHECK_CONFIG,
    FAILURE_REASON_PRECHECK_CONTRACT,
    FAILURE_REASON_PRECHECK_INVALID_TRIAL,
    FAILURE_REASON_LOCK_CONFLICT,
    FAILURE_REASON_OOM,
    FAILURE_REASON_EXCEPTION_TRAINING,
    FAILURE_REASON_EXCEPTION_INTERPRETABILITY,
    FAILURE_REASON_EXCEPTION_RESULTS_PERSISTENCE,
    FAILURE_REASON_INVALID_F7,
    FAILURE_REASON_MISSING_RUN_ARTIFACTS,
    FAILURE_REASON_MANUAL_ABORT,
    FAILURE_REASON_PARENT_NOT_CLOSED,
}


@dataclass(frozen=True)
class CampaignPaths:
    campaign_id: str
    root: Path
    inputs_dir: Path
    state_dir: Path
    trial_state_dir: Path
    locks_dir: Path
    effective_configs_dir: Path
    tracebacks_dir: Path
    warnings_dir: Path
    registry_path: Path
    attempts_path: Path
    ledger_path: Path
    summary_path: Path
    campaign_manifest_path: Path
    campaign_closeout_path: Path
    preflight_report_path: Path


def derive_lineage_trial_group_id(
    *,
    campaign_lineage_id: str | None,
    model_family: str | None,
    dataset_candidate_id: str | None,
    run_spec_id: str | None,
) -> str | None:
    if not campaign_lineage_id or not model_family or not dataset_candidate_id or not run_spec_id:
        return None
    return (
        f"lineage_group__{campaign_lineage_id}__{model_family}__"
        f"{dataset_candidate_id}__{run_spec_id}"
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(ROOT_PATH) / candidate


def build_campaign_paths(campaign_id: str) -> CampaignPaths:
    root = Path(ROOT_PATH) / "outputs" / "campaigns" / campaign_id
    return CampaignPaths(
        campaign_id=campaign_id,
        root=root,
        inputs_dir=root / "inputs",
        state_dir=root / "state",
        trial_state_dir=root / "state" / "trials",
        locks_dir=root / "state" / "locks",
        effective_configs_dir=root / "effective_configs",
        tracebacks_dir=root / "tracebacks",
        warnings_dir=root / "warnings",
        registry_path=Path(ROOT_PATH) / "outputs" / "campaigns" / "f7_campaign_registry.csv",
        attempts_path=root / "trial_attempts.jsonl",
        ledger_path=root / "trial_ledger.csv",
        summary_path=root / "summary.json",
        campaign_manifest_path=root / "campaign_manifest.json",
        campaign_closeout_path=root / "campaign_closeout.json",
        preflight_report_path=root / "preflight_report.json",
    )


def ensure_campaign_dirs(paths: CampaignPaths) -> None:
    for directory in (
        paths.root,
        paths.inputs_dir,
        paths.state_dir,
        paths.trial_state_dir,
        paths.locks_dir,
        paths.effective_configs_dir,
        paths.tracebacks_dir,
        paths.warnings_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_needed(src: str | Path, dst: Path) -> None:
    source = _repo_path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == dst.resolve():
        return
    shutil.copy2(source, dst)


def freeze_campaign_inputs(
    *,
    paths: CampaignPaths,
    spec_source_path: str | Path,
    materialized_output_paths: dict[str, Path],
) -> dict[str, str]:
    ensure_campaign_dirs(paths)
    frozen_paths = {
        "campaign_spec_path": paths.inputs_dir / Path(spec_source_path).name,
        "dataset_candidate_inventory_path": paths.inputs_dir / materialized_output_paths["dataset_candidate_inventory_path"].name,
        "run_spec_inventory_path": paths.inputs_dir / materialized_output_paths["run_spec_inventory_path"].name,
        "trial_inventory_path": paths.inputs_dir / materialized_output_paths["trial_inventory_path"].name,
        "expansion_manifest_path": paths.inputs_dir / materialized_output_paths["expansion_manifest_path"].name,
        "expected_replication_manifest_path": paths.inputs_dir / materialized_output_paths["expected_replication_manifest_path"].name,
    }
    _copy_if_needed(spec_source_path, frozen_paths["campaign_spec_path"])
    for key, path in materialized_output_paths.items():
        target_key = key
        frozen_key = {
            "dataset_candidate_inventory_path": "dataset_candidate_inventory_path",
            "run_spec_inventory_path": "run_spec_inventory_path",
            "trial_inventory_path": "trial_inventory_path",
            "expansion_manifest_path": "expansion_manifest_path",
            "expected_replication_manifest_path": "expected_replication_manifest_path",
        }[target_key]
        _copy_if_needed(path, frozen_paths[frozen_key])
    return {key: path_relative_to_root(value) for key, value in frozen_paths.items()}


def load_trial_inventory_from_csv(path: str | Path) -> list[dict[str, str]]:
    with _repo_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_frozen_trial_inventory(paths: CampaignPaths) -> list[dict[str, str]]:
    trial_inventory_path = paths.inputs_dir / "f7_campaign_trials_v1.csv"
    if not trial_inventory_path.exists():
        trial_inventory_candidates = sorted(paths.inputs_dir.glob("*trial*.csv"))
        if len(trial_inventory_candidates) != 1:
            raise FileNotFoundError("Could not resolve frozen trial inventory under campaign inputs/")
        trial_inventory_path = trial_inventory_candidates[0]
    return load_trial_inventory_from_csv(trial_inventory_path)


def build_initial_trial_state(trial_row: dict[str, Any]) -> dict[str, Any]:
    lineage_trial_group_id = derive_lineage_trial_group_id(
        campaign_lineage_id=trial_row.get("campaign_lineage_id"),
        model_family=trial_row.get("model_family"),
        dataset_candidate_id=trial_row.get("dataset_candidate_id"),
        run_spec_id=trial_row.get("run_spec_id"),
    )
    return {
        "trial_id": trial_row["trial_id"],
        "campaign_id": trial_row["campaign_id"],
        "campaign_spec_id": trial_row["campaign_spec_id"],
        "campaign_kind": trial_row.get("campaign_kind"),
        "extension_type": trial_row.get("extension_type"),
        "campaign_lineage_id": trial_row.get("campaign_lineage_id"),
        "root_campaign_id": trial_row.get("root_campaign_id"),
        "parent_campaign_id": trial_row.get("parent_campaign_id"),
        "pooling_group_id": trial_row.get("pooling_group_id"),
        "is_primary_analysis_campaign": bool(str(trial_row.get("is_primary_analysis_campaign")).lower() == "true"),
        "eligible_for_pooled_seed_analysis": bool(
            str(trial_row.get("eligible_for_pooled_seed_analysis")).lower() == "true"
        ),
        "model_family": trial_row["model_family"],
        "dataset_candidate_id": trial_row["dataset_candidate_id"],
        "run_spec_id": trial_row["run_spec_id"],
        "comparison_group_id": trial_row["comparison_group_id"],
        "lineage_trial_group_id": lineage_trial_group_id,
        "x_transform": trial_row.get("x_transform"),
        "y_transform": trial_row.get("y_transform"),
        "synthetic_policy": trial_row.get("synthetic_policy"),
        "run_policy": trial_row.get("run_policy"),
        "flowpre_usage": bool(str(trial_row.get("flowpre_usage")).lower() == "true"),
        "flowgen_usage": bool(str(trial_row.get("flowgen_usage")).lower() == "true"),
        "seed_set_id": trial_row["seed_set_id"],
        "seed_panel_path": trial_row.get("seed_panel_path"),
        "seed": int(trial_row["seed"]),
        "replication_index": int(trial_row["replication_index"]),
        "expected_seed_count": int(trial_row.get("expected_seed_count") or 0) or None,
        "observed_seed_count": None,
        "dataset_manifest_path": trial_row["dataset_manifest_path"],
        "base_config_id": trial_row["base_config_id"],
        "objective_metric_id": trial_row["objective_metric_id"],
        "run_mode": trial_row["run_mode"],
        "allow_test_holdout": str(trial_row["allow_test_holdout"]).lower() == "true",
        "test_enabled": str(trial_row["test_enabled"]).lower() == "true",
        "native_interpretability_layer": trial_row["native_interpretability_layer"],
        "bridge_interpretability_layer": trial_row["bridge_interpretability_layer"],
        "execution_status": EXECUTION_STATUS_PENDING,
        "validity_status": VALIDITY_STATUS_UNKNOWN,
        "failure_reason_code": None,
        "failure_reason_detail": None,
        "exception_type": None,
        "exception_message": None,
        "traceback_path": None,
        "warning_count": 0,
        "warning_count_total": 0,
        "warning_count_silenced_known_noise": 0,
        "warning_count_surfaced": 0,
        "warning_log_path": None,
        "warning_category_counts": {},
        "warning_policy_counts": {},
        "warning_signature_counts": {},
        "attempt_count": 0,
        "current_attempt_id": None,
        "successful_attempt_id": None,
        "current_run_id": None,
        "current_run_dir": None,
        "successful_run_dir": None,
        "run_manifest_path": None,
        "results_path": None,
        "metrics_long_path": None,
        "prediction_sidecar_path": None,
        "interpretability_summary_path": None,
        "input_feature_influence_global_path": None,
        "input_feature_influence_per_class_path": None,
        "feature_influence_global_path": None,
        "feature_influence_per_class_path": None,
        "top_features_global_path": None,
        "top_features_per_class_path": None,
        "latent_feature_influence_global_path": None,
        "latent_feature_influence_per_class_path": None,
        "flowpre_projection_manifest_path": None,
        "flowpre_projection_cache_path": None,
        "xgb_shap_feature_influence_global_path": None,
        "xgb_shap_feature_influence_per_class_path": None,
        "xgb_shap_top_features_global_path": None,
        "xgb_shap_top_features_per_class_path": None,
        "xgb_perturbation_feature_influence_global_path": None,
        "xgb_perturbation_feature_influence_per_class_path": None,
        "xgb_perturbation_top_features_global_path": None,
        "xgb_perturbation_top_features_per_class_path": None,
        "contract_id": None,
        "raw_metric_contract_id": None,
        "raw_metric_contract_validation_status": None,
        "raw_real_available": None,
        "requires_raw_inversion": None,
        "raw_inversion_status": None,
        "value_space_default": None,
        "class_ontology_contract_id": trial_row.get("class_ontology_contract_id"),
        "class_ontology_contract_version": trial_row.get("class_ontology_contract_version"),
        "target_contract_id": trial_row.get("target_contract_id"),
        "target_contract_version": trial_row.get("target_contract_version"),
        "metric_grammar_contract_id": trial_row.get("metric_grammar_contract_id"),
        "metric_grammar_contract_version": trial_row.get("metric_grammar_contract_version"),
        "metric_availability_contract_id": trial_row.get("metric_availability_contract_id"),
        "metric_availability_contract_version": trial_row.get("metric_availability_contract_version"),
        "metric_aggregation_contract_id": trial_row.get("metric_aggregation_contract_id"),
        "metric_aggregation_contract_version": trial_row.get("metric_aggregation_contract_version"),
        "evaluation_population_contract_id": trial_row.get("evaluation_population_contract_id"),
        "evaluation_population_contract_version": trial_row.get("evaluation_population_contract_version"),
        "prediction_row_join_contract_id": trial_row.get("prediction_row_join_contract_id"),
        "prediction_row_join_contract_version": trial_row.get("prediction_row_join_contract_version"),
        "prediction_row_join_key_kind": trial_row.get("prediction_row_join_key_kind"),
        "feature_schema_contract_id": trial_row.get("feature_schema_contract_id"),
        "feature_schema_contract_version": trial_row.get("feature_schema_contract_version"),
        "factor_parser_contract_id": trial_row.get("factor_parser_contract_id"),
        "factor_parser_contract_version": trial_row.get("factor_parser_contract_version"),
        "factor_parser_version": trial_row.get("factor_parser_version"),
        "metric_grammar_version": trial_row.get("metric_grammar_version"),
        "lineage_aggregate_build_version": trial_row.get("lineage_aggregate_build_version"),
        "panel_build_version": trial_row.get("panel_build_version"),
        "panel_build_timestamp": trial_row.get("panel_build_timestamp"),
        "target_name": trial_row.get("target_name"),
        "target_space": trial_row.get("target_space"),
        "target_unit_public": trial_row.get("target_unit_public"),
        "analysis_ready_comparable": False,
        "analysis_ready_blockers": [],
        "split_class_support_source_path": None,
        "metric_availability_reference": None,
        "evaluation_population_reference": None,
        "feature_schema_surface_id": None,
        "feature_namespace": None,
        "primary_interpretability_surface_id": None,
        "git_commit": None,
        "variant_fingerprint": None,
        "config_path": None,
        "config_sha256": None,
        "campaign_valid": None,
        "campaign_valid_interpretability": None,
        "campaign_valid_f7": None,
        "training_runtime_s": None,
        "interpretability_runtime_s": None,
        "total_runtime_s": None,
        "started_at": None,
        "finished_at": None,
        "last_event_at": None,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }


def normalize_trial_state_schema(state: dict[str, Any]) -> dict[str, Any]:
    state.setdefault("warning_count", 0)
    state.setdefault("warning_count_total", int(state.get("warning_count") or 0))
    state.setdefault("warning_count_silenced_known_noise", 0)
    state.setdefault("warning_count_surfaced", 0)
    state.setdefault("warning_log_path", None)
    state.setdefault("warning_category_counts", {})
    state.setdefault("warning_policy_counts", {})
    state.setdefault("warning_signature_counts", {})
    state.setdefault("lineage_trial_group_id", derive_lineage_trial_group_id(
        campaign_lineage_id=state.get("campaign_lineage_id"),
        model_family=state.get("model_family"),
        dataset_candidate_id=state.get("dataset_candidate_id"),
        run_spec_id=state.get("run_spec_id"),
    ))
    for key, default in (
        ("x_transform", None),
        ("y_transform", None),
        ("synthetic_policy", None),
        ("run_policy", None),
        ("flowpre_usage", False),
        ("flowgen_usage", False),
        ("expected_seed_count", None),
        ("observed_seed_count", None),
    ):
        state.setdefault(key, default)
    state.setdefault("analysis_ready_blockers", [])
    for key in (
        "run_manifest_path",
        "results_path",
        "metrics_long_path",
        "prediction_sidecar_path",
        "interpretability_summary_path",
        "input_feature_influence_global_path",
        "input_feature_influence_per_class_path",
        "feature_influence_global_path",
        "feature_influence_per_class_path",
        "top_features_global_path",
        "top_features_per_class_path",
        "latent_feature_influence_global_path",
        "latent_feature_influence_per_class_path",
        "flowpre_projection_manifest_path",
        "flowpre_projection_cache_path",
        "xgb_shap_feature_influence_global_path",
        "xgb_shap_feature_influence_per_class_path",
        "xgb_shap_top_features_global_path",
        "xgb_shap_top_features_per_class_path",
        "xgb_perturbation_feature_influence_global_path",
        "xgb_perturbation_feature_influence_per_class_path",
        "xgb_perturbation_top_features_global_path",
        "xgb_perturbation_top_features_per_class_path",
        "contract_id",
        "raw_metric_contract_id",
        "raw_metric_contract_validation_status",
        "raw_real_available",
        "requires_raw_inversion",
        "raw_inversion_status",
        "value_space_default",
        "class_ontology_contract_id",
        "class_ontology_contract_version",
        "target_contract_id",
        "target_contract_version",
        "metric_grammar_contract_id",
        "metric_grammar_contract_version",
        "metric_availability_contract_id",
        "metric_availability_contract_version",
        "metric_aggregation_contract_id",
        "metric_aggregation_contract_version",
        "evaluation_population_contract_id",
        "evaluation_population_contract_version",
        "prediction_row_join_contract_id",
        "prediction_row_join_contract_version",
        "prediction_row_join_key_kind",
        "feature_schema_contract_id",
        "feature_schema_contract_version",
        "factor_parser_contract_id",
        "factor_parser_contract_version",
        "factor_parser_version",
        "metric_grammar_version",
        "lineage_aggregate_build_version",
        "panel_build_version",
        "panel_build_timestamp",
        "target_name",
        "target_space",
        "target_unit_public",
        "analysis_ready_comparable",
        "split_class_support_source_path",
        "metric_availability_reference",
        "evaluation_population_reference",
        "feature_schema_surface_id",
        "feature_namespace",
        "primary_interpretability_surface_id",
        "git_commit",
        "variant_fingerprint",
        "config_path",
        "config_sha256",
    ):
        state.setdefault(key, None)
    return state


def _trial_state_path(paths: CampaignPaths, trial_id: str) -> Path:
    return paths.trial_state_dir / f"{trial_id}.json"


def initialize_trial_state_files(paths: CampaignPaths, trial_rows: Iterable[dict[str, Any]]) -> None:
    ensure_campaign_dirs(paths)
    for trial_row in trial_rows:
        path = _trial_state_path(paths, str(trial_row["trial_id"]))
        if path.exists():
            continue
        _write_json(path, build_initial_trial_state(trial_row))


def load_trial_state(paths: CampaignPaths, trial_id: str) -> dict[str, Any]:
    return normalize_trial_state_schema(_read_json(_trial_state_path(paths, trial_id)))


def save_trial_state(paths: CampaignPaths, state: dict[str, Any]) -> None:
    normalize_trial_state_schema(state)
    state["updated_at"] = utc_now_iso()
    _write_json(_trial_state_path(paths, str(state["trial_id"])), state)


def iter_trial_states(paths: CampaignPaths) -> list[dict[str, Any]]:
    if not paths.trial_state_dir.exists():
        return []
    states: list[dict[str, Any]] = []
    for path in sorted(paths.trial_state_dir.glob("*.json")):
        states.append(normalize_trial_state_schema(_read_json(path)))
    return states


def append_attempt_event(paths: CampaignPaths, payload: dict[str, Any]) -> None:
    ensure_campaign_dirs(paths)
    event = {"event_at": utc_now_iso(), **payload}
    with paths.attempts_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def derive_attempt_id(trial_id: str, attempt_index: int) -> str:
    return f"attempt__{trial_id}__{int(attempt_index):04d}"


def derive_run_id(trial_id: str, attempt_index: int) -> str:
    return f"{trial_id}__attempt-{int(attempt_index):04d}"


def next_attempt_index(state: dict[str, Any]) -> int:
    return int(state.get("attempt_count") or 0) + 1


def lock_path(paths: CampaignPaths, trial_id: str) -> Path:
    return paths.locks_dir / f"{trial_id}.lock"


def acquire_trial_lock(paths: CampaignPaths, trial_id: str, runner_id: str) -> tuple[bool, Path]:
    ensure_campaign_dirs(paths)
    path = lock_path(paths, trial_id)
    if path.exists():
        return False, path
    payload = {
        "campaign_id": paths.campaign_id,
        "trial_id": trial_id,
        "runner_id": runner_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "acquired_at": utc_now_iso(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return True, path


def release_trial_lock(paths: CampaignPaths, trial_id: str) -> None:
    path = lock_path(paths, trial_id)
    if path.exists():
        path.unlink()


def _safe_mean(values: list[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 6)


def _count_by_field(states: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in states:
        value = state.get(field)
        if value is None or value == "":
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def assess_campaign_lineage_pool_readiness(
    *,
    states: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    expected_total = int((((manifest.get("expected_counts") or {}).get("trials") or {}).get("total") or 0))
    completed_valid_count = sum(
        1
        for state in states
        if state.get("execution_status") == EXECUTION_STATUS_COMPLETED and bool(state.get("campaign_valid_f7"))
    )
    if reduce_campaign_status(states) != CAMPAIGN_STATUS_CLOSED_SUCCESS:
        blockers.append("campaign_not_closed_success")
    if expected_total > 0 and completed_valid_count != expected_total:
        blockers.append("completed_valid_f7_count_mismatch_expected_total")
    valid_states = [
        state
        for state in states
        if state.get("execution_status") == EXECUTION_STATUS_COMPLETED and bool(state.get("campaign_valid_f7"))
    ]
    for field in (
        "run_manifest_path",
        "results_path",
        "metrics_long_path",
        "prediction_sidecar_path",
        "interpretability_summary_path",
        "raw_metric_contract_id",
        "raw_metric_contract_validation_status",
        "value_space_default",
        "lineage_trial_group_id",
        "class_ontology_contract_id",
        "target_contract_id",
        "metric_grammar_contract_id",
        "metric_availability_contract_id",
        "metric_aggregation_contract_id",
        "evaluation_population_contract_id",
        "prediction_row_join_contract_id",
        "feature_schema_contract_id",
        "factor_parser_contract_id",
        "target_name",
        "target_space",
    ):
        if any(not state.get(field) for state in valid_states):
            blockers.append(f"missing_{field}")
    if any(str(state.get("raw_metric_contract_validation_status")) != "ok" for state in valid_states):
        blockers.append("raw_metric_contract_validation_status_not_ok")
    if any(state.get("raw_real_available") is not True for state in valid_states):
        blockers.append("raw_real_not_available_for_all_valid_trials")
    if any(state.get("analysis_ready_comparable") is not True for state in valid_states):
        blockers.append("analysis_ready_comparable_not_true_for_all_valid_trials")
    return len(blockers) == 0, sorted(set(blockers))


def build_trial_ledger_rows(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ledger_rows: list[dict[str, Any]] = []
    for state in sorted(
        states,
        key=lambda row: (
            str(row.get("model_family")),
            str(row.get("dataset_candidate_id")),
            str(row.get("run_spec_id")),
            int(row.get("seed") or 0),
        ),
    ):
        ledger_rows.append(dict(state))
    return ledger_rows


def write_trial_ledger(paths: CampaignPaths, states: list[dict[str, Any]]) -> None:
    rows = build_trial_ledger_rows(states)
    if not rows:
        return
    paths.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def reduce_campaign_status(states: list[dict[str, Any]], *, aborted: bool = False) -> str:
    if aborted:
        return CAMPAIGN_STATUS_ABORTED
    if not states:
        return CAMPAIGN_STATUS_OPEN
    statuses = [str(state.get("execution_status") or EXECUTION_STATUS_PENDING) for state in states]
    if all(status == EXECUTION_STATUS_PENDING for status in statuses):
        return CAMPAIGN_STATUS_OPEN
    if any(status in {EXECUTION_STATUS_PENDING, EXECUTION_STATUS_RUNNING} for status in statuses):
        return CAMPAIGN_STATUS_IN_PROGRESS
    if any(status in {EXECUTION_STATUS_FAILED, EXECUTION_STATUS_BLOCKED} for status in statuses):
        return CAMPAIGN_STATUS_CLOSED_WITH_FAILURES
    return CAMPAIGN_STATUS_CLOSED_SUCCESS


def build_summary_payload(paths: CampaignPaths, states: list[dict[str, Any]]) -> dict[str, Any]:
    counts_by_status: dict[str, int] = {}
    counts_by_family: dict[str, dict[str, int]] = {}
    failure_counts: dict[str, int] = {}
    warning_category_counts: dict[str, int] = {}
    warning_policy_counts: dict[str, int] = {}
    warning_signature_counts: dict[str, int] = {}
    total_warning_count = 0
    total_warning_count_silenced = 0
    total_warning_count_surfaced = 0
    training_values: list[float] = []
    interpretability_values: list[float] = []
    total_values: list[float] = []

    for state in states:
        status = str(state.get("execution_status") or EXECUTION_STATUS_PENDING)
        family = str(state.get("model_family"))
        counts_by_status[status] = counts_by_status.get(status, 0) + 1
        counts_by_family.setdefault(family, {})
        counts_by_family[family][status] = counts_by_family[family].get(status, 0) + 1
        reason = state.get("failure_reason_code")
        if reason:
            failure_counts[str(reason)] = failure_counts.get(str(reason), 0) + 1
        total_warning_count += int(state.get("warning_count_total") or state.get("warning_count") or 0)
        total_warning_count_silenced += int(state.get("warning_count_silenced_known_noise") or 0)
        total_warning_count_surfaced += int(state.get("warning_count_surfaced") or 0)
        for category_name, count in dict(state.get("warning_category_counts") or {}).items():
            warning_category_counts[str(category_name)] = warning_category_counts.get(str(category_name), 0) + int(count)
        for policy_name, count in dict(state.get("warning_policy_counts") or {}).items():
            warning_policy_counts[str(policy_name)] = warning_policy_counts.get(str(policy_name), 0) + int(count)
        for signature, count in dict(state.get("warning_signature_counts") or {}).items():
            warning_signature_counts[str(signature)] = warning_signature_counts.get(str(signature), 0) + int(count)
        if state.get("training_runtime_s") is not None:
            training_values.append(float(state["training_runtime_s"]))
        if state.get("interpretability_runtime_s") is not None:
            interpretability_values.append(float(state["interpretability_runtime_s"]))
        if state.get("total_runtime_s") is not None:
            total_values.append(float(state["total_runtime_s"]))

    completed_valid = sum(1 for state in states if bool(state.get("campaign_valid_f7")) and state.get("execution_status") == EXECUTION_STATUS_COMPLETED)
    analysis_ready_completed_valid = sum(
        1
        for state in states
        if bool(state.get("campaign_valid_f7"))
        and state.get("execution_status") == EXECUTION_STATUS_COMPLETED
        and bool(state.get("analysis_ready_comparable"))
    )
    manifest = load_campaign_manifest(paths)
    lineage_pool_ready, lineage_pool_blockers = assess_campaign_lineage_pool_readiness(states=states, manifest=manifest)
    summary = {
        "campaign_id": paths.campaign_id,
        "trial_count_total": len(states),
        "counts_by_status": counts_by_status,
        "counts_by_family": counts_by_family,
        "failure_reason_counts": failure_counts,
        "warning_count_total": total_warning_count,
        "warning_count_silenced_known_noise": total_warning_count_silenced,
        "warning_count_surfaced": total_warning_count_surfaced,
        "warning_category_counts": warning_category_counts,
        "warning_policy_counts": warning_policy_counts,
        "warning_signature_counts": warning_signature_counts,
        "raw_metric_contract_validation_status_counts": _count_by_field(states, "raw_metric_contract_validation_status"),
        "value_space_default_counts": _count_by_field(states, "value_space_default"),
        "raw_inversion_status_counts": _count_by_field(states, "raw_inversion_status"),
        "completed_valid_f7_count": completed_valid,
        "analysis_ready_comparable_count": analysis_ready_completed_valid,
        "analysis_ready_comparable_counts": _count_by_field(states, "analysis_ready_comparable"),
        "lineage_pool_ready": lineage_pool_ready,
        "lineage_pool_blockers": lineage_pool_blockers,
        "contract_coverage": {
            "class_ontology_contract_id_counts": _count_by_field(states, "class_ontology_contract_id"),
            "target_contract_id_counts": _count_by_field(states, "target_contract_id"),
            "metric_grammar_contract_id_counts": _count_by_field(states, "metric_grammar_contract_id"),
            "metric_availability_contract_id_counts": _count_by_field(states, "metric_availability_contract_id"),
            "metric_aggregation_contract_id_counts": _count_by_field(states, "metric_aggregation_contract_id"),
            "evaluation_population_contract_id_counts": _count_by_field(states, "evaluation_population_contract_id"),
            "prediction_row_join_contract_id_counts": _count_by_field(states, "prediction_row_join_contract_id"),
            "feature_schema_contract_id_counts": _count_by_field(states, "feature_schema_contract_id"),
            "factor_parser_contract_id_counts": _count_by_field(states, "factor_parser_contract_id"),
        },
        "aggregate_runtime": {
            "training_runtime_s_sum": round(sum(training_values), 6) if training_values else None,
            "interpretability_runtime_s_sum": round(sum(interpretability_values), 6) if interpretability_values else None,
            "total_runtime_s_sum": round(sum(total_values), 6) if total_values else None,
            "training_runtime_s_mean": _safe_mean(training_values),
            "interpretability_runtime_s_mean": _safe_mean(interpretability_values),
            "total_runtime_s_mean": _safe_mean(total_values),
        },
        "campaign_status": reduce_campaign_status(states),
        "updated_at": utc_now_iso(),
    }
    return summary


def write_summary(paths: CampaignPaths, states: list[dict[str, Any]]) -> dict[str, Any]:
    summary = build_summary_payload(paths, states)
    _write_json(paths.summary_path, summary)
    return summary


def initialize_campaign_manifest(
    *,
    paths: CampaignPaths,
    spec: dict[str, Any],
    frozen_inputs: dict[str, str],
    expansion_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "campaign_id": spec["campaign_id"],
        "campaign_spec_id": spec["campaign_spec_id"],
        "campaign_kind": spec["campaign_kind"],
        "extension_type": spec.get("extension_type"),
        "campaign_lineage_id": spec["campaign_lineage_id"],
        "root_campaign_id": spec["root_campaign_id"],
        "parent_campaign_id": spec.get("parent_campaign_id"),
        "seed_set_id": spec["seed_set_id"],
        "seed_panel_path": spec.get("seed_panel_path"),
        "pooling_group_id": spec["pooling_group_id"],
        "is_primary_analysis_campaign": bool(spec["is_primary_analysis_campaign"]),
        "eligible_for_pooled_seed_analysis": bool(spec["eligible_for_pooled_seed_analysis"]),
        "campaign_scope": spec["campaign_scope"],
        "campaign_status": CAMPAIGN_STATUS_OPEN,
        "created_at": utc_now_iso(),
        "closed_at": None,
        "inputs": frozen_inputs,
        "analysis_contracts": dict(spec.get("analysis_contracts") or {}),
        "expected_counts": dict(expansion_manifest.get("counts") or {}),
        "expected_replication": dict(expansion_manifest.get("expected_replication") or {}),
        "current_counts": {
            "completed_valid_f7_count": 0,
            "failed_count": 0,
            "blocked_count": 0,
        },
    }
    _write_json(paths.campaign_manifest_path, manifest)
    return manifest


def load_campaign_manifest(paths: CampaignPaths) -> dict[str, Any]:
    return _read_json(paths.campaign_manifest_path)


def write_campaign_manifest(paths: CampaignPaths, manifest: dict[str, Any]) -> None:
    _write_json(paths.campaign_manifest_path, manifest)


def update_campaign_manifest_from_states(paths: CampaignPaths, states: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = load_campaign_manifest(paths)
    campaign_status = reduce_campaign_status(states)
    if campaign_status in {CAMPAIGN_STATUS_CLOSED_SUCCESS, CAMPAIGN_STATUS_CLOSED_WITH_FAILURES} and manifest.get("closed_at") is None:
        manifest["closed_at"] = utc_now_iso()
    manifest["campaign_status"] = campaign_status
    manifest["current_counts"] = {
        "completed_valid_f7_count": sum(
            1
            for state in states
            if state.get("execution_status") == EXECUTION_STATUS_COMPLETED and bool(state.get("campaign_valid_f7"))
        ),
        "failed_count": sum(1 for state in states if state.get("execution_status") == EXECUTION_STATUS_FAILED),
        "blocked_count": sum(1 for state in states if state.get("execution_status") == EXECUTION_STATUS_BLOCKED),
    }
    manifest["updated_at"] = utc_now_iso()
    write_campaign_manifest(paths, manifest)
    return manifest


def update_campaign_registry(paths: CampaignPaths, manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    paths.registry_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if paths.registry_path.exists():
        with paths.registry_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    filtered = [row for row in rows if row.get("campaign_id") != manifest["campaign_id"]]
    filtered.append(
        {
            "campaign_id": manifest["campaign_id"],
            "campaign_spec_id": manifest["campaign_spec_id"],
            "campaign_kind": manifest["campaign_kind"],
            "extension_type": manifest.get("extension_type"),
            "campaign_lineage_id": manifest["campaign_lineage_id"],
            "root_campaign_id": manifest["root_campaign_id"],
            "parent_campaign_id": manifest.get("parent_campaign_id"),
            "seed_set_id": manifest["seed_set_id"],
            "pooling_group_id": manifest["pooling_group_id"],
            "is_primary_analysis_campaign": manifest["is_primary_analysis_campaign"],
            "eligible_for_pooled_seed_analysis": manifest["eligible_for_pooled_seed_analysis"],
            "trial_count_expected": manifest.get("expected_counts", {}).get("trials", {}).get("total"),
            "trial_count_completed_valid": manifest.get("current_counts", {}).get("completed_valid_f7_count"),
            "failed_count": manifest.get("current_counts", {}).get("failed_count"),
            "blocked_count": manifest.get("current_counts", {}).get("blocked_count"),
            "created_at": manifest.get("created_at"),
            "closed_at": manifest.get("closed_at"),
            "campaign_status": manifest.get("campaign_status"),
            "summary_path": path_relative_to_root(paths.summary_path),
            "campaign_manifest_path": path_relative_to_root(paths.campaign_manifest_path),
        }
    )
    filtered.sort(key=lambda row: str(row.get("campaign_id")))
    with paths.registry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(filtered[0].keys()))
        writer.writeheader()
        writer.writerows(filtered)


def refresh_campaign_reporting(paths: CampaignPaths) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    states = iter_trial_states(paths)
    write_trial_ledger(paths, states)
    summary = write_summary(paths, states)
    manifest = update_campaign_manifest_from_states(paths, states)
    update_campaign_registry(paths, manifest, summary)
    return states, summary, manifest


def write_preflight_report(paths: CampaignPaths, payload: dict[str, Any]) -> None:
    _write_json(paths.preflight_report_path, payload)


def write_campaign_closeout(paths: CampaignPaths, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    states, summary, manifest = refresh_campaign_reporting(paths)
    completed_valid_f7_count_by_comparison_group: dict[str, int] = {}
    counts_by_family: dict[str, int] = {}
    for state in states:
        if state.get("execution_status") == EXECUTION_STATUS_COMPLETED and bool(state.get("campaign_valid_f7")):
            comparison_group_id = str(state.get("comparison_group_id"))
            completed_valid_f7_count_by_comparison_group[comparison_group_id] = (
                completed_valid_f7_count_by_comparison_group.get(comparison_group_id, 0) + 1
            )
            family = str(state.get("model_family"))
            counts_by_family[family] = counts_by_family.get(family, 0) + 1
    payload = {
        "campaign_id": manifest["campaign_id"],
        "campaign_spec_id": manifest["campaign_spec_id"],
        "campaign_status": manifest["campaign_status"],
        "campaign_kind": manifest["campaign_kind"],
        "extension_type": manifest.get("extension_type"),
        "campaign_lineage_id": manifest["campaign_lineage_id"],
        "root_campaign_id": manifest["root_campaign_id"],
        "parent_campaign_id": manifest.get("parent_campaign_id"),
        "closed_at": utc_now_iso(),
        "eligible_for_chaining": manifest["campaign_status"] == CAMPAIGN_STATUS_CLOSED_SUCCESS,
        "lineage_pool_ready": summary.get("lineage_pool_ready"),
        "lineage_pool_blockers": summary.get("lineage_pool_blockers"),
        "analysis_ready_comparable_count": summary.get("analysis_ready_comparable_count"),
        "contract_coverage": summary.get("contract_coverage"),
        "completed_valid_f7_count_by_comparison_group": completed_valid_f7_count_by_comparison_group,
        "completed_valid_f7_count_by_family": counts_by_family,
        "warning_policy_counts": summary.get("warning_policy_counts"),
        "summary": summary,
        "unresolved_trials": [
            {
                "trial_id": state["trial_id"],
                "execution_status": state["execution_status"],
                "failure_reason_code": state.get("failure_reason_code"),
                "failure_reason_detail": state.get("failure_reason_detail"),
                "warning_count": int(state.get("warning_count") or 0),
                "warning_log_path": state.get("warning_log_path"),
            }
            for state in states
            if state.get("execution_status") in {EXECUTION_STATUS_FAILED, EXECUTION_STATUS_BLOCKED, EXECUTION_STATUS_PENDING, EXECUTION_STATUS_RUNNING}
        ],
    }
    if extra:
        payload.update(extra)
    _write_json(paths.campaign_closeout_path, payload)
    return payload
