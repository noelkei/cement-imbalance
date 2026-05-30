from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import tempfile
import sys
import time
import traceback
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import joblib
import pandas as pd
import torch
import yaml

from data.utils import ROOT_PATH, path_relative_to_root
from evaluation.f7_campaign_spec import DEFAULT_SPEC_PATH, MaterializedCampaignSpec, materialize_f7_campaign_spec
from evaluation.f7_campaign_state import (
    CAMPAIGN_STATUS_CLOSED_SUCCESS,
    CAMPAIGN_STATUS_CLOSED_WITH_FAILURES,
    EXECUTION_STATUS_BLOCKED,
    EXECUTION_STATUS_COMPLETED,
    EXECUTION_STATUS_FAILED,
    EXECUTION_STATUS_PENDING,
    EXECUTION_STATUS_RUNNING,
    FAILURE_REASON_EXCEPTION_INTERPRETABILITY,
    FAILURE_REASON_EXCEPTION_RESULTS_PERSISTENCE,
    FAILURE_REASON_EXCEPTION_TRAINING,
    FAILURE_REASON_INVALID_F7,
    FAILURE_REASON_LOCK_CONFLICT,
    FAILURE_REASON_MANUAL_ABORT,
    FAILURE_REASON_MISSING_RUN_ARTIFACTS,
    FAILURE_REASON_OOM,
    FAILURE_REASON_PARENT_NOT_CLOSED,
    FAILURE_REASON_PRECHECK_CONFIG,
    FAILURE_REASON_PRECHECK_CONTRACT,
    FAILURE_REASON_PRECHECK_INVALID_TRIAL,
    FAILURE_REASON_PRECHECK_MANIFEST,
    VALIDITY_STATUS_INVALID_F7,
    VALIDITY_STATUS_UNKNOWN,
    VALIDITY_STATUS_VALID_F7,
    acquire_trial_lock,
    append_attempt_event,
    build_campaign_paths,
    derive_lineage_trial_group_id,
    derive_attempt_id,
    derive_run_id,
    freeze_campaign_inputs,
    initialize_campaign_manifest,
    initialize_trial_state_files,
    iter_trial_states,
    load_campaign_manifest,
    load_frozen_trial_inventory,
    load_trial_state,
    next_attempt_index,
    refresh_campaign_reporting,
    release_trial_lock,
    save_trial_state,
    update_campaign_manifest_from_states,
    write_campaign_closeout,
    write_preflight_report,
)
from evaluation.f7_campaign_trial_consumption import (
    build_trial_consumption_payload,
    validate_trial_consumption_row,
    validate_trial_inventory_rows,
)
from evaluation.flowpre_projection import (
    load_flowpre_decoder_runtime,
    resolve_flowpre_promotion_manifest_from_dataset_manifest,
    resolve_or_build_flowpre_projection_cache,
)
from evaluation.meta_context import (
    get_f7_analysis_contract_bundle_with_paths,
    parse_f7_factor_fields,
    resolve_feature_namespace,
)
from training.train_mlp import train_mlp_pipeline
from training.train_xgboost import train_xgboost_model


@dataclass(frozen=True)
class RunnerSelection:
    rows: list[dict[str, str]]
    reason: str


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(ROOT_PATH) / candidate


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with _repo_path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML mapping at {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _read_bundle_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(_repo_path(path))
    if "post_cleaning_index" not in df.columns:
        raise ValueError(f"Missing post_cleaning_index in {path}")
    df["post_cleaning_index"] = df["post_cleaning_index"].astype(int)
    return df.sort_values("post_cleaning_index").reset_index(drop=True)


def _load_joblib_artifact_with_upgrade(path: str | Path) -> Any:
    resolved_path = _repo_path(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        artifact = joblib.load(resolved_path)
    version_warnings = [item for item in caught if item.category.__name__ == "InconsistentVersionWarning"]
    if not version_warnings:
        return artifact
    with tempfile.NamedTemporaryFile(dir=resolved_path.parent, suffix=".tmp", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        joblib.dump(artifact, temp_path)
        os.replace(temp_path, resolved_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return artifact


def _load_bundle_frames_from_manifest(
    manifest_payload: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifacts = dict(manifest_payload.get("artifacts") or {})
    x_artifacts = dict(artifacts.get("X") or {})
    y_artifacts = dict(artifacts.get("y") or {})
    return (
        _read_bundle_csv(x_artifacts["train"]),
        _read_bundle_csv(x_artifacts["val"]),
        _read_bundle_csv(x_artifacts["test"]),
        _read_bundle_csv(y_artifacts["train"]),
        _read_bundle_csv(y_artifacts["val"]),
        _read_bundle_csv(y_artifacts["test"]),
    )


def _bool_from_row(value: Any) -> bool:
    return str(value).lower() == "true"


def _load_trial_rows_from_file(path: str | Path) -> list[dict[str, str]]:
    with _repo_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_frozen_spec_path(paths: Any) -> Path:
    preferred = sorted(paths.inputs_dir.glob("f7_campaign_spec*.yaml"))
    if len(preferred) == 1:
        return preferred[0]
    candidates = sorted(paths.inputs_dir.glob("*.yaml"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not resolve frozen campaign spec under {paths.inputs_dir}")


def _build_trial_row_index(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row["trial_id"]): dict(row) for row in rows}


def _warning_log_path(paths: Any, attempt_id: str) -> Path:
    return paths.warnings_dir / f"{attempt_id}.jsonl"


_WARNING_POLICY_SILENCED_KNOWN_NOISE = "silenced_known_noise"
_WARNING_POLICY_SURFACED = "surfaced"


def _infer_warning_source_module(filename: str | None) -> str | None:
    if not filename:
        return None
    path = Path(filename)
    parts = list(path.parts)
    if "site-packages" in parts:
        idx = parts.index("site-packages") + 1
        module_parts = list(parts[idx:])
    else:
        module_parts = list(path.parts[-3:])
    if not module_parts:
        return None
    if module_parts[-1].endswith(".py"):
        module_parts[-1] = module_parts[-1][:-3]
    return ".".join(part for part in module_parts if part and part != "__init__")


def _normalize_warning_message(message: str) -> str:
    return " ".join(str(message).split())


def _classify_warning(item: warnings.WarningMessage) -> dict[str, Any]:
    category_name = str(item.category.__name__)
    message = str(item.message)
    normalized_message = _normalize_warning_message(message)
    source_module = _infer_warning_source_module(item.filename)
    policy = _WARNING_POLICY_SURFACED
    is_known_noise = False
    if (
        category_name == "FutureWarning"
        and source_module == "torch.nn.utils.weight_norm"
        and normalized_message == "`torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`."
    ):
        policy = _WARNING_POLICY_SILENCED_KNOWN_NOISE
        is_known_noise = True
    elif (
        category_name == "UserWarning"
        and source_module == "nflows.transforms.coupling"
        and normalized_message == "Inputs to the softmax are not scaled down: initialization might be bad."
    ):
        policy = _WARNING_POLICY_SILENCED_KNOWN_NOISE
        is_known_noise = True
    signature = f"{category_name}::{source_module or 'unknown'}::{normalized_message}"
    message_hash = hashlib.sha256(normalized_message.encode("utf-8")).hexdigest()
    return {
        "category": category_name,
        "message": message,
        "normalized_message": normalized_message,
        "warning_policy": policy,
        "warning_source_module": source_module,
        "warning_signature": signature,
        "warning_message_hash": message_hash,
        "warning_is_known_noise": is_known_noise,
    }


def _apply_warning_capture_to_state(state: dict[str, Any], warning_payload: dict[str, Any]) -> None:
    state["warning_count"] = int(warning_payload["warning_count_total"])
    state["warning_count_total"] = int(warning_payload["warning_count_total"])
    state["warning_count_silenced_known_noise"] = int(warning_payload["warning_count_silenced_known_noise"])
    state["warning_count_surfaced"] = int(warning_payload["warning_count_surfaced"])
    state["warning_log_path"] = warning_payload["warning_log_path"]
    state["warning_category_counts"] = dict(warning_payload["warning_category_counts"])
    state["warning_policy_counts"] = dict(warning_payload["warning_policy_counts"])
    state["warning_signature_counts"] = dict(warning_payload["warning_signature_counts"])


def _load_warning_payload_from_log(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {
            "warning_count_total": 0,
            "warning_count_silenced_known_noise": 0,
            "warning_count_surfaced": 0,
            "warning_log_path": None,
            "warning_category_counts": {},
            "warning_policy_counts": {},
            "warning_signature_counts": {},
        }
    resolved = _repo_path(path)
    if not resolved.exists():
        return {
            "warning_count_total": 0,
            "warning_count_silenced_known_noise": 0,
            "warning_count_surfaced": 0,
            "warning_log_path": None,
            "warning_category_counts": {},
            "warning_policy_counts": {},
            "warning_signature_counts": {},
        }
    category_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    signature_counts: Counter[str] = Counter()
    total = 0
    for line in resolved.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        total += 1
        category_counts[str(payload.get("category") or "unknown")] += 1
        policy_counts[str(payload.get("warning_policy") or _WARNING_POLICY_SURFACED)] += 1
        signature_counts[str(payload.get("warning_signature") or "unknown")] += 1
    return {
        "warning_count_total": total,
        "warning_count_silenced_known_noise": int(policy_counts.get(_WARNING_POLICY_SILENCED_KNOWN_NOISE, 0)),
        "warning_count_surfaced": int(policy_counts.get(_WARNING_POLICY_SURFACED, 0)),
        "warning_log_path": path_relative_to_root(resolved),
        "warning_category_counts": dict(category_counts),
        "warning_policy_counts": dict(policy_counts),
        "warning_signature_counts": dict(signature_counts),
    }


def _try_git_commit() -> str | None:
    head_path = Path(ROOT_PATH) / ".git" / "HEAD"
    try:
        if not head_path.exists():
            return None
        head_value = head_path.read_text(encoding="utf-8").strip()
        if head_value.startswith("ref:"):
            ref_path = Path(ROOT_PATH) / ".git" / head_value.split(":", 1)[1].strip()
            return ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else None
        return head_value or None
    except Exception:
        return None


def _read_csv_header(path: str | Path) -> list[str]:
    df = pd.read_csv(_repo_path(path), nrows=0)
    return [str(column) for column in df.columns]


def _validate_prediction_sidecar_contract(path: str | Path | None) -> list[str]:
    if not path:
        return ["missing_prediction_sidecar_path"]
    try:
        columns = set(_read_csv_header(path))
    except Exception as exc:  # noqa: BLE001
        return [f"prediction_sidecar_unreadable:{type(exc).__name__}"]
    required = set((get_f7_analysis_contract_bundle_with_paths().get("prediction_row_join_key_kind") and ["post_cleaning_index", "split"]) or [])
    missing = sorted(required - columns)
    return [f"prediction_sidecar_missing_column:{column}" for column in missing]


def _validate_metrics_long_contract(path: str | Path | None, *, model_family: str) -> list[str]:
    if not path:
        return ["missing_metrics_long_path"]
    try:
        df = pd.read_csv(_repo_path(path))
    except Exception as exc:  # noqa: BLE001
        return [f"metrics_long_unreadable:{type(exc).__name__}"]
    required_columns = {"split", "metric_name", "metric_scope", "value_space", "metric_value"}
    missing_columns = sorted(required_columns - set(str(column) for column in df.columns))
    if missing_columns:
        return [f"metrics_long_missing_column:{column}" for column in missing_columns]
    contract = get_f7_analysis_contract_bundle_with_paths()
    availability = _load_yaml(contract["metric_availability_contract_path"])
    by_family = (
        ((availability.get("metric_availability_contract") or {}).get("by_model_family") or {}).get(str(model_family), {})
    )
    required_value_spaces = {
        str(item)
        for item in (((by_family.get("predictive_surfaces") or {}).get("required_value_spaces")) or [])
    }
    required_scopes = {str(item) for item in (by_family.get("required_metric_scopes") or [])}
    blockers: list[str] = []
    observed_value_spaces = {str(item) for item in df["value_space"].dropna().unique()}
    observed_scopes = {str(item) for item in df["metric_scope"].dropna().unique()}
    for value_space in sorted(required_value_spaces):
        if value_space not in observed_value_spaces:
            blockers.append(f"metrics_long_missing_value_space:{value_space}")
    for scope in sorted(required_scopes):
        if scope not in observed_scopes:
            blockers.append(f"metrics_long_missing_scope:{scope}")
    return blockers


def _validate_interpretability_surface_contract(
    *,
    state_like: dict[str, Any],
    model_family: str,
) -> list[str]:
    namespace = str(state_like.get("feature_namespace") or "")
    if not namespace:
        return ["missing_feature_namespace"]
    if str(model_family) == "xgboost":
        global_path = state_like.get("xgb_perturbation_feature_influence_global_path")
        per_class_path = state_like.get("xgb_perturbation_feature_influence_per_class_path")
    elif namespace == "flowpre_projected_semantic_input":
        global_path = state_like.get("input_feature_influence_global_path")
        per_class_path = state_like.get("input_feature_influence_per_class_path")
    else:
        global_path = state_like.get("feature_influence_global_path")
        per_class_path = state_like.get("feature_influence_per_class_path")
    blockers: list[str] = []
    if not global_path:
        blockers.append("missing_primary_interpretability_global_path")
    if not per_class_path:
        blockers.append("missing_primary_interpretability_per_class_path")
    if blockers:
        return blockers
    try:
        global_columns = set(_read_csv_header(global_path))
        per_class_columns = set(_read_csv_header(per_class_path))
    except Exception as exc:  # noqa: BLE001
        return [f"interpretability_surface_unreadable:{type(exc).__name__}"]
    for column in ("split", "feature_name"):
        if column not in global_columns:
            blockers.append(f"interpretability_global_missing_column:{column}")
    if "mean_abs_delta_pred_raw" not in global_columns and "mean_abs_shap" not in global_columns:
        blockers.append("interpretability_global_missing_importance_column")
    if "split" not in per_class_columns:
        blockers.append("interpretability_per_class_missing_column:split")
    if "feature_name" not in per_class_columns and "latent_name" not in per_class_columns:
        blockers.append("interpretability_per_class_missing_feature_identifier")
    if "class_id" not in per_class_columns and "type" not in per_class_columns:
        blockers.append("interpretability_per_class_missing_class_identifier")
    return blockers


def _collect_analysis_ready_blockers(state: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    required_values = (
        state.get("campaign_valid_f7") is True,
        state.get("raw_real_available") is True,
        str(state.get("raw_metric_contract_validation_status")) == "ok",
        bool(state.get("run_manifest_path")),
        bool(state.get("results_path")),
        bool(state.get("metrics_long_path")),
        bool(state.get("prediction_sidecar_path")),
        bool(state.get("interpretability_summary_path")),
        bool(state.get("class_ontology_contract_id")),
        bool(state.get("target_contract_id")),
        bool(state.get("metric_grammar_contract_id")),
        bool(state.get("metric_availability_contract_id")),
        bool(state.get("metric_aggregation_contract_id")),
        bool(state.get("evaluation_population_contract_id")),
        bool(state.get("prediction_row_join_contract_id")),
        bool(state.get("prediction_row_join_key_kind")),
        bool(state.get("feature_schema_contract_id")),
        bool(state.get("factor_parser_contract_id")),
        bool(state.get("target_name")),
        bool(state.get("target_space")),
        bool(state.get("target_unit_public")),
        bool(state.get("metric_availability_reference")),
        bool(state.get("evaluation_population_reference")),
        bool(state.get("feature_namespace")),
        bool(state.get("primary_interpretability_surface_id")),
        bool(state.get("split_class_support_source_path") or state.get("metrics_long_path")),
    )
    if not all(required_values):
        blockers.append("missing_required_analysis_surface_fields")
    blockers.extend(_validate_prediction_sidecar_contract(state.get("prediction_sidecar_path")))
    blockers.extend(
        _validate_metrics_long_contract(
            state.get("metrics_long_path"),
            model_family=str(state.get("model_family") or ""),
        )
    )
    blockers.extend(
        _validate_interpretability_surface_contract(
            state_like=state,
            model_family=str(state.get("model_family") or ""),
        )
    )
    return sorted(set(blockers))


def _derive_analysis_ready_comparable(state: dict[str, Any]) -> bool:
    return len(_collect_analysis_ready_blockers(state)) == 0


def _emit_surfaced_warning_messages(
    *,
    trial_id: str,
    selection_index: int | None,
    selection_total: int | None,
    surfaced_records: list[dict[str, Any]],
) -> None:
    if not surfaced_records:
        return
    progress_prefix = (
        f"[{selection_index}/{selection_total}] "
        if selection_index is not None and selection_total is not None
        else ""
    )
    for record in surfaced_records:
        print(
            (
                f"{progress_prefix}WARNING "
                f"trial_id={trial_id} "
                f"category={record['category']} "
                f"source={record.get('warning_source_module') or 'unknown'} "
                f"message={record['normalized_message']}"
            ),
            flush=True,
        )


def _persist_captured_warnings(
    *,
    paths: Any,
    attempt_id: str,
    trial_id: str,
    run_id: str,
    selection_index: int | None,
    selection_total: int | None,
    caught: list[warnings.WarningMessage],
) -> dict[str, Any]:
    if not caught:
        return {
            "warning_count_total": 0,
            "warning_count_silenced_known_noise": 0,
            "warning_count_surfaced": 0,
            "warning_log_path": None,
            "warning_category_counts": {},
            "warning_policy_counts": {},
            "warning_signature_counts": {},
        }
    warning_path = _warning_log_path(paths, attempt_id)
    warning_path.parent.mkdir(parents=True, exist_ok=True)
    category_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    signature_counts: Counter[str] = Counter()
    surfaced_records: list[dict[str, Any]] = []
    with warning_path.open("w", encoding="utf-8") as handle:
        for warning_idx, item in enumerate(caught, start=1):
            classification = _classify_warning(item)
            category_name = str(classification["category"])
            category_counts[category_name] += 1
            policy_counts[str(classification["warning_policy"])] += 1
            signature_counts[str(classification["warning_signature"])] += 1
            payload = dict(classification)
            payload.update(
                {
                    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "trial_id": trial_id,
                    "attempt_id": attempt_id,
                    "run_id": run_id,
                    "selection_index": selection_index,
                    "selection_total": selection_total,
                    "warning_index": warning_idx,
                    "filename": item.filename,
                    "lineno": int(item.lineno),
                    "line": item.line,
                }
            )
            if classification["warning_policy"] == _WARNING_POLICY_SURFACED:
                surfaced_records.append(payload)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    _emit_surfaced_warning_messages(
        trial_id=trial_id,
        selection_index=selection_index,
        selection_total=selection_total,
        surfaced_records=surfaced_records,
    )
    return {
        "warning_count_total": len(caught),
        "warning_count_silenced_known_noise": int(policy_counts.get(_WARNING_POLICY_SILENCED_KNOWN_NOISE, 0)),
        "warning_count_surfaced": int(policy_counts.get(_WARNING_POLICY_SURFACED, 0)),
        "warning_log_path": path_relative_to_root(warning_path),
        "warning_category_counts": dict(category_counts),
        "warning_policy_counts": dict(policy_counts),
        "warning_signature_counts": dict(signature_counts),
    }


def _prewarm_flowpre_assets_for_rows(rows: Iterable[dict[str, str]]) -> None:
    warmed: set[str] = set()
    for row in rows:
        if str(row.get("model_family")) != "mlp":
            continue
        manifest_path = row.get("dataset_manifest_path")
        if not manifest_path:
            continue
        dataset_manifest = _load_json(manifest_path)
        promotion_manifest_path = resolve_flowpre_promotion_manifest_from_dataset_manifest(dataset_manifest)
        if promotion_manifest_path is None:
            continue
        resolved = str(_repo_path(promotion_manifest_path).resolve())
        if resolved in warmed:
            continue
        resolve_or_build_flowpre_projection_cache(
            promotion_manifest_path=resolved,
            device="cpu",
            force_rebuild=False,
        )
        load_flowpre_decoder_runtime(
            promotion_manifest_path=resolved,
            device="cpu",
            condition_col="type",
        )
        warmed.add(resolved)


def _status_filter_tokens(status_filter: str | None) -> set[str] | None:
    if not status_filter:
        return None
    return {token.strip() for token in str(status_filter).split(",") if token.strip()}


def _stable_trial_sort_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["model_family"]),
        str(row["dataset_candidate_id"]),
        str(row["run_spec_id"]),
        int(row["seed"]),
    )


def filter_trial_rows(
    rows: list[dict[str, str]],
    *,
    states_by_id: dict[str, dict[str, Any]] | None = None,
    model_family: str | None = None,
    dataset_candidate_id: str | None = None,
    run_spec_id: str | None = None,
    trial_id: str | None = None,
    trial_id_file: str | Path | None = None,
    offset: int | None = None,
    limit: int | None = None,
    status_filter: str | None = None,
) -> RunnerSelection:
    selected = list(rows)
    if model_family:
        selected = [row for row in selected if str(row["model_family"]) == str(model_family)]
    if dataset_candidate_id:
        selected = [row for row in selected if str(row["dataset_candidate_id"]) == str(dataset_candidate_id)]
    if run_spec_id:
        selected = [row for row in selected if str(row["run_spec_id"]) == str(run_spec_id)]
    if trial_id:
        selected = [row for row in selected if str(row["trial_id"]) == str(trial_id)]
    if trial_id_file:
        wanted = {
            line.strip()
            for line in _repo_path(trial_id_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        selected = [row for row in selected if str(row["trial_id"]) in wanted]
    selected.sort(key=_stable_trial_sort_key)
    status_tokens = _status_filter_tokens(status_filter)
    if status_tokens is not None and states_by_id is not None:
        selected = [
            row
            for row in selected
            if str(states_by_id.get(str(row["trial_id"]), {}).get("execution_status", EXECUTION_STATUS_PENDING)) in status_tokens
        ]
    if offset:
        selected = selected[int(offset):]
    if limit is not None:
        selected = selected[: int(limit)]
    return RunnerSelection(rows=selected, reason="filtered_trial_inventory")


def _materialize_campaign_inputs_for_root(spec_path: str | Path, campaign_id: str) -> MaterializedCampaignSpec:
    paths = build_campaign_paths(campaign_id)
    return materialize_f7_campaign_spec(spec_path=spec_path, output_root=paths.inputs_dir, write_outputs=True)


def initialize_campaign_from_spec(spec_path: str | Path) -> tuple[dict[str, Any], MaterializedCampaignSpec]:
    materialized = materialize_f7_campaign_spec(spec_path=spec_path, write_outputs=False)
    spec = materialized.spec
    paths = build_campaign_paths(str(spec["campaign_id"]))
    if paths.campaign_manifest_path.exists():
        return spec, materialized

    materialized = _materialize_campaign_inputs_for_root(spec_path, str(spec["campaign_id"]))
    frozen_inputs = freeze_campaign_inputs(
        paths=paths,
        spec_source_path=spec_path,
        materialized_output_paths=materialized.output_paths,
    )
    initialize_trial_state_files(paths, materialized.trials)
    initialize_campaign_manifest(
        paths=paths,
        spec={**materialized.spec, "analysis_contracts": materialized.expansion_manifest.get("analysis_contracts") or {}},
        frozen_inputs=frozen_inputs,
        expansion_manifest=materialized.expansion_manifest,
    )
    refresh_campaign_reporting(paths)
    return materialized.spec, materialized


def load_existing_campaign(campaign_id: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    paths = build_campaign_paths(campaign_id)
    manifest = load_campaign_manifest(paths)
    spec_path = _resolve_frozen_spec_path(paths)
    spec = materialize_f7_campaign_spec(spec_path=spec_path, write_outputs=False).spec
    return spec, load_frozen_trial_inventory(paths)


def _read_parent_lineage_rows(parent_campaign_id: str | None) -> list[dict[str, str]]:
    if not parent_campaign_id:
        return []
    lineage_rows: list[dict[str, str]] = []
    current_id = str(parent_campaign_id)
    while current_id:
        parent_paths = build_campaign_paths(current_id)
        parent_manifest = load_campaign_manifest(parent_paths)
        lineage_rows.extend(load_frozen_trial_inventory(parent_paths))
        current_id = str(parent_manifest.get("parent_campaign_id") or "")
    return lineage_rows


def validate_extension_lineage(
    *,
    spec: dict[str, Any],
    materialized: MaterializedCampaignSpec,
) -> dict[str, Any]:
    if str(spec.get("campaign_kind")) != "extension":
        return {"ok": True, "issues": []}

    issues: list[str] = []
    parent_campaign_id = str(spec.get("parent_campaign_id") or "")
    parent_paths = build_campaign_paths(parent_campaign_id)
    if not parent_paths.campaign_manifest_path.exists():
        issues.append("missing_parent_campaign_manifest")
        return {"ok": False, "issues": issues}

    parent_manifest = load_campaign_manifest(parent_paths)
    if parent_manifest.get("campaign_status") not in {CAMPAIGN_STATUS_CLOSED_SUCCESS, CAMPAIGN_STATUS_CLOSED_WITH_FAILURES}:
        issues.append(FAILURE_REASON_PARENT_NOT_CLOSED)

    parent_inputs = dict(parent_manifest.get("inputs") or {})
    parent_dataset_input = parent_inputs.get("dataset_candidate_inventory_path")
    parent_run_spec_input = parent_inputs.get("run_spec_inventory_path")
    if not parent_dataset_input or not parent_run_spec_input:
        issues.append("missing_parent_frozen_inputs")
        return {"ok": False, "issues": issues}

    parent_dataset_path = _repo_path(parent_dataset_input)
    parent_run_spec_path = _repo_path(parent_run_spec_input)
    if not parent_dataset_path.exists() or not parent_run_spec_path.exists():
        issues.append("missing_parent_frozen_inputs")
        return {"ok": False, "issues": issues}

    parent_dataset_rows = _load_trial_rows_from_file(parent_dataset_path)
    parent_run_spec_rows = _load_trial_rows_from_file(parent_run_spec_path)
    child_dataset_rows = materialized.dataset_candidates
    child_run_spec_rows = materialized.run_specs

    parent_dataset_ids = {row["dataset_candidate_id"] for row in parent_dataset_rows}
    child_dataset_ids = {row["dataset_candidate_id"] for row in child_dataset_rows}
    if parent_dataset_ids != child_dataset_ids:
        issues.append("dataset_candidate_set_mismatch")

    parent_run_spec_ids = {row["run_spec_id"] for row in parent_run_spec_rows}
    child_run_spec_ids = {row["run_spec_id"] for row in child_run_spec_rows}
    if parent_run_spec_ids != child_run_spec_ids:
        issues.append("run_spec_set_mismatch")

    signature_fields = (
        "raw_metric_contract_id",
        "artifact_policy_id",
        "class_ontology_contract_id",
        "target_contract_id",
        "metric_grammar_contract_id",
        "metric_availability_contract_id",
        "metric_aggregation_contract_id",
        "evaluation_population_contract_id",
        "prediction_row_join_contract_id",
        "feature_schema_contract_id",
        "factor_parser_contract_id",
    )

    def _signature_set(rows: list[dict[str, str]]) -> set[tuple[tuple[str, str], ...]]:
        signatures: set[tuple[tuple[str, str], ...]] = set()
        for row in rows:
            signatures.add(
                tuple((field, str(row.get(field))) for field in signature_fields)
            )
        return signatures

    parent_contract_signatures = _signature_set(parent_run_spec_rows)
    child_contract_signatures = _signature_set(child_run_spec_rows)
    if len(parent_contract_signatures) != 1:
        issues.append("parent_run_spec_contract_signature_inconsistent")
    if len(child_contract_signatures) != 1:
        issues.append("child_run_spec_contract_signature_inconsistent")
    if parent_contract_signatures != child_contract_signatures:
        issues.append("global_contract_signature_mismatch")

    parent_spec_path = _resolve_frozen_spec_path(parent_paths)
    parent_spec = materialize_f7_campaign_spec(spec_path=parent_spec_path, write_outputs=False).spec
    structural_keys = ("campaign_scope", "run_mode", "allow_test_holdout", "test_enabled", "meta_grammar_id")
    for key in structural_keys:
        if str(parent_spec.get(key)) != str(spec.get(key)):
            issues.append(f"structural_spec_mismatch:{key}")

    parent_trial_rows = load_frozen_trial_inventory(parent_paths)
    parent_seeds = {int(row["seed"]) for row in parent_trial_rows}
    child_seeds = {int(row["seed"]) for row in materialized.trials}
    parent_expected_replication = dict(parent_manifest.get("expected_replication") or {})
    parent_expected_seeds = {int(value) for value in list(parent_expected_replication.get("expected_seed_values") or [])}
    if parent_expected_seeds and parent_seeds != parent_expected_seeds:
        issues.append("parent_expected_seed_values_mismatch")
    child_expected_replication = dict(materialized.expansion_manifest.get("expected_replication") or {})
    child_expected_seeds = {int(value) for value in list(child_expected_replication.get("expected_seed_values") or [])}
    if child_expected_seeds and child_seeds != child_expected_seeds:
        issues.append("child_expected_seed_values_mismatch")
    lineage_rows = _read_parent_lineage_rows(parent_campaign_id)
    parent_lineage_seeds = {int(row["seed"]) for row in lineage_rows}
    overlap = sorted(parent_lineage_seeds & child_seeds)
    if overlap:
        issues.append("seed_overlap_with_parent_lineage")

    return {"ok": not issues, "issues": issues}


def _preflight_reason_code(issues: list[str]) -> str:
    if "missing_dataset_manifest" in issues:
        return FAILURE_REASON_PRECHECK_MANIFEST
    if "missing_base_config" in issues:
        return FAILURE_REASON_PRECHECK_CONFIG
    if any(issue.startswith("missing_contract") for issue in issues):
        return FAILURE_REASON_PRECHECK_CONTRACT
    return FAILURE_REASON_PRECHECK_INVALID_TRIAL


def _validate_runtime_trial_requirements(row: dict[str, str]) -> list[str]:
    issues: list[str] = []
    if not row.get("x_transform") or not row.get("y_transform") or not row.get("synthetic_policy"):
        issues.append("missing_parsed_dataset_factor_fields")
    if not row.get("run_policy"):
        issues.append("missing_parsed_run_factor_field")
    if not row.get("expected_seed_count"):
        issues.append("missing_expected_seed_count")
    for contract_field in (
        "class_ontology_contract_id",
        "target_contract_id",
        "metric_grammar_contract_id",
        "metric_availability_contract_id",
        "metric_aggregation_contract_id",
        "evaluation_population_contract_id",
        "prediction_row_join_contract_id",
        "feature_schema_contract_id",
        "factor_parser_contract_id",
    ):
        if not row.get(contract_field):
            issues.append(f"missing_contract_field:{contract_field}")
    if str(row["model_family"]) != "mlp":
        return issues
    manifest = _load_json(row["dataset_manifest_path"])
    dataset_axes = dict(manifest.get("dataset_level_axes") or {})
    y_transform = str(dataset_axes.get("y_transform") or row.get("y_transform") or "raw").lower()
    if y_transform == "raw":
        return issues
    scaler_artifacts = dict(manifest.get("scaler_artifacts") or {})
    target_scaler_artifact = (
        manifest.get("target_scaler_artifact")
        or manifest.get("y_scaler_artifact")
        or scaler_artifacts.get("y")
    )
    if not target_scaler_artifact:
        issues.append("missing_target_scaler_artifact")
    else:
        scaler_path = _repo_path(target_scaler_artifact)
        if not scaler_path.exists():
            issues.append("missing_target_scaler_file")
    return issues


def preflight_trials(
    *,
    spec: dict[str, Any],
    trial_rows: list[dict[str, str]],
    lineage_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation_rows = validate_trial_inventory_rows(trial_rows)
    for validation_row, trial_row in zip(validation_rows, trial_rows, strict=False):
        runtime_issues = _validate_runtime_trial_requirements(trial_row)
        if runtime_issues:
            existing = [item for item in str(validation_row["issues"]).split(";") if item]
            combined = existing + runtime_issues
            validation_row["issues"] = ";".join(combined)
            validation_row["status"] = "failed"
    ok_count = sum(1 for row in validation_rows if row["status"] == "ok")
    failed_rows = [row for row in validation_rows if row["status"] != "ok"]
    payload = {
        "campaign_id": spec["campaign_id"],
        "campaign_spec_id": spec["campaign_spec_id"],
        "checked_trial_count": len(trial_rows),
        "ok_count": ok_count,
        "failed_count": len(failed_rows),
        "lineage_validation": lineage_validation or {"ok": True, "issues": []},
        "sample_failed_rows": failed_rows[:20],
        "blocking_contract_failures": sorted(
            {
                issue
                for row in failed_rows
                for issue in str(row.get("issues") or "").split(";")
                if issue.startswith("missing_contract")
            }
        ),
        "blocking_structural_failures": sorted(
            {
                issue
                for row in failed_rows
                for issue in str(row.get("issues") or "").split(";")
                if issue and not issue.startswith("missing_contract")
            }
        ),
        "missing_optional_surfaces": [],
        "informational_warnings": [],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload["ok"] = len(trial_rows) > 0 and ok_count == len(trial_rows) and bool((lineage_validation or {"ok": True})["ok"])
    return payload


def _build_effective_config(
    *,
    paths: Any,
    trial_row: dict[str, str],
) -> Path:
    base_config_id = str(trial_row["base_config_id"])
    base_cfg = _load_yaml(Path(ROOT_PATH) / "config" / f"{base_config_id}.yaml")
    cfg = deepcopy(base_cfg)
    family = str(trial_row["model_family"])
    seed = int(trial_row["seed"])
    allow_test_holdout = _bool_from_row(trial_row["allow_test_holdout"])
    cfg.setdefault("contract", {})
    cfg["contract"]["seed_set_id"] = trial_row["seed_set_id"]
    cfg["contract"]["allow_test_holdout_default"] = allow_test_holdout
    cfg["contract"]["objective_metric_id"] = trial_row["objective_metric_id"]
    if family == "mlp":
        cfg["contract"]["mlp_base_config_id"] = base_config_id
        cfg.setdefault("training", {})
        cfg["training"]["seed"] = seed
        cfg["training"]["dataloader_mode"] = trial_row["dataloader_mode"]
        cfg["training"]["cycle_reals"] = _bool_from_row(trial_row["cycle_reals"])
        cfg["training"]["allow_synth"] = _bool_from_row(trial_row["allow_synth"])
        cfg["training"]["loss_reduction"] = trial_row["loss_reduction"]
        cfg["training"]["regression_group_metric"] = trial_row["regression_group_metric"]
        cfg["seed"] = seed
    elif family == "xgboost":
        cfg["contract"]["xgb_base_config_id"] = base_config_id
        cfg.setdefault("training", {})
        cfg["training"]["random_state"] = seed
        cfg["seed"] = seed
    else:
        raise ValueError(f"Unsupported model family: {family}")
    attempt_id = str(trial_row["_attempt_id"])
    out_path = paths.effective_configs_dir / f"{attempt_id}.yaml"
    _write_yaml(out_path, cfg)
    return out_path


def _build_evaluation_context(
    *,
    trial_row: dict[str, str],
    dataset_manifest: dict[str, Any],
) -> dict[str, Any]:
    dataset_axes = dict(dataset_manifest.get("dataset_level_axes") or {})
    analysis_contracts = {
        **get_f7_analysis_contract_bundle_with_paths(),
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
    }
    parsed_factors = parse_f7_factor_fields(
        model_family=str(trial_row["model_family"]),
        dataset_level_axes=dataset_axes,
        run_level_axes={
            "batch_policy_id": trial_row.get("batch_policy_id"),
            "cycling_policy_id": trial_row.get("cycling_policy_id"),
            "loss_policy_id": trial_row.get("loss_policy_id"),
            "allow_synth": _bool_from_row(trial_row.get("allow_synth")),
        },
        fallback_dataset_candidate_id=str(trial_row["dataset_candidate_id"]),
        fallback_run_spec_id=str(trial_row["run_spec_id"]),
    )
    y_scaler_rel = ((dataset_manifest.get("scaler_artifacts") or {}).get("y"))
    y_scaler = None if y_scaler_rel is None else _load_joblib_artifact_with_upgrade(y_scaler_rel)
    return {
        "contract_id": "f7_contract_v1",
        "campaign_id": trial_row["campaign_id"],
        "dataset_candidate_id": trial_row["dataset_candidate_id"],
        "run_spec_id": trial_row["run_spec_id"],
        "trial_id": trial_row["trial_id"],
        "comparison_group_id": trial_row["comparison_group_id"],
        "seed_set_id": trial_row["seed_set_id"],
        "seed_panel_path": trial_row.get("seed_panel_path"),
        "base_config_id": trial_row["base_config_id"],
        "objective_metric_id": trial_row["objective_metric_id"],
        "dataset_name": dataset_manifest.get("dataset_name"),
        "dataset_manifest_path": trial_row["dataset_manifest_path"],
        "dataset_level_axes": dataset_axes,
        "y_transform": dataset_axes.get("y_transform"),
        "y_scaler": y_scaler,
        "target_scaler_artifact": y_scaler_rel,
        "split_id": dataset_manifest.get("split_id"),
        "run_mode": trial_row["run_mode"],
        "test_enabled": _bool_from_row(trial_row["test_enabled"]),
        "upstream_variant_fingerprint": f"{trial_row['campaign_id']}::{trial_row['trial_id']}",
        "analysis_contracts": analysis_contracts,
        "parsed_factor_fields": parsed_factors,
    }


def _family_run_dir(campaign_id: str, model_family: str, run_id: str) -> Path:
    return Path(ROOT_PATH) / "outputs" / "models" / model_family / "campaigns" / campaign_id / run_id


def _attempt_index_from_run_id(run_id: str | None) -> int | None:
    if not run_id or "__attempt-" not in str(run_id):
        return None
    try:
        return int(str(run_id).rsplit("__attempt-", 1)[1])
    except Exception:
        return None


def _light_memory_cleanup() -> None:
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _safe_mean_runtime(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 3)


def _format_runtime(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def _load_interpretability_summary_payload(run_manifest: dict[str, Any], candidate_paths: Iterable[str | Path | None]) -> dict[str, Any] | None:
    for candidate in candidate_paths:
        if not candidate:
            continue
        try:
            payload = _load_json(candidate)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            return payload
    return None


def _extract_interpretability_runtime(run_manifest: dict[str, Any], family: str) -> float | None:
    artifact_paths = dict(run_manifest.get("artifact_paths") or {})
    interpretability_artifacts = dict((run_manifest.get("interpretability_status") or {}).get("interpretability_artifacts") or {})
    if family == "mlp":
        summary = run_manifest.get("mlp_interpretability")
        if isinstance(summary, dict):
            value = summary.get("interpretability_runtime_s")
            return None if value is None else float(value)
        payload = _load_interpretability_summary_payload(
            run_manifest,
            (
                run_manifest.get("interpretability_summary_path"),
                artifact_paths.get("interpretability_summary_json"),
                interpretability_artifacts.get("interpretability_summary_json"),
            ),
        )
        if isinstance(payload, dict):
            value = payload.get("interpretability_runtime_s")
            return None if value is None else float(value)
        return None
    if family == "xgboost":
        summary = run_manifest.get("xgb_interpretability")
        if isinstance(summary, dict):
            value = summary.get("interpretability_runtime_s_total")
            return None if value is None else float(value)
        payload = _load_interpretability_summary_payload(
            run_manifest,
            (
                run_manifest.get("xgb_interpretability_summary_path"),
                run_manifest.get("interpretability_summary_path"),
                artifact_paths.get("xgb_interpretability_summary_json"),
                artifact_paths.get("interpretability_summary_json"),
                interpretability_artifacts.get("xgb_interpretability_summary_json"),
                interpretability_artifacts.get("interpretability_summary_json"),
            ),
        )
        if isinstance(payload, dict):
            value = payload.get("interpretability_runtime_s_total")
            return None if value is None else float(value)
        return None
    return None


def _resolve_interpretability_summary_path(run_manifest: dict[str, Any], family: str) -> str | None:
    artifact_paths = dict(run_manifest.get("artifact_paths") or {})
    if family == "mlp":
        return (
            run_manifest.get("interpretability_summary_path")
            or artifact_paths.get("interpretability_summary_json")
        )
    if family == "xgboost":
        return (
            run_manifest.get("xgb_interpretability_summary_path")
            or run_manifest.get("interpretability_summary_path")
            or artifact_paths.get("xgb_interpretability_summary_json")
            or artifact_paths.get("interpretability_summary_json")
        )
    return None


def _relative_artifact_path(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return path_relative_to_root(value)


def _extract_interpretability_artifact_paths(run_manifest: dict[str, Any], family: str) -> dict[str, Any]:
    artifact_paths = dict(run_manifest.get("artifact_paths") or {})
    interpretability_artifacts = dict((run_manifest.get("interpretability_status") or {}).get("interpretability_artifacts") or {})

    def resolve(*keys: str) -> str | None:
        for key in keys:
            value = (
                run_manifest.get(key)
                or interpretability_artifacts.get(key)
                or artifact_paths.get(key)
            )
            if value not in (None, ""):
                return _relative_artifact_path(value)
        return None

    payload = {
        "input_feature_influence_global_path": resolve(
            "input_feature_influence_global_path",
            "input_feature_influence_global_csv",
        ),
        "input_feature_influence_per_class_path": resolve(
            "input_feature_influence_per_class_path",
            "input_feature_influence_per_class_csv",
        ),
        "feature_influence_global_path": resolve(
            "feature_influence_global_path",
            "feature_influence_global_csv",
        ),
        "feature_influence_per_class_path": resolve(
            "feature_influence_per_class_path",
            "feature_influence_per_class_csv",
        ),
        "top_features_global_path": resolve(
            "top_features_global_path",
            "top_features_global_csv",
        ),
        "top_features_per_class_path": resolve(
            "top_features_per_class_path",
            "top_features_per_class_csv",
        ),
        "latent_feature_influence_global_path": resolve(
            "latent_feature_influence_global_path",
            "latent_feature_influence_global_csv",
        ),
        "latent_feature_influence_per_class_path": resolve(
            "latent_feature_influence_per_class_path",
            "latent_feature_influence_per_class_csv",
        ),
        "flowpre_projection_manifest_path": resolve(
            "flowpre_projection_manifest_path",
            "flowpre_projection_manifest_json",
        ),
        "flowpre_projection_cache_path": resolve(
            "flowpre_projection_cache_path",
        ),
        "xgb_shap_feature_influence_global_path": resolve(
            "xgb_shap_feature_influence_global_path",
            "xgb_shap_feature_influence_global_csv",
        ),
        "xgb_shap_feature_influence_per_class_path": resolve(
            "xgb_shap_feature_influence_per_class_path",
            "xgb_shap_feature_influence_per_class_csv",
        ),
        "xgb_shap_top_features_global_path": resolve(
            "xgb_shap_top_features_global_path",
            "xgb_shap_top_features_global_csv",
        ),
        "xgb_shap_top_features_per_class_path": resolve(
            "xgb_shap_top_features_per_class_path",
            "xgb_shap_top_features_per_class_csv",
        ),
        "xgb_perturbation_feature_influence_global_path": resolve(
            "xgb_perturbation_feature_influence_global_path",
            "xgb_perturbation_feature_influence_global_csv",
        ),
        "xgb_perturbation_feature_influence_per_class_path": resolve(
            "xgb_perturbation_feature_influence_per_class_path",
            "xgb_perturbation_feature_influence_per_class_csv",
        ),
        "xgb_perturbation_top_features_global_path": resolve(
            "xgb_perturbation_top_features_global_path",
            "xgb_perturbation_top_features_global_csv",
        ),
        "xgb_perturbation_top_features_per_class_path": resolve(
            "xgb_perturbation_top_features_per_class_path",
            "xgb_perturbation_top_features_per_class_csv",
        ),
    }
    if family == "mlp":
        payload.setdefault("xgb_shap_feature_influence_global_path", None)
    return payload


def _extract_state_fields_from_run_manifest(
    *,
    run_manifest: dict[str, Any],
    family: str,
    run_dir: Path,
) -> dict[str, Any]:
    raw_metric_contract_validation = dict(run_manifest.get("raw_metric_contract_validation") or {})
    raw_inversion_status_payload = dict(
        raw_metric_contract_validation.get("raw_inversion_status")
        or run_manifest.get("raw_inversion_status")
        or {}
    )
    results_path = (
        run_manifest.get("results_path")
        or dict(run_manifest.get("artifact_paths") or {}).get("results_yaml")
    )
    metrics_long_path = (
        run_manifest.get("metrics_long_path")
        or dict(run_manifest.get("artifact_paths") or {}).get("metrics_long_csv")
        or str(run_dir / "metrics_long.csv")
    )
    prediction_sidecar_path = (
        run_manifest.get("prediction_sidecar_path")
        or dict(run_manifest.get("artifact_paths") or {}).get("predictions_eval_raw_csv_gz")
    )
    config_path = run_manifest.get("config_path") or run_manifest.get("config_snapshot_path")
    analysis_contracts = dict(run_manifest.get("analysis_contracts") or {})
    parsed_factor_fields = dict(run_manifest.get("parsed_factor_fields") or {})
    feature_namespace, primary_surface = resolve_feature_namespace(
        model_family=family,
        has_input_projection=bool(
            run_manifest.get("input_feature_influence_global_path")
            or dict(run_manifest.get("artifact_paths") or {}).get("input_feature_influence_global_csv")
        ),
        has_latent_surface=bool(
            run_manifest.get("latent_feature_influence_global_path")
            or dict(run_manifest.get("artifact_paths") or {}).get("latent_feature_influence_global_csv")
        ),
        surface_hint="xgb_shap" if family == "xgboost" and bool(run_manifest.get("xgb_shap_feature_influence_global_path")) else None,
    )
    payload = {
        "run_manifest_path": path_relative_to_root(run_dir / "run_manifest.json"),
        "results_path": None if not results_path else path_relative_to_root(results_path),
        "metrics_long_path": None if not metrics_long_path else path_relative_to_root(metrics_long_path),
        "prediction_sidecar_path": None if not prediction_sidecar_path else path_relative_to_root(prediction_sidecar_path),
        "interpretability_summary_path": (
            None
            if not _resolve_interpretability_summary_path(run_manifest, family)
            else path_relative_to_root(_resolve_interpretability_summary_path(run_manifest, family))
        ),
        "contract_id": run_manifest.get("contract_id"),
        "raw_metric_contract_id": run_manifest.get("raw_metric_contract_id"),
        "raw_metric_contract_validation_status": raw_metric_contract_validation.get("validation_status"),
        "raw_real_available": raw_inversion_status_payload.get("raw_real_available"),
        "requires_raw_inversion": raw_inversion_status_payload.get("requires_raw_inversion"),
        "raw_inversion_status": raw_inversion_status_payload.get("status"),
        "value_space_default": raw_metric_contract_validation.get("value_space_default"),
        "x_transform": parsed_factor_fields.get("x_transform"),
        "y_transform": parsed_factor_fields.get("y_transform"),
        "synthetic_policy": parsed_factor_fields.get("synthetic_policy"),
        "run_policy": parsed_factor_fields.get("run_policy"),
        "flowpre_usage": parsed_factor_fields.get("flowpre_usage"),
        "flowgen_usage": parsed_factor_fields.get("flowgen_usage"),
        "class_ontology_contract_id": analysis_contracts.get("class_ontology_contract_id"),
        "class_ontology_contract_version": analysis_contracts.get("class_ontology_contract_version"),
        "target_contract_id": analysis_contracts.get("target_contract_id"),
        "target_contract_version": analysis_contracts.get("target_contract_version"),
        "metric_grammar_contract_id": analysis_contracts.get("metric_grammar_contract_id"),
        "metric_grammar_contract_version": analysis_contracts.get("metric_grammar_contract_version"),
        "metric_availability_contract_id": analysis_contracts.get("metric_availability_contract_id"),
        "metric_availability_contract_version": analysis_contracts.get("metric_availability_contract_version"),
        "metric_aggregation_contract_id": analysis_contracts.get("metric_aggregation_contract_id"),
        "metric_aggregation_contract_version": analysis_contracts.get("metric_aggregation_contract_version"),
        "evaluation_population_contract_id": analysis_contracts.get("evaluation_population_contract_id"),
        "evaluation_population_contract_version": analysis_contracts.get("evaluation_population_contract_version"),
        "prediction_row_join_contract_id": analysis_contracts.get("prediction_row_join_contract_id"),
        "prediction_row_join_contract_version": analysis_contracts.get("prediction_row_join_contract_version"),
        "prediction_row_join_key_kind": analysis_contracts.get("prediction_row_join_key_kind"),
        "feature_schema_contract_id": analysis_contracts.get("feature_schema_contract_id"),
        "feature_schema_contract_version": analysis_contracts.get("feature_schema_contract_version"),
        "factor_parser_contract_id": analysis_contracts.get("factor_parser_contract_id"),
        "factor_parser_contract_version": analysis_contracts.get("factor_parser_contract_version"),
        "factor_parser_version": analysis_contracts.get("factor_parser_version"),
        "metric_grammar_version": analysis_contracts.get("metric_grammar_version"),
        "lineage_aggregate_build_version": analysis_contracts.get("lineage_aggregate_build_version"),
        "panel_build_version": analysis_contracts.get("panel_build_version"),
        "panel_build_timestamp": analysis_contracts.get("panel_build_timestamp"),
        "target_name": analysis_contracts.get("target_name"),
        "target_space": analysis_contracts.get("target_space"),
        "target_unit_public": analysis_contracts.get("target_unit_public"),
        "split_class_support_source_path": None if not metrics_long_path else path_relative_to_root(metrics_long_path),
        "metric_availability_reference": analysis_contracts.get("metric_availability_contract_path"),
        "evaluation_population_reference": analysis_contracts.get("evaluation_population_contract_path"),
        "feature_schema_surface_id": primary_surface,
        "feature_namespace": feature_namespace,
        "primary_interpretability_surface_id": primary_surface,
        "git_commit": _try_git_commit(),
        "variant_fingerprint": run_manifest.get("variant_fingerprint"),
        "config_path": None if not config_path else path_relative_to_root(config_path),
        "config_sha256": run_manifest.get("config_sha256"),
        **_extract_interpretability_artifact_paths(run_manifest, family),
    }
    analysis_ready_state = {**payload, "campaign_valid_f7": run_manifest.get("campaign_valid_f7")}
    payload["analysis_ready_blockers"] = _collect_analysis_ready_blockers(analysis_ready_state)
    payload["analysis_ready_comparable"] = len(payload["analysis_ready_blockers"]) == 0
    return payload

def _detect_failure_code(exc: BaseException) -> str:
    message = str(exc).lower()
    if isinstance(exc, MemoryError) or "out of memory" in message or "cuda out of memory" in message or "oom" in message:
        return FAILURE_REASON_OOM
    if "interpretability" in message:
        return FAILURE_REASON_EXCEPTION_INTERPRETABILITY
    return FAILURE_REASON_EXCEPTION_TRAINING


def _update_state_for_blocked(
    state: dict[str, Any],
    *,
    reason_code: str,
    reason_detail: str,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    state = dict(state)
    state["execution_status"] = EXECUTION_STATUS_BLOCKED
    state["validity_status"] = VALIDITY_STATUS_UNKNOWN
    state["failure_reason_code"] = reason_code
    state["failure_reason_detail"] = reason_detail
    state["current_attempt_id"] = attempt_id
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state["last_event_at"] = state["finished_at"]
    return state


def execute_trial(
    *,
    paths: Any,
    trial_row: dict[str, str],
    device: str,
    runner_id: str,
    selection_index: int | None = None,
    selection_total: int | None = None,
) -> dict[str, Any]:
    trial_id = str(trial_row["trial_id"])
    state = load_trial_state(paths, trial_id)
    current_status = str(state.get("execution_status") or EXECUTION_STATUS_PENDING)
    if current_status == EXECUTION_STATUS_COMPLETED and bool(state.get("campaign_valid_f7")):
        append_attempt_event(
            paths,
            {
                "trial_id": trial_id,
                "event_type": "skip_completed_valid",
                "execution_status": current_status,
            },
        )
        return state

    lock_ok, lock_file = acquire_trial_lock(paths, trial_id, runner_id)
    if not lock_ok:
        state = _update_state_for_blocked(
            state,
            reason_code=FAILURE_REASON_LOCK_CONFLICT,
            reason_detail=f"Lock already present at {path_relative_to_root(lock_file)}",
        )
        save_trial_state(paths, state)
        append_attempt_event(
            paths,
            {
                "trial_id": trial_id,
                "event_type": "blocked_lock_conflict",
                "failure_reason_code": FAILURE_REASON_LOCK_CONFLICT,
            },
        )
        refresh_campaign_reporting(paths)
        return state

    attempt_index = next_attempt_index(state)
    attempt_id = derive_attempt_id(trial_id, attempt_index)
    run_id = derive_run_id(trial_id, attempt_index)
    run_dir = _family_run_dir(str(trial_row["campaign_id"]), str(trial_row["model_family"]), run_id)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    state = dict(state)
    state.update(
        {
            "attempt_count": attempt_index,
            "current_attempt_id": attempt_id,
            "current_run_id": run_id,
            "current_run_dir": path_relative_to_root(run_dir),
            "execution_status": EXECUTION_STATUS_RUNNING,
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
            "analysis_ready_comparable": False,
            "analysis_ready_blockers": [],
            "variant_fingerprint": None,
            "config_path": None,
            "config_sha256": None,
            "started_at": started_at,
            "last_event_at": started_at,
        }
    )
    save_trial_state(paths, state)
    append_attempt_event(
        paths,
        {
            "trial_id": trial_id,
            "attempt_id": attempt_id,
            "run_id": run_id,
            "event_type": "attempt_started",
        },
    )
    refresh_campaign_reporting(paths)

    caught_warnings: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            caught_warnings = caught
            warnings.simplefilter("always")
            validation = validate_trial_consumption_row(trial_row)
            runtime_issues = _validate_runtime_trial_requirements(trial_row)
            issues = [item for item in str(validation["issues"]).split(";") if item] + runtime_issues
            if validation["status"] != "ok" or runtime_issues:
                state = _update_state_for_blocked(
                    state,
                    reason_code=_preflight_reason_code(issues),
                    reason_detail=";".join(issues),
                    attempt_id=attempt_id,
                )
                warning_payload = _persist_captured_warnings(
                    paths=paths,
                    attempt_id=attempt_id,
                    trial_id=trial_id,
                    run_id=run_id,
                    selection_index=selection_index,
                    selection_total=selection_total,
                    caught=caught_warnings,
                )
                _apply_warning_capture_to_state(state, warning_payload)
                save_trial_state(paths, state)
                append_attempt_event(
                    paths,
                    {
                        "trial_id": trial_id,
                        "attempt_id": attempt_id,
                        "event_type": "blocked_preflight",
                        "failure_reason_code": state["failure_reason_code"],
                        "failure_reason_detail": state["failure_reason_detail"],
                        "warning_count": warning_payload["warning_count_total"],
                    },
                )
                refresh_campaign_reporting(paths)
                return state

            trial_row = dict(trial_row)
            trial_row["_attempt_id"] = attempt_id
            config_path = _build_effective_config(paths=paths, trial_row=trial_row)
            dataset_manifest = _load_json(trial_row["dataset_manifest_path"])
            X_train, X_val, X_test, y_train, y_val, y_test = _load_bundle_frames_from_manifest(dataset_manifest)
            evaluation_context = _build_evaluation_context(trial_row=trial_row, dataset_manifest=dataset_manifest)

            started_perf = time.perf_counter()
            if str(trial_row["model_family"]) == "mlp":
                _, _ = train_mlp_pipeline(
                    condition_col="type",
                    config_filename=str(config_path),
                    base_name="mlp",
                    run_id=run_id,
                    run_dir=run_dir,
                    device=device,
                    seed=int(trial_row["seed"]),
                    verbose=False,
                    allow_test_holdout=_bool_from_row(trial_row["allow_test_holdout"]),
                    X_train=X_train,
                    X_val=X_val,
                    X_test=X_test,
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                    evaluation_context=evaluation_context,
                )
            elif str(trial_row["model_family"]) == "xgboost":
                _, _ = train_xgboost_model(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    allow_test_holdout=_bool_from_row(trial_row["allow_test_holdout"]),
                    seed=int(trial_row["seed"]),
                    config_filename=str(config_path),
                    base_name="xgboost",
                    run_id=run_id,
                    run_dir=run_dir,
                    verbose=False,
                    evaluation_context=evaluation_context,
                )
            else:
                raise ValueError(f"Unsupported model family: {trial_row['model_family']}")
            _ = round(time.perf_counter() - started_perf, 3)

            stable_manifest_path = run_dir / "run_manifest.json"
            if not stable_manifest_path.exists():
                raise FileNotFoundError(f"Missing stable run manifest after execution: {stable_manifest_path}")
            run_manifest = _load_json(stable_manifest_path)
            training_summary = dict(run_manifest.get("training_summary") or {})
            training_runtime_s = training_summary.get("runtime_s")
            interpretability_runtime_s = _extract_interpretability_runtime(run_manifest, str(trial_row["model_family"]))

            warning_payload = _persist_captured_warnings(
                paths=paths,
                attempt_id=attempt_id,
                trial_id=trial_id,
                run_id=run_id,
                selection_index=selection_index,
                selection_total=selection_total,
                caught=caught_warnings,
            )
            state.update(
                {
                    "campaign_valid": run_manifest.get("campaign_valid"),
                    "campaign_valid_interpretability": run_manifest.get("campaign_valid_interpretability"),
                    "campaign_valid_f7": run_manifest.get("campaign_valid_f7"),
                    "observed_seed_count": 1,
                    "training_runtime_s": None if training_runtime_s is None else float(training_runtime_s),
                    "interpretability_runtime_s": interpretability_runtime_s,
                    "total_runtime_s": (
                        None
                        if training_runtime_s is None and interpretability_runtime_s is None
                        else round(float(training_runtime_s or 0.0) + float(interpretability_runtime_s or 0.0), 6)
                    ),
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "last_event_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            state.update(
                _extract_state_fields_from_run_manifest(
                    run_manifest=run_manifest,
                    family=str(trial_row["model_family"]),
                    run_dir=run_dir,
                )
            )
            analysis_ready_state = {**state, "campaign_valid_f7": run_manifest.get("campaign_valid_f7")}
            state["analysis_ready_blockers"] = _collect_analysis_ready_blockers(analysis_ready_state)
            state["analysis_ready_comparable"] = len(state["analysis_ready_blockers"]) == 0
            _apply_warning_capture_to_state(state, warning_payload)
            if not bool(run_manifest.get("campaign_valid_f7")):
                state["execution_status"] = EXECUTION_STATUS_FAILED
                state["validity_status"] = VALIDITY_STATUS_INVALID_F7
                state["failure_reason_code"] = FAILURE_REASON_INVALID_F7
                state["failure_reason_detail"] = "campaign_valid_f7=false in stable run manifest"
            else:
                state["execution_status"] = EXECUTION_STATUS_COMPLETED
                state["validity_status"] = VALIDITY_STATUS_VALID_F7
                state["successful_attempt_id"] = attempt_id
                state["successful_run_dir"] = path_relative_to_root(run_dir)
                state["failure_reason_code"] = None
                state["failure_reason_detail"] = None
            save_trial_state(paths, state)
            append_attempt_event(
                paths,
                {
                    "trial_id": trial_id,
                    "attempt_id": attempt_id,
                    "run_id": run_id,
                    "event_type": "attempt_finished",
                    "execution_status": state["execution_status"],
                    "campaign_valid_f7": state["campaign_valid_f7"],
                    "warning_count": warning_payload["warning_count_total"],
                    "warning_log_path": warning_payload["warning_log_path"],
                    "warning_policy_counts": warning_payload["warning_policy_counts"],
                },
            )
            refresh_campaign_reporting(paths)
            return state
    except FileNotFoundError as exc:
        warning_payload = _persist_captured_warnings(
            paths=paths,
            attempt_id=attempt_id,
            trial_id=trial_id,
            run_id=run_id,
            selection_index=selection_index,
            selection_total=selection_total,
            caught=caught_warnings,
        )
        state["execution_status"] = EXECUTION_STATUS_FAILED
        state["validity_status"] = VALIDITY_STATUS_UNKNOWN
        state["failure_reason_code"] = FAILURE_REASON_MISSING_RUN_ARTIFACTS
        state["failure_reason_detail"] = str(exc)
        state["exception_type"] = type(exc).__name__
        state["exception_message"] = str(exc)
        _apply_warning_capture_to_state(state, warning_payload)
        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["last_event_at"] = state["finished_at"]
        save_trial_state(paths, state)
        append_attempt_event(
            paths,
            {
                "trial_id": trial_id,
                "attempt_id": attempt_id,
                "run_id": run_id,
                "event_type": "attempt_failed",
                "failure_reason_code": FAILURE_REASON_MISSING_RUN_ARTIFACTS,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "warning_count": warning_payload["warning_count_total"],
                "warning_log_path": warning_payload["warning_log_path"],
            },
        )
        refresh_campaign_reporting(paths)
        return state
    except KeyboardInterrupt:
        warning_payload = _persist_captured_warnings(
            paths=paths,
            attempt_id=attempt_id,
            trial_id=trial_id,
            run_id=run_id,
            selection_index=selection_index,
            selection_total=selection_total,
            caught=caught_warnings,
        )
        state["execution_status"] = EXECUTION_STATUS_FAILED
        state["validity_status"] = VALIDITY_STATUS_UNKNOWN
        state["failure_reason_code"] = FAILURE_REASON_MANUAL_ABORT
        state["failure_reason_detail"] = "Interrupted by user"
        _apply_warning_capture_to_state(state, warning_payload)
        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["last_event_at"] = state["finished_at"]
        save_trial_state(paths, state)
        append_attempt_event(
            paths,
            {
                "trial_id": trial_id,
                "attempt_id": attempt_id,
                "run_id": run_id,
                "event_type": "attempt_aborted",
                "failure_reason_code": FAILURE_REASON_MANUAL_ABORT,
                "warning_count": warning_payload["warning_count_total"],
                "warning_log_path": warning_payload["warning_log_path"],
            },
        )
        refresh_campaign_reporting(paths)
        raise
    except Exception as exc:  # noqa: BLE001
        warning_payload = _persist_captured_warnings(
            paths=paths,
            attempt_id=attempt_id,
            trial_id=trial_id,
            run_id=run_id,
            selection_index=selection_index,
            selection_total=selection_total,
            caught=caught_warnings,
        )
        failure_code = _detect_failure_code(exc)
        traceback_path = paths.tracebacks_dir / f"{attempt_id}.log"
        traceback_path.parent.mkdir(parents=True, exist_ok=True)
        traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
        state["execution_status"] = EXECUTION_STATUS_FAILED
        state["validity_status"] = VALIDITY_STATUS_INVALID_F7 if failure_code == FAILURE_REASON_INVALID_F7 else VALIDITY_STATUS_UNKNOWN
        state["failure_reason_code"] = failure_code
        state["failure_reason_detail"] = str(exc)
        state["exception_type"] = type(exc).__name__
        state["exception_message"] = str(exc)
        state["traceback_path"] = path_relative_to_root(traceback_path)
        _apply_warning_capture_to_state(state, warning_payload)
        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["last_event_at"] = state["finished_at"]
        save_trial_state(paths, state)
        append_attempt_event(
            paths,
            {
                "trial_id": trial_id,
                "attempt_id": attempt_id,
                "run_id": run_id,
                "event_type": "attempt_failed",
                "failure_reason_code": failure_code,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback_path": path_relative_to_root(traceback_path),
                "warning_count": warning_payload["warning_count_total"],
                "warning_log_path": warning_payload["warning_log_path"],
            },
        )
        refresh_campaign_reporting(paths)
        return state
    finally:
        release_trial_lock(paths, trial_id)


def _blocked_states_for_selection(
    *,
    paths: Any,
    trial_rows: list[dict[str, str]],
    reason_code: str,
    reason_detail: str,
) -> None:
    for trial_row in trial_rows:
        state = load_trial_state(paths, str(trial_row["trial_id"]))
        state = _update_state_for_blocked(state, reason_code=reason_code, reason_detail=reason_detail)
        save_trial_state(paths, state)
        append_attempt_event(
            paths,
            {
                "trial_id": trial_row["trial_id"],
                "event_type": "blocked_campaign_level",
                "failure_reason_code": reason_code,
                "failure_reason_detail": reason_detail,
            },
        )
    refresh_campaign_reporting(paths)


def _resolve_spec_and_paths_for_run(spec_path: str | Path) -> tuple[dict[str, Any], MaterializedCampaignSpec, Any]:
    spec, materialized = initialize_campaign_from_spec(spec_path)
    paths = build_campaign_paths(str(spec["campaign_id"]))
    return spec, materialized, paths


def run_preflight(
    *,
    spec_path: str | Path = DEFAULT_SPEC_PATH,
    model_family: str | None = None,
    dataset_candidate_id: str | None = None,
    run_spec_id: str | None = None,
    trial_id: str | None = None,
    trial_id_file: str | Path | None = None,
    offset: int | None = None,
    limit: int | None = None,
    status_filter: str | None = None,
) -> dict[str, Any]:
    spec, materialized, paths = _resolve_spec_and_paths_for_run(spec_path)
    states = {state["trial_id"]: state for state in iter_trial_states(paths)}
    selection = filter_trial_rows(
        load_frozen_trial_inventory(paths),
        states_by_id=states,
        model_family=model_family,
        dataset_candidate_id=dataset_candidate_id,
        run_spec_id=run_spec_id,
        trial_id=trial_id,
        trial_id_file=trial_id_file,
        offset=offset,
        limit=limit,
        status_filter=status_filter,
    )
    lineage = validate_extension_lineage(spec=spec, materialized=materialized)
    report = preflight_trials(spec=spec, trial_rows=selection.rows, lineage_validation=lineage)
    write_preflight_report(paths, report)
    return report


def _run_selected_trials(
    *,
    spec: dict[str, Any],
    materialized: MaterializedCampaignSpec,
    paths: Any,
    selection: RunnerSelection,
    device: str,
    runner_id: str,
    cleanup_every: int = 1,
) -> dict[str, Any]:
    if not selection.rows:
        report = {
            "campaign_id": spec["campaign_id"],
            "campaign_spec_id": spec["campaign_spec_id"],
            "checked_trial_count": 0,
            "ok": False,
            "message": "No trials selected after filtering",
        }
        write_preflight_report(paths, report)
        return report

    lineage = validate_extension_lineage(spec=spec, materialized=materialized)
    report = preflight_trials(spec=spec, trial_rows=selection.rows, lineage_validation=lineage)
    write_preflight_report(paths, report)
    if not lineage["ok"]:
        reason_detail = ";".join(str(item) for item in lineage["issues"])
        _blocked_states_for_selection(
            paths=paths,
            trial_rows=selection.rows,
            reason_code=FAILURE_REASON_PARENT_NOT_CLOSED if FAILURE_REASON_PARENT_NOT_CLOSED in lineage["issues"] else FAILURE_REASON_PRECHECK_INVALID_TRIAL,
            reason_detail=reason_detail,
        )
        return report

    _prewarm_flowpre_assets_for_rows(selection.rows)
    wall_clock_started = time.perf_counter()
    selected_trial_count = len(selection.rows)
    executed_trial_count = 0
    completed_total_runtimes: list[float | None] = []
    completed_interp_runtimes: list[float | None] = []
    for idx, trial_row in enumerate(selection.rows, start=1):
        state = execute_trial(
            paths=paths,
            trial_row=trial_row,
            device=device,
            runner_id=runner_id,
            selection_index=idx,
            selection_total=selected_trial_count,
        )
        executed_trial_count += 1
        last_train = state.get("training_runtime_s")
        last_interp = state.get("interpretability_runtime_s")
        last_total = state.get("total_runtime_s")
        completed_total_runtimes.append(last_total)
        completed_interp_runtimes.append(last_interp)
        avg_total = _safe_mean_runtime(completed_total_runtimes)
        avg_interp = _safe_mean_runtime(completed_interp_runtimes)
        progress_prefix = f"[{idx}/{selected_trial_count}]"
        if state.get("execution_status") in {EXECUTION_STATUS_FAILED, EXECUTION_STATUS_BLOCKED}:
            print(
                (
                    f"{progress_prefix} {state.get('execution_status').upper()} "
                    f"trial_id={state.get('trial_id')} "
                    f"reason={state.get('failure_reason_code')} "
                    f"last_train={_format_runtime(last_train)} "
                    f"last_interp={_format_runtime(last_interp)} "
                    f"last_total={_format_runtime(last_total)} "
                    f"avg_total={_format_runtime(avg_total)} "
                    f"avg_interp={_format_runtime(avg_interp)}"
                ),
                flush=True,
            )
        if cleanup_every > 0 and idx % cleanup_every == 0:
            _light_memory_cleanup()
            print(
                (
                    f"{progress_prefix} cleanup "
                    f"last_train={_format_runtime(last_train)} "
                    f"last_interp={_format_runtime(last_interp)} "
                    f"last_total={_format_runtime(last_total)} "
                    f"avg_total={_format_runtime(avg_total)} "
                    f"avg_interp={_format_runtime(avg_interp)}"
                ),
                flush=True,
            )

    states, summary, manifest = refresh_campaign_reporting(paths)
    if manifest["campaign_status"] in {CAMPAIGN_STATUS_CLOSED_SUCCESS, CAMPAIGN_STATUS_CLOSED_WITH_FAILURES}:
        write_campaign_closeout(paths)
    wall_clock_s = round(time.perf_counter() - wall_clock_started, 3)
    return {
        "campaign_id": spec["campaign_id"],
        "campaign_status": manifest["campaign_status"],
        "counts_by_status": summary["counts_by_status"],
        "completed_valid_f7_count": summary["completed_valid_f7_count"],
        "trial_count_total": summary["trial_count_total"],
        "selected_trial_count": selected_trial_count,
        "executed_trial_count": executed_trial_count,
        "cleanup_every": cleanup_every,
        "runner_wall_clock_s": wall_clock_s,
    }


def run_campaign(
    *,
    spec_path: str | Path = DEFAULT_SPEC_PATH,
    device: str = "cpu",
    runner_id: str | None = None,
    model_family: str | None = None,
    dataset_candidate_id: str | None = None,
    run_spec_id: str | None = None,
    trial_id: str | None = None,
    trial_id_file: str | Path | None = None,
    offset: int | None = None,
    limit: int | None = None,
    status_filter: str | None = None,
    cleanup_every: int = 1,
) -> dict[str, Any]:
    spec, materialized, paths = _resolve_spec_and_paths_for_run(spec_path)
    states = {state["trial_id"]: state for state in iter_trial_states(paths)}
    selection = filter_trial_rows(
        load_frozen_trial_inventory(paths),
        states_by_id=states,
        model_family=model_family,
        dataset_candidate_id=dataset_candidate_id,
        run_spec_id=run_spec_id,
        trial_id=trial_id,
        trial_id_file=trial_id_file,
        offset=offset,
        limit=limit,
        status_filter=status_filter,
    )
    return _run_selected_trials(
        spec=spec,
        materialized=materialized,
        paths=paths,
        selection=selection,
        device=device,
        runner_id=runner_id or f"runner-{os.getpid()}",
        cleanup_every=cleanup_every,
    )


def resume_campaign(
    *,
    campaign_id: str,
    device: str = "cpu",
    runner_id: str | None = None,
    model_family: str | None = None,
    dataset_candidate_id: str | None = None,
    run_spec_id: str | None = None,
    trial_id: str | None = None,
    trial_id_file: str | Path | None = None,
    offset: int | None = None,
    limit: int | None = None,
    status_filter: str | None = None,
    cleanup_every: int = 1,
) -> dict[str, Any]:
    spec, trial_rows = load_existing_campaign(campaign_id)
    paths = build_campaign_paths(campaign_id)
    states = {state["trial_id"]: state for state in iter_trial_states(paths)}
    effective_status_filter = status_filter or f"{EXECUTION_STATUS_PENDING},{EXECUTION_STATUS_COMPLETED}"
    selection = filter_trial_rows(
        trial_rows,
        states_by_id=states,
        model_family=model_family,
        dataset_candidate_id=dataset_candidate_id,
        run_spec_id=run_spec_id,
        trial_id=trial_id,
        trial_id_file=trial_id_file,
        offset=offset,
        limit=limit,
        status_filter=effective_status_filter,
    )
    materialized = materialize_f7_campaign_spec(spec_path=_resolve_frozen_spec_path(paths), write_outputs=False)
    return _run_selected_trials(
        spec=spec,
        materialized=materialized,
        paths=paths,
        selection=selection,
        device=device,
        runner_id=runner_id or f"runner-{os.getpid()}",
        cleanup_every=cleanup_every,
    )


def rerun_failed_campaign(
    *,
    campaign_id: str,
    device: str = "cpu",
    runner_id: str | None = None,
    reason_code_filter: str | None = None,
    model_family: str | None = None,
    dataset_candidate_id: str | None = None,
    run_spec_id: str | None = None,
    trial_id: str | None = None,
    trial_id_file: str | Path | None = None,
    offset: int | None = None,
    limit: int | None = None,
    cleanup_every: int = 1,
) -> dict[str, Any]:
    spec, trial_rows = load_existing_campaign(campaign_id)
    paths = build_campaign_paths(campaign_id)
    states_list = iter_trial_states(paths)
    states = {state["trial_id"]: state for state in states_list}
    filtered_ids = {
        state["trial_id"]
        for state in states_list
        if state.get("execution_status") in {EXECUTION_STATUS_FAILED, EXECUTION_STATUS_BLOCKED}
        and (reason_code_filter is None or str(state.get("failure_reason_code")) == str(reason_code_filter))
    }
    selection = filter_trial_rows(
        [row for row in trial_rows if str(row["trial_id"]) in filtered_ids],
        states_by_id=states,
        model_family=model_family,
        dataset_candidate_id=dataset_candidate_id,
        run_spec_id=run_spec_id,
        trial_id=trial_id,
        trial_id_file=trial_id_file,
        offset=offset,
        limit=limit,
    )
    materialized = materialize_f7_campaign_spec(spec_path=_resolve_frozen_spec_path(paths), write_outputs=False)
    return _run_selected_trials(
        spec=spec,
        materialized=materialized,
        paths=paths,
        selection=selection,
        device=device,
        runner_id=runner_id or f"runner-{os.getpid()}",
        cleanup_every=cleanup_every,
    )


def close_campaign(*, campaign_id: str) -> dict[str, Any]:
    paths = build_campaign_paths(campaign_id)
    return write_campaign_closeout(paths)


def rebuild_campaign_state(*, campaign_id: str) -> dict[str, Any]:
    paths = build_campaign_paths(campaign_id)
    trial_rows = load_frozen_trial_inventory(paths)
    trial_index = _build_trial_row_index(trial_rows)
    initialize_trial_state_files(paths, trial_rows)

    run_manifest_paths: list[Path] = []
    for model_family in ("mlp", "xgboost"):
        family_root = Path(ROOT_PATH) / "outputs" / "models" / model_family
        for candidate_root in (
            family_root / "campaigns" / campaign_id,
            family_root / campaign_id,
        ):
            if candidate_root.exists():
                run_manifest_paths.extend(candidate_root.glob("**/run_manifest.json"))
    grouped: dict[str, list[Path]] = {}
    for path in run_manifest_paths:
        payload = _load_json(path)
        trial_id = payload.get("trial_id")
        if trial_id not in trial_index:
            run_id = payload.get("run_id")
            if isinstance(run_id, str) and "__attempt-" in run_id:
                fallback_trial_id = run_id.rsplit("__attempt-", 1)[0]
                if fallback_trial_id in trial_index:
                    trial_id = fallback_trial_id
        if trial_id:
            grouped.setdefault(str(trial_id), []).append(path)

    for trial_id, manifest_paths in grouped.items():
        if trial_id not in trial_index:
            continue
        latest_path = sorted(manifest_paths, key=lambda item: item.stat().st_mtime)[-1]
        latest_manifest = _load_json(latest_path)
        state = load_trial_state(paths, trial_id)
        recovered_run_id = latest_manifest.get("run_id")
        recovered_attempt_index = _attempt_index_from_run_id(recovered_run_id)
        recovered_attempt_id = (
            None
            if recovered_attempt_index is None
            else derive_attempt_id(trial_id, recovered_attempt_index)
        )
        state["current_run_dir"] = path_relative_to_root(latest_path.parent)
        state["current_run_id"] = recovered_run_id
        state["current_attempt_id"] = recovered_attempt_id
        state["expected_seed_count"] = state.get("expected_seed_count") or trial_index[trial_id].get("expected_seed_count")
        state["observed_seed_count"] = 1
        state["campaign_valid"] = latest_manifest.get("campaign_valid")
        state["campaign_valid_interpretability"] = latest_manifest.get("campaign_valid_interpretability")
        state["campaign_valid_f7"] = latest_manifest.get("campaign_valid_f7")
        state.update(
            _extract_state_fields_from_run_manifest(
                run_manifest=latest_manifest,
                family=str(state["model_family"]),
                run_dir=latest_path.parent,
            )
        )
        training_summary = dict(latest_manifest.get("training_summary") or {})
        state["training_runtime_s"] = training_summary.get("runtime_s")
        state["interpretability_runtime_s"] = _extract_interpretability_runtime(latest_manifest, str(state["model_family"]))
        state["total_runtime_s"] = (
            None
            if state["training_runtime_s"] is None and state["interpretability_runtime_s"] is None
            else round(float(state["training_runtime_s"] or 0.0) + float(state["interpretability_runtime_s"] or 0.0), 6)
        )
        if bool(latest_manifest.get("campaign_valid_f7")):
            state["execution_status"] = EXECUTION_STATUS_COMPLETED
            state["validity_status"] = VALIDITY_STATUS_VALID_F7
            state["successful_attempt_id"] = recovered_attempt_id
            state["successful_run_dir"] = path_relative_to_root(latest_path.parent)
        else:
            state["execution_status"] = EXECUTION_STATUS_FAILED
            state["validity_status"] = VALIDITY_STATUS_INVALID_F7
            state["failure_reason_code"] = FAILURE_REASON_INVALID_F7
            state["failure_reason_detail"] = "Recovered from run manifests with campaign_valid_f7=false"
        if recovered_attempt_index is not None:
            state["attempt_count"] = max(int(state.get("attempt_count") or 0), recovered_attempt_index)
        warning_payload = _load_warning_payload_from_log(
            None if recovered_attempt_id is None else _warning_log_path(paths, recovered_attempt_id)
        )
        _apply_warning_capture_to_state(state, warning_payload)
        analysis_ready_state = {**state, "campaign_valid_f7": state.get("campaign_valid_f7")}
        state["analysis_ready_blockers"] = _collect_analysis_ready_blockers(analysis_ready_state)
        state["analysis_ready_comparable"] = len(state["analysis_ready_blockers"]) == 0
        save_trial_state(paths, state)

    _, summary, manifest = refresh_campaign_reporting(paths)
    return {
        "campaign_id": campaign_id,
        "campaign_status": manifest["campaign_status"],
        "counts_by_status": summary["counts_by_status"],
        "trial_count_total": summary["trial_count_total"],
    }


def maybe_chain_campaigns(
    *,
    current_campaign_id: str,
    next_spec_paths: list[str | Path],
    allow_chain_on_closed_with_failures: bool,
    device: str,
    runner_id: str | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    current_paths = build_campaign_paths(current_campaign_id)
    current_manifest = load_campaign_manifest(current_paths)
    allowed_statuses = {CAMPAIGN_STATUS_CLOSED_SUCCESS}
    if allow_chain_on_closed_with_failures:
        allowed_statuses.add(CAMPAIGN_STATUS_CLOSED_WITH_FAILURES)
    if current_manifest.get("campaign_status") not in allowed_statuses:
        return results
    for spec_path in next_spec_paths:
        result = run_campaign(spec_path=spec_path, device=device, runner_id=runner_id)
        results.append(result)
        chained_paths = build_campaign_paths(str(result["campaign_id"]))
        chained_manifest = load_campaign_manifest(chained_paths)
        if chained_manifest.get("campaign_status") not in allowed_statuses:
            break
    return results
