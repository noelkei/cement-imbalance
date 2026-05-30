from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.utils import ROOT_PATH
from evaluation.f7_campaign_runner import preflight_trials
from evaluation.f7_campaign_spec import MaterializedCampaignSpec, materialize_f7_campaign_spec


DEFAULT_PRIMARY_SPEC_PATH = Path(ROOT_PATH) / "config" / "f7_campaign_spec_v1.yaml"
DEFAULT_EXTENSION_SPEC_PATHS = [
    Path(ROOT_PATH) / "config" / "f7_campaign_extension1_v1.yaml",
    Path(ROOT_PATH) / "config" / "f7_campaign_extension2_v1.yaml",
    Path(ROOT_PATH) / "config" / "f7_campaign_extension3_v1.yaml",
]
DEFAULT_OUTPUT_DIR = Path(ROOT_PATH) / "outputs" / "reports" / "f7_launch_readiness"
DEFAULT_JSON_PATH = DEFAULT_OUTPUT_DIR / "f7_launch_readiness_v1.json"
DEFAULT_MD_PATH = DEFAULT_OUTPUT_DIR / "f7_launch_readiness_v1.md"


@dataclass(frozen=True)
class LaunchReadinessRecord:
    spec_path: Path
    spec: dict[str, Any]
    materialized: MaterializedCampaignSpec


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", None) or "")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_PATH,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def collect_environment_freeze() -> dict[str, Any]:
    return {
        "captured_at": _utc_now_iso(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": _safe_module_version("torch"),
        "xgboost_version": _safe_module_version("xgboost"),
        "git_commit": _git_commit(),
        "root_path": str(ROOT_PATH),
    }


def _run_spec_signature_set(rows: list[dict[str, Any]]) -> set[tuple[tuple[str, str], ...]]:
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
    signatures: set[tuple[tuple[str, str], ...]] = set()
    for row in rows:
        signatures.add(tuple((field, str(row.get(field) or "")) for field in signature_fields))
    return signatures


def _materialize_records(primary_spec_path: str | Path, extension_spec_paths: list[str | Path]) -> list[LaunchReadinessRecord]:
    spec_paths = [Path(primary_spec_path), *[Path(path) for path in extension_spec_paths]]
    records: list[LaunchReadinessRecord] = []
    for path in spec_paths:
        materialized = materialize_f7_campaign_spec(spec_path=path, write_outputs=False)
        records.append(LaunchReadinessRecord(spec_path=Path(path), spec=materialized.spec, materialized=materialized))
    return records


def validate_planned_campaign_chain(
    primary_spec_path: str | Path = DEFAULT_PRIMARY_SPEC_PATH,
    extension_spec_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    records = _materialize_records(primary_spec_path, extension_spec_paths or DEFAULT_EXTENSION_SPEC_PATHS)
    root_record = records[0]
    lineage_issues: list[str] = []
    per_campaign: dict[str, dict[str, Any]] = {}
    accumulated_seed_values: set[int] = set()
    expected_total_trials = 0

    lineage_id = str(root_record.spec.get("campaign_lineage_id") or "")
    root_campaign_id = str(root_record.spec.get("campaign_id") or "")
    pooling_group_id = str(root_record.spec.get("pooling_group_id") or "")

    for index, record in enumerate(records):
        spec = record.spec
        materialized = record.materialized
        campaign_id = str(spec.get("campaign_id") or "")
        issues: list[str] = []

        dataset_count = len(materialized.dataset_candidates)
        run_spec_count = len(materialized.run_specs)
        trial_count = len(materialized.trials)
        expected_counts = dict(spec.get("expected_counts") or {})
        expected_dataset_total = int((expected_counts.get("dataset_candidates") or {}).get("total") or 0)
        expected_run_spec_total = int((expected_counts.get("run_specs") or {}).get("total") or 0)
        expected_trial_total = int((expected_counts.get("trials") or {}).get("total") or 0)
        if expected_dataset_total and dataset_count != expected_dataset_total:
            issues.append("dataset_candidate_count_mismatch")
        if expected_run_spec_total and run_spec_count != expected_run_spec_total:
            issues.append("run_spec_count_mismatch")
        if expected_trial_total and trial_count != expected_trial_total:
            issues.append("trial_count_mismatch")

        expected_replication = dict(materialized.expansion_manifest.get("expected_replication") or {})
        expected_seed_values = {int(value) for value in list(expected_replication.get("expected_seed_values") or [])}
        observed_seed_values = {int(row["seed"]) for row in materialized.trials}
        if expected_seed_values and observed_seed_values != expected_seed_values:
            issues.append("expected_seed_values_mismatch")

        if index == 0:
            if str(spec.get("campaign_kind")) != "primary":
                issues.append("root_campaign_not_primary")
            if str(spec.get("root_campaign_id")) != campaign_id:
                issues.append("root_campaign_id_mismatch")
        else:
            parent_record = records[index - 1]
            parent_spec = parent_record.spec
            parent_materialized = parent_record.materialized
            if str(spec.get("campaign_kind")) != "extension":
                issues.append("non_root_campaign_not_extension")
            if str(spec.get("extension_type")) != "seed_extension":
                issues.append("invalid_extension_type")
            if str(spec.get("parent_campaign_id")) != str(parent_spec.get("campaign_id")):
                issues.append("parent_campaign_id_mismatch")
            if str(spec.get("root_campaign_id")) != root_campaign_id:
                issues.append("root_campaign_id_mismatch")
            if str(spec.get("campaign_lineage_id")) != lineage_id:
                issues.append("campaign_lineage_id_mismatch")
            if str(spec.get("pooling_group_id")) != pooling_group_id:
                issues.append("pooling_group_id_mismatch")

            structural_keys = ("campaign_scope", "run_mode", "allow_test_holdout", "test_enabled", "meta_grammar_id")
            for key in structural_keys:
                if str(parent_spec.get(key)) != str(spec.get(key)):
                    issues.append(f"structural_spec_mismatch:{key}")

            parent_dataset_ids = {str(row["dataset_candidate_id"]) for row in parent_materialized.dataset_candidates}
            child_dataset_ids = {str(row["dataset_candidate_id"]) for row in materialized.dataset_candidates}
            if parent_dataset_ids != child_dataset_ids:
                issues.append("dataset_candidate_set_mismatch")

            parent_run_spec_ids = {str(row["run_spec_id"]) for row in parent_materialized.run_specs}
            child_run_spec_ids = {str(row["run_spec_id"]) for row in materialized.run_specs}
            if parent_run_spec_ids != child_run_spec_ids:
                issues.append("run_spec_set_mismatch")

            parent_signatures = _run_spec_signature_set(parent_materialized.run_specs)
            child_signatures = _run_spec_signature_set(materialized.run_specs)
            if len(parent_signatures) != 1:
                issues.append("parent_run_spec_contract_signature_inconsistent")
            if len(child_signatures) != 1:
                issues.append("child_run_spec_contract_signature_inconsistent")
            if parent_signatures != child_signatures:
                issues.append("global_contract_signature_mismatch")

            overlap = sorted(accumulated_seed_values & observed_seed_values)
            if overlap:
                issues.append("seed_overlap_with_prior_lineage")

        accumulated_seed_values |= observed_seed_values
        expected_total_trials += expected_trial_total
        per_campaign[campaign_id] = {
            "campaign_id": campaign_id,
            "spec_path": str(record.spec_path),
            "campaign_kind": spec.get("campaign_kind"),
            "ok": not issues,
            "issues": issues,
            "dataset_candidate_count": dataset_count,
            "run_spec_count": run_spec_count,
            "trial_count": trial_count,
            "expected_trial_count": expected_trial_total,
            "seed_count": len(observed_seed_values),
            "expected_seed_count": len(expected_seed_values),
            "seed_values": sorted(observed_seed_values),
        }
        lineage_issues.extend(f"{campaign_id}:{issue}" for issue in issues)

    total_trial_count = sum(len(record.materialized.trials) for record in records)
    if expected_total_trials and total_trial_count != expected_total_trials:
        lineage_issues.append("global_trial_count_mismatch")

    return {
        "ok": not lineage_issues,
        "issues": lineage_issues,
        "root_campaign_id": root_campaign_id,
        "campaign_lineage_id": lineage_id,
        "campaign_count": len(records),
        "expected_total_trial_count": expected_total_trials,
        "observed_total_trial_count": total_trial_count,
        "expected_total_seed_count": len(accumulated_seed_values),
        "observed_total_seed_count": len(accumulated_seed_values),
        "per_campaign": per_campaign,
    }


def _planned_lineage_issue_map(validation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(campaign_id): {
            "ok": bool(payload.get("ok")),
            "issues": list(payload.get("issues") or []),
        }
        for campaign_id, payload in dict(validation.get("per_campaign") or {}).items()
    }


def _readiness_markers(preflight_reports: list[dict[str, Any]], chain_validation: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not chain_validation.get("ok"):
        blockers.extend([f"planned_chain:{issue}" for issue in list(chain_validation.get("issues") or [])])
    for report in preflight_reports:
        if not report.get("ok"):
            campaign_id = str(report.get("campaign_id") or "")
            blockers.extend([f"{campaign_id}:trial_preflight_failed"])
        blockers.extend([f"{report.get('campaign_id')}:{issue}" for issue in list(report.get("blocking_contract_failures") or [])])
        blockers.extend([f"{report.get('campaign_id')}:{issue}" for issue in list(report.get("blocking_structural_failures") or [])])
    return {"ok": not blockers, "blockers": sorted(set(blockers))}


def _build_readiness_markdown(payload: dict[str, Any]) -> str:
    markers = dict(payload.get("readiness_markers") or {})
    lines = [
        "# F7 Launch Readiness Report",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- go_no_go: `{payload.get('go_no_go')}`",
        f"- blocker_count: `{len(list(markers.get('blockers') or []))}`",
        "",
        "## Chain",
        "",
        f"- root_campaign_id: `{payload.get('root_campaign_id')}`",
        f"- campaign_count: `{payload.get('campaign_count')}`",
        f"- expected_total_trial_count: `{payload.get('expected_total_trial_count')}`",
        f"- observed_total_trial_count: `{payload.get('observed_total_trial_count')}`",
        f"- total_seed_count: `{payload.get('total_seed_count')}`",
        "",
        "## Per Campaign",
        "",
    ]
    for report in list(payload.get("campaign_preflight_reports") or []):
        lines.extend(
            [
                f"- `{report.get('campaign_id')}`",
                f"  - checked_trial_count: `{report.get('checked_trial_count')}`",
                f"  - ok_count: `{report.get('ok_count')}`",
                f"  - failed_count: `{report.get('failed_count')}`",
                f"  - ok: `{report.get('ok')}`",
            ]
        )
    blockers = list(markers.get("blockers") or [])
    lines.extend(["", "## Go / No-Go", ""])
    if blockers:
        lines.append("- blockers:")
        for blocker in blockers:
            lines.append(f"  - `{blocker}`")
    else:
        lines.append("- blockers: `[]`")
    env = dict(payload.get("environment_freeze") or {})
    lines.extend(
        [
            "",
            "## Environment Freeze",
            "",
            f"- python_version: `{env.get('python_version')}`",
            f"- torch_version: `{env.get('torch_version')}`",
            f"- xgboost_version: `{env.get('xgboost_version')}`",
            f"- git_commit: `{env.get('git_commit')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_f7_launch_readiness_report(
    *,
    primary_spec_path: str | Path = DEFAULT_PRIMARY_SPEC_PATH,
    extension_spec_paths: list[str | Path] | None = None,
    json_output_path: str | Path = DEFAULT_JSON_PATH,
    markdown_output_path: str | Path = DEFAULT_MD_PATH,
) -> dict[str, Any]:
    extension_spec_paths = extension_spec_paths or DEFAULT_EXTENSION_SPEC_PATHS
    records = _materialize_records(primary_spec_path, extension_spec_paths)
    chain_validation = validate_planned_campaign_chain(primary_spec_path, extension_spec_paths)
    lineage_issue_map = _planned_lineage_issue_map(chain_validation)

    preflight_reports: list[dict[str, Any]] = []
    for record in records:
        campaign_id = str(record.spec.get("campaign_id") or "")
        preflight = preflight_trials(
            spec=record.spec,
            trial_rows=record.materialized.trials,
            lineage_validation=lineage_issue_map.get(campaign_id) or {"ok": True, "issues": []},
        )
        preflight_reports.append(preflight)

    readiness_markers = _readiness_markers(preflight_reports, chain_validation)
    payload = {
        "generated_at": _utc_now_iso(),
        "root_campaign_id": chain_validation.get("root_campaign_id"),
        "campaign_lineage_id": chain_validation.get("campaign_lineage_id"),
        "campaign_count": chain_validation.get("campaign_count"),
        "expected_total_trial_count": chain_validation.get("expected_total_trial_count"),
        "observed_total_trial_count": chain_validation.get("observed_total_trial_count"),
        "total_seed_count": chain_validation.get("observed_total_seed_count"),
        "go_no_go": "go" if readiness_markers.get("ok") else "no_go",
        "environment_freeze": collect_environment_freeze(),
        "planned_chain_validation": chain_validation,
        "campaign_preflight_reports": preflight_reports,
        "readiness_markers": readiness_markers,
    }

    json_path = Path(json_output_path)
    markdown_path = Path(markdown_output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_build_readiness_markdown(payload), encoding="utf-8")
    return payload
