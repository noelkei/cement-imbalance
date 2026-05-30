from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from data.utils import ROOT_PATH, path_relative_to_root
from evaluation.f7_campaign_state import build_campaign_paths, load_campaign_manifest


LINEAGE_SURFACE_SEMANTIC_BRIDGE = "semantic_bridge_perturbation"
LINEAGE_SURFACE_XGB_NATIVE_SHAP = "xgb_native_shap"
LINEAGE_SURFACE_MLP_FLOWPRE_NATIVE_LATENT = "mlp_flowpre_native_latent_perturbation"
LINEAGE_REQUIRED_LEDGER_COLUMNS = {
    "campaign_id",
    "trial_id",
    "model_family",
    "dataset_candidate_id",
    "run_spec_id",
    "comparison_group_id",
    "lineage_trial_group_id",
    "seed",
    "objective_metric_id",
    "value_space_default",
    "metrics_long_path",
    "run_manifest_path",
    "raw_metric_contract_id",
    "raw_metric_contract_validation_status",
    "execution_status",
    "campaign_valid_f7",
    "analysis_ready_comparable",
    "class_ontology_contract_id",
    "target_contract_id",
    "metric_grammar_contract_id",
    "metric_availability_contract_id",
    "metric_aggregation_contract_id",
    "evaluation_population_contract_id",
    "prediction_row_join_contract_id",
    "feature_schema_contract_id",
    "factor_parser_contract_id",
    "prediction_row_join_key_kind",
    "feature_namespace",
    "primary_interpretability_surface_id",
    "expected_seed_count",
    "panel_build_version",
    "panel_build_timestamp",
    "lineage_aggregate_build_version",
}
LINEAGE_AUX_METRICS = ("mse", "rmse", "rrmse", "r2", "mape")
LINEAGE_SPLITS = ("train", "val", "test")
LINEAGE_DETAILED_SCOPES = (
    "overall",
    "overall_quantile",
    "macro",
    "worst_class",
    "per_class",
    "per_class_quantile",
)
LINEAGE_CONTRACT_FIELDS = (
    "class_ontology_contract_id",
    "target_contract_id",
    "metric_grammar_contract_id",
    "metric_availability_contract_id",
    "metric_aggregation_contract_id",
    "evaluation_population_contract_id",
    "feature_schema_contract_id",
    "factor_parser_contract_id",
    "raw_metric_contract_id",
)


@dataclass(frozen=True)
class LineageCampaignRecord:
    campaign_id: str
    paths: Any
    manifest: dict[str, Any]
    summary: dict[str, Any]
    ledger: pd.DataFrame


def _manifest_analysis_contracts(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = dict(manifest.get("analysis_contracts") or {})
    return payload


def _manifest_expected_replication(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = dict(manifest.get("expected_replication") or {})
    if payload:
        return payload
    frozen_inputs = dict(manifest.get("frozen_inputs") or {})
    candidate_path = frozen_inputs.get("expected_replication_manifest_path")
    if candidate_path:
        resolved = _repo_path(candidate_path)
        if resolved.exists():
            loaded = json.loads(resolved.read_text(encoding="utf-8"))
            return dict(loaded or {})
    return {}


def _campaign_seed_values(record: LineageCampaignRecord) -> set[int]:
    expected_replication = _manifest_expected_replication(record.manifest)
    expected_seed_values = expected_replication.get("expected_seed_values")
    if isinstance(expected_seed_values, list):
        return {int(value) for value in expected_seed_values}
    if "seed" in record.ledger.columns:
        return {
            int(value)
            for value in pd.to_numeric(record.ledger["seed"], errors="coerce").dropna().astype(int).tolist()
        }
    return set()


def _contract_signature_for_record(record: LineageCampaignRecord) -> dict[str, str]:
    analysis_contracts = _manifest_analysis_contracts(record.manifest)
    ledger = record.ledger
    signature: dict[str, str] = {}
    for field in LINEAGE_CONTRACT_FIELDS:
        manifest_value = analysis_contracts.get(field)
        if manifest_value is not None:
            signature[field] = str(manifest_value)
            continue
        values = {str(item) for item in ledger.get(field, pd.Series(dtype=object)).dropna().unique()}
        signature[field] = next(iter(values), "") if len(values) == 1 else ""
    return signature


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(ROOT_PATH) / candidate


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() == "true"


def _safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _path_value_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    candidate = str(value).strip()
    if not candidate or candidate.lower() == "nan":
        return None
    return candidate


def _campaign_manifest_records(root_campaign_id: str) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    campaigns_root = Path(ROOT_PATH) / "outputs" / "campaigns"
    if not campaigns_root.exists():
        return records
    for manifest_path in sorted(campaigns_root.glob("*/campaign_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(payload.get("root_campaign_id") or payload.get("campaign_id")) != str(root_campaign_id):
            continue
        records.append((manifest_path, payload))
    records.sort(key=lambda item: (str(item[1].get("created_at") or ""), str(item[1].get("campaign_id") or "")))
    return records


def _read_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing campaign ledger: {path}")
    return pd.read_csv(path)


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_lineage_campaign_records(root_campaign_id: str) -> list[LineageCampaignRecord]:
    records: list[LineageCampaignRecord] = []
    for _, manifest in _campaign_manifest_records(root_campaign_id):
        campaign_id = str(manifest["campaign_id"])
        paths = build_campaign_paths(campaign_id)
        records.append(
            LineageCampaignRecord(
                campaign_id=campaign_id,
                paths=paths,
                manifest=manifest,
                summary=_read_summary(paths.summary_path),
                ledger=_read_ledger(paths.ledger_path),
            )
        )
    return records


def _surface_id_for_row(row: pd.Series) -> str | None:
    explicit_surface = _path_value_or_none(row.get("primary_interpretability_surface_id"))
    if explicit_surface:
        return explicit_surface
    family = str(row["model_family"])
    if family == "xgboost":
        return LINEAGE_SURFACE_SEMANTIC_BRIDGE
    if family != "mlp":
        return None
    if pd.notna(row.get("input_feature_influence_global_path")) and str(row.get("input_feature_influence_global_path")):
        return LINEAGE_SURFACE_SEMANTIC_BRIDGE
    return LINEAGE_SURFACE_SEMANTIC_BRIDGE


def validate_lineage_pool_readiness(root_campaign_id: str) -> dict[str, Any]:
    candidate_records = load_lineage_campaign_records(root_campaign_id)
    blockers: list[str] = []
    if not candidate_records:
        blockers.append("missing_root_campaign_lineage")
        return {
            "root_campaign_id": root_campaign_id,
            "candidate_campaign_ids": [],
            "included_campaign_ids": [],
            "excluded_campaigns": [],
            "lineage_pool_ready": False,
            "lineage_pool_blockers": blockers,
        }

    excluded_campaigns: list[dict[str, Any]] = []
    included_records: list[LineageCampaignRecord] = []
    lineage_ids = {str(record.manifest.get("campaign_lineage_id") or "") for record in candidate_records}
    if len(lineage_ids) != 1:
        blockers.append("campaign_lineage_id_mismatch")

    root_ids = {str(record.manifest.get("root_campaign_id") or "") for record in candidate_records}
    if root_ids != {str(root_campaign_id)}:
        blockers.append("root_campaign_id_mismatch")

    for record in candidate_records:
        status = str(record.manifest.get("campaign_status") or "")
        if status != "closed_success":
            blockers.append(f"campaign_not_closed_success:{record.campaign_id}")
            excluded_campaigns.append({"campaign_id": record.campaign_id, "campaign_status": status})
            continue
        included_records.append(record)

    if not included_records:
        blockers.append("no_closed_success_campaigns_in_lineage")
        return {
            "root_campaign_id": root_campaign_id,
            "candidate_campaign_ids": [record.campaign_id for record in candidate_records],
            "included_campaign_ids": [],
            "excluded_campaigns": excluded_campaigns,
            "lineage_pool_ready": False,
            "lineage_pool_blockers": sorted(set(blockers)),
        }

    manifest_fields = {
        "campaign_scope": {str(record.manifest.get("campaign_scope")) for record in included_records},
    }
    if len(manifest_fields["campaign_scope"]) != 1:
        blockers.append("campaign_scope_mismatch")

    per_campaign_dataset_sets: dict[str, set[str]] = {}
    per_campaign_run_spec_sets: dict[str, set[str]] = {}
    per_campaign_structural_values: dict[str, dict[str, str]] = {}
    per_campaign_expected_replication: dict[str, dict[str, Any]] = {}
    per_campaign_contract_signatures: dict[str, dict[str, str]] = {}
    raw_contract_ids: set[str] = set()
    interpretability_surface_by_family: dict[str, set[str]] = {}
    feature_namespace_by_family: dict[str, set[str]] = {}
    panel_build_versions: set[str] = set()
    lineage_aggregate_build_versions: set[str] = set()
    contract_reference_values: dict[str, set[str]] = {field: set() for field in LINEAGE_CONTRACT_FIELDS}

    for record in included_records:
        missing_columns = sorted(LINEAGE_REQUIRED_LEDGER_COLUMNS - set(record.ledger.columns))
        if missing_columns:
            blockers.append(f"missing_required_ledger_columns:{record.campaign_id}")
            continue
        per_campaign_expected_replication[record.campaign_id] = _manifest_expected_replication(record.manifest)
        per_campaign_contract_signatures[record.campaign_id] = _contract_signature_for_record(record)
        ledger = record.ledger.copy()
        expected_replication = per_campaign_expected_replication[record.campaign_id]
        if not expected_replication:
            blockers.append(f"missing_expected_replication_manifest:{record.campaign_id}")
        else:
            expected_groups = {
                str(item)
                for item in list(expected_replication.get("expected_structural_group_ids") or [])
            }
            ledger_groups = {str(item) for item in ledger["lineage_trial_group_id"].dropna().unique()}
            if expected_groups and expected_groups != ledger_groups:
                blockers.append(f"expected_structural_group_set_mismatch:{record.campaign_id}")
            expected_seed_values = expected_replication.get("expected_seed_values")
            if not isinstance(expected_seed_values, list) or not expected_seed_values:
                blockers.append(f"missing_expected_seed_values:{record.campaign_id}")
            else:
                expected_seed_set = {int(value) for value in expected_seed_values}
                observed_seed_set = {
                    int(value)
                    for value in pd.to_numeric(ledger["seed"], errors="coerce").dropna().astype(int).tolist()
                }
                if expected_seed_set != observed_seed_set:
                    blockers.append(f"expected_seed_values_mismatch:{record.campaign_id}")
        per_campaign_dataset_sets[record.campaign_id] = {str(item) for item in ledger["dataset_candidate_id"].dropna().unique()}
        per_campaign_run_spec_sets[record.campaign_id] = {str(item) for item in ledger["run_spec_id"].dropna().unique()}
        per_campaign_structural_values[record.campaign_id] = {}
        for column in ("run_mode", "allow_test_holdout", "test_enabled"):
            values = {str(item) for item in ledger[column].dropna().unique()} if column in ledger.columns else set()
            if len(values) > 1:
                blockers.append(f"campaign_inconsistent_{column}:{record.campaign_id}")
            per_campaign_structural_values[record.campaign_id][column] = next(iter(values), "")

        valid_rows = ledger[
            ledger["execution_status"].astype(str).eq("completed")
            & ledger["campaign_valid_f7"].map(_coerce_bool)
        ].copy()
        if valid_rows.empty:
            blockers.append(f"no_completed_valid_trials:{record.campaign_id}")
            continue
        if any(str(item) != "ok" for item in valid_rows["raw_metric_contract_validation_status"].fillna("")):
            blockers.append(f"raw_metric_contract_validation_status_not_ok:{record.campaign_id}")
        if not valid_rows["analysis_ready_comparable"].map(_coerce_bool).all():
            blockers.append(f"analysis_ready_comparable_incomplete:{record.campaign_id}")
        raw_contract_ids.update(str(item) for item in valid_rows["raw_metric_contract_id"].dropna().unique())
        for field in LINEAGE_CONTRACT_FIELDS:
            contract_reference_values[field].update(str(item) for item in valid_rows[field].dropna().unique())
        panel_build_versions.update(str(item) for item in valid_rows["panel_build_version"].dropna().unique())
        lineage_aggregate_build_versions.update(
            str(item) for item in valid_rows["lineage_aggregate_build_version"].dropna().unique()
        )
        for family, group in valid_rows.groupby("model_family", dropna=False):
            surface_values = {_surface_id_for_row(row) for _, row in group.iterrows()}
            interpretability_surface_by_family.setdefault(str(family), set()).update(
                value for value in surface_values if value is not None
            )
            namespace_values = {
                str(item)
                for item in group.get("feature_namespace", pd.Series(dtype=object)).dropna().unique()
                if str(item)
            }
            feature_namespace_by_family.setdefault(str(family), set()).update(namespace_values)

    dataset_sets = {frozenset(values) for values in per_campaign_dataset_sets.values()}
    run_spec_sets = {frozenset(values) for values in per_campaign_run_spec_sets.values()}
    structural_signatures = {
        tuple(sorted(structural_values.items()))
        for structural_values in per_campaign_structural_values.values()
    }
    if len(dataset_sets) > 1:
        blockers.append("dataset_candidate_set_mismatch")
    if len(run_spec_sets) > 1:
        blockers.append("run_spec_set_mismatch")
    if len(structural_signatures) > 1:
        blockers.append("structural_runtime_policy_mismatch")
    if len(raw_contract_ids) > 1:
        blockers.append("raw_metric_contract_id_mismatch")
    for field in LINEAGE_CONTRACT_FIELDS:
        manifest_signature_values = {
            str(signature.get(field) or "")
            for signature in per_campaign_contract_signatures.values()
            if str(signature.get(field) or "")
        }
        if len(manifest_signature_values) > 1:
            blockers.append(f"{field}_mismatch")
    for field, values in contract_reference_values.items():
        values = {item for item in values if item}
        if len(values) > 1:
            blockers.append(f"{field}_mismatch")
    for family, values in interpretability_surface_by_family.items():
        if len(values) > 1:
            blockers.append(f"interpretability_surface_primary_mismatch:{family}")
    for family, values in feature_namespace_by_family.items():
        if len(values) > 1:
            blockers.append(f"feature_namespace_mismatch:{family}")
    if len(panel_build_versions - {""}) > 1:
        blockers.append("panel_build_version_mismatch")
    if len(lineage_aggregate_build_versions - {""}) > 1:
        blockers.append("lineage_aggregate_build_version_mismatch")

    seed_sets_by_campaign = {
        campaign_id: _campaign_seed_values(record)
        for campaign_id, record in ((record.campaign_id, record) for record in included_records)
    }
    campaign_ids = sorted(seed_sets_by_campaign)
    for idx, left_campaign_id in enumerate(campaign_ids):
        left = seed_sets_by_campaign[left_campaign_id]
        for right_campaign_id in campaign_ids[idx + 1:]:
            overlap = sorted(left & seed_sets_by_campaign[right_campaign_id])
            if overlap:
                blockers.append(f"seed_overlap_across_campaigns:{left_campaign_id}:{right_campaign_id}")

    expected_seed_counts_by_group: dict[str, int] = {}
    for expected_replication in per_campaign_expected_replication.values():
        by_group = dict(expected_replication.get("expected_seed_count_by_structural_group") or {})
        if not by_group:
            continue
        for group_id, value in by_group.items():
            expected_seed_counts_by_group[str(group_id)] = (
                int(expected_seed_counts_by_group.get(str(group_id), 0)) + int(value)
            )
    if not expected_seed_counts_by_group:
        blockers.append("missing_expected_replication_manifest")

    observed_registry = _valid_trial_registry(included_records)
    observed_seed_counts_by_group: dict[str, int] = {}
    if not observed_registry.empty:
        observed_series = observed_registry.groupby("lineage_trial_group_id")["seed"].nunique(dropna=True)
        observed_seed_counts_by_group = {str(key): int(value) for key, value in observed_series.items()}
    all_groups = sorted(set(expected_seed_counts_by_group) | set(observed_seed_counts_by_group))
    for group_id in all_groups:
        expected = int(expected_seed_counts_by_group.get(group_id, 0))
        observed = int(observed_seed_counts_by_group.get(group_id, 0))
        if expected != observed:
            blockers.append(f"expected_seed_count_mismatch:{group_id}")

    return {
        "root_campaign_id": root_campaign_id,
        "campaign_lineage_id": str(included_records[0].manifest.get("campaign_lineage_id") or ""),
        "candidate_campaign_ids": [record.campaign_id for record in candidate_records],
        "included_campaign_ids": [record.campaign_id for record in included_records],
        "excluded_campaigns": excluded_campaigns,
        "contract_signature": per_campaign_contract_signatures.get(included_records[0].campaign_id, {}),
        "expected_seed_counts_by_group": expected_seed_counts_by_group,
        "observed_seed_counts_by_group": observed_seed_counts_by_group,
        "panel_build_version": next(iter(panel_build_versions), ""),
        "lineage_aggregate_build_version": next(iter(lineage_aggregate_build_versions), ""),
        "lineage_pool_ready": len(set(blockers)) == 0,
        "lineage_pool_blockers": sorted(set(blockers)),
    }


def _valid_trial_registry(records: list[LineageCampaignRecord]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for record in records:
        frame = record.ledger.copy()
        required_columns = {"execution_status", "campaign_valid_f7", "lineage_trial_group_id", "seed"}
        if not required_columns.issubset(set(frame.columns)):
            continue
        frame["campaign_status"] = str(record.manifest.get("campaign_status") or "")
        frame = frame[
            frame["execution_status"].astype(str).eq("completed")
            & frame["campaign_valid_f7"].map(_coerce_bool)
        ].copy()
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    registry = pd.concat(frames, ignore_index=True)
    registry = registry.sort_values(
        ["model_family", "dataset_candidate_id", "run_spec_id", "seed", "campaign_id"],
        kind="stable",
    ).reset_index(drop=True)
    return registry


def _expected_seed_counts(records: list[LineageCampaignRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        expected_replication = _manifest_expected_replication(record.manifest)
        by_group = dict(expected_replication.get("expected_seed_count_by_structural_group") or {})
        for group_id, value in by_group.items():
            counts[str(group_id)] = int(counts.get(str(group_id), 0)) + int(value)
    return counts


def _extract_metric_panel_row(metrics_path: str | Path) -> tuple[dict[str, float | None], list[str]]:
    df = pd.read_csv(_repo_path(metrics_path))
    blockers: list[str] = []
    subset = df[
        df["value_space"].astype(str).eq("raw_real")
        & df["metric_name"].astype(str).isin(LINEAGE_AUX_METRICS)
        & df["metric_scope"].astype(str).eq("macro")
        & df["split"].astype(str).isin(LINEAGE_SPLITS)
    ].copy()
    values: dict[str, float | None] = {}
    for split in LINEAGE_SPLITS:
        split_df = subset[subset["split"].astype(str).eq(split)]
        for metric_name in LINEAGE_AUX_METRICS:
            rows = split_df[split_df["metric_name"].astype(str).eq(metric_name)]
            key = f"{split}_raw_real_macro_{metric_name}"
            if rows.empty:
                blockers.append(f"missing_metric:{split}:{metric_name}")
                values[key] = None
                continue
            values[key] = _safe_float(rows.iloc[0]["metric_value"])
    return values, blockers


def _extract_metric_detail_rows(metrics_path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    df = pd.read_csv(_repo_path(metrics_path))
    blockers: list[str] = []
    required_columns = {
        "split",
        "metric_name",
        "metric_scope",
        "component",
        "class_id",
        "target_name",
        "value_space",
        "metric_value",
        "n_obs",
    }
    missing_columns = sorted(required_columns - set(str(column) for column in df.columns))
    if missing_columns:
        return [], [f"missing_detailed_metric_column:{column}" for column in missing_columns]
    subset = df[
        df["value_space"].astype(str).eq("raw_real")
        & df["metric_scope"].astype(str).isin(LINEAGE_DETAILED_SCOPES)
        & df["split"].astype(str).isin(LINEAGE_SPLITS)
    ].copy()
    if subset.empty:
        return [], ["missing_detailed_metric_surface:raw_real"]
    detail_rows: list[dict[str, Any]] = []
    for row in subset.to_dict("records"):
        detail_rows.append(
            {
                "split": str(row.get("split") or ""),
                "metric_scope": str(row.get("metric_scope") or ""),
                "metric_name": str(row.get("metric_name") or ""),
                "component": str(row.get("component") or ""),
                "class_id": None if row.get("class_id") in ("", None) or pd.isna(row.get("class_id")) else str(row.get("class_id")),
                "target_name": str(row.get("target_name") or ""),
                "value_space": str(row.get("value_space") or ""),
                "metric_value": _safe_float(row.get("metric_value")),
                "n_obs": _safe_float(row.get("n_obs")),
            }
        )
    return detail_rows, blockers


def build_lineage_metric_panel(registry_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for record in registry_df.to_dict("records"):
        metric_values, metric_blockers = _extract_metric_panel_row(record["metrics_long_path"])
        if metric_blockers:
            blockers.extend(
                f"{record['trial_id']}::{item}"
                for item in metric_blockers
            )
        row = {
            "campaign_id": record["campaign_id"],
            "trial_id": record["trial_id"],
            "seed": int(record["seed"]),
            "model_family": record["model_family"],
            "dataset_candidate_id": record["dataset_candidate_id"],
            "run_spec_id": record["run_spec_id"],
            "comparison_group_id": record["comparison_group_id"],
            "lineage_trial_group_id": record["lineage_trial_group_id"],
            "objective_metric_id": record["objective_metric_id"],
            "value_space_default": record.get("value_space_default"),
            "raw_metric_contract_id": record.get("raw_metric_contract_id"),
            "analysis_ready_comparable": record.get("analysis_ready_comparable"),
            "feature_namespace": record.get("feature_namespace"),
            "primary_interpretability_surface_id": record.get("primary_interpretability_surface_id"),
            "variant_fingerprint": record.get("variant_fingerprint"),
        }
        row.update(metric_values)
        rows.append(row)
    return pd.DataFrame(rows), sorted(set(blockers))


def build_lineage_metric_panel_detailed(registry_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for record in registry_df.to_dict("records"):
        detail_rows, detail_blockers = _extract_metric_detail_rows(record["metrics_long_path"])
        if detail_blockers:
            blockers.extend(f"{record['trial_id']}::{item}" for item in detail_blockers)
        for metric_row in detail_rows:
            rows.append(
                {
                    "campaign_id": record["campaign_id"],
                    "trial_id": record["trial_id"],
                    "seed": int(record["seed"]),
                    "model_family": record["model_family"],
                    "dataset_candidate_id": record["dataset_candidate_id"],
                    "run_spec_id": record["run_spec_id"],
                    "comparison_group_id": record["comparison_group_id"],
                    "lineage_trial_group_id": record["lineage_trial_group_id"],
                    "objective_metric_id": record["objective_metric_id"],
                    "value_space_default": record.get("value_space_default"),
                    "raw_metric_contract_id": record.get("raw_metric_contract_id"),
                    "analysis_ready_comparable": record.get("analysis_ready_comparable"),
                    "variant_fingerprint": record.get("variant_fingerprint"),
                    **metric_row,
                }
            )
    return pd.DataFrame(rows), sorted(set(blockers))


def build_lineage_metric_aggregate(
    panel_df: pd.DataFrame,
    *,
    expected_seed_counts: dict[str, int],
) -> pd.DataFrame:
    if panel_df.empty:
        return pd.DataFrame()
    metric_columns = [
        column
        for column in panel_df.columns
        if column.startswith(("train_raw_real_macro_", "val_raw_real_macro_", "test_raw_real_macro_"))
    ]
    rows: list[dict[str, Any]] = []
    for lineage_trial_group_id, group in panel_df.groupby("lineage_trial_group_id", dropna=False):
        row = {
            "lineage_trial_group_id": lineage_trial_group_id,
            "model_family": group["model_family"].iloc[0],
            "dataset_candidate_id": group["dataset_candidate_id"].iloc[0],
            "run_spec_id": group["run_spec_id"].iloc[0],
            "comparison_group_id": group["comparison_group_id"].iloc[0],
            "objective_metric_id": group["objective_metric_id"].iloc[0],
            "observed_seed_count": int(group["seed"].nunique()),
            "expected_seed_count": int(expected_seed_counts.get(str(lineage_trial_group_id), group["seed"].nunique())),
        }
        row["seed_completeness_ratio"] = (
            None
            if row["expected_seed_count"] <= 0
            else round(float(row["observed_seed_count"]) / float(row["expected_seed_count"]), 6)
        )
        for column in metric_columns:
            series = pd.to_numeric(group[column], errors="coerce").dropna()
            prefix = column
            row[f"{prefix}__mean"] = None if series.empty else round(float(series.mean()), 6)
            row[f"{prefix}__std"] = None if series.empty else round(float(series.std(ddof=0)), 6)
            row[f"{prefix}__stderr"] = None if series.empty else round(float(series.std(ddof=0) / math.sqrt(len(series))), 6)
            row[f"{prefix}__min"] = None if series.empty else round(float(series.min()), 6)
            row[f"{prefix}__max"] = None if series.empty else round(float(series.max()), 6)
            row[f"{prefix}__median"] = None if series.empty else round(float(series.median()), 6)
            row[f"{prefix}__p25"] = None if series.empty else round(float(series.quantile(0.25)), 6)
            row[f"{prefix}__p75"] = None if series.empty else round(float(series.quantile(0.75)), 6)
        val_mean = row.get("val_raw_real_macro_rrmse__mean")
        test_mean = row.get("test_raw_real_macro_rrmse__mean")
        row["test_minus_val_mean"] = (
            None
            if val_mean is None or test_mean is None
            else round(float(test_mean) - float(val_mean), 6)
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["model_family", "dataset_candidate_id", "run_spec_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_lineage_metric_aggregate_detailed(
    panel_df: pd.DataFrame,
    *,
    expected_seed_counts: dict[str, int],
) -> pd.DataFrame:
    if panel_df.empty:
        return pd.DataFrame()
    group_cols = [
        "lineage_trial_group_id",
        "model_family",
        "dataset_candidate_id",
        "run_spec_id",
        "comparison_group_id",
        "objective_metric_id",
        "split",
        "metric_scope",
        "metric_name",
        "component",
        "class_id",
        "target_name",
        "value_space",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in panel_df.groupby(group_cols, dropna=False):
        key_values = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,), strict=False))
        metric_values = pd.to_numeric(group["metric_value"], errors="coerce").dropna()
        n_obs_values = pd.to_numeric(group["n_obs"], errors="coerce").dropna()
        lineage_trial_group_id = str(key_values["lineage_trial_group_id"])
        row = {
            **key_values,
            "observed_seed_count": int(group["seed"].nunique()),
            "expected_seed_count": int(expected_seed_counts.get(lineage_trial_group_id, group["seed"].nunique())),
        }
        row["seed_completeness_ratio"] = (
            None
            if row["expected_seed_count"] <= 0
            else round(float(row["observed_seed_count"]) / float(row["expected_seed_count"]), 6)
        )
        row["metric_value__mean"] = None if metric_values.empty else round(float(metric_values.mean()), 6)
        row["metric_value__std"] = None if metric_values.empty else round(float(metric_values.std(ddof=0)), 6)
        row["metric_value__stderr"] = (
            None if metric_values.empty else round(float(metric_values.std(ddof=0) / math.sqrt(len(metric_values))), 6)
        )
        row["metric_value__min"] = None if metric_values.empty else round(float(metric_values.min()), 6)
        row["metric_value__max"] = None if metric_values.empty else round(float(metric_values.max()), 6)
        row["metric_value__median"] = None if metric_values.empty else round(float(metric_values.median()), 6)
        row["metric_value__p25"] = None if metric_values.empty else round(float(metric_values.quantile(0.25)), 6)
        row["metric_value__p75"] = None if metric_values.empty else round(float(metric_values.quantile(0.75)), 6)
        row["n_obs__mean"] = None if n_obs_values.empty else round(float(n_obs_values.mean()), 6)
        row["n_obs__min"] = None if n_obs_values.empty else round(float(n_obs_values.min()), 6)
        row["n_obs__max"] = None if n_obs_values.empty else round(float(n_obs_values.max()), 6)
        row["n_obs__median"] = None if n_obs_values.empty else round(float(n_obs_values.median()), 6)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "model_family",
            "dataset_candidate_id",
            "run_spec_id",
            "split",
            "metric_scope",
            "metric_name",
            "class_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _surface_paths_for_row(row: pd.Series, surface_id: str) -> tuple[str | None, str | None]:
    if surface_id == LINEAGE_SURFACE_SEMANTIC_BRIDGE:
        if str(row["model_family"]) == "xgboost":
            return (
                _path_value_or_none(row.get("xgb_perturbation_feature_influence_global_path")),
                _path_value_or_none(row.get("xgb_perturbation_feature_influence_per_class_path")),
            )
        input_global = _path_value_or_none(row.get("input_feature_influence_global_path"))
        input_per_class = _path_value_or_none(row.get("input_feature_influence_per_class_path"))
        if input_global:
            return (
                input_global,
                input_per_class,
            )
        return (
            _path_value_or_none(row.get("feature_influence_global_path")),
            _path_value_or_none(row.get("feature_influence_per_class_path")),
        )
    if surface_id == LINEAGE_SURFACE_XGB_NATIVE_SHAP:
        if str(row["model_family"]) != "xgboost":
            return (None, None)
        return (
            _path_value_or_none(row.get("xgb_shap_feature_influence_global_path")),
            _path_value_or_none(row.get("xgb_shap_feature_influence_per_class_path")),
        )
    if surface_id == LINEAGE_SURFACE_MLP_FLOWPRE_NATIVE_LATENT:
        if str(row["model_family"]) != "mlp":
            return (None, None)
        return (
            _path_value_or_none(row.get("latent_feature_influence_global_path")),
            _path_value_or_none(row.get("latent_feature_influence_per_class_path")),
        )
    return (None, None)


def _normalize_interpretability_frame(
    *,
    trial_row: pd.Series,
    path: str | Path,
    kind: str,
    surface_id: str,
) -> pd.DataFrame:
    df = pd.read_csv(_repo_path(path))
    renamed = df.copy()
    if "latent_name" in renamed.columns:
        renamed = renamed.rename(columns={"latent_name": "feature_name"})
    if "type" in renamed.columns and "class_id" not in renamed.columns:
        renamed = renamed.rename(columns={"type": "class_id"})
    importance_column = None
    for candidate in ("mean_abs_delta_pred_raw", "mean_abs_shap"):
        if candidate in renamed.columns:
            importance_column = candidate
            break
    if importance_column is None:
        raise ValueError(f"Unsupported interpretability frame: {path}")
    base = pd.DataFrame(
        {
            "surface_id": surface_id,
            "kind": kind,
            "campaign_id": str(trial_row["campaign_id"]),
            "trial_id": str(trial_row["trial_id"]),
            "seed": int(trial_row["seed"]),
            "model_family": str(trial_row["model_family"]),
            "dataset_candidate_id": str(trial_row["dataset_candidate_id"]),
            "run_spec_id": str(trial_row["run_spec_id"]),
            "comparison_group_id": str(trial_row["comparison_group_id"]),
            "lineage_trial_group_id": str(trial_row["lineage_trial_group_id"]),
            "split": renamed.get("split"),
            "class_id": renamed.get("class_id"),
            "feature_name": renamed["feature_name"],
            "importance_mean_abs": pd.to_numeric(renamed[importance_column], errors="coerce"),
            "rank_abs": pd.to_numeric(renamed.get("rank_abs"), errors="coerce"),
            "feature_space_kind": renamed.get("feature_space_kind"),
            "projection_status": renamed.get("projection_status"),
        }
    )
    if kind == "global":
        base["class_id"] = None
    return base


def _aggregate_interpretability_surface(
    registry_df: pd.DataFrame,
    *,
    surface_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    global_frames: list[pd.DataFrame] = []
    per_class_frames: list[pd.DataFrame] = []
    blockers: list[str] = []
    for _, row in registry_df.iterrows():
        global_path, per_class_path = _surface_paths_for_row(row, surface_id)
        if not global_path:
            continue
        try:
            global_frames.append(
                _normalize_interpretability_frame(
                    trial_row=row,
                    path=global_path,
                    kind="global",
                    surface_id=surface_id,
                )
            )
            if per_class_path:
                per_class_frames.append(
                    _normalize_interpretability_frame(
                        trial_row=row,
                        path=per_class_path,
                        kind="per_class",
                        surface_id=surface_id,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"{row['trial_id']}::{surface_id}::{type(exc).__name__}:{exc}")

    def aggregate_frame(frame: pd.DataFrame, *, include_class: bool) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        group_cols = [
            "surface_id",
            "lineage_trial_group_id",
            "model_family",
            "dataset_candidate_id",
            "run_spec_id",
            "comparison_group_id",
            "split",
            "feature_name",
        ]
        if include_class:
            group_cols.append("class_id")
        rows: list[dict[str, Any]] = []
        for keys, group in frame.groupby(group_cols, dropna=False):
            key_values = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,), strict=False))
            importance = pd.to_numeric(group["importance_mean_abs"], errors="coerce").dropna()
            ranks = pd.to_numeric(group["rank_abs"], errors="coerce").dropna()
            observed_seed_count = int(group["seed"].nunique())
            row = {
                **key_values,
                "observed_seed_count": observed_seed_count,
                "mean_importance": None if importance.empty else round(float(importance.mean()), 6),
                "std_importance": None if importance.empty else round(float(importance.std(ddof=0)), 6),
                "stderr_importance": None if importance.empty else round(float(importance.std(ddof=0) / math.sqrt(len(importance))), 6),
                "mean_rank": None if ranks.empty else round(float(ranks.mean()), 6),
                "topk_frequency": None if ranks.empty else round(float((ranks <= 10).mean()), 6),
            }
            rows.append(row)
        return pd.DataFrame(rows).sort_values(
            ["model_family", "dataset_candidate_id", "run_spec_id", "split", "feature_name"],
            kind="stable",
        ).reset_index(drop=True)

    global_df = pd.concat(global_frames, ignore_index=True) if global_frames else pd.DataFrame()
    per_class_df = pd.concat(per_class_frames, ignore_index=True) if per_class_frames else pd.DataFrame()
    stability = build_interpretability_stability_summary(global_df, surface_id=surface_id)
    return aggregate_frame(global_df, include_class=False), aggregate_frame(per_class_df, include_class=True), stability, sorted(set(blockers))


def build_interpretability_stability_summary(global_df: pd.DataFrame, *, surface_id: str) -> dict[str, Any]:
    if global_df.empty:
        return {
            "surface_id": surface_id,
            "group_summaries": [],
            "mean_pairwise_rank_correlation": None,
            "mean_topk_intersection": None,
        }
    group_summaries: list[dict[str, Any]] = []
    pairwise_corrs: list[float] = []
    pairwise_topk: list[float] = []
    focus = global_df[global_df["split"].astype(str).eq("val")].copy()
    if focus.empty:
        focus = global_df.copy()
    for lineage_trial_group_id, group in focus.groupby("lineage_trial_group_id", dropna=False):
        seeds = sorted(group["seed"].dropna().unique())
        seed_feature_tables: dict[int, pd.DataFrame] = {}
        for seed in seeds:
            seed_df = group[group["seed"] == seed][["feature_name", "rank_abs", "importance_mean_abs"]].copy()
            seed_df = seed_df.dropna(subset=["feature_name"]).drop_duplicates(subset=["feature_name"], keep="first")
            seed_feature_tables[int(seed)] = seed_df
        corr_values: list[float] = []
        topk_values: list[float] = []
        for seed_a, seed_b in combinations(seeds, 2):
            left = seed_feature_tables[int(seed_a)].set_index("feature_name")
            right = seed_feature_tables[int(seed_b)].set_index("feature_name")
            features = sorted(set(left.index) | set(right.index))
            if not features:
                continue
            left_ranks = left.reindex(features)["rank_abs"]
            right_ranks = right.reindex(features)["rank_abs"]
            max_left = float(left["rank_abs"].max()) if not left.empty else 0.0
            max_right = float(right["rank_abs"].max()) if not right.empty else 0.0
            left_ranks = left_ranks.fillna(max_left + 1.0)
            right_ranks = right_ranks.fillna(max_right + 1.0)
            corr = left_ranks.corr(right_ranks, method="spearman")
            if corr is not None and not math.isnan(float(corr)):
                corr_values.append(float(corr))
                pairwise_corrs.append(float(corr))
            left_topk = set(left[left["rank_abs"] <= 10].index)
            right_topk = set(right[right["rank_abs"] <= 10].index)
            topk_ratio = float(len(left_topk & right_topk)) / 10.0
            topk_values.append(topk_ratio)
            pairwise_topk.append(topk_ratio)
        stable_features: list[dict[str, Any]] = []
        feature_groups = group.groupby("feature_name", dropna=False)
        for feature_name, feature_group in feature_groups:
            ranks = pd.to_numeric(feature_group["rank_abs"], errors="coerce")
            stable_features.append(
                {
                    "feature_name": feature_name,
                    "topk_frequency": round(float((ranks <= 10).mean()), 6) if not ranks.dropna().empty else None,
                    "mean_rank": round(float(ranks.mean()), 6) if not ranks.dropna().empty else None,
                }
            )
        stable_features = sorted(
            stable_features,
            key=lambda item: (
                -1.0 if item["topk_frequency"] is None else -float(item["topk_frequency"]),
                float("inf") if item["mean_rank"] is None else float(item["mean_rank"]),
                str(item["feature_name"]),
            ),
        )[:10]
        group_summaries.append(
            {
                "lineage_trial_group_id": lineage_trial_group_id,
                "observed_seed_count": len(seeds),
                "mean_pairwise_rank_correlation": None if not corr_values else round(sum(corr_values) / len(corr_values), 6),
                "mean_topk_intersection": None if not topk_values else round(sum(topk_values) / len(topk_values), 6),
                "stable_features": stable_features,
            }
        )
    return {
        "surface_id": surface_id,
        "mean_pairwise_rank_correlation": None if not pairwise_corrs else round(sum(pairwise_corrs) / len(pairwise_corrs), 6),
        "mean_topk_intersection": None if not pairwise_topk else round(sum(pairwise_topk) / len(pairwise_topk), 6),
        "group_summaries": group_summaries,
    }


def lineage_output_dir(root_campaign_id: str) -> Path:
    return build_campaign_paths(root_campaign_id).root / "lineage_aggregate"


def write_lineage_aggregate(root_campaign_id: str) -> dict[str, Any]:
    validation = validate_lineage_pool_readiness(root_campaign_id)
    output_dir = lineage_output_dir(root_campaign_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload: dict[str, Any] = dict(validation)
    summary_payload["output_dir"] = path_relative_to_root(output_dir)

    if not validation["lineage_pool_ready"]:
        summary_payload["outputs"] = {}
        (output_dir / "lineage_summary.json").write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary_payload

    records = [
        record
        for record in load_lineage_campaign_records(root_campaign_id)
        if record.campaign_id in set(validation["included_campaign_ids"])
    ]
    registry_df = _valid_trial_registry(records)
    metric_panel_df, metric_panel_blockers = build_lineage_metric_panel(registry_df)
    metric_panel_detailed_df, metric_panel_detailed_blockers = build_lineage_metric_panel_detailed(registry_df)
    expected_seed_counts = _expected_seed_counts(records)
    metric_aggregate_df = build_lineage_metric_aggregate(metric_panel_df, expected_seed_counts=expected_seed_counts)
    metric_aggregate_detailed_df = build_lineage_metric_aggregate_detailed(
        metric_panel_detailed_df,
        expected_seed_counts=expected_seed_counts,
    )

    interpretability_outputs: dict[str, dict[str, str | None]] = {}
    interpretability_blockers: list[str] = []
    for surface_id in (
        LINEAGE_SURFACE_SEMANTIC_BRIDGE,
        LINEAGE_SURFACE_XGB_NATIVE_SHAP,
        LINEAGE_SURFACE_MLP_FLOWPRE_NATIVE_LATENT,
    ):
        global_df, per_class_df, stability_payload, surface_blockers = _aggregate_interpretability_surface(
            registry_df,
            surface_id=surface_id,
        )
        interpretability_blockers.extend(surface_blockers)
        global_path = output_dir / f"lineage_interpretability_aggregate__{surface_id}__global.csv"
        per_class_path = output_dir / f"lineage_interpretability_aggregate__{surface_id}__per_class.csv"
        stability_path = output_dir / f"lineage_interpretability_stability__{surface_id}.json"
        global_df.to_csv(global_path, index=False)
        per_class_df.to_csv(per_class_path, index=False)
        stability_path.write_text(json.dumps(stability_payload, indent=2, sort_keys=True), encoding="utf-8")
        interpretability_outputs[surface_id] = {
            "global_csv": path_relative_to_root(global_path),
            "per_class_csv": path_relative_to_root(per_class_path),
            "stability_json": path_relative_to_root(stability_path),
        }

    registry_path = output_dir / "lineage_trial_registry.csv"
    metric_panel_path = output_dir / "lineage_metric_panel.csv"
    metric_aggregate_path = output_dir / "lineage_metric_aggregate.csv"
    metric_panel_detailed_path = output_dir / "lineage_metric_panel_detailed.csv"
    metric_aggregate_detailed_path = output_dir / "lineage_metric_aggregate_detailed.csv"
    registry_df.to_csv(registry_path, index=False)
    metric_panel_df.to_csv(metric_panel_path, index=False)
    metric_aggregate_df.to_csv(metric_aggregate_path, index=False)
    metric_panel_detailed_df.to_csv(metric_panel_detailed_path, index=False)
    metric_aggregate_detailed_df.to_csv(metric_aggregate_detailed_path, index=False)

    lineage_pool_blockers = sorted(
        set(
            validation["lineage_pool_blockers"]
            + metric_panel_blockers
            + metric_panel_detailed_blockers
            + interpretability_blockers
        )
    )
    summary_payload.update(
        {
            "lineage_pool_ready": len(lineage_pool_blockers) == 0,
            "lineage_pool_blockers": lineage_pool_blockers,
            "trial_registry_count": int(len(registry_df)),
            "lineage_metric_panel_count": int(len(metric_panel_df)),
            "lineage_metric_aggregate_count": int(len(metric_aggregate_df)),
            "lineage_metric_panel_detailed_count": int(len(metric_panel_detailed_df)),
            "lineage_metric_aggregate_detailed_count": int(len(metric_aggregate_detailed_df)),
            "expected_seed_counts_by_group": expected_seed_counts,
            "observed_seed_counts_by_group": (
                {}
                if registry_df.empty
                else {
                    str(key): int(value)
                    for key, value in registry_df.groupby("lineage_trial_group_id")["seed"].nunique(dropna=True).items()
                }
            ),
            "analysis_ready_comparable_count": (
                0
                if registry_df.empty or "analysis_ready_comparable" not in registry_df.columns
                else int(registry_df["analysis_ready_comparable"].map(_coerce_bool).sum())
            ),
            "analysis_ready_comparable_counts": (
                {}
                if registry_df.empty or "analysis_ready_comparable" not in registry_df.columns
                else {
                    "true": int(registry_df["analysis_ready_comparable"].map(_coerce_bool).sum()),
                    "false": int((~registry_df["analysis_ready_comparable"].map(_coerce_bool)).sum()),
                }
            ),
            "metric_availability_coverage": {
                "metrics_long_path_present_count": 0
                if registry_df.empty
                else int(registry_df["metrics_long_path"].notna().sum()),
            },
            "interpretability_surface_coverage": {
                "primary_interpretability_surface_counts": (
                    {}
                    if registry_df.empty or "primary_interpretability_surface_id" not in registry_df.columns
                    else {
                        str(key): int(value)
                        for key, value in registry_df["primary_interpretability_surface_id"].fillna("").value_counts().items()
                        if str(key)
                    }
                ),
            },
            "outputs": {
                "lineage_trial_registry_csv": path_relative_to_root(registry_path),
                "lineage_metric_panel_csv": path_relative_to_root(metric_panel_path),
                "lineage_metric_aggregate_csv": path_relative_to_root(metric_aggregate_path),
                "lineage_metric_panel_detailed_csv": path_relative_to_root(metric_panel_detailed_path),
                "lineage_metric_aggregate_detailed_csv": path_relative_to_root(metric_aggregate_detailed_path),
                "interpretability": interpretability_outputs,
            },
        }
    )
    (output_dir / "lineage_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_payload
