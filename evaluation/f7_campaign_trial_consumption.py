from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.utils import ROOT_PATH, path_relative_to_root
from evaluation.meta_context import (
    derive_comparison_group_id,
    derive_run_spec_id,
    derive_trial_id,
    resolve_dataset_candidate_id,
    resolve_replication_index,
)


DEFAULT_TRIAL_INVENTORY_PATH = (
    Path(ROOT_PATH) / "outputs" / "reports" / "f7_campaign_spec" / "f7_campaign_trials_v1.csv"
)
DEFAULT_VALIDATION_ROOT = (
    Path(ROOT_PATH) / "outputs" / "reports" / "f7_campaign_spec" / "trial_consumption_validation_v1"
)


@dataclass(frozen=True)
class TrialConsumptionValidationBundle:
    sample_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    output_paths: dict[str, Path]


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(ROOT_PATH) / candidate


def _load_json_dict(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve_repo_path(path).read_text(encoding="utf-8"))


def _load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with _resolve_repo_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, Any], validation_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# F7 Campaign Trial Consumption Validation",
        "",
        f"- trial_inventory_path: `{summary['trial_inventory_path']}`",
        f"- sample_size: `{summary['sample_size']}`",
        f"- ok_count: `{summary['ok_count']}`",
        f"- failed_count: `{summary['failed_count']}`",
        f"- model_family_counts: `{summary['model_family_counts']}`",
        "",
        "## Sample",
        "",
        "| model_family | dataset_candidate_id | run_spec_id | seed | status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in validation_rows:
        lines.append(
            f"| {row['model_family']} | {row['dataset_candidate_id']} | {row['run_spec_id']} | {row['seed']} | {row['status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_trial_inventory(path: str | Path = DEFAULT_TRIAL_INVENTORY_PATH) -> list[dict[str, str]]:
    return _load_csv_rows(path)


def select_structured_trial_sample(
    rows: list[dict[str, str]],
    *,
    sample_size: int = 10,
) -> list[dict[str, str]]:
    mlp_rows = [row for row in rows if row["model_family"] == "mlp"]
    xgb_rows = [row for row in rows if row["model_family"] == "xgboost"]

    selected: list[dict[str, str]] = []

    seen_mlp_run_specs: set[str] = set()
    for row in mlp_rows:
        run_spec_id = row["run_spec_id"]
        if run_spec_id in seen_mlp_run_specs:
            continue
        selected.append(row)
        seen_mlp_run_specs.add(run_spec_id)
        if len(seen_mlp_run_specs) == 6:
            break

    seen_xgb_datasets: set[str] = set()
    for row in xgb_rows:
        dataset_candidate_id = row["dataset_candidate_id"]
        if dataset_candidate_id in seen_xgb_datasets:
            continue
        selected.append(row)
        seen_xgb_datasets.add(dataset_candidate_id)
        if len(seen_xgb_datasets) == 4:
            break

    if len(selected) != sample_size:
        raise ValueError(f"Unable to build structured sample of size {sample_size}; built {len(selected)}")
    return selected


def _mlp_run_level_axes_from_trial(row: dict[str, str]) -> dict[str, Any]:
    return {
        "batch_policy": {
            "id": "balanced" if row["batch_policy_id"] == "imbalance_aware" else "baseline",
            "implemented_now": True,
        },
        "cycling_policy": {
            "cycle_reals": str(row["cycle_reals"]).lower() == "true",
            "implemented_now": True,
        },
        "allow_synth": {
            "enabled": str(row["allow_synth"]).lower() == "true",
            "implemented_now": True,
        },
        "loss_policy": {
            "loss_reduction": row["loss_reduction"],
            "regression_group_metric": row["regression_group_metric"],
            "implemented_now": True,
        },
    }


def _xgb_run_level_axes_from_trial(row: dict[str, str]) -> dict[str, Any]:
    return {
        "objective_metric": {
            "id": row["objective_metric_id"],
            "implemented_now_scope": "post_run_selection",
        }
    }


def build_trial_consumption_payload(row: dict[str, str]) -> dict[str, Any]:
    model_family = row["model_family"]
    base_config_id = row["base_config_id"]
    config_path = Path(ROOT_PATH) / "config" / f"{base_config_id}.yaml"
    manifest_path = _resolve_repo_path(row["dataset_manifest_path"])
    manifest = _load_json_dict(manifest_path) if manifest_path.exists() else {}
    if model_family == "mlp":
        run_level_axes = _mlp_run_level_axes_from_trial(row)
    elif model_family == "xgboost":
        run_level_axes = _xgb_run_level_axes_from_trial(row)
    else:
        raise ValueError(f"Unsupported model family in trial row: {model_family}")
    return {
        "model_family": model_family,
        "dataset_candidate_id": row["dataset_candidate_id"],
        "dataset_manifest_path": path_relative_to_root(manifest_path),
        "dataset_name": manifest.get("dataset_name"),
        "split_id": manifest.get("split_id"),
        "base_config_id": base_config_id,
        "config_path": path_relative_to_root(config_path),
        "seed": int(row["seed"]),
        "seed_set_id": row["seed_set_id"],
        "seed_panel_path": row.get("seed_panel_path"),
        "run_mode": row["run_mode"],
        "allow_test_holdout": str(row["allow_test_holdout"]).lower() == "true",
        "test_enabled": str(row["test_enabled"]).lower() == "true",
        "run_spec_id": row["run_spec_id"],
        "trial_id": row["trial_id"],
        "comparison_group_id": row["comparison_group_id"],
        "run_level_axes": run_level_axes,
        "dataset_level_axes": dict(manifest.get("dataset_level_axes") or {}),
        "contracts": {
            "raw_metric_contract_id": row["raw_metric_contract_id"],
            "artifact_policy_id": row["artifact_policy_id"],
            "family_interpretability_contract_id": row["family_interpretability_contract_id"],
        },
        "interpretability_layers": {
            "native": row["native_interpretability_layer"],
            "bridge": row["bridge_interpretability_layer"],
        },
    }


def validate_trial_consumption_row(row: dict[str, str]) -> dict[str, Any]:
    issues: list[str] = []
    payload = build_trial_consumption_payload(row)
    manifest_path = _resolve_repo_path(payload["dataset_manifest_path"])
    config_path = _resolve_repo_path(payload["config_path"])
    manifest = _load_json_dict(manifest_path) if manifest_path.exists() else {}

    if not manifest_path.exists():
        issues.append("missing_dataset_manifest")
    if not config_path.exists():
        issues.append("missing_base_config")

    for contract_id in payload["contracts"].values():
        candidate = Path(ROOT_PATH) / "config" / f"{contract_id}.yaml"
        if not candidate.exists():
            issues.append(f"missing_contract:{contract_id}")

    resolved_candidate = resolve_dataset_candidate_id(dataset_manifest_path=payload["dataset_manifest_path"])
    if resolved_candidate != payload["dataset_candidate_id"]:
        issues.append("dataset_candidate_id_resolution_mismatch")

    derived_run_spec = derive_run_spec_id(
        model_family=payload["model_family"],
        base_config_id=payload["base_config_id"],
        run_level_axes=payload["run_level_axes"],
    )
    if derived_run_spec != payload["run_spec_id"]:
        issues.append("run_spec_id_mismatch")

    derived_trial_id = derive_trial_id(
        campaign_id=row["campaign_id"],
        dataset_candidate_id=payload["dataset_candidate_id"],
        run_spec_id=payload["run_spec_id"],
        seed=payload["seed"],
    )
    if derived_trial_id != payload["trial_id"]:
        issues.append("trial_id_mismatch")

    derived_group_id = derive_comparison_group_id(
        dataset_candidate_id=payload["dataset_candidate_id"],
        run_spec_id=payload["run_spec_id"],
        contrast_id=row["contrast_id"],
    )
    if derived_group_id != payload["comparison_group_id"]:
        issues.append("comparison_group_id_mismatch")

    replication_index = resolve_replication_index(
        seed=payload["seed"],
        seed_set_id=payload["seed_set_id"],
        seed_panel_path=payload["seed_panel_path"],
    )
    if replication_index != int(row["replication_index"]):
        issues.append("replication_index_mismatch")

    dataset_axes = dict(manifest.get("dataset_level_axes") or {})
    if str(dataset_axes.get("x_transform")) != str(row["x_transform"]):
        issues.append("x_transform_mismatch")
    if str(dataset_axes.get("y_transform")) != str(row["y_transform"]):
        issues.append("y_transform_mismatch")
    if str(dataset_axes.get("synthetic_policy")) != str(row["synthetic_policy"]):
        issues.append("synthetic_policy_mismatch")
    if str(manifest.get("split_id")) != str(row["split_id"]):
        issues.append("split_id_mismatch")

    status = "ok" if not issues else "failed"
    return {
        "status": status,
        "issues": ";".join(issues),
        "model_family": row["model_family"],
        "dataset_candidate_id": row["dataset_candidate_id"],
        "run_spec_id": row["run_spec_id"],
        "trial_id": row["trial_id"],
        "seed": int(row["seed"]),
        "replication_index": int(row["replication_index"]),
        "dataset_manifest_path": payload["dataset_manifest_path"],
        "config_path": payload["config_path"],
        "raw_metric_contract_id": row["raw_metric_contract_id"],
        "artifact_policy_id": row["artifact_policy_id"],
        "family_interpretability_contract_id": row["family_interpretability_contract_id"],
        "native_interpretability_layer": row["native_interpretability_layer"],
        "bridge_interpretability_layer": row["bridge_interpretability_layer"],
        "payload_json": json.dumps(payload, sort_keys=True, ensure_ascii=True),
    }


def validate_trial_inventory_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [validate_trial_consumption_row(row) for row in rows]


def validate_trial_consumption_sample(
    *,
    trial_inventory_path: str | Path = DEFAULT_TRIAL_INVENTORY_PATH,
    output_root: str | Path = DEFAULT_VALIDATION_ROOT,
    sample_size: int = 10,
) -> TrialConsumptionValidationBundle:
    rows = load_trial_inventory(trial_inventory_path)
    sample_rows = select_structured_trial_sample(rows, sample_size=sample_size)
    validation_rows = [validate_trial_consumption_row(row) for row in sample_rows]
    ok_count = sum(1 for row in validation_rows if row["status"] == "ok")
    failed_count = len(validation_rows) - ok_count
    model_family_counts: dict[str, int] = {}
    for row in validation_rows:
        model_family_counts[row["model_family"]] = model_family_counts.get(row["model_family"], 0) + 1

    output_dir = _resolve_repo_path(output_root)
    sample_path = output_dir / "trial_sample.csv"
    validation_path = output_dir / "validation_results.csv"
    report_path = output_dir / "report.md"

    _write_csv_rows(sample_path, sample_rows)
    _write_csv_rows(validation_path, validation_rows)
    summary = {
        "trial_inventory_path": path_relative_to_root(trial_inventory_path),
        "sample_size": sample_size,
        "ok_count": ok_count,
        "failed_count": failed_count,
        "model_family_counts": model_family_counts,
    }
    _write_markdown(report_path, summary, validation_rows)

    return TrialConsumptionValidationBundle(
        sample_rows=sample_rows,
        validation_rows=validation_rows,
        summary=summary,
        output_paths={
            "sample_path": sample_path,
            "validation_path": validation_path,
            "report_path": report_path,
        },
    )
