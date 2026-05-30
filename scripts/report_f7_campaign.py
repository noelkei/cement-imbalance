from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.f7_campaign_state import build_campaign_paths


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _build_runtime_summary(df: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty or group_col not in df.columns:
        return rows
    for key, group in df.groupby(group_col, dropna=False):
        total = _safe_numeric(group["total_runtime_s"]) if "total_runtime_s" in group.columns else pd.Series(dtype=float)
        train = _safe_numeric(group["training_runtime_s"]) if "training_runtime_s" in group.columns else pd.Series(dtype=float)
        interp = _safe_numeric(group["interpretability_runtime_s"]) if "interpretability_runtime_s" in group.columns else pd.Series(dtype=float)
        rows.append(
            {
                group_col: key,
                "n_trials": int(len(group)),
                "completed": int((group["execution_status"] == "completed").sum()),
                "failed": int((group["execution_status"] == "failed").sum()),
                "blocked": int((group["execution_status"] == "blocked").sum()),
                "training_runtime_s_mean": None if train.dropna().empty else round(float(train.mean()), 6),
                "interpretability_runtime_s_mean": None if interp.dropna().empty else round(float(interp.mean()), 6),
                "total_runtime_s_mean": None if total.dropna().empty else round(float(total.mean()), 6),
                "total_runtime_s_p95": None if total.dropna().empty else round(float(total.quantile(0.95)), 6),
            }
        )
    return rows


def build_report(campaign_id: str) -> dict[str, Any]:
    paths = build_campaign_paths(campaign_id)
    ledger_path = paths.ledger_path
    summary_path = paths.summary_path
    closeout_path = paths.campaign_closeout_path
    manifest_path = paths.campaign_manifest_path

    if not ledger_path.exists():
        raise FileNotFoundError(f"Campaign ledger not found: {ledger_path}")

    ledger = pd.read_csv(ledger_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    closeout = json.loads(closeout_path.read_text(encoding="utf-8")) if closeout_path.exists() else {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    by_family = _build_runtime_summary(ledger, "model_family")
    by_run_spec = _build_runtime_summary(ledger, "run_spec_id")
    by_dataset = _build_runtime_summary(ledger, "dataset_candidate_id")

    failure_rows: list[dict[str, Any]] = []
    if "failure_reason_code" in ledger.columns:
        failed = ledger[ledger["failure_reason_code"].notna()].copy()
        if not failed.empty:
            counts = failed.groupby("failure_reason_code").size().reset_index(name="count")
            failure_rows = counts.sort_values(["count", "failure_reason_code"], ascending=[False, True]).to_dict("records")

    report = {
        "campaign_id": campaign_id,
        "campaign_manifest": manifest,
        "summary": summary,
        "closeout": closeout,
        "analysis_contracts": manifest.get("analysis_contracts"),
        "expected_replication": manifest.get("expected_replication"),
        "lineage_pool_ready": summary.get("lineage_pool_ready"),
        "lineage_pool_blockers": summary.get("lineage_pool_blockers"),
        "warning_policy_counts": summary.get("warning_policy_counts"),
        "warning_signature_counts": summary.get("warning_signature_counts"),
        "raw_metric_contract_validation_status_counts": summary.get("raw_metric_contract_validation_status_counts"),
        "value_space_default_counts": summary.get("value_space_default_counts"),
        "raw_inversion_status_counts": summary.get("raw_inversion_status_counts"),
        "runtime_by_family": by_family,
        "runtime_by_run_spec": by_run_spec,
        "runtime_by_dataset_candidate": by_dataset,
        "failure_reason_counts_from_ledger": failure_rows,
        "artifact_paths": {
            "campaign_manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
            "campaign_closeout_path": str(closeout_path) if closeout_path.exists() else None,
            "trial_ledger_path": str(ledger_path),
        },
    }
    return report


def write_markdown_report(output_path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    summary = dict(report.get("summary") or {})
    manifest = dict(report.get("campaign_manifest") or {})
    lines.append(f"# {report['campaign_id']}")
    lines.append("")
    lines.append("## Campaign")
    lines.append("")
    lines.append(f"- campaign_status: `{manifest.get('campaign_status')}`")
    lines.append(f"- campaign_kind: `{manifest.get('campaign_kind')}`")
    lines.append(f"- seed_set_id: `{manifest.get('seed_set_id')}`")
    analysis_contracts = dict(report.get("analysis_contracts") or {})
    expected_replication = dict(report.get("expected_replication") or {})
    lines.append(f"- panel_build_version: `{analysis_contracts.get('panel_build_version')}`")
    lines.append(f"- panel_build_timestamp: `{analysis_contracts.get('panel_build_timestamp')}`")
    lines.append(f"- trial_count_total: `{summary.get('trial_count_total')}`")
    lines.append(f"- completed_valid_f7_count: `{summary.get('completed_valid_f7_count')}`")
    lines.append(f"- counts_by_status: `{summary.get('counts_by_status')}`")
    lines.append(f"- failure_reason_counts: `{summary.get('failure_reason_counts')}`")
    lines.append(f"- lineage_pool_ready: `{report.get('lineage_pool_ready')}`")
    lines.append(f"- lineage_pool_blockers: `{report.get('lineage_pool_blockers')}`")
    lines.append("")
    lines.append("## Contracts And Warnings")
    lines.append("")
    lines.append(f"- raw_metric_contract_validation_status_counts: `{report.get('raw_metric_contract_validation_status_counts')}`")
    lines.append(f"- value_space_default_counts: `{report.get('value_space_default_counts')}`")
    lines.append(f"- raw_inversion_status_counts: `{report.get('raw_inversion_status_counts')}`")
    lines.append(f"- warning_policy_counts: `{report.get('warning_policy_counts')}`")
    lines.append(f"- warning_signature_counts: `{report.get('warning_signature_counts')}`")
    lines.append("")
    lines.append("## Replication")
    lines.append("")
    lines.append(f"- expected_seed_values: `{expected_replication.get('expected_seed_values')}`")
    lines.append(f"- expected_seed_count: `{expected_replication.get('expected_seed_count')}`")
    lines.append(f"- expected_structural_group_count: `{expected_replication.get('expected_structural_group_count')}`")
    lines.append("")

    def _append_table(title: str, rows: list[dict[str, Any]], key_col: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("No rows.")
            lines.append("")
            return
        lines.append(f"| {key_col} | n_trials | completed | failed | blocked | train_mean_s | interp_mean_s | total_mean_s | total_p95_s |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row[key_col]} | {row['n_trials']} | {row['completed']} | {row['failed']} | {row['blocked']} | "
                f"{row['training_runtime_s_mean']} | {row['interpretability_runtime_s_mean']} | "
                f"{row['total_runtime_s_mean']} | {row['total_runtime_s_p95']} |"
            )
        lines.append("")

    _append_table("Runtime By Family", list(report.get("runtime_by_family") or []), "model_family")
    _append_table("Runtime By Run Spec", list(report.get("runtime_by_run_spec") or []), "run_spec_id")
    _append_table("Runtime By Dataset Candidate", list(report.get("runtime_by_dataset_candidate") or []), "dataset_candidate_id")

    lines.append("## Artifacts")
    lines.append("")
    for key, value in dict(report.get("artifact_paths") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a post-run report for an F7 campaign.")
    parser.add_argument("--campaign-id", type=str, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = build_report(args.campaign_id)
    paths = build_campaign_paths(args.campaign_id)
    output_json = args.output_json or (paths.root / "campaign_report.json")
    output_md = args.output_md or (paths.root / "campaign_report.md")
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(output_md, report)
    print(f"Campaign report JSON written to: {output_json}")
    print(f"Campaign report Markdown written to: {output_md}")


if __name__ == "__main__":
    main()
