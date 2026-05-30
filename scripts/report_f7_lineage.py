from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.f7_campaign_lineage import write_lineage_aggregate  # noqa: E402
from evaluation.f7_campaign_state import build_campaign_paths  # noqa: E402


def _read_csv_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if not candidate.exists():
        return []
    with candidate.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_report(root_campaign_id: str) -> dict[str, Any]:
    lineage_summary = write_lineage_aggregate(root_campaign_id)
    outputs = dict(lineage_summary.get("outputs") or {})
    trial_registry = _read_csv_rows(outputs.get("lineage_trial_registry_csv"))
    metric_aggregate = _read_csv_rows(outputs.get("lineage_metric_aggregate_csv"))
    campaign_summaries: dict[str, dict[str, Any]] = {}
    for campaign_id in list(lineage_summary.get("included_campaign_ids") or []):
        summary_path = build_campaign_paths(campaign_id).summary_path
        if summary_path.exists():
            campaign_summaries[campaign_id] = json.loads(summary_path.read_text(encoding="utf-8"))

    runtime_by_campaign: list[dict[str, Any]] = []
    runtime_by_family: dict[str, dict[str, float]] = {}
    coverage_by_group: list[dict[str, Any]] = []
    top_groups_val: list[dict[str, Any]] = []
    top_groups_test: list[dict[str, Any]] = []
    warning_policy_counts: dict[str, int] = {}

    for campaign_id, payload in campaign_summaries.items():
        aggregate_runtime = dict(payload.get("aggregate_runtime") or {})
        runtime_by_campaign.append(
            {
                "campaign_id": campaign_id,
                "training_runtime_s_sum": aggregate_runtime.get("training_runtime_s_sum"),
                "interpretability_runtime_s_sum": aggregate_runtime.get("interpretability_runtime_s_sum"),
                "total_runtime_s_sum": aggregate_runtime.get("total_runtime_s_sum"),
            }
        )
        for key, count in dict(payload.get("warning_policy_counts") or {}).items():
            warning_policy_counts[str(key)] = warning_policy_counts.get(str(key), 0) + int(count)

    for row in trial_registry:
        family = str(row.get("model_family"))
        bucket = runtime_by_family.setdefault(
            family,
            {"training_runtime_s_sum": 0.0, "interpretability_runtime_s_sum": 0.0, "total_runtime_s_sum": 0.0, "n_trials": 0},
        )
        for field in ("training_runtime_s", "interpretability_runtime_s", "total_runtime_s"):
            try:
                bucket[f"{field}_sum"] += float(row.get(field) or 0.0)
            except Exception:
                pass
        bucket["n_trials"] += 1

    for row in metric_aggregate:
        coverage_by_group.append(
            {
                "lineage_trial_group_id": row.get("lineage_trial_group_id"),
                "observed_seed_count": row.get("observed_seed_count"),
                "expected_seed_count": row.get("expected_seed_count"),
                "seed_completeness_ratio": row.get("seed_completeness_ratio"),
            }
        )

    def _sorted_top(metric_key: str) -> list[dict[str, Any]]:
        eligible = [row for row in metric_aggregate if row.get(metric_key) not in (None, "", "nan")]
        eligible.sort(key=lambda row: float(row[metric_key]))
        return eligible[:10]

    top_groups_val = _sorted_top("val_raw_real_macro_rrmse__mean")
    top_groups_test = _sorted_top("test_raw_real_macro_rrmse__mean")

    return {
        "root_campaign_id": root_campaign_id,
        "campaign_ids_included": lineage_summary.get("included_campaign_ids"),
        "candidate_campaign_ids": lineage_summary.get("candidate_campaign_ids"),
        "excluded_campaigns": lineage_summary.get("excluded_campaigns"),
        "lineage_pool_ready": lineage_summary.get("lineage_pool_ready"),
        "lineage_pool_blockers": lineage_summary.get("lineage_pool_blockers"),
        "trial_registry_count": lineage_summary.get("trial_registry_count"),
        "lineage_metric_panel_count": lineage_summary.get("lineage_metric_panel_count"),
        "lineage_metric_aggregate_count": lineage_summary.get("lineage_metric_aggregate_count"),
        "lineage_metric_panel_detailed_count": lineage_summary.get("lineage_metric_panel_detailed_count"),
        "lineage_metric_aggregate_detailed_count": lineage_summary.get("lineage_metric_aggregate_detailed_count"),
        "expected_seed_counts_by_group": lineage_summary.get("expected_seed_counts_by_group"),
        "observed_seed_counts_by_group": lineage_summary.get("observed_seed_counts_by_group"),
        "panel_build_version": lineage_summary.get("panel_build_version"),
        "lineage_aggregate_build_version": lineage_summary.get("lineage_aggregate_build_version"),
        "coverage_by_lineage_trial_group": coverage_by_group,
        "runtime_by_campaign": runtime_by_campaign,
        "runtime_by_family": runtime_by_family,
        "warning_policy_counts": warning_policy_counts,
        "top_groups_by_val_objective": top_groups_val,
        "top_groups_by_test_objective": top_groups_test,
        "outputs": lineage_summary.get("outputs"),
        "lineage_summary": lineage_summary,
    }


def write_markdown_report(output_path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# {report['root_campaign_id']}")
    lines.append("")
    lines.append("## Lineage")
    lines.append("")
    lines.append(f"- lineage_pool_ready: `{report.get('lineage_pool_ready')}`")
    lines.append(f"- lineage_pool_blockers: `{report.get('lineage_pool_blockers')}`")
    lines.append(f"- candidate_campaign_ids: `{report.get('candidate_campaign_ids')}`")
    lines.append(f"- campaign_ids_included: `{report.get('campaign_ids_included')}`")
    lines.append(f"- excluded_campaigns: `{report.get('excluded_campaigns')}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- trial_registry_count: `{report.get('trial_registry_count')}`")
    lines.append(f"- lineage_metric_panel_count: `{report.get('lineage_metric_panel_count')}`")
    lines.append(f"- lineage_metric_aggregate_count: `{report.get('lineage_metric_aggregate_count')}`")
    lines.append(f"- lineage_metric_panel_detailed_count: `{report.get('lineage_metric_panel_detailed_count')}`")
    lines.append(f"- lineage_metric_aggregate_detailed_count: `{report.get('lineage_metric_aggregate_detailed_count')}`")
    lines.append(f"- expected_seed_counts_by_group: `{report.get('expected_seed_counts_by_group')}`")
    lines.append(f"- observed_seed_counts_by_group: `{report.get('observed_seed_counts_by_group')}`")
    lines.append(f"- panel_build_version: `{report.get('panel_build_version')}`")
    lines.append(f"- lineage_aggregate_build_version: `{report.get('lineage_aggregate_build_version')}`")
    lines.append("")
    lines.append("## Runtime And Warnings")
    lines.append("")
    lines.append(f"- runtime_by_campaign: `{report.get('runtime_by_campaign')}`")
    lines.append(f"- runtime_by_family: `{report.get('runtime_by_family')}`")
    lines.append(f"- warning_policy_counts: `{report.get('warning_policy_counts')}`")
    lines.append("")
    lines.append("## Top Groups")
    lines.append("")
    lines.append(f"- top_groups_by_val_objective: `{report.get('top_groups_by_val_objective')}`")
    lines.append(f"- top_groups_by_test_objective: `{report.get('top_groups_by_test_objective')}`")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for key, value in dict(report.get("outputs") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build canonical lineage aggregation/reporting for an F7 campaign root.")
    parser.add_argument("--root-campaign-id", type=str, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = build_report(args.root_campaign_id)
    output_root = build_campaign_paths(args.root_campaign_id).root / "lineage_aggregate"
    output_root.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (output_root / "lineage_report.json")
    output_md = args.output_md or (output_root / "lineage_report.md")
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(output_md, report)
    print(f"Lineage report JSON written to: {output_json}")
    print(f"Lineage report Markdown written to: {output_md}")


if __name__ == "__main__":
    main()
