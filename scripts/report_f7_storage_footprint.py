from __future__ import annotations

import argparse
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

from evaluation.f7_storage_footprint import (  # noqa: E402
    DEFAULT_EXTENSION_SPEC_PATH,
    DEFAULT_PRIMARY_SPEC_PATH,
    build_storage_footprint_report,
)


def _fmt_size(payload: dict[str, Any] | None) -> str:
    if not payload or payload.get("bytes") is None:
        return "n/a"
    return f"{payload['bytes']} B | {payload['mb']} MB | {payload['gib']} GiB"


def write_markdown_report(output_path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    observed = dict(report.get("observed") or {})
    projections = dict(report.get("projections") or {})
    lines.append(f"# F7 Storage Footprint: {report['root_campaign_id']}")
    lines.append("")
    lines.append("## Observed")
    lines.append("")
    lines.append(f"- run_count: `{observed.get('run_count')}`")
    lines.append(f"- total_size: `{_fmt_size(observed.get('total_size'))}`")
    lines.append(f"- mean_size: `{_fmt_size(observed.get('mean_size'))}`")
    lines.append(f"- median_size: `{_fmt_size(observed.get('median_size'))}`")
    lines.append(f"- p90_size: `{_fmt_size(observed.get('p90_size'))}`")
    lines.append(f"- p95_size: `{_fmt_size(observed.get('p95_size'))}`")
    lines.append("")

    def _append_group_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("No rows.")
            lines.append("")
            return
        header = " | ".join(columns + ["run_count", "mean_size", "total_size"])
        lines.append(f"| {header} |")
        lines.append("| " + " | ".join(["---"] * (len(columns) + 3)) + " |")
        for row in rows:
            values = [str(row.get(column)) for column in columns]
            values.extend(
                [
                    str(row.get("run_count")),
                    _fmt_size(row.get("mean_size")),
                    _fmt_size(row.get("total_size")),
                ]
            )
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    _append_group_table("By Campaign", list(observed.get("by_campaign_id") or []), ["campaign_id"])
    _append_group_table("By Family", list(observed.get("by_model_family") or []), ["model_family"])
    _append_group_table("By FlowPre Usage", list(observed.get("by_flowpre_usage") or []), ["flowpre_usage"])
    _append_group_table(
        "By Family And FlowPre Usage",
        list(observed.get("by_model_family_and_flowpre_usage") or []),
        ["model_family", "flowpre_usage"],
    )

    for key in ("primary_17400", "extension_17400", "full_4x17400"):
        payload = dict(projections.get(key) or {})
        lines.append(f"## Projection: {key}")
        lines.append("")
        lines.append(f"- expected_trial_count: `{payload.get('expected_trial_count')}`")
        lines.append(f"- projected_total_size: `{_fmt_size(payload.get('projected_total_size'))}`")
        lines.append(f"- fallbacks: `{payload.get('fallbacks')}`")
        lines.append("")
        rows = list(payload.get("group_rows") or [])
        if rows:
            lines.append("| model_family | flowpre_usage | expected_trial_count | mean_size | projected_total_size |")
            lines.append("| --- | --- | --- | --- | --- |")
            for row in rows:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.get("model_family")),
                            str(row.get("flowpre_usage")),
                            str(row.get("expected_trial_count")),
                            _fmt_size(row.get("mean_size")),
                            _fmt_size(row.get("projected_total_size")),
                        ]
                    )
                    + " |"
                )
            lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report observed and projected F7 campaign storage footprint.")
    parser.add_argument("--root-campaign-id", type=str, required=True)
    parser.add_argument("--primary-spec-path", type=Path, default=DEFAULT_PRIMARY_SPEC_PATH)
    parser.add_argument("--extension-spec-path", type=Path, default=DEFAULT_EXTENSION_SPEC_PATH)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = build_storage_footprint_report(
        root_campaign_id=args.root_campaign_id,
        primary_spec_path=args.primary_spec_path,
        extension_spec_path=args.extension_spec_path,
    )
    output_dir = REPO_ROOT / "outputs" / "reports" / "f7_storage_footprint"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (output_dir / f"{args.root_campaign_id}.json")
    output_md = args.output_md or (output_dir / f"{args.root_campaign_id}.md")
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(output_md, report)
    print(f"Storage footprint JSON written to: {output_json}")
    print(f"Storage footprint Markdown written to: {output_md}")


if __name__ == "__main__":
    main()
