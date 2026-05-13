#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs" / "models" / "official" / "flowgen"
POSTHOC_SCRIPT = ROOT / "scripts" / "f6_flowgen_temporal_posthoc.py"
EXCLUDED_DIR_NAMES = {"bases", "campaign_summaries", "shortlist"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan official FlowGen runs and backfill val.temporal_realism only where it is missing. "
            "Execution is strictly sequential and delegates the actual update to "
            "scripts/f6_flowgen_temporal_posthoc.py."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="FlowGen root directory to scan.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only scan and report what would be updated; do not run backfills.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the post-hoc updater even when val.temporal_realism already exists.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-run logging.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device forwarded to scripts/f6_flowgen_temporal_posthoc.py.",
    )
    parser.add_argument(
        "--condition-col",
        type=str,
        default="type",
        help="Condition column forwarded to scripts/f6_flowgen_temporal_posthoc.py.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _find_results_path(run_dir: Path) -> Path:
    preferred = run_dir / "results.yaml"
    if preferred.exists():
        return preferred
    matches = sorted(run_dir.glob("*_results.yaml"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate results.yaml under {run_dir}")


def _has_temporal_realism(results_path: Path) -> bool:
    payload = _load_yaml(results_path)
    val = payload.get("val")
    if not isinstance(val, dict):
        return False
    temporal = val.get("temporal_realism")
    return isinstance(temporal, dict)


def _iter_subdirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"FlowGen root does not exist: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir())


def _run_posthoc(run_dir: Path, *, device: str, condition_col: str, quiet: bool) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(POSTHOC_SCRIPT),
        str(run_dir),
        "--device",
        str(device),
        "--condition-col",
        str(condition_col),
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    output_parts = []
    if completed.stdout:
        output_parts.append(completed.stdout.strip())
    if completed.stderr:
        output_parts.append(completed.stderr.strip())
    combined_output = "\n".join(part for part in output_parts if part)
    if (not quiet) and combined_output:
        print(combined_output)
    return completed.returncode == 0, combined_output


def _print(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(msg)


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()

    if not POSTHOC_SCRIPT.exists():
        raise FileNotFoundError(f"Required post-hoc script does not exist: {POSTHOC_SCRIPT}")

    subdirs = _iter_subdirs(root)
    excluded = []
    candidates = []
    for run_dir in subdirs:
        if run_dir.name in EXCLUDED_DIR_NAMES:
            excluded.append(run_dir)
            continue
        if not run_dir.name.startswith("flowgen"):
            excluded.append(run_dir)
            continue
        candidates.append(run_dir)

    total_detected = len(candidates)
    total_excluded = len(excluded)
    total_already_updated = 0
    total_pending_found = 0
    total_updated_now = 0
    total_failed = 0

    runs_to_execute: list[tuple[Path, Path, bool]] = []

    for idx, run_dir in enumerate(candidates, start=1):
        prefix = f"[{idx}/{total_detected}]"
        _print(f"{prefix} checking {run_dir.name}", quiet=args.quiet)
        try:
            results_path = _find_results_path(run_dir)
            has_temporal = _has_temporal_realism(results_path)
            if has_temporal:
                total_already_updated += 1
                if args.force:
                    runs_to_execute.append((run_dir, results_path, True))
                    _print(f"{prefix} already updated, but will re-run because --force", quiet=args.quiet)
                else:
                    _print(f"{prefix} already updated", quiet=args.quiet)
            else:
                total_pending_found += 1
                runs_to_execute.append((run_dir, results_path, False))
                action = "would backfill" if args.summary_only else "pending backfill"
                _print(f"{prefix} {action}", quiet=args.quiet)
        except Exception as exc:
            total_failed += 1
            print(f"{prefix} failed during scan: {exc}")

    if args.summary_only:
        print("")
        print("Summary")
        print(f"- total runs detected: {total_detected}")
        print(f"- total excluded: {total_excluded}")
        print(f"- total already updated: {total_already_updated}")
        print(f"- total pending found: {total_pending_found}")
        print(f"- total updated in this execution: {total_updated_now}")
        print(f"- total failed: {total_failed}")
        return 0 if total_failed == 0 else 1

    for exec_idx, (run_dir, _results_path, forced) in enumerate(runs_to_execute, start=1):
        prefix = f"[{exec_idx}/{len(runs_to_execute)}]"
        mode = "re-running" if forced else "backfilling"
        _print(f"{prefix} {mode} {run_dir.name}", quiet=args.quiet)
        ok, combined_output = _run_posthoc(
            run_dir,
            device=args.device,
            condition_col=args.condition_col,
            quiet=args.quiet,
        )
        if ok:
            total_updated_now += 1
            _print(f"{prefix} done", quiet=args.quiet)
        else:
            total_failed += 1
            print(f"{prefix} failed: post-hoc updater returned non-zero exit status")
            if args.quiet and combined_output:
                print(combined_output)

    print("")
    print("Summary")
    print(f"- total runs detected: {total_detected}")
    print(f"- total excluded: {total_excluded}")
    print(f"- total already updated: {total_already_updated}")
    print(f"- total pending found: {total_pending_found}")
    print(f"- total updated in this execution: {total_updated_now}")
    print(f"- total failed: {total_failed}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
