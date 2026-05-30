from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.f7_campaign_runner import (  # noqa: E402
    close_campaign,
    maybe_chain_campaigns,
    rebuild_campaign_state,
    rerun_failed_campaign,
    resume_campaign,
    run_campaign,
    run_preflight,
)
from evaluation.f7_campaign_spec import DEFAULT_SPEC_PATH  # noqa: E402


def _add_common_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-family", type=str, default=None)
    parser.add_argument("--dataset-candidate-id", type=str, default=None)
    parser.add_argument("--run-spec-id", type=str, default=None)
    parser.add_argument("--trial-id", type=str, default=None)
    parser.add_argument("--trial-id-file", type=Path, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--status-filter", type=str, default=None)


def _add_runner_runtime_controls(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cleanup-every",
        type=int,
        default=1,
        help="Run lightweight memory cleanup every N executed trials. Default: 1.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical runner for F7 campaigns.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Initialize/validate a campaign without launching trials.")
    preflight.add_argument("--campaign-spec", type=Path, default=DEFAULT_SPEC_PATH)
    _add_common_filters(preflight)

    run = subparsers.add_parser("run", help="Initialize a campaign if needed and run selected trials.")
    run.add_argument("--campaign-spec", type=Path, default=DEFAULT_SPEC_PATH)
    run.add_argument("--device", type=str, default="cpu")
    run.add_argument("--runner-id", type=str, default=None)
    run.add_argument("--chain-next-spec", type=Path, action="append", default=[])
    run.add_argument("--allow-chain-on-closed-with-failures", action="store_true")
    _add_common_filters(run)
    _add_runner_runtime_controls(run)

    resume = subparsers.add_parser("resume", help="Resume a previously initialized campaign.")
    resume.add_argument("--campaign-id", type=str, required=True)
    resume.add_argument("--device", type=str, default="cpu")
    resume.add_argument("--runner-id", type=str, default=None)
    resume.add_argument("--chain-next-spec", type=Path, action="append", default=[])
    resume.add_argument("--allow-chain-on-closed-with-failures", action="store_true")
    _add_common_filters(resume)
    _add_runner_runtime_controls(resume)

    rerun_failed = subparsers.add_parser("rerun-failed", help="Rerun failed or blocked trials.")
    rerun_failed.add_argument("--campaign-id", type=str, required=True)
    rerun_failed.add_argument("--device", type=str, default="cpu")
    rerun_failed.add_argument("--runner-id", type=str, default=None)
    rerun_failed.add_argument("--reason-code-filter", type=str, default=None)
    rerun_failed.add_argument("--chain-next-spec", type=Path, action="append", default=[])
    rerun_failed.add_argument("--allow-chain-on-closed-with-failures", action="store_true")
    _add_common_filters(rerun_failed)
    _add_runner_runtime_controls(rerun_failed)

    close = subparsers.add_parser("close", help="Compute final status and write campaign_closeout.json.")
    close.add_argument("--campaign-id", type=str, required=True)

    rebuild = subparsers.add_parser("rebuild-state", help="Rebuild campaign state explicitly from run manifests.")
    rebuild.add_argument("--campaign-id", type=str, required=True)
    rebuild.add_argument("--from-run-manifests", action="store_true", default=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preflight":
        result = run_preflight(
            spec_path=args.campaign_spec,
            model_family=args.model_family,
            dataset_candidate_id=args.dataset_candidate_id,
            run_spec_id=args.run_spec_id,
            trial_id=args.trial_id,
            trial_id_file=args.trial_id_file,
            offset=args.offset,
            limit=args.limit,
            status_filter=args.status_filter,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "run":
        result = run_campaign(
            spec_path=args.campaign_spec,
            device=args.device,
            runner_id=args.runner_id,
            model_family=args.model_family,
            dataset_candidate_id=args.dataset_candidate_id,
            run_spec_id=args.run_spec_id,
            trial_id=args.trial_id,
            trial_id_file=args.trial_id_file,
            offset=args.offset,
            limit=args.limit,
            status_filter=args.status_filter,
            cleanup_every=args.cleanup_every,
        )
        chained = maybe_chain_campaigns(
            current_campaign_id=result["campaign_id"],
            next_spec_paths=args.chain_next_spec,
            allow_chain_on_closed_with_failures=args.allow_chain_on_closed_with_failures,
            device=args.device,
            runner_id=args.runner_id,
        )
        print(json.dumps({"result": result, "chained": chained}, indent=2, sort_keys=True))
        return

    if args.command == "resume":
        result = resume_campaign(
            campaign_id=args.campaign_id,
            device=args.device,
            runner_id=args.runner_id,
            model_family=args.model_family,
            dataset_candidate_id=args.dataset_candidate_id,
            run_spec_id=args.run_spec_id,
            trial_id=args.trial_id,
            trial_id_file=args.trial_id_file,
            offset=args.offset,
            limit=args.limit,
            status_filter=args.status_filter,
            cleanup_every=args.cleanup_every,
        )
        chained = maybe_chain_campaigns(
            current_campaign_id=result["campaign_id"],
            next_spec_paths=args.chain_next_spec,
            allow_chain_on_closed_with_failures=args.allow_chain_on_closed_with_failures,
            device=args.device,
            runner_id=args.runner_id,
        )
        print(json.dumps({"result": result, "chained": chained}, indent=2, sort_keys=True))
        return

    if args.command == "rerun-failed":
        result = rerun_failed_campaign(
            campaign_id=args.campaign_id,
            device=args.device,
            runner_id=args.runner_id,
            reason_code_filter=args.reason_code_filter,
            model_family=args.model_family,
            dataset_candidate_id=args.dataset_candidate_id,
            run_spec_id=args.run_spec_id,
            trial_id=args.trial_id,
            trial_id_file=args.trial_id_file,
            offset=args.offset,
            limit=args.limit,
            cleanup_every=args.cleanup_every,
        )
        chained = maybe_chain_campaigns(
            current_campaign_id=result["campaign_id"],
            next_spec_paths=args.chain_next_spec,
            allow_chain_on_closed_with_failures=args.allow_chain_on_closed_with_failures,
            device=args.device,
            runner_id=args.runner_id,
        )
        print(json.dumps({"result": result, "chained": chained}, indent=2, sort_keys=True))
        return

    if args.command == "close":
        result = close_campaign(campaign_id=args.campaign_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "rebuild-state":
        result = rebuild_campaign_state(campaign_id=args.campaign_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
