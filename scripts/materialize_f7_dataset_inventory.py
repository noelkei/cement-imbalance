from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.f7_dataset_materialization import (
    load_f7_materialization_inventory,
    materialize_f7_inventory_phase,
    validate_f7_materialization_prerequisites,
    write_f7_materialization_batch_report,
)


def _batch_id_from_now() -> str:
    return datetime.now(timezone.utc).strftime("f7_dataset_materialization_%Y%m%dT%H%M%S%fZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize canonical F7 dataset inventory by phase.")
    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "preflight",
            "mlp_base",
            "mlp_kmeans",
            "flowgen_official_pool",
            "mlp_flowgen_official",
            "flowgen_trainonly_pool",
            "mlp_flowgen_trainonly",
            "xgb",
            "full",
        ],
    )
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    verbose = not bool(args.quiet)
    batch_id = args.batch_id or _batch_id_from_now()

    inventory_cfg, inventory_df = load_f7_materialization_inventory()
    del inventory_cfg

    if args.phase == "preflight":
        preflight = validate_f7_materialization_prerequisites()
        report_dir = write_f7_materialization_batch_report(
            batch_id=batch_id,
            phase_results={"preflight": {"rows": [], "failures": []}},
        )
        with open(report_dir / "preflight.json", "w", encoding="utf-8") as handle:
            json.dump(preflight, handle, indent=2, sort_keys=True)
        print(f"preflight_all_ok={preflight['all_ok']}")
        print(f"report_dir={report_dir}")
        if not preflight["all_ok"]:
            raise SystemExit(1)
        return

    phase_order = [
        "mlp_base",
        "mlp_kmeans",
        "flowgen_official_pool",
        "mlp_flowgen_official",
        "flowgen_trainonly_pool",
        "mlp_flowgen_trainonly",
        "xgb",
    ]
    selected_phases = phase_order if args.phase == "full" else [args.phase]

    phase_results: dict[str, dict] = {}
    for phase in selected_phases:
        result = materialize_f7_inventory_phase(
            phase=phase,
            inventory_df=inventory_df,
            batch_id=batch_id,
            force=bool(args.force),
            strict=bool(args.strict),
            verbose=verbose,
            dataset_id=args.dataset_id,
        )
        phase_results[phase] = result

    report_dir = write_f7_materialization_batch_report(
        batch_id=batch_id,
        phase_results=phase_results,
    )
    print(f"batch_id={batch_id}")
    print(f"report_dir={report_dir}")
    total_failures = sum(len(payload.get("failures") or []) for payload in phase_results.values())
    if total_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
