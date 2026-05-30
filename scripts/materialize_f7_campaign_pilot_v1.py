from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.f7_campaign_spec import materialize_f7_campaign_spec


DEFAULT_SPEC_PATH = REPO_ROOT / "config" / "f7_campaign_pilot_v1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the canonical F7 pilot campaign inventories.")
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=DEFAULT_SPEC_PATH,
        help="Path to the F7 pilot campaign spec YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory override. Uses spec output paths by default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the expansion summary without writing artifacts.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.dry_run:
        bundle = materialize_f7_campaign_spec(
            spec_path=args.spec_path,
            output_root=args.output_root,
            write_outputs=False,
        )
        print(json.dumps(bundle.expansion_manifest, indent=2, sort_keys=True))
        return

    bundle = materialize_f7_campaign_spec(spec_path=args.spec_path, output_root=args.output_root)
    print(f"Dataset candidate inventory written to: {bundle.output_paths['dataset_candidate_inventory_path']}")
    print(f"Run spec inventory written to: {bundle.output_paths['run_spec_inventory_path']}")
    print(f"Trial inventory written to: {bundle.output_paths['trial_inventory_path']}")
    print(f"Expansion manifest written to: {bundle.output_paths['expansion_manifest_path']}")


if __name__ == "__main__":
    main()
