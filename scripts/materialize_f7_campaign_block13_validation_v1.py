from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.f7_campaign_spec import materialize_f7_campaign_spec


PRIMARY_SPEC_PATH = REPO_ROOT / "config" / "f7_campaign_block13_validation_primary_v1.yaml"
EXTENSION_SPEC_PATH = REPO_ROOT / "config" / "f7_campaign_block13_validation_extension_v1.yaml"
EXTENSION2_SPEC_PATH = REPO_ROOT / "config" / "f7_campaign_block13_validation_extension2_v1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the Block 13 runner-validation campaigns.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory override. Uses spec output paths by default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print both expansion manifests without writing artifacts.",
    )
    return parser


def _print_manifest(label: str, payload: dict) -> None:
    print(f"## {label}")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    specs = [
      ("primary", PRIMARY_SPEC_PATH),
      ("extension", EXTENSION_SPEC_PATH),
      ("extension2", EXTENSION2_SPEC_PATH),
    ]
    if args.dry_run:
        for label, spec_path in specs:
            bundle = materialize_f7_campaign_spec(
                spec_path=spec_path,
                output_root=args.output_root,
                write_outputs=False,
            )
            _print_manifest(label, bundle.expansion_manifest)
        return

    for label, spec_path in specs:
        bundle = materialize_f7_campaign_spec(spec_path=spec_path, output_root=args.output_root)
        print(f"[{label}] dataset candidates: {bundle.output_paths['dataset_candidate_inventory_path']}")
        print(f"[{label}] run specs: {bundle.output_paths['run_spec_inventory_path']}")
        print(f"[{label}] trials: {bundle.output_paths['trial_inventory_path']}")
        print(f"[{label}] expansion manifest: {bundle.output_paths['expansion_manifest_path']}")


if __name__ == "__main__":
    main()
