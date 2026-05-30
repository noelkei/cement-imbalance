from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.f7_campaign_trial_consumption import (  # noqa: E402
    DEFAULT_TRIAL_INVENTORY_PATH,
    DEFAULT_VALIDATION_ROOT,
    validate_trial_consumption_sample,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate consumption of a structured 10-trial sample from the F7 campaign trial inventory.")
    parser.add_argument(
        "--trial-inventory-path",
        type=Path,
        default=DEFAULT_TRIAL_INVENTORY_PATH,
        help="Path to the materialized F7 trial inventory CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_VALIDATION_ROOT,
        help="Output directory for validation artifacts.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of structured sample trials to validate.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    bundle = validate_trial_consumption_sample(
        trial_inventory_path=args.trial_inventory_path,
        output_root=args.output_root,
        sample_size=int(args.sample_size),
    )
    print(f"Sample CSV written to: {bundle.output_paths['sample_path']}")
    print(f"Validation CSV written to: {bundle.output_paths['validation_path']}")
    print(f"Markdown report written to: {bundle.output_paths['report_path']}")
    print(
        "Validation summary: "
        f"sample_size={bundle.summary['sample_size']} "
        f"ok={bundle.summary['ok_count']} "
        f"failed={bundle.summary['failed_count']} "
        f"family_counts={bundle.summary['model_family_counts']}"
    )


if __name__ == "__main__":
    main()
