from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.f7_launch_readiness import (  # noqa: E402
    DEFAULT_EXTENSION_SPEC_PATHS,
    DEFAULT_JSON_PATH,
    DEFAULT_MD_PATH,
    DEFAULT_PRIMARY_SPEC_PATH,
    generate_f7_launch_readiness_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the canonical F7 launch-readiness report.")
    parser.add_argument("--primary-spec", type=Path, default=DEFAULT_PRIMARY_SPEC_PATH)
    parser.add_argument("--extension-spec", type=Path, action="append", default=[])
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_PATH)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    extension_specs = args.extension_spec or DEFAULT_EXTENSION_SPEC_PATHS
    payload = generate_f7_launch_readiness_report(
        primary_spec_path=args.primary_spec,
        extension_spec_paths=extension_specs,
        json_output_path=args.json_output,
        markdown_output_path=args.markdown_output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
