from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_contract import build_base_scaling_matrix, build_supported_dataset_matrix


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Emit the F5 dataset-level supported space matrix.")
    ap.add_argument(
        "--include-synthetic-policies",
        action="store_true",
        help="Expand the matrix beyond the 28 base XxY combinations.",
    )
    ap.add_argument(
        "--include-out-of-scope",
        action="store_true",
        help="Include combinations that the current architecture marks as out_of_scope.",
    )
    ap.add_argument(
        "--format",
        choices=("json", "csv", "pretty"),
        default="pretty",
        help="Output format.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path. Defaults to stdout.",
    )
    return ap.parse_args()


def _build_rows(args: argparse.Namespace) -> list[dict]:
    if args.include_synthetic_policies:
        return build_supported_dataset_matrix(
            include_synthetic_policies=True,
            include_out_of_scope=args.include_out_of_scope,
        )
    return build_base_scaling_matrix()


def _emit_pretty(rows: list[dict], handle) -> None:
    for row in rows:
        axes = row["dataset_level_axes"]
        handle.write(
            f"{row['dataset_name']}\n"
            f"  x_transform      : {axes['x_transform']}\n"
            f"  y_transform      : {axes['y_transform']}\n"
            f"  synthetic_policy : {axes['synthetic_policy']}\n"
            f"  support_status   : {row['support_status']}\n"
            f"  storage_family   : {row['dataset_storage_family']}\n"
            f"  reason           : {row['status_reason']}\n"
            "\n"
        )


def _emit_csv(rows: list[dict], handle) -> None:
    fieldnames = [
        "dataset_name",
        "x_transform",
        "y_transform",
        "synthetic_policy",
        "support_status",
        "dataset_storage_family",
        "status_reason",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        axes = row["dataset_level_axes"]
        writer.writerow(
            {
                "dataset_name": row["dataset_name"],
                "x_transform": axes["x_transform"],
                "y_transform": axes["y_transform"],
                "synthetic_policy": axes["synthetic_policy"],
                "support_status": row["support_status"],
                "dataset_storage_family": row["dataset_storage_family"],
                "status_reason": row["status_reason"],
            }
        )


def main() -> int:
    args = _parse_args()
    rows = _build_rows(args)
    if args.out is None:
        handle = sys.stdout
        should_close = False
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        handle = args.out.open("w", encoding="utf-8", newline="")
        should_close = True

    try:
        if args.format == "json":
            json.dump(rows, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        elif args.format == "csv":
            _emit_csv(rows, handle)
        else:
            _emit_pretty(rows, handle)
    finally:
        if should_close:
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
