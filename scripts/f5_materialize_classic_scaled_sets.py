from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_contract import CLASSICAL_X_TRANSFORMS, SUPPORTED_Y_TRANSFORMS
from data.sets import DEFAULT_OFFICIAL_DATASET_NAME, load_or_create_scaled_sets


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Materialize the 16 classical F5 scaled datasets from the canonical official raw bundle."
    )
    ap.add_argument("--x", dest="x_scalers", action="append", choices=CLASSICAL_X_TRANSFORMS, default=None)
    ap.add_argument("--y", dest="y_scalers", action="append", choices=SUPPORTED_Y_TRANSFORMS, default=None)
    ap.add_argument(
        "--no-force",
        action="store_true",
        help="Reuse existing folders instead of forcing canonical rematerialization.",
    )
    ap.add_argument("--quiet", action="store_true", help="Reduce builder logs.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    x_scalers = args.x_scalers or list(CLASSICAL_X_TRANSFORMS)
    y_scalers = args.y_scalers or list(SUPPORTED_Y_TRANSFORMS)

    for x_scaler in x_scalers:
        for y_scaler in y_scalers:
            load_or_create_scaled_sets(
                raw_df_name=DEFAULT_OFFICIAL_DATASET_NAME,
                x_scaler_type=x_scaler,
                y_scaler_type=y_scaler,
                force=not args.no_force,
                verbose=not args.quiet,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
