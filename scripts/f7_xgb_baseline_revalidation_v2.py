from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.f7_xgb_baseline_revalidation import ROOT_PATH, run_shap_smoke, run_val_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-revalidacion final 3x9 del baseline XGBoost para F7.")
    parser.add_argument(
        "--spec",
        default="config/f7_xgb_baseline_revalidation_v2.yaml",
        help="Ruta al spec YAML de la micro-revalidacion.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Etiqueta para separar runs, por ejemplo 'cpu_xgb_v2'.",
    )
    parser.add_argument(
        "--phase",
        choices=["val_grid", "shap_smoke"],
        default="val_grid",
        help="Fase a ejecutar: grid principal o smoke test de SHAP.",
    )
    args = parser.parse_args()

    spec_path = ROOT_PATH / args.spec
    if args.phase == "shap_smoke":
        run_shap_smoke(spec_path=spec_path, run_label=args.run_label)
    else:
        run_val_grid(spec_path=spec_path, run_label=args.run_label)


if __name__ == "__main__":
    main()
