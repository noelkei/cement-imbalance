from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.f7_mlp_baseline_revalidation_v2 import ROOT_PATH, run_phase


def main() -> None:
    parser = argparse.ArgumentParser(description="Tercera y ultima revalidacion 3x40 del baseline MLP de F7.")
    parser.add_argument(
        "--spec",
        default="config/f7_mlp_baseline_revalidation_v3.yaml",
        help="Ruta al spec YAML de la mini-revalidacion.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device preferido para training: auto | mps | cuda | cpu.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Etiqueta para separar benchmark runs, por ejemplo 'cpu_v3'.",
    )
    parser.add_argument(
        "--phase",
        choices=["val_grid", "winner_test_check"],
        default="val_grid",
        help="Fase a ejecutar: grid principal en val o chequeo final del ganador con test.",
    )
    parser.add_argument(
        "--include-test-for-all",
        action="store_true",
        help="Solo para debugging. Habilita test en todas las runs del grid; no recomendado.",
    )
    args = parser.parse_args()

    spec_path = ROOT_PATH / args.spec
    if args.phase == "winner_test_check":
        run_phase(spec_path=spec_path, device=args.device, run_label=args.run_label, winner_only=True)
    else:
        run_phase(
            spec_path=spec_path,
            device=args.device,
            run_label=args.run_label,
            include_test_for_all=bool(args.include_test_for_all),
            winner_only=False,
        )


if __name__ == "__main__":
    main()
