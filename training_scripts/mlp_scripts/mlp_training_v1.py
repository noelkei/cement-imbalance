# run_all_scaled_sets.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

from training.utils import ROOT_PATH, list_scaled_sets
from training.optuna_mlp import (
    run_mlp_study_pipeline,
    make_storage_path,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna MLP studies over all scaled sets except an excluded one."
    )
    parser.add_argument(
        "--exclude",
        type=str,
        required=True,
        help="Name of the scaled set to skip (e.g. 'df_scaled_x_flowpre_fair_yminmax').",
    )
    parser.add_argument(
        "--metric-path",
        type=str,
        default="per_class.rrmse",
        help="Metric path (e.g. 'overall.rmse', 'per_class.rrmse', or 'per_class.<cls>.rmse').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base seed used for the study sampler and (by default) the objective.",
    )
    parser.add_argument(
        "--warmup-trials",
        type=int,
        default=150,
        help="Warmup trials (used only when --n-trials is not provided).",
    )
    parser.add_argument(
        "--wait-trials",
        type=int,
        default=10,
        help="Stop after this many consecutive non-improving trials (improvement mode).",
    )
    parser.add_argument(
        "--improve-pct",
        type=float,
        default=0.05,
        help="Required relative improvement in percent (e.g., 0.05 means 0.05%).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="If provided, run a fixed number of trials; if omitted, use improvement-based early stop.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Training device preference.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose training logs for each run.",
    )
    args = parser.parse_args()

    # Base config
    base_config_path = Path(ROOT_PATH) / "config" / "mlp.yaml"

    # Discover sets and build run list
    all_sets = sorted(list_scaled_sets())
    print("Available scaled sets:", all_sets)
    run_sets = [s for s in all_sets if s != args.exclude]
    if not run_sets:
        print("Nothing to run: after exclusion, no scaled sets remain.")
        return

    print(f"\nExcluding: {args.exclude}")
    print("Will run:", run_sets, "\n")

    results: List[Tuple[str, float, Dict[str, Any]]] = []
    failures: List[Tuple[str, str]] = []

    for scaled_set_name in run_sets:
        metric_path = args.metric_path
        # Make a filesystem-friendly study name
        safe_metric = metric_path.replace(".", "_")
        study_name = f"mlp_{safe_metric}_{scaled_set_name}"

        print("=" * 100)
        print(f"▶️  Running study for: {scaled_set_name}")
        print(f"    study_name   : {study_name}")
        print(f"    metric_path  : {metric_path}")
        print(f"    base_config  : {base_config_path}")
        print("=" * 100)

        try:
            study, best_params, best_value = run_mlp_study_pipeline(
                scaled_set_name=scaled_set_name,
                base_config_path=base_config_path,
                metric_path=metric_path,

                # trials policy
                n_trials=None,              # None -> improvement-based
                warmup_trials=args.warmup_trials,
                wait_trials=args.wait_trials,
                improve_pct=args.improve_pct,

                # optuna study creation (auto SQLite storage)
                study_name=study_name,
                storage=make_storage_path(study_name),
                direction="minimize",
                sampler_seed=args.seed,
                pruner=None,

                # data/model knobs
                condition_col="type",

                # objective passthrough (everything explicit, but defaults preserved)
                objective_kwargs={
                    "device": args.device,
                    "metric_path": metric_path,
                    "base_config_path": base_config_path,
                    "condition_col": "type",
                    "seed": args.seed,
                    "seed_base": None,
                    "num_epochs_override": None,
                    "extra_overrides": None,
                    "verbose": bool(args.verbose),
                    # Note: splits injected by pipeline
                },
            )
            print("\nStudy done.")
            print("Best value :", best_value)
            print("Best params:", best_params)
            results.append((scaled_set_name, float(best_value), best_params))

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"\n❌ Study failed for {scaled_set_name}: {msg}")
            failures.append((scaled_set_name, msg))

        print("\n")

    # Summary
    print("#" * 100)
    print("SUMMARY")
    if results:
        print("\nSuccessful runs (sorted by best value):")
        for name, val, _ in sorted(results, key=lambda t: t[1]):
            print(f"  {name:<45}  best={val:.6f}")
    else:
        print("\nNo successful runs.")

    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"  {name:<45}  {msg}")

    print("#" * 100)


if __name__ == "__main__":
    main()
