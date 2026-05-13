from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SEED = 2468
EPOCH_BUDGETS = (30, 50, 100)
BASE_RUN_ID = "flowprers1_rrmse_tpv1_hf256_l4_rq5_lr1e-3_mson_skoff_seed2468_v1"
BASE_CONFIG_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "official"
    / "flowpre_finalists"
    / "candidate_2"
    / BASE_RUN_ID
    / f"{BASE_RUN_ID}.yaml"
)
CONFIG_ROOT = (
    ROOT
    / "outputs"
    / "models"
    / "experimental"
    / "train_only"
    / "flow_pre"
    / "configs"
    / "candidate2_epochs"
)
OUTPUT_NAMESPACE = "experimental/train_only"
OUTPUT_SUBDIR = "candidate2_epochs"
CONTRACT_ID = "experimental_flowpre_train_only_candidate2_epochs_v1"
DEFAULT_OFFICIAL_DATASET_NAME = "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run the experimental FlowPre train-only candidate_2 epoch-budget micro-round "
            "for budgets 30, 50 and 100 with seed 2468."
        )
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quiet", action="store_true", help="Reduce trainer stdout logging.")
    ap.add_argument("--dry-run", action="store_true", help="Print the planned runs without writing configs or training.")
    ap.add_argument(
        "--epoch-budget",
        type=int,
        choices=EPOCH_BUDGETS,
        action="append",
        help="Run only one or more selected epoch budgets. Defaults to all three.",
    )
    return ap.parse_args()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Base candidate_2 config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Base candidate_2 config is not a YAML mapping: {path}")
    if "training" not in loaded or not isinstance(loaded["training"], dict):
        raise ValueError(f"Base candidate_2 config has no training mapping: {path}")
    return loaded


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _run_id(epoch_budget: int) -> str:
    return f"flowpre_trainonly_c2prior_e{int(epoch_budget)}_seed{SEED}_v1"


def _config_path(epoch_budget: int) -> Path:
    return CONFIG_ROOT / f"{_run_id(epoch_budget)}.yaml"


def _run_dir(epoch_budget: int) -> Path:
    return ROOT / "outputs" / "models" / OUTPUT_NAMESPACE / "flow_pre" / OUTPUT_SUBDIR / _run_id(epoch_budget)


def _build_config(base_config: dict, epoch_budget: int) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["training"]["num_epochs"] = int(epoch_budget)
    return cfg


def _evaluation_context(epoch_budget: int) -> dict:
    return {
        "dataset_name": DEFAULT_OFFICIAL_DATASET_NAME,
        "split_id": "init_temporal_processed_v1",
        "contract_id": CONTRACT_ID,
        "seed_set_id": f"flowpre_trainonly_candidate2_seed{SEED}",
        "base_config_id": "flowpre_candidate_2_official_prior",
        "objective_metric_id": "flowpre_trainonly_monitor_loss",
        "run_level_axes": {
            "phase": "experimental_train_only_micro_round",
            "prior": "candidate_2",
            "source_run_id": BASE_RUN_ID,
            "epoch_budget": int(epoch_budget),
            "seed": SEED,
        },
    }


def _planned_budgets(args: argparse.Namespace) -> tuple[int, ...]:
    if args.epoch_budget:
        return tuple(dict.fromkeys(int(v) for v in args.epoch_budget))
    return EPOCH_BUDGETS


def main() -> None:
    args = _parse_args()
    budgets = _planned_budgets(args)
    base_config = _load_yaml(BASE_CONFIG_PATH)

    print("Experimental FlowPre train-only candidate_2 epoch micro-round")
    print(f"Base config: {BASE_CONFIG_PATH}")
    print(f"Seed: {SEED}")
    print(f"Epoch budgets: {', '.join(str(v) for v in budgets)}")
    print(f"Output root: {ROOT / 'outputs' / 'models' / OUTPUT_NAMESPACE / 'flow_pre' / OUTPUT_SUBDIR}")

    for epoch_budget in budgets:
        run_id = _run_id(epoch_budget)
        cfg_path = _config_path(epoch_budget)
        run_dir = _run_dir(epoch_budget)

        print("")
        print(f"Run: {run_id}")
        print(f"  override: training.num_epochs = {epoch_budget}")
        print(f"  derived config: {cfg_path}")
        print(f"  run dir: {run_dir}")

        if args.dry_run:
            continue
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")

        cfg = _build_config(base_config, epoch_budget)
        _write_yaml(cfg_path, cfg)

        from training.train_flow_pre import train_flowpre_pipeline

        model = train_flowpre_pipeline(
            config_filename=str(cfg_path),
            base_name=f"flowpre_trainonly_c2prior_e{epoch_budget}_seed{SEED}",
            device=args.device,
            seed=SEED,
            verbose=not args.quiet,
            allow_test_holdout=False,
            evaluation_context=_evaluation_context(epoch_budget),
            output_namespace=OUTPUT_NAMESPACE,
            output_subdir=OUTPUT_SUBDIR,
            fixed_run_id=run_id,
            log_in_run_dir=True,
            monitoring_policy="train_only",
        )
        artifacts = dict(getattr(model, "run_artifacts", {}) or {})
        results_path = artifacts.get("results_path")
        if not results_path:
            raise RuntimeError(f"Run finished without a results artifact: {run_id}")
        print(f"  results: {results_path}")


if __name__ == "__main__":
    main()
