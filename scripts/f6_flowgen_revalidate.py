from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sets import DEFAULT_OFFICIAL_DATASET_NAME, materialize_official_flowgen_augmented_sets
from evaluation.f6_reporting import write_f6_reports
from evaluation.f6_selection import rank_flowgen, summarize_flowgen_results
from evaluation.results import save_promotion_manifest
from scripts.f6_common import (
    CONFIGS_ROOT,
    DEFAULT_Y_SCALERS,
    FLOWGEN_LIMIT,
    OFFICIAL_SPLIT_ID,
    REPORTS_ROOT,
    ensure_budget,
    ensure_campaign_budget,
    load_json,
    load_yaml,
    source_id_matches_flowpre_pattern,
    sync_campaign_budget_ledger,
    write_yaml,
)


FLOWPRE_FINALISTS_ROOT = ROOT / "outputs" / "models" / "official" / "flowpre_finalists"
FLOWGEN_WORK_BASES = ("candidate_1", "candidate_2")
FLOWGEN_LEGACY_PRIOR = ROOT / "outputs" / "models" / "flowgen" / "W_flowgen_seed2898_v1" / "W_flowgen_seed2898_v1.yaml"
FLOWGEN_LEGACY_PRETRAINED = ROOT / "outputs" / "models" / "flowgen" / "W_flowgen_seed2898_v1" / "W_flowgen_seed2898_v1.pt"
SCREENING_SEED = 2898
RESEED_SEEDS = [2899, 2900, 2901]
MAX_CONFIGS = 6
MAX_TOTAL_RUNS = FLOWGEN_LIMIT

SHARED_MODEL_KEYS = [
    "embedding_dim",
    "hidden_features",
    "num_layers",
    "use_actnorm",
    "use_learnable_permutations",
    "num_bins",
    "tail_bound",
    "initial_affine_layers",
    "affine_rq_ratio",
    "n_repeat_blocks",
    "final_rq_layers",
    "lulinear_finisher",
]
SHARED_TRAIN_KEYS = [
    "batch_size",
    "learning_rate",
    "use_nll",
    "use_logdet_penalty",
    "logdet_penalty_weight",
    "use_logdet_abs",
    "use_logdet_sq",
    "use_mean_penalty",
    "mean_penalty_weight",
    "use_mean_abs",
    "use_mean_sq",
    "use_std_penalty",
    "std_penalty_weight",
    "use_std_abs",
    "use_std_sq",
    "use_skew_penalty",
    "skew_penalty_weight",
    "use_skew_abs",
    "use_skew_sq",
    "use_kurtosis_penalty",
    "kurtosis_penalty_weight",
    "use_kurtosis_abs",
    "use_kurtosis_sq",
    "use_logpz_centering",
    "logpz_centering_weight",
    "logpz_target",
    "early_stopping_patience",
    "lr_decay_patience",
    "lr_decay_factor",
    "min_improvement",
    "min_improvement_floor",
    "lr_patience_factor",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the canonical F6c FlowGen revalidation campaign.")
    ap.add_argument(
        "--work-base",
        choices=FLOWGEN_WORK_BASES,
        required=True,
        help="Canonical FlowGen work base to use (candidate_1 or candidate_2).",
    )
    ap.add_argument("--y", dest="y_scalers", action="append", choices=DEFAULT_Y_SCALERS, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-materialize", action="store_true", help="Promote FlowGen only; skip synthetic dataset materialization.")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def _resolve_flowgen_work_base_manifest(work_base_id: str) -> Path:
    finalist_root = FLOWPRE_FINALISTS_ROOT / work_base_id
    candidates = sorted(finalist_root.glob("*/*_promotion_manifest.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one promotion manifest under {finalist_root}, found {len(candidates)}."
        )
    return candidates[0]


def _scale_active_realism(training_cfg: dict, factor: float) -> None:
    for weight_key, toggle_key in [
        ("mmd_x_weight", "use_mmd_x"),
        ("mmd_y_weight", "use_mmd_y"),
        ("mmd_xy_weight", "use_mmd_xy"),
        ("w1_x_weight", "use_w1_x"),
        ("w1_y_weight", "use_w1_y"),
    ]:
        if training_cfg.get(toggle_key, False):
            training_cfg[weight_key] = float(training_cfg.get(weight_key, 0.0)) * factor


def _pretrained_compatible(flowpre_cfg: dict, flowgen_prior_cfg: dict) -> bool:
    for key in SHARED_MODEL_KEYS:
        if flowpre_cfg["model"].get(key) != flowgen_prior_cfg["model"].get(key):
            return False
    return True


def _validate_flowgen_work_base_manifest(path: Path, *, work_base_id: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"FlowGen work-base promotion manifest not found: {path}")
    payload = load_json(path)
    if str(payload.get("model_family")) != "flowpre":
        raise ValueError("FlowGen work-base manifest must have model_family='flowpre'.")
    branch_id = str(payload.get("branch_id"))
    if branch_id not in FLOWGEN_WORK_BASES:
        raise ValueError(
            "FlowGen work-base manifest must have branch_id in "
            f"{FLOWGEN_WORK_BASES}, got '{branch_id}'."
        )
    if branch_id != work_base_id:
        raise ValueError(
            f"FlowGen work-base manifest branch_id='{branch_id}' does not match requested work base '{work_base_id}'."
        )
    if str(payload.get("split_id")) != OFFICIAL_SPLIT_ID:
        raise ValueError(f"FlowGen work-base manifest must have split_id='{OFFICIAL_SPLIT_ID}'.")
    if not source_id_matches_flowpre_pattern(
        str(payload.get("source_id", "")),
        branch_id=branch_id,
        split_id=OFFICIAL_SPLIT_ID,
    ):
        raise ValueError(
            "FlowGen work-base manifest must use a source_id compatible with "
            f"'flowpre__{branch_id}__{OFFICIAL_SPLIT_ID}__*'."
        )
    run_manifest_path = _resolve_repo_path(str(payload.get("source_run_manifest", "")))
    if not run_manifest_path.exists():
        raise FileNotFoundError(f"FlowGen work-base source_run_manifest not found: {run_manifest_path}")
    expected_root = FLOWPRE_FINALISTS_ROOT / branch_id
    if expected_root not in run_manifest_path.resolve().parents:
        raise ValueError(
            "FlowGen work-base source_run_manifest must live under "
            f"{expected_root.relative_to(ROOT)}/."
        )
    return path


def _build_flowgen_candidates(flowgen_work_base_manifest_path: Path) -> tuple[list[tuple[str, dict]], dict]:
    work_base_promotion = load_json(flowgen_work_base_manifest_path)
    work_base_id = str(work_base_promotion["branch_id"])
    work_base_run_manifest_path = _resolve_repo_path(work_base_promotion["source_run_manifest"])
    work_base_run_manifest = load_json(work_base_run_manifest_path)
    work_base_run_dir = work_base_run_manifest_path.parent
    flowpre_cfg = load_yaml(work_base_run_dir / f"{work_base_run_manifest['run_id']}.yaml")
    flowgen_prior = load_yaml(FLOWGEN_LEGACY_PRIOR)

    baseline = copy.deepcopy(flowgen_prior)
    for key in SHARED_MODEL_KEYS:
        if key in flowpre_cfg["model"]:
            baseline["model"][key] = copy.deepcopy(flowpre_cfg["model"][key])
    for key in SHARED_TRAIN_KEYS:
        if key in flowpre_cfg["training"]:
            baseline["training"][key] = copy.deepcopy(flowpre_cfg["training"][key])

    baseline["training"]["seed"] = SCREENING_SEED
    compatible = _pretrained_compatible(flowpre_cfg, flowgen_prior)
    if compatible and FLOWGEN_LEGACY_PRETRAINED.exists():
        baseline["training"]["skip_phase1"] = True
        baseline["training"]["pretrained_path"] = str(FLOWGEN_LEGACY_PRETRAINED)
    else:
        baseline["training"]["skip_phase1"] = False
        baseline["training"].pop("pretrained_path", None)

    candidates = [("baseline", baseline)]

    low = copy.deepcopy(baseline)
    _scale_active_realism(low["training"], 0.75)
    candidates.append(("realism075", low))

    high = copy.deepcopy(baseline)
    _scale_active_realism(high["training"], 1.25)
    candidates.append(("realism125", high))

    ks_y = copy.deepcopy(baseline)
    ks_y["training"]["use_ks_y"] = True
    candidates.append(("ksy_on", ks_y))

    mmd_xy = copy.deepcopy(baseline)
    mmd_xy["training"]["use_mmd_xy"] = True
    candidates.append(("mmdxy_on", mmd_xy))

    ks_y_mmd_xy = copy.deepcopy(baseline)
    ks_y_mmd_xy["training"]["use_ks_y"] = True
    ks_y_mmd_xy["training"]["use_mmd_xy"] = True
    candidates.append(("ksy_mmdxy", ks_y_mmd_xy))

    if len(candidates) > MAX_CONFIGS:
        raise RuntimeError(f"FlowGen generated {len(candidates)} configs, exceeding limit {MAX_CONFIGS}.")
    meta = {
        "work_base_id": work_base_id,
        "work_base_promotion": work_base_promotion,
        "work_base_run_manifest": work_base_run_manifest,
        "compatible_with_legacy_w2898": compatible,
    }
    return candidates, meta


def _evaluation_context(flowgen_meta: dict, cfg_id: str, phase: str, seed: int) -> dict:
    work_base_id = str(flowgen_meta["work_base_id"])
    return {
        "dataset_name": DEFAULT_OFFICIAL_DATASET_NAME,
        "split_id": OFFICIAL_SPLIT_ID,
        "contract_id": "f6_flowgen_revalidation_v1",
        "seed_set_id": f"f6_flowgen_{phase}",
        "base_config_id": f"flowgen_{work_base_id}_work_base_v1",
        "objective_metric_id": "flowgen_realism_selection",
        "upstream_variant_fingerprint": flowgen_meta["work_base_run_manifest"].get("variant_fingerprint"),
        "run_level_axes": {
            "flowgen_work_base_id": work_base_id,
            "paired_flowpre_source_id": flowgen_meta["work_base_promotion"]["source_id"],
            "cfg_id": cfg_id,
            "phase": phase,
            "seed": seed,
        },
    }


def _run_candidate(cfg_id: str, cfg_path: Path, seed: int, *, phase: str, flowgen_meta: dict, device: str, verbose: bool) -> dict:
    from training.train_flowgen import train_flowgen_pipeline

    baseline_training = load_yaml(cfg_path)["training"]
    pretrained_path = baseline_training.get("pretrained_path")
    skip_phase1 = bool(baseline_training.get("skip_phase1", False))
    work_base_id = str(flowgen_meta["work_base_id"])

    model = train_flowgen_pipeline(
        config_filename=str(cfg_path),
        base_name=f"flowgen_{work_base_id}_tpv1_{cfg_id}_seed{seed}",
        device=device,
        seed=seed,
        verbose=verbose,
        allow_test_holdout=False,
        finetuning=True,
        skip_phase1=skip_phase1,
        pretrained_path=pretrained_path,
        evaluation_context=_evaluation_context(flowgen_meta, cfg_id, phase, seed),
        output_namespace="official",
    )
    artifacts = dict(getattr(model, "run_artifacts", {}) or {})
    if not artifacts.get("results_path"):
        raise RuntimeError(f"Missing FlowGen run artifacts for cfg={cfg_id} seed={seed}")
    row = summarize_flowgen_results(
        artifacts["results_path"],
        run_id=artifacts["run_id"],
        cfg_id=cfg_id,
        phase=phase,
        seed=seed,
    )
    row["results_path"] = artifacts["results_path"]
    row["metrics_long_path"] = artifacts.get("metrics_long_path")
    row["run_dir"] = artifacts["run_dir"]
    row["config_path"] = artifacts["saved_config_path"]
    return row


def main() -> int:
    args = _parse_args()
    y_scalers = args.y_scalers or list(DEFAULT_Y_SCALERS)
    verbose = not args.quiet

    flowgen_work_base_manifest = _validate_flowgen_work_base_manifest(
        _resolve_flowgen_work_base_manifest(args.work_base),
        work_base_id=args.work_base,
    )

    candidates, flowgen_meta = _build_flowgen_candidates(Path(flowgen_work_base_manifest))
    planned_runs = len(candidates) + (2 * len(RESEED_SEEDS))
    ensure_budget(planned=planned_runs, limit=MAX_TOTAL_RUNS, label="F6 FlowGen")
    ensure_campaign_budget(planned_flowpre=0, planned_flowgen=planned_runs)

    report_dir = REPORTS_ROOT
    report_dir.mkdir(parents=True, exist_ok=True)

    screening_rows = []
    cfg_lookup = {}
    for cfg_id, cfg_payload in candidates:
        cfg_path = write_yaml(CONFIGS_ROOT / "flowgen" / f"{cfg_id}.yaml", cfg_payload)
        cfg_lookup[cfg_id] = cfg_path
        screening_rows.append(
            _run_candidate(
                cfg_id,
                cfg_path,
                SCREENING_SEED,
                phase="screen",
                flowgen_meta=flowgen_meta,
                device=args.device,
                verbose=verbose,
            )
        )

    screening_df = pd.DataFrame(screening_rows)
    screening_df.to_csv(report_dir / "flowgen_screening.csv", index=False)
    top_cfgs = rank_flowgen(screening_df).head(2)["cfg_id"].tolist()

    reseed_rows = []
    for cfg_id in top_cfgs:
        for seed in RESEED_SEEDS:
            reseed_rows.append(
                _run_candidate(
                    cfg_id,
                    cfg_lookup[cfg_id],
                    seed,
                    phase="reseed",
                    flowgen_meta=flowgen_meta,
                    device=args.device,
                    verbose=verbose,
                )
            )

    all_df = pd.DataFrame(screening_rows + reseed_rows)
    all_df.to_csv(report_dir / "flowgen_all_runs.csv", index=False)

    numeric_cols = [col for col in all_df.columns if pd.api.types.is_numeric_dtype(all_df[col]) and col not in {"seed"}]
    agg = all_df.groupby(["cfg_id"], as_index=False)[numeric_cols].mean()
    run_counts = all_df.groupby(["cfg_id"]).size().rename("n_runs").reset_index()
    agg = agg.merge(run_counts, on="cfg_id", how="left")
    agg_ranked = rank_flowgen(agg)
    agg_ranked.to_csv(report_dir / "flowgen_aggregate.csv", index=False)

    winner_cfg = str(agg_ranked.iloc[0]["cfg_id"])
    winner_runs = rank_flowgen(all_df[all_df["cfg_id"] == winner_cfg].copy())
    winner = winner_runs.iloc[0].to_dict()

    promotion_path = Path(str(winner["run_dir"])) / f"{winner['run_id']}_promotion_manifest.json"
    work_base_id = str(flowgen_meta["work_base_id"])
    flowgen_source_id = f"flowgen__{work_base_id}__{OFFICIAL_SPLIT_ID}__v1"
    save_promotion_manifest(
        out_path=promotion_path,
        model_family="flowgen",
        source_id=flowgen_source_id,
        source_run_manifest_path=Path(str(winner["run_dir"])) / f"{winner['run_id']}_run_manifest.json",
        source_metrics_long_path=winner["metrics_long_path"],
        split_id=OFFICIAL_SPLIT_ID,
        cleaning_policy_id="trainfit_overlap_cap1pct_holdoutflag_v1",
        raw_bundle_manifest_path=ROOT / "data" / "sets" / "official" / "init_temporal_processed_v1" / "raw" / DEFAULT_OFFICIAL_DATASET_NAME / "manifest.json",
        branch_id=work_base_id,
        paired_flowpre_source_id=flowgen_meta["work_base_promotion"]["source_id"],
        extra_fields={
            "selection_cfg_id": winner_cfg,
            "selection_phase": "f6c",
            "historical_support_only": False,
            "n_runs_aggregated": int(agg_ranked.iloc[0]["n_runs"]),
            "flowgen_work_base_id": work_base_id,
        },
    )

    selected = dict(winner)
    selected["branch_id"] = work_base_id
    selected["flowgen_work_base_id"] = work_base_id
    selected["flowgen_source_id"] = flowgen_source_id
    selected["promotion_manifest_path"] = str(promotion_path)
    selected["paired_flowpre_source_id"] = flowgen_meta["work_base_promotion"]["source_id"]
    pd.DataFrame([selected]).to_csv(report_dir / "flowgen_selected.csv", index=False)

    budget_snapshot = sync_campaign_budget_ledger()
    write_f6_reports(budget_snapshot=budget_snapshot)

    if not args.no_materialize:
        materialize_official_flowgen_augmented_sets(
            flowgen_work_base_manifest_path=str(flowgen_work_base_manifest),
            flowgen_promotion_manifest_path=str(promotion_path),
            y_scalers=y_scalers,
            source_dataset_name=DEFAULT_OFFICIAL_DATASET_NAME,
            force=args.force,
            device=args.device,
            verbose=verbose,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
