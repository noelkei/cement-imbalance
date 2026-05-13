from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sets import DEFAULT_OFFICIAL_DATASET_NAME, materialize_official_flowpre_sets
from evaluation.f6_reporting import write_f6_reports
from evaluation.f6_selection import FLOWPRE_BRANCHES, rank_flowpre_branch, summarize_flowpre_results
from evaluation.results import save_promotion_manifest
from scripts.f6_common import (
    CONFIGS_ROOT,
    DEFAULT_Y_SCALERS,
    FLOWPRE_LIMIT,
    REPORTS_ROOT,
    ensure_budget,
    ensure_campaign_budget,
    load_yaml,
    sync_campaign_budget_ledger,
    write_yaml,
)


ANCHORS = {
    "rrmse": ROOT / "outputs" / "models" / "flow_pre" / "flow_pre_rrmse_r131_s5678_v1" / "flow_pre_rrmse_r131_s5678_v1.yaml",
    "mvn": ROOT / "outputs" / "models" / "flow_pre" / "flow_pre_mvn_r349_s9101_v1" / "flow_pre_mvn_r349_s9101_v1.yaml",
    "fair": ROOT / "outputs" / "models" / "flow_pre" / "flow_pre_fair_r25_s1234_v1" / "flow_pre_fair_r25_s1234_v1.yaml",
}
SCREENING_SEEDS = {"rrmse": 5678, "mvn": 9101, "fair": 1234}
COMMON_SEEDS = [1234, 5678, 9101]
MAX_CONFIGS = {"rrmse": 5, "mvn": 6, "fair": 5}
MAX_TOTAL_RUNS = FLOWPRE_LIMIT


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the canonical F6a/F6b FlowPre revalidation campaign.")
    ap.add_argument("--branch", dest="branches", action="append", choices=FLOWPRE_BRANCHES, default=None)
    ap.add_argument("--y", dest="y_scalers", action="append", choices=DEFAULT_Y_SCALERS, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true", help="Re-run training even if official FlowPre bundles already exist.")
    ap.add_argument("--no-materialize", action="store_true", help="Stop after promotion manifests; do not materialize official FlowPre-based datasets.")
    ap.add_argument("--quiet", action="store_true", help="Reduce trainer logs.")
    return ap.parse_args()


def _deep_set(cfg: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    cur = cfg
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def _build_branch_candidates(branch: str) -> list[tuple[str, dict]]:
    anchor = load_yaml(ANCHORS[branch])
    specs: list[tuple[str, dict]] = [("anchor", anchor)]

    mutations = {
        "rrmse": [
            ("lr1e-4", {"training.learning_rate": 1e-4}),
            ("hidden192", {"model.hidden_features": 192}),
            ("layers3", {"model.num_layers": 3}),
            ("rq5", {"model.affine_rq_ratio": [1, 5], "model.final_rq_layers": 5}),
        ],
        "mvn": [
            ("lr1e-4", {"training.learning_rate": 1e-4}),
            ("hidden128", {"model.hidden_features": 128}),
            ("layers3", {"model.num_layers": 3}),
            ("rq6", {"model.affine_rq_ratio": [1, 6], "model.final_rq_layers": 6}),
            ("skewkurt_on", {"training.use_skew_penalty": True, "training.use_kurtosis_penalty": True}),
        ],
        "fair": [
            ("lr1e-4", {"training.learning_rate": 1e-4}),
            ("hidden192", {"model.hidden_features": 192}),
            ("layers2", {"model.num_layers": 2}),
            ("rq5", {"model.affine_rq_ratio": [1, 5], "model.final_rq_layers": 5}),
        ],
    }[branch]

    for cfg_id, overrides in mutations:
        cfg = copy.deepcopy(anchor)
        for dotted_key, value in overrides.items():
            _deep_set(cfg, dotted_key, value)
        specs.append((cfg_id, cfg))

    if len(specs) > MAX_CONFIGS[branch]:
        raise RuntimeError(f"Branch {branch} generated {len(specs)} configs, exceeding limit {MAX_CONFIGS[branch]}.")
    return specs


def _evaluation_context(branch: str, cfg_id: str, phase: str, seed: int) -> dict:
    return {
        "dataset_name": DEFAULT_OFFICIAL_DATASET_NAME,
        "split_id": "init_temporal_processed_v1",
        "contract_id": "f6_flowpre_revalidation_v1",
        "seed_set_id": f"f6_flowpre_{phase}",
        "base_config_id": f"flowpre_anchor_{branch}",
        "objective_metric_id": f"flowpre_{branch}_selection",
        "run_level_axes": {
            "branch_id": branch,
            "cfg_id": cfg_id,
            "phase": phase,
            "seed": seed,
        },
    }


def _run_candidate(branch: str, cfg_id: str, cfg_path: Path, seed: int, *, phase: str, device: str, verbose: bool) -> dict:
    from training.train_flow_pre import train_flowpre_pipeline

    model = train_flowpre_pipeline(
        config_filename=str(cfg_path),
        base_name=f"flowpre_{branch}_tpv1_{cfg_id}_seed{seed}",
        device=device,
        seed=seed,
        verbose=verbose,
        allow_test_holdout=False,
        evaluation_context=_evaluation_context(branch, cfg_id, phase, seed),
        output_namespace="official",
    )
    artifacts = dict(getattr(model, "run_artifacts", {}) or {})
    if not artifacts.get("results_path"):
        raise RuntimeError(f"Missing run artifacts for FlowPre run branch={branch} cfg={cfg_id} seed={seed}")
    row = summarize_flowpre_results(
        artifacts["results_path"],
        branch_id=branch,
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


def _aggregate_branch(df: pd.DataFrame, branch: str) -> pd.DataFrame:
    branch_df = df[df["branch_id"] == branch].copy()
    numeric_cols = [
        col for col in branch_df.columns
        if pd.api.types.is_numeric_dtype(branch_df[col]) and col not in {"seed"}
    ]
    agg = branch_df.groupby(["branch_id", "cfg_id"], as_index=False)[numeric_cols].mean()
    run_counts = branch_df.groupby(["branch_id", "cfg_id"]).size().rename("n_runs").reset_index()
    agg = agg.merge(run_counts, on=["branch_id", "cfg_id"], how="left")
    return rank_flowpre_branch(agg, branch)


def main() -> int:
    args = _parse_args()
    branches = args.branches or list(FLOWPRE_BRANCHES)
    y_scalers = args.y_scalers or list(DEFAULT_Y_SCALERS)
    verbose = not args.quiet

    planned_screen = sum(len(_build_branch_candidates(branch)) for branch in branches)
    planned_reseed = len(branches) * 2 * 2
    planned_runs = planned_screen + planned_reseed
    ensure_budget(planned=planned_runs, limit=MAX_TOTAL_RUNS, label="F6 FlowPre")
    ensure_campaign_budget(planned_flowpre=planned_runs, planned_flowgen=0)

    report_dir = REPORTS_ROOT
    report_dir.mkdir(parents=True, exist_ok=True)

    screening_rows: list[dict] = []
    candidate_specs = {branch: _build_branch_candidates(branch) for branch in branches}

    for branch in branches:
        screening_seed = SCREENING_SEEDS[branch]
        for cfg_id, cfg_payload in candidate_specs[branch]:
            cfg_path = write_yaml(CONFIGS_ROOT / "flowpre" / branch / f"{cfg_id}.yaml", cfg_payload)
            screening_rows.append(
                _run_candidate(
                    branch,
                    cfg_id,
                    cfg_path,
                    screening_seed,
                    phase="screen",
                    device=args.device,
                    verbose=verbose,
                )
            )

    screening_df = pd.DataFrame(screening_rows)
    screening_df.to_csv(report_dir / "flowpre_screening.csv", index=False)

    top_cfgs_by_branch: dict[str, list[str]] = {}
    for branch in branches:
        ranked_screen = rank_flowpre_branch(screening_df[screening_df["branch_id"] == branch], branch)
        top_cfgs_by_branch[branch] = ranked_screen.head(2)["cfg_id"].tolist()

    reseed_rows: list[dict] = []
    for branch in branches:
        screening_seed = SCREENING_SEEDS[branch]
        reseed_seeds = [seed for seed in COMMON_SEEDS if seed != screening_seed]
        cfg_lookup = {cfg_id: cfg for cfg_id, cfg in candidate_specs[branch]}
        for cfg_id in top_cfgs_by_branch[branch]:
            cfg_path = write_yaml(CONFIGS_ROOT / "flowpre" / branch / f"{cfg_id}.yaml", cfg_lookup[cfg_id])
            for seed in reseed_seeds:
                reseed_rows.append(
                    _run_candidate(
                        branch,
                        cfg_id,
                        cfg_path,
                        seed,
                        phase="reseed",
                        device=args.device,
                        verbose=verbose,
                    )
                )

    all_rows = screening_rows + reseed_rows
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(report_dir / "flowpre_all_runs.csv", index=False)

    branch_aggregates = []
    selected_rows = []
    promoted_upstreams: dict[str, str] = {}

    for branch in branches:
        agg = _aggregate_branch(all_df, branch)
        branch_aggregates.append(agg)
        winner_cfg = str(agg.iloc[0]["cfg_id"])
        winner_runs = rank_flowpre_branch(
            all_df[(all_df["branch_id"] == branch) & (all_df["cfg_id"] == winner_cfg)].copy(),
            branch,
        )
        winner = winner_runs.iloc[0].to_dict()
        promotion_path = Path(str(winner["run_dir"])) / f"{winner['run_id']}_promotion_manifest.json"
        source_id = f"flowpre__{branch}__init_temporal_processed_v1__v1"
        save_promotion_manifest(
            out_path=promotion_path,
            model_family="flowpre",
            source_id=source_id,
            branch_id=branch,
            source_run_manifest_path=Path(str(winner["run_dir"])) / f"{winner['run_id']}_run_manifest.json",
            source_metrics_long_path=winner["metrics_long_path"],
            split_id="init_temporal_processed_v1",
            cleaning_policy_id="trainfit_overlap_cap1pct_holdoutflag_v1",
            raw_bundle_manifest_path=ROOT / "data" / "sets" / "official" / "init_temporal_processed_v1" / "raw" / DEFAULT_OFFICIAL_DATASET_NAME / "manifest.json",
            extra_fields={
                "selection_cfg_id": winner_cfg,
                "selection_phase": "f6b",
                "historical_support_only": bool(agg.iloc[0].get("historical_support_only", False)),
                "n_runs_aggregated": int(agg.iloc[0]["n_runs"]),
            },
        )
        winner["flowpre_source_id"] = source_id
        winner["promotion_manifest_path"] = str(promotion_path)
        promoted_upstreams[branch] = str(promotion_path)
        selected_rows.append(winner)

    pd.concat(branch_aggregates, axis=0, ignore_index=True).to_csv(report_dir / "flowpre_aggregate.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(report_dir / "flowpre_selected.csv", index=False)

    budget_snapshot = sync_campaign_budget_ledger()
    write_f6_reports(budget_snapshot=budget_snapshot)

    if not args.no_materialize:
        materialize_official_flowpre_sets(
            promoted_upstreams,
            y_scalers,
            df_name=DEFAULT_OFFICIAL_DATASET_NAME,
            device=args.device,
            force=args.force,
            verbose=verbose,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
