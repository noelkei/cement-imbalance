from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sets import DEFAULT_OFFICIAL_DATASET_NAME
from scripts.f6_common import write_yaml
from scripts.f6_flowpre_evaluate_current import _build_current_all_runs, _build_inventory
from scripts.f6_flowpre_explore_v2 import _config_from_spec, _config_signature
from scripts.f6_flowpre_revalidate import COMMON_SEEDS


OFFICIAL_FLOWPRE_ROOT = ROOT / "outputs" / "models" / "official" / "flow_pre"
REPORT_ROOT = ROOT / "outputs" / "reports" / "f6_reseed_topcfgs_v1"
CONFIG_ROOT = REPORT_ROOT / "configs" / "flowpre"
RESEED_CONTRACT_ID = "f6_flowpre_reseed_topcfgs_v1"
OFFICIAL_SPLIT_ID = "init_temporal_processed_v1"
TARGET_SEED_SET = [*COMMON_SEEDS, 2468]

SIGNATURE_RE = re.compile(
    r"hf(?P<hf>\d+)\|l(?P<layers>\d+)\|rq1x(?P<rq>\d+)\|frq(?P<frq>\d+)\|lr(?P<lr>[^|]+)\|ms(?P<ms>on|off)\|sk(?P<sk>on|off)"
)


def _candidate(signature: str, anchor_branch: str) -> dict[str, Any]:
    return {
        "cfg_signature": str(signature),
        "anchor_branch": str(anchor_branch),
        "proposal_id": _proposal_id_from_signature(signature),
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plan or train the FlowPre reseed round for top cfg signatures.")
    ap.add_argument("--dry-run", action="store_true", help="Write the reseed reports only.")
    ap.add_argument("--train", action="store_true", help="Train the missing cfg+seed runs after writing the reports.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true", help="Re-run planned cfg+seed pairs even if they already exist.")
    ap.add_argument("--quiet", action="store_true", help="Reduce trainer logs in --train mode.")
    return ap.parse_args()


def _mode_from_args(args: argparse.Namespace) -> str:
    if args.dry_run and args.train:
        raise RuntimeError("Use either --dry-run or --train, not both.")
    if args.train:
        return "train"
    return "dry-run"


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_no rows_"
    cols = list(frame.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = [
        "| " + " | ".join(str(row[col]) for col in cols) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, sep, *body])


def _seed_list(values: list[int] | tuple[int, ...] | pd.Series) -> list[int]:
    if isinstance(values, pd.Series):
        cleaned = [int(v) for v in values.dropna().tolist()]
    else:
        cleaned = [int(v) for v in values]
    return sorted(set(cleaned))


def _seed_list_str(values: list[int] | tuple[int, ...]) -> str:
    seeds = _seed_list(list(values))
    return "|".join(str(seed) for seed in seeds)


def _sorted_unique(values: pd.Series) -> str:
    cleaned = sorted({str(value) for value in values if pd.notna(value) and str(value) != ""})
    return "|".join(cleaned)


def _parse_signature(signature: str) -> dict[str, Any]:
    match = SIGNATURE_RE.fullmatch(str(signature))
    if not match:
        raise RuntimeError(f"Unparseable cfg_signature: {signature}")
    groups = match.groupdict()
    lr_token = str(groups["lr"])
    lr_value = {"1e-3": 1e-3, "1e-4": 1e-4, "1e-5": 1e-5}[lr_token]
    meanstd_on = groups["ms"] == "on"
    skewkurt_on = groups["sk"] == "on"
    spec = {
        "hidden_features": int(groups["hf"]),
        "num_layers": int(groups["layers"]),
        "affine_rq_ratio": [1, int(groups["rq"])],
        "final_rq_layers": int(groups["frq"]),
        "learning_rate": float(lr_value),
        "use_mean_penalty": bool(meanstd_on),
        "use_std_penalty": bool(meanstd_on),
        "use_skew_penalty": bool(skewkurt_on),
        "use_kurtosis_penalty": bool(skewkurt_on),
    }
    if _config_signature(spec) != signature:
        raise RuntimeError(f"Signature roundtrip mismatch for {signature}")
    spec.update(
        {
            "hf": str(groups["hf"]),
            "layers": str(groups["layers"]),
            "rq": str(groups["rq"]),
            "frq": str(groups["frq"]),
            "lr": lr_token,
            "ms": groups["ms"],
            "sk": groups["sk"],
        }
    )
    return spec


def _proposal_id_from_signature(signature: str) -> str:
    parsed = _parse_signature(signature)
    return (
        f"hf{parsed['hf']}_"
        f"l{parsed['layers']}_"
        f"rq{parsed['rq']}_"
        f"lr{parsed['lr']}_"
        f"ms{parsed['ms']}_"
        f"sk{parsed['sk']}"
    )


CANDIDATE_CONFIGS: list[dict[str, Any]] = [
    _candidate("hf256|l4|rq1x5|frq5|lr1e-3|mson|skoff", "rrmse"),
    _candidate("hf256|l4|rq1x3|frq3|lr1e-4|mson|skoff", "rrmse"),
    _candidate("hf192|l2|rq1x3|frq3|lr1e-3|mson|skoff", "rrmse"),
    _candidate("hf256|l4|rq1x5|frq5|lr1e-3|mson|skon", "rrmse"),
    _candidate("hf192|l3|rq1x6|frq6|lr1e-3|mson|skoff", "rrmse"),
    _candidate("hf192|l3|rq1x3|frq3|lr1e-3|mson|skoff", "rrmse"),
    _candidate("hf256|l2|rq1x3|frq3|lr1e-3|mson|skoff", "rrmse"),
    _candidate("hf256|l3|rq1x3|frq3|lr1e-3|mson|skoff", "rrmse"),
    _candidate("hf128|l2|rq1x6|frq6|lr1e-3|msoff|skoff", "fair"),
    _candidate("hf256|l3|rq1x3|frq3|lr1e-4|mson|skoff", "rrmse"),
    _candidate("hf128|l3|rq1x6|frq6|lr1e-3|msoff|skoff", "fair"),
    _candidate("hf192|l3|rq1x3|frq3|lr1e-4|mson|skoff", "rrmse"),
    _candidate("hf192|l3|rq1x6|frq6|lr1e-4|mson|skoff", "rrmse"),
    _candidate("hf256|l4|rq1x8|frq8|lr1e-3|mson|skoff", "rrmse"),
    _candidate("hf128|l3|rq1x5|frq5|lr1e-3|msoff|skoff", "fair"),
    _candidate("hf192|l3|rq1x8|frq8|lr1e-5|msoff|skoff", "mvn"),
    _candidate("hf192|l4|rq1x6|frq6|lr1e-4|mson|skoff", "rrmse"),
    _candidate("hf192|l2|rq1x6|frq6|lr1e-3|msoff|skoff", "fair"),
    _candidate("hf192|l2|rq1x6|frq6|lr1e-5|msoff|skoff", "mvn"),
    _candidate("hf192|l3|rq1x5|frq5|lr1e-4|mson|skoff", "rrmse"),
    _candidate("hf192|l2|rq1x8|frq8|lr1e-5|msoff|skoff", "mvn"),
    _candidate("hf128|l2|rq1x6|frq6|lr1e-3|msoff|skon", "fair"),
    _candidate("hf192|l2|rq1x8|frq8|lr1e-5|msoff|skon", "mvn"),
    _candidate("hf128|l2|rq1x8|frq8|lr1e-5|msoff|skoff", "mvn"),
    _candidate("hf256|l3|rq1x5|frq5|lr1e-4|mson|skoff", "rrmse"),
    _candidate("hf192|l2|rq1x5|frq5|lr1e-3|msoff|skoff", "fair"),
    _candidate("hf256|l2|rq1x5|frq5|lr1e-5|msoff|skoff", "fair"),
]


def _candidate_lookup() -> dict[str, dict[str, Any]]:
    lookup = {row["cfg_signature"]: dict(row) for row in CANDIDATE_CONFIGS}
    if len(lookup) != len(CANDIDATE_CONFIGS):
        raise RuntimeError("Duplicate cfg_signature detected in the fixed reseed candidate list.")
    return lookup


def _build_candidate_inventory(all_runs: pd.DataFrame) -> pd.DataFrame:
    lookup = _candidate_lookup()
    subset = all_runs[all_runs["cfg_signature"].isin(lookup)].copy()
    rows: list[dict[str, Any]] = []

    for slot, candidate in enumerate(CANDIDATE_CONFIGS, start=1):
        signature = str(candidate["cfg_signature"])
        anchor_branch = str(candidate["anchor_branch"])
        proposal_id = str(candidate["proposal_id"])
        cfg_runs = subset[subset["cfg_signature"] == signature].copy()

        observed_run_count = int(len(cfg_runs))
        observed_seeds = _seed_list(cfg_runs["seed"]) if not cfg_runs.empty else []
        observed_branches = sorted(set(cfg_runs["branch_id"].dropna().astype(str))) if not cfg_runs.empty else []
        observed_cfg_ids = sorted(set(cfg_runs["cfg_id"].dropna().astype(str))) if not cfg_runs.empty else []
        observed_run_ids = sorted(set(cfg_runs["run_id"].dropna().astype(str))) if not cfg_runs.empty else []

        if observed_branches and (len(observed_branches) != 1 or observed_branches[0] != anchor_branch):
            raise RuntimeError(
                f"cfg_signature {signature} appears with unexpected branches {observed_branches}; expected {anchor_branch}."
            )

        n_unique_seeds = int(len(observed_seeds))
        if n_unique_seeds >= len(TARGET_SEED_SET):
            missing_seeds: list[int] = []
        else:
            missing_seeds = [seed for seed in TARGET_SEED_SET if seed not in observed_seeds]
            missing_seeds = missing_seeds[: len(TARGET_SEED_SET) - n_unique_seeds]

        rows.append(
            {
                "candidate_slot": int(slot),
                "cfg_signature": signature,
                "anchor_branch": anchor_branch,
                "proposal_id": proposal_id,
                "observed_run_count": observed_run_count,
                "observed_seeds_list": observed_seeds,
                "observed_seeds": _seed_list_str(observed_seeds),
                "n_unique_seeds": n_unique_seeds,
                "observed_branches": "|".join(observed_branches),
                "observed_cfg_ids": "|".join(observed_cfg_ids),
                "observed_run_ids": "|".join(observed_run_ids),
                "target_seed_set_list": list(TARGET_SEED_SET),
                "target_seed_set": _seed_list_str(TARGET_SEED_SET),
                "missing_seeds_list": missing_seeds,
                "missing_seeds": _seed_list_str(missing_seeds),
                "n_missing_runs": int(len(missing_seeds)),
            }
        )

    return pd.DataFrame(rows).sort_values(["candidate_slot"]).reset_index(drop=True)


def _build_training_plan(candidate_inventory_df: pd.DataFrame) -> pd.DataFrame:
    plan_rows: list[dict[str, Any]] = []
    existing_pairs = {
        (str(row["cfg_signature"]), int(seed))
        for _, row in candidate_inventory_df.iterrows()
        for seed in row["observed_seeds_list"]
    }
    seen_pairs: set[tuple[str, int]] = set()

    for _, row in candidate_inventory_df.iterrows():
        for seed in row["missing_seeds_list"]:
            key = (str(row["cfg_signature"]), int(seed))
            if key in existing_pairs or key in seen_pairs:
                continue
            seen_pairs.add(key)
            run_id = f"flowprers1_{row['anchor_branch']}_tpv1_{row['proposal_id']}_seed{int(seed)}_v1"
            config_path = CONFIG_ROOT / str(row["anchor_branch"]) / f"{row['proposal_id']}.yaml"
            plan_rows.append(
                {
                    "candidate_slot": int(row["candidate_slot"]),
                    "cfg_signature": str(row["cfg_signature"]),
                    "anchor_branch": str(row["anchor_branch"]),
                    "proposal_id": str(row["proposal_id"]),
                    "seed": int(seed),
                    "planned_run_id": run_id,
                    "planned_config_path": str(config_path),
                    "planned_output_dir": str(OFFICIAL_FLOWPRE_ROOT / run_id),
                    "contract_id": RESEED_CONTRACT_ID,
                    "split_id": OFFICIAL_SPLIT_ID,
                    "uses_test": False,
                    "creates_promotion_manifest": False,
                }
            )

    if not plan_rows:
        return pd.DataFrame(
            columns=[
                "candidate_slot",
                "cfg_signature",
                "anchor_branch",
                "proposal_id",
                "seed",
                "planned_run_id",
                "planned_config_path",
                "planned_output_dir",
                "contract_id",
                "split_id",
                "uses_test",
                "creates_promotion_manifest",
            ]
        )

    return pd.DataFrame(plan_rows).sort_values(["candidate_slot", "seed"]).reset_index(drop=True)


def _report_inventory_frame(candidate_inventory_df: pd.DataFrame) -> pd.DataFrame:
    return candidate_inventory_df[
        [
            "candidate_slot",
            "cfg_signature",
            "anchor_branch",
            "proposal_id",
            "observed_run_count",
            "observed_seeds",
            "n_unique_seeds",
            "target_seed_set",
            "missing_seeds",
            "n_missing_runs",
            "observed_branches",
            "observed_cfg_ids",
            "observed_run_ids",
        ]
    ].copy()


def _summary_counts(candidate_inventory_df: pd.DataFrame) -> dict[str, int]:
    counts = candidate_inventory_df["n_missing_runs"].value_counts().to_dict()
    return {
        "complete_4plus": int(counts.get(0, 0)),
        "needs_1": int(counts.get(1, 0)),
        "needs_2": int(counts.get(2, 0)),
        "needs_3": int(counts.get(3, 0)),
        "needs_4": int(counts.get(4, 0)),
        "total_new_runs": int(candidate_inventory_df["n_missing_runs"].sum()),
    }


def _write_summary(
    *,
    inventory_df: pd.DataFrame,
    candidate_inventory_df: pd.DataFrame,
    training_plan_df: pd.DataFrame,
) -> Path:
    total_runs = int(len(inventory_df))
    counts = _summary_counts(candidate_inventory_df)
    candidates_ready = _report_inventory_frame(candidate_inventory_df)
    missing_only = candidates_ready[candidates_ready["n_missing_runs"] > 0].reset_index(drop=True)

    lines = [
        "# FlowPre Reseed Top CFGs V1 Summary",
        "",
        "## Estado de partida",
        f"- Runs oficiales observadas actualmente: `{total_runs}`.",
        f"- `cfg_signature` candidatas fijas: `{len(candidate_inventory_df)}`.",
        f"- Seeds comunes historicas reutilizadas: `{_seed_list_str(COMMON_SEEDS)}`.",
        f"- Seed nueva comun para completar el set de 4: `{TARGET_SEED_SET[-1]}`.",
        f"- Seed policy objetivo total por cfg: `{_seed_list_str(TARGET_SEED_SET)}`.",
        "",
        "## Resumen global de faltantes",
        f"- Configs ya completas a 4 seeds o mas: `{counts['complete_4plus']}`.",
        f"- Configs que necesitan 1 run mas: `{counts['needs_1']}`.",
        f"- Configs que necesitan 2 runs mas: `{counts['needs_2']}`.",
        f"- Configs que necesitan 3 runs mas: `{counts['needs_3']}`.",
        f"- Configs que necesitan 4 runs mas: `{counts['needs_4']}`.",
        f"- Total de runs nuevas planificadas: `{counts['total_new_runs']}`.",
        "",
        "## Inventario por cfg",
        _frame_to_markdown(
            candidates_ready[
                [
                    "candidate_slot",
                    "cfg_signature",
                    "observed_run_count",
                    "observed_seeds",
                    "n_unique_seeds",
                    "target_seed_set",
                    "missing_seeds",
                    "n_missing_runs",
                ]
            ]
        ),
        "",
        "## Training plan",
        _frame_to_markdown(
            training_plan_df[
                [
                    "candidate_slot",
                    "cfg_signature",
                    "seed",
                    "planned_run_id",
                ]
            ]
            if not training_plan_df.empty
            else pd.DataFrame(columns=["candidate_slot", "cfg_signature", "seed", "planned_run_id"])
        ),
        "",
        "## Comandos a ejecutar manualmente",
        "```bash",
        "caffeinate -dimsu env MPLCONFIGDIR=/tmp/codex-mpl .venv/bin/python scripts/f6_flowpre_reseed_topcfgs_v1.py --dry-run",
        "caffeinate -dimsu env MPLCONFIGDIR=/tmp/codex-mpl .venv/bin/python scripts/f6_flowpre_reseed_topcfgs_v1.py --train --device auto",
        "```",
        "",
        "- Este runner solo reseedea cfgs ya elegidas; no cambia la logica de evaluacion ni promociona nada.",
    ]
    out_path = REPORT_ROOT / "reseed_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_reports(
    *,
    candidate_inventory_df: pd.DataFrame,
    training_plan_df: pd.DataFrame,
    summary_path: Path,
) -> dict[str, Path]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    inventory_report_df = _report_inventory_frame(candidate_inventory_df)
    missing_plan_df = inventory_report_df[inventory_report_df["n_missing_runs"] > 0].reset_index(drop=True)

    paths = {
        "inventory": REPORT_ROOT / "reseed_inventory.csv",
        "missing_plan": REPORT_ROOT / "reseed_missing_plan.csv",
        "training_plan": REPORT_ROOT / "reseed_training_plan.csv",
        "summary": summary_path,
    }
    inventory_report_df.to_csv(paths["inventory"], index=False)
    missing_plan_df.to_csv(paths["missing_plan"], index=False)
    training_plan_df.to_csv(paths["training_plan"], index=False)
    return paths


def _write_train_configs(training_plan_df: pd.DataFrame) -> None:
    written: set[str] = set()
    for _, row in training_plan_df.iterrows():
        signature = str(row["cfg_signature"])
        if signature in written:
            continue
        written.add(signature)
        spec = _parse_signature(signature)
        config = _config_from_spec(str(row["anchor_branch"]), spec)
        config_path = CONFIG_ROOT / str(row["anchor_branch"]) / f"{row['proposal_id']}.yaml"
        write_yaml(config_path, config)


def _run_train_plan(training_plan_df: pd.DataFrame, *, device: str, verbose: bool, force: bool) -> None:
    from training.train_flow_pre import train_flowpre_pipeline

    if training_plan_df.empty:
        return

    _write_train_configs(training_plan_df)
    inventory_df = _build_inventory()
    all_runs = _build_current_all_runs(inventory_df)
    existing_pairs = {
        (str(row["cfg_signature"]), int(row["seed"]))
        for _, row in all_runs[all_runs["cfg_signature"].notna() & all_runs["seed"].notna()].iterrows()
    }

    for _, row in training_plan_df.iterrows():
        key = (str(row["cfg_signature"]), int(row["seed"]))
        if key in existing_pairs and not force:
            continue

        run_id = str(row["planned_run_id"])
        run_dir = OFFICIAL_FLOWPRE_ROOT / run_id
        results_path = run_dir / f"{run_id}_results.yaml"
        if results_path.exists() and not force:
            continue

        config_path = CONFIG_ROOT / str(row["anchor_branch"]) / f"{row['proposal_id']}.yaml"
        eval_ctx = {
            "dataset_name": DEFAULT_OFFICIAL_DATASET_NAME,
            "split_id": OFFICIAL_SPLIT_ID,
            "contract_id": RESEED_CONTRACT_ID,
            "seed_set_id": "f6_flowpre_reseed_topcfgs_v1",
            "base_config_id": f"flowpre_reseed_topcfgs_v1_{row['anchor_branch']}",
            "objective_metric_id": f"flowpre_{row['anchor_branch']}_selection",
            "run_level_axes": {
                "campaign_id": "f6_reseed_topcfgs_v1",
                "branch_id": str(row["anchor_branch"]),
                "cfg_id": str(row["proposal_id"]),
                "phase": "reseed",
                "seed": int(row["seed"]),
                "candidate_slot": int(row["candidate_slot"]),
            },
        }
        train_flowpre_pipeline(
            config_filename=str(config_path),
            base_name=run_id.removesuffix("_v1"),
            device=device,
            seed=int(row["seed"]),
            verbose=verbose,
            allow_test_holdout=False,
            evaluation_context=eval_ctx,
            output_namespace="official",
        )
        existing_pairs.add(key)


def build_reseed_plan_data() -> dict[str, Any]:
    inventory_df = _build_inventory()
    if inventory_df.empty:
        raise RuntimeError("No FlowPre official runs found under outputs/models/official/flow_pre.")
    all_runs = _build_current_all_runs(inventory_df)
    candidate_inventory_df = _build_candidate_inventory(all_runs)
    training_plan_df = _build_training_plan(candidate_inventory_df)
    return {
        "inventory_df": inventory_df,
        "all_runs": all_runs,
        "candidate_inventory_df": candidate_inventory_df,
        "training_plan_df": training_plan_df,
    }


def write_reseed_reports(plan: dict[str, Any]) -> dict[str, Path]:
    summary_path = _write_summary(
        inventory_df=plan["all_runs"],
        candidate_inventory_df=plan["candidate_inventory_df"],
        training_plan_df=plan["training_plan_df"],
    )
    return _write_reports(
        candidate_inventory_df=plan["candidate_inventory_df"],
        training_plan_df=plan["training_plan_df"],
        summary_path=summary_path,
    )


def main() -> int:
    args = _parse_args()
    mode = _mode_from_args(args)
    plan = build_reseed_plan_data()
    write_reseed_reports(plan)
    if mode == "train":
        _run_train_plan(
            plan["training_plan_df"],
            device=args.device,
            verbose=not args.quiet,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
