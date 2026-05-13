from __future__ import annotations

import argparse
import gc
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sets import DEFAULT_OFFICIAL_DATASET_NAME
from scripts.f6_common import write_yaml
from scripts.f6_flowpre_explore_v2 import _config_from_spec, _config_signature


OFFICIAL_FLOWPRE_ROOT = ROOT / "outputs" / "models" / "official" / "flow_pre"
V1_REPORT_ROOT = ROOT / "outputs" / "reports" / "f6_reseed_topcfgs_v1"
REPORT_ROOT = ROOT / "outputs" / "reports" / "f6_reseed_topcfgs_v3_resume"
CONFIG_ROOT = REPORT_ROOT / "configs" / "flowpre"
ORIGINAL_PLAN_PATH = V1_REPORT_ROOT / "reseed_training_plan.csv"
ORIGINAL_INVENTORY_PATH = V1_REPORT_ROOT / "reseed_inventory.csv"
RESUME_RUNNER_ID = "f6_flowpre_reseed_topcfgs_v3_resume"

SIGNATURE_RE = re.compile(
    r"hf(?P<hf>\d+)\|l(?P<layers>\d+)\|rq1x(?P<rq>\d+)\|frq(?P<frq>\d+)\|lr(?P<lr>[^|]+)\|ms(?P<ms>on|off)\|sk(?P<sk>on|off)"
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resume pending FlowPre reseed runs with fresh subprocesses per run.")
    ap.add_argument("--dry-run", action="store_true", help="Audit the real state and write resume reports without deleting or training.")
    ap.add_argument("--train", action="store_true", help="Clean incomplete runs, rebuild the real remaining plan, and train only pending runs.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quiet", action="store_true", help="Reduce trainer logs in child runs.")
    ap.add_argument("--run-one", default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def _mode_from_args(args: argparse.Namespace) -> str:
    selected = int(bool(args.dry_run)) + int(bool(args.train)) + int(bool(args.run_one))
    if selected != 1:
        raise RuntimeError("Use exactly one mode: --dry-run, --train, or --run-one.")
    if args.run_one:
        return "run-one"
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
    return "|".join(str(seed) for seed in _seed_list(values))


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
    return spec


def _core_artifact_paths(run_id: str) -> dict[str, Path]:
    run_dir = OFFICIAL_FLOWPRE_ROOT / run_id
    return {
        "run_dir": run_dir,
        "manifest": run_dir / f"{run_id}_run_manifest.json",
        "results": run_dir / f"{run_id}_results.yaml",
        "metrics_long": run_dir / f"{run_id}_metrics_long.csv",
        "snapshots_dir": run_dir / "snapshots",
    }


def _classify_run_row(row: pd.Series) -> dict[str, Any]:
    run_id = str(row["planned_run_id"])
    paths = _core_artifact_paths(run_id)
    run_dir_exists = paths["run_dir"].exists()
    manifest_exists = paths["manifest"].exists()
    results_exists = paths["results"].exists()
    metrics_exists = paths["metrics_long"].exists()
    snapshot_count = len(list(paths["snapshots_dir"].glob("*"))) if paths["snapshots_dir"].exists() else 0
    status = "completed" if (manifest_exists and results_exists and metrics_exists) else ("incomplete" if run_dir_exists else "pending")
    return {
        "candidate_slot": int(row["candidate_slot"]),
        "cfg_signature": str(row["cfg_signature"]),
        "anchor_branch": str(row["anchor_branch"]),
        "proposal_id": str(row["proposal_id"]),
        "seed": int(row["seed"]),
        "planned_run_id": run_id,
        "planned_output_dir": str(row["planned_output_dir"]),
        "planned_config_path": str(row["planned_config_path"]),
        "contract_id": str(row["contract_id"]),
        "split_id": str(row["split_id"]),
        "run_dir_exists": bool(run_dir_exists),
        "manifest_exists": bool(manifest_exists),
        "results_exists": bool(results_exists),
        "metrics_exists": bool(metrics_exists),
        "snapshot_count": int(snapshot_count),
        "status": status,
        "core_complete": bool(manifest_exists and results_exists and metrics_exists),
    }


def _load_original_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ORIGINAL_PLAN_PATH.exists():
        raise FileNotFoundError(f"Missing original reseed plan: {ORIGINAL_PLAN_PATH}")
    if not ORIGINAL_INVENTORY_PATH.exists():
        raise FileNotFoundError(f"Missing original reseed inventory: {ORIGINAL_INVENTORY_PATH}")

    plan_df = pd.read_csv(ORIGINAL_PLAN_PATH)
    inventory_df = pd.read_csv(ORIGINAL_INVENTORY_PATH)
    required_plan_cols = {
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
    }
    missing_cols = required_plan_cols.difference(plan_df.columns)
    if missing_cols:
        raise RuntimeError(f"Original plan is missing required columns: {sorted(missing_cols)}")
    if plan_df.duplicated(subset=["cfg_signature", "seed"]).any():
        dupes = plan_df.loc[plan_df.duplicated(subset=["cfg_signature", "seed"], keep=False), ["cfg_signature", "seed"]]
        raise RuntimeError(f"Original plan has duplicate cfg_signature+seed rows:\n{dupes}")
    return plan_df.copy(), inventory_df.copy()


def _audit_plan(plan_df: pd.DataFrame) -> pd.DataFrame:
    rows = [_classify_run_row(row) for _, row in plan_df.iterrows()]
    return pd.DataFrame(rows).sort_values(["candidate_slot", "seed"]).reset_index(drop=True)


def _simulate_post_cleanup_audit(current_audit_df: pd.DataFrame) -> pd.DataFrame:
    simulated = current_audit_df.copy()
    mask = simulated["status"] == "incomplete"
    simulated.loc[mask, "status"] = "pending"
    simulated.loc[mask, "run_dir_exists"] = False
    simulated.loc[mask, "manifest_exists"] = False
    simulated.loc[mask, "results_exists"] = False
    simulated.loc[mask, "metrics_exists"] = False
    simulated.loc[mask, "core_complete"] = False
    return simulated


def _cleanup_incomplete_runs(incomplete_df: pd.DataFrame) -> pd.DataFrame:
    cleaned_rows: list[dict[str, Any]] = []
    for _, row in incomplete_df.iterrows():
        run_dir = Path(str(row["planned_output_dir"]))
        if run_dir.exists():
            shutil.rmtree(run_dir)
        cleaned_rows.append(
            {
                "candidate_slot": int(row["candidate_slot"]),
                "cfg_signature": str(row["cfg_signature"]),
                "seed": int(row["seed"]),
                "planned_run_id": str(row["planned_run_id"]),
                "cleaned_run_dir": str(run_dir),
                "removed": True,
                "snapshot_count_before_cleanup": int(row["snapshot_count"]),
                "manifest_exists_before_cleanup": bool(row["manifest_exists"]),
                "results_exists_before_cleanup": bool(row["results_exists"]),
                "metrics_exists_before_cleanup": bool(row["metrics_exists"]),
            }
        )
    return pd.DataFrame(cleaned_rows)


def _build_cfg_inventory(plan_df: pd.DataFrame, effective_audit_df: pd.DataFrame, original_inventory_df: pd.DataFrame) -> pd.DataFrame:
    original_by_sig = (
        original_inventory_df[["cfg_signature", "target_seed_set", "observed_seeds"]]
        .drop_duplicates(subset=["cfg_signature"])
        .set_index("cfg_signature")
        .to_dict(orient="index")
    )
    target_by_sig = {
        signature: str(values.get("target_seed_set", ""))
        for signature, values in original_by_sig.items()
    }
    observed_by_sig = {
        signature: _seed_list(
            [int(token) for token in str(values.get("observed_seeds", "")).split("|") if token]
        )
        for signature, values in original_by_sig.items()
    }
    rows: list[dict[str, Any]] = []
    grouped = plan_df.groupby(["candidate_slot", "cfg_signature", "anchor_branch"], sort=False)
    for (slot, signature, anchor_branch), _group in grouped:
        completed_plan_seeds = sorted(
            effective_audit_df[
                (effective_audit_df["cfg_signature"] == signature) & (effective_audit_df["status"] == "completed")
            ]["seed"].astype(int).tolist()
        )
        completed_seeds = _seed_list(observed_by_sig.get(signature, []) + completed_plan_seeds)
        pending_seeds = sorted(
            effective_audit_df[
                (effective_audit_df["cfg_signature"] == signature) & (effective_audit_df["status"] == "pending")
            ]["seed"].astype(int).tolist()
        )
        rows.append(
            {
                "candidate_slot": int(slot),
                "cfg_signature": str(signature),
                "anchor_branch": str(anchor_branch),
                "target_seed_set": str(target_by_sig.get(signature, "")),
                "completed_seeds": _seed_list_str(completed_seeds),
                "pending_seeds": _seed_list_str(pending_seeds),
                "n_completed_runs": int(len(completed_seeds)),
                "n_pending_runs": int(len(pending_seeds)),
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate_slot"]).reset_index(drop=True)


def _build_resume_training_plan(plan_df: pd.DataFrame, effective_audit_df: pd.DataFrame) -> pd.DataFrame:
    pending_keys = {
        (str(row["cfg_signature"]), int(row["seed"]))
        for _, row in effective_audit_df[effective_audit_df["status"] == "pending"].iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, row in plan_df.iterrows():
        key = (str(row["cfg_signature"]), int(row["seed"]))
        if key in pending_keys:
            rows.append(dict(row))
    if not rows:
        return plan_df.head(0).copy()
    return pd.DataFrame(rows).sort_values(["candidate_slot", "seed"]).reset_index(drop=True)


def _write_train_configs(resume_plan_df: pd.DataFrame) -> None:
    written: set[str] = set()
    for _, row in resume_plan_df.iterrows():
        signature = str(row["cfg_signature"])
        if signature in written:
            continue
        written.add(signature)
        spec = _parse_signature(signature)
        config = _config_from_spec(str(row["anchor_branch"]), spec)
        config_path = CONFIG_ROOT / str(row["anchor_branch"]) / f"{row['proposal_id']}.yaml"
        write_yaml(config_path, config)


def _release_run_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass


def _build_plan_state(*, cleanup_mode: str) -> dict[str, Any]:
    plan_df, original_inventory_df = _load_original_inputs()
    current_audit_df = _audit_plan(plan_df)
    incomplete_detected_df = current_audit_df[current_audit_df["status"] == "incomplete"].reset_index(drop=True)

    cleaned_df = pd.DataFrame(
        columns=[
            "candidate_slot",
            "cfg_signature",
            "seed",
            "planned_run_id",
            "cleaned_run_dir",
            "removed",
            "snapshot_count_before_cleanup",
            "manifest_exists_before_cleanup",
            "results_exists_before_cleanup",
            "metrics_exists_before_cleanup",
        ]
    )

    if cleanup_mode == "perform" and not incomplete_detected_df.empty:
        cleaned_df = _cleanup_incomplete_runs(incomplete_detected_df)
        effective_audit_df = _audit_plan(plan_df)
    elif cleanup_mode == "perform":
        effective_audit_df = current_audit_df.copy()
    elif cleanup_mode == "simulate":
        effective_audit_df = _simulate_post_cleanup_audit(current_audit_df)
    elif cleanup_mode == "none":
        effective_audit_df = current_audit_df.copy()
    else:
        raise RuntimeError(f"Unsupported cleanup_mode: {cleanup_mode}")

    cfg_inventory_df = _build_cfg_inventory(plan_df, effective_audit_df, original_inventory_df)
    resume_plan_df = _build_resume_training_plan(plan_df, effective_audit_df)
    return {
        "plan_df": plan_df,
        "original_inventory_df": original_inventory_df,
        "current_audit_df": current_audit_df,
        "incomplete_detected_df": incomplete_detected_df,
        "cleaned_df": cleaned_df,
        "effective_audit_df": effective_audit_df,
        "cfg_inventory_df": cfg_inventory_df,
        "resume_plan_df": resume_plan_df,
    }


def _write_summary(*, mode: str, state: dict[str, Any]) -> Path:
    current_counts = state["current_audit_df"]["status"].value_counts().to_dict()
    effective_counts = state["effective_audit_df"]["status"].value_counts().to_dict()
    lines = [
        "# FlowPre Reseed Top CFGs V3 Resume Summary",
        "",
        "## Estado real auditado",
        f"- Runs planeadas en el plan original: `{len(state['plan_df'])}`.",
        f"- Estado actual detectado en filesystem: `{current_counts}`.",
        f"- Runs incompletas detectadas: `{len(state['incomplete_detected_df'])}`.",
        f"- Runs que quedarian tras limpieza de incompletas: `{effective_counts}`.",
        f"- Runs que lanzaria este runner: `{len(state['resume_plan_df'])}`.",
        "",
        "## Runs incompletas detectadas",
        _frame_to_markdown(
            state["incomplete_detected_df"][
                [
                    "candidate_slot",
                    "cfg_signature",
                    "seed",
                    "planned_run_id",
                    "snapshot_count",
                    "manifest_exists",
                    "results_exists",
                    "metrics_exists",
                ]
            ]
            if not state["incomplete_detected_df"].empty
            else pd.DataFrame(
                columns=[
                    "candidate_slot",
                    "cfg_signature",
                    "seed",
                    "planned_run_id",
                    "snapshot_count",
                    "manifest_exists",
                    "results_exists",
                    "metrics_exists",
                ]
            )
        ),
        "",
        "## Runs incompletas limpiadas en este modo",
        _frame_to_markdown(state["cleaned_df"]),
        "",
        "## Seeds por cfg_signature",
        _frame_to_markdown(state["cfg_inventory_df"]),
        "",
        "## Resume training plan",
        _frame_to_markdown(
            state["resume_plan_df"][["candidate_slot", "cfg_signature", "seed", "planned_run_id"]]
            if not state["resume_plan_df"].empty
            else pd.DataFrame(columns=["candidate_slot", "cfg_signature", "seed", "planned_run_id"])
        ),
        "",
        "## Modo",
        f"- Runner ejecutado en modo: `{mode}`.",
        f"- En `--dry-run` no se borra nada; en `--train` se limpian primero las incompletas detectadas.",
        "",
        "## Comandos a ejecutar manualmente",
        "```bash",
        "caffeinate -dimsu env MPLCONFIGDIR=/tmp/codex-mpl .venv/bin/python scripts/f6_flowpre_reseed_topcfgs_v3_resume.py --dry-run",
        "caffeinate -dimsu env MPLCONFIGDIR=/tmp/codex-mpl .venv/bin/python scripts/f6_flowpre_reseed_topcfgs_v3_resume.py --train --device auto",
        "```",
    ]
    out_path = REPORT_ROOT / "resume_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_reports(*, mode: str, state: dict[str, Any]) -> dict[str, Path]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    completed_df = state["current_audit_df"][state["current_audit_df"]["status"] == "completed"].reset_index(drop=True)
    paths = {
        "run_inventory": REPORT_ROOT / "resume_run_inventory.csv",
        "completed_runs": REPORT_ROOT / "resume_completed_runs.csv",
        "incomplete_detected": REPORT_ROOT / "resume_incomplete_detected.csv",
        "incomplete_cleaned": REPORT_ROOT / "resume_incomplete_cleaned.csv",
        "cfg_inventory": REPORT_ROOT / "resume_cfg_inventory.csv",
        "training_plan": REPORT_ROOT / "resume_training_plan.csv",
        "summary": _write_summary(mode=mode, state=state),
    }
    state["current_audit_df"].to_csv(paths["run_inventory"], index=False)
    completed_df.to_csv(paths["completed_runs"], index=False)
    state["incomplete_detected_df"].to_csv(paths["incomplete_detected"], index=False)
    state["cleaned_df"].to_csv(paths["incomplete_cleaned"], index=False)
    state["cfg_inventory_df"].to_csv(paths["cfg_inventory"], index=False)
    state["resume_plan_df"].to_csv(paths["training_plan"], index=False)
    return paths


def _plan_row_by_run_id(plan_df: pd.DataFrame, planned_run_id: str) -> pd.Series:
    matches = plan_df[plan_df["planned_run_id"].astype(str) == str(planned_run_id)]
    if matches.empty:
        raise RuntimeError(f"planned_run_id not found in original plan: {planned_run_id}")
    if len(matches) != 1:
        raise RuntimeError(f"planned_run_id appears multiple times in original plan: {planned_run_id}")
    return matches.iloc[0]


def _run_one(planned_run_id: str, *, device: str, verbose: bool) -> None:
    from training.train_flow_pre import train_flowpre_pipeline

    plan_df, _inventory_df = _load_original_inputs()
    row = _plan_row_by_run_id(plan_df, planned_run_id)
    paths = _core_artifact_paths(planned_run_id)
    if paths["manifest"].exists() and paths["results"].exists() and paths["metrics_long"].exists():
        return
    if paths["run_dir"].exists():
        raise RuntimeError(
            f"Run directory already exists for pending child run {planned_run_id}: {paths['run_dir']}. "
            "Refusing to create a silent _v2."
        )

    one_row_df = pd.DataFrame([dict(row)])
    _write_train_configs(one_row_df)
    config_path = CONFIG_ROOT / str(row["anchor_branch"]) / f"{row['proposal_id']}.yaml"
    eval_ctx = {
        "dataset_name": DEFAULT_OFFICIAL_DATASET_NAME,
        "split_id": str(row["split_id"]),
        "contract_id": str(row["contract_id"]),
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
            "resume_runner_id": RESUME_RUNNER_ID,
        },
    }

    model = train_flowpre_pipeline(
        config_filename=str(config_path),
        base_name=planned_run_id.removesuffix("_v1"),
        device=device,
        seed=int(row["seed"]),
        verbose=verbose,
        allow_test_holdout=False,
        evaluation_context=eval_ctx,
        output_namespace="official",
    )
    del model
    _release_run_memory()

    refreshed = _classify_run_row(row)
    if refreshed["status"] != "completed":
        raise RuntimeError(f"Run {planned_run_id} did not finish with all core artifacts.")


def _run_resume_plan(resume_plan_df: pd.DataFrame, *, device: str, quiet: bool) -> None:
    if resume_plan_df.empty:
        return

    _write_train_configs(resume_plan_df)
    script_path = Path(__file__).resolve()
    for _, row in resume_plan_df.iterrows():
        run_id = str(row["planned_run_id"])
        child_cmd = [
            sys.executable,
            str(script_path),
            "--run-one",
            run_id,
            "--device",
            str(device),
        ]
        if quiet:
            child_cmd.append("--quiet")
        result = subprocess.run(child_cmd, cwd=str(ROOT))
        _release_run_memory()
        if result.returncode != 0:
            raise RuntimeError(f"Child process failed for {run_id} with return code {result.returncode}.")


def _write_progress_reports(*, mode: str, cleaned_df: pd.DataFrame | None = None) -> dict[str, Path]:
    state = _build_plan_state(cleanup_mode="simulate")
    if cleaned_df is not None:
        state["cleaned_df"] = cleaned_df.copy()
    return _write_reports(mode=mode, state=state)


def main() -> int:
    args = _parse_args()
    mode = _mode_from_args(args)

    if mode == "run-one":
        _run_one(str(args.run_one), device=args.device, verbose=not args.quiet)
        return 0

    if mode == "dry-run":
        state = _build_plan_state(cleanup_mode="simulate")
        _write_reports(mode=mode, state=state)
        return 0

    pre_state = _build_plan_state(cleanup_mode="perform")
    _write_reports(mode=mode, state=pre_state)
    try:
        _run_resume_plan(pre_state["resume_plan_df"], device=args.device, quiet=args.quiet)
    except Exception:
        _write_progress_reports(mode=mode, cleaned_df=pre_state["cleaned_df"])
        raise
    _write_progress_reports(mode=mode, cleaned_df=pre_state["cleaned_df"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
