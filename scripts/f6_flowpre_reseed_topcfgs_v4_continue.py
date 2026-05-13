from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

import scripts.f6_flowpre_reseed_topcfgs_v3_resume as base


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "reports" / "f6_reseed_topcfgs_v4_continue"
CONFIG_ROOT = REPORT_ROOT / "configs" / "flowpre"
RUNNER_ID = "f6_flowpre_reseed_topcfgs_v4_continue"

base.REPORT_ROOT = REPORT_ROOT
base.CONFIG_ROOT = CONFIG_ROOT
base.RESUME_RUNNER_ID = RUNNER_ID


def _write_summary(*, mode: str, state: dict) -> Path:
    current_counts = state["current_audit_df"]["status"].value_counts().to_dict()
    effective_counts = state["effective_audit_df"]["status"].value_counts().to_dict()
    lines = [
        "# FlowPre Reseed Top CFGs V4 Continue Summary",
        "",
        "## Estado real auditado",
        f"- Runs planeadas en el plan original: `{len(state['plan_df'])}`.",
        f"- Estado actual detectado en filesystem: `{current_counts}`.",
        f"- Runs incompletas detectadas: `{len(state['incomplete_detected_df'])}`.",
        f"- Runs que quedarian tras limpieza de incompletas: `{effective_counts}`.",
        f"- Runs que lanzaria este runner: `{len(state['resume_plan_df'])}`.",
        "",
        "## Runs incompletas detectadas",
        base._frame_to_markdown(
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
        base._frame_to_markdown(state["cleaned_df"]),
        "",
        "## Seeds por cfg_signature",
        base._frame_to_markdown(state["cfg_inventory_df"]),
        "",
        "## Resume training plan",
        base._frame_to_markdown(
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
        "caffeinate -dimsu env MPLCONFIGDIR=/tmp/codex-mpl .venv/bin/python scripts/f6_flowpre_reseed_topcfgs_v4_continue.py --dry-run",
        "caffeinate -dimsu env MPLCONFIGDIR=/tmp/codex-mpl .venv/bin/python scripts/f6_flowpre_reseed_topcfgs_v4_continue.py --train --device auto",
        "```",
    ]
    out_path = REPORT_ROOT / "resume_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_reports(*, mode: str, state: dict) -> dict[str, Path]:
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


def _write_progress_reports(*, mode: str, cleaned_df: pd.DataFrame | None = None) -> dict[str, Path]:
    state = base._build_plan_state(cleanup_mode="simulate")
    if cleaned_df is not None:
        state["cleaned_df"] = cleaned_df.copy()
    return _write_reports(mode=mode, state=state)


def _run_resume_plan(resume_plan_df: pd.DataFrame, *, device: str, quiet: bool) -> None:
    if resume_plan_df.empty:
        return

    base._write_train_configs(resume_plan_df)
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
        base._release_run_memory()
        if result.returncode != 0:
            raise RuntimeError(f"Child process failed for {run_id} with return code {result.returncode}.")


def main() -> int:
    args = base._parse_args()
    mode = base._mode_from_args(args)

    if mode == "run-one":
        base._run_one(str(args.run_one), device=args.device, verbose=not args.quiet)
        return 0

    if mode == "dry-run":
        state = base._build_plan_state(cleanup_mode="simulate")
        _write_reports(mode=mode, state=state)
        return 0

    pre_state = base._build_plan_state(cleanup_mode="perform")
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
