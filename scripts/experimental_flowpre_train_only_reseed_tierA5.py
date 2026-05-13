from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUTPUT_NAMESPACE = "experimental/train_only"
OUTPUT_SUBDIR = "reseed_tierA5"
CONTRACT_ID = "experimental_flowpre_train_only_reseed_tierA5_v1"
DEFAULT_OFFICIAL_DATASET_NAME = "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"

TRAINONLY_FLOWPRE_ROOT = ROOT / "outputs" / "models" / "experimental" / "train_only" / "flow_pre"
RUN_ROOT = TRAINONLY_FLOWPRE_ROOT / OUTPUT_SUBDIR
CONFIG_ROOT = TRAINONLY_FLOWPRE_ROOT / "configs" / OUTPUT_SUBDIR
REPORT_ROOT = ROOT / "outputs" / "reports" / "experimental" / "train_only" / "flowpre_reseed_tierA5"

# Used previously in FlowPre / FlowGen scripts and materialized artifacts at the
# time this runner was created. The new panel below intentionally avoids them.
PREVIOUSLY_USED_FLOW_SEEDS = (1117, 1234, 2221, 2468, 2898, 3331, 4447, 5678, 6769, 9101, 96024)
RESEED_SEEDS = (7319, 8423, 9547, 10627)

CORE_ARTIFACT_SUFFIXES = (
    ".pt",
    ".yaml",
    "_results.yaml",
    "_metrics_long.csv",
    "_run_manifest.json",
    "_influence.json",
    ".log",
)


@dataclass(frozen=True)
class TierASpec:
    slot: int
    source_subdir: str
    source_run_id: str
    cfg_signature: str
    ranking_position: int
    ranking_score: float
    overfit_risk: str
    note: str


TIER_A_SPECS: tuple[TierASpec, ...] = (
    TierASpec(
        slot=1,
        source_subdir="top20",
        source_run_id="flowpre_trainonly_top20_cfg11_hf256_l4_rq6_frq6_lr1e-3_mson_skoff_seed6769_v1",
        cfg_signature="hf256|l4|rq1x6|frq6|lr1e-3|mson|skoff",
        ranking_position=1,
        ranking_score=-0.417,
        overfit_risk="medium",
        note="Best raw train-only prior score; strong but worth reseeding because risk is medium.",
    ),
    TierASpec(
        slot=2,
        source_subdir="explore12",
        source_run_id="flowpre_trainonly_explore12_e09_hf224_l3_rq5_frq5_lr1e-3_mson_skoff_seed6769_v1",
        cfg_signature="hf224|l3|rq1x5|frq5|lr1e-3|mson|skoff",
        ranking_position=2,
        ranking_score=-0.403,
        overfit_risk="low",
        note="Cleanest new Tier A from explore12; strong balance and low overfit risk.",
    ),
    TierASpec(
        slot=3,
        source_subdir="top20",
        source_run_id="flowpre_trainonly_top20_cfg19_hf128_l4_rq6_frq6_lr1e-3_msoff_skoff_seed6769_v1",
        cfg_signature="hf128|l4|rq1x6|frq6|lr1e-3|msoff|skoff",
        ranking_position=3,
        ranking_score=-0.365,
        overfit_risk="medium",
        note="Tier A but diagnostic: excellent MVN signal with high logdet flags in the first seed.",
    ),
    TierASpec(
        slot=4,
        source_subdir="top20",
        source_run_id="flowpre_trainonly_top20_cfg03_hf256_l4_rq5_frq5_lr1e-3_mson_skoff_seed6769_v1",
        cfg_signature="hf256|l4|rq1x5|frq5|lr1e-3|mson|skoff",
        ranking_position=4,
        ranking_score=-0.351,
        overfit_risk="low",
        note="Balanced canonical-like train-only candidate; good anchor for seed robustness.",
    ),
    TierASpec(
        slot=5,
        source_subdir="top20",
        source_run_id="flowpre_trainonly_top20_cfg08_hf256_l4_rq5_frq5_lr1e-3_mson_skon_seed6769_v1",
        cfg_signature="hf256|l4|rq1x5|frq5|lr1e-3|mson|skon",
        ranking_position=5,
        ranking_score=-0.328,
        overfit_risk="medium",
        note="Skew/kurtosis-on coverage candidate; useful to test whether skon survives reseed.",
    ),
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Reseed the five FlowPre train-only Tier A configs. The runner audits existing outputs, "
            "cleans incomplete run directories only in --train mode, and executes pending runs in "
            "fresh subprocesses to avoid carrying model state in memory."
        )
    )
    ap.add_argument("--dry-run", action="store_true", help="Audit state and write reports without deleting or training.")
    ap.add_argument("--train", action="store_true", help="Clean incomplete runs and train only pending runs.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quiet", action="store_true", help="Reduce trainer stdout logging in child runs.")
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


def _validate_seed_panel() -> None:
    if len(set(RESEED_SEEDS)) != len(RESEED_SEEDS):
        raise RuntimeError(f"Duplicate reseed seeds are not allowed: {RESEED_SEEDS}")
    overlap = sorted(set(RESEED_SEEDS).intersection(PREVIOUSLY_USED_FLOW_SEEDS))
    if overlap:
        raise RuntimeError(f"Reseed seeds overlap with previously used FlowPre/FlowGen seeds: {overlap}")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML is not a mapping: {path}")
    if "model" not in loaded or not isinstance(loaded["model"], dict):
        raise ValueError(f"Config has no model mapping: {path}")
    if "training" not in loaded or not isinstance(loaded["training"], dict):
        raise ValueError(f"Config has no training mapping: {path}")
    return loaded


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def _nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _lr_slug(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric == 0.001:
        return "1e-3"
    if numeric == 0.0007:
        return "7e-4"
    if numeric == 0.0001:
        return "1e-4"
    if numeric == 0.00001:
        return "1e-5"
    return f"{numeric:g}"


def _cfg_signature(config: dict[str, Any]) -> str:
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    ratio = model_cfg.get("affine_rq_ratio") or []
    if len(ratio) >= 2:
        rq = f"rq{ratio[0]}x{ratio[1]}"
    else:
        rq = f"rq{model_cfg.get('final_rq_layers')}"
    mean_std = "mson" if train_cfg.get("use_mean_penalty") and train_cfg.get("use_std_penalty") else "msoff"
    skew_kurt = "skon" if train_cfg.get("use_skew_penalty") or train_cfg.get("use_kurtosis_penalty") else "skoff"
    return (
        f"hf{model_cfg.get('hidden_features')}"
        f"|l{model_cfg.get('num_layers')}"
        f"|{rq}"
        f"|frq{model_cfg.get('final_rq_layers')}"
        f"|lr{_lr_slug(train_cfg.get('learning_rate'))}"
        f"|{mean_std}"
        f"|{skew_kurt}"
    )


def _safe_cfg_slug(cfg_signature: str) -> str:
    return cfg_signature.replace("|", "_").replace("rq1x", "rq")


def _source_run_dir(spec: TierASpec) -> Path:
    return TRAINONLY_FLOWPRE_ROOT / spec.source_subdir / spec.source_run_id


def _source_config_path(spec: TierASpec) -> Path:
    return _source_run_dir(spec) / f"{spec.source_run_id}.yaml"


def _run_id(spec: TierASpec, seed: int) -> str:
    return f"flowpre_trainonly_reseed_tierA5_a{spec.slot:02d}_{_safe_cfg_slug(spec.cfg_signature)}_seed{seed}_v1"


def _run_dir(spec: TierASpec, seed: int) -> Path:
    return RUN_ROOT / _run_id(spec, seed)


def _config_path(spec: TierASpec, seed: int) -> Path:
    return CONFIG_ROOT / f"{_run_id(spec, seed)}.yaml"


def _build_reseed_config(source_config: dict[str, Any], *, seed: int) -> dict[str, Any]:
    config = copy.deepcopy(source_config)
    training = config.setdefault("training", {})
    training["seed"] = int(seed)
    training["save_states"] = False
    training["log_training"] = True
    training["save_results"] = True
    training["save_model"] = True
    config["seed"] = int(seed)
    return config


def _override_summary(seed: int) -> str:
    return (
        f"seed={int(seed)}; training.seed={int(seed)}; "
        "training.save_states=False; training.log_training=True; "
        "training.save_results=True; training.save_model=True"
    )


def _evaluation_context(spec: TierASpec, *, seed: int, base_cfg_signature: str) -> dict[str, Any]:
    seed_panel = "_".join(str(item) for item in RESEED_SEEDS)
    return {
        "dataset_name": DEFAULT_OFFICIAL_DATASET_NAME,
        "split_id": "init_temporal_processed_v1",
        "contract_id": CONTRACT_ID,
        "seed_set_id": f"flowpre_trainonly_reseed_tierA5_seeds{seed_panel}",
        "base_config_id": f"experimental_train_only_flowpre::{spec.source_run_id}",
        "objective_metric_id": "flowpre_trainonly_prior_rank_v1",
        "run_level_axes": {
            "phase": "experimental_train_only_reseed_tierA5",
            "tier": "A",
            "tier_a_slot": int(spec.slot),
            "source_ranking_position": int(spec.ranking_position),
            "source_ranking_score": float(spec.ranking_score),
            "cfg_signature": spec.cfg_signature,
            "base_cfg_signature": base_cfg_signature,
            "source_trainonly_run_id": spec.source_run_id,
            "source_trainonly_subdir": spec.source_subdir,
            "source_trainonly_seed": 6769,
            "source_overfit_risk": spec.overfit_risk,
            "source_note": spec.note,
            "seed": int(seed),
            "run_seed_source": "new_explicit_flowpre_trainonly_reseed_panel_v1",
            "overrides": _override_summary(seed),
        },
    }


def _plan_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in TIER_A_SPECS:
        source_config_path = _source_config_path(spec)
        base_cfg_signature = ""
        source_exists = source_config_path.exists()
        if source_exists:
            base_cfg_signature = _cfg_signature(_load_yaml(source_config_path))
        for seed in RESEED_SEEDS:
            run_id = _run_id(spec, seed)
            rows.append(
                {
                    "tier_a_slot": int(spec.slot),
                    "source_ranking_position": int(spec.ranking_position),
                    "source_ranking_score": float(spec.ranking_score),
                    "cfg_signature": spec.cfg_signature,
                    "base_cfg_signature": base_cfg_signature,
                    "source_run_id": spec.source_run_id,
                    "source_subdir": spec.source_subdir,
                    "source_config_path": str(source_config_path),
                    "source_config_exists": bool(source_exists),
                    "source_overfit_risk": spec.overfit_risk,
                    "source_note": spec.note,
                    "seed": int(seed),
                    "planned_run_id": run_id,
                    "planned_output_dir": str(_run_dir(spec, seed)),
                    "planned_config_path": str(_config_path(spec, seed)),
                    "contract_id": CONTRACT_ID,
                    "split_id": "init_temporal_processed_v1",
                    "output_namespace": OUTPUT_NAMESPACE,
                    "output_subdir": OUTPUT_SUBDIR,
                }
            )
    return rows


def _spec_by_slot(slot: int) -> TierASpec:
    matches = [spec for spec in TIER_A_SPECS if spec.slot == int(slot)]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown tier A slot: {slot}")
    return matches[0]


def _spec_seed_from_run_id(run_id: str) -> tuple[TierASpec, int]:
    for spec in TIER_A_SPECS:
        for seed in RESEED_SEEDS:
            if _run_id(spec, seed) == run_id:
                return spec, int(seed)
    raise RuntimeError(f"Run id not found in current Tier A reseed plan: {run_id}")


def _classify_plan_row(row: dict[str, Any]) -> dict[str, Any]:
    run_id = str(row["planned_run_id"])
    run_dir = Path(str(row["planned_output_dir"]))
    row = dict(row)
    row.update(
        {
            "run_dir_exists": bool(run_dir.exists()),
            "snapshot_count": len(list((run_dir / "snapshots").glob("*.pt"))) if (run_dir / "snapshots").exists() else 0,
            "manifest_exists": False,
            "results_exists": False,
            "metrics_exists": False,
            "model_exists": False,
            "config_exists": False,
            "influence_exists": False,
            "log_exists": False,
            "missing_artifacts": "",
            "status": "pending",
            "reason": "run_dir missing",
        }
    )
    if not run_dir.exists():
        return row

    missing: list[str] = []
    for suffix in CORE_ARTIFACT_SUFFIXES:
        artifact = run_dir / f"{run_id}{suffix}"
        exists = _nonempty_file(artifact)
        if suffix == "_run_manifest.json":
            row["manifest_exists"] = exists
        elif suffix == "_results.yaml":
            row["results_exists"] = exists
        elif suffix == "_metrics_long.csv":
            row["metrics_exists"] = exists
        elif suffix == ".pt":
            row["model_exists"] = exists
        elif suffix == ".yaml":
            row["config_exists"] = exists
        elif suffix == "_influence.json":
            row["influence_exists"] = exists
        elif suffix == ".log":
            row["log_exists"] = exists
        if not exists:
            missing.append(artifact.name)

    if not missing:
        results_path = run_dir / f"{run_id}_results.yaml"
        try:
            results = yaml.safe_load(results_path.read_text(encoding="utf-8")) or {}
            for key in ("best_epoch", "total_epochs", "seed", "train", "val", "monitoring"):
                if key not in results:
                    missing.append(f"results missing {key}")
            if int(results.get("seed")) != int(row["seed"]):
                missing.append("results seed mismatch")
            for split_key in ("train", "val"):
                split = results.get(split_key) or {}
                for metric_key in ("rrmse_recon", "rrmse_mean_whole", "rrmse_std_whole"):
                    if metric_key not in split:
                        missing.append(f"results.{split_key} missing {metric_key}")
            monitoring = results.get("monitoring") or {}
            if monitoring.get("policy") != "train_only":
                missing.append("results monitoring policy is not train_only")
        except Exception as exc:
            missing.append(f"results unreadable: {exc}")

        manifest_path = run_dir / f"{run_id}_run_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("run_id") != run_id:
                missing.append("manifest run_id mismatch")
            if int(manifest.get("seed")) != int(row["seed"]):
                missing.append("manifest seed mismatch")
            if manifest.get("contract_id") != CONTRACT_ID:
                missing.append("manifest contract_id mismatch")
            run_axes = manifest.get("run_level_axes") or {}
            if run_axes.get("monitoring_policy") != "train_only":
                missing.append("manifest monitoring_policy is not train_only")
            if run_axes.get("phase") != "experimental_train_only_reseed_tierA5":
                missing.append("manifest phase mismatch")
            if int(run_axes.get("tier_a_slot")) != int(row["tier_a_slot"]):
                missing.append("manifest tier_a_slot mismatch")
            if run_axes.get("cfg_signature") != row["cfg_signature"]:
                missing.append("manifest cfg_signature mismatch")
        except Exception as exc:
            missing.append(f"manifest unreadable: {exc}")

        metrics_path = run_dir / f"{run_id}_metrics_long.csv"
        try:
            with open(metrics_path, newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                row_count = sum(1 for _ in reader)
            if row_count < 2:
                missing.append("metrics_long has no data rows")
        except Exception as exc:
            missing.append(f"metrics unreadable: {exc}")

        config_path = run_dir / f"{run_id}.yaml"
        try:
            actual_signature = _cfg_signature(_load_yaml(config_path))
            if actual_signature != row["cfg_signature"]:
                missing.append(f"config signature mismatch: {actual_signature}")
        except Exception as exc:
            missing.append(f"config unreadable: {exc}")

    if missing:
        row["status"] = "incomplete"
        row["missing_artifacts"] = "; ".join(missing)
        row["reason"] = row["missing_artifacts"]
        return row

    row["status"] = "completed"
    row["missing_artifacts"] = ""
    row["reason"] = "all required artifacts present and parseable"
    return row


def _audit_plan(plan_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_classify_plan_row(row) for row in plan_rows]


def _simulate_post_cleanup(current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    simulated: list[dict[str, Any]] = []
    for row in current_rows:
        row = dict(row)
        if row["status"] == "incomplete":
            row.update(
                {
                    "status": "pending",
                    "run_dir_exists": False,
                    "manifest_exists": False,
                    "results_exists": False,
                    "metrics_exists": False,
                    "model_exists": False,
                    "config_exists": False,
                    "influence_exists": False,
                    "log_exists": False,
                    "snapshot_count": 0,
                    "missing_artifacts": "",
                    "reason": "would remove incomplete run_dir before training",
                }
            )
        simulated.append(row)
    return simulated


def _cleanup_incomplete_runs(incomplete_rows: list[dict[str, Any]], *, perform: bool) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in incomplete_rows:
        run_dir = Path(str(row["planned_output_dir"]))
        removed = False
        if perform and run_dir.exists():
            shutil.rmtree(run_dir)
            removed = True
        cleaned.append(
            {
                "tier_a_slot": int(row["tier_a_slot"]),
                "cfg_signature": str(row["cfg_signature"]),
                "seed": int(row["seed"]),
                "planned_run_id": str(row["planned_run_id"]),
                "cleaned_run_dir": str(run_dir),
                "removed": bool(removed),
                "snapshot_count_before_cleanup": int(row["snapshot_count"]),
                "manifest_exists_before_cleanup": bool(row["manifest_exists"]),
                "results_exists_before_cleanup": bool(row["results_exists"]),
                "metrics_exists_before_cleanup": bool(row["metrics_exists"]),
                "model_exists_before_cleanup": bool(row["model_exists"]),
                "influence_exists_before_cleanup": bool(row["influence_exists"]),
                "log_exists_before_cleanup": bool(row["log_exists"]),
                "reason": str(row["reason"]),
            }
        )
    return cleaned


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"completed": 0, "incomplete": 0, "pending": 0}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    return counts


def _build_cfg_inventory(effective_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in TIER_A_SPECS:
        spec_rows = [row for row in effective_rows if int(row["tier_a_slot"]) == spec.slot]
        completed = sorted(int(row["seed"]) for row in spec_rows if row["status"] == "completed")
        pending = sorted(int(row["seed"]) for row in spec_rows if row["status"] == "pending")
        incomplete = sorted(int(row["seed"]) for row in spec_rows if row["status"] == "incomplete")
        rows.append(
            {
                "tier_a_slot": int(spec.slot),
                "cfg_signature": spec.cfg_signature,
                "source_run_id": spec.source_run_id,
                "source_ranking_position": int(spec.ranking_position),
                "source_overfit_risk": spec.overfit_risk,
                "target_seed_set": "|".join(str(seed) for seed in RESEED_SEEDS),
                "completed_seeds": "|".join(str(seed) for seed in completed),
                "pending_seeds": "|".join(str(seed) for seed in pending),
                "incomplete_seeds": "|".join(str(seed) for seed in incomplete),
                "n_completed_runs": len(completed),
                "n_pending_runs": len(pending),
                "n_incomplete_runs": len(incomplete),
                "note": spec.note,
            }
        )
    return rows


def _build_plan_state(*, cleanup_mode: str) -> dict[str, Any]:
    plan = _plan_rows()
    current_audit = _audit_plan(plan)
    incomplete = [row for row in current_audit if row["status"] == "incomplete"]
    cleaned: list[dict[str, Any]] = []

    if cleanup_mode == "perform":
        cleaned = _cleanup_incomplete_runs(incomplete, perform=True)
        effective_audit = _audit_plan(plan)
    elif cleanup_mode == "simulate":
        cleaned = _cleanup_incomplete_runs(incomplete, perform=False)
        effective_audit = _simulate_post_cleanup(current_audit)
    elif cleanup_mode == "none":
        effective_audit = list(current_audit)
    else:
        raise RuntimeError(f"Unsupported cleanup_mode: {cleanup_mode}")

    training_plan = [row for row in effective_audit if row["status"] == "pending"]
    cfg_inventory = _build_cfg_inventory(effective_audit)
    return {
        "plan": plan,
        "current_audit": current_audit,
        "incomplete_detected": incomplete,
        "cleaned": cleaned,
        "effective_audit": effective_audit,
        "training_plan": training_plan,
        "cfg_inventory": cfg_inventory,
    }


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    normalized = [
        {key: _stable_json(value) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()}
        for row in rows
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _rows_to_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_no rows_"
    def cell(value: Any) -> str:
        return str(value).replace("\n", " ").replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(cell(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _write_summary(*, mode: str, state: dict[str, Any]) -> Path:
    current_counts = _status_counts(state["current_audit"])
    effective_counts = _status_counts(state["effective_audit"])
    training_rows = state["training_plan"]
    commands = [
        "caffeinate -dimsu env MPLCONFIGDIR=/tmp/codex-mpl .venv/bin/python scripts/experimental_flowpre_train_only_reseed_tierA5.py --dry-run",
        "caffeinate -dimsu env MPLCONFIGDIR=/tmp/codex-mpl .venv/bin/python scripts/experimental_flowpre_train_only_reseed_tierA5.py --train --device auto",
    ]
    lines = [
        "# FlowPre Train-Only Tier A5 Reseed Summary",
        "",
        "## Estado real auditado",
        f"- CFGs Tier A seleccionadas: `{len(TIER_A_SPECS)}`.",
        f"- Seeds nuevas por cfg: `{len(RESEED_SEEDS)}`.",
        f"- Runs planeadas: `{len(state['plan'])}`.",
        f"- Estado actual detectado: `{current_counts}`.",
        f"- Incompletas detectadas: `{len(state['incomplete_detected'])}`.",
        f"- Estado efectivo tras limpieza {'real' if mode == 'train' else 'simulada'}: `{effective_counts}`.",
        f"- Runs que lanzaria este runner ahora: `{len(training_rows)}`.",
        "",
        "## Seeds",
        f"- Seeds nuevas: `{', '.join(str(seed) for seed in RESEED_SEEDS)}`.",
        f"- Seeds evitadas por uso previo FlowPre/FlowGen: `{', '.join(str(seed) for seed in PREVIOUSLY_USED_FLOW_SEEDS)}`.",
        "",
        "## CFGs Tier A",
        _rows_to_markdown(
            [asdict(spec) for spec in TIER_A_SPECS],
            ["slot", "ranking_position", "cfg_signature", "overfit_risk", "source_run_id", "note"],
        ),
        "",
        "## Runs incompletas detectadas",
        _rows_to_markdown(
            state["incomplete_detected"],
            [
                "tier_a_slot",
                "cfg_signature",
                "seed",
                "planned_run_id",
                "snapshot_count",
                "manifest_exists",
                "results_exists",
                "metrics_exists",
                "reason",
            ],
        ),
        "",
        "## Runs incompletas limpiadas / marcadas",
        _rows_to_markdown(
            state["cleaned"],
            [
                "tier_a_slot",
                "cfg_signature",
                "seed",
                "planned_run_id",
                "removed",
                "snapshot_count_before_cleanup",
                "reason",
            ],
        ),
        "",
        "## Seeds por cfg",
        _rows_to_markdown(
            state["cfg_inventory"],
            [
                "tier_a_slot",
                "cfg_signature",
                "source_overfit_risk",
                "completed_seeds",
                "pending_seeds",
                "n_completed_runs",
                "n_pending_runs",
            ],
        ),
        "",
        "## Resume training plan",
        _rows_to_markdown(
            training_rows,
            ["tier_a_slot", "cfg_signature", "seed", "planned_run_id", "planned_output_dir"],
        ),
        "",
        "## Politica de ejecucion",
        "- Modo `--dry-run`: escribe auditoria y plan, no borra nada y no entrena.",
        "- Modo `--train`: borra solo directorios de runs planificadas que esten incompletas y entrena pendientes.",
        "- Cada run real se ejecuta en un subprocess aislado y se liberan caches de Python/Torch despues de cada hijo.",
        "- Las configs de reseed mantienen la arquitectura y loss de la fuente; solo cambian seed y `training.save_states=False` para evitar snapshots pesados.",
        "",
        "## Comandos",
        "```bash",
        *commands,
        "```",
    ]
    out_path = REPORT_ROOT / "summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_reports(*, mode: str, state: dict[str, Any]) -> dict[str, Path]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    completed = [row for row in state["current_audit"] if row["status"] == "completed"]
    paths = {
        "plan_json": REPORT_ROOT / "reseed_plan.json",
        "plan_csv": REPORT_ROOT / "reseed_plan.csv",
        "run_inventory": REPORT_ROOT / "reseed_run_inventory.csv",
        "completed_runs": REPORT_ROOT / "reseed_completed_runs.csv",
        "incomplete_detected": REPORT_ROOT / "reseed_incomplete_detected.csv",
        "incomplete_cleaned": REPORT_ROOT / "reseed_incomplete_cleaned.csv",
        "cfg_inventory": REPORT_ROOT / "reseed_cfg_inventory.csv",
        "training_plan": REPORT_ROOT / "reseed_training_plan.csv",
        "summary": _write_summary(mode=mode, state=state),
    }
    _write_json(
        paths["plan_json"],
        {
            "script": str(Path(__file__).resolve()),
            "contract_id": CONTRACT_ID,
            "output_root": str(RUN_ROOT),
            "config_root": str(CONFIG_ROOT),
            "report_root": str(REPORT_ROOT),
            "output_namespace": OUTPUT_NAMESPACE,
            "output_subdir": OUTPUT_SUBDIR,
            "reseed_seeds": list(RESEED_SEEDS),
            "previously_used_flow_seeds": list(PREVIOUSLY_USED_FLOW_SEEDS),
            "tier_a_specs": [asdict(spec) for spec in TIER_A_SPECS],
            "runs": state["plan"],
        },
    )
    _write_rows_csv(paths["plan_csv"], state["plan"])
    _write_rows_csv(paths["run_inventory"], state["current_audit"])
    _write_rows_csv(paths["completed_runs"], completed)
    _write_rows_csv(paths["incomplete_detected"], state["incomplete_detected"])
    _write_rows_csv(paths["incomplete_cleaned"], state["cleaned"])
    _write_rows_csv(paths["cfg_inventory"], state["cfg_inventory"])
    _write_rows_csv(paths["training_plan"], state["training_plan"])
    return paths


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


def _print_state_summary(title: str, rows: list[dict[str, Any]]) -> None:
    print("")
    print(title)
    print(f"  counts: {_status_counts(rows)}")
    for row in rows:
        print(f"  [A{int(row['tier_a_slot']):02d} seed{int(row['seed'])}] {row['status']:10s} {row['planned_run_id']}")
        if row["status"] != "completed":
            print(f"       reason: {row['reason']}")


def _write_train_config_for(spec: TierASpec, seed: int) -> Path:
    source_cfg = _load_yaml(_source_config_path(spec))
    source_signature = _cfg_signature(source_cfg)
    if source_signature != spec.cfg_signature:
        raise RuntimeError(
            f"Source config signature mismatch for {spec.source_run_id}: expected {spec.cfg_signature}, got {source_signature}"
        )
    config = _build_reseed_config(source_cfg, seed=seed)
    actual_signature = _cfg_signature(config)
    if actual_signature != spec.cfg_signature:
        raise RuntimeError(
            f"Reseed config signature mismatch for {spec.source_run_id}: expected {spec.cfg_signature}, got {actual_signature}"
        )
    return _write_yaml(_config_path(spec, seed), config)


def _run_one(run_id: str, *, device: str, quiet: bool) -> None:
    from training.train_flow_pre import train_flowpre_pipeline

    spec, seed = _spec_seed_from_run_id(run_id)
    row = _classify_plan_row(
        next(plan_row for plan_row in _plan_rows() if str(plan_row["planned_run_id"]) == run_id)
    )
    run_dir = _run_dir(spec, seed)
    if row["status"] == "completed":
        print(f"Run already complete: {run_id} -> {run_dir}")
        return
    if row["status"] == "incomplete" and run_dir.exists():
        print(f"Removing incomplete run dir before retry: {run_dir}")
        shutil.rmtree(run_dir)

    cfg_path = _write_train_config_for(spec, seed)
    base_cfg_signature = _cfg_signature(_load_yaml(_source_config_path(spec)))
    eval_ctx = _evaluation_context(spec, seed=seed, base_cfg_signature=base_cfg_signature)

    print("")
    print(f"Running FlowPre train-only Tier A reseed: {run_id}")
    print(f"  cfg_signature: {spec.cfg_signature}")
    print(f"  source: {_source_config_path(spec)}")
    print(f"  seed: {seed}")
    print(f"  config: {cfg_path}")
    print(f"  output_dir: {run_dir}")

    model = None
    try:
        model = train_flowpre_pipeline(
            config_filename=str(cfg_path),
            base_name=run_id.removesuffix("_v1"),
            device=device,
            seed=int(seed),
            verbose=not quiet,
            allow_test_holdout=False,
            evaluation_context=eval_ctx,
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
    finally:
        if model is not None:
            del model
        _release_run_memory()

    refreshed = _classify_plan_row(
        next(plan_row for plan_row in _plan_rows() if str(plan_row["planned_run_id"]) == run_id)
    )
    if refreshed["status"] != "completed":
        raise RuntimeError(f"Run {run_id} did not finish complete: {refreshed['reason']}")


def _child_failure_message(run_id: str, returncode: int) -> str:
    if returncode == -9:
        return f"Child process killed for {run_id} (returncode={returncode}, probable OOM / killed)."
    if returncode < 0:
        return f"Child process terminated by signal {-returncode} for {run_id} (returncode={returncode})."
    return f"Child process failed for {run_id} with return code {returncode}."


def _run_resume_plan(training_rows: list[dict[str, Any]], *, device: str, quiet: bool) -> None:
    if not training_rows:
        return

    for row in training_rows:
        spec = _spec_by_slot(int(row["tier_a_slot"]))
        _write_train_config_for(spec, int(row["seed"]))

    script_path = Path(__file__).resolve()
    for index, row in enumerate(training_rows, start=1):
        run_id = str(row["planned_run_id"])
        print("")
        print("-" * 100)
        print(f"[{index}/{len(training_rows)}] Queueing child run: {run_id}")
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
            raise RuntimeError(_child_failure_message(run_id, result.returncode))


def _write_progress_reports(*, mode: str, cleaned: list[dict[str, Any]] | None = None) -> dict[str, Path]:
    state = _build_plan_state(cleanup_mode="simulate")
    if cleaned is not None:
        state["cleaned"] = list(cleaned)
    return _write_reports(mode=mode, state=state)


def main() -> int:
    args = _parse_args()
    mode = _mode_from_args(args)
    _validate_seed_panel()

    if mode == "run-one":
        _run_one(str(args.run_one), device=args.device, quiet=args.quiet)
        return 0

    print("FlowPre train-only Tier A5 reseed runner")
    print(f"Output root: {RUN_ROOT}")
    print(f"Config root: {CONFIG_ROOT}")
    print(f"Report root: {REPORT_ROOT}")
    print(f"Reseed seeds: {', '.join(str(seed) for seed in RESEED_SEEDS)}")

    if mode == "dry-run":
        state = _build_plan_state(cleanup_mode="simulate")
        paths = _write_reports(mode=mode, state=state)
        _print_state_summary("Current state", state["current_audit"])
        print("")
        print(f"Dry run complete. Summary: {paths['summary']}")
        return 0

    pre_state = _build_plan_state(cleanup_mode="perform")
    paths = _write_reports(mode=mode, state=pre_state)
    _print_state_summary("State after cleanup", pre_state["effective_audit"])
    try:
        _run_resume_plan(pre_state["training_plan"], device=args.device, quiet=args.quiet)
    except Exception:
        _write_progress_reports(mode=mode, cleaned=pre_state["cleaned"])
        raise

    final_state = _build_plan_state(cleanup_mode="simulate")
    _write_reports(mode=mode, state=final_state)
    _print_state_summary("Final state", final_state["current_audit"])
    final_counts = _status_counts(final_state["current_audit"])
    if final_counts.get("incomplete", 0) or final_counts.get("pending", 0):
        raise RuntimeError(f"Tier A5 reseed still has unfinished runs after training: {final_counts}")
    print("")
    print(f"Tier A5 reseed complete. Summary: {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
