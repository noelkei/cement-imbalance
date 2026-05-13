from __future__ import annotations

import argparse
import csv
import copy
import gc
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUTPUT_NAMESPACE = "experimental/train_only"
OUTPUT_SUBDIR = "explore12"
CONFIG_ROOT = (
    ROOT
    / "outputs"
    / "models"
    / "experimental"
    / "train_only"
    / "flow_pre"
    / "configs"
    / "explore12"
)
OFFICIAL_FLOWPRE_ROOT = ROOT / "outputs" / "models" / "official" / "flow_pre"
CONTRACT_ID = "experimental_flowpre_train_only_explore12_v1"
DEFAULT_OFFICIAL_DATASET_NAME = "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"
RUN_SEED = 6769
CORE_ARTIFACT_SUFFIXES = (
    ".pt",
    ".yaml",
    "_results.yaml",
    "_metrics_long.csv",
    "_run_manifest.json",
)


@dataclass(frozen=True)
class ExploreSpec:
    rank: int
    explore_id: str
    cfg_signature: str
    source_run_id: str
    source_seed: int
    source_view: str
    source_rank: int | None
    overrides: tuple[tuple[str, object], ...] = ()


EXPLORE12_SPECS: tuple[ExploreSpec, ...] = (
    ExploreSpec(
        1,
        "E01",
        "hf192|l4|rq1x6|frq6|lr1e-3|mson|skoff",
        "flowprex3_rrmse_tpv1_fgr_hf192_l4_rq6_lr1e-3_mson_skoff_seed5678_v1",
        5678,
        "exact_official_local_gap",
        None,
    ),
    ExploreSpec(
        2,
        "E02",
        "hf224|l4|rq1x5|frq5|lr1e-3|mson|skoff",
        "flowpre_rrmse_tpv1_rq5_seed5678_v1",
        5678,
        "winner_interpolation",
        2,
        (("model.hidden_features", 224),),
    ),
    ExploreSpec(
        3,
        "E03",
        "hf224|l4|rq1x6|frq6|lr1e-3|mson|skoff",
        "flowprex3_rrmse_tpv1_fgp_hf256_l4_rq6_lr1e-3_mson_skoff_seed5678_v1",
        5678,
        "rq6_interpolation",
        10,
        (("model.hidden_features", 224),),
    ),
    ExploreSpec(
        4,
        "E04",
        "hf256|l4|rq1x4|frq4|lr1e-3|mson|skoff",
        "flowpre_rrmse_tpv1_rq5_seed5678_v1",
        5678,
        "rq4_underfit_probe",
        2,
        (("model.affine_rq_ratio", [1, 4]), ("model.final_rq_layers", 4)),
    ),
    ExploreSpec(
        5,
        "E05",
        "hf192|l4|rq1x4|frq4|lr1e-3|mson|skoff",
        "flowprex3_rrmse_tpv1_fgp_hf192_l4_rq5_lr1e-3_mson_skoff_seed5678_v1",
        5678,
        "rq4_stable_probe",
        12,
        (("model.affine_rq_ratio", [1, 4]), ("model.final_rq_layers", 4)),
    ),
    ExploreSpec(
        6,
        "E06",
        "hf224|l4|rq1x4|frq4|lr1e-3|mson|skoff",
        "flowpre_rrmse_tpv1_rq5_seed5678_v1",
        5678,
        "rq4_midwidth_probe",
        2,
        (
            ("model.hidden_features", 224),
            ("model.affine_rq_ratio", [1, 4]),
            ("model.final_rq_layers", 4),
        ),
    ),
    ExploreSpec(
        7,
        "E07",
        "hf256|l3|rq1x5|frq5|lr1e-3|mson|skoff",
        "flowprex3_rrmse_tpv1_fgr_hf256_l3_rq5_lr1e-3_mson_skoff_seed5678_v1",
        5678,
        "exact_official_depth_probe",
        None,
    ),
    ExploreSpec(
        8,
        "E08",
        "hf192|l3|rq1x5|frq5|lr1e-3|mson|skoff",
        "flowprex4_rrmse_tpv1_hf192_l3_rq6_lr1e-3_mson_skoff_seed5678_v1",
        5678,
        "depth_rq5_stable_probe",
        7,
        (("model.affine_rq_ratio", [1, 5]), ("model.final_rq_layers", 5)),
    ),
    ExploreSpec(
        9,
        "E09",
        "hf224|l3|rq1x5|frq5|lr1e-3|mson|skoff",
        "flowprex3_rrmse_tpv1_fgr_hf256_l3_rq5_lr1e-3_mson_skoff_seed5678_v1",
        5678,
        "depth_midwidth_probe",
        None,
        (("model.hidden_features", 224),),
    ),
    ExploreSpec(
        10,
        "E10",
        "hf192|l4|rq1x5|frq5|lr1e-3|mson|skon",
        "flowprex3_rrmse_tpv1_fgp_hf192_l4_rq5_lr1e-3_mson_skoff_seed5678_v1",
        5678,
        "skon_stable_probe",
        12,
        (("training.use_skew_penalty", True), ("training.use_kurtosis_penalty", True)),
    ),
    ExploreSpec(
        11,
        "E11",
        "hf224|l4|rq1x5|frq5|lr1e-3|mson|skon",
        "flowprex3_rrmse_tpv1_fgh_hf256_l4_rq5_lr1e-3_mson_skon_seed5678_v1",
        5678,
        "skon_midwidth_probe",
        3,
        (("model.hidden_features", 224),),
    ),
    ExploreSpec(
        12,
        "E12",
        "hf256|l4|rq1x6|frq6|lr7e-4|mson|skoff",
        "flowprex3_rrmse_tpv1_fgp_hf256_l4_rq6_lr1e-3_mson_skoff_seed5678_v1",
        5678,
        "lr_damping_probe",
        10,
        (("training.learning_rate", 0.0007),),
    ),
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run the experimental FlowPre train-only explore12 local campaign. "
            "Base configs are copied from nearby official FlowPre runs, lightly overridden, and trained with "
            "monitoring_policy='train_only'."
        )
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quiet", action="store_true", help="Reduce trainer stdout logging.")
    ap.add_argument("--dry-run", action="store_true", help="Print the planned runs without writing configs or training.")
    ap.add_argument("--run-one", type=int, choices=range(1, 13), default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--only-rank",
        type=int,
        choices=range(1, 13),
        action="append",
        metavar="N",
        help="Run only one or more explore12 ranks. Defaults to all 12.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Deprecated compatibility flag. Complete runs are always skipped by the resume logic.",
    )
    return ap.parse_args()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Base config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Base config is not a YAML mapping: {path}")
    if "model" not in loaded or not isinstance(loaded["model"], dict):
        raise ValueError(f"Base config has no model mapping: {path}")
    if "training" not in loaded or not isinstance(loaded["training"], dict):
        raise ValueError(f"Base config has no training mapping: {path}")
    return loaded


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


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


def _cfg_signature(config: dict) -> str:
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


def _source_config_path(spec: ExploreSpec) -> Path:
    return OFFICIAL_FLOWPRE_ROOT / spec.source_run_id / f"{spec.source_run_id}.yaml"


def _run_id(spec: ExploreSpec) -> str:
    return f"flowpre_trainonly_explore12_e{spec.rank:02d}_{_safe_cfg_slug(spec.cfg_signature)}_seed{RUN_SEED}_v1"


def _config_path(spec: ExploreSpec) -> Path:
    return CONFIG_ROOT / f"{_run_id(spec)}.yaml"


def _run_dir(spec: ExploreSpec) -> Path:
    return ROOT / "outputs" / "models" / OUTPUT_NAMESPACE / "flow_pre" / OUTPUT_SUBDIR / _run_id(spec)


def _apply_override(config: dict, dotted_path: str, value: object) -> None:
    current = config
    parts = dotted_path.split(".")
    for key in parts[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"Cannot apply override {dotted_path}: {key!r} is not a mapping.")
        current = child
    current[parts[-1]] = copy.deepcopy(value)


def _build_config(base_config: dict, spec: ExploreSpec) -> dict:
    config = copy.deepcopy(base_config)
    for dotted_path, value in spec.overrides:
        _apply_override(config, dotted_path, value)
    return config


def _override_summary(spec: ExploreSpec) -> str:
    if not spec.overrides:
        return "none"
    return "; ".join(f"{path}={value!r}" for path, value in spec.overrides)


def _config_for_completion_check(spec: ExploreSpec, run_id: str, run_dir: Path) -> dict:
    for path in (_config_path(spec), run_dir / f"{run_id}.yaml", _source_config_path(spec)):
        if path.exists():
            return _load_yaml(path)
    return {}


def _evaluation_context(spec: ExploreSpec, base_cfg_signature: str) -> dict:
    return {
        "dataset_name": DEFAULT_OFFICIAL_DATASET_NAME,
        "split_id": "init_temporal_processed_v1",
        "contract_id": CONTRACT_ID,
        "seed_set_id": f"flowpre_trainonly_explore12_seed{RUN_SEED}",
        "base_config_id": f"official_flowpre::{spec.source_run_id}",
        "objective_metric_id": "flowpre_trainonly_monitor_loss",
        "run_level_axes": {
            "phase": "experimental_train_only_explore12",
            "explore_id": spec.explore_id,
            "explore_rank": spec.rank,
            "cfg_signature": spec.cfg_signature,
            "base_cfg_signature": base_cfg_signature,
            "source_run_id": spec.source_run_id,
            "source_seed": spec.source_seed,
            "source_view": spec.source_view,
            "source_rank": spec.source_rank,
            "overrides": _override_summary(spec),
            "seed": RUN_SEED,
        },
    }


def _selected_specs(args: argparse.Namespace) -> tuple[ExploreSpec, ...]:
    if args.run_one is not None:
        return tuple(spec for spec in EXPLORE12_SPECS if spec.rank == int(args.run_one))
    if not args.only_rank:
        return EXPLORE12_SPECS
    requested = set(args.only_rank)
    return tuple(spec for spec in EXPLORE12_SPECS if spec.rank in requested)


def _validate_spec(spec: ExploreSpec, config: dict) -> None:
    actual = _cfg_signature(config)
    if actual != spec.cfg_signature:
        raise ValueError(
            "Derived config signature mismatch for "
            f"{spec.source_run_id}: expected {spec.cfg_signature}, got {actual}"
        )


def _inspect_run(spec: ExploreSpec) -> dict:
    run_id = _run_id(spec)
    run_dir = _run_dir(spec)
    row = {
        "rank": spec.rank,
        "cfg_signature": spec.cfg_signature,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": "not_started",
        "missing_artifacts": "",
        "reason": "run_dir missing",
    }
    if not run_dir.exists():
        return row

    missing: list[str] = []
    for suffix in CORE_ARTIFACT_SUFFIXES:
        artifact = run_dir / f"{run_id}{suffix}"
        if not _nonempty_file(artifact):
            missing.append(artifact.name)

    try:
        cfg = _config_for_completion_check(spec, run_id, run_dir)
        if (cfg.get("interpretability") or {}).get("save_influence"):
            influence_path = run_dir / f"{run_id}_influence.json"
            if not _nonempty_file(influence_path):
                missing.append(influence_path.name)
    except Exception as exc:
        missing.append(f"config unreadable: {exc}")

    if not missing:
        results_path = run_dir / f"{run_id}_results.yaml"
        try:
            results = yaml.safe_load(results_path.read_text(encoding="utf-8")) or {}
            for key in ("best_epoch", "total_epochs", "seed", "train", "val", "monitoring"):
                if key not in results:
                    missing.append(f"results missing {key}")
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
            if manifest.get("seed") != RUN_SEED:
                missing.append("manifest seed mismatch")
            run_axes = manifest.get("run_level_axes") or {}
            if run_axes.get("monitoring_policy") != "train_only":
                missing.append("manifest monitoring_policy is not train_only")
            if run_axes.get("explore_rank") != spec.rank:
                missing.append("manifest explore_rank mismatch")
            if run_axes.get("explore_id") != spec.explore_id:
                missing.append("manifest explore_id mismatch")
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

    if missing:
        row["status"] = "incomplete"
        row["missing_artifacts"] = "; ".join(missing)
        row["reason"] = row["missing_artifacts"]
        return row

    row["status"] = "complete"
    row["missing_artifacts"] = ""
    row["reason"] = "all required artifacts present and parseable"
    return row


def _audit_runs(specs: tuple[ExploreSpec, ...]) -> list[dict]:
    return [_inspect_run(spec) for spec in specs]


def _status_counts(audit_rows: list[dict]) -> dict[str, int]:
    counts = {"complete": 0, "incomplete": 0, "not_started": 0}
    for row in audit_rows:
        counts[row["status"]] += 1
    return counts


def _print_audit(title: str, audit_rows: list[dict]) -> None:
    print("")
    print(title)
    print(f"  counts: {_status_counts(audit_rows)}")
    for row in audit_rows:
        print(f"  [{row['rank']:02d}] {row['status']:12s} {row['run_id']}")
        if row["status"] != "complete":
            print(f"       reason: {row['reason']}")


def _cleanup_incomplete_runs(audit_rows: list[dict], *, perform: bool) -> list[dict]:
    cleaned: list[dict] = []
    for row in audit_rows:
        if row["status"] != "incomplete":
            continue
        run_dir = Path(row["run_dir"])
        removed = False
        if perform and run_dir.exists():
            shutil.rmtree(run_dir)
            removed = True
        cleaned.append({**row, "removed": removed})
    return cleaned


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


def _print_plan_header(specs: tuple[ExploreSpec, ...]) -> None:
    print("Experimental FlowPre train-only explore12 local campaign")
    print(f"Runs selected: {len(specs)}")
    print(f"Run seed: {RUN_SEED}")
    print(f"Output root: {ROOT / 'outputs' / 'models' / OUTPUT_NAMESPACE / 'flow_pre' / OUTPUT_SUBDIR}")
    print(f"Derived config root: {CONFIG_ROOT}")


def _print_spec_plan(spec: ExploreSpec, *, base_cfg_signature: str | None = None) -> None:
    run_id = _run_id(spec)
    print("")
    print(f"[{spec.rank:02d} / {spec.explore_id}] {run_id}")
    print(f"  cfg_signature: {spec.cfg_signature}")
    print(f"  source config: {_source_config_path(spec)}")
    if base_cfg_signature is not None:
        print(f"  base cfg_signature: {base_cfg_signature}")
    source_rank = "NA" if spec.source_rank is None else f"#{spec.source_rank}"
    print(f"  source view/rank: {spec.source_view} {source_rank}")
    print(f"  source seed: {spec.source_seed}")
    print(f"  run seed: {RUN_SEED}")
    print(f"  config overrides: {_override_summary(spec)}")
    print("  execution overrides: monitoring_policy='train_only'; experimental namespace/output naming")
    print(f"  derived config: {_config_path(spec)}")
    print(f"  run dir: {_run_dir(spec)}")


def _train_one(spec: ExploreSpec, *, device: str, quiet: bool) -> None:
    source_config = _source_config_path(spec)
    base_config = _load_yaml(source_config)
    base_cfg_signature = _cfg_signature(base_config)
    derived_config = _build_config(base_config, spec)
    _validate_spec(spec, derived_config)

    run_id = _run_id(spec)
    cfg_path = _config_path(spec)
    run_dir = _run_dir(spec)

    current = _inspect_run(spec)
    if current["status"] == "complete":
        print(f"[{spec.rank:02d}] skip_existing_complete {run_id}")
        return
    if current["status"] == "incomplete" and run_dir.exists():
        print(f"[{spec.rank:02d}] removing incomplete run dir before retry: {run_dir}")
        shutil.rmtree(run_dir)

    _print_spec_plan(spec, base_cfg_signature=base_cfg_signature)
    _write_yaml(cfg_path, derived_config)

    model = None
    try:
        from training.train_flow_pre import train_flowpre_pipeline

        model = train_flowpre_pipeline(
            config_filename=str(cfg_path),
            base_name=run_id.removesuffix("_v1"),
            device=device,
            seed=RUN_SEED,
            verbose=not quiet,
            allow_test_holdout=False,
            evaluation_context=_evaluation_context(spec, base_cfg_signature),
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
        del model
        _release_run_memory()

    refreshed = _inspect_run(spec)
    if refreshed["status"] != "complete":
        raise RuntimeError(f"Run {run_id} did not finish complete: {refreshed['reason']}")


def _run_child(spec: ExploreSpec, *, device: str, quiet: bool) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-one",
        str(spec.rank),
        "--device",
        str(device),
    ]
    if quiet:
        cmd.append("--quiet")
    result = subprocess.run(cmd, cwd=str(ROOT))
    _release_run_memory()
    if result.returncode != 0:
        raise RuntimeError(f"Child process failed for rank {spec.rank:02d} with return code {result.returncode}.")

    refreshed = _inspect_run(spec)
    if refreshed["status"] != "complete":
        raise RuntimeError(f"Rank {spec.rank:02d} did not finish complete: {refreshed['reason']}")


def main() -> None:
    args = _parse_args()
    specs = _selected_specs(args)

    if args.run_one is not None:
        if not specs:
            raise RuntimeError(f"Unknown run-one rank: {args.run_one}")
        _train_one(specs[0], device=args.device, quiet=args.quiet)
        return

    _print_plan_header(specs)
    for spec in specs:
        source_config = _source_config_path(spec)
        base_config = _load_yaml(source_config)
        base_cfg_signature = _cfg_signature(base_config)
        derived_config = _build_config(base_config, spec)
        _validate_spec(spec, derived_config)
        _print_spec_plan(spec, base_cfg_signature=base_cfg_signature)

    initial_audit = _audit_runs(specs)
    _print_audit("Initial campaign state", initial_audit)

    cleaned = _cleanup_incomplete_runs(initial_audit, perform=not args.dry_run)
    if cleaned:
        print("")
        print("Incomplete run dirs marked for cleanup")
        for row in cleaned:
            action = "removed" if row["removed"] else "would_remove"
            print(f"  [{row['rank']:02d}] {action}: {row['run_dir']}")

    if args.dry_run:
        pending = [row for row in initial_audit if row["status"] != "complete"]
        print("")
        print(f"Dry run: would execute {len(pending)} run(s) after cleaning incomplete dirs.")
        return

    post_cleanup_audit = _audit_runs(specs)
    _print_audit("State after cleanup", post_cleanup_audit)
    pending_specs = [spec for spec in specs if _inspect_run(spec)["status"] != "complete"]

    if not pending_specs:
        print("")
        print("Nothing to do: all selected runs are already complete.")
        return

    print("")
    print(f"Resume queue: {len(pending_specs)} run(s)")
    for spec in pending_specs:
        print(f"  [{spec.rank:02d}] {_run_id(spec)}")

    for spec in pending_specs:
        _run_child(spec, device=args.device, quiet=args.quiet)

    final_audit = _audit_runs(specs)
    _print_audit("Final campaign state", final_audit)
    counts = _status_counts(final_audit)
    if counts["incomplete"] or counts["not_started"]:
        raise RuntimeError(f"Campaign still has unfinished runs after resume: {counts}")


if __name__ == "__main__":
    main()
