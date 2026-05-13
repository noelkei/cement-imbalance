from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.f6_common import OFFICIAL_SPLIT_ID, load_json, load_yaml


MODEL_FAMILY = "flowgen"
OUTPUT_NAMESPACE = "official"
OUTPUT_SUBDIR = "reseed_final"
CONTRACT_ID = "f6_flowgen_reseed_final_v1"
OBJECTIVE_METRIC_ID = "flowgen_realism_official"
DEFAULT_OFFICIAL_DATASET_NAME = "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"

OFFICIAL_FLOWGEN_ROOT = ROOT / "outputs" / "models" / OUTPUT_NAMESPACE / MODEL_FAMILY
SUMMARY_ROOT = OFFICIAL_FLOWGEN_ROOT / "campaign_summaries"

SOURCE_RUN_IDS = [
    "flowgen_tpv1_c2_train_s01_e38_softclip_seed2468_v2",
    "flowgen_tpv1_c2_train_h01_bridge300_lowmmd_seed2468_v2",
    "flowgen_tpv1_c2_train_k01_e36_ksy_seed2468_v2",
    "flowgen_tpv1_c2_train_h02_bridge500_lowmmd_seed2468_v2",
    "flowgen_tpv1_c2_train_e03_seed2468_v1",
]

# Fixed, explicit reseed panel for every selected source run.
RESEED_SEEDS = [1117, 2221, 3331, 4447]


@dataclass
class SourceRunContext:
    run_id: str
    run_dir: Path
    config_path: Path
    results_path: Path
    metrics_long_path: Path
    run_manifest_path: Path
    checkpoint_path: Path
    config: dict[str, Any]
    results: dict[str, Any]
    run_manifest: dict[str, Any]
    base_token: str
    source_seed: int
    source_version: str
    policy_slug: str
    source_policy_id: str | None
    source_policy_origin: str | None
    source_policy_signature: str | None
    historical_source_run_ids: list[str]
    base_flowgen_run_id: str
    base_flowgen_seed: int | None
    base_flowgen_work_base_id: str | None
    base_flowgen_checkpoint_path: Path
    paired_flowpre_source_id: str | None
    paired_flowpre_run_id: str | None
    paired_flowpre_seed: int | None


@dataclass
class PlanEntry:
    source_run_id: str
    source_version: str
    source_policy_id: str | None
    source_seed: int
    reseed_seed: int
    run_id: str
    output_dir: str
    existing_status: str
    status: str = "planned"
    error: str = ""
    result_paths: dict[str, str] = field(default_factory=dict)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Official FlowGen final reseed runner. It reuses already selected official FlowGen "
            "source runs, keeps the same effective config, and changes only the training seed."
        )
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--base-run", dest="source_run_ids", action="append", choices=SOURCE_RUN_IDS, default=None)
    ap.add_argument("--run-one", default=None, help="Execute exactly one planned reseed run id.")
    ap.add_argument("--summary-only", action="store_true", help="Build and print the plan without training.")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def _campaign_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _release_process_memory() -> None:
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


def _official_raw_bundle_manifest_path(
    *,
    split_id: str = OFFICIAL_SPLIT_ID,
    dataset_name: str = DEFAULT_OFFICIAL_DATASET_NAME,
) -> Path:
    return ROOT / "data" / "sets" / "official" / split_id / "raw" / dataset_name / "manifest.json"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _resolve_required_artifact(run_dir: Path, run_id: str, kind: str) -> Path:
    if kind == "config":
        candidates = [run_dir / "config.yaml", run_dir / f"{run_id}.yaml"]
    elif kind == "results":
        candidates = [run_dir / "results.yaml", run_dir / f"{run_id}_results.yaml"]
    elif kind == "metrics_long":
        candidates = [run_dir / "metrics_long.csv", run_dir / f"{run_id}_metrics_long.csv"]
    elif kind == "run_manifest":
        candidates = [run_dir / "run_manifest.json", run_dir / f"{run_id}_run_manifest.json"]
    elif kind == "checkpoint":
        candidates = [run_dir / "checkpoint.pt", run_dir / f"{run_id}.pt"]
    else:
        raise ValueError(f"Unsupported artifact kind: {kind}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing required source artifact '{kind}' under {run_dir}.")


def _parse_source_run_id(run_id: str) -> tuple[str, str, int, str]:
    match = re.match(r"^flowgen_tpv1_(c[12])_train_(.+)_seed(\d+)_v(\d+)$", str(run_id))
    if match is None:
        raise ValueError(f"Unsupported FlowGen source run id format: {run_id}")
    base_token = str(match.group(1))
    policy_slug = str(match.group(2))
    source_seed = int(match.group(3))
    source_version = f"v{match.group(4)}"
    return base_token, policy_slug, source_seed, source_version


def _resolve_base_flowgen_checkpoint(
    *,
    run_id: str,
    config: dict[str, Any],
    run_manifest: dict[str, Any],
) -> tuple[str, Path]:
    official_training = config.get("official_training") or {}
    run_axes = run_manifest.get("run_level_axes") or {}

    base_run_id = str(
        official_training.get("base_run_id")
        or run_axes.get("flowgen_base_run_id")
        or ""
    ).strip()
    if not base_run_id:
        raise ValueError(f"Unable to resolve base FlowGen run id for {run_id}.")

    base_checkpoint_raw = official_training.get("base_checkpoint")
    if base_checkpoint_raw not in (None, ""):
        base_checkpoint_path = Path(str(base_checkpoint_raw))
    else:
        base_checkpoint_path = OFFICIAL_FLOWGEN_ROOT / "bases" / base_run_id / "checkpoint.pt"
    if not base_checkpoint_path.exists():
        raise FileNotFoundError(f"Base FlowGen checkpoint not found for {run_id}: {base_checkpoint_path}")

    return base_run_id, base_checkpoint_path


def _resolve_source_context(run_id: str) -> SourceRunContext:
    run_dir = OFFICIAL_FLOWGEN_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Official FlowGen source run not found: {run_dir}")

    config_path = _resolve_required_artifact(run_dir, run_id, "config")
    results_path = _resolve_required_artifact(run_dir, run_id, "results")
    metrics_long_path = _resolve_required_artifact(run_dir, run_id, "metrics_long")
    run_manifest_path = _resolve_required_artifact(run_dir, run_id, "run_manifest")
    checkpoint_path = _resolve_required_artifact(run_dir, run_id, "checkpoint")

    config = load_yaml(config_path)
    results = load_yaml(results_path)
    run_manifest = load_json(run_manifest_path)

    if str(run_manifest.get("model_family")) != MODEL_FAMILY:
        raise ValueError(f"Source run manifest must have model_family='{MODEL_FAMILY}': {run_manifest_path}")
    if str(run_manifest.get("split_id")) != OFFICIAL_SPLIT_ID:
        raise ValueError(
            f"Source run manifest must have split_id='{OFFICIAL_SPLIT_ID}': {run_manifest_path}"
        )
    if bool(run_manifest.get("test_enabled")):
        raise ValueError(f"Source run manifest unexpectedly enables test holdout: {run_manifest_path}")
    if "model" not in config or not isinstance(config["model"], dict):
        raise ValueError(f"Source config missing model block: {config_path}")
    if "training" not in config or not isinstance(config["training"], dict):
        raise ValueError(f"Source config missing training block: {config_path}")

    base_token, policy_slug, source_seed_from_id, source_version = _parse_source_run_id(run_id)
    config_seed = config.get("seed")
    training_seed = (config.get("training") or {}).get("seed")
    seed_candidates = {int(value) for value in [source_seed_from_id, config_seed, training_seed] if value is not None}
    if len(seed_candidates) != 1:
        detail = {
            "run_id_seed": source_seed_from_id,
            "config.seed": config_seed,
            "training.seed": training_seed,
        }
        raise ValueError(f"Inconsistent source seed values for {run_id}: {detail}")
    source_seed = seed_candidates.pop()

    official_training = config.get("official_training") or {}
    run_axes = run_manifest.get("run_level_axes") or {}

    base_run_id, base_checkpoint_path = _resolve_base_flowgen_checkpoint(
        run_id=run_id,
        config=config,
        run_manifest=run_manifest,
    )

    base_flowgen_seed_raw = official_training.get("base_seed") or run_axes.get("flowgen_base_seed")
    base_flowgen_seed = None if base_flowgen_seed_raw is None else int(base_flowgen_seed_raw)
    base_work_base_id = (
        official_training.get("base_work_base_id")
        or run_axes.get("flowgen_base_work_base_id")
    )
    paired_flowpre_source_id = (
        official_training.get("paired_flowpre_source_id")
        or run_axes.get("paired_flowpre_source_id")
    )
    paired_flowpre_run_id = (
        official_training.get("paired_flowpre_run_id")
        or run_axes.get("paired_flowpre_run_id")
    )
    paired_flowpre_seed_raw = (
        official_training.get("paired_flowpre_seed")
        or run_axes.get("paired_flowpre_seed")
    )
    paired_flowpre_seed = None if paired_flowpre_seed_raw is None else int(paired_flowpre_seed_raw)

    source_policy_id = official_training.get("policy_id") or run_axes.get("policy_id")
    source_policy_origin = official_training.get("policy_origin") or run_axes.get("policy_origin")
    source_policy_signature = official_training.get("policy_signature") or run_axes.get("policy_signature")
    historical_source_run_ids = (
        official_training.get("historical_source_run_ids")
        or run_axes.get("historical_source_run_ids")
        or []
    )

    return SourceRunContext(
        run_id=run_id,
        run_dir=run_dir,
        config_path=config_path,
        results_path=results_path,
        metrics_long_path=metrics_long_path,
        run_manifest_path=run_manifest_path,
        checkpoint_path=checkpoint_path,
        config=config,
        results=results,
        run_manifest=run_manifest,
        base_token=base_token,
        source_seed=source_seed,
        source_version=source_version,
        policy_slug=policy_slug,
        source_policy_id=None if source_policy_id in (None, "") else str(source_policy_id),
        source_policy_origin=None if source_policy_origin in (None, "") else str(source_policy_origin),
        source_policy_signature=None if source_policy_signature in (None, "") else str(source_policy_signature),
        historical_source_run_ids=[str(item) for item in historical_source_run_ids],
        base_flowgen_run_id=base_run_id,
        base_flowgen_seed=base_flowgen_seed,
        base_flowgen_work_base_id=None if base_work_base_id in (None, "") else str(base_work_base_id),
        base_flowgen_checkpoint_path=base_checkpoint_path,
        paired_flowpre_source_id=None if paired_flowpre_source_id in (None, "") else str(paired_flowpre_source_id),
        paired_flowpre_run_id=None if paired_flowpre_run_id in (None, "") else str(paired_flowpre_run_id),
        paired_flowpre_seed=paired_flowpre_seed,
    )


def _run_id_for(source: SourceRunContext, reseed_seed: int) -> str:
    return (
        f"flowgen_tpv1_{source.base_token}_reseed_final_"
        f"{source.policy_slug}_src{source.source_version}_seed{int(reseed_seed)}_v1"
    )


def _run_dir_for(run_id: str) -> Path:
    return OFFICIAL_FLOWGEN_ROOT / OUTPUT_SUBDIR / run_id


def _run_materialization_status(run_id: str) -> str:
    run_dir = _run_dir_for(run_id)
    required = [
        run_dir / "config.yaml",
        run_dir / "results.yaml",
        run_dir / "metrics_long.csv",
        run_dir / "run_manifest.json",
        run_dir / "checkpoint.pt",
        run_dir / "run.log",
    ]
    if not run_dir.exists():
        return "missing"
    if all(path.exists() for path in required):
        return "complete"
    return "incomplete"


def _build_effective_training(training_cfg: dict[str, Any], *, run_seed: int) -> dict[str, Any]:
    effective = copy.deepcopy(training_cfg)
    effective["seed"] = int(run_seed)
    effective["save_states"] = False
    effective["log_training"] = True
    effective["save_results"] = True
    effective["save_model"] = True
    return effective


def _build_config_payload(*, source: SourceRunContext, reseed_seed: int) -> dict[str, Any]:
    official_training = source.config.get("official_training") or {}
    return {
        "model": copy.deepcopy(source.config["model"]),
        "training": _build_effective_training(source.config["training"], run_seed=reseed_seed),
        "interpretability": copy.deepcopy(source.config.get("interpretability", {})),
        "seed": int(reseed_seed),
        "official_training": {
            "mode": "flowgen_reseed_final_v1",
            "reseed_panel_id": "flowgen_final_reseed_wide_v1",
            "source_run_id": source.run_id,
            "source_run_manifest": str(source.run_manifest_path),
            "source_config": str(source.config_path),
            "source_results": str(source.results_path),
            "source_metrics_long": str(source.metrics_long_path),
            "source_checkpoint": str(source.checkpoint_path),
            "source_version": source.source_version,
            "source_seed": int(source.source_seed),
            "policy_id": source.source_policy_id,
            "policy_signature": source.source_policy_signature,
            "policy_origin": source.source_policy_origin,
            "historical_source_run_ids": list(source.historical_source_run_ids),
            "run_seed": int(reseed_seed),
            "seed_source": "explicit_reseed_final_v1",
            "condition_col": "type",
            "target": "init",
            "allow_test_holdout": False,
            "skip_phase1": True,
            "temperature_tuning": False,
            "materialize_datasets": False,
            "base_run_id": source.base_flowgen_run_id,
            "base_seed": source.base_flowgen_seed,
            "base_checkpoint": str(source.base_flowgen_checkpoint_path),
            "base_work_base_id": source.base_flowgen_work_base_id,
            "paired_flowpre_source_id": source.paired_flowpre_source_id,
            "paired_flowpre_run_id": source.paired_flowpre_run_id,
            "paired_flowpre_seed": source.paired_flowpre_seed,
            "source_mode": official_training.get("mode"),
            "source_contract_id": source.run_manifest.get("contract_id"),
        },
    }


def _build_evaluation_context(*, source: SourceRunContext, reseed_seed: int) -> dict[str, Any]:
    dataset_manifest_path = source.run_manifest.get(
        "dataset_manifest_path",
        _official_raw_bundle_manifest_path(),
    )
    split_manifest_path = source.run_manifest.get("split_manifest_path")
    return {
        "dataset_name": str(source.run_manifest.get("dataset_name", DEFAULT_OFFICIAL_DATASET_NAME)),
        "dataset_manifest_path": dataset_manifest_path,
        "split_id": str(source.run_manifest.get("split_id", OFFICIAL_SPLIT_ID)),
        "split_manifest_path": split_manifest_path,
        "contract_id": CONTRACT_ID,
        "seed_set_id": f"flowgen_reseed_final_v1_seed{int(reseed_seed)}",
        "base_config_id": f"{source.run_id}__reseed_seed{int(reseed_seed)}",
        "objective_metric_id": OBJECTIVE_METRIC_ID,
        "upstream_variant_fingerprint": source.run_manifest.get("variant_fingerprint"),
        "run_level_axes": {
            "phase": "reseed_final_v1",
            "flowgen_base_run_id": source.base_flowgen_run_id,
            "flowgen_base_work_base_id": source.base_flowgen_work_base_id,
            "flowgen_base_seed": source.base_flowgen_seed,
            "paired_flowpre_source_id": source.paired_flowpre_source_id,
            "paired_flowpre_run_id": source.paired_flowpre_run_id,
            "paired_flowpre_seed": source.paired_flowpre_seed,
            "policy_id": source.source_policy_id,
            "policy_signature": source.source_policy_signature,
            "policy_origin": source.source_policy_origin,
            "historical_source_run_ids": list(source.historical_source_run_ids),
            "reseed_source_run_id": source.run_id,
            "reseed_source_seed": int(source.source_seed),
            "reseed_source_version": source.source_version,
            "reseed_source_contract_id": source.run_manifest.get("contract_id"),
            "run_seed": int(reseed_seed),
            "run_seed_source": "explicit_reseed_final_v1",
            "skip_phase1": True,
            "allow_test_holdout": False,
        },
        "test_enabled": False,
    }


def _write_temp_config(payload: dict[str, Any], *, stem: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"{stem}_",
        suffix=".yaml",
        encoding="utf-8",
        delete=False,
    ) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        return Path(handle.name)


def _rewrite_manifest_config_path(manifest_path: Path, saved_config_path: str | Path | None) -> None:
    if saved_config_path in (None, "") or not manifest_path.exists():
        return
    payload = load_json(manifest_path)
    payload["config_path"] = str(Path(saved_config_path))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _copy_if_exists(src: str | Path | None, dst: Path) -> None:
    if src in (None, ""):
        return
    src_path = Path(src)
    if not src_path.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)


def _materialize_run_aliases(model: Any) -> Path:
    artifacts = getattr(model, "run_artifacts", None)
    if not isinstance(artifacts, dict):
        raise RuntimeError("FlowGen model did not expose run_artifacts after training.")

    run_dir = Path(artifacts["run_dir"])
    run_id = str(artifacts["run_id"])
    saved_config_path = artifacts.get("saved_config_path")
    canonical_manifest_path = run_dir / f"{run_id}_run_manifest.json"

    _rewrite_manifest_config_path(canonical_manifest_path, saved_config_path)

    alias_map = {
        saved_config_path: run_dir / "config.yaml",
        artifacts.get("results_path"): run_dir / "results.yaml",
        artifacts.get("metrics_long_path"): run_dir / "metrics_long.csv",
        artifacts.get("model_path"): run_dir / "checkpoint.pt",
        artifacts.get("log_file_path"): run_dir / "run.log",
        canonical_manifest_path: run_dir / "run_manifest.json",
    }
    for src, dst in alias_map.items():
        _copy_if_exists(src, dst)
    return run_dir


def _validate_results_artifact(results_path: Path) -> None:
    results = load_yaml(results_path)
    for split in ("train", "val"):
        split_payload = results.get(split)
        if not isinstance(split_payload, dict):
            raise RuntimeError(f"FlowGen reseed results are missing split '{split}': {results_path}")
        realism = split_payload.get("realism")
        if not isinstance(realism, dict):
            raise RuntimeError(
                f"FlowGen reseed results are missing realism metrics for split '{split}': {results_path}"
            )


def _validate_materialized_run_dir(run_dir: Path) -> None:
    required = [
        run_dir / "config.yaml",
        run_dir / "results.yaml",
        run_dir / "metrics_long.csv",
        run_dir / "run_manifest.json",
        run_dir / "checkpoint.pt",
        run_dir / "run.log",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"FlowGen reseed run is missing required artifacts under {run_dir}: {missing}")
    _validate_results_artifact(run_dir / "results.yaml")


def _result_paths_for_run_dir(run_dir: str | Path) -> dict[str, str]:
    run_dir = Path(run_dir)
    return {
        "run_dir": str(run_dir),
        "config": str(run_dir / "config.yaml"),
        "results": str(run_dir / "results.yaml"),
        "metrics_long": str(run_dir / "metrics_long.csv"),
        "run_manifest": str(run_dir / "run_manifest.json"),
        "checkpoint": str(run_dir / "checkpoint.pt"),
        "log": str(run_dir / "run.log"),
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["status"])
            writer.writeheader()
        return path

    normalized_rows = [
        {key: _stable_json(value) if isinstance(value, (dict, list)) else value for key, value in row.items()}
        for row in rows
    ]
    fieldnames = sorted({key for row in normalized_rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _build_plan(*, sources: list[SourceRunContext], reseed_seeds: list[int]) -> list[PlanEntry]:
    entries: list[PlanEntry] = []
    for source in sources:
        for reseed_seed in reseed_seeds:
            run_id = _run_id_for(source, reseed_seed)
            entries.append(
                PlanEntry(
                    source_run_id=source.run_id,
                    source_version=source.source_version,
                    source_policy_id=source.source_policy_id,
                    source_seed=int(source.source_seed),
                    reseed_seed=int(reseed_seed),
                    run_id=run_id,
                    output_dir=str(_run_dir_for(run_id)),
                    existing_status=_run_materialization_status(run_id),
                )
            )
    entries.sort(key=lambda entry: (SOURCE_RUN_IDS.index(entry.source_run_id), entry.reseed_seed))
    return entries


def _summarize_counts(entries: list[PlanEntry]) -> dict[str, int]:
    return {
        "planned_total": len(entries),
        "complete_existing": sum(1 for entry in entries if entry.existing_status == "complete"),
        "incomplete_existing": sum(1 for entry in entries if entry.existing_status == "incomplete"),
        "missing": sum(1 for entry in entries if entry.existing_status == "missing"),
    }


def _print_plan_summary(
    *,
    sources: list[SourceRunContext],
    entries: list[PlanEntry],
    summary_paths: dict[str, Path],
) -> None:
    counts = _summarize_counts(entries)
    print("\n" + "=" * 100)
    print("FlowGen final reseed campaign v1")
    print("Mode: official reseed only | same effective config as source run | only the seed changes")
    print("Condition: type | target: init | skip_phase1=True | allow_test_holdout=False")
    print(f"Output root: {OFFICIAL_FLOWGEN_ROOT / OUTPUT_SUBDIR}")

    print("\nSelected source runs:")
    for source in sources:
        print(
            f"  - {source.run_id} | policy_id={source.source_policy_id} | "
            f"source_seed={source.source_seed} | source_version={source.source_version} | "
            f"base_run={source.base_flowgen_run_id}"
        )

    print("\nFixed new reseed seeds:")
    for seed in RESEED_SEEDS:
        print(f"  - {seed}")

    print("\nExpected reseed run ids:")
    for entry in entries:
        print(f"  - {entry.run_id}")

    print("\nPlan counts:")
    print(f"  - source runs: {len(sources)}")
    print(f"  - new seeds per source run: {len(RESEED_SEEDS)}")
    print(f"  - total logical reseed runs: {counts['planned_total']}")
    print(f"  - already complete: {counts['complete_existing']}")
    print(f"  - incomplete existing dirs: {counts['incomplete_existing']}")
    print(f"  - pending new runs: {counts['missing']}")

    print("\nPlan summaries:")
    print(f"  - JSON: {summary_paths['plan_json']}")
    print(f"  - CSV:  {summary_paths['plan_csv']}")


def _resolve_entry_by_run_id(entries: list[PlanEntry], run_id: str) -> PlanEntry:
    matches = [entry for entry in entries if entry.run_id == run_id]
    if not matches:
        raise RuntimeError(f"Run id not found in current FlowGen reseed plan: {run_id}")
    if len(matches) != 1:
        raise RuntimeError(f"Run id appears multiple times in current FlowGen reseed plan: {run_id}")
    return matches[0]


def _run_one(
    *,
    source: SourceRunContext,
    entry: PlanEntry,
    device: str,
    verbose: bool,
) -> dict[str, str]:
    from training.train_flowgen import train_flowgen_pipeline

    config_payload = _build_config_payload(source=source, reseed_seed=entry.reseed_seed)
    evaluation_context = _build_evaluation_context(source=source, reseed_seed=entry.reseed_seed)
    temp_cfg_path = _write_temp_config(config_payload, stem=entry.run_id)
    model = None

    try:
        model = train_flowgen_pipeline(
            condition_col="type",
            config_filename=str(temp_cfg_path),
            base_name=entry.run_id,
            device=device,
            seed=int(entry.reseed_seed),
            verbose=verbose,
            allow_test_holdout=False,
            finetuning=True,
            skip_phase1=True,
            pretrained_path=str(source.base_flowgen_checkpoint_path),
            evaluation_context=evaluation_context,
            output_namespace=OUTPUT_NAMESPACE,
            output_subdir=OUTPUT_SUBDIR,
            fixed_run_id=entry.run_id,
            log_in_run_dir=True,
        )
    finally:
        temp_cfg_path.unlink(missing_ok=True)

    try:
        run_dir = _materialize_run_aliases(model)
        _validate_materialized_run_dir(run_dir)
        return _result_paths_for_run_dir(run_dir)
    finally:
        if model is not None:
            del model
        _release_process_memory()


def _child_failure_message(run_id: str, returncode: int) -> str:
    if returncode == -9:
        return f"Child process killed for {run_id} (returncode={returncode}, probable OOM / killed)."
    if returncode < 0:
        return f"Child process terminated by signal {-returncode} for {run_id} (returncode={returncode})."
    return f"Child process failed for {run_id} with return code {returncode}."


def _run_entry_in_child(
    *,
    entry: PlanEntry,
    script_path: Path,
    device: str,
    quiet: bool,
) -> subprocess.CompletedProcess:
    child_cmd = [
        sys.executable,
        str(script_path),
        "--run-one",
        entry.run_id,
        "--device",
        str(device),
    ]
    if quiet:
        child_cmd.append("--quiet")
    return subprocess.run(child_cmd, cwd=str(ROOT))


def main() -> int:
    args = _parse_args()
    verbose = not args.quiet

    selected_source_run_ids = args.source_run_ids or list(SOURCE_RUN_IDS)
    sources = [_resolve_source_context(run_id) for run_id in selected_source_run_ids]
    source_lookup = {source.run_id: source for source in sources}
    entries = _build_plan(sources=sources, reseed_seeds=list(RESEED_SEEDS))
    if not entries:
        raise RuntimeError("No FlowGen reseed runs remain after filtering.")

    if args.run_one:
        entry = _resolve_entry_by_run_id(entries, str(args.run_one))
        if entry.existing_status == "complete":
            print(f"Run already complete: {entry.run_id} -> {entry.output_dir}")
            return 0
        if entry.existing_status == "incomplete":
            raise RuntimeError(
                f"Target reseed run directory already exists but is incomplete: {entry.output_dir}. "
                "Remove it manually before rerunning."
            )
        source = source_lookup[entry.source_run_id]
        print(f"Running isolated FlowGen reseed run: {entry.run_id}")
        _run_one(
            source=source,
            entry=entry,
            device=args.device,
            verbose=verbose,
        )
        print(f"Completed isolated FlowGen reseed run: {entry.run_id}")
        return 0

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    campaign_id = f"flowgen_reseed_final_v1_{_campaign_timestamp()}"
    plan_json_path = SUMMARY_ROOT / f"{campaign_id}_plan.json"
    plan_csv_path = SUMMARY_ROOT / f"{campaign_id}_plan.csv"

    plan_payload = {
        "campaign_id": campaign_id,
        "script": str(Path(__file__).resolve()),
        "model_family": MODEL_FAMILY,
        "contract_id": CONTRACT_ID,
        "output_root": str(OFFICIAL_FLOWGEN_ROOT / OUTPUT_SUBDIR),
        "source_run_ids": list(selected_source_run_ids),
        "fixed_new_reseed_seeds": list(RESEED_SEEDS),
        "sources": [
            asdict(source)
            | {
                "run_dir": str(source.run_dir),
                "config_path": str(source.config_path),
                "results_path": str(source.results_path),
                "metrics_long_path": str(source.metrics_long_path),
                "run_manifest_path": str(source.run_manifest_path),
                "checkpoint_path": str(source.checkpoint_path),
                "base_flowgen_checkpoint_path": str(source.base_flowgen_checkpoint_path),
            }
            for source in sources
        ],
        "runs": [asdict(entry) for entry in entries],
    }
    _write_json(plan_json_path, plan_payload)
    _write_rows_csv(plan_csv_path, [asdict(entry) for entry in entries])

    _print_plan_summary(
        sources=sources,
        entries=entries,
        summary_paths={"plan_json": plan_json_path, "plan_csv": plan_csv_path},
    )

    if args.summary_only:
        print("\nSummary-only mode enabled. No training started.")
        return 0

    completed = 0
    skipped = 0
    failed = 0
    result_rows: list[dict[str, Any]] = []
    script_path = Path(__file__).resolve()

    for index, entry in enumerate(entries, start=1):
        print("\n" + "-" * 100)
        print(f"[{index}/{len(entries)}] FlowGen reseed run: {entry.run_id}")
        print(f"  source_run: {entry.source_run_id}")
        print(f"  reseed_seed: {entry.reseed_seed}")
        print(f"  output_dir: {entry.output_dir}")

        if entry.existing_status == "complete":
            skipped += 1
            entry.status = "skipped_complete"
            entry.result_paths = _result_paths_for_run_dir(entry.output_dir)
            result_rows.append(asdict(entry))
            print("  status: already complete, skipping")
            continue
        if entry.existing_status == "incomplete":
            failed += 1
            entry.status = "failed_existing_incomplete"
            entry.error = (
                "Output directory already exists but is incomplete. Remove it manually before rerunning."
            )
            result_rows.append(asdict(entry))
            print(f"  status: {entry.error}")
            continue

        try:
            proc = _run_entry_in_child(
                entry=entry,
                script_path=script_path,
                device=args.device,
                quiet=args.quiet,
            )
            if proc.returncode != 0:
                raise RuntimeError(_child_failure_message(entry.run_id, proc.returncode))
            entry.status = "completed"
            entry.result_paths = _result_paths_for_run_dir(entry.output_dir)
            completed += 1
            print("  status: completed")
        except Exception as exc:
            entry.status = "failed"
            entry.error = f"{type(exc).__name__}: {exc}"
            failed += 1
            print(f"  status: FAILED -> {entry.error}")
        finally:
            result_rows.append(asdict(entry))
            _release_process_memory()

    results_json_path = SUMMARY_ROOT / f"{campaign_id}_results.json"
    results_csv_path = SUMMARY_ROOT / f"{campaign_id}_results.csv"
    _write_json(
        results_json_path,
        {
            "campaign_id": campaign_id,
            "script": str(Path(__file__).resolve()),
            "contract_id": CONTRACT_ID,
            "output_root": str(OFFICIAL_FLOWGEN_ROOT / OUTPUT_SUBDIR),
            "fixed_new_reseed_seeds": list(RESEED_SEEDS),
            "summary": {
                "planned_total": len(entries),
                "completed": completed,
                "skipped_complete": skipped,
                "failed": failed,
            },
            "runs": result_rows,
        },
    )
    _write_rows_csv(results_csv_path, result_rows)

    print("\n" + "=" * 100)
    print("FlowGen reseed campaign finished")
    print(f"  planned_total: {len(entries)}")
    print(f"  completed: {completed}")
    print(f"  skipped_complete: {skipped}")
    print(f"  failed: {failed}")
    print(f"  results_json: {results_json_path}")
    print(f"  results_csv: {results_csv_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
