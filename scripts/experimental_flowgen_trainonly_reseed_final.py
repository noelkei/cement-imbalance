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
OUTPUT_NAMESPACE = "experimental/train_only"
OUTPUT_SUBDIR_ROOT = "reseed_final"
CONTRACT_ID = "flowgen_trainonly_reseed_final_v1"
OBJECTIVE_METRIC_ID = "flowgen_trainonly_realism_reseed_final"
DEFAULT_OFFICIAL_DATASET_NAME = "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"
TRAIN_ONLY_POLICY = "train_only"

FLOWGEN_TRAINONLY_ROOT = ROOT / "outputs" / "models" / "experimental" / "train_only" / "flowgen"
SUMMARY_ROOT = FLOWGEN_TRAINONLY_ROOT / "campaign_summaries" / OUTPUT_SUBDIR_ROOT

BASE_TOKENS = {
    "candidate_trainonly_1": "ct1",
    "candidate_trainonly_2": "ct2",
}

# Audited from existing path-level seed material in this repository when the
# runner was created. The reseed panel below must stay outside this set.
PREVIOUSLY_USED_FLOW_SEEDS = (
    1117,
    1234,
    2221,
    2468,
    2887,
    2888,
    2889,
    2890,
    2891,
    2892,
    2893,
    2894,
    2895,
    2896,
    2897,
    2898,
    2899,
    2900,
    2901,
    2902,
    3331,
    3901,
    4321,
    4447,
    5270,
    5678,
    6479,
    6769,
    7319,
    8423,
    9101,
    9547,
    10627,
    96024,
)

RESEED_SEEDS = (11863, 12979, 14143, 15427)


@dataclass(frozen=True)
class SourceSpec:
    slot: int
    source_group: str
    base_token: str
    source_run_id: str
    selection_role: str
    selection_note: str


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        slot=1,
        source_group="round3",
        base_token="candidate_trainonly_1",
        source_run_id="flowgen_trainonly_tpv1_ct1_round3_r3b1_t06_ksx_light_seed6769_v1",
        selection_role="balanced_anchor",
        selection_note=(
            "Best balanced train-only candidate after round3: very strong X fit while keeping KS and XY structure "
            "reasonably controlled."
        ),
    ),
    SourceSpec(
        slot=2,
        source_group="round3",
        base_token="candidate_trainonly_1",
        source_run_id="flowgen_trainonly_tpv1_ct1_round3_r3a2_t06_clip125_seed6769_v1",
        selection_role="x_priority",
        selection_note=(
            "Strongest ct1 X-priority candidate. It gave the biggest X trainaligned gain and kept the rest sane enough "
            "to deserve full seed robustness."
        ),
    ),
    SourceSpec(
        slot=3,
        source_group="round3",
        base_token="candidate_trainonly_2",
        source_run_id="flowgen_trainonly_tpv1_ct2_round3_r3a1_t06_w1x120_seed6769_v1",
        selection_role="ct2_balanced_anchor",
        selection_note=(
            "Best balanced ct2 candidate from the main round3 panel. Keeps the ct2 branch alive with its own strongest "
            "T06-derived geometry."
        ),
    ),
    SourceSpec(
        slot=4,
        source_group="round3_confirm",
        base_token="candidate_trainonly_2",
        source_run_id="flowgen_trainonly_tpv1_ct2_round3confirm_r3a2_t06_clip125_seed6769_v1",
        selection_role="ct2_transfer_confirm",
        selection_note=(
            "Cross-base confirmation that the clip125 corridor transfers from ct1 to ct2 and deserves to enter the final reseed."
        ),
    ),
)


@dataclass
class SourceRunContext:
    spec: SourceSpec
    run_id: str
    run_dir: Path
    config_path: Path
    results_path: Path
    run_manifest_path: Path
    config: dict[str, Any]
    results: dict[str, Any]
    run_manifest: dict[str, Any]
    source_seed: int
    source_version: str
    source_stage_slug: str
    source_policy_id: str
    source_policy_signature: str | None
    source_policy_origin: str | None
    source_policy_set_id: str | None
    historical_source_run_ids: list[str]
    source_contract_id: str | None
    source_objective_metric_id: str | None
    source_base_config_id: str
    source_seed_set_id: str | None
    source_variant_fingerprint: str | None
    base_run_id: str
    base_seed: int
    base_work_base_id: str
    base_checkpoint_path: Path
    base_run_manifest_path: Path | None
    paired_flowpre_source_id: str | None
    paired_flowpre_run_id: str | None
    paired_flowpre_seed: int | None
    monitor_x_w1_trainaligned: float | None
    monitor_ks_mean: float | None
    monitor_y_w1_trainaligned: float | None
    monitor_xy_pearson_rel: float | None
    monitor_xy_spearman_rel: float | None


@dataclass
class PlanEntry:
    source_run_id: str
    source_policy_id: str
    selection_role: str
    base_token: str
    base_run_id: str
    reseed_seed: int
    run_id: str
    output_dir: str
    existing_status: str
    error: str = ""
    status: str = "planned"
    result_paths: dict[str, str] = field(default_factory=dict)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "FlowGen train-only final reseed runner. It reuses the four shortlisted "
            "train-only source configs and expands each family to a five-seed panel "
            "by adding four new seeds that are not already used elsewhere in the project."
        )
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--source-run",
        dest="source_run_ids",
        action="append",
        choices=[spec.source_run_id for spec in SOURCE_SPECS],
        default=None,
        help="Optional subset of shortlisted source runs to reseed.",
    )
    ap.add_argument(
        "--reseed-seed",
        dest="reseed_seeds",
        action="append",
        type=int,
        choices=RESEED_SEEDS,
        default=None,
        help="Optional subset of the new reseed seeds to run.",
    )
    ap.add_argument("--run-one", default=None, help=argparse.SUPPRESS)
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


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _resolve_required_artifact(run_dir: Path, run_id: str, kind: str) -> Path:
    if kind == "config":
        candidates = [run_dir / "config.yaml", run_dir / f"{run_id}.yaml"]
    elif kind == "results":
        candidates = [run_dir / "results.yaml", run_dir / f"{run_id}_results.yaml"]
    elif kind == "run_manifest":
        candidates = [run_dir / "run_manifest.json", run_dir / f"{run_id}_run_manifest.json"]
    else:
        raise ValueError(f"Unsupported artifact kind: {kind}")

    for candidate in candidates:
        if _nonempty_file(candidate):
            return candidate
    raise FileNotFoundError(f"Missing required source artifact '{kind}' under {run_dir}.")


def _parse_source_run_id(run_id: str) -> tuple[str, str, int, str]:
    match = re.match(r"^flowgen_trainonly_tpv1_(ct[12])_(.+)_seed(\d+)_v(\d+)$", str(run_id))
    if match is None:
        raise ValueError(f"Unsupported FlowGen train-only source run id format: {run_id}")
    base_token = str(match.group(1))
    stage_slug = str(match.group(2))
    source_seed = int(match.group(3))
    source_version = f"v{match.group(4)}"
    return base_token, stage_slug, source_seed, source_version


def _validate_seed_panel() -> None:
    if len(set(RESEED_SEEDS)) != len(RESEED_SEEDS):
        raise RuntimeError(f"Duplicate reseed seeds are not allowed: {RESEED_SEEDS}")
    overlap = sorted(set(RESEED_SEEDS).intersection(PREVIOUSLY_USED_FLOW_SEEDS))
    if overlap:
        raise RuntimeError(f"Reseed seeds overlap with previously used project seeds: {overlap}")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_source_context(spec: SourceSpec) -> SourceRunContext:
    run_dir = FLOWGEN_TRAINONLY_ROOT / spec.source_group / spec.base_token / spec.source_run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Shortlisted train-only FlowGen source run not found: {run_dir}")

    config_path = _resolve_required_artifact(run_dir, spec.source_run_id, "config")
    results_path = _resolve_required_artifact(run_dir, spec.source_run_id, "results")
    run_manifest_path = _resolve_required_artifact(run_dir, spec.source_run_id, "run_manifest")

    config = load_yaml(config_path)
    results = load_yaml(results_path)
    run_manifest = load_json(run_manifest_path)

    if str(run_manifest.get("model_family")) != MODEL_FAMILY:
        raise ValueError(f"Source run manifest must have model_family='{MODEL_FAMILY}': {run_manifest_path}")
    if str(run_manifest.get("split_id")) != OFFICIAL_SPLIT_ID:
        raise ValueError(f"Source run manifest must have split_id='{OFFICIAL_SPLIT_ID}': {run_manifest_path}")
    if bool(run_manifest.get("test_enabled")):
        raise ValueError(f"Source run manifest unexpectedly enables test: {run_manifest_path}")
    monitoring = run_manifest.get("monitoring") or {}
    if str(monitoring.get("policy")) != TRAIN_ONLY_POLICY:
        raise ValueError(f"Source run manifest must have monitoring.policy='{TRAIN_ONLY_POLICY}': {run_manifest_path}")
    if "model" not in config or not isinstance(config["model"], dict):
        raise ValueError(f"Source config missing model block: {config_path}")
    if "training" not in config or not isinstance(config["training"], dict):
        raise ValueError(f"Source config missing training block: {config_path}")

    finetune = results.get("finetune") or {}
    if not bool(finetune.get("enabled")):
        raise ValueError(f"Source results must report finetune.enabled=true: {results_path}")

    ct_token_from_id, source_stage_slug, source_seed_from_id, source_version = _parse_source_run_id(spec.source_run_id)
    expected_ct = BASE_TOKENS[spec.base_token]
    if ct_token_from_id != expected_ct:
        raise ValueError(
            f"Source run id/base token mismatch for {spec.source_run_id}: "
            f"id says {ct_token_from_id}, spec says {expected_ct}"
        )

    config_seed = config.get("seed")
    training_seed = (config.get("training") or {}).get("seed")
    seed_candidates = {int(value) for value in [source_seed_from_id, config_seed, training_seed] if value is not None}
    if len(seed_candidates) != 1:
        detail = {
            "run_id_seed": source_seed_from_id,
            "config.seed": config_seed,
            "training.seed": training_seed,
        }
        raise ValueError(f"Inconsistent source seed values for {spec.source_run_id}: {detail}")
    source_seed = seed_candidates.pop()

    trainonly_training = config.get("trainonly_training") or {}
    run_axes = run_manifest.get("run_level_axes") or {}

    source_policy_id = (
        trainonly_training.get("policy_id")
        or run_axes.get("policy_id")
    )
    if not source_policy_id:
        raise ValueError(f"Unable to resolve source policy id for {spec.source_run_id}")
    source_policy_id = str(source_policy_id)

    base_run_id = (
        trainonly_training.get("base_run_id")
        or run_axes.get("flowgen_base_run_id")
    )
    if not base_run_id:
        raise ValueError(f"Unable to resolve base FlowGen run id for {spec.source_run_id}")
    base_run_id = str(base_run_id)

    base_seed_raw = trainonly_training.get("base_seed") or run_axes.get("flowgen_base_seed")
    if base_seed_raw is None:
        raise ValueError(f"Unable to resolve base FlowGen seed for {spec.source_run_id}")
    base_seed = int(base_seed_raw)

    base_work_base_id = (
        trainonly_training.get("base_work_base_id")
        or run_axes.get("flowgen_base_work_base_id")
        or spec.base_token
    )
    base_work_base_id = str(base_work_base_id)
    if base_work_base_id != spec.base_token:
        raise ValueError(
            f"Source base token mismatch for {spec.source_run_id}: "
            f"training says {base_work_base_id}, spec says {spec.base_token}"
        )

    base_checkpoint_raw = trainonly_training.get("base_checkpoint")
    if base_checkpoint_raw in (None, ""):
        base_checkpoint_path = (
            FLOWGEN_TRAINONLY_ROOT / "bases" / base_work_base_id / base_run_id / "checkpoint.pt"
        )
    else:
        base_checkpoint_path = Path(str(base_checkpoint_raw))
    if not base_checkpoint_path.exists():
        raise FileNotFoundError(f"Base FlowGen checkpoint not found for {spec.source_run_id}: {base_checkpoint_path}")

    base_run_manifest_raw = trainonly_training.get("base_run_manifest")
    base_run_manifest_path = None if base_run_manifest_raw in (None, "") else Path(str(base_run_manifest_raw))

    monitor_realism = ((results.get("val") or {}).get("realism") or {})
    monitor_x = (monitor_realism.get("x") or {})
    monitor_y = (monitor_realism.get("y") or {})
    monitor_overall = (monitor_realism.get("overall") or {})

    historical_source_run_ids_raw = (
        trainonly_training.get("historical_source_run_ids")
        or run_axes.get("historical_source_run_ids")
        or []
    )
    historical_source_run_ids = [str(item) for item in historical_source_run_ids_raw]

    source_base_config_id = run_manifest.get("base_config_id")
    if not source_base_config_id:
        raise ValueError(f"Source run manifest missing base_config_id: {run_manifest_path}")

    return SourceRunContext(
        spec=spec,
        run_id=spec.source_run_id,
        run_dir=run_dir,
        config_path=config_path,
        results_path=results_path,
        run_manifest_path=run_manifest_path,
        config=config,
        results=results,
        run_manifest=run_manifest,
        source_seed=source_seed,
        source_version=source_version,
        source_stage_slug=source_stage_slug,
        source_policy_id=source_policy_id,
        source_policy_signature=None if trainonly_training.get("policy_signature") is None else str(trainonly_training.get("policy_signature")),
        source_policy_origin=None if trainonly_training.get("policy_origin") is None else str(trainonly_training.get("policy_origin")),
        source_policy_set_id=None if trainonly_training.get("policy_set_id") is None else str(trainonly_training.get("policy_set_id")),
        historical_source_run_ids=historical_source_run_ids,
        source_contract_id=None if run_manifest.get("contract_id") is None else str(run_manifest.get("contract_id")),
        source_objective_metric_id=None if run_manifest.get("objective_metric_id") is None else str(run_manifest.get("objective_metric_id")),
        source_base_config_id=str(source_base_config_id),
        source_seed_set_id=None if run_manifest.get("seed_set_id") is None else str(run_manifest.get("seed_set_id")),
        source_variant_fingerprint=None if run_manifest.get("variant_fingerprint") is None else str(run_manifest.get("variant_fingerprint")),
        base_run_id=base_run_id,
        base_seed=base_seed,
        base_work_base_id=base_work_base_id,
        base_checkpoint_path=base_checkpoint_path,
        base_run_manifest_path=base_run_manifest_path,
        paired_flowpre_source_id=None if trainonly_training.get("paired_flowpre_source_id") is None else str(trainonly_training.get("paired_flowpre_source_id")),
        paired_flowpre_run_id=None if trainonly_training.get("paired_flowpre_run_id") is None else str(trainonly_training.get("paired_flowpre_run_id")),
        paired_flowpre_seed=_float_or_none(trainonly_training.get("paired_flowpre_seed")) and int(trainonly_training.get("paired_flowpre_seed")),
        monitor_x_w1_trainaligned=_float_or_none(monitor_x.get("w1_mean_trainaligned")),
        monitor_ks_mean=_float_or_none(monitor_overall.get("ks_mean")),
        monitor_y_w1_trainaligned=_float_or_none(monitor_y.get("w1_mean_trainaligned")),
        monitor_xy_pearson_rel=_float_or_none(monitor_overall.get("xy_pearson_fro_rel")),
        monitor_xy_spearman_rel=_float_or_none(monitor_overall.get("xy_spearman_fro_rel")),
    )


def _policy_slug_for_run_id(source: SourceRunContext) -> str:
    return source.source_policy_id.lower()


def _run_id_for(source: SourceRunContext, reseed_seed: int) -> str:
    token = BASE_TOKENS[source.base_work_base_id]
    policy_slug = _policy_slug_for_run_id(source)
    return f"flowgen_trainonly_tpv1_{token}_reseedfinal_{policy_slug}_seed{int(reseed_seed)}_v1"


def _run_dir_for(source: SourceRunContext, run_id: str) -> Path:
    return FLOWGEN_TRAINONLY_ROOT / OUTPUT_SUBDIR_ROOT / source.base_work_base_id / run_id


def _run_materialization_status(source: SourceRunContext, run_id: str) -> str:
    run_dir = _run_dir_for(source, run_id)
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


def _reset_incomplete_run_dir(path: str | Path) -> Path:
    run_dir = Path(path)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return run_dir


def _build_effective_training(source_training: dict[str, Any], *, reseed_seed: int) -> dict[str, Any]:
    training = copy.deepcopy(source_training)
    training["seed"] = int(reseed_seed)
    training["save_states"] = False
    training["log_training"] = True
    training["save_results"] = True
    training["save_model"] = True
    training["realism_bootstrap"] = 10
    training["realism_rvr_bootstrap"] = 10
    training["realism_seed_offset"] = int(training.get("realism_seed_offset", 0))
    return training


def _seed_panel_slug(source_seed: int) -> str:
    full_panel = [int(source_seed), *[int(seed) for seed in RESEED_SEEDS]]
    return "_".join(str(seed) for seed in full_panel)


def _build_config_payload(source: SourceRunContext, *, reseed_seed: int) -> dict[str, Any]:
    training = _build_effective_training(source.config["training"], reseed_seed=reseed_seed)
    source_training_meta = source.config.get("trainonly_training") or {}
    seed_panel = [int(source.source_seed), *[int(seed) for seed in RESEED_SEEDS]]

    return {
        "model": copy.deepcopy(source.config["model"]),
        "training": training,
        "interpretability": copy.deepcopy(source.config.get("interpretability", {})),
        "seed": int(reseed_seed),
        "trainonly_training": {
            "mode": "flowgen_trainonly_reseed_final_finetune",
            "policy_set_id": CONTRACT_ID,
            "policy_id": source.source_policy_id,
            "policy_signature": source.source_policy_signature,
            "policy_origin": source.source_policy_origin,
            "historical_source_run_ids": list(source.historical_source_run_ids),
            "run_seed": int(reseed_seed),
            "source_seed": int(source.source_seed),
            "seed_source": "trainonly_reseed_panel_v1",
            "seed_panel": seed_panel,
            "source_run_id": source.run_id,
            "source_contract_id": source.source_contract_id,
            "source_objective_metric_id": source.source_objective_metric_id,
            "source_policy_set_id": source.source_policy_set_id,
            "selection_role": source.spec.selection_role,
            "selection_note": source.spec.selection_note,
            "condition_col": "type",
            "target": "init",
            "monitoring_policy": TRAIN_ONLY_POLICY,
            "allow_test_holdout": False,
            "skip_phase1": True,
            "temperature_tuning": False,
            "materialize_datasets": False,
            "base_run_id": source.base_run_id,
            "base_seed": int(source.base_seed),
            "base_run_manifest": None if source.base_run_manifest_path is None else str(source.base_run_manifest_path),
            "base_checkpoint": str(source.base_checkpoint_path),
            "base_work_base_id": source.base_work_base_id,
            "paired_flowpre_source_id": source.paired_flowpre_source_id,
            "paired_flowpre_run_id": source.paired_flowpre_run_id,
            "paired_flowpre_seed": source.paired_flowpre_seed,
        },
    }


def _build_evaluation_context(source: SourceRunContext, *, reseed_seed: int) -> dict[str, Any]:
    seed_panel_slug = _seed_panel_slug(source.source_seed)
    dataset_manifest_path = source.run_manifest.get("dataset_manifest_path")
    split_manifest_path = source.run_manifest.get("split_manifest_path")
    return {
        "dataset_name": str(source.run_manifest.get("dataset_name", DEFAULT_OFFICIAL_DATASET_NAME)),
        "dataset_manifest_path": dataset_manifest_path,
        "split_id": str(source.run_manifest.get("split_id", OFFICIAL_SPLIT_ID)),
        "split_manifest_path": split_manifest_path,
        "contract_id": CONTRACT_ID,
        "seed_set_id": f"{source.base_work_base_id}_{_policy_slug_for_run_id(source)}_{seed_panel_slug}",
        "base_config_id": source.source_base_config_id,
        "objective_metric_id": OBJECTIVE_METRIC_ID,
        "upstream_variant_fingerprint": source.run_manifest.get("upstream_variant_fingerprint"),
        "run_level_axes": {
            "phase": "trainonly_reseed_final_finetune",
            "line": "experimental_train_only",
            "flowgen_base_run_id": source.base_run_id,
            "flowgen_base_work_base_id": source.base_work_base_id,
            "flowgen_base_seed": int(source.base_seed),
            "paired_flowpre_source_id": source.paired_flowpre_source_id,
            "paired_flowpre_run_id": source.paired_flowpre_run_id,
            "paired_flowpre_seed": source.paired_flowpre_seed,
            "policy_id": source.source_policy_id,
            "policy_signature": source.source_policy_signature,
            "policy_origin": source.source_policy_origin,
            "historical_source_run_ids": list(source.historical_source_run_ids),
            "source_run_id": source.run_id,
            "source_stage_slug": source.source_stage_slug,
            "source_seed": int(source.source_seed),
            "source_seed_set_id": source.source_seed_set_id,
            "source_contract_id": source.source_contract_id,
            "source_objective_metric_id": source.source_objective_metric_id,
            "selection_role": source.spec.selection_role,
            "selection_note": source.spec.selection_note,
            "reseed_seed": int(reseed_seed),
            "run_seed": int(reseed_seed),
            "run_seed_source": "trainonly_reseed_panel_v1",
            "full_seed_panel": [int(source.source_seed), *[int(seed) for seed in RESEED_SEEDS]],
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
    finetune = results.get("finetune")
    if not isinstance(finetune, dict) or not bool(finetune.get("enabled")):
        raise RuntimeError(f"Reseed results must report finetune.enabled=true: {results_path}")
    for split in ("train", "val"):
        split_payload = results.get(split)
        if not isinstance(split_payload, dict):
            raise RuntimeError(f"Reseed results are missing split '{split}': {results_path}")
        realism = split_payload.get("realism")
        if not isinstance(realism, dict):
            raise RuntimeError(f"Reseed results are missing realism metrics for split '{split}': {results_path}")


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
        raise RuntimeError(f"Reseed FlowGen run is missing required artifacts under {run_dir}: {missing}")
    run_manifest = load_json(run_dir / "run_manifest.json")
    monitoring = run_manifest.get("monitoring") or {}
    if str(monitoring.get("policy")) != TRAIN_ONLY_POLICY:
        raise RuntimeError(f"Reseed FlowGen run manifest is missing monitoring.policy=train_only: {run_dir}")
    if bool(run_manifest.get("test_enabled")):
        raise RuntimeError(f"Reseed FlowGen run manifest unexpectedly enables test: {run_dir}")
    _validate_results_artifact(run_dir / "results.yaml")


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = ["status"]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return path

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_rows.append(
            {
                key: _stable_json(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )

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


def _source_plan_payload(source: SourceRunContext) -> dict[str, Any]:
    return {
        "slot": source.spec.slot,
        "source_group": source.spec.source_group,
        "source_run_id": source.run_id,
        "selection_role": source.spec.selection_role,
        "selection_note": source.spec.selection_note,
        "source_seed": int(source.source_seed),
        "source_policy_id": source.source_policy_id,
        "source_policy_signature": source.source_policy_signature,
        "source_policy_set_id": source.source_policy_set_id,
        "source_contract_id": source.source_contract_id,
        "base_config_id": source.source_base_config_id,
        "run_dir": str(source.run_dir),
        "config_path": str(source.config_path),
        "results_path": str(source.results_path),
        "run_manifest_path": str(source.run_manifest_path),
        "base_run_id": source.base_run_id,
        "base_seed": int(source.base_seed),
        "base_work_base_id": source.base_work_base_id,
        "base_checkpoint_path": str(source.base_checkpoint_path),
        "paired_flowpre_source_id": source.paired_flowpre_source_id,
        "paired_flowpre_run_id": source.paired_flowpre_run_id,
        "paired_flowpre_seed": source.paired_flowpre_seed,
        "monitor_x_w1_trainaligned": source.monitor_x_w1_trainaligned,
        "monitor_ks_mean": source.monitor_ks_mean,
        "monitor_y_w1_trainaligned": source.monitor_y_w1_trainaligned,
        "monitor_xy_pearson_rel": source.monitor_xy_pearson_rel,
        "monitor_xy_spearman_rel": source.monitor_xy_spearman_rel,
    }


def _build_plan(
    *,
    sources: list[SourceRunContext],
    allowed_source_run_ids: set[str] | None,
    allowed_reseed_seeds: set[int] | None,
) -> list[PlanEntry]:
    entries: list[PlanEntry] = []
    selected_sources = [
        source for source in sources if allowed_source_run_ids is None or source.run_id in allowed_source_run_ids
    ]
    seed_panel = [seed for seed in RESEED_SEEDS if allowed_reseed_seeds is None or seed in allowed_reseed_seeds]

    for source in selected_sources:
        for reseed_seed in seed_panel:
            run_id = _run_id_for(source, reseed_seed)
            entries.append(
                PlanEntry(
                    source_run_id=source.run_id,
                    source_policy_id=source.source_policy_id,
                    selection_role=source.spec.selection_role,
                    base_token=source.base_work_base_id,
                    base_run_id=source.base_run_id,
                    reseed_seed=int(reseed_seed),
                    run_id=run_id,
                    output_dir=str(_run_dir_for(source, run_id)),
                    existing_status=_run_materialization_status(source, run_id),
                )
            )

    entries.sort(key=lambda entry: (entry.base_token, entry.source_policy_id, entry.reseed_seed))
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
    print("FlowGen train-only final reseed")
    print("Run seed policy: 4 new project-unique seeds per shortlisted cfg (5 seeds total including source seed 6769)")
    print("Condition: type | target: init | skip_phase1=True | allow_test_holdout=False")
    print("Monitoring policy: train_only | dataset materialization: disabled | temperature tuning: disabled")
    print("Realism bootstrap: 10/10 | campaign goal: robustness reseed of the final T06-derived shortlist")
    print(f"New reseed seeds: {', '.join(str(seed) for seed in RESEED_SEEDS)}")

    print("\nShortlisted source configs:")
    for source in sources:
        print(
            f"  - slot={source.spec.slot} | base={source.base_work_base_id} | "
            f"policy={source.source_policy_id} | source_run={source.run_id}"
        )
        print(
            f"    source metrics: x_ta={source.monitor_x_w1_trainaligned} | ks={source.monitor_ks_mean} | "
            f"y_ta={source.monitor_y_w1_trainaligned} | xyP={source.monitor_xy_pearson_rel} | "
            f"xyS={source.monitor_xy_spearman_rel}"
        )
        print(f"    rationale: {source.spec.selection_note}")

    print("\nCurrent plan run ids:")
    for entry in entries:
        print(f"  - {entry.run_id}")

    print("\nPlan counts:")
    print(f"  - total logical runs: {counts['planned_total']}")
    print(f"  - already complete: {counts['complete_existing']}")
    print(f"  - incomplete existing dirs: {counts['incomplete_existing']}")
    print(f"  - pending new runs: {counts['missing']}")

    print("\nPlan summaries:")
    print(f"  - JSON: {summary_paths['plan_json']}")
    print(f"  - CSV:  {summary_paths['plan_csv']}")


def _run_one(
    *,
    source: SourceRunContext,
    entry: PlanEntry,
    device: str,
    verbose: bool,
) -> dict[str, str]:
    from training.train_flowgen import train_flowgen_pipeline

    config_payload = _build_config_payload(source, reseed_seed=entry.reseed_seed)
    evaluation_context = _build_evaluation_context(source, reseed_seed=entry.reseed_seed)
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
            pretrained_path=str(source.base_checkpoint_path),
            evaluation_context=evaluation_context,
            monitoring_policy=TRAIN_ONLY_POLICY,
            output_namespace=OUTPUT_NAMESPACE,
            output_subdir=f"{OUTPUT_SUBDIR_ROOT}/{source.base_work_base_id}",
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


def _resolve_plan_inputs(args: argparse.Namespace) -> tuple[set[str] | None, set[int] | None]:
    if args.run_one:
        return None, None
    source_run_ids = None if not args.source_run_ids else set(args.source_run_ids)
    reseed_seeds = None if not args.reseed_seeds else {int(seed) for seed in args.reseed_seeds}
    return source_run_ids, reseed_seeds


def _resolve_entry_by_run_id(entries: list[PlanEntry], run_id: str) -> PlanEntry:
    matches = [entry for entry in entries if entry.run_id == run_id]
    if not matches:
        raise RuntimeError(f"Run id not found in current FlowGen train-only reseed plan: {run_id}")
    if len(matches) != 1:
        raise RuntimeError(f"Run id appears multiple times in current FlowGen train-only reseed plan: {run_id}")
    return matches[0]


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

    _validate_seed_panel()

    selected_source_run_ids, selected_reseed_seeds = _resolve_plan_inputs(args)
    sources = [_resolve_source_context(spec) for spec in SOURCE_SPECS]
    entries = _build_plan(
        sources=sources,
        allowed_source_run_ids=selected_source_run_ids,
        allowed_reseed_seeds=selected_reseed_seeds,
    )
    if not entries:
        raise RuntimeError("No FlowGen train-only reseed runs remain after filtering.")

    if args.run_one:
        entry = _resolve_entry_by_run_id(entries, str(args.run_one))
        if entry.existing_status == "complete":
            print(f"Run already complete: {entry.run_id} -> {entry.output_dir}")
            return 0
        if entry.existing_status == "incomplete":
            removed = _reset_incomplete_run_dir(entry.output_dir)
            print(f"Removed incomplete run directory before retry: {removed}")
        source_lookup = {source.run_id: source for source in sources}
        print(f"Running isolated FlowGen train-only reseed run: {entry.run_id}")
        _run_one(
            source=source_lookup[entry.source_run_id],
            entry=entry,
            device=args.device,
            verbose=verbose,
        )
        print(f"Completed isolated FlowGen train-only reseed run: {entry.run_id}")
        return 0

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    campaign_id = f"flowgen_trainonly_reseed_final_{_campaign_timestamp()}"
    plan_json_path = SUMMARY_ROOT / f"{campaign_id}_plan.json"
    plan_csv_path = SUMMARY_ROOT / f"{campaign_id}_plan.csv"

    plan_payload = {
        "campaign_id": campaign_id,
        "script": str(Path(__file__).resolve()),
        "seed_policy": {
            "source_seed": 6769,
            "new_reseed_seeds": list(RESEED_SEEDS),
            "panel_size_per_cfg": 5,
            "new_runs_per_cfg": 4,
        },
        "model_family": MODEL_FAMILY,
        "contract_id": CONTRACT_ID,
        "output_root": str(FLOWGEN_TRAINONLY_ROOT / OUTPUT_SUBDIR_ROOT),
        "sources": [_source_plan_payload(source) for source in sources],
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
    script_path = Path(__file__).resolve()

    print("\n" + "=" * 100)
    print(f"Starting FlowGen train-only reseed final | planned runs={len(entries)}")

    for idx, entry in enumerate(entries, start=1):
        print("\n" + "-" * 100)
        print(f"[{idx}/{len(entries)}] {entry.run_id}")
        print(
            f"  source={entry.source_run_id} | policy={entry.source_policy_id} | "
            f"base={entry.base_run_id} | reseed_seed={entry.reseed_seed} | existing={entry.existing_status}"
        )

        if entry.existing_status == "complete":
            entry.status = "skipped_existing"
            entry.result_paths = _result_paths_for_run_dir(entry.output_dir)
            skipped += 1
            print("  status=skip_existing_complete")
            continue

        if entry.existing_status == "incomplete":
            removed = _reset_incomplete_run_dir(entry.output_dir)
            print(f"  status=reset_incomplete | removed={removed}")

        try:
            child = _run_entry_in_child(
                entry=entry,
                script_path=script_path,
                device=args.device,
                quiet=args.quiet,
            )
            if child.returncode == 0:
                run_dir = Path(entry.output_dir)
                _validate_materialized_run_dir(run_dir)
                entry.result_paths = _result_paths_for_run_dir(run_dir)
                entry.status = "completed"
                completed += 1
                print(f"  status=completed | output={entry.output_dir}")
            else:
                entry.status = "failed"
                entry.error = _child_failure_message(entry.run_id, child.returncode)
                failed += 1
                print(f"  status=failed | error={entry.error}")
        except Exception as exc:
            entry.status = "failed"
            entry.error = f"{type(exc).__name__}: {exc}"
            failed += 1
            print(f"  status=failed | error={entry.error}")
        finally:
            _release_process_memory()

    result_rows = [asdict(entry) for entry in entries]
    results_payload = {
        "campaign_id": campaign_id,
        "script": str(Path(__file__).resolve()),
        "seed_policy": {
            "source_seed": 6769,
            "new_reseed_seeds": list(RESEED_SEEDS),
            "panel_size_per_cfg": 5,
            "new_runs_per_cfg": 4,
        },
        "contract_id": CONTRACT_ID,
        "output_root": str(FLOWGEN_TRAINONLY_ROOT / OUTPUT_SUBDIR_ROOT),
        "completed": completed,
        "skipped_existing": skipped,
        "failed": failed,
        "runs": result_rows,
    }
    results_json_path = SUMMARY_ROOT / f"{campaign_id}_results.json"
    results_csv_path = SUMMARY_ROOT / f"{campaign_id}_results.csv"
    _write_json(results_json_path, results_payload)
    _write_rows_csv(results_csv_path, result_rows)

    print("\n" + "=" * 100)
    print("FlowGen train-only reseed final finished")
    print(f"Completed: {completed}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")
    print(f"Results JSON: {results_json_path}")
    print(f"Results CSV:  {results_csv_path}")

    if completed or skipped:
        print("\nOutput paths:")
        for entry in entries:
            if entry.status in {"completed", "skipped_existing"}:
                print(f"  - {entry.run_id}: {entry.output_dir}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
