from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sets import DEFAULT_OFFICIAL_DATASET_NAME
from scripts.f6_common import OFFICIAL_SPLIT_ID, load_json, load_yaml

FLOWGEN_WORK_BASES = ("candidate_1", "candidate_2")

BOOTSTRAP_CONTRACT_ID = "flowgen_base_bootstrap_v1"
OUTPUT_ROOT = ROOT / "outputs" / "models" / "official" / "flowgen" / "bases"
OUTPUT_NAMESPACE = "official"
OUTPUT_SUBDIR = "bases"


BOOTSTRAP_TRAINING_BASE = {
    "batch_size": 256,
    "learning_rate": 1.0e-4,
    "num_epochs": 500,
    "early_stopping_patience": 20,
    "lr_decay_patience": 10,
    "lr_decay_factor": 0.5,
    "min_improvement": 0.05,
    "min_improvement_floor": 0.0025,
    "lr_patience_factor": 0.8,
    "save_states": False,
    "log_training": True,
    "save_results": True,
    "save_model": True,
    "use_nll": True,
    "nll_weight": 1.0,
    "use_logdet_penalty": False,
    "logdet_penalty_weight": 0.0,
    "use_logdet_abs": True,
    "use_logdet_sq": True,
    "clamp_logabsdet_range": None,
    "use_logpz_centering": False,
    "logpz_centering_weight": 0.0,
    "logpz_target": None,
    # The trainer still reads the legacy latent-penalty keys.
    "use_mean_penalty": False,
    "mean_penalty_weight": 2.5,
    "use_mean_abs": True,
    "use_mean_sq": True,
    "use_std_penalty": False,
    "std_penalty_weight": 1.25,
    "use_std_abs": True,
    "use_std_sq": True,
    "use_skew_penalty": False,
    "skew_penalty_weight": 0.004,
    "use_skew_abs": True,
    "use_skew_sq": True,
    "use_kurtosis_penalty": False,
    "kurtosis_penalty_weight": 0.0002,
    "use_kurtosis_abs": True,
    "use_kurtosis_sq": True,
    # Bootstrap bases are pure Phase-1 pretraining; realism stays off here.
    "use_full_ref": False,
    "ref_min": 100,
    "syn_min": 100,
    "class_weighting": "uniform",
    "min_per_class": 1,
    "use_mmd_as_ratio": False,
    "mmd_ratio_eps": 1.0e-6,
    "mmd_ratio_mode": "sq",
    "use_mmd_xy": False,
    "mmd_xy_weight": 0.0,
    "mmd_xy_norm": "iqr",
    "mmd_xy_scales": [0.5, 1.0, 2.0],
    "use_corr_xy_pearson": False,
    "corr_xy_pearson_weight": 0.0,
    "use_corr_xy_spearman": False,
    "corr_xy_spearman_weight": 0.0,
    "corr_xy_tau": 0.05,
    "corr_xy_use_relative": True,
    "use_mmd_x": False,
    "mmd_x_weight": 0.0,
    "use_corr_pearson_x": False,
    "corr_pearson_x_weight": 0.0,
    "corr_pearson_use_relative_x": True,
    "use_corr_spearman_x": False,
    "corr_spearman_x_weight": 0.0,
    "corr_spearman_use_relative_x": True,
    "use_ks_x": False,
    "ks_x_weight": 0.0,
    "use_w1_x": False,
    "w1_x_weight": 0.0,
    "w1_x_norm": "iqr",
    "ks_grid_points_x": 32,
    "ks_tau_x": 0.05,
    "use_mmd_y": False,
    "mmd_y_weight": 0.0,
    "use_corr_pearson_y": False,
    "corr_pearson_y_weight": 0.0,
    "corr_pearson_use_relative_y": True,
    "use_corr_spearman_y": False,
    "corr_spearman_y_weight": 0.0,
    "corr_spearman_use_relative_y": True,
    "use_ks_y": False,
    "ks_y_weight": 0.0,
    "ks_grid_points_y": 64,
    "ks_tau_y": 0.04,
    "use_w1_y": False,
    "w1_y_weight": 0.0,
    "w1_y_norm": "iqr",
    "ks_grid_points_all": 64,
    "ks_tau_all": 0.05,
    "w1_norm_all": "iqr",
    "realism_stride_batches": 1,
    "realism_stride_epochs": 1,
    "realism_scale_mode": "keep_mean",
    "realism_warmup_epochs": 0,
    "realism_ramp_epochs": 0,
    "realism_bootstrap": 100,
    "realism_rvr_bootstrap": 100,
    "realism_seed_offset": 0,
    "enforce_realism": False,
    "w1_x_softclip_s": 1.25,
    "w1_y_softclip_s": 1.25,
    "w1_x_clip_perdim": 2.0,
    "w1_y_clip_perdim": 2.0,
    "realism_z_trunc": 3.0,
    "w1_x_agg_softcap": 2.0,
    "w1_y_agg_softcap": 2.0,
}

BOOTSTRAP_INTERPRETABILITY = {
    "save_influence": False,
    "influence_step_fraction": 0.005,
    "sweep_range": [-3.0, 3.0],
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Bootstrap canonical FlowGen bases from the official FlowPre work bases "
            "(candidate_1 and candidate_2)."
        )
    )
    ap.add_argument(
        "--work-base",
        dest="work_bases",
        action="append",
        choices=FLOWGEN_WORK_BASES,
        default=None,
        help="Optional subset to run. If omitted, bootstraps both candidates.",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def _candidate_token(work_base_id: str) -> str:
    mapping = {
        "candidate_1": "c1",
        "candidate_2": "c2",
    }
    if work_base_id not in mapping:
        raise ValueError(f"Unsupported work base '{work_base_id}'.")
    return mapping[work_base_id]


def _expected_run_id(work_base_id: str, seed: int) -> str:
    return f"flowgen_tpv1_{_candidate_token(work_base_id)}_base_seed{int(seed)}"


def _resolve_candidate_seed(run_manifest: dict[str, Any], upstream_cfg: dict[str, Any]) -> tuple[int, str]:
    candidates = [
        ("run_manifest.seed", run_manifest.get("seed")),
        ("run_manifest.run_level_axes.seed", (run_manifest.get("run_level_axes") or {}).get("seed")),
        ("config.seed", upstream_cfg.get("seed")),
        ("config.training.seed", (upstream_cfg.get("training") or {}).get("seed")),
    ]
    for source, value in candidates:
        if value is None:
            continue
        return int(value), source

    run_id = str(run_manifest.get("run_id", ""))
    match = re.search(r"_seed(\d+)", run_id)
    if match:
        return int(match.group(1)), "run_id_tag"

    raise ValueError("Unable to resolve upstream candidate seed from run manifest or config.")


def _resolve_candidate_checkpoint(run_dir: Path, run_id: str) -> Path:
    preferred = [
        run_dir / f"{run_id}.pt",
        run_dir / f"{run_id}_finetuned.pt",
        run_dir / f"{run_id}_phase1.pt",
    ]
    for path in preferred:
        if path.exists():
            return path

    discovered = sorted(run_dir.glob("*.pt"))
    if len(discovered) == 1:
        return discovered[0]

    raise FileNotFoundError(f"Unable to resolve upstream checkpoint under {run_dir}.")


def _resolve_candidate_context(work_base_id: str) -> dict[str, Any]:
    from scripts.f6_flowgen_revalidate import (
        _resolve_flowgen_work_base_manifest,
        _resolve_repo_path,
        _validate_flowgen_work_base_manifest,
    )

    promotion_manifest_path = _resolve_flowgen_work_base_manifest(work_base_id)
    _validate_flowgen_work_base_manifest(promotion_manifest_path, work_base_id=work_base_id)

    promotion_manifest = load_json(promotion_manifest_path)
    run_manifest_path = _resolve_repo_path(str(promotion_manifest["source_run_manifest"]))
    run_manifest = load_json(run_manifest_path)
    run_dir = run_manifest_path.parent
    run_id = str(run_manifest["run_id"])

    if str(run_manifest.get("model_family")) != "flowpre":
        raise ValueError(f"Upstream run manifest must have model_family='flowpre': {run_manifest_path}")
    if str(run_manifest.get("split_id")) != OFFICIAL_SPLIT_ID:
        raise ValueError(f"Upstream run manifest must have split_id='{OFFICIAL_SPLIT_ID}': {run_manifest_path}")

    upstream_yaml_path = run_dir / f"{run_id}.yaml"
    if not upstream_yaml_path.exists():
        raise FileNotFoundError(f"Upstream FlowPre config not found: {upstream_yaml_path}")

    upstream_cfg = load_yaml(upstream_yaml_path)
    upstream_checkpoint_path = _resolve_candidate_checkpoint(run_dir, run_id)
    seed, seed_source = _resolve_candidate_seed(run_manifest, upstream_cfg)

    return {
        "work_base_id": work_base_id,
        "candidate_token": _candidate_token(work_base_id),
        "promotion_manifest_path": promotion_manifest_path,
        "promotion_manifest": promotion_manifest,
        "run_manifest_path": run_manifest_path,
        "run_manifest": run_manifest,
        "run_dir": run_dir,
        "run_id": run_id,
        "upstream_yaml_path": upstream_yaml_path,
        "upstream_cfg": upstream_cfg,
        "upstream_checkpoint_path": upstream_checkpoint_path,
        "seed": seed,
        "seed_source": seed_source,
    }


def _build_bootstrap_config(context: dict[str, Any]) -> dict[str, Any]:
    seed = int(context["seed"])
    training_cfg = copy.deepcopy(BOOTSTRAP_TRAINING_BASE)
    training_cfg["seed"] = seed

    return {
        "model": copy.deepcopy(context["upstream_cfg"]["model"]),
        "training": training_cfg,
        "interpretability": copy.deepcopy(BOOTSTRAP_INTERPRETABILITY),
        "seed": seed,
        "bootstrap": {
            "mode": "flowgen_base_bootstrap",
            "condition_col": "type",
            "target": "init",
            "skip_phase1": False,
            "finetuning": False,
            "work_base_id": context["work_base_id"],
            "paired_flowpre_source_id": context["promotion_manifest"]["source_id"],
            "paired_flowpre_run_id": context["run_id"],
            "paired_flowpre_seed": seed,
            "paired_flowpre_seed_source": context["seed_source"],
            "paired_flowpre_promotion_manifest": str(context["promotion_manifest_path"]),
            "paired_flowpre_run_manifest": str(context["run_manifest_path"]),
            "paired_flowpre_config": str(context["upstream_yaml_path"]),
            "paired_flowpre_checkpoint": str(context["upstream_checkpoint_path"]),
        },
    }


def _build_evaluation_context(context: dict[str, Any], *, run_id: str, seed: int) -> dict[str, Any]:
    run_manifest = context["run_manifest"]
    promotion_manifest = context["promotion_manifest"]

    return {
        "dataset_name": str(run_manifest.get("dataset_name", DEFAULT_OFFICIAL_DATASET_NAME)),
        "dataset_manifest_path": run_manifest.get("dataset_manifest_path"),
        "split_id": str(run_manifest.get("split_id", OFFICIAL_SPLIT_ID)),
        "split_manifest_path": run_manifest.get("split_manifest_path"),
        "contract_id": BOOTSTRAP_CONTRACT_ID,
        "seed_set_id": f"{context['work_base_id']}_bootstrap_seed",
        "base_config_id": f"{run_id}_bootstrap_config",
        "objective_metric_id": "flowgen_base_pretrain",
        "upstream_variant_fingerprint": run_manifest.get("variant_fingerprint"),
        "dataset_level_axes": run_manifest.get("dataset_level_axes"),
        "run_level_axes": {
            "phase": "bootstrap_base",
            "flowgen_work_base_id": context["work_base_id"],
            "paired_flowpre_source_id": promotion_manifest.get("source_id"),
            "paired_flowpre_run_id": context["run_id"],
            "paired_flowpre_seed": seed,
            "paired_flowpre_checkpoint_path": str(context["upstream_checkpoint_path"]),
        },
    }


def _target_run_dir(run_id: str) -> Path:
    return OUTPUT_ROOT / run_id


def _run_already_materialized(run_dir: Path) -> bool:
    required = [
        run_dir / "config.yaml",
        run_dir / "results.yaml",
        run_dir / "metrics_long.csv",
        run_dir / "run_manifest.json",
        run_dir / "checkpoint.pt",
        run_dir / "run.log",
    ]
    return run_dir.is_dir() and all(path.exists() for path in required)


def _copy_alias(src: str | Path | None, dst: Path) -> None:
    if src in (None, ""):
        return
    src_path = Path(src)
    if not src_path.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)


def _rewrite_manifest_config_path(manifest_path: Path, saved_config_path: str | Path | None) -> None:
    if saved_config_path in (None, "") or not manifest_path.exists():
        return
    payload = load_json(manifest_path)
    payload["config_path"] = str(Path(saved_config_path))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _materialize_artifact_aliases(model: Any) -> Path:
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
        _copy_alias(src, dst)

    return run_dir


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


def _bootstrap_one(work_base_id: str, *, device: str, verbose: bool) -> Path:
    from training.train_flowgen import train_flowgen_pipeline

    context = _resolve_candidate_context(work_base_id)
    run_id = _expected_run_id(work_base_id, context["seed"])
    target_dir = _target_run_dir(run_id)

    if _run_already_materialized(target_dir):
        print(f"⏭️  Skipping existing base: {target_dir}")
        return target_dir
    if target_dir.exists():
        raise FileExistsError(
            f"Target directory already exists but is incomplete: {target_dir}. "
            "Remove it manually before rerunning."
        )

    config = _build_bootstrap_config(context)
    evaluation_context = _build_evaluation_context(context, run_id=run_id, seed=context["seed"])
    temp_cfg_path = _write_temp_config(config, stem=run_id)

    print(
        f"▶️  Bootstrapping {run_id} from {work_base_id} "
        f"(seed={context['seed']} via {context['seed_source']})"
    )
    print(f"    promotion manifest: {context['promotion_manifest_path']}")
    print(f"    upstream run manifest: {context['run_manifest_path']}")
    print(f"    upstream checkpoint: {context['upstream_checkpoint_path']}")

    try:
        model = train_flowgen_pipeline(
            condition_col="type",
            config_filename=str(temp_cfg_path),
            base_name=run_id,
            device=device,
            seed=int(context["seed"]),
            verbose=verbose,
            allow_test_holdout=False,
            finetuning=False,
            skip_phase1=False,
            evaluation_context=evaluation_context,
            output_namespace=OUTPUT_NAMESPACE,
            output_subdir=OUTPUT_SUBDIR,
            fixed_run_id=run_id,
            log_in_run_dir=True,
        )
    finally:
        temp_cfg_path.unlink(missing_ok=True)

    run_dir = _materialize_artifact_aliases(model)
    print(f"✅ Bootstrapped base ready at: {run_dir}")
    return run_dir


def main() -> None:
    args = _parse_args()
    work_bases = args.work_bases or list(FLOWGEN_WORK_BASES)
    verbose = not args.quiet

    for work_base_id in work_bases:
        _bootstrap_one(work_base_id, device=args.device, verbose=verbose)


if __name__ == "__main__":
    main()
