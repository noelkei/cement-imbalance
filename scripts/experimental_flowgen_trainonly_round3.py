from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
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


OUTPUT_NAMESPACE = "experimental/train_only"
OUTPUT_SUBDIR_ROOT = "round3"
MODEL_FAMILY = "flowgen"
CONTRACT_ID = "flowgen_trainonly_round3_v1"
OBJECTIVE_METRIC_ID = "flowgen_trainonly_realism_round3"
DEFAULT_OFFICIAL_DATASET_NAME = "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"
TRAIN_ONLY_POLICY = "train_only"

FLOWGEN_TRAINONLY_ROOT = ROOT / "outputs" / "models" / "experimental" / "train_only" / "flowgen"
SUMMARY_ROOT = FLOWGEN_TRAINONLY_ROOT / "campaign_summaries" / "round3"

BASE_RUN_IDS = {
    "candidate_trainonly_1": "flowgen_trainonly_tpv1_ct1_base_seed6769",
    "candidate_trainonly_2": "flowgen_trainonly_tpv1_ct2_base_seed6769",
}

BASE_TOKENS = {
    "candidate_trainonly_1": "ct1",
    "candidate_trainonly_2": "ct2",
}

PREFERRED_POLICY_ORDER = [
    "R3A1_t06_w1x120",
    "R3A2_t06_clip125",
    "R3A3_t06_clip175",
    "R3B1_t06_ksx_light",
    "R3B2_t06_ksx_ksy_light",
    "R3C1_t06_mmdxy_light",
    "R3D1_now1x_rms_mmdheavy",
    "R3E1_t06_prior_ramp_light",
]

NON_POLICY_TRAIN_KEYS = {
    "log_training",
    "pretrained_path",
    "realism_bootstrap",
    "realism_rvr_bootstrap",
    "realism_seed_offset",
    "save_model",
    "save_results",
    "save_states",
    "seed",
    "skip_phase1",
}

LATENT_ALIAS_MAP = {
    "use_latent_mean_penalty": "use_mean_penalty",
    "latent_mean_weight": "mean_penalty_weight",
    "use_latent_mean_sq": "use_mean_sq",
    "use_latent_mean_abs": "use_mean_abs",
    "use_latent_std_penalty": "use_std_penalty",
    "latent_std_weight": "std_penalty_weight",
    "use_latent_std_sq": "use_std_sq",
    "use_latent_std_abs": "use_std_abs",
    "use_latent_skew_penalty": "use_skew_penalty",
    "latent_skew_weight": "skew_penalty_weight",
    "use_latent_skew_sq": "use_skew_sq",
    "use_latent_skew_abs": "use_skew_abs",
    "use_latent_kurtosis_penalty": "use_kurtosis_penalty",
    "latent_kurtosis_weight": "kurtosis_penalty_weight",
    "use_latent_kurtosis_sq": "use_kurtosis_sq",
    "use_latent_kurtosis_abs": "use_kurtosis_abs",
}

GATED_FIELDS = {
    "use_nll": ["nll_weight"],
    "use_logdet_penalty": ["logdet_penalty_weight"],
    "use_logpz_centering": ["logpz_centering_weight", "logpz_target"],
    "use_mean_penalty": ["mean_penalty_weight", "use_mean_abs", "use_mean_sq"],
    "use_std_penalty": ["std_penalty_weight", "use_std_abs", "use_std_sq"],
    "use_skew_penalty": ["skew_penalty_weight", "use_skew_abs", "use_skew_sq"],
    "use_kurtosis_penalty": ["kurtosis_penalty_weight", "use_kurtosis_abs", "use_kurtosis_sq"],
    "use_mmd_xy": ["mmd_xy_weight", "mmd_xy_norm", "mmd_xy_scales"],
    "use_corr_xy_pearson": ["corr_xy_pearson_weight"],
    "use_corr_xy_spearman": ["corr_xy_spearman_weight", "corr_xy_tau", "corr_xy_use_relative"],
    "use_mmd_x": ["mmd_x_weight"],
    "use_corr_pearson_x": ["corr_pearson_x_weight", "corr_pearson_use_relative_x"],
    "use_corr_spearman_x": ["corr_spearman_x_weight", "corr_spearman_use_relative_x"],
    "use_ks_x": ["ks_x_weight", "ks_grid_points_x", "ks_tau_x"],
    "use_w1_x": ["w1_x_weight", "w1_x_norm", "w1_x_softclip_s", "w1_x_clip_perdim", "w1_x_agg_softcap"],
    "use_mmd_y": ["mmd_y_weight"],
    "use_corr_pearson_y": ["corr_pearson_y_weight", "corr_pearson_use_relative_y"],
    "use_corr_spearman_y": ["corr_spearman_y_weight", "corr_spearman_use_relative_y"],
    "use_ks_y": ["ks_y_weight", "ks_grid_points_y", "ks_tau_y"],
    "use_w1_y": ["w1_y_weight", "w1_y_norm", "w1_y_softclip_s", "w1_y_clip_perdim", "w1_y_agg_softcap"],
}

FLOWGEN_TRAINONLY_POLICY_BASE_TRAINING = {
    "batch_size": 192,
    "learning_rate": 1.0e-4,
    "num_epochs": 3,
    "finetune_num_epochs": 100,
    "early_stopping_patience": 20,
    "lr_decay_patience": 10,
    "lr_decay_factor": 0.5,
    "min_improvement": 0.05,
    "min_improvement_floor": 0.0025,
    "lr_patience_factor": 0.8,
    "use_full_ref": False,
    "ref_min": 100,
    "syn_min": 100,
    "class_weighting": "uniform",
    "min_per_class": 1,
    "use_nll": True,
    "nll_weight": 1.0,
    "use_logdet_penalty": False,
    "logdet_penalty_weight": 0.0,
    "use_logdet_sq": True,
    "use_logdet_abs": True,
    "clamp_logabsdet_range": None,
    "use_logpz_centering": False,
    "logpz_centering_weight": 0.0,
    "logpz_target": None,
    "use_mean_penalty": False,
    "mean_penalty_weight": 2.5,
    "use_mean_sq": True,
    "use_mean_abs": True,
    "use_std_penalty": False,
    "std_penalty_weight": 1.25,
    "use_std_sq": True,
    "use_std_abs": True,
    "use_skew_penalty": False,
    "skew_penalty_weight": 0.004,
    "use_skew_sq": True,
    "use_skew_abs": True,
    "use_kurtosis_penalty": False,
    "kurtosis_penalty_weight": 0.0002,
    "use_kurtosis_sq": True,
    "use_kurtosis_abs": True,
    "use_mmd_as_ratio": False,
    "mmd_ratio_eps": 1.0e-6,
    "mmd_ratio_mode": "sq",
    "use_mmd_xy": False,
    "mmd_xy_weight": 0.05,
    "mmd_xy_norm": "iqr",
    "mmd_xy_scales": [0.5, 1.0, 2.0],
    "use_corr_xy_pearson": False,
    "corr_xy_pearson_weight": 0.0,
    "use_corr_xy_spearman": False,
    "corr_xy_spearman_weight": 0.0,
    "corr_xy_tau": 0.05,
    "corr_xy_use_relative": True,
    "use_mmd_x": True,
    "mmd_x_weight": 0.48,
    "use_corr_pearson_x": False,
    "corr_pearson_x_weight": 0.0,
    "corr_pearson_use_relative_x": True,
    "use_corr_spearman_x": False,
    "corr_spearman_x_weight": 0.0,
    "corr_spearman_use_relative_x": True,
    "use_ks_x": False,
    "ks_x_weight": 0.0,
    "use_w1_x": True,
    "w1_x_weight": 70.0,
    "w1_x_norm": "iqr",
    "ks_grid_points_x": 32,
    "ks_tau_x": 0.05,
    "use_mmd_y": True,
    "mmd_y_weight": 1.25,
    "use_corr_pearson_y": False,
    "corr_pearson_y_weight": 0.0,
    "corr_pearson_use_relative_y": True,
    "use_corr_spearman_y": False,
    "corr_spearman_y_weight": 0.0,
    "corr_spearman_use_relative_y": True,
    "use_ks_y": False,
    "ks_y_weight": 0.01,
    "ks_grid_points_y": 64,
    "ks_tau_y": 0.04,
    "use_w1_y": True,
    "w1_y_weight": 70.0,
    "w1_y_norm": "iqr",
    "ks_grid_points_all": 64,
    "ks_tau_all": 0.05,
    "w1_norm_all": "iqr",
    "realism_stride_batches": 1,
    "realism_stride_epochs": 1,
    "realism_scale_mode": "keep_mean",
    "realism_warmup_epochs": 0,
    "realism_ramp_epochs": 0,
    "realism_bootstrap": 10,
    "realism_rvr_bootstrap": 10,
    "realism_seed_offset": 0,
    "enforce_realism": True,
    "w1_x_softclip_s": 1.25,
    "w1_y_softclip_s": 1.25,
    "w1_x_clip_perdim": 2.0,
    "w1_y_clip_perdim": 2.0,
    "realism_z_trunc": 3.0,
    "w1_x_agg_softcap": 2.0,
    "w1_y_agg_softcap": 2.0,
}

FLOWGEN_TRAINONLY_POLICY_SPECS = {
    "R3A1_t06_w1x120": {
        "allowed_base_tokens": ["candidate_trainonly_1", "candidate_trainonly_2"],
        "training_overrides": {
            "w1_x_weight": 120.0,
            "w1_y_weight": 20.0,
            "mmd_x_weight": 0.58,
            "mmd_y_weight": 1.50,
            "w1_x_norm": "rms",
            "w1_y_norm": "rms",
            "w1_x_softclip_s": 1.0,
            "w1_y_softclip_s": 1.0,
            "w1_x_clip_perdim": 1.5,
            "w1_y_clip_perdim": 1.5,
            "w1_x_agg_softcap": 1.5,
            "w1_y_agg_softcap": 1.5,
        },
        "historical_origin": (
            "Round3 family A / variant 1. Direct exploitation of the T06 corridor after round2 showed the clearest "
            "X-trainaligned improvement there. This lowers W1_x from the round2 T06 setting while keeping the same "
            "RMS+tight-clip geometry to test whether KS can improve without losing the X gain."
        ),
        "historical_source_run_ids": [
            "flowgen_trainonly_tpv1_ct1_round2_t06_loww1x_rms_tightclip_seed6769_v1",
            "flowgen_trainonly_tpv1_ct2_round2_t06_loww1x_rms_tightclip_seed6769_v1",
        ],
    },
    "R3A2_t06_clip125": {
        "allowed_base_tokens": ["candidate_trainonly_1"],
        "training_overrides": {
            "w1_x_weight": 140.0,
            "w1_y_weight": 20.0,
            "mmd_x_weight": 0.58,
            "mmd_y_weight": 1.50,
            "w1_x_norm": "rms",
            "w1_y_norm": "rms",
            "w1_x_softclip_s": 1.0,
            "w1_y_softclip_s": 1.0,
            "w1_x_clip_perdim": 1.25,
            "w1_y_clip_perdim": 1.25,
            "w1_x_agg_softcap": 1.25,
            "w1_y_agg_softcap": 1.25,
        },
        "historical_origin": (
            "Round3 family A / variant 2. Tightens the successful T06 corridor further to test whether the "
            "remaining pain is concentrated in a handful of X dimensions that still need more aggressive clipping."
        ),
        "historical_source_run_ids": [
            "flowgen_trainonly_tpv1_ct1_round2_t06_loww1x_rms_tightclip_seed6769_v1",
        ],
    },
    "R3A3_t06_clip175": {
        "allowed_base_tokens": ["candidate_trainonly_1"],
        "training_overrides": {
            "w1_x_weight": 140.0,
            "w1_y_weight": 20.0,
            "mmd_x_weight": 0.58,
            "mmd_y_weight": 1.50,
            "w1_x_norm": "rms",
            "w1_y_norm": "rms",
            "w1_x_softclip_s": 1.0,
            "w1_y_softclip_s": 1.0,
            "w1_x_clip_perdim": 1.75,
            "w1_y_clip_perdim": 1.75,
            "w1_x_agg_softcap": 1.75,
            "w1_y_agg_softcap": 1.75,
        },
        "historical_origin": (
            "Round3 family A / variant 3. Symmetric upper-side bracket of the T06 clipping corridor. This checks "
            "whether round2 T06 was succeeding because it hit a sweet spot or simply because the RMS regime matters."
        ),
        "historical_source_run_ids": [
            "flowgen_trainonly_tpv1_ct1_round2_t06_loww1x_rms_tightclip_seed6769_v1",
        ],
    },
    "R3B1_t06_ksx_light": {
        "allowed_base_tokens": ["candidate_trainonly_1", "candidate_trainonly_2"],
        "training_overrides": {
            "w1_x_weight": 140.0,
            "w1_y_weight": 20.0,
            "mmd_x_weight": 0.58,
            "mmd_y_weight": 1.50,
            "w1_x_norm": "rms",
            "w1_y_norm": "rms",
            "w1_x_softclip_s": 1.0,
            "w1_y_softclip_s": 1.0,
            "w1_x_clip_perdim": 1.5,
            "w1_y_clip_perdim": 1.5,
            "w1_x_agg_softcap": 1.5,
            "w1_y_agg_softcap": 1.5,
            "use_ks_x": True,
            "ks_x_weight": 0.006,
            "ks_grid_points_x": 80,
            "ks_tau_x": 0.045,
        },
        "historical_origin": (
            "Round3 family B / variant 1. Keeps the T06 geometry but adds a very light KS_x term. The hypothesis is "
            "that round2 found the right X geometry but not enough explicit shape control to convert it into KS gains."
        ),
        "historical_source_run_ids": [
            "flowgen_trainonly_tpv1_ct1_round2_t06_loww1x_rms_tightclip_seed6769_v1",
            "flowgen_trainonly_tpv1_ct2_round2_t06_loww1x_rms_tightclip_seed6769_v1",
            "flowgen_trainonly_tpv1_ct1_round2_t03_now1x_ksx_ksy_seed6769_v1",
            "flowgen_trainonly_tpv1_ct2_round2_t03_now1x_ksx_ksy_seed6769_v1",
        ],
    },
    "R3B2_t06_ksx_ksy_light": {
        "allowed_base_tokens": ["candidate_trainonly_1", "candidate_trainonly_2"],
        "training_overrides": {
            "w1_x_weight": 140.0,
            "w1_y_weight": 20.0,
            "mmd_x_weight": 0.58,
            "mmd_y_weight": 1.50,
            "w1_x_norm": "rms",
            "w1_y_norm": "rms",
            "w1_x_softclip_s": 1.0,
            "w1_y_softclip_s": 1.0,
            "w1_x_clip_perdim": 1.5,
            "w1_y_clip_perdim": 1.5,
            "w1_x_agg_softcap": 1.5,
            "w1_y_agg_softcap": 1.5,
            "use_ks_x": True,
            "ks_x_weight": 0.006,
            "ks_grid_points_x": 80,
            "ks_tau_x": 0.045,
            "use_ks_y": True,
            "ks_y_weight": 0.006,
            "ks_grid_points_y": 80,
            "ks_tau_y": 0.040,
        },
        "historical_origin": (
            "Round3 family B / variant 2. Same idea as B1, but adding a light KS_y term as well to test whether "
            "the missing shape signal is joint X+Y rather than X-only."
        ),
        "historical_source_run_ids": [
            "flowgen_trainonly_tpv1_ct1_round2_t06_loww1x_rms_tightclip_seed6769_v1",
            "flowgen_trainonly_tpv1_ct2_round2_t06_loww1x_rms_tightclip_seed6769_v1",
            "flowgen_trainonly_tpv1_ct1_round2_t03_now1x_ksx_ksy_seed6769_v1",
            "flowgen_trainonly_tpv1_ct2_round2_t03_now1x_ksx_ksy_seed6769_v1",
        ],
    },
    "R3C1_t06_mmdxy_light": {
        "allowed_base_tokens": ["candidate_trainonly_1"],
        "training_overrides": {
            "w1_x_weight": 140.0,
            "w1_y_weight": 20.0,
            "mmd_x_weight": 0.58,
            "mmd_y_weight": 1.50,
            "w1_x_norm": "rms",
            "w1_y_norm": "rms",
            "w1_x_softclip_s": 1.0,
            "w1_y_softclip_s": 1.0,
            "w1_x_clip_perdim": 1.5,
            "w1_y_clip_perdim": 1.5,
            "w1_x_agg_softcap": 1.5,
            "w1_y_agg_softcap": 1.5,
            "use_mmd_xy": True,
            "mmd_xy_weight": 0.02,
        },
        "historical_origin": (
            "Round3 family C / variant 1. Reintroduces a very light MMD_XY term on top of the successful T06 "
            "geometry, explicitly avoiding the correlation losses that failed in round2 while still probing joint structure."
        ),
        "historical_source_run_ids": [
            "flowgen_trainonly_tpv1_ct1_round2_t06_loww1x_rms_tightclip_seed6769_v1",
            "flowgen_trainonly_tpv1_ct1_round2_t02_now1x_mmdheavy_mmdxy_seed6769_v1",
        ],
    },
    "R3D1_now1x_rms_mmdheavy": {
        "allowed_base_tokens": ["candidate_trainonly_1"],
        "training_overrides": {
            "use_w1_x": False,
            "w1_y_weight": 25.0,
            "mmd_x_weight": 0.65,
            "mmd_y_weight": 1.60,
            "w1_x_norm": "rms",
            "w1_y_norm": "rms",
            "w1_x_softclip_s": 1.0,
            "w1_y_softclip_s": 1.0,
            "w1_x_clip_perdim": 1.5,
            "w1_y_clip_perdim": 1.5,
            "w1_x_agg_softcap": 1.5,
            "w1_y_agg_softcap": 1.5,
        },
        "historical_origin": (
            "Round3 family D / variant 1. A deliberately orthogonal probe inspired by the early shuffled broad sweeps: "
            "drop W1_x entirely, keep the new RMS+tight geometry, and let MMD drive X. This tests whether the real gain "
            "in round2 came from geometry alone or from geometry interacting with residual W1_x."
        ),
        "historical_source_run_ids": [
            "flowgen_trainonly_tpv1_ct1_round2_t01_now1x_mmdheavy_seed6769_v1",
            "flowgen_trainonly_tpv1_ct1_round2_t06_loww1x_rms_tightclip_seed6769_v1",
        ],
    },
    "R3E1_t06_prior_ramp_light": {
        "allowed_base_tokens": ["candidate_trainonly_1", "candidate_trainonly_2"],
        "training_overrides": {
            "w1_x_weight": 140.0,
            "w1_y_weight": 20.0,
            "mmd_x_weight": 0.58,
            "mmd_y_weight": 1.50,
            "w1_x_norm": "rms",
            "w1_y_norm": "rms",
            "w1_x_softclip_s": 1.0,
            "w1_y_softclip_s": 1.0,
            "w1_x_clip_perdim": 1.5,
            "w1_y_clip_perdim": 1.5,
            "w1_x_agg_softcap": 1.5,
            "w1_y_agg_softcap": 1.5,
            "class_weighting": "prior",
            "realism_warmup_epochs": 5,
            "realism_ramp_epochs": 10,
        },
        "historical_origin": (
            "Round3 family E / variant 1. A lightweight, safer rescue of the round2 schedule hypothesis. It keeps "
            "the T06 geometry, uses prior class weighting, and shortens warmup/ramp to avoid the heavier failure pattern of T07."
        ),
        "historical_source_run_ids": [
            "flowgen_trainonly_tpv1_ct1_round2_t06_loww1x_rms_tightclip_seed6769_v1",
            "flowgen_trainonly_tpv1_ct2_round2_t06_loww1x_rms_tightclip_seed6769_v1",
            "flowgen_trainonly_tpv1_ct1_round2_t07_now1x_prior_ramp_seed6769_v1",
            "flowgen_trainonly_tpv1_ct2_round2_t07_now1x_prior_ramp_seed6769_v1",
        ],
    },
}


@dataclass
class BaseContext:
    token: str
    run_id: str
    base_seed: int
    run_dir: Path
    checkpoint_path: Path
    config_path: Path
    results_path: Path
    run_manifest_path: Path
    config: dict[str, Any]
    results: dict[str, Any]
    run_manifest: dict[str, Any]
    work_base_id: str
    paired_flowpre_source_id: str | None
    paired_flowpre_run_id: str | None
    paired_flowpre_seed: int | None


@dataclass
class TrainOnlyPolicy:
    policy_id: str
    allowed_base_tokens: list[str]
    training_template: dict[str, Any]
    historical_origin: str
    historical_source_run_ids: list[str]
    policy_signature: str


@dataclass
class PlanEntry:
    base_token: str
    base_run_id: str
    base_run_dir: str
    base_work_base_id: str
    base_checkpoint_path: str
    policy_id: str
    policy_signature: str
    historical_origin: str
    historical_source_run_ids: list[str]
    run_seed: int
    run_id: str
    output_dir: str
    existing_status: str
    error: str = ""
    status: str = "planned"
    result_paths: dict[str, str] = field(default_factory=dict)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "FlowGen train-only finetune round 3. Runs a small exploitation campaign "
            "centered on the round2 T06 geometry plus a few orthogonal probes."
        )
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--base",
        dest="base_tokens",
        action="append",
        choices=sorted(BASE_RUN_IDS),
        default=None,
        help="Optional subset of bases to run.",
    )
    ap.add_argument(
        "--policy",
        action="append",
        choices=PREFERRED_POLICY_ORDER,
        default=None,
        help="Optional subset of policies to run.",
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


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_round_floats(item) for item in value)
    if isinstance(value, dict):
        return {key: _round_floats(val) for key, val in sorted(value.items())}
    return value


def _resolve_required_artifact(run_dir: Path, run_id: str, kind: str) -> Path:
    if kind == "config":
        candidates = [run_dir / "config.yaml", run_dir / f"{run_id}.yaml"]
    elif kind == "results":
        candidates = [run_dir / "results.yaml", run_dir / f"{run_id}_results.yaml"]
    elif kind == "run_manifest":
        candidates = [run_dir / "run_manifest.json", run_dir / f"{run_id}_run_manifest.json"]
    elif kind == "checkpoint":
        candidates = [run_dir / "checkpoint.pt", run_dir / f"{run_id}.pt"]
    else:
        raise ValueError(f"Unsupported artifact kind: {kind}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing required base artifact '{kind}' under {run_dir}.")


def _resolve_base_seed(run_id: str, run_manifest: dict[str, Any], config: dict[str, Any]) -> int:
    candidates: list[tuple[str, int]] = []
    raw_candidates = [
        ("run_manifest.seed", run_manifest.get("seed")),
        ("config.seed", config.get("seed")),
        ("config.training.seed", (config.get("training") or {}).get("seed")),
    ]
    for source, raw_value in raw_candidates:
        if raw_value is None:
            continue
        candidates.append((source, int(raw_value)))

    if not candidates:
        raise ValueError(f"Unable to resolve base seed for {run_id}.")

    unique_values = sorted({value for _, value in candidates})
    if len(unique_values) != 1:
        detail = ", ".join(f"{source}={value}" for source, value in candidates)
        raise ValueError(f"Inconsistent base seed sources for {run_id}: {detail}")

    return unique_values[0]


def _resolve_base_context(token: str) -> BaseContext:
    run_id = BASE_RUN_IDS[token]
    run_dir = FLOWGEN_TRAINONLY_ROOT / "bases" / token / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Train-only FlowGen base not found: {run_dir}")

    checkpoint_path = _resolve_required_artifact(run_dir, run_id, "checkpoint")
    config_path = _resolve_required_artifact(run_dir, run_id, "config")
    results_path = _resolve_required_artifact(run_dir, run_id, "results")
    run_manifest_path = _resolve_required_artifact(run_dir, run_id, "run_manifest")

    config = load_yaml(config_path)
    results = load_yaml(results_path)
    run_manifest = load_json(run_manifest_path)

    if str(run_manifest.get("model_family")) != MODEL_FAMILY:
        raise ValueError(f"Base run manifest must have model_family='{MODEL_FAMILY}': {run_manifest_path}")
    if str(run_manifest.get("split_id")) != OFFICIAL_SPLIT_ID:
        raise ValueError(f"Base run manifest must have split_id='{OFFICIAL_SPLIT_ID}': {run_manifest_path}")
    if bool(run_manifest.get("test_enabled")):
        raise ValueError(f"Base run manifest unexpectedly enables test holdout: {run_manifest_path}")
    monitoring = run_manifest.get("monitoring") or {}
    if str(monitoring.get("policy")) != TRAIN_ONLY_POLICY:
        raise ValueError(f"Base run manifest must have monitoring.policy='{TRAIN_ONLY_POLICY}': {run_manifest_path}")
    if "model" not in config or not isinstance(config["model"], dict):
        raise ValueError(f"Base config missing model block: {config_path}")

    finetune_block = results.get("finetune") or {}
    if bool(finetune_block.get("enabled")):
        raise ValueError(f"Base results unexpectedly report finetune.enabled=true: {results_path}")

    base_seed = _resolve_base_seed(run_id, run_manifest, config)
    run_axes = run_manifest.get("run_level_axes") or {}
    bootstrap = config.get("bootstrap") or {}
    work_base_id = str(run_axes.get("flowgen_work_base_id") or bootstrap.get("work_base_id") or token)

    paired_flowpre_source_id = (
        run_axes.get("paired_flowpre_source_id") or bootstrap.get("paired_flowpre_source_id")
    )
    paired_flowpre_run_id = run_axes.get("paired_flowpre_run_id") or bootstrap.get("paired_flowpre_run_id")
    paired_flowpre_seed_raw = run_axes.get("paired_flowpre_seed") or bootstrap.get("paired_flowpre_seed")
    paired_flowpre_seed = None if paired_flowpre_seed_raw is None else int(paired_flowpre_seed_raw)

    return BaseContext(
        token=token,
        run_id=run_id,
        base_seed=base_seed,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        results_path=results_path,
        run_manifest_path=run_manifest_path,
        config=config,
        results=results,
        run_manifest=run_manifest,
        work_base_id=work_base_id,
        paired_flowpre_source_id=None if paired_flowpre_source_id is None else str(paired_flowpre_source_id),
        paired_flowpre_run_id=None if paired_flowpre_run_id is None else str(paired_flowpre_run_id),
        paired_flowpre_seed=paired_flowpre_seed,
    )


def _canonicalize_training_aliases(training_cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(training_cfg)
    for src_key, dst_key in LATENT_ALIAS_MAP.items():
        if src_key in normalized and dst_key not in normalized:
            normalized[dst_key] = normalized[src_key]
    for src_key in LATENT_ALIAS_MAP:
        normalized.pop(src_key, None)
    return normalized


def _normalize_effective_training_for_signature(training_cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = _canonicalize_training_aliases(training_cfg)
    signature_dict = {
        key: copy.deepcopy(value)
        for key, value in normalized.items()
        if key not in NON_POLICY_TRAIN_KEYS
    }
    for toggle_key, gated_fields in GATED_FIELDS.items():
        if not bool(signature_dict.get(toggle_key, False)):
            for gated_key in gated_fields:
                signature_dict.pop(gated_key, None)
    return _round_floats(signature_dict)


def _merge_training_template(overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(FLOWGEN_TRAINONLY_POLICY_BASE_TRAINING)
    merged.update(copy.deepcopy(overrides))
    return _canonicalize_training_aliases(merged)


def _build_policies() -> list[TrainOnlyPolicy]:
    policies: list[TrainOnlyPolicy] = []
    for policy_id in PREFERRED_POLICY_ORDER:
        spec = FLOWGEN_TRAINONLY_POLICY_SPECS[policy_id]
        training_template = _merge_training_template(spec["training_overrides"])
        policy_signature = _hash_payload(_normalize_effective_training_for_signature(training_template))
        policies.append(
            TrainOnlyPolicy(
                policy_id=policy_id,
                allowed_base_tokens=list(spec["allowed_base_tokens"]),
                training_template=training_template,
                historical_origin=str(spec["historical_origin"]),
                historical_source_run_ids=list(spec["historical_source_run_ids"]),
                policy_signature=policy_signature,
            )
        )
    return policies


def _build_effective_training(training_cfg: dict[str, Any], *, run_seed: int) -> dict[str, Any]:
    effective = _canonicalize_training_aliases(training_cfg)
    effective["seed"] = int(run_seed)
    effective["save_states"] = False
    effective["log_training"] = True
    effective["save_results"] = True
    effective["save_model"] = True
    return effective


def _policy_sort_key(policy_id: str) -> tuple[int, str]:
    return PREFERRED_POLICY_ORDER.index(policy_id), policy_id


def _policy_slug_for_run_id(policy_id: str) -> str:
    return policy_id.lower()


def _run_id_for(base: BaseContext, policy: TrainOnlyPolicy, run_seed: int) -> str:
    token = BASE_TOKENS[base.token]
    policy_slug = _policy_slug_for_run_id(policy.policy_id)
    return f"flowgen_trainonly_tpv1_{token}_round3_{policy_slug}_seed{int(run_seed)}_v1"


def _run_dir_for(base: BaseContext, run_id: str) -> Path:
    return FLOWGEN_TRAINONLY_ROOT / OUTPUT_SUBDIR_ROOT / base.token / run_id


def _run_materialization_status(base: BaseContext, run_id: str) -> str:
    run_dir = _run_dir_for(base, run_id)
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


def _build_config_payload(*, base: BaseContext, policy: TrainOnlyPolicy, run_seed: int) -> dict[str, Any]:
    training = _build_effective_training(policy.training_template, run_seed=run_seed)
    return {
        "model": copy.deepcopy(base.config["model"]),
        "training": training,
        "interpretability": copy.deepcopy(base.config.get("interpretability", {})),
        "seed": int(run_seed),
        "trainonly_training": {
            "mode": "flowgen_trainonly_round3_finetune",
            "policy_set_id": "flowgen_trainonly_round3_v1",
            "policy_id": policy.policy_id,
            "policy_signature": policy.policy_signature,
            "policy_origin": policy.historical_origin,
            "historical_source_run_ids": list(policy.historical_source_run_ids),
            "run_seed": int(run_seed),
            "seed_source": "base_flowgen_seed",
            "condition_col": "type",
            "target": "init",
            "monitoring_policy": TRAIN_ONLY_POLICY,
            "allow_test_holdout": False,
            "skip_phase1": True,
            "temperature_tuning": False,
            "materialize_datasets": False,
            "base_run_id": base.run_id,
            "base_seed": int(base.base_seed),
            "base_run_manifest": str(base.run_manifest_path),
            "base_checkpoint": str(base.checkpoint_path),
            "base_work_base_id": base.work_base_id,
            "paired_flowpre_source_id": base.paired_flowpre_source_id,
            "paired_flowpre_run_id": base.paired_flowpre_run_id,
            "paired_flowpre_seed": base.paired_flowpre_seed,
        },
    }


def _build_evaluation_context(*, base: BaseContext, policy: TrainOnlyPolicy, run_seed: int) -> dict[str, Any]:
    dataset_manifest_path = base.run_manifest.get("dataset_manifest_path")
    split_manifest_path = base.run_manifest.get("split_manifest_path")
    return {
        "dataset_name": str(base.run_manifest.get("dataset_name", DEFAULT_OFFICIAL_DATASET_NAME)),
        "dataset_manifest_path": dataset_manifest_path,
        "split_id": str(base.run_manifest.get("split_id", OFFICIAL_SPLIT_ID)),
        "split_manifest_path": split_manifest_path,
        "contract_id": CONTRACT_ID,
        "seed_set_id": f"{base.token}_round3_seed{int(run_seed)}",
        "base_config_id": f"{base.run_id}__{policy.policy_id.lower()}",
        "objective_metric_id": OBJECTIVE_METRIC_ID,
        "upstream_variant_fingerprint": base.run_manifest.get("variant_fingerprint"),
        "run_level_axes": {
            "phase": "trainonly_round3_finetune",
            "line": "experimental_train_only",
            "flowgen_base_run_id": base.run_id,
            "flowgen_base_work_base_id": base.work_base_id,
            "flowgen_base_seed": int(base.base_seed),
            "paired_flowpre_source_id": base.paired_flowpre_source_id,
            "paired_flowpre_run_id": base.paired_flowpre_run_id,
            "paired_flowpre_seed": base.paired_flowpre_seed,
            "policy_id": policy.policy_id,
            "policy_signature": policy.policy_signature,
            "policy_origin": policy.historical_origin,
            "historical_source_run_ids": list(policy.historical_source_run_ids),
            "run_seed": int(run_seed),
            "run_seed_source": "base_flowgen_seed",
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
        raise RuntimeError(f"Train-only FlowGen results must report finetune.enabled=true: {results_path}")
    for split in ("train", "val"):
        split_payload = results.get(split)
        if not isinstance(split_payload, dict):
            raise RuntimeError(f"Train-only FlowGen results are missing split '{split}': {results_path}")
        realism = split_payload.get("realism")
        if not isinstance(realism, dict):
            raise RuntimeError(
                f"Train-only FlowGen results are missing realism metrics for split '{split}': {results_path}"
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
        raise RuntimeError(f"Train-only FlowGen run is missing required artifacts under {run_dir}: {missing}")
    run_manifest = load_json(run_dir / "run_manifest.json")
    monitoring = run_manifest.get("monitoring") or {}
    if str(monitoring.get("policy")) != TRAIN_ONLY_POLICY:
        raise RuntimeError(f"Train-only FlowGen run manifest is missing monitoring.policy=train_only: {run_dir}")
    if bool(run_manifest.get("test_enabled")):
        raise RuntimeError(f"Train-only FlowGen run manifest unexpectedly enables test: {run_dir}")
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


def _base_plan_payload(base: BaseContext) -> dict[str, Any]:
    monitoring = base.run_manifest.get("monitoring") or {}
    return {
        "token": base.token,
        "run_id": base.run_id,
        "base_seed": int(base.base_seed),
        "work_base_id": base.work_base_id,
        "run_dir": str(base.run_dir),
        "checkpoint_path": str(base.checkpoint_path),
        "config_path": str(base.config_path),
        "results_path": str(base.results_path),
        "run_manifest_path": str(base.run_manifest_path),
        "dataset_name": base.run_manifest.get("dataset_name"),
        "split_id": base.run_manifest.get("split_id"),
        "monitoring_policy": monitoring.get("policy"),
        "test_enabled": bool(base.run_manifest.get("test_enabled")),
        "paired_flowpre_source_id": base.paired_flowpre_source_id,
        "paired_flowpre_run_id": base.paired_flowpre_run_id,
        "paired_flowpre_seed": base.paired_flowpre_seed,
    }


def _build_plan(
    *,
    bases: list[BaseContext],
    policies: list[TrainOnlyPolicy],
    allowed_policy_ids: set[str] | None,
) -> list[PlanEntry]:
    entries: list[PlanEntry] = []
    selected_policies = [
        policy for policy in policies if allowed_policy_ids is None or policy.policy_id in allowed_policy_ids
    ]

    for base in bases:
        run_seed = int(base.base_seed)
        for policy in selected_policies:
            if base.token not in policy.allowed_base_tokens:
                continue
            run_id = _run_id_for(base, policy, run_seed)
            entries.append(
                PlanEntry(
                    base_token=base.token,
                    base_run_id=base.run_id,
                    base_run_dir=str(base.run_dir),
                    base_work_base_id=base.work_base_id,
                    base_checkpoint_path=str(base.checkpoint_path),
                    policy_id=policy.policy_id,
                    policy_signature=policy.policy_signature,
                    historical_origin=policy.historical_origin,
                    historical_source_run_ids=list(policy.historical_source_run_ids),
                    run_seed=run_seed,
                    run_id=run_id,
                    output_dir=str(_run_dir_for(base, run_id)),
                    existing_status=_run_materialization_status(base, run_id),
                )
            )

    entries.sort(key=lambda entry: (entry.base_token, _policy_sort_key(entry.policy_id)))
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
    bases: list[BaseContext],
    policies: list[TrainOnlyPolicy],
    entries: list[PlanEntry],
    summary_paths: dict[str, Path],
) -> None:
    counts = _summarize_counts(entries)

    print("\n" + "=" * 100)
    print("FlowGen train-only finetune round 3")
    print("Run seed policy: reuse the FlowGen base seed")
    print("Condition: type | target: init | skip_phase1=True | allow_test_holdout=False")
    print("Monitoring policy: train_only | dataset materialization: disabled | temperature tuning: disabled")
    print("Realism bootstrap: 10/10 | policy goal: round3 targeted exploitation around T06 + orthogonal probes")

    print("\nBases detected:")
    for base in bases:
        print(
            f"  - {base.token}: {base.run_id} | base_seed={base.base_seed} | "
            f"work_base={base.work_base_id} | checkpoint={base.checkpoint_path}"
        )

    print("\nPolicy set:")
    for policy in policies:
        print(
            f"  - {policy.policy_id}: bases={','.join(policy.allowed_base_tokens)} | signature={policy.policy_signature[:12]} | "
            f"historical_sources={len(policy.historical_source_run_ids)}"
        )
        print(f"    provenance: {policy.historical_origin}")

    print("\nSelected policies per base:")
    by_base: dict[str, list[str]] = {}
    for entry in entries:
        by_base.setdefault(entry.base_token, []).append(entry.policy_id)
    for base_token, policy_ids in by_base.items():
        print(f"  - {base_token}: {', '.join(policy_ids)}")

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
    base: BaseContext,
    policy: TrainOnlyPolicy,
    entry: PlanEntry,
    device: str,
    verbose: bool,
) -> dict[str, str]:
    from training.train_flowgen import train_flowgen_pipeline

    config_payload = _build_config_payload(base=base, policy=policy, run_seed=entry.run_seed)
    evaluation_context = _build_evaluation_context(base=base, policy=policy, run_seed=entry.run_seed)
    temp_cfg_path = _write_temp_config(config_payload, stem=entry.run_id)
    model = None

    try:
        model = train_flowgen_pipeline(
            condition_col="type",
            config_filename=str(temp_cfg_path),
            base_name=entry.run_id,
            device=device,
            seed=int(entry.run_seed),
            verbose=verbose,
            allow_test_holdout=False,
            finetuning=True,
            skip_phase1=True,
            pretrained_path=str(base.checkpoint_path),
            evaluation_context=evaluation_context,
            monitoring_policy=TRAIN_ONLY_POLICY,
            output_namespace=OUTPUT_NAMESPACE,
            output_subdir=f"{OUTPUT_SUBDIR_ROOT}/{base.token}",
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


def _resolve_plan_inputs(args: argparse.Namespace) -> tuple[list[str], set[str] | None]:
    if args.run_one:
        return sorted(BASE_RUN_IDS), None
    selected_base_tokens = args.base_tokens or sorted(BASE_RUN_IDS)
    selected_policy_ids = None if not args.policy else set(args.policy)
    return selected_base_tokens, selected_policy_ids


def _resolve_entry_by_run_id(entries: list[PlanEntry], run_id: str) -> PlanEntry:
    matches = [entry for entry in entries if entry.run_id == run_id]
    if not matches:
        raise RuntimeError(f"Run id not found in current FlowGen train-only round3 plan: {run_id}")
    if len(matches) != 1:
        raise RuntimeError(f"Run id appears multiple times in current FlowGen train-only round3 plan: {run_id}")
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

    selected_base_tokens, selected_policy_ids = _resolve_plan_inputs(args)
    bases = [_resolve_base_context(token) for token in selected_base_tokens]
    policies = _build_policies()
    entries = _build_plan(
        bases=bases,
        policies=policies,
        allowed_policy_ids=selected_policy_ids,
    )
    if not entries:
        raise RuntimeError("No FlowGen train-only round3 runs remain after filtering.")

    if args.run_one:
        entry = _resolve_entry_by_run_id(entries, str(args.run_one))
        if entry.existing_status == "complete":
            print(f"Run already complete: {entry.run_id} -> {entry.output_dir}")
            return 0
        if entry.existing_status == "incomplete":
            removed = _reset_incomplete_run_dir(entry.output_dir)
            print(f"Removed incomplete run directory before retry: {removed}")
        base_lookup = {base.token: base for base in bases}
        policy_lookup = {policy.policy_id: policy for policy in policies}
        print(f"Running isolated FlowGen train-only round3 run: {entry.run_id}")
        _run_one(
            base=base_lookup[entry.base_token],
            policy=policy_lookup[entry.policy_id],
            entry=entry,
            device=args.device,
            verbose=verbose,
        )
        print(f"Completed isolated FlowGen train-only round3 run: {entry.run_id}")
        return 0

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    campaign_id = f"flowgen_trainonly_round3_{_campaign_timestamp()}"
    plan_json_path = SUMMARY_ROOT / f"{campaign_id}_plan.json"
    plan_csv_path = SUMMARY_ROOT / f"{campaign_id}_plan.csv"

    plan_payload = {
        "campaign_id": campaign_id,
        "script": str(Path(__file__).resolve()),
        "seed_policy": "per_base_flowgen_seed",
        "model_family": MODEL_FAMILY,
        "contract_id": CONTRACT_ID,
        "output_root": str(FLOWGEN_TRAINONLY_ROOT / OUTPUT_SUBDIR_ROOT),
        "bases": [_base_plan_payload(base) for base in bases],
        "policy_set": [
            {
                "policy_id": policy.policy_id,
                "policy_signature": policy.policy_signature,
                "historical_origin": policy.historical_origin,
                "historical_source_run_ids": list(policy.historical_source_run_ids),
            }
            for policy in policies
        ],
        "runs": [asdict(entry) for entry in entries],
    }
    _write_json(plan_json_path, plan_payload)
    _write_rows_csv(plan_csv_path, [asdict(entry) for entry in entries])

    _print_plan_summary(
        bases=bases,
        policies=policies,
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
    print(f"Starting FlowGen train-only round3 | planned runs={len(entries)}")

    for idx, entry in enumerate(entries, start=1):
        print("\n" + "-" * 100)
        print(f"[{idx}/{len(entries)}] {entry.run_id}")
        print(
            f"  base={entry.base_run_id} | policy={entry.policy_id} | "
            f"seed={entry.run_seed} | existing={entry.existing_status}"
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
        "seed_policy": "per_base_flowgen_seed",
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
    print("FlowGen train-only round3 finished")
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
