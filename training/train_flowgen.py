# === train_flowgen.py (first part) ===
import gc
import os, random, secrets, math, json, shutil, re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import yaml

import time
from scipy.optimize import minimize


from models.flowgen import FlowGen
from losses.flowgen_loss import (
    flowgen_loss, _ks_w1_matrix, _mmd_rbf_biased, _pearson_corr, _spearman_corr, _fro_rel, _iqr,
    # NEW: joint + XY-block helpers used by eval
    _pearson_xyblock_fro_gap,         # XY Pearson Fro gap (abs/rel)
    _softspearman_xyblock_fro_gap,    # differentiable Spearman XY Fro gap (abs/rel)
)

from training.utils import (
    load_yaml_config,
    ROOT_PATH,
    flowpre_log,                    # we'll reuse logging utilities; can rename later if you want
    setup_training_logs_and_dirs,
    log_epoch_diagnostics,
)
from data.sets import (
    DEFAULT_OFFICIAL_DATASET_NAME,
    load_or_create_raw_splits,
    official_raw_bundle_manifest_path,
)
from nflows.distributions.normal import StandardNormal
from evaluation.metrics import (
    compute_flowgen_iso_rrmse_per_class as _canonical_compute_flowgen_iso_rrmse_per_class,
    compute_flowgen_latent_isotropy_stats as _canonical_compute_flowgen_latent_isotropy_stats,
    compute_flowgen_latent_isotropy_stats_per_class as _canonical_compute_flowgen_latent_isotropy_stats_per_class,
    compute_flowgen_split_metrics,
    format_iso_dict as _canonical_format_iso_dict,
    latent_isotropy_stats_from_z as _canonical_latent_isotropy_stats_from_z,
)
from evaluation.realism import compute_realism_metrics_for_set as _canonical_compute_realism_metrics_for_set
from evaluation.results import build_run_context, save_canonical_run_artifacts
from training.monitoring import (
    OFFICIAL_VAL_POLICY,
    TRAIN_ONLY_POLICY,
    ensure_holdout_policy,
    experimental_output_namespace,
    normalize_monitoring_policy,
    with_monitoring_context,
)
from evaluation.temporal_realism import (
    build_temporal_realism_block,
    resolve_temporal_realism_config,
    write_temporal_realism_sidecars,
)

# --- Device selection helper (MPS → CUDA → CPU) ---
def select_device(prefer: str | None = None) -> torch.device:
    """
    If `prefer` is provided, try to honor it with fallback to CPU.
    Otherwise prefer MPS, then CUDA, then CPU.
    """
    def has_mps():
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if isinstance(prefer, str):
        p = prefer.lower()
        if p in ("auto", "default", "best"):
            prefer = None  # fall through to auto logic below
        elif p == "mps" and has_mps():
            return torch.device("mps")
        elif p == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            prefer = None  # requested device unavailable → try auto

    # No explicit preference: pick best available
    if has_mps():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ───────────────────────────────────────────────────────────────────────────────
# Latent diagnostics (unchanged math; now driven by FlowGen.forward_xy)
# ───────────────────────────────────────────────────────────────────────────────
def _latent_isotropy_stats_from_z(z_np: np.ndarray) -> dict:
    return _canonical_latent_isotropy_stats_from_z(z_np)


def _latent_isotropy_stats(model: FlowGen, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> dict:
    return _canonical_compute_flowgen_latent_isotropy_stats(model, x, y, c)


def _latent_isotropy_stats_per_class(model: FlowGen, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> dict[int, dict]:
    return _canonical_compute_flowgen_latent_isotropy_stats_per_class(model, x, y, c)


# --- Helper: compute per-class isotropy RRMSE (from loss function outputs) ---
def _iso_rrmse_per_class_from_loss(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    loss_kwargs: dict,
):
    return _canonical_compute_flowgen_iso_rrmse_per_class(model, x, y, c, loss_kwargs)


# ───────────────────────────────────────────────────────────────────────────────
# Compact pretty-printer for per-class isotropy dicts (handles X/Y or legacy)
# ───────────────────────────────────────────────────────────────────────────────

def _format_iso_dict(iso_dict: Dict[int, Dict]) -> str:
    return _canonical_format_iso_dict(iso_dict)



# ───────────────────────────────────────────────────────────────────────────────
# Column filtering for FlowGen (keeps order: [condition, features..., y...])
# ───────────────────────────────────────────────────────────────────────────────
def filter_flowgen_columns(
    df: pd.DataFrame,
    cols_to_exclude: List[str],
    condition_col: str,
    y_cols: List[str] | str
) -> pd.DataFrame:
    """
    Removes specified columns while preserving original order, and reorders as:
      [condition_col] + feature_cols + y_cols

    Args:
        df: DataFrame with features, y targets, and condition_col.
        cols_to_exclude: columns to drop (e.g., ["post_cleaning_index"])
        condition_col: name of condition column (kept and moved to front)
        y_cols: list (or single name) of target columns to place at the end

    Returns:
        Filtered/reordered DataFrame.
    """
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    if condition_col not in df.columns:
        raise ValueError(f"Condition column '{condition_col}' not found in DataFrame.")
    for y in y_cols:
        if y not in df.columns:
            raise ValueError(f"Target column '{y}' not found in DataFrame.")

    excluded = set(cols_to_exclude or [])
    # never exclude condition_col or targets
    if condition_col in excluded:
        raise ValueError(f"Condition column '{condition_col}' cannot be excluded.")
    excluded = excluded.difference(set(y_cols))

    # base feature candidates = df minus excluded, minus condition/y
    feature_cols = [col for col in df.columns
                    if col not in excluded and col != condition_col and col not in y_cols]

    ordered_cols = [condition_col] + feature_cols + list(y_cols)
    return df[ordered_cols].copy()


# ───────────────────────────────────────────────────────────────────────────────
# Latent feature influence (sweep z_i; report influence on X and y separately)
# ───────────────────────────────────────────────────────────────────────────────
def compute_latent_feature_influence_flowgen(
    model: FlowGen,
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    feature_names_x: List[str],
    target_names: List[str],
    influence_step_fraction: float = 0.01,
    sweep_range: Tuple[float, float] = (-3.0, 3.0),
) -> dict:
    """
    For each latent dimension z_i, sweep it and estimate variability in decoded [X, y].
    Returns a nested dict with per-latent influence scores on each feature/target.

    Output shape:
      {
        "z_0": {
          "X": {feat_name: (raw_std, norm), ...},
          "y": {tgt_name:  (raw_std, norm), ...}
        },
        ...
      }
    """
    model.eval()
    device = model.device

    with torch.no_grad():
        z, _ = model.forward_xy(x.to(device), y.to(device), c.to(device))

    z = z.cpu()
    c = c.cpu()

    base_z = z.mean(dim=0)
    sweep_min, sweep_max = sweep_range
    step = max(1, int(round(1.0 / max(influence_step_fraction, 1e-4))))
    sweep_vals = torch.linspace(sweep_min, sweep_max, steps=step)

    influence = {}
    for i in range(z.shape[1]):
        z_sweep = base_z.repeat(step, 1)
        z_sweep[:, i] = sweep_vals

        # repeat a single class label for context (use the first)
        c_rep = c[:1].repeat(step)

        # inverse -> (x_rec, y_rec)
        (x_rec, y_rec), _ = model.inverse_xy(z_sweep.to(device), c_rep.to(device))
        x_rec = x_rec.detach().cpu().numpy()
        y_rec = y_rec.detach().cpu().numpy()

        # variability across sweep => raw influence
        x_std = np.std(x_rec, axis=0)
        y_std = np.std(y_rec, axis=0)

        # normalized within group (X and y separately)
        x_total = x_std.sum()
        y_total = y_std.sum()
        x_norm = (x_std / x_total) if x_total > 0 else np.zeros_like(x_std)
        y_norm = (y_std / y_total) if y_total > 0 else np.zeros_like(y_std)

        influence[f"z_{i}"] = {
            "X": {name: (round(float(r), 6), round(float(n), 6))
                  for name, r, n in zip(feature_names_x, x_std, x_norm)},
            "y": {name: (round(float(r), 6), round(float(n), 6))
                  for name, r, n in zip(target_names,    y_std, y_norm)},
        }
    return influence


# ───────────────────────────────────────────────────────────────────────────────
# Build FlowGen from config
# ───────────────────────────────────────────────────────────────────────────────
def build_flowgen_model(
    model_cfg: dict,
    x_dim: int,
    y_dim: int,
    num_classes: int,
    device: str = "cpu"
) -> FlowGen:
    """
    Constructs a FlowGen model from config dict (same arch defaults as FlowPre).
    """
    return FlowGen(
        x_dim=x_dim,
        y_dim=y_dim,
        num_classes=num_classes,
        embedding_dim=model_cfg.get("embedding_dim", 8),
        hidden_features=model_cfg.get("hidden_features", 64),
        num_layers=model_cfg.get("num_layers", 2),
        use_actnorm=model_cfg.get("use_actnorm", True),
        use_learnable_permutations=model_cfg.get("use_learnable_permutations", True),
        num_bins=model_cfg.get("num_bins", 8),
        tail_bound=model_cfg.get("tail_bound", 3.0),
        initial_affine_layers=model_cfg.get("initial_affine_layers", 2),
        affine_rq_ratio=tuple(model_cfg.get("affine_rq_ratio", [1, 3])),
        n_repeat_blocks=model_cfg.get("n_repeat_blocks", 4),
        final_rq_layers=model_cfg.get("final_rq_layers", 3),
        lulinear_finisher=model_cfg.get("lulinear_finisher", True),
        device=device
    )


# ───────────────────────────────────────────────────────────────────────────────
# Dataloader prep for FlowGen (matches how train_flowgen_model currently calls it)
# ───────────────────────────────────────────────────────────────────────────────

def prepare_flowgen_dataloader(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    condition_col: str,
    batch_size: int,
    device: torch.device,
    df_test: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None,
    y_cols: Optional[Union[str, List[str]]] = "init",
):
    """
    Validate and prepare tensors + dataloader for FlowGen training.

    Expects the *full* merged DataFrames (cXy_*), i.e., containing:
      - 'post_cleaning_index' (optional)
      - the condition column (e.g., 'type')
      - feature columns X (one or more)
      - target column(s) y (default name 'init', or list of names)

    The function infers feature columns as:
        feature_names_x = all columns EXCEPT {'post_cleaning_index', condition_col} ∪ set(y_cols)

    Returns (in the exact order the current training function unpacks):
        x_train, y_train,
        x_val,   y_val,
        x_test,  y_test,
        c_train, c_val, c_test,
        feature_names_x, target_names,
        train_dataset, train_dataloader
    """
    # --- Normalize target columns list
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    target_names = list(y_cols)

    # --- Basic schema checks
    def _check_required(df: pd.DataFrame, name: str):
        if condition_col not in df.columns:
            raise ValueError(f"Missing condition_col '{condition_col}' in {name}.")
        for yc in target_names:
            if yc not in df.columns:
                raise ValueError(f"Missing target column '{yc}' in {name}.")

    _check_required(df_train, "df_train")
    _check_required(df_val,   "df_val")
    if df_test is not None:
        _check_required(df_test, "df_test")

    # --- Determine feature columns from train (preserve original order)
    excluded = {"post_cleaning_index", condition_col, *target_names}
    feature_names_x = [col for col in df_train.columns if col not in excluded]

    # Validate the same feature schema in val/test
    def _features(df: pd.DataFrame) -> List[str]:
        return [col for col in df.columns if col not in excluded]

    if feature_names_x != _features(df_val):
        raise ValueError("Feature columns mismatch between df_train and df_val — names or order differ.")

    if df_test is not None and feature_names_x != _features(df_test):
        raise ValueError("Feature columns mismatch between df_train and df_test — names or order differ.")

    # --- Build tensors
    def _to_tensors(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.tensor(df[feature_names_x].values, dtype=torch.float32, device=device)
        y = torch.tensor(df[target_names].values,    dtype=torch.float32, device=device)
        c = torch.tensor(df[condition_col].values,   dtype=torch.long,    device=device)
        return x, y, c

    x_train, y_train, c_train = _to_tensors(df_train)
    x_val,   y_val,   c_val   = _to_tensors(df_val)

    x_test = y_test = c_test = None
    if df_test is not None:
        x_test, y_test, c_test = _to_tensors(df_test)

    # --- DataLoader (deterministic if seeded earlier)
    train_dataset = TensorDataset(x_train, y_train, c_train)

    g = torch.Generator(device="cpu")
    gen_seed = int(seed if seed is not None else torch.initial_seed())
    g.manual_seed(gen_seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
    )

    # Return in the exact order the current training code expects
    return (
        x_train, y_train,
        x_val,   y_val,
        x_test,  y_test,
        c_train, c_val, c_test,
        feature_names_x, target_names,
        train_dataset, train_dataloader
    )



# ───────────────────────────────────────────────────────────────────────────────
# Determinism helper
# ───────────────────────────────────────────────────────────────────────────────
def _maybe_set_seed(seed: int | None) -> int:
    """
    Deterministic seeding across CPU/CUDA. (MPS ignores cudnn flags.)
    """
    if seed is None:
        seed = secrets.randbits(64) % (2**63 - 1)

    random.seed(seed)

    np_seed = int(seed % (2**32 - 1))  # keep in [0, 2**32-2]
    np.random.seed(np_seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    return int(seed)


def _state_dict_to_cpu(state_dict: dict) -> dict:
    cpu_state: dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state[key] = value.detach().cpu().clone()
        else:
            cpu_state[key] = deepcopy(value)
    return cpu_state


def _release_training_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _loss_kwargs_from_train_cfg(train_cfg: dict) -> dict:
    """
    Map legacy training config keys to flowgen_loss(...) kwargs.
    Every value is tunable via config; defaults are only used via .get(...).
    """
    def _tuple_or_default(v, default):
        if v is None:
            return default
        if isinstance(v, (list, tuple)) and len(v) == 2:
            a, b = v
            return (float(a), float(b))
        return default

    clamp_range = _tuple_or_default(train_cfg.get("clamp_logabsdet_range", None), (-250.0, 250.0))

    kw = {
        # ---- base flow terms (existing keys) ----
        "use_nll": bool(train_cfg.get("use_nll", True)),
        "nll_weight": float(train_cfg.get("nll_weight", 1.0)),

        "use_logdet_penalty": bool(train_cfg.get("use_logdet_penalty", False)),
        "logdet_penalty_weight": float(train_cfg.get("logdet_penalty_weight", 0.0)),
        "use_logdet_sq": bool(train_cfg.get("use_logdet_sq", True)),
        "use_logdet_abs": bool(train_cfg.get("use_logdet_abs", True)),
        "clamp_logabsdet_range": clamp_range,

        "use_logpz_centering": bool(train_cfg.get("use_logpz_centering", False)),
        "logpz_centering_weight": float(train_cfg.get("logpz_centering_weight", 0.0)),
        "logpz_target": (None if train_cfg.get("logpz_target", None) in [None, "null"] else float(train_cfg["logpz_target"])),

        # ---- latent isotropy penalties (legacy names → new names) ----
        "use_latent_mean_penalty": bool(train_cfg.get("use_mean_penalty", False)),
        "latent_mean_weight": float(train_cfg.get("mean_penalty_weight", 0.0)),
        "use_latent_mean_sq": bool(train_cfg.get("use_mean_sq", True)),
        "use_latent_mean_abs": bool(train_cfg.get("use_mean_abs", True)),

        "use_latent_std_penalty": bool(train_cfg.get("use_std_penalty", False)),
        "latent_std_weight": float(train_cfg.get("std_penalty_weight", 0.0)),
        "use_latent_std_sq": bool(train_cfg.get("use_std_sq", True)),
        "use_latent_std_abs": bool(train_cfg.get("use_std_abs", True)),

        "use_latent_skew_penalty": bool(train_cfg.get("use_skew_penalty", False)),
        "latent_skew_weight": float(train_cfg.get("skew_penalty_weight", 0.0)),
        "use_latent_skew_sq": bool(train_cfg.get("use_skew_sq", True)),
        "use_latent_skew_abs": bool(train_cfg.get("use_skew_abs", True)),

        "use_latent_kurtosis_penalty": bool(train_cfg.get("use_kurtosis_penalty", False)),
        "latent_kurtosis_weight": float(train_cfg.get("kurtosis_penalty_weight", 0.0)),
        "use_latent_kurtosis_sq": bool(train_cfg.get("use_kurtosis_sq", True)),
        "use_latent_kurtosis_abs": bool(train_cfg.get("use_kurtosis_abs", True)),

        # ---- realism toggles (all tunable via config) ----

        # ---------------- NEW: MMD ratio mode ----------------
        "use_mmd_as_ratio": bool(train_cfg.get("use_mmd_as_ratio", False)),
        "mmd_ratio_eps": float(train_cfg.get("mmd_ratio_eps", 1e-6)),
        "mmd_ratio_mode": str(train_cfg.get("mmd_ratio_mode", "logsq")),
        # ---------------- NEW: joint XY realism ----------------
        "use_mmd_xy": bool(train_cfg.get("use_mmd_xy", False)),
        "mmd_xy_weight": float(train_cfg.get("mmd_xy_weight", 0.0)),
        "mmd_xy_norm": (None if (train_cfg.get("mmd_xy_norm", "iqr") in [None, "none", "None"])
                        else str(train_cfg.get("mmd_xy_norm", "iqr"))),
        "mmd_xy_scales": tuple(train_cfg.get("mmd_xy_scales", [0.5, 1.0, 2.0])),

        "use_corr_xy_pearson": bool(train_cfg.get("use_corr_xy_pearson", False)),
        "corr_xy_pearson_weight": float(train_cfg.get("corr_xy_pearson_weight", 0.0)),

        "use_corr_xy_spearman": bool(train_cfg.get("use_corr_xy_spearman", False)),
        "corr_xy_spearman_weight": float(train_cfg.get("corr_xy_spearman_weight", 0.0)),
        "corr_xy_tau": float(train_cfg.get("corr_xy_tau", 0.05)),

        "corr_xy_use_relative": bool(train_cfg.get("corr_xy_use_relative", True)),

        # X
        "use_mmd_x": bool(train_cfg.get("use_mmd_x", False)),
        "mmd_x_weight": float(train_cfg.get("mmd_x_weight", 0.0)),
        "use_corr_pearson_x": bool(train_cfg.get("use_corr_pearson_x", False)),
        "corr_pearson_x_weight": float(train_cfg.get("corr_pearson_x_weight", 0.0)),
        "corr_pearson_use_relative_x": bool(train_cfg.get("corr_pearson_use_relative_x", True)),
        "use_corr_spearman_x": bool(train_cfg.get("use_corr_spearman_x", False)),
        "corr_spearman_x_weight": float(train_cfg.get("corr_spearman_x_weight", 0.0)),
        "corr_spearman_use_relative_x": bool(train_cfg.get("corr_spearman_use_relative_x", True)),
        "use_ks_x": bool(train_cfg.get("use_ks_x", False)),
        "ks_x_weight": float(train_cfg.get("ks_x_weight", 0.0)),
        "use_w1_x": bool(train_cfg.get("use_w1_x", False)),
        "w1_x_weight": float(train_cfg.get("w1_x_weight", 0.0)),
        "w1_x_norm": str(train_cfg.get("w1_x_norm", "iqr")),
        "ks_grid_points_x": int(train_cfg.get("ks_grid_points_x", 64)),
        "ks_tau_x": float(train_cfg.get("ks_tau_x", 0.05)),

        # y
        "use_mmd_y": bool(train_cfg.get("use_mmd_y", False)),
        "mmd_y_weight": float(train_cfg.get("mmd_y_weight", 0.0)),
        "use_corr_pearson_y": bool(train_cfg.get("use_corr_pearson_y", False)),
        "corr_pearson_y_weight": float(train_cfg.get("corr_pearson_y_weight", 0.0)),
        "corr_pearson_use_relative_y": bool(train_cfg.get("corr_pearson_use_relative_y", True)),
        "use_corr_spearman_y": bool(train_cfg.get("use_corr_spearman_y", False)),
        "corr_spearman_y_weight": float(train_cfg.get("corr_spearman_y_weight", 0.0)),
        "corr_spearman_use_relative_y": bool(train_cfg.get("corr_spearman_use_relative_y", True)),
        "use_ks_y": bool(train_cfg.get("use_ks_y", False)),
        "ks_y_weight": float(train_cfg.get("ks_y_weight", 0.0)),
        "use_w1_y": bool(train_cfg.get("use_w1_y", False)),
        "w1_y_weight": float(train_cfg.get("w1_y_weight", 0.0)),
        "w1_y_norm": str(train_cfg.get("w1_y_norm", "iqr")),
        "ks_grid_points_y": int(train_cfg.get("ks_grid_points_y", 64)),
        "ks_tau_y": float(train_cfg.get("ks_tau_y", 0.05)),

        # ---- realism scheduling / striding ----  (NEW)
        "realism_stride_batches": int(train_cfg.get("realism_stride_batches", 1)),
        "realism_stride_epochs": int(train_cfg.get("realism_stride_epochs", 1)),
        "realism_scale_mode": str(train_cfg.get("realism_scale_mode", "keep_mean")),
        "realism_warmup_epochs": int(train_cfg.get("realism_warmup_epochs", 200)),
        "realism_ramp_epochs": int(train_cfg.get("realism_ramp_epochs", 50)),

        # ---- enforcement switch ----  (NEW)
        "enforce_realism": bool(train_cfg.get("enforce_realism", False)),

        # ---- reference sampling / eval knobs ----
        "use_full_ref": bool(train_cfg.get("use_full_ref", False)),
        # FIX: these two keys must match the loss signature (ref_min/syn_min), not ref_total/syn_total
        "ref_min": int(train_cfg.get("ref_min", 100)),  # CHANGED
        "syn_min": int(train_cfg.get("syn_min", 100)),  # CHANGED
        "class_weighting": str(train_cfg.get("class_weighting", "prior")),
        "min_per_class": int(train_cfg.get("min_per_class", 1)),

        # evaluation-only extras (keep as-is if you use them elsewhere)
        "realism_bootstrap": int(train_cfg.get("realism_bootstrap", 10)),
        "realism_seed_offset": int(train_cfg.get("realism_seed_offset", 0)),
        "realism_rvr_bootstrap": int(train_cfg.get("realism_rvr_bootstrap", 10)),

        "w1_x_softclip_s": float(train_cfg.get("w1_x_softclip_s", 0.75)),
        "w1_y_softclip_s": float(train_cfg.get("w1_y_softclip_s", 0.75)),
        "w1_x_clip_perdim": float(train_cfg.get("w1_x_clip_perdim", 2.0)),
        "w1_y_clip_perdim": float(train_cfg.get("w1_y_clip_perdim", 2.0)),
        "w1_x_agg_softcap": float(train_cfg.get("w1_x_agg_softcap", 2.0)),
        "w1_y_agg_softcap": float(train_cfg.get("w1_y_agg_softcap", 2.0)),
        "realism_z_trunc": float(train_cfg.get("realism_z_trunc", 2.5)),

        "ks_grid_points_all": int(train_cfg.get("ks_grid_points_all", max(
            int(train_cfg.get("ks_grid_points_x", 64)),
            int(train_cfg.get("ks_grid_points_y", 64)),
        ))),
        "ks_tau_all": float(train_cfg.get("ks_tau_all", 0.5 * (
                float(train_cfg.get("ks_tau_x", 0.05)) + float(train_cfg.get("ks_tau_y", 0.05))
        ))),
        "w1_norm_all": str(train_cfg.get("w1_norm_all", "iqr")),

    }
    return kw


def compute_realism_metrics_for_set(
    model,
    x_ref: torch.Tensor,
    y_ref: torch.Tensor,
    c_ref: torch.Tensor,
    *,
    loss_like_kwargs: dict,
    device: torch.device,
    seed: int | None = None,
) -> dict:
    return _canonical_compute_realism_metrics_for_set(
        model,
        x_ref,
        y_ref,
        c_ref,
        loss_like_kwargs=loss_like_kwargs,
        device=device,
        seed=seed,
    )

def compute_realism_metrics_for_set_with_temperature(
    model,
    x_ref: torch.Tensor,
    y_ref: torch.Tensor,
    c_ref: torch.Tensor,
    *,
    temps_by_class: dict,
    loss_like_kwargs: dict,
    device: torch.device,
    seed: int | None = None,
) -> dict:
    """
    EXACTLY like compute_realism_metrics_for_set, but synthetic samples
    are generated using FlowGen.sample_xy_with_temperature().
    """
    """
    Full-set realism metrics using ALL reference data, averaged over bootstrap replicates.
    For each replicate:
      • sample per-class synthetic with counts == full real per-class counts,
      • compute KS/W1 (normalized), Pearson/Spearman Fro (abs/rel), and MMD² ratio
        for overall | X | y and per-class,
      • RvR median uses the same σ as the replicate's RvS to keep ratios fair.
    All knobs come from loss_like_kwargs via .get(...).
    """

    EPS_METRIC = 1e-6  # metric-scale stabilization for MMD

    def _finite_and_clamp_to_real(xr_c, yr_c, xs_c, ys_c):
        # drop non-finite synth rows
        finite = torch.isfinite(xs_c).all(dim=1) & torch.isfinite(ys_c).all(dim=1)
        xs_c = xs_c[finite];
        ys_c = ys_c[finite]

        # robust per-class bounds from REAL (5 * IQR)
        q25x = xr_c.quantile(0.25, dim=0);
        q75x = xr_c.quantile(0.75, dim=0)
        iqr_x = (q75x - q25x).clamp_min(1e-6)
        lo_x = q25x - 5.0 * iqr_x;
        hi_x = q75x + 5.0 * iqr_x

        q25y = yr_c.quantile(0.25, dim=0);
        q75y = yr_c.quantile(0.75, dim=0)
        iqr_y = (q75y - q25y).clamp_min(1e-6)
        lo_y = q25y - 5.0 * iqr_y;
        hi_y = q75y + 5.0 * iqr_y

        # clamp synth to real support
        xs_c = xs_c.clamp(min=lo_x, max=hi_x)
        ys_c = ys_c.clamp(min=lo_y, max=hi_y)
        return xs_c, ys_c

    model.set_temperature_table_xy(temps_by_class)
    model.eval()

    t0 = time.perf_counter()
    print(
        f"[Realism] START | boots={loss_like_kwargs.get('realism_bootstrap')} "
        f"| rvr={loss_like_kwargs.get('realism_rvr_bootstrap')} "
        f"| device={device}"
    )

    # ---- RNG (tunable offset) ----
    gen_cpu = torch.Generator(device="cpu")
    if seed is not None:
        gen_cpu.manual_seed(int(seed) + int(loss_like_kwargs.get("realism_seed_offset", 0)))

    with torch.no_grad():
        Dx = int(x_ref.shape[1]); Dy = int(y_ref.shape[1]); Dxy = Dx + Dy

        # ---- eval knobs (all tunable) ----
        boots       = int(loss_like_kwargs.get("realism_bootstrap", 10))  # number of RvS replicates (averaged)
        rvr_boots   = int(loss_like_kwargs.get("realism_rvr_bootstrap", boots))  # RvR median bootstraps per replicate

        ks_grid_x   = int(loss_like_kwargs.get("ks_grid_points_x", 64))
        ks_tau_x    = float(loss_like_kwargs.get("ks_tau_x", 0.05))
        w1_norm_x   = str(loss_like_kwargs.get("w1_x_norm", "iqr"))

        ks_grid_y   = int(loss_like_kwargs.get("ks_grid_points_y", 64))
        ks_tau_y    = float(loss_like_kwargs.get("ks_tau_y", 0.05))
        w1_norm_y   = str(loss_like_kwargs.get("w1_y_norm", "iqr"))

        ks_grid_all = int(loss_like_kwargs.get("ks_grid_points_all", max(ks_grid_x, ks_grid_y)))
        ks_tau_all  = float(loss_like_kwargs.get("ks_tau_all", 0.5 * (ks_tau_x + ks_tau_y)))
        w1_norm_all = str(loss_like_kwargs.get("w1_norm_all", "iqr"))

        # ---- full real per-class pools (no subsampling) ----
        classes = [int(ci) for ci in torch.unique(c_ref).tolist()]
        counts = {cls: int((c_ref == cls).sum().item()) for cls in classes}

        real_x_by_cls = {cls: x_ref[c_ref == cls] for cls in classes}
        real_y_by_cls = {cls: y_ref[c_ref == cls] for cls in classes}

        Xr_all = torch.cat([real_x_by_cls[cls] for cls in classes], dim=0)
        Yr_all = torch.cat([real_y_by_cls[cls] for cls in classes], dim=0)

        real_all_cat = torch.cat([Xr_all, Yr_all], dim=1)

        # --- global denominators (from full real pools; fixed across reps/classes) ---
        denom_all_global = _iqr(real_all_cat).to(torch.float32).clamp_min(1e-4)
        denom_x_global = _iqr(Xr_all).to(torch.float32).clamp_min(1e-4)
        denom_y_global = _iqr(Yr_all).to(torch.float32).clamp_min(1e-4)

        # ---- accumulator helpers (no nested defs) ----
        def _empty_suite(include_corr: bool) -> Dict[str, Optional[float]]:
            base = {
                "ks_mean": 0.0, "ks_median": 0.0,
                "w1_mean": 0.0, "w1_median": 0.0,
                "mmd2_rvs": 0.0, "mmd2_rvr_med": 0.0, "mmd2_ratio": 0.0,
                # NEW: XY block corr (filled only for “overall”; kept here for shape consistency)
                "xy_pearson_fro": 0.0, "xy_pearson_fro_rel": 0.0,
                "xy_spearman_fro": 0.0, "xy_spearman_fro_rel": 0.0,
                "_xycorr_count": 0,
            }
            if include_corr:
                base.update({
                    "pearson_fro": 0.0, "pearson_fro_rel": 0.0,
                    "spearman_fro": 0.0, "spearman_fro_rel": 0.0,
                    "_corr_count": 0,
                })
            return base

        overall_acc = _empty_suite(include_corr=False)
        x_acc       = _empty_suite(include_corr=True)
        y_acc       = _empty_suite(include_corr=True)
        per_class_acc: Dict[int, Dict[str, Dict[str, Optional[float]]]] = {
            cls: {
                "overall": _empty_suite(include_corr=False),
                "x": _empty_suite(include_corr=True),
                "y": _empty_suite(include_corr=True),
            } for cls in counts.keys()
        }

        # ======================================================
        # Replicates: generate full-size synthetic and evaluate
        # ======================================================
        for _rep in range(boots):
            rep_t0 = time.perf_counter()
            print(f"[Realism] bootstrap {_rep + 1}/{boots} – generating synth...")

            # ---- generate synthetic per class with FULL real counts ----
            synth_x_by_cls: Dict[int, torch.Tensor] = {}
            synth_y_by_cls: Dict[int, torch.Tensor] = {}
            for cls, n_real in counts.items():
                ns = int(n_real)
                if ns <= 0:
                    # keep empty tensors if class absent (safety)
                    synth_x_by_cls[cls] = real_x_by_cls[cls].new_zeros((0, Dx))
                    synth_y_by_cls[cls] = real_y_by_cls[cls].new_zeros((0, Dy))
                    continue

                c_s = torch.full((ns,), cls, dtype=torch.long, device=device)

                # ONLY CHANGE
                xs_c, ys_c = model.sample_xy_with_temperature(ns, c_s)

                xs_c, ys_c = _finite_and_clamp_to_real(real_x_by_cls[cls], real_y_by_cls[cls], xs_c, ys_c)
                synth_x_by_cls[cls] = xs_c
                synth_y_by_cls[cls] = ys_c

            Xs_all = torch.cat([synth_x_by_cls[cls] for cls in counts.keys()], dim=0)
            Ys_all = torch.cat([synth_y_by_cls[cls] for cls in counts.keys()], dim=0)
            synth_all_cat = torch.cat([Xs_all, Ys_all], dim=1)

            # ---------- helper to compute RvR median with fixed sigma ----------
            def _mmd_rvr_median(real_mat: torch.Tensor, sigma: float, n_draws: int) -> float:
                n = real_mat.size(0)
                # draw with replacement, size = n each time (MPS-safe: sample on CPU)
                vals = []
                for _ in range(n_draws):
                    idx1 = torch.randint(0, n, (n,), generator=gen_cpu, device="cpu").to(device)
                    idx2 = torch.randint(0, n, (n,), generator=gen_cpu, device="cpu").to(device)
                    r1 = real_mat.index_select(0, idx1)
                    r2 = real_mat.index_select(0, idx2)
                    m2, _ = _mmd_rbf_biased(r1, r2, sigma=sigma)
                    vals.append(float(m2.detach()))
                return float(real_mat.new_tensor(vals).median().detach())

            # ===================== overall =====================
            ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                real_all_cat, synth_all_cat,
                grid_points=ks_grid_all, tau=ks_tau_all, norm=w1_norm_all,
                denom_override=denom_all_global
            )
            mmd2_rvs_all, sigma_all = _mmd_rbf_biased(real_all_cat, synth_all_cat)
            mmd2_rvr_med_all = _mmd_rvr_median(real_all_cat, sigma_all, rvr_boots)
            overall_acc["ks_mean"]      += float(ks_mean.detach())
            overall_acc["ks_median"]    += float(ks_med.detach())
            overall_acc["w1_mean"]      += float(w1_mean.detach())
            overall_acc["w1_median"]    += float(w1_med.detach())
            overall_acc["mmd2_rvs"]     += float(mmd2_rvs_all.detach())
            overall_acc["mmd2_rvr_med"] += float(mmd2_rvr_med_all)
            overall_acc["mmd2_ratio"]   += float((mmd2_rvs_all / (mmd2_rvr_med_all + EPS_METRIC)).detach())

            # ===================== X-only =====================
            ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                Xr_all, Xs_all,
                grid_points=ks_grid_x, tau=ks_tau_x, norm=w1_norm_x,
                denom_override=denom_x_global
            )
            mmd2_rvs_x, sigma_x = _mmd_rbf_biased(Xr_all, Xs_all)
            mmd2_rvr_med_x = _mmd_rvr_median(Xr_all, sigma_x, rvr_boots)
            x_acc["ks_mean"]      += float(ks_mean.detach())
            x_acc["ks_median"]    += float(ks_med.detach())
            x_acc["w1_mean"]      += float(w1_mean.detach())
            x_acc["w1_median"]    += float(w1_med.detach())
            x_acc["mmd2_rvs"]     += float(mmd2_rvs_x.detach())
            x_acc["mmd2_rvr_med"] += float(mmd2_rvr_med_x)
            x_acc["mmd2_ratio"]   += float((mmd2_rvs_x / (mmd2_rvr_med_x + EPS_METRIC)).detach())

            if Dx >= 2:
                Cr = _pearson_corr(Xr_all);  Cs = _pearson_corr(Xs_all)
                pe_abs, pe_rel = _fro_rel(Cr, Cs)
                Crs = _spearman_corr(Xr_all); Css = _spearman_corr(Xs_all)
                sp_abs, sp_rel = _fro_rel(Crs, Css)
                x_acc["pearson_fro"]     += float(pe_abs.detach())
                x_acc["pearson_fro_rel"] += float(pe_rel.detach())
                x_acc["spearman_fro"]    += float(sp_abs.detach())
                x_acc["spearman_fro_rel"]+= float(sp_rel.detach())
                x_acc["_corr_count"]     += 1  # mark that we accumulated corr

            # ===================== y-only =====================
            ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                Yr_all, Ys_all,
                grid_points=ks_grid_y, tau=ks_tau_y, norm=w1_norm_y,
                denom_override=denom_y_global
            )
            mmd2_rvs_y, sigma_y = _mmd_rbf_biased(Yr_all, Ys_all)
            mmd2_rvr_med_y = _mmd_rvr_median(Yr_all, sigma_y, rvr_boots)
            y_acc["ks_mean"]      += float(ks_mean.detach())
            y_acc["ks_median"]    += float(ks_med.detach())
            y_acc["w1_mean"]      += float(w1_mean.detach())
            y_acc["w1_median"]    += float(w1_med.detach())
            y_acc["mmd2_rvs"]     += float(mmd2_rvs_y.detach())
            y_acc["mmd2_rvr_med"] += float(mmd2_rvr_med_y)
            y_acc["mmd2_ratio"]   += float((mmd2_rvs_y / (mmd2_rvr_med_y + EPS_METRIC)).detach())

            if Dy >= 2:
                Cr = _pearson_corr(Yr_all);  Cs = _pearson_corr(Ys_all)
                pe_abs, pe_rel = _fro_rel(Cr, Cs)
                Crs = _spearman_corr(Yr_all); Css = _spearman_corr(Ys_all)
                sp_abs, sp_rel = _fro_rel(Crs, Css)
                y_acc["pearson_fro"]     += float(pe_abs.detach())
                y_acc["pearson_fro_rel"] += float(pe_rel.detach())
                y_acc["spearman_fro"]    += float(sp_abs.detach())
                y_acc["spearman_fro_rel"]+= float(sp_rel.detach())
                y_acc["_corr_count"]     += 1

            # === overall (concat) — existing KS/W1 & MMD keep as-is ===

            # NEW XY correlation (on the X↔y block):
            pe_abs_xy, pe_rel_xy = _pearson_xyblock_fro_gap(
                model, Xr_all, Yr_all, Xs_all, Ys_all, relative=True
            )
            sp_abs_xy, sp_rel_xy = _softspearman_xyblock_fro_gap(
                model, Xr_all, Yr_all, Xs_all, Ys_all, tau=float(loss_like_kwargs.get("corr_xy_tau", 0.05)),
                relative=True
            )
            overall_acc["xy_pearson_fro"] += float(pe_abs_xy.detach())
            overall_acc["xy_pearson_fro_rel"] += float(pe_rel_xy.detach())
            overall_acc["xy_spearman_fro"] += float(sp_abs_xy.detach())
            overall_acc["xy_spearman_fro_rel"] += float(sp_rel_xy.detach())
            overall_acc["_xycorr_count"] += 1

            # ===================== per-class =====================
            for cls in counts.keys():
                xr_c = real_x_by_cls[cls];  yr_c = real_y_by_cls[cls]
                xs_c = synth_x_by_cls[cls]; ys_c = synth_y_by_cls[cls]

                # overall (concat X|y)
                rc_all = torch.cat([xr_c, yr_c], dim=1)
                sc_all = torch.cat([xs_c, ys_c], dim=1)
                ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                    rc_all, sc_all,
                    grid_points=ks_grid_all, tau=ks_tau_all, norm=w1_norm_all,
                    denom_override=denom_all_global
                )
                mmd2_rvs, sigma = _mmd_rbf_biased(rc_all, sc_all)
                mmd2_rvr_med = _mmd_rvr_median(rc_all, sigma, rvr_boots)

                pc_over = per_class_acc[cls]["overall"]
                pc_over["ks_mean"]      += float(ks_mean.detach())
                pc_over["ks_median"]    += float(ks_med.detach())
                pc_over["w1_mean"]      += float(w1_mean.detach())
                pc_over["w1_median"]    += float(w1_med.detach())
                pc_over["mmd2_rvs"]     += float(mmd2_rvs.detach())
                pc_over["mmd2_rvr_med"] += float(mmd2_rvr_med)
                pc_over["mmd2_ratio"]   += float((mmd2_rvs / (mmd2_rvr_med + EPS_METRIC)).detach())

                # X-only
                ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                    xr_c, xs_c,
                    grid_points=ks_grid_x, tau=ks_tau_x, norm=w1_norm_x,
                    denom_override=denom_x_global
                )
                mmd2_rvs_x, sigma_x = _mmd_rbf_biased(xr_c, xs_c)
                mmd2_rvr_med_x = _mmd_rvr_median(xr_c, sigma_x, rvr_boots)

                pc_x = per_class_acc[cls]["x"]
                pc_x["ks_mean"]      += float(ks_mean.detach())
                pc_x["ks_median"]    += float(ks_med.detach())
                pc_x["w1_mean"]      += float(w1_mean.detach())
                pc_x["w1_median"]    += float(w1_med.detach())
                pc_x["mmd2_rvs"]     += float(mmd2_rvs_x.detach())
                pc_x["mmd2_rvr_med"] += float(mmd2_rvr_med_x)
                pc_x["mmd2_ratio"]   += float((mmd2_rvs_x / (mmd2_rvr_med_x + EPS_METRIC)).detach())

                if xr_c.shape[1] >= 2:
                    Cr = _pearson_corr(xr_c);  Cs = _pearson_corr(xs_c)
                    pe_abs, pe_rel = _fro_rel(Cr, Cs)
                    Crs = _spearman_corr(xr_c); Css = _spearman_corr(xs_c)
                    sp_abs, sp_rel = _fro_rel(Crs, Css)
                    pc_x["pearson_fro"]      += float(pe_abs.detach())
                    pc_x["pearson_fro_rel"]  += float(pe_rel.detach())
                    pc_x["spearman_fro"]     += float(sp_abs.detach())
                    pc_x["spearman_fro_rel"] += float(sp_rel.detach())
                    pc_x["_corr_count"]      += 1

                # y-only
                ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                    yr_c, ys_c,
                    grid_points=ks_grid_y, tau=ks_tau_y, norm=w1_norm_y,
                    denom_override=denom_y_global
                )
                mmd2_rvs_y, sigma_y = _mmd_rbf_biased(yr_c, ys_c)
                mmd2_rvr_med_y = _mmd_rvr_median(yr_c, sigma_y, rvr_boots)

                pc_y = per_class_acc[cls]["y"]
                pc_y["ks_mean"]      += float(ks_mean.detach())
                pc_y["ks_median"]    += float(ks_med.detach())
                pc_y["w1_mean"]      += float(w1_mean.detach())
                pc_y["w1_median"]    += float(w1_med.detach())
                pc_y["mmd2_rvs"]     += float(mmd2_rvs_y.detach())
                pc_y["mmd2_rvr_med"] += float(mmd2_rvr_med_y)
                pc_y["mmd2_ratio"]   += float((mmd2_rvs_y / (mmd2_rvr_med_y + EPS_METRIC)).detach())

                if yr_c.shape[1] >= 2:
                    Cr = _pearson_corr(yr_c);  Cs = _pearson_corr(ys_c)
                    pe_abs, pe_rel = _fro_rel(Cr, Cs)
                    Crs = _spearman_corr(yr_c); Css = _spearman_corr(ys_c)
                    sp_abs, sp_rel = _fro_rel(Crs, Css)
                    pc_y["pearson_fro"]      += float(pe_abs.detach())
                    pc_y["pearson_fro_rel"]  += float(pe_rel.detach())
                    pc_y["spearman_fro"]     += float(sp_abs.detach())
                    pc_y["spearman_fro_rel"] += float(sp_rel.detach())
                    pc_y["_corr_count"]      += 1

                # per-class overall XY block correlation
                pe_abs_xy, pe_rel_xy = _pearson_xyblock_fro_gap(
                    model, xr_c, yr_c, xs_c, ys_c, relative=True
                )
                sp_abs_xy, sp_rel_xy = _softspearman_xyblock_fro_gap(
                    model, xr_c, yr_c, xs_c, ys_c, tau=float(loss_like_kwargs.get("corr_xy_tau", 0.05)), relative=True
                )
                pc_over["xy_pearson_fro"] += float(pe_abs_xy.detach())
                pc_over["xy_pearson_fro_rel"] += float(pe_rel_xy.detach())
                pc_over["xy_spearman_fro"] += float(sp_abs_xy.detach())
                pc_over["xy_spearman_fro_rel"] += float(sp_rel_xy.detach())
                pc_over["_xycorr_count"] += 1

            rep_elapsed = time.perf_counter() - rep_t0
            print(f"[Realism] bootstrap {_rep + 1}/{boots} DONE in {rep_elapsed:.1f}s")

        # ---- finalize averages over replicates ----
        def _finalize_suite(acc: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
            out = {
                "ks_mean": acc["ks_mean"] / boots,
                "ks_median": acc["ks_median"] / boots,
                "w1_mean": acc["w1_mean"] / boots,
                "w1_median": acc["w1_median"] / boots,
                "pearson_fro": None, "pearson_fro_rel": None,
                "spearman_fro": None, "spearman_fro_rel": None,
                # NEW: XY block corr (only for “overall”, so may stay None elsewhere)
                "xy_pearson_fro": None, "xy_pearson_fro_rel": None,
                "xy_spearman_fro": None, "xy_spearman_fro_rel": None,
                "mmd2_rvs": acc["mmd2_rvs"] / boots,
                "mmd2_rvr_med": acc["mmd2_rvr_med"] / boots,
                "mmd2_ratio": acc["mmd2_ratio"] / boots,
            }
            if "_corr_count" in acc and acc["_corr_count"] > 0:
                k = float(acc["_corr_count"])
                out["pearson_fro"] = acc["pearson_fro"] / k
                out["pearson_fro_rel"] = acc["pearson_fro_rel"] / k
                out["spearman_fro"] = acc["spearman_fro"] / k
                out["spearman_fro_rel"] = acc["spearman_fro_rel"] / k
            if "_xycorr_count" in acc and acc["_xycorr_count"] > 0:
                kxy = float(acc["_xycorr_count"])
                out["xy_pearson_fro"] = acc["xy_pearson_fro"] / kxy
                out["xy_pearson_fro_rel"] = acc["xy_pearson_fro_rel"] / kxy
                out["xy_spearman_fro"] = acc["xy_spearman_fro"] / kxy
                out["xy_spearman_fro_rel"] = acc["xy_spearman_fro_rel"] / kxy
            return {kk: (None if vv is None else float(vv)) for kk, vv in out.items()}

        realism = {
            "overall": _finalize_suite(overall_acc),
            "x": _finalize_suite(x_acc),
            "y": _finalize_suite(y_acc),
            "per_class": {},
        }
        for cls, suites in per_class_acc.items():
            realism["per_class"][int(cls)] = {
                "overall": _finalize_suite(suites["overall"]),
                "x": _finalize_suite(suites["x"]),
                "y": _finalize_suite(suites["y"]),
            }

        total_elapsed = time.perf_counter() - t0
        print(f"[Realism] END | total time = {total_elapsed / 60:.2f} min")

        return realism

def compute_class_weights_subset(counts: dict, classes: list[int], mode: str) -> dict:
    sub = {c: counts[c] for c in classes}

    if mode == "proportional":
        s = sum(sub.values())
        return {c: sub[c] / s for c in sub}

    if mode == "equal":
        k = len(sub)
        return {c: 1.0 / k for c in sub}

    if mode == "inverse":
        inv = {c: 1.0 / max(1, sub[c]) for c in sub}
        s = sum(inv.values())
        return {c: inv[c] / s for c in inv}

    raise ValueError(f"Unknown class_weight_mode: {mode}")


def optimize_temperature_table(
    model,
    x_ref,
    y_ref,
    c_ref,
    *,
    classes_to_optimize: list[int],
    metric_weights: dict,
    loss_like_kwargs: dict,
    device,
    class_weight_mode: str = "inverse",
    tx_bounds=(0.6, 1.4),
    ty_bounds=(0.6, 1.4),
    max_evals: int = 40,
    seed: int | None = None,
    save_yaml_path: str | None = None,
):
    """
    Deterministic post-hoc optimization of per-class (T_x, T_y)
    using Nelder–Mead.
    """

    model.eval()

    EPS_METRIC = 1e-6  # metric-scale stabilization

    def _normalize_metric(path: str, cls_metrics: dict, raw_val: float) -> float:

        """
        Apply deadzones + robust MMD handling.
        Returns the comparable metric value used for optimization.
        """
        val = float(raw_val)

        # ---- deadzones ----
        if path.endswith("ks_mean"):
            val = max(0.0, val - 0.01)
        elif path.endswith("w1_mean"):
            val = max(0.0, val - 0.02)
        elif path.endswith("pearson_fro_rel"):
            val = max(0.0, val - 0.05)
        elif path.endswith("xy_pearson_fro_rel"):
            val = max(0.0, val - 0.05)
        elif path.endswith("xy_spearman_fro_rel"):
            val = max(0.0, val - 0.05)

        # ---- ROBUST MMD handling (scale-invariant) ----
        if path.endswith("mmd2_ratio"):
            scope = path.split(".", 1)[0]  # overall / x / y
            scoped = cls_metrics[scope]

            rvs = max(scoped["mmd2_rvs"], EPS_METRIC)
            rvr = max(scoped["mmd2_rvr_med"], EPS_METRIC)

            # symmetric, scale-invariant distance
            val = abs(math.log(rvs) - math.log(rvr))

        return float(val)

    def _get_by_path(d: dict, path: str):
        obj = d
        for p in path.split("."):
            obj = obj[p]
        return obj

    classes_all = sorted(int(c) for c in torch.unique(c_ref).tolist())
    classes_opt = sorted(int(c) for c in classes_to_optimize)

    # --- counts ---
    counts = {
        int(c): int((c_ref == c).sum().item())
        for c in classes_all
    }

    # --- weights ONLY over optimized classes ---
    class_weights = compute_class_weights_subset(
        counts, classes_opt, class_weight_mode
    )

    # ---- parameter layout ----
    # theta = [Tx_c1, Ty_c1, Tx_c2, Ty_c2, ...]
    idx = []
    for c in classes_opt:
        idx.append((c, "T_x"))
        idx.append((c, "T_y"))

    dim = len(idx)

    # ---- initial point = identity ----
    x0 = np.ones(dim, dtype=np.float64)

    # ---- bounds ----
    bounds = []
    for (_, k) in idx:
        bounds.append(tx_bounds if k == "T_x" else ty_bounds)

    # ---- timing ----
    t_start = time.perf_counter()
    print(f"Optimizing {len(classes_opt)} classes")
    eval_counter = 0

    print("Computing BASELINE (identity temperatures)...")

    identity_temps = {
        int(c): {"T_x": 1.0, "T_y": 1.0}
        for c in torch.unique(c_ref).tolist()
    }

    baseline_realism = compute_realism_metrics_for_set_with_temperature(
        model=model,
        x_ref=x_ref,
        y_ref=y_ref,
        c_ref=c_ref,
        temps_by_class=identity_temps,
        loss_like_kwargs=loss_like_kwargs,
        device=device,
        seed=seed,
    )

    # ---- best trackers for 3 objectives ----
    best = {
        "conservative": {"score": float("inf"), "theta": None, "temps": None},
        "tradeoff": {"score": float("inf"), "theta": None, "temps": None},
        "mixed": {"score": float("inf"), "theta": None, "temps": None},
    }

    # allowed relative degradation for MMD (5–20%)
    MMD_TOLERANCE = 0.10  # 10% (cámbialo a 0.05–0.20 si quieres)

    # ---- objective ----
    def objective(theta: np.ndarray) -> float:
        nonlocal eval_counter
        eval_counter += 1

        temps = {}

        # identity for all
        for c in classes_all:
            temps[c] = {"T_x": 1.0, "T_y": 1.0}

        # optimized ones
        for i, (c, k) in enumerate(idx):
            temps[c][k] = float(theta[i])

        realism = compute_realism_metrics_for_set_with_temperature(
            model=model,
            x_ref=x_ref,
            y_ref=y_ref,
            c_ref=c_ref,
            temps_by_class=temps,
            loss_like_kwargs=loss_like_kwargs,
            device=device,
            seed=seed,
        )

        score_conservative = 0.0
        score_tradeoff = 0.0

        # para debug: guardamos contribuciones por objetivo
        metric_debug = []

        for c in classes_opt:
            w_c = class_weights[c]

            cls_metrics = realism["per_class"][c]
            base_cls_metrics = baseline_realism["per_class"][c]

            for path, w_m in metric_weights.items():
                # raw values
                raw_obj = _get_by_path(cls_metrics, path)
                raw_base_obj = _get_by_path(base_cls_metrics, path)

                if raw_obj is None or raw_base_obj is None:
                    continue

                raw_val = float(raw_obj)
                raw_base_val = float(raw_base_obj)

                # normalized comparable values (SAME pipeline)
                val = _normalize_metric(path, cls_metrics, raw_val)
                base_val = _normalize_metric(path, base_cls_metrics, raw_base_val)

                delta = val - base_val  # >0 peor que baseline, <0 mejora

                # =========================================================
                # 1) CONSERVATIVE: penaliza cualquier empeoramiento vs baseline
                # =========================================================
                cons_term = max(0.0, delta)  # solo empeoramientos
                cons_contrib = w_c * w_m * cons_term
                score_conservative += cons_contrib

                # =========================================================
                # 2) TRADEOFF:
                #    - permite empeorar MMD hasta MMD_TOLERANCE (relativo)
                #    - recompensa mejoras XY (delta negativo)
                #    - el resto lo tratamos conservador (penaliza empeoramientos)
                # =========================================================
                if path.endswith("mmd2_ratio"):
                    # tolerancia RELATIVA: delta/base
                    rel = delta / max(base_val, EPS_METRIC) if base_val is not None else delta
                    # solo penaliza si superas la tolerancia
                    trade_term = max(0.0, rel - MMD_TOLERANCE)
                    trade_contrib = w_c * w_m * trade_term

                elif ("xy_pearson_fro_rel" in path) or ("xy_spearman_fro_rel" in path):
                    # recompensa solo mejoras (delta<0): suma negativo => mejor
                    trade_term = min(0.0, delta)
                    trade_contrib = w_c * w_m * trade_term

                else:
                    # resto: conservador
                    trade_term = max(0.0, delta)
                    trade_contrib = w_c * w_m * trade_term

                score_tradeoff += trade_contrib

                metric_debug.append((
                    c, path,
                    val, base_val, delta,
                    w_m, w_c,
                    cons_contrib, trade_contrib
                ))

        # =========================================================
        # 3) MIXED: media entre conservative y tradeoff
        #    (si tradeoff puede ser negativo por recompensas XY,
        #     mixed recoge esa preferencia pero no se desmadra)
        # =========================================================
        score_mixed = 0.5 * score_conservative + 0.5 * score_tradeoff

        # ---- timing info ----
        elapsed = time.perf_counter() - t_start
        avg_time = elapsed / eval_counter

        # ---- update best trackers (3 objetivos) ----
        def _update_best(tag: str, score_val: float):
            if score_val < best[tag]["score"]:
                best[tag]["score"] = float(score_val)
                best[tag]["theta"] = theta.copy()
                best[tag]["temps"] = {int(k): {"T_x": float(v["T_x"]), "T_y": float(v["T_y"])} for k, v in
                                      temps.items()}

        _update_best("conservative", score_conservative)
        _update_best("tradeoff", score_tradeoff)
        _update_best("mixed", score_mixed)

        print(
            f"\n[TempOpt] eval {eval_counter:03d} | elapsed={elapsed / 60:.2f} min | avg/eval={avg_time:.2f} s | "
            f"S_cons={score_conservative:.6f} | S_trade={score_tradeoff:.6f} | S_mixed={score_mixed:.6f} | "
            f"BEST_cons={best['conservative']['score']:.6f} | BEST_trade={best['tradeoff']['score']:.6f} | BEST_mixed={best['mixed']['score']:.6f}"
        )

        # ---- pretty debug: ordena por |impacto| en mixed (aprox) ----
        # contrib_mixed = 0.5*cons + 0.5*trade
        metric_debug.sort(key=lambda x: abs(0.5 * x[-2] + 0.5 * x[-1]), reverse=True)

        for (c, path, val, base_val, delta, w_m, w_c, cons_contrib, trade_contrib) in metric_debug:
            mixed_contrib = 0.5 * cons_contrib + 0.5 * trade_contrib
            print(
                f"  class {c:>2} | {path:35s} "
                f"val={val:7.4f} base={base_val:7.4f} Δ={delta:+7.4f} "
                f"(w_m={w_m:.2f}, w_c={w_c:.2f}) "
                f"cons={cons_contrib:+.6f} trade={trade_contrib:+.6f} mixed={mixed_contrib:+.6f}"
            )

        # =========================================================
        # >>>> LO QUE MINIMIZA NELDER-MEAD <<<<
        # por defecto: mixed (recomendado)
        # =========================================================
        return float(score_mixed)

    # ---- improved initial simplex ----
    simplex = [x0]

    rng = np.random.default_rng(seed)

    for _ in range(len(x0)):
        step = rng.normal(0.0, 0.08, size=len(x0))
        step = np.clip(step, -0.15, 0.15)
        simplex.append(np.clip(x0 + step,
                               [b[0] for b in bounds],
                               [b[1] for b in bounds]))

    initial_simplex = np.vstack(simplex)

    # ---- optimize ----
    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "maxfev": max_evals,
            "xatol": 1e-3,
            "fatol": 1e-4,
            "initial_simplex": initial_simplex,
            "adaptive": True,  # <-- VERY IMPORTANT
        },
    )

    # ---- build best temperature table ----
    def _build_result(tag: str):
        theta_best = best[tag]["theta"]
        temps_best = best[tag]["temps"]

        return {
            "tag": tag,
            "best_temps": temps_best,
            "best_score": float(best[tag]["score"]),
            "best_theta": None if theta_best is None else [float(x) for x in theta_best],
            "success": bool(res.success),
            "message": str(res.message),
            "n_evals": int(res.nfev),
            "class_weights": class_weights,
            "classes_optimized": classes_opt,
            "metric_weights": metric_weights,
            "mmd_tolerance": float(MMD_TOLERANCE),
            "minimize_target": "mixed",
        }

    result_all = {
        "conservative": _build_result("conservative"),
        "tradeoff": _build_result("tradeoff"),
        "mixed": _build_result("mixed"),
        "optimizer": {
            "method": "Nelder-Mead",
            "success": bool(res.success),
            "message": str(res.message),
            "n_evals": int(res.nfev),
        }
    }

    # ---- optional YAML save ----
    if save_yaml_path is not None:
        with open(save_yaml_path, "w") as f:
            yaml.dump(result_all, f, sort_keys=False)

    return result_all


def train_flowgen_model(
        cXy_train: pd.DataFrame,
        cXy_val: pd.DataFrame,
        condition_col: str,
        cXy_test: pd.DataFrame | None = None,
        r_train: pd.DataFrame | None = None,
        r_val: pd.DataFrame | None = None,
        allow_test_holdout: bool = False,
        seed: int | None = None,
        config_filename: str = "flowgen.yaml",
        base_name: str = "flow_gen_v1",
        device: str = "auto",
        verbose: bool = True,
        finetuning: bool = True,
        pretrained_model: torch.nn.Module | None = None,
        pretrained_path: str | Path | None = None,
        skip_phase1: bool = False,
        evaluation_context: Optional[dict] = None,
        monitoring_policy: str = OFFICIAL_VAL_POLICY,
        output_namespace: str | None = None,
        output_subdir: str | None = None,
        fixed_run_id: str | None = None,
        log_in_run_dir: bool = False,
):
    """
    Train a conditional normalizing flow on concatenated [X, y] given class labels using FlowGen.

    Assumes `flowgen_loss(model, x, y, c, **kwargs)` returns:
        loss, diagnostics, rrmse_x_mean, rrmse_x_std, rrmse_y_mean, rrmse_y_std
    """
    # -------------------- Setup & config --------------------
    seed = _maybe_set_seed(seed)

    cfg_path = Path(config_filename)
    if not cfg_path.parent or cfg_path.parent == Path("."):
        cfg_path = ROOT_PATH / "config" / cfg_path
    if not cfg_path.suffix:
        cfg_path = cfg_path.with_suffix(".yaml")

    config    = load_yaml_config(cfg_path)
    model_cfg = config["model"]
    train_cfg = config["training"]
    interp_cfg = config.get("interpretability", {})
    monitoring_policy = normalize_monitoring_policy(monitoring_policy)
    ensure_holdout_policy(monitoring_policy, allow_test_holdout=allow_test_holdout)
    eval_ctx, monitoring_info = with_monitoring_context(evaluation_context, monitoring_policy)
    monitoring_artifact = monitoring_info if monitoring_policy == TRAIN_ONLY_POLICY else None
    output_namespace = experimental_output_namespace(output_namespace, monitoring_policy)
    monitor_label = "Train-monitor" if monitoring_policy == TRAIN_ONLY_POLICY else "Val"
    improvement_label = "Monitor Improvement" if monitoring_policy == TRAIN_ONLY_POLICY else "Validation Improvement"
    best_loss_label = "Best monitor loss" if monitoring_policy == TRAIN_ONLY_POLICY else "Best validation loss"
    baseline_label = "monitor" if monitoring_policy == TRAIN_ONLY_POLICY else "validation"
    loss_tag = "monitorloss" if monitoring_policy == TRAIN_ONLY_POLICY else "valloss"
    if monitoring_policy == TRAIN_ONLY_POLICY:
        cXy_val = cXy_train.copy()
        r_val = None
    if not allow_test_holdout:
        cXy_test = None

    # Accept str or torch.device; prefer MPS → CUDA → CPU with fallback
    if isinstance(device, torch.device):
        device = device
    elif isinstance(device, str):
        device = select_device(device)
    else:
        device = select_device(None)

    # -------------------- Logging dirs --------------------
    versioned_dir, versioned_name, log_file_path, snapshots_dir = setup_training_logs_and_dirs(
        base_name,
        str(cfg_path),
        config,
        verbose,
        train_cfg.get("save_states", False),
        train_cfg.get("log_training", True),
        subdir="flowgen",
        namespace=output_namespace,
        relative_subdir=output_subdir,
        fixed_run_id=fixed_run_id,
        log_in_run_dir=log_in_run_dir,
    )

    flowpre_log(f"Using device: {device}", log_training=train_cfg.get("log_training", True),
                filename_or_path=log_file_path, verbose=verbose)

    # Tensors + DataLoader
    (
        x_train, y_train,
        x_val,   y_val,
        x_test,  y_test,
        c_train, c_val, c_test,
        feature_names_x, target_names_y,
        train_dataset, train_dataloader
    ) = prepare_flowgen_dataloader(
        df_train=cXy_train,
        df_val=cXy_val,
        condition_col=condition_col,
        batch_size=train_cfg["batch_size"],
        device=device,
        df_test=cXy_test,
        seed=seed,
    )

    # ----- Build or use provided model (also support .pt path) -----
    if pretrained_model is not None:
        model = pretrained_model.to(device)
        flowpre_log("📦 Using provided pretrained model object.", filename_or_path=log_file_path, verbose=verbose)
    else:
        model = build_flowgen_model(
            model_cfg=model_cfg,
            x_dim=x_train.shape[1],
            y_dim=y_train.shape[1],
            num_classes=int(c_train.max().item()) + 1,
            device=device,
        )
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location=device)
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                model.load_state_dict(sd["state_dict"], strict=False)
            elif isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values()):
                # raw state_dict
                model.load_state_dict(sd, strict=False)
            elif hasattr(sd, "state_dict"):
                model.load_state_dict(sd.state_dict(), strict=False)
            else:
                raise ValueError(
                    "Unsupported checkpoint format at pretrained_path. "
                    "Expect a raw state_dict, a {'state_dict': ...} dict, or an nn.Module."
                )
            flowpre_log(f"📥 Loaded weights from: {pretrained_path}",
                        filename_or_path=log_file_path, verbose=verbose)

    # Optional ActNorm warmup (use forward_xy)
    if (pretrained_model is None) and (pretrained_path is None) and model_cfg.get("use_actnorm", False):
        model.eval()
        with torch.no_grad():
            try:
                bx, by, bc = next(iter(train_dataloader))
                _ = model.forward_xy(bx.to(device), by.to(device), bc.to(device))
            except StopIteration:
                raise ValueError("Empty dataloader – check your input data.")
        model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    # Build loss kwargs *entirely* from config (all tunable via training.* keys)
    loss_kwargs = _loss_kwargs_from_train_cfg(train_cfg)

    # Always pass TRAIN references (used by loss if realism toggles are on, and by eval helper)
    loss_kwargs.update({
        "x_ref_all": x_train,
        "y_ref_all": y_train,
        "c_ref_all": c_train,
    })

    def _any_realism_enabled_from_cfg(cfg: dict) -> bool:
        return any([
            bool(cfg.get("use_mmd_xy", False)) and float(cfg.get("mmd_xy_weight", 0.0)) > 0.0,
            bool(cfg.get("use_corr_xy_pearson", False)) and float(cfg.get("corr_xy_pearson_weight", 0.0)) > 0.0,
            bool(cfg.get("use_corr_xy_spearman", False)) and float(cfg.get("corr_xy_spearman_weight", 0.0)) > 0.0,

            bool(cfg.get("use_mmd_x", False)) and float(cfg.get("mmd_x_weight", 0.0)) > 0.0,
            bool(cfg.get("use_corr_pearson_x", False)) and float(cfg.get("corr_pearson_x_weight", 0.0)) > 0.0,
            bool(cfg.get("use_corr_spearman_x", False)) and float(cfg.get("corr_spearman_x_weight", 0.0)) > 0.0,
            bool(cfg.get("use_ks_x", False)) and float(cfg.get("ks_x_weight", 0.0)) > 0.0,
            bool(cfg.get("use_w1_x", False)) and float(cfg.get("w1_x_weight", 0.0)) > 0.0,

            bool(cfg.get("use_mmd_y", False)) and float(cfg.get("mmd_y_weight", 0.0)) > 0.0,
            bool(cfg.get("use_corr_pearson_y", False)) and float(cfg.get("corr_pearson_y_weight", 0.0)) > 0.0,
            bool(cfg.get("use_corr_spearman_y", False)) and float(cfg.get("corr_spearman_y_weight", 0.0)) > 0.0,
            bool(cfg.get("use_ks_y", False)) and float(cfg.get("ks_y_weight", 0.0)) > 0.0,
            bool(cfg.get("use_w1_y", False)) and float(cfg.get("w1_y_weight", 0.0)) > 0.0,
        ])

    # -------------------- Training state --------------------
    best_val_loss       = float("inf")
    best_model_state    = None
    state_loss_epoch    = None

    early_stopping_patience = train_cfg.get("early_stopping_patience", 40)
    lr_decay_patience      = train_cfg.get("lr_decay_patience", 16)
    min_improvement        = train_cfg.get("min_improvement", 0.04)
    min_improvement_floor  = train_cfg.get("min_improvement_floor", 0.0025)
    lr_decay_factor        = train_cfg.get("lr_decay_factor", 0.5)
    lr_patience_factor     = train_cfg.get("lr_patience_factor", 0.8)

    epochs_no_improve = 0
    lr_decay_wait     = 0
    lr                = train_cfg["learning_rate"]
    lr_factor         = 1.0
    patience_factor   = 1.0
    initial_lr_decay_patience = lr_decay_patience
    total_epochs      = train_cfg["num_epochs"]
    log_training      = train_cfg.get("log_training", True)

    flowpre_log(f"Using device: {device}", log_training=log_training,
                filename_or_path=log_file_path, verbose=verbose)
    flowpre_log(f"🎲 Using seed: {seed}", log_training=log_training,
                filename_or_path=log_file_path, verbose=verbose)
    flowpre_log(
        f"Monitoring policy: {monitoring_policy} | {monitoring_info['monitor_result_key']} role: "
        f"{monitoring_info['monitor_role']}",
        log_training=log_training,
        filename_or_path=log_file_path,
        verbose=verbose,
    )

    # Base distribution sanity check: dim = X + Y
    base = StandardNormal([x_train.shape[1] + y_train.shape[1]])
    z0 = torch.zeros(1, x_train.shape[1] + y_train.shape[1]).to(device)
    flowpre_log(f"✅ Base log_prob at zero: {base.log_prob(z0).item():.6f}",
                log_training=log_training, filename_or_path=log_file_path, verbose=verbose)

    did_early_stop = False
    did_reach_max_epochs = False

    state_loss_epoch_ft = None
    finetune_total_epochs = 0
    ft_rebase_epoch = None
    loss_kwargs_ft = None

    # Track the first time we complete the realism ramp (phase ≈ 1) in Phase-1
    p1_ramp_rebased = False
    p1_rebase_epoch = None  # NEW

    # If we skip Phase-1, we still want a coherent "Phase-1 best" = provided weights
    if skip_phase1:
        flowpre_log("⏭️ Skipping Phase-1 (will only finetune).",
                    filename_or_path=log_file_path, verbose=verbose)
        best_model_state = _state_dict_to_cpu(model.state_dict())
        state_loss_epoch = None
        # mark as if Phase-1 concluded so finetune can run
        did_early_stop = True

    if not skip_phase1:
        # -------------------- Train loop (Phase-1) --------------------
        for epoch in range(total_epochs):
            model.train()
            total_loss = 0.0

            ep_rmx, ep_rxstd, ep_rmy, ep_rystd = [], [], [], []
            diagnostics_accum = {}

            for batch_idx, batch in enumerate(train_dataloader, start=1):
                batch_x, batch_y, batch_c = batch
                batch_x, batch_y, batch_c = batch_x.to(device), batch_y.to(device), batch_c.to(device)

                loss, diagnostics, rmx, rxs, rmy, rys = flowgen_loss(
                    model=model,
                    batch_x=batch_x, batch_y=batch_y, batch_c=batch_c,
                    epoch=epoch,  # keep this (0-based epoch is fine)
                    batch_index=batch_idx,  # <<< ADD THIS
                    **loss_kwargs
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * batch_x.size(0)
                ep_rmx.append(float(rmx));   ep_rxstd.append(float(rxs))
                ep_rmy.append(float(rmy));   ep_rystd.append(float(rys))

                for k, v in diagnostics.items():
                    diagnostics_accum.setdefault(k, []).append(float(v))

            avg_train_loss = total_loss / len(train_dataloader.dataset)
            avg_rmx   = (sum(ep_rmx)   / max(1, len(ep_rmx)))
            avg_rxstd = (sum(ep_rxstd) / max(1, len(ep_rxstd)))
            avg_rmy   = (sum(ep_rmy)   / max(1, len(ep_rmy)))
            avg_rystd = (sum(ep_rystd) / max(1, len(ep_rystd)))

            # --- Validation ---
            model.eval()
            with torch.no_grad():
                val_loss, val_diag, vrmx, vrxs, vrmy, vrystd = flowgen_loss(
                    model,
                    batch_x=x_val, batch_y=y_val, batch_c=c_val,
                    epoch=epoch,  # <<< gate stays closed
                    batch_index=0,  # <<< ditto
                    **loss_kwargs
                )

            log_epoch_diagnostics(epoch, diagnostics_accum, log_file_path, verbose)

            flowpre_log(
                f"📉 Epoch {epoch + 1}/{total_epochs} — "
                f"Train Loss: {avg_train_loss:.4f} | {monitor_label} Loss: {float(val_loss):.4f}",
                filename_or_path=log_file_path, verbose=verbose
            )
            flowpre_log(
                f"🔍 Train RRMSE — X: mean={avg_rmx:.4f}, std={avg_rxstd:.4f} | "
                f"Y: mean={avg_rmy:.4f}, std={avg_rystd:.4f}",
                filename_or_path=log_file_path, verbose=verbose
            )
            flowpre_log(
                f"🔍 {monitor_label} RRMSE — X: mean={float(vrmx):.4f}, std={float(vrxs):.4f} | "
                f"Y: mean={float(vrmy):.4f}, std={float(vrystd):.4f}",
                filename_or_path=log_file_path, verbose=verbose
            )

            # === Rebase baseline when realism ramp completes (Phase-1) ===
            # Prefer diagnostics from the loss; fall back to epoch-based estimate.
            realism_any_cfg = _any_realism_enabled_from_cfg(train_cfg)
            phase_from_diag = float(val_diag.get("realism_phase", -1.0))
            if phase_from_diag < 0.0:
                warm = int(train_cfg.get("realism_warmup_epochs", 0))
                ramp = int(train_cfg.get("realism_ramp_epochs", 0))
                if ramp <= 0:
                    phase_from_diag = 1.0
                else:
                    phase_from_diag = max(0.0, min(1.0, (epoch - warm + 1) / max(1, ramp)))

            any_enabled = bool(val_diag.get("realism_any_enabled", realism_any_cfg))

            # If this is the FIRST time the ramp finishes, rebase to the current val_loss
            # and SKIP any scheduler updates for this epoch.
            if (not p1_ramp_rebased) and any_enabled and (phase_from_diag >= 0.999):
                p1_ramp_rebased = True
                p1_rebase_epoch = epoch + 1

                # Reset counters/thresholds to config defaults
                epochs_no_improve = 0
                lr_decay_wait = 0
                min_improvement = train_cfg.get("min_improvement", min_improvement)
                lr_decay_patience = train_cfg.get("lr_decay_patience", lr_decay_patience)
                early_stopping_patience = train_cfg.get("early_stopping_patience", early_stopping_patience)
                patience_factor = 1.0
                initial_lr_decay_patience = lr_decay_patience

                # Rebase baseline to *current* validation loss and snapshot the state
                best_val_loss = float(val_loss)
                state_loss_epoch = epoch + 1
                best_model_state = _state_dict_to_cpu(model.state_dict())

                flowpre_log(
                    f"🔁 Rebased {baseline_label} baseline at realism ramp completion (Phase-1); skipping scheduler this epoch.",
                    filename_or_path=log_file_path, verbose=verbose)

                # Skip improvement/LR/early-stop logic for this epoch
                continue

            improvement = (best_val_loss - float(val_loss)) / (abs(best_val_loss) + 1e-8) if best_val_loss < float("inf") else float("inf")
            flowpre_log(f"📈 {improvement_label}: {improvement:.4f}",
                        filename_or_path=log_file_path, verbose=verbose)

            if float(val_loss) < best_val_loss:
                best_val_loss    = float(val_loss)
                state_loss_epoch = epoch + 1
                best_model_state = _state_dict_to_cpu(model.state_dict())

                if improvement >= min_improvement:
                    epochs_no_improve = 0
                    lr_decay_wait     = 0

                    if train_cfg.get("save_states", False):
                        snapshot_path = snapshots_dir / f"{versioned_name}_epoch{epoch + 1}_{loss_tag}{float(val_loss):.4f}.pt"
                        torch.save(best_model_state, snapshot_path)
                        flowpre_log(f"💾 Saved snapshot: {snapshot_path.name}",
                                    filename_or_path=log_file_path, verbose=verbose)
            else:
                epochs_no_improve += 1
                lr_decay_wait     += 1

                if lr_decay_wait >= lr_decay_patience:
                    last_lr = lr
                    lr_factor *= lr_decay_factor
                    lr = train_cfg["learning_rate"] * lr_factor
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr
                    flowpre_log(f"🔽 LR decay: {last_lr:.6f} → {lr:.6f}",
                                filename_or_path=log_file_path, verbose=verbose)

                    old_impr = min_improvement
                    min_improvement = max(min_improvement_floor, min_improvement * lr_decay_factor)
                    if min_improvement < old_impr:
                        flowpre_log(f"🔽 Reducing minimum improvement: {old_impr:.4f} → {min_improvement:.4f}",
                                    log_training=log_training, filename_or_path=log_file_path, verbose=verbose)

                    old_pat = lr_decay_patience
                    patience_factor   *= lr_patience_factor
                    lr_decay_patience  = max(5, int(math.ceil(initial_lr_decay_patience * patience_factor)))
                    if lr_decay_patience < old_pat:
                        flowpre_log(f"🔽 Reducing LR-decay patience: {old_pat} → {lr_decay_patience}",
                                    log_training=log_training, filename_or_path=log_file_path, verbose=verbose)

                    if early_stopping_patience > 5:
                        epochs_no_improve = 0
                    lr_decay_wait = 0

                    if old_impr * lr_decay_factor < min_improvement:
                        old_esp = early_stopping_patience
                        early_stopping_patience = max(5, int(early_stopping_patience * 0.5))
                        if early_stopping_patience < old_esp:
                            flowpre_log(f"🔽 Reducing early-stopping patience: {old_esp} → {early_stopping_patience}",
                                        log_training=log_training, filename_or_path=log_file_path, verbose=verbose)

                if epochs_no_improve >= early_stopping_patience:
                    flowpre_log(f"🛌 Early stopping at epoch {epoch + 1}",
                                filename_or_path=log_file_path, verbose=verbose)
                    did_early_stop = True
                    break

        did_reach_max_epochs = (not did_early_stop) and ((epoch + 1) >= total_epochs)
        phase1_total_epochs = (epoch + 1)
        phase1_best_epoch = state_loss_epoch

        flowpre_log(f"✅ {best_loss_label}: {best_val_loss:.4f} (epoch {state_loss_epoch})",
                    filename_or_path=log_file_path, verbose=verbose)
    else:
        phase1_total_epochs = 0
        phase1_best_epoch = None

    phase1_best_state = (
        _state_dict_to_cpu(best_model_state)
        if best_model_state is not None
        else _state_dict_to_cpu(model.state_dict())
    )

    # -------------------- Phase 2: realism finetuning (optional) --------------------
    finetune_enabled = _any_realism_enabled_from_cfg(train_cfg)

    if finetuning and finetune_enabled and (did_early_stop or did_reach_max_epochs or skip_phase1):
        flowpre_log("🔁 Starting realism finetuning phase (enforce_realism=True)…",
                    filename_or_path=log_file_path, verbose=verbose)

        # Load the best Phase-1 weights as the starting point
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Reset optimizer & all training state
        lr = float(train_cfg["learning_rate"])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # fresh counters & thresholds (new objective now includes realism)
        best_val_loss = float("inf")
        state_loss_epoch_ft = None
        epochs_no_improve = 0
        lr_decay_wait = 0
        lr_factor = 1.0
        patience_factor = 1.0
        lr_decay_patience = train_cfg.get("lr_decay_patience", 16)
        early_stopping_patience = train_cfg.get("early_stopping_patience", 40)
        min_improvement = train_cfg.get("min_improvement", 0.04)
        min_improvement_floor = train_cfg.get("min_improvement_floor", 0.0025)
        lr_decay_factor = train_cfg.get("lr_decay_factor", 0.5)
        lr_patience_factor = train_cfg.get("lr_patience_factor", 0.8)
        initial_lr_decay_patience = lr_decay_patience

        flowpre_log("🔁 [FT] Baseline reset for finetune (new objective includes realism).",
                    filename_or_path=log_file_path, verbose=verbose)

        # Force realism regardless of warmup/epoch/stride and keep “keep_mean” scaling logic
        loss_kwargs_ft = {
            **loss_kwargs,
            "enforce_realism": True,
            # set warmup to total_epochs: phase would be 0, but enforce_realism bypasses the gate anyway
            "realism_warmup_epochs": total_epochs,
        }

        total_epochs_finetune = int(train_cfg.get("finetune_num_epochs", total_epochs))

        # Track the first time we complete the realism ramp (phase ≈ 1) in Finetune
        ft_ramp_rebased = False
        ft_rebase_epoch = None  # NEW

        for ft_epoch in range(total_epochs_finetune):
            model.train()
            total_loss = 0.0

            ep_rmx, ep_rxstd, ep_rmy, ep_rystd = [], [], [], []
            diagnostics_accum = {}

            for batch_idx, batch in enumerate(train_dataloader, start=1):
                batch_x, batch_y, batch_c = batch
                batch_x, batch_y, batch_c = batch_x.to(device), batch_y.to(device), batch_c.to(device)

                loss, diagnostics, rmx, rxs, rmy, rys = flowgen_loss(
                    model=model,
                    batch_x=batch_x, batch_y=batch_y, batch_c=batch_c,
                    epoch=ft_epoch, batch_index=batch_idx,
                    **loss_kwargs_ft
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += float(loss.item()) * batch_x.size(0)
                ep_rmx.append(float(rmx));
                ep_rxstd.append(float(rxs))
                ep_rmy.append(float(rmy));
                ep_rystd.append(float(rys))
                for k, v in diagnostics.items():
                    diagnostics_accum.setdefault(k, []).append(float(v))

            avg_train_loss = total_loss / len(train_dataloader.dataset)
            avg_rmx = (sum(ep_rmx) / max(1, len(ep_rmx)))
            avg_rxstd = (sum(ep_rxstd) / max(1, len(ep_rxstd)))
            avg_rmy = (sum(ep_rmy) / max(1, len(ep_rmy)))
            avg_rystd = (sum(ep_rystd) / max(1, len(ep_rystd)))

            # --- Validation ---
            model.eval()
            with torch.no_grad():
                val_loss, val_diag, vrmx, vrxs, vrmy, vrystd = flowgen_loss(
                    model,
                    batch_x=x_val, batch_y=y_val, batch_c=c_val,
                    epoch=ft_epoch, batch_index=0,
                    **loss_kwargs_ft
                )

            log_epoch_diagnostics(ft_epoch, diagnostics_accum, log_file_path, verbose)

            flowpre_log(
                f"📉 [FT] Epoch {ft_epoch + 1}/{total_epochs_finetune} — "
                f"Train Loss: {avg_train_loss:.4f} | {monitor_label} Loss: {float(val_loss):.4f}",
                filename_or_path=log_file_path, verbose=verbose
            )
            flowpre_log(
                f"🔍 [FT] Train RRMSE — X: mean={avg_rmx:.4f}, std={avg_rxstd:.4f} | "
                f"Y: mean={avg_rmy:.4f}, std={avg_rystd:.4f}",
                filename_or_path=log_file_path, verbose=verbose
            )
            flowpre_log(
                f"🔍 [FT] {monitor_label} RRMSE — X: mean={float(vrmx):.4f}, std={float(vrxs):.4f} | "
                f"Y: mean={float(vrmy):.4f}, std={float(vrystd):.4f}",
                filename_or_path=log_file_path, verbose=verbose
            )

            # === Baseline rebase when realism ramp completes (Finetune) ===
            realism_any_cfg = _any_realism_enabled_from_cfg(train_cfg)
            phase_from_diag = float(val_diag.get("realism_phase", -1.0))

            # If we forced realism, treat the phase as complete from the start.
            if phase_from_diag < 0.0:
                warm = int(train_cfg.get("realism_warmup_epochs", 0))
                ramp = int(train_cfg.get("realism_ramp_epochs", 0))
                if loss_kwargs_ft.get("enforce_realism", False):
                    phase_from_diag = 1.0
                else:
                    if ramp <= 0:
                        phase_from_diag = 1.0
                    else:
                        phase_from_diag = max(0.0, min(1.0, (ft_epoch - warm + 1) / max(1, ramp)))

            any_enabled = bool(val_diag.get("realism_any_enabled", realism_any_cfg)) or bool(
                loss_kwargs_ft.get("enforce_realism", False))

            if (not ft_ramp_rebased) and any_enabled and (phase_from_diag >= 0.999):
                ft_ramp_rebased = True
                ft_rebase_epoch = ft_epoch + 1

                epochs_no_improve = 0
                lr_decay_wait = 0
                min_improvement = train_cfg.get("min_improvement", min_improvement)
                lr_decay_patience = train_cfg.get("lr_decay_patience", lr_decay_patience)
                early_stopping_patience = train_cfg.get("early_stopping_patience", early_stopping_patience)
                patience_factor = 1.0
                initial_lr_decay_patience = lr_decay_patience

                # Rebase to current loss and snapshot; then skip this epoch's scheduler logic
                best_val_loss = float(val_loss)
                state_loss_epoch_ft = ft_epoch + 1
                best_model_state = _state_dict_to_cpu(model.state_dict())

                flowpre_log(
                    f"🔁 [FT] Rebased {baseline_label} baseline at realism ramp completion; skipping scheduler this epoch.",
                    filename_or_path=log_file_path, verbose=verbose)
                continue

            improvement = (best_val_loss - float(val_loss)) / (abs(best_val_loss) + 1e-8) if best_val_loss < float(
                "inf") else float("inf")
            flowpre_log(f"📈 [FT] {improvement_label}: {improvement:.4f}",
                        filename_or_path=log_file_path, verbose=verbose)

            if float(val_loss) < best_val_loss:
                best_val_loss = float(val_loss)
                state_loss_epoch_ft = ft_epoch + 1
                best_model_state = _state_dict_to_cpu(model.state_dict())

                if improvement >= min_improvement:
                    epochs_no_improve = 0
                    lr_decay_wait = 0

                    if train_cfg.get("save_states", False):
                        snapshot_path = snapshots_dir / f"{versioned_name}_FT_epoch{ft_epoch + 1}_{loss_tag}{float(val_loss):.4f}.pt"
                        torch.save(best_model_state, snapshot_path)
                        flowpre_log(f"💾 Saved [FT] snapshot: {snapshot_path.name}",
                                    filename_or_path=log_file_path, verbose=verbose)
            else:
                epochs_no_improve += 1
                lr_decay_wait += 1

                if lr_decay_wait >= lr_decay_patience:
                    last_lr = lr
                    lr_factor *= lr_decay_factor
                    lr = train_cfg["learning_rate"] * lr_factor
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr
                    flowpre_log(f"🔽 [FT] LR decay: {last_lr:.6f} → {lr:.6f}",
                                filename_or_path=log_file_path, verbose=verbose)

                    old_impr = min_improvement
                    min_improvement = max(min_improvement_floor, min_improvement * lr_decay_factor)
                    if min_improvement < old_impr:
                        flowpre_log(f"🔽 [FT] Reducing minimum improvement: {old_impr:.4f} → {min_improvement:.4f}",
                                    filename_or_path=log_file_path, verbose=verbose)

                    old_pat = lr_decay_patience
                    patience_factor *= lr_patience_factor
                    lr_decay_patience = max(5, int(math.ceil(initial_lr_decay_patience * patience_factor)))
                    if lr_decay_patience < old_pat:
                        flowpre_log(f"🔽 [FT] Reducing LR-decay patience: {old_pat} → {lr_decay_patience}",
                                    filename_or_path=log_file_path, verbose=verbose)

                    if early_stopping_patience > 5:
                        epochs_no_improve = 0
                    lr_decay_wait = 0

                    if old_impr * lr_decay_factor < min_improvement:
                        old_esp = early_stopping_patience
                        early_stopping_patience = max(5, int(early_stopping_patience * 0.5))
                        if early_stopping_patience < old_esp:
                            flowpre_log(
                                f"🔽 [FT] Reducing early-stopping patience: {old_esp} → {early_stopping_patience}",
                                filename_or_path=log_file_path, verbose=verbose)

                if epochs_no_improve >= early_stopping_patience:
                    flowpre_log(f"🛌 [FT] Early stopping at epoch {ft_epoch + 1}",
                                filename_or_path=log_file_path, verbose=verbose)
                    break

        finetune_total_epochs = (ft_epoch + 1)
        flowpre_log(f"✅ [FT] {best_loss_label}: {best_val_loss:.4f} (epoch {state_loss_epoch_ft})",
                    filename_or_path=log_file_path, verbose=verbose)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final snapshot (optional)
    if train_cfg.get("save_states", False):
        best_epoch_tag = state_loss_epoch_ft if state_loss_epoch_ft is not None else state_loss_epoch
        final_path = snapshots_dir / f"{versioned_name}_best_epoch{best_epoch_tag}_{loss_tag}{best_val_loss:.4f}.pt"
        torch.save(best_model_state, final_path)

    flowpre_log("🧱 Postprocess start: split_metrics_train", filename_or_path=log_file_path, verbose=verbose)
    split_metrics_train = compute_flowgen_split_metrics(
        model=model,
        x=x_train,
        y=y_train,
        c=c_train,
        loss_kwargs=loss_kwargs,
        include_realism=True,
        device=device,
        seed=seed,
    )
    flowpre_log("✅ Postprocess end: split_metrics_train", filename_or_path=log_file_path, verbose=verbose)
    flowpre_log("🧱 Postprocess start: split_metrics_val", filename_or_path=log_file_path, verbose=verbose)
    split_metrics_val = compute_flowgen_split_metrics(
        model=model,
        x=x_val,
        y=y_val,
        c=c_val,
        loss_kwargs=loss_kwargs,
        include_realism=True,
        device=device,
        seed=seed,
    )
    flowpre_log("✅ Postprocess end: split_metrics_val", filename_or_path=log_file_path, verbose=verbose)
    split_metrics_test = None
    if allow_test_holdout and (x_test is not None and y_test is not None and c_test is not None):
        flowpre_log("🧱 Postprocess start: split_metrics_test", filename_or_path=log_file_path, verbose=verbose)
        split_metrics_test = compute_flowgen_split_metrics(
            model=model,
            x=x_test,
            y=y_test,
            c=c_test,
            loss_kwargs=loss_kwargs,
            include_realism=True,
            device=device,
            seed=seed,
        )
        flowpre_log("✅ Postprocess end: split_metrics_test", filename_or_path=log_file_path, verbose=verbose)

    temporal_realism_block = None
    temporal_realism_artifacts = None
    temporal_cfg = resolve_temporal_realism_config(train_cfg=train_cfg, loss_like_kwargs=loss_kwargs)
    if monitoring_policy == TRAIN_ONLY_POLICY and temporal_cfg.get("enabled", False):
        temporal_cfg = {**temporal_cfg, "enabled": False}
        flowpre_log(
            "Temporal realism disabled: train-only monitoring does not provide a temporal validation holdout.",
            filename_or_path=log_file_path,
            verbose=verbose,
        )
    if train_cfg.get("save_results", False) and temporal_cfg.get("enabled", False):
        if r_val is None:
            flowpre_log(
                "⚠️ Temporal realism skipped because r_val was not provided to the closeout phase.",
                filename_or_path=log_file_path,
                verbose=verbose,
            )
        else:
            flowpre_log("🧱 Postprocess start: temporal_realism", filename_or_path=log_file_path, verbose=verbose)
            try:
                temporal_payload = build_temporal_realism_block(
                    model=model,
                    x_train=x_train,
                    y_train=y_train,
                    c_train=c_train,
                    x_val=x_val,
                    y_val=y_val,
                    c_val=c_val,
                    cxy_val=cXy_val,
                    r_val=r_val,
                    loss_like_kwargs=loss_kwargs,
                    temporal_cfg=temporal_cfg,
                    device=device,
                    seed=seed,
                    run_id=versioned_name,
                    split_id=str(eval_ctx.get("split_id", "init_temporal_processed_v1")),
                    condition_col=condition_col,
                )
                if temporal_cfg.get("write_sidecars", True):
                    temporal_realism_artifacts = write_temporal_realism_sidecars(
                        out_dir=versioned_dir,
                        payload=temporal_payload,
                    )
                temporal_realism_block = dict(temporal_payload["block"])
                temporal_realism_block["artifacts"] = dict(temporal_realism_artifacts or {})
                flowpre_log("✅ Postprocess end: temporal_realism", filename_or_path=log_file_path, verbose=verbose)
            except Exception as exc:
                flowpre_log(
                    f"⚠️ Temporal realism skipped after an additive-phase failure: {exc}",
                    filename_or_path=log_file_path,
                    verbose=verbose,
                )

    flowpre_log(
        f"📈 Train RRMSE — X: {split_metrics_train['rrmse_x_recon']:.4f}, Y: {split_metrics_train['rrmse_y_recon']:.4f} | "
        f"R² — X: {split_metrics_train['r2_x_recon']:.4f}, Y: {split_metrics_train['r2_y_recon']:.4f}",
        filename_or_path=log_file_path, verbose=verbose
    )
    flowpre_log(
        f"📈 {monitor_label} RRMSE — X: {split_metrics_val['rrmse_x_recon']:.4f}, Y: {split_metrics_val['rrmse_y_recon']:.4f} | "
        f"R² — X: {split_metrics_val['r2_x_recon']:.4f}, Y: {split_metrics_val['r2_y_recon']:.4f}",
        filename_or_path=log_file_path, verbose=verbose
    )

    # ---- Latent influence over X and y (FlowGen-specific helper)
    flowpre_log("🧱 Postprocess start: influence", filename_or_path=log_file_path, verbose=verbose)
    influence_dict = compute_latent_feature_influence_flowgen(
        model,
        x_train, y_train, c_train,
        feature_names_x=feature_names_x,
        target_names=target_names_y,
        influence_step_fraction=float(interp_cfg.get("influence_step_fraction", 0.01)),
        sweep_range=tuple(interp_cfg.get("sweep_range", [-model_cfg.get("tail_bound", 3.0),
                                                         model_cfg.get("tail_bound", 3.0)])),
    )
    flowpre_log("✅ Postprocess end: influence", filename_or_path=log_file_path, verbose=verbose)

    flowpre_log(f"📊 Per-class isotropy (Train): {_format_iso_dict(split_metrics_train['per_class_iso_rrmse'])}",
                filename_or_path=log_file_path, verbose=verbose)
    flowpre_log(f"📊 Per-class isotropy ({monitor_label}):   {_format_iso_dict(split_metrics_val['per_class_iso_rrmse'])}",
                filename_or_path=log_file_path, verbose=verbose)

    metrics_long_path = None
    results_path = None
    results = None

    # --- Save results with overall isotropy (X/Y separated) ---
    if train_cfg.get("save_results", False):
        flowpre_log("🧱 Postprocess start: save_results", filename_or_path=log_file_path, verbose=verbose)
        results = {
            "seed": seed,
            "phase1": {
                "best_epoch": phase1_best_epoch,
                "total_epochs": phase1_total_epochs,
                "ramp_rebase_epoch": p1_rebase_epoch,  # NEW
            },

            "finetune": {
                "enabled": bool(finetuning and finetune_enabled and (did_early_stop or did_reach_max_epochs or skip_phase1)),
                "best_epoch": state_loss_epoch_ft,
                "total_epochs": finetune_total_epochs if state_loss_epoch_ft is not None else 0,
                "ramp_rebase_epoch": ft_rebase_epoch,  # NEW
            },
            "train": split_metrics_train,
            "val": split_metrics_val,
        }
        if monitoring_artifact is not None:
            results["monitoring"] = monitoring_artifact

        if temporal_realism_block is not None:
            results["val"]["temporal_realism"] = temporal_realism_block

        if split_metrics_test is not None:
            results["test"] = split_metrics_test

        # === Add best-epoch total loss AND NLL loss (train & val) ===
        # Decide which section holds the "best" state (already loaded into `model`)
        best_section = "finetune" if (state_loss_epoch_ft is not None) else "phase1"

        model.eval()
        with torch.no_grad():
            best_train_total, train_diag, *_ = flowgen_loss(
                model, x_train, y_train, c_train,
                epoch=0, batch_index=0, **loss_kwargs
            )
            best_val_total,   val_diag,   *_ = flowgen_loss(
                model, x_val,   y_val,   c_val,
                epoch=0, batch_index=0, **loss_kwargs
            )

        # Total losses (scalars)
        results[best_section]["best_train_loss_total"] = round(float(best_train_total), 6)
        results[best_section]["best_val_loss_total"]   = round(float(best_val_total), 6)
        results[best_section]["best_monitor_loss_total"] = round(float(best_val_total), 6)

        # NLL losses from diagnostics (if present)
        train_nll = train_diag.get("loss_nll", None)
        val_nll   = val_diag.get("loss_nll", None)
        results[best_section]["best_train_loss_nll"] = (round(float(train_nll), 6) if train_nll is not None else None)
        results[best_section]["best_val_loss_nll"]   = (round(float(val_nll), 6)   if val_nll   is not None else None)
        results[best_section]["best_monitor_loss_nll"] = (
            round(float(val_nll), 6) if val_nll is not None else None
        )


        results_path = versioned_dir / f"{versioned_name}_results.yaml"
        with open(results_path, "w") as f:
            yaml.dump(results, f, sort_keys=False)
        flowpre_log(f"📝 Saved enriched results to: {results_path}",
                    filename_or_path=log_file_path, verbose=verbose)

        run_context = build_run_context(
            model_family="flowgen",
            run_id=versioned_name,
            seed=seed,
            config=config,
            config_path=cfg_path,
            dataset_name=str(eval_ctx.get("dataset_name", DEFAULT_OFFICIAL_DATASET_NAME)),
            dataset_manifest_path=eval_ctx.get("dataset_manifest_path", official_raw_bundle_manifest_path()),
            split_id=str(eval_ctx.get("split_id", "init_temporal_processed_v1")),
            split_manifest_path=eval_ctx.get("split_manifest_path"),
            upstream_variant_fingerprint=eval_ctx.get("upstream_variant_fingerprint"),
            contract_id=eval_ctx.get("contract_id"),
            comparison_group_id=eval_ctx.get("comparison_group_id"),
            seed_set_id=eval_ctx.get("seed_set_id"),
            base_config_id=eval_ctx.get("base_config_id"),
            objective_metric_id=eval_ctx.get("objective_metric_id"),
            dataset_level_axes=eval_ctx.get("dataset_level_axes"),
            run_level_axes=eval_ctx.get("run_level_axes"),
            split_role_map=eval_ctx.get("split_role_map"),
            monitoring=monitoring_artifact,
            test_enabled=bool(split_metrics_test is not None),
        )
        _, metrics_long_path = save_canonical_run_artifacts(
            results=results,
            context=run_context,
            out_dir=versioned_dir,
            stem=versioned_name,
        )
        flowpre_log(
            f"🧾 Saved canonical metrics table to: {metrics_long_path}",
            filename_or_path=log_file_path,
            verbose=verbose,
        )
        flowpre_log("✅ Postprocess end: save_results", filename_or_path=log_file_path, verbose=verbose)

    # Always save the config
    shutil.copy(cfg_path, versioned_dir / f"{versioned_name}.yaml")

    # =========================
    # Saving models
    # =========================
    final_path = None
    phase1_path = None
    finetuned_path = None

    if train_cfg.get("save_model", False):
        flowpre_log("🧱 Postprocess start: save_model", filename_or_path=log_file_path, verbose=verbose)
        # Always save Phase-1 (pre-finetune) best
        phase1_path = versioned_dir / f"{versioned_name}_phase1.pt"
        torch.save(phase1_best_state, phase1_path)
        flowpre_log(f"💾 Saved Phase-1 (pre-finetune) model: {phase1_path.name}",
                    filename_or_path=log_file_path, verbose=verbose)

        if state_loss_epoch_ft is not None:
            # We ran FT and have a best FT state in best_model_state
            finetuned_path = versioned_dir / f"{versioned_name}_finetuned.pt"
            torch.save(best_model_state, finetuned_path)
            final_path = versioned_dir / f"{versioned_name}.pt"
            torch.save(best_model_state, final_path)  # canonical points to finetuned
            flowpre_log(f"💾 Saved finetuned model: {finetuned_path.name} (also as {final_path.name})",
                        filename_or_path=log_file_path, verbose=verbose)
        else:
            # No FT → canonical is Phase-1
            final_path = versioned_dir / f"{versioned_name}.pt"
            torch.save(phase1_best_state, final_path)
            flowpre_log(f"💾 Saved model: {final_path.name} (Phase-1)",
                        filename_or_path=log_file_path, verbose=verbose)
        flowpre_log("✅ Postprocess end: save_model", filename_or_path=log_file_path, verbose=verbose)

    # Log influence only if saved
    if interp_cfg.get("save_influence", False):
        with open(versioned_dir / f"{versioned_name}_influence.json", "w") as f:
            json.dump(influence_dict, f, indent=2)
        flowpre_log(f"🧠 Influence saved: {versioned_name}_influence.json",
                    filename_or_path=log_file_path, verbose=verbose)

    flowpre_log(f"✅ Config saved under {versioned_dir}",
                filename_or_path=log_file_path, verbose=verbose)

    model.run_artifacts = {
        "run_id": versioned_name,
        "run_dir": str(versioned_dir),
        "config_path": str(cfg_path),
        "saved_config_path": str(versioned_dir / f"{versioned_name}.yaml"),
        "results_path": None if results_path is None else str(results_path),
        "metrics_long_path": None if metrics_long_path is None else str(metrics_long_path),
        "model_path": str(versioned_dir / f"{versioned_name}.pt") if train_cfg.get("save_model", False) else None,
        "phase1_model_path": None if phase1_path is None else str(phase1_path),
        "finetuned_model_path": None if finetuned_path is None else str(finetuned_path),
        "log_file_path": None if log_file_path is None else str(log_file_path),
        "output_namespace": output_namespace,
        "output_subdir": output_subdir,
        "monitoring_policy": monitoring_policy,
    }

    del optimizer, train_dataset, train_dataloader, base, loss_kwargs
    del best_model_state, phase1_best_state
    del split_metrics_train, split_metrics_val, split_metrics_test
    del influence_dict
    del x_train, y_train, x_val, y_val, x_test, y_test, c_train, c_val, c_test
    del feature_names_x, target_names_y
    if loss_kwargs_ft is not None:
        del loss_kwargs_ft
    if results is not None:
        del results
    _release_training_memory()

    return model


def train_flowgen_pipeline(
    condition_col: str = "type",
    config_filename: str = "flowgen.yaml",
    base_name: str = "flow_gen",
    device: str = "auto",
    seed: int | None = None,
    verbose: bool = True,
    allow_test_holdout: bool = False,
    *,
    # new knobs (forwarded to train_flowgen_model)
    finetuning: bool = True,
    skip_phase1: bool = False,
    pretrained_model: torch.nn.Module | None = None,
    pretrained_path: str | None = None,
    evaluation_context: Optional[dict] = None,
    monitoring_policy: str = OFFICIAL_VAL_POLICY,
    output_namespace: str | None = None,
    output_subdir: str | None = None,
    fixed_run_id: str | None = None,
    log_in_run_dir: bool = False,
):
    """
    Load (X, y, removed) splits, merge X and y on 'post_cleaning_index' to build cXy_*:
    column order → [post_cleaning_index, condition_col, X..., y...],
    then train FlowGen via `train_flowgen_model`.

    If `skip_phase1=True`, only finetuning (Phase-2) will run.
    Optionally provide `pretrained_model` or `pretrained_path` (path to .pt) to start from saved weights.
    """
    monitoring_policy = normalize_monitoring_policy(monitoring_policy)
    ensure_holdout_policy(monitoring_policy, allow_test_holdout=allow_test_holdout)
    X_train = X_val = X_test = None
    y_train = y_val = y_test = None
    r_train = r_val = r_test = None
    cXy_train = cXy_val = cXy_test = None
    model = None

    try:
        # 1) Load splits
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         r_train, r_val, r_test) = load_or_create_raw_splits(
            condition_col=condition_col, verbose=verbose
        )

        # 2) Merge helper
        def _build_cXy(X_df: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
            X_df = X_df.sort_values("post_cleaning_index").reset_index(drop=True).copy()
            y_df = y_df.sort_values("post_cleaning_index").reset_index(drop=True).copy()

            # y DOES NOT include the condition; merge on index
            df = pd.merge(
                X_df, y_df,
                on="post_cleaning_index",
                how="inner",
                validate="one_to_one",
            )

            x_cols = [c for c in X_df.columns if c not in ("post_cleaning_index", condition_col)]
            y_cols = [c for c in y_df.columns if c != "post_cleaning_index"]

            ordered = ["post_cleaning_index", condition_col] + x_cols + y_cols
            return df[ordered].copy()

        cXy_train = _build_cXy(X_train, y_train)
        cXy_val = (
            cXy_train.copy()
            if monitoring_policy == TRAIN_ONLY_POLICY
            else _build_cXy(X_val, y_val)
        )
        cXy_test = _build_cXy(X_test, y_test) if allow_test_holdout else None
        r_val_for_training = None if monitoring_policy == TRAIN_ONLY_POLICY else r_val

        # small safety: warn if skipping phase-1 without weights
        if skip_phase1 and (pretrained_model is None) and (pretrained_path is None):
            print("⚠️  skip_phase1=True but no pretrained_model/pretrained_path provided; "
                  "will finetune from freshly initialized weights.")

        # 3) Train
        model = train_flowgen_model(
            cXy_train=cXy_train,
            cXy_val=cXy_val,
            cXy_test=cXy_test,
            r_train=r_train,
            r_val=r_val_for_training,
            allow_test_holdout=allow_test_holdout,
            condition_col=condition_col,
            config_filename=config_filename,
            base_name=base_name,
            device=device,
            seed=seed,
            verbose=verbose,
            finetuning=finetuning,
            skip_phase1=skip_phase1,
            pretrained_model=pretrained_model,
            pretrained_path=pretrained_path,
            evaluation_context=evaluation_context,
            monitoring_policy=monitoring_policy,
            output_namespace=output_namespace,
            output_subdir=output_subdir,
            fixed_run_id=fixed_run_id,
            log_in_run_dir=log_in_run_dir,
        )
        return model
    finally:
        del X_train, X_val, X_test
        del y_train, y_val, y_test
        del r_train, r_val, r_test
        del cXy_train, cXy_val, cXy_test
        _release_training_memory()


