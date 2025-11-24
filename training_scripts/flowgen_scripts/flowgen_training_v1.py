#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
flowgen_training_v1.py

Finetune-only sweep for FlowGen focused on KS / W1 / MMD realism metrics (marginal & joint).

What it does
------------
• Clones ROOT_PATH/config/flowgen.yaml per run, applies only the relevant overrides for that run, then launches finetune-only training.
• If --pretrained_path is omitted or looks like a placeholder, it falls back to:
    ROOT_PATH/outputs/models/flowgen/flowgen_T2_v1/snapshots/flowgen_T2_v1_epoch233_valloss-31.5234.pt
• Run names are descriptive (weights/norms/grids/clips/trunc) + seed + prefix.
• Robust retries with exponential backoff; state is persisted in outputs/sweeps/flowgen/flowgen_training_v1_state.json.
• Already-finished runs are auto-skipped (detected by results YAML and by state).

Example
-------
python flowgen_training_v1.py ^
  --device cuda ^
  --seed 1234 ^
  --run_prefix FT25 ^
  --max_retries 6 ^
  --finetune_epochs 10
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

# ---- project utils ----
from training.utils import ROOT_PATH, load_yaml_config
# Your wrapper that loads data and calls training internally
from training.train_flowgen import train_flowgen_pipeline  # this worked in your later run logs

# -------------------------
# Paths
# -------------------------
ROOT_PATH = Path(ROOT_PATH)
CONFIG_DIR = ROOT_PATH / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_ROOT = ROOT_PATH / "outputs" / "models" / "flowgen"
OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

SWEEP_ROOT = ROOT_PATH / "outputs" / "sweeps" / "flowgen"
SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

STATE_PATH = SWEEP_ROOT / "flowgen_training_v1_state.json"

DEFAULT_PRETRAINED = (
    ROOT_PATH
    / "outputs"
    / "models"
    / "flowgen"
    / "flowgen_T2_v1"
    / "snapshots"
    / "flowgen_T2_v1_epoch233_valloss-31.5234.pt"
)


# -------------------------
# Helpers
# -------------------------
def deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    # IMPORTANT: copy full base config; only override the keys for the combo
    cfg2 = copy.deepcopy(cfg)
    for k, v in overrides.items():
        deep_set(cfg2, k, v)
    return cfg2


def _sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def write_temp_config(cfg_dict: dict, stem: str) -> Tuple[str, Path]:
    """Write a temporary YAML under ROOT_PATH/config and return (filenameOnly, fullPath)."""
    safe_stem = _sanitize_name(stem)
    fname = f"{safe_stem}.yaml"
    full_path = CONFIG_DIR / fname
    with open(full_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    if not full_path.exists():
        raise FileNotFoundError(f"Temp config not created: {full_path}")
    return fname, full_path


def fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}h {m:02d}m {s:02d}s"


def find_latest_run_dir(base_name: str) -> Path | None:
    """Find the latest run directory that starts with base_name."""
    if not OUTPUTS_ROOT.exists():
        return None
    cands = [p for p in OUTPUTS_ROOT.iterdir() if p.is_dir() and p.name.startswith(base_name)]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def has_results_yaml(run_dir: Path | None) -> bool:
    if run_dir is None or not run_dir.exists():
        return False
    ymls = [p for p in run_dir.iterdir() if p.is_file() and p.name.endswith("_results.yaml")]
    return len(ymls) > 0


def run_already_finished(base_name: str) -> bool:
    """Robust finished detection: outputs results file OR state file says finished."""
    run_dir = find_latest_run_dir(base_name)
    if has_results_yaml(run_dir):
        return True
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
            return state.get("statuses", {}).get(base_name, {}).get("status") == "ok"
        except Exception:
            return False
    return False


def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"statuses": {}, "attempts": {}}


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


# -------------------------
# Sweep spec (25 runs)
# -------------------------
def common_finetune_overrides(finetune_epochs: int) -> Dict[str, Any]:
    """
    Minimal, focused overrides that every finetune run needs.
    We do NOT touch unrelated keys; we clone the full base config first.
    """
    return {
        # Finetune scheduling
        "training.finetune_num_epochs": int(finetune_epochs),
        "training.enforce_realism": True,
        "training.realism_warmup_epochs": 0,
        "training.realism_ramp_epochs": 0,
        "training.realism_stride_batches": 1,
        "training.realism_stride_epochs": 1,
        "training.realism_scale_mode": "keep_mean",

        # Keep NLL; turn OFF all latent isotropy penalties (we care about realism / sampling)
        "training.use_nll": True,
        "training.nll_weight": 1.0,
        "training.use_latent_mean_penalty": False,
        "training.use_latent_std_penalty": False,
        "training.use_latent_skew_penalty": False,
        "training.use_latent_kurtosis_penalty": False,

        # Results/models — do not change other base config fields
        "training.save_results": True,
        "training.save_states": True,
        "training.save_model": True,
    }


def ks_w1_mmd_recipe(
    *,
    w1x: float | None, w1x_norm: str | None, w1x_soft: float | None, w1x_clip: float | None,
    w1y: float | None, w1y_norm: str | None, w1y_soft: float | None, w1y_clip: float | None,
    ksx_w: float | None, ksx_grid: int | None, ksx_tau: float | None,
    ksy_w: float | None, ksy_grid: int | None, ksy_tau: float | None,
    mmdx: float | None, mmdy: float | None,
    z_trunc: float | None,
    name_tag_hint: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Build overrides + concise tag. Only set keys that belong to the combo.
    """
    ov: Dict[str, Any] = {}

    # W1 X
    if w1x is not None:
        ov["training.use_w1_x"] = True
        ov["training.w1_x_weight"] = float(w1x)
        if w1x_norm is not None: ov["training.w1_x_norm"] = str(w1x_norm)
        if w1x_soft is not None: ov["training.w1_x_softclip_s"] = float(w1x_soft)
        if w1x_clip is not None: ov["training.w1_x_clip_perdim"] = float(w1x_clip)
    else:
        ov["training.use_w1_x"] = False

    # W1 Y
    if w1y is not None:
        ov["training.use_w1_y"] = True
        ov["training.w1_y_weight"] = float(w1y)
        if w1y_norm is not None: ov["training.w1_y_norm"] = str(w1y_norm)
        if w1y_soft is not None: ov["training.w1_y_softclip_s"] = float(w1y_soft)
        if w1y_clip is not None: ov["training.w1_y_clip_perdim"] = float(w1y_clip)
    else:
        ov["training.use_w1_y"] = False

    # KS X
    if ksx_w is not None:
        ov["training.use_ks_x"] = True
        ov["training.ks_x_weight"] = float(ksx_w)
        if ksx_grid is not None: ov["training.ks_grid_points_x"] = int(ksx_grid)
        if ksx_tau is not None: ov["training.ks_tau_x"] = float(ksx_tau)
    else:
        ov["training.use_ks_x"] = False

    # KS Y
    if ksy_w is not None:
        ov["training.use_ks_y"] = True
        ov["training.ks_y_weight"] = float(ksy_w)
        if ksy_grid is not None: ov["training.ks_grid_points_y"] = int(ksy_grid)
        if ksy_tau is not None: ov["training.ks_tau_y"] = float(ksy_tau)
    else:
        ov["training.use_ks_y"] = False

    # MMD
    if mmdx is not None:
        ov["training.use_mmd_x"] = True
        ov["training.mmd_x_weight"] = float(mmdx)
    else:
        ov["training.use_mmd_x"] = False

    if mmdy is not None:
        ov["training.use_mmd_y"] = True
        ov["training.mmd_y_weight"] = float(mmdy)
    else:
        ov["training.use_mmd_y"] = False

    # Truncation
    if z_trunc is not None:
        ov["training.realism_z_trunc"] = float(z_trunc)

    # If not set by the combo, gently increase KS grid defaults from base
    if "training.ks_grid_points_x" not in ov:
        ov["training.ks_grid_points_x"] = 32
    if "training.ks_grid_points_y" not in ov:
        ov["training.ks_grid_points_y"] = 32

    # --- name tag ---
    def tok(label: str, v) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            if v == 0:
                s = "0"
            else:
                mag = abs(math.log10(abs(v))) if v != 0 else 0
                s = f"{v:.2g}" if mag >= 2 else f"{v:.3f}".rstrip("0").rstrip(".")
        else:
            s = str(v)
        return f"{label}{s}"

    parts = [
        name_tag_hint,
        tok("w1x", w1x), tok("nx_", w1x_norm), tok("sx_", w1x_soft), tok("cx_", w1x_clip),
        tok("w1y", w1y), tok("ny_", w1y_norm), tok("sy_", w1y_soft), tok("cy_", w1y_clip),
        tok("ksx", ksx_w), tok("gx", ksx_grid), tok("tx", ksx_tau),
        tok("ksy", ksy_w), tok("gy", ksy_grid), tok("ty", ksy_tau),
        tok("mmdx", mmdx), tok("mmdy", mmdy),
        tok("tr", z_trunc),
    ]
    name_tag = "_".join([p for p in parts if p]).replace("__", "_")
    return ov, name_tag


def build_25_specs(finetune_epochs: int) -> List[Dict[str, Any]]:
    """
    25 realism-focused finetune runs (20 core + 5 extra).
    """
    specs: List[Dict[str, Any]] = []
    common = common_finetune_overrides(finetune_epochs)

    # ---- 20 core ----
    core_params = [
        # A) W1-centric baselines & small shifts
        dict(w1x=5e-6,  w1x_norm="iqr", w1x_soft=1.0, w1x_clip=0.0,
             w1y=0.05,  w1y_norm="iqr", w1y_soft=1.0, w1y_clip=2.0,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=None, ksy_grid=None, ksy_tau=None,
             mmdx=0.15, mmdy=0.50, z_trunc=3.0, name_tag_hint="FT01"),
        dict(w1x=1e-4,  w1x_norm="iqr", w1x_soft=1.0, w1x_clip=0.0,
             w1y=0.05,  w1y_norm="iqr", w1y_soft=1.0, w1y_clip=2.0,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=None, ksy_grid=None, ksy_tau=None,
             mmdx=0.15, mmdy=0.50, z_trunc=3.0, name_tag_hint="FT02"),
        dict(w1x=0.01,  w1x_norm="iqr", w1x_soft=1.0, w1x_clip=0.0,
             w1y=0.01,  w1y_norm="iqr", w1y_soft=1.0, w1y_clip=2.0,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=None, ksy_grid=None, ksy_tau=None,
             mmdx=0.15, mmdy=0.50, z_trunc=3.0, name_tag_hint="FT03"),
        dict(w1x=1e-4,  w1x_norm="rms", w1x_soft=1.0, w1x_clip=0.0,
             w1y=0.05,  w1y_norm="rms", w1y_soft=1.0, w1y_clip=2.0,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=None, ksy_grid=None, ksy_tau=None,
             mmdx=0.15, mmdy=0.50, z_trunc=3.0, name_tag_hint="FT04"),
        dict(w1x=1e-4,  w1x_norm="iqr", w1x_soft=1.0, w1x_clip=0.0,
             w1y=0.05,  w1y_norm="iqr", w1y_soft=1.0, w1y_clip=2.0,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=None, ksy_grid=None, ksy_tau=None,
             mmdx=0.15, mmdy=0.50, z_trunc=2.5, name_tag_hint="FT05"),

        # B) Add KS
        dict(w1x=1e-4,  w1x_norm="iqr", w1x_soft=1.0, w1x_clip=0.0,
             w1y=0.05,  w1y_norm="iqr", w1y_soft=1.0, w1y_clip=2.0,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=0.02, ksy_grid=64,  ksy_tau=0.04,
             mmdx=0.15, mmdy=0.50, z_trunc=3.0, name_tag_hint="FT06"),
        dict(w1x=1e-4,  w1x_norm="iqr", w1x_soft=1.0, w1x_clip=0.0,
             w1y=0.05,  w1y_norm="iqr", w1y_soft=1.0, w1y_clip=2.0,
             ksx_w=0.01, ksx_grid=64,  ksx_tau=0.04,
             ksy_w=0.01, ksy_grid=64,  ksy_tau=0.04,
             mmdx=0.15, mmdy=0.50, z_trunc=3.0, name_tag_hint="FT07"),
        dict(w1x=5e-6,  w1x_norm="iqr", w1x_soft=1.0, w1x_clip=0.0,
             w1y=0.02,  w1y_norm="iqr", w1y_soft=1.0, w1y_clip=2.0,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=0.06, ksy_grid=64,  ksy_tau=0.04,
             mmdx=0.15, mmdy=0.50, z_trunc=3.0, name_tag_hint="FT08"),

        # C) MMD-centric anchors
        dict(w1x=1e-5,  w1x_norm="iqr", w1x_soft=1.0, w1x_clip=0.0,
             w1y=0.02,  w1y_norm="iqr", w1y_soft=1.0, w1y_clip=2.0,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=None, ksy_grid=None, ksy_tau=None,
             mmdx=0.15, mmdy=1.00, z_trunc=3.0, name_tag_hint="FT09"),
        dict(w1x=1e-5,  w1x_norm="iqr", w1x_soft=1.0, w1x_clip=0.0,
             w1y=0.02,  w1y_norm="iqr", w1y_soft=1.0, w1y_clip=2.0,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=None, ksy_grid=None, ksy_tau=None,
             mmdx=0.50, mmdy=1.00, z_trunc=3.0, name_tag_hint="FT10"),
        dict(w1x=None,  w1x_norm=None, w1x_soft=None, w1x_clip=None,
             w1y=None,  w1y_norm=None, w1y_soft=None, w1y_clip=None,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=None, ksy_grid=None, ksy_tau=None,
             mmdx=0.50, mmdy=1.00, z_trunc=3.0, name_tag_hint="FT11"),

        # D) More W1 variants
        dict(w1x=2e-4,  w1x_norm="iqr", w1x_soft=0.9, w1x_clip=1.0,
             w1y=0.03,  w1y_norm="iqr", w1y_soft=0.9, w1y_clip=1.5,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=None, ksy_grid=None, ksy_tau=None,
             mmdx=0.20, mmdy=0.60, z_trunc=3.0, name_tag_hint="FT12"),
        dict(w1x=0.005, w1x_norm="iqr", w1x_soft=1.2, w1x_clip=2.0,
             w1y=0.02,  w1y_norm="iqr", w1y_soft=1.2, w1y_clip=2.0,
             ksx_w=0.008, ksx_grid=48, ksx_tau=0.05,
             ksy_w=None,  ksy_grid=None, ksy_tau=None,
             mmdx=0.15, mmdy=0.50, z_trunc=2.5, name_tag_hint="FT13"),
        dict(w1x=0.002, w1x_norm="rms", w1x_soft=1.0, w1x_clip=1.8,
             w1y=0.04,  w1y_norm="rms", w1y_soft=1.0, w1y_clip=1.8,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=0.02,  ksy_grid=48, ksy_tau=0.05,
             mmdx=0.20, mmdy=0.70, z_trunc=3.0, name_tag_hint="FT14"),

        # E) Robustness / clipping
        dict(w1x=1e-4,  w1x_norm="iqr", w1x_soft=0.75, w1x_clip=1.5,
             w1y=0.05,  w1y_norm="iqr", w1y_soft=0.75, w1y_clip=1.5,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=None, ksy_grid=None, ksy_tau=None,
             mmdx=0.15, mmdy=0.50, z_trunc=3.0, name_tag_hint="FT15"),
        dict(w1x=1e-4,  w1x_norm="iqr", w1x_soft=1.5,  w1x_clip=3.0,
             w1y=0.05,  w1y_norm="iqr", w1y_soft=1.5,  w1y_clip=3.0,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=None, ksy_grid=None, ksy_tau=None,
             mmdx=0.15, mmdy=0.50, z_trunc=3.0, name_tag_hint="FT16"),

        # F) KS sampling stress
        dict(w1x=1e-4,  w1x_norm="iqr", w1x_soft=1.0, w1x_clip=0.0,
             w1y=0.05,  w1y_norm="iqr", w1y_soft=1.0, w1y_clip=2.0,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=0.02, ksy_grid=64,  ksy_tau=0.04,
             mmdx=0.15, mmdy=0.50, z_trunc=2.0, name_tag_hint="FT17"),

        # G) Sampling/no trunc or alt norms
        dict(w1x=1e-4,  w1x_norm="iqr", w1x_soft=1.25, w1x_clip=2.5,
             w1y=0.05,  w1y_norm="iqr", w1y_soft=1.25, w1y_clip=2.5,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=None, ksy_grid=None, ksy_tau=None,
             mmdx=0.15, mmdy=0.50, z_trunc=0.0, name_tag_hint="FT18"),
        dict(w1x=0.005, w1x_norm="rms", w1x_soft=1.0, w1x_clip=0.0,
             w1y=0.02,  w1y_norm="rms", w1y_soft=1.0, w1y_clip=2.0,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=0.04, ksy_grid=64,  ksy_tau=0.04,
             mmdx=None, mmdy=1.00, z_trunc=2.5, name_tag_hint="FT19"),
        dict(w1x=0.02,  w1x_norm="iqr", w1x_soft=0.9, w1x_clip=2.0,
             w1y=0.02,  w1y_norm="iqr", w1y_soft=0.9, w1y_clip=2.0,
             ksx_w=0.008, ksx_grid=64, ksx_tau=0.05,
             ksy_w=0.008, ksy_grid=64, ksy_tau=0.05,
             mmdx=0.20, mmdy=0.80, z_trunc=2.5, name_tag_hint="FT20"),
    ]
    assert len(core_params) == 20

    # ---- 5 extras ----
    extra_params = [
        dict(w1x=5e-5, w1x_norm="iqr", w1x_soft=1.0, w1x_clip=1.5,
             w1y=0.02, w1y_norm="iqr", w1y_soft=1.0, w1y_clip=1.5,
             ksx_w=0.03, ksx_grid=96, ksx_tau=0.035,
             ksy_w=0.06, ksy_grid=96, ksy_tau=0.035,
             mmdx=0.25, mmdy=1.20, z_trunc=2.5, name_tag_hint="FT21"),

        dict(w1x=1e-4, w1x_norm="iqr", w1x_soft=1.0, w1x_clip=0.0,
             w1y=0.08, w1y_norm="iqr", w1y_soft=0.9, w1y_clip=2.0,
             ksx_w=0.01, ksx_grid=64, ksx_tau=0.04,
             ksy_w=None,  ksy_grid=None, ksy_tau=None,
             mmdx=0.30, mmdy=0.80, z_trunc=3.0, name_tag_hint="FT22"),

        dict(w1x=0.02, w1x_norm="iqr", w1x_soft=1.0, w1x_clip=2.0,
             w1y=0.02, w1y_norm="iqr", w1y_soft=1.0, w1y_clip=2.0,
             ksx_w=0.02, ksx_grid=64, ksx_tau=0.05,
             ksy_w=0.02, ksy_grid=64, ksy_tau=0.05,
             mmdx=0.20, mmdy=0.80, z_trunc=2.5, name_tag_hint="FT23"),

        dict(w1x=None, w1x_norm=None, w1x_soft=None, w1x_clip=None,
             w1y=None, w1y_norm=None, w1y_soft=None, w1y_clip=None,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=None, ksy_grid=None, ksy_tau=None,
             mmdx=0.6, mmdy=1.4, z_trunc=3.0, name_tag_hint="FT24"),

        dict(w1x=1e-4, w1x_norm="iqr", w1x_soft=1.1, w1x_clip=1.5,
             w1y=0.01, w1y_norm="iqr", w1y_soft=1.1, w1y_clip=1.8,
             ksx_w=None, ksx_grid=None, ksx_tau=None,
             ksy_w=0.12, ksy_grid=128, ksy_tau=0.03,
             mmdx=0.15, mmdy=0.9, z_trunc=2.0, name_tag_hint="FT25"),
    ]
    assert len(extra_params) == 5

    all_params = core_params + extra_params
    assert len(all_params) == 25

    for p in all_params:
        ov, tag = ks_w1_mmd_recipe(**p)
        specs.append({
            "name_tag": tag,
            "overrides": {**common, **ov},
        })
    return specs


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="FlowGen finetune-only sweep (KS/W1/MMD realism).")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--base_config", type=str, default="flowgen.yaml",
                    help="YAML filename under ROOT_PATH/config to clone per run.")
    ap.add_argument("--pretrained_path", type=str, default="",
                    help="If omitted or placeholder-like, uses DEFAULT_PRETRAINED.")
    ap.add_argument("--run_prefix", type=str, default="FT",
                    help="Prefix added to each run's base_name.")
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--finetune_epochs", type=int, default=10)
    ap.add_argument("--sleep_on_fail_sec", type=int, default=60,
                    help="Initial backoff on failure; doubles each retry until 10x.")
    args = ap.parse_args()

    device = args.device
    seed = int(args.seed)
    finetune_epochs = int(args.finetune_epochs)

    # Resolve pretrained checkpoint with default fallback
    passed = (args.pretrained_path or "").strip()
    if (not passed) or passed.startswith("/abs/path/to"):
        pretrained_path = str(DEFAULT_PRETRAINED)
    else:
        pretrained_path = passed

    # Base config — copy full tree and only tweak per-run keys
    base_cfg = load_yaml_config(args.base_config)

    # Sweep specs (25)
    specs = build_25_specs(finetune_epochs)

    # Build names
    def make_base_name(tag: str) -> str:
        prefix = f"{args.run_prefix}_" if args.run_prefix else ""
        return f"{prefix}flowgen_{tag}_seed{seed}"

    planned = [(spec, make_base_name(spec["name_tag"])) for spec in specs]

    # Load state and skip finished
    state = load_state()
    statuses = state.get("statuses", {})
    attempts = state.get("attempts", {})

    todo, skipped = [], []
    for spec, base in planned:
        if run_already_finished(base):
            statuses[base] = {"status": "ok", "error": "", "attempts": attempts.get(base, 0)}
            skipped.append(base)
        else:
            todo.append((spec, base))
            statuses.setdefault(base, {"status": "pending", "error": "", "attempts": attempts.get(base, 0)})

    state["statuses"] = statuses
    state["attempts"] = attempts
    save_state(state)

    print(f"🧪 Planned: {len(specs)}  |  🟢 To run: {len(todo)}  |  ⏭️ Skipped (already finished): {len(skipped)}")
    if skipped:
        preview = "\n".join(f"   - {b}" for b in skipped[:10])
        more = "" if len(skipped) <= 10 else f"\n   ... and {len(skipped)-10} more"
        print(f"⏭️ Skipped:\n{preview}{more}")

    summary_rows: List[Dict[str, Any]] = []
    t_start = time.time()

    # Retry loop
    pending = list(todo)
    cycle = 0

    try:
        while pending:
            cycle += 1
            print(f"\n🔁 Retry cycle #{cycle} — remaining runs: {len(pending)}")
            next_pending: List[Tuple[Dict[str, Any], str]] = []

            for idx, (spec, base_name) in enumerate(pending, start=1):
                a = attempts.get(base_name, 0)
                if a >= args.max_retries:
                    print(f"⛔ Max retries reached: {base_name} — skipping permanently.")
                    statuses[base_name] = {"status": "failed", "error": statuses.get(base_name, {}).get("error", ""), "attempts": a}
                    continue

                # If finished meanwhile, skip
                if run_already_finished(base_name):
                    statuses[base_name] = {"status": "ok", "error": "", "attempts": a}
                    print(f"✅ Detected finished: {base_name}")
                    continue

                # Build config for this run: full copy + tiny patch set
                run_cfg = apply_overrides(base_cfg, spec["overrides"])
                # Ensure realism is enforced in FT even if base file differs
                deep_set(run_cfg, "training.enforce_realism", True)

                # Write temp YAML under config/
                cfg_filename, cfg_full_path = write_temp_config(run_cfg, spec["name_tag"])

                print(f"\n[{idx}/{len(pending)}] 🚀 {base_name}")
                t0 = time.time()
                status, error_msg = "ok", ""
                try:
                    _ = train_flowgen_pipeline(
                        condition_col="type",
                        config_filename=cfg_filename,  # file in ROOT_PATH/config
                        base_name=base_name,
                        device=device,
                        seed=seed,
                        verbose=True,
                        skip_phase1=True,
                        pretrained_path=pretrained_path,
                    )
                except Exception as e:
                    status, error_msg = "failed", f"{type(e).__name__}: {e}"
                    print(f"❌ Run failed: {base_name}\n    → {error_msg}")

                dt = time.time() - t0
                print(f"⏱️ Duration: {fmt_hms(dt)}")

                # Best-effort clean of temp YAML
                try:
                    if cfg_full_path.exists():
                        cfg_full_path.unlink()
                except Exception:
                    pass

                # Consider it successful iff results YAML exists
                run_dir = find_latest_run_dir(base_name)
                success = (status == "ok") and has_results_yaml(run_dir)

                # Update state
                attempts[base_name] = a + 1
                if success:
                    statuses[base_name] = {"status": "ok", "error": "", "attempts": attempts[base_name]}
                else:
                    statuses[base_name] = {"status": "failed", "error": error_msg, "attempts": attempts[base_name]}

                state["statuses"] = statuses
                state["attempts"] = attempts
                save_state(state)

                # Summary row
                summary_rows.append({
                    "base_name": base_name,
                    "name_tag": spec["name_tag"],
                    "status": "ok" if success else "failed",
                    "error": "" if success else error_msg,
                    "attempts": attempts[base_name],
                    "duration_sec": round(dt, 2),
                })

                # Queue for retry
                if not success and attempts[base_name] < args.max_retries:
                    next_pending.append((spec, base_name))
                    # Backoff sleep (per run)
                    k = attempts[base_name]
                    sleep_s = min(args.sleep_on_fail_sec * (2 ** max(0, k - 1)), args.sleep_on_fail_sec * 10)
                    print(f"🕒 Backing off {sleep_s}s before next run...")
                    time.sleep(sleep_s)

            pending = next_pending

    except KeyboardInterrupt:
        print("\n🛑 Interrupted — writing partial summary...")

    # Final summary
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_csv = SWEEP_ROOT / f"sweep_flowgen_finetune_{ts}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    total_elapsed = time.time() - t_start
    ok_ct = sum(1 for r in summary_rows if r["status"] == "ok")
    fail_ct = sum(1 for r in summary_rows if r["status"] != "ok")
    print(f"\n✅ Sweep finished. OK: {ok_ct}  |  Failed: {fail_ct}")
    print(f"📝 Summary CSV: {summary_csv}")
    print(f"⏲️ Total elapsed: {fmt_hms(total_elapsed)}")


if __name__ == "__main__":
    main()
