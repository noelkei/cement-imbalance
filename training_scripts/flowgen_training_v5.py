#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
flowgen_training_v180.py

Runs the 180-config plan:
- Block A (57): 10:1 ladder w1x=10*w1y with w1y∈[2..20]; baseline + KS-Y probe + slight MMD tweak
- Block B (60): original MMD/KS grid & W1 variants (kept from v90)
- Block C (25): inverse/spacing variants incl. mmd_xy toggles and heavy MMD stress
- Block E (38): high-magnitude 10:1 ladder w1x 700→1000, w1y 70→100 + light probes

Prefix is FT180_{finetune_epochs}. Prints per-run duration and total elapsed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

# ---- project utils ----
from training.utils import ROOT_PATH, load_yaml_config
from training.train_flowgen import train_flowgen_pipeline  # your wrapper

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

STATE_PATH = SWEEP_ROOT / "flowgen_training_v180_state.json"

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
    cfg2 = copy.deepcopy(cfg)
    for k, v in overrides.items():
        deep_set(cfg2, k, v)
    return cfg2

def _sanitize_name(s: str) -> str:
    import re as _re
    return _re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def write_temp_config(cfg_dict: dict, stem: str) -> Tuple[str, Path]:
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
# Name-tag utilities
# -------------------------
def _ftok(label: str, v: float | int | str | None, prec: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return f"{label}{v}"
    if isinstance(v, int):
        return f"{label}{v}"
    if v == 0:
        s = "0"
    elif 1e-2 <= v < 100:
        s = f"{v:.{prec}f}".rstrip("0").rstrip(".")
    else:
        s = f"{v:.0e}".replace("+0", "").replace("+", "")
    return f"{label}{s}"

def make_name_tag(hint: str, ov: Dict[str, Any], extra: str = "") -> str:
    toks = [hint]
    g = ov.get
    toks += [
        _ftok("w1x", g("training.w1_x_weight")),
        _ftok("nx_", g("training.w1_x_norm")),
        _ftok("sx_", g("training.w1_x_softclip_s")),
        _ftok("cx_", g("training.w1_x_clip_perdim")),
        _ftok("w1y", g("training.w1_y_weight")),
        _ftok("ny_", g("training.w1_y_norm")),
        _ftok("sy_", g("training.w1_y_softclip_s")),
        _ftok("cy_", g("training.w1_y_clip_perdim")),
        _ftok("ksx", g("training.ks_x_weight")),
        _ftok("gx",  g("training.ks_grid_points_x")),
        _ftok("tx",  g("training.ks_tau_x")),
        _ftok("ksy", g("training.ks_y_weight")),
        _ftok("gy",  g("training.ks_grid_points_y")),
        _ftok("ty",  g("training.ks_tau_y")),
        _ftok("mmdx", g("training.mmd_x_weight")),
        _ftok("mmdy", g("training.mmd_y_weight")),
        _ftok("tr",   g("training.realism_z_trunc")),
    ]
    if "training.learning_rate" in ov:
        lr = ov["training.learning_rate"]
        toks.append("lr1e-4" if abs(lr - 1e-4) < 1e-12 else _ftok("lr", lr))
    if "training.realism_ramp_epochs" in ov and ov["training.realism_ramp_epochs"]:
        toks.append(_ftok("ramp", ov["training.realism_ramp_epochs"]))
    if extra:
        toks.append(extra)
    name = "_".join([t for t in toks if t]).replace("__", "_")
    return name

# -------------------------
# Common finetune overrides
# -------------------------
def common_finetune_overrides(finetune_epochs: int) -> Dict[str, Any]:
    return {
        "training.finetune_num_epochs": int(finetune_epochs),
        "training.enforce_realism": True,
        "training.realism_warmup_epochs": 0,
        "training.realism_ramp_epochs": 0,
        "training.realism_stride_batches": 1,
        "training.realism_stride_epochs": 1,
        "training.realism_scale_mode": "keep_mean",
        "training.use_nll": True,
        "training.nll_weight": 1.0,
        "training.class_weighting": "uniform",
        "training.ref_min": 100,
        "training.syn_min": 100,
        "training.save_results": True,
        "training.save_states": False,
        "training.save_model": False,
    }

# -------------------------
# Spec builder (180 runs)
# -------------------------
def build_180_specs(finetune_epochs: int) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    common = common_finetune_overrides(finetune_epochs)

    base = {
        # W1
        "training.use_w1_x": True,
        "training.use_w1_y": True,
        "training.w1_x_weight": 1e-4,
        "training.w1_y_weight": 0.05,
        "training.w1_x_norm": "iqr",
        "training.w1_y_norm": "iqr",
        "training.w1_x_softclip_s": 1.25,
        "training.w1_y_softclip_s": 1.25,
        "training.w1_x_clip_perdim": 2.0,
        "training.w1_y_clip_perdim": 2.0,
        # MMD
        "training.use_mmd_x": True,
        "training.use_mmd_y": True,
        "training.mmd_x_weight": 0.50,  # default mmdx
        "training.mmd_y_weight": 1.50,
        "training.use_mmd_xy": False,
        # KS defaults off
        "training.use_ks_x": False,
        "training.use_ks_y": False,
        # Misc
        "training.realism_ramp_epochs": 0,
        "training.realism_z_trunc": 3.0,
        "training.learning_rate": 1e-4,
    }

    def add(hint: str, **ov):
        ov2 = {**base, **ov}
        name_tag = make_name_tag(hint, ov2)
        specs.append({"name_tag": name_tag, "overrides": {**common, **ov2}})

    # ---------- Block A (57) — 10:1 ladder in low/mid range
    # y ∈ [2..20], x = 10*y (so x ∈ [20..200])
    for i, wy in enumerate(range(2, 21), start=1):  # 19 values
        wx = wy * 10
        # A01..A19 baseline
        add(f"A{i:02d}", **{"training.w1_y_weight": float(wy), "training.w1_x_weight": float(wx)})
        # A20..A38 KS-Y light probe (ky=0.01, tau=0.04, gy=64)
        add(f"A{i+19:02d}",
            **{"training.w1_y_weight": float(wy), "training.w1_x_weight": float(wx),
               "training.use_ks_y": True, "training.ks_y_weight": 0.01,
               "training.ks_grid_points_y": 64, "training.ks_tau_y": 0.04})
        # A39..A57 slight MMD tweak
        add(f"A{i+38:02d}",
            **{"training.w1_y_weight": float(wy), "training.w1_x_weight": float(wx),
               "training.mmd_x_weight": 0.48, "training.mmd_y_weight": 1.25})

    # ---------- Block B (60) — original v90 block kept
    mmdx_vals = [0.48, 0.52, 0.58, 0.62]
    mmdy_vals = [1.0, 1.25, 1.5, 1.75]
    idx = 1
    for mx in mmdx_vals:
        for my in mmdy_vals:
            add(f"B{idx:02d}", **{"training.mmd_x_weight": mx, "training.mmd_y_weight": my}); idx += 1
    b_ks_y = [
        (0.48, 1.25, 0.012, 80), (0.48, 1.50, 0.015, 80),
        (0.52, 1.25, 0.012, 112), (0.52, 1.50, 0.015, 112),
        (0.58, 1.25, 0.012, 80),  (0.58, 1.50, 0.018, 112),
        (0.62, 1.25, 0.015, 80),  (0.62, 1.50, 0.018, 112),
    ]
    for (mx, my, ky, gy) in b_ks_y:
        add(f"B{idx:02d}",
            **{"training.mmd_x_weight": mx, "training.mmd_y_weight": my,
               "training.use_ks_y": True, "training.ks_y_weight": ky,
               "training.ks_grid_points_y": gy, "training.ks_tau_y": 0.045}); idx += 1
    b_both = [
        (0.52, 1.50, 0.007, 80, 0.045, 0.012, 80, 0.045),
        (0.52, 1.25, 0.009, 112, 0.045, 0.012, 80, 0.045),
        (0.58, 1.50, 0.011, 112, 0.045, 0.015, 112, 0.045),
        (0.58, 1.25, 0.007, 80, 0.045, 0.018, 112, 0.045),
        (0.48, 1.50, 0.009, 112, 0.045, 0.012, 112, 0.045),
        (0.62, 1.25, 0.011, 80, 0.045, 0.015, 80, 0.045),
        (0.48, 1.25, 0.007, 112, 0.045, 0.018, 80, 0.045),
        (0.62, 1.50, 0.009, 80, 0.045, 0.015, 112, 0.045),
        (0.58, 1.75, 0.011, 112, 0.045, 0.012, 112, 0.045),
        (0.52, 1.75, 0.009, 80, 0.045, 0.018, 80, 0.045),
        (0.52, 1.00, 0.007, 112, 0.045, 0.015, 112, 0.045),
        (0.58, 1.00, 0.011, 80, 0.045, 0.012, 80, 0.045),
    ]
    for (mx, my, kx, gx, tx, ky, gy, ty) in b_both:
        add(f"B{idx:02d}",
            **{"training.mmd_x_weight": mx, "training.mmd_y_weight": my,
               "training.use_ks_x": True, "training.ks_x_weight": kx,
               "training.ks_grid_points_x": gx, "training.ks_tau_x": tx,
               "training.use_ks_y": True, "training.ks_y_weight": ky,
               "training.ks_grid_points_y": gy, "training.ks_tau_y": ty}); idx += 1
    b_w1 = [
        (0.00015, 0.06, 0.55, 1.25),
        (0.00020, 0.06, 0.55, 1.50),
        (0.00025, 0.06, 0.57, 1.50),
        (0.00030, 0.06, 0.57, 1.25),
        (0.00020, 0.08, 0.52, 1.25),
        (0.00025, 0.08, 0.52, 1.50),
        (0.00030, 0.08, 0.48, 1.25),
        (0.00015, 0.08, 0.48, 1.50),
    ]
    for (wx, wy, mx, my) in b_w1:
        add(f"B{idx:02d}",
            **{"training.w1_x_weight": wx, "training.w1_y_weight": wy,
               "training.mmd_x_weight": mx, "training.mmd_y_weight": my}); idx += 1
    add(f"B{idx:02d}",
        **{"training.w1_x_norm": "rms", "training.w1_y_norm": "iqr",
           "training.mmd_x_weight": 0.52, "training.mmd_y_weight": 1.50}); idx += 1
    add(f"B{idx:02d}",
        **{"training.w1_x_norm": "iqr", "training.w1_y_norm": "rms",
           "training.mmd_x_weight": 0.58, "training.mmd_y_weight": 1.25}); idx += 1
    add(f"B{idx:02d}",
        **{"training.w1_x_softclip_s": 1.5, "training.w1_y_softclip_s": 1.5,
           "training.w1_x_clip_perdim": 2.5, "training.w1_y_clip_perdim": 2.5,
           "training.mmd_x_weight": 0.52, "training.mmd_y_weight": 1.50}); idx += 1
    add(f"B{idx:02d}",
        **{"training.w1_x_softclip_s": 0.9, "training.w1_y_softclip_s": 0.9,
           "training.w1_x_clip_perdim": 1.5, "training.w1_y_clip_perdim": 1.5,
           "training.mmd_x_weight": 0.58, "training.mmd_y_weight": 1.25}); idx += 1
    add(f"B{idx:02d}",
        **{"training.w1_x_norm": "rms", "training.w1_y_norm": "rms",
           "training.mmd_x_weight": 0.62, "training.mmd_y_weight": 1.25}); idx += 1
    add(f"B{idx:02d}",
        **{"training.w1_x_norm": "rms", "training.w1_y_norm": "rms",
           "training.mmd_x_weight": 0.48, "training.mmd_y_weight": 1.75}); idx += 1
    add(f"B{idx:02d}",
        **{"training.w1_x_clip_perdim": 3.0, "training.w1_y_clip_perdim": 3.0,
           "training.mmd_x_weight": 0.52, "training.mmd_y_weight": 1.25}); idx += 1
    add(f"B{idx:02d}",
        **{"training.w1_x_clip_perdim": 1.8, "training.w1_y_clip_perdim": 1.8,
           "training.mmd_x_weight": 0.58, "training.mmd_y_weight": 1.50}); idx += 1
    for rr, ss, mx, my in [
        (200, 200, 0.58, 1.50),
        (300, 300, 0.52, 1.50),
        (400, 400, 0.52, 1.25),
        (500, 500, 0.58, 1.25),
    ]:
        add(f"B{idx:02d}", **{"training.ref_min": rr, "training.syn_min": ss,
                              "training.mmd_x_weight": mx, "training.mmd_y_weight": my}); idx += 1
    add(f"B{idx:02d}",
        **{"training.learning_rate": 2e-4,
           "training.mmd_x_weight": 0.52, "training.mmd_y_weight": 1.50}); idx += 1
    add(f"B{idx:02d}",
        **{"training.learning_rate": 2e-4,
           "training.mmd_x_weight": 0.58, "training.mmd_y_weight": 1.25}); idx += 1
    add(f"B{idx:02d}",
        **{"training.realism_ramp_epochs": 4,
           "training.mmd_x_weight": 0.52, "training.mmd_y_weight": 1.25}); idx += 1
    add(f"B{idx:02d}",
        **{"training.realism_ramp_epochs": 4,
           "training.mmd_x_weight": 0.58, "training.mmd_y_weight": 1.50}); idx += 1

    # ---------- Block C (25) — inverse / spacing + mmd_xy & heavy MMDs
    # C01–C20: for x in {20,40,...,200}, y = clamp(round(400/x), 2..20)
    def clamp(v, lo, hi): return max(lo, min(hi, v))
    xs = list(range(20, 201, 20))  # 10 values
    cidx = 1
    for x in xs:
        y = clamp(int(round(400 / x)), 2, 20)
        # baseline
        add(f"C{cidx:02d}", **{"training.w1_x_weight": float(x), "training.w1_y_weight": float(y)}); cidx += 1
        # + mmd_xy = 5.0
        add(f"C{cidx:02d}",
            **{"training.w1_x_weight": float(x), "training.w1_y_weight": float(y),
               "training.use_mmd_xy": True, "training.mmd_xy_weight": 5.0}); cidx += 1

    # C21–C25: heavier MMD stress at representative pairs (and mmd_xy=5)
    reps = [(20, 2), (60, 3), (100, 5), (140, 7), (200, 10)]
    for (x, y) in reps:
        add(f"C{cidx:02d}",
            **{"training.w1_x_weight": float(x), "training.w1_y_weight": float(y),
               "training.mmd_x_weight": 5.0, "training.mmd_y_weight": 3.0,
               "training.use_mmd_xy": True, "training.mmd_xy_weight": 5.0})
        cidx += 1

    # ---------- Block E (38) — high magnitude ladder + light probes
    # E01–E31: x 700→1000 step 10, y 70→100
    anchors = list(range(700, 1001, 10))
    for j, wx in enumerate(anchors, start=1):
        wy = 70 + (j - 1)
        add(f"E{j:02d}", **{"training.w1_x_weight": float(wx), "training.w1_y_weight": float(wy)})
    # E32–E35: alt MMDs at four anchors
    for k, wx in enumerate([700, 800, 900, 1000], start=32):
        wy = wx // 10
        add(f"E{k:02d}",
            **{"training.w1_x_weight": float(wx), "training.w1_y_weight": float(wy),
               "training.mmd_x_weight": 0.48, "training.mmd_y_weight": 1.25})
    # E36–E37: light KS-Y uptick
    for k, wx in zip([36, 37], [800, 900]):
        wy = wx // 10
        add(f"E{k:02d}",
            **{"training.w1_x_weight": float(wx), "training.w1_y_weight": float(wy),
               "training.use_ks_y": True, "training.ks_y_weight": 0.012,
               "training.ks_grid_points_y": 80, "training.ks_tau_y": 0.04})
    # E38: softclip/clip bump at 900/90
    add("E38",
        **{"training.w1_x_weight": 900.0, "training.w1_y_weight": 90.0,
           "training.w1_x_softclip_s": 1.5, "training.w1_y_softclip_s": 1.5,
           "training.w1_x_clip_perdim": 2.5, "training.w1_y_clip_perdim": 2.5})

    assert len(specs) == 180, f"Expected 180 specs, got {len(specs)}"
    return specs

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="FlowGen finetune-only sweep (180 combos).")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--base_config", type=str, default="flowgen.yaml",
                    help="YAML filename under ROOT_PATH/config to clone per run.")
    ap.add_argument("--pretrained_path", type=str, default="",
                    help="If omitted or placeholder-like, uses DEFAULT_PRETRAINED.")
    ap.add_argument("--finetune_epochs_list", type=str, default="50",
                    help="Comma-separated list of FT epochs to try, e.g. '30,50'.")
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--sleep_on_fail_sec", type=int, default=60,
                    help="Initial backoff on failure; doubles each retry until 10x.")
    args = ap.parse_args()

    device = args.device
    seed = int(args.seed)

    passed = (args.pretrained_path or "").strip()
    if (not passed) or passed.startswith("/abs/path/to"):
        pretrained_path = str(DEFAULT_PRETRAINED)
    else:
        pretrained_path = passed

    base_cfg = load_yaml_config(args.base_config)
    ft_buckets = [int(x.strip()) for x in args.finetune_epochs_list.split(",") if x.strip()]

    state = load_state()
    statuses = state.get("statuses", {})
    attempts = state.get("attempts", {})

    all_planned: List[Tuple[Dict[str, Any], str]] = []
    total_to_run = 0
    total_skipped = 0

    for fe in ft_buckets:
        specs = build_180_specs(finetune_epochs=fe)

        def make_base_name(tag: str) -> str:
            prefix = f"FT180_{fe}_"
            return f"{prefix}flowgen_{tag}_seed{seed}"

        planned = [(spec, make_base_name(spec["name_tag"])) for spec in specs]
        for spec, base in planned:
            if run_already_finished(base):
                statuses[base] = {"status": "ok", "error": "", "attempts": attempts.get(base, 0)}
                total_skipped += 1
            else:
                all_planned.append((spec, base))
                if base not in statuses:
                    statuses[base] = {"status": "pending", "error": "", "attempts": attempts.get(base, 0)}
                total_to_run += 1

    state["statuses"] = statuses
    state["attempts"] = attempts
    save_state(state)

    print(f"🧪 Planned total: {len(ft_buckets)}×180 = {len(ft_buckets)*180}")
    print(f"🟢 To run now: {total_to_run}  |  ⏭️ Skipped (already finished): {total_skipped}")

    summary_rows: List[Dict[str, Any]] = []
    t_start = time.time()

    pending = list(all_planned)
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

                if run_already_finished(base_name):
                    statuses[base_name] = {"status": "ok", "error": "", "attempts": a}
                    print(f"✅ Detected finished: {base_name}")
                    continue

                run_cfg = apply_overrides(base_cfg, spec["overrides"])
                deep_set(run_cfg, "training.enforce_realism", True)

                fe_match = re.search(r"FT180_(\d+)_", base_name)
                fe_tag = fe_match.group(1) if fe_match else "xx"
                cfg_stem = f"{fe_tag}_{spec['name_tag']}"

                cfg_filename, cfg_full_path = write_temp_config(run_cfg, cfg_stem)

                print(f"\n[{idx}/{len(pending)}] 🚀 {base_name}")
                t0 = time.time()
                status, error_msg = "ok", ""
                try:
                    _ = train_flowgen_pipeline(
                        condition_col="type",
                        config_filename=cfg_filename,
                        base_name=base_name,
                        device=device,
                        seed=seed,
                        verbose=False,
                        skip_phase1=True,
                        pretrained_path=pretrained_path,
                    )
                except Exception as e:
                    status, error_msg = "failed", f"{type(e).__name__}: {e}"
                    print(f"❌ Run failed: {base_name}\n    → {error_msg}")

                dt = time.time() - t0
                print(f"⏱️ Duration: {fmt_hms(dt)}")

                try:
                    if cfg_full_path.exists():
                        cfg_full_path.unlink()
                except Exception:
                    pass

                run_dir = find_latest_run_dir(base_name)
                success = (status == "ok") and has_results_yaml(run_dir)

                attempts[base_name] = a + 1
                statuses[base_name] = {"status": "ok" if success else "failed",
                                       "error": "" if success else error_msg,
                                       "attempts": attempts[base_name]}
                state["statuses"] = statuses
                state["attempts"] = attempts
                save_state(state)

                summary_rows.append({
                    "base_name": base_name,
                    "name_tag": spec["name_tag"],
                    "status": "ok" if success else "failed",
                    "error": "" if success else error_msg,
                    "attempts": attempts[base_name],
                    "duration_sec": round(dt, 2),
                })

                if not success and attempts[base_name] < args.max_retries:
                    next_pending.append((spec, base_name))
                    k = attempts[base_name]
                    sleep_s = min(args.sleep_on_fail_sec * (2 ** max(0, k - 1)), args.sleep_on_fail_sec * 10)
                    print(f"🕒 Backing off {sleep_s}s before next run...")
                    time.sleep(sleep_s)

            pending = next_pending

    except KeyboardInterrupt:
        print("\n🛑 Interrupted — writing partial summary...")

    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_csv = SWEEP_ROOT / f"sweep_flowgen_finetune_v180_{ts}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    total_elapsed = time.time() - t_start
    ok_ct = sum(1 for r in summary_rows if r["status"] == "ok")
    fail_ct = sum(1 for r in summary_rows if r["status"] != "ok")
    print(f"\n✅ Sweep finished. OK: {ok_ct}  |  Failed: {fail_ct}")
    print(f"📝 Summary CSV: {summary_csv}")
    print(f"⏲️ Total elapsed: {fmt_hms(total_elapsed)}")

if __name__ == "__main__":
    main()
