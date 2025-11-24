#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
flowgen_training_v3.py

Finetune-only sweep for FlowGen derived from Top-10 patterns, expressed as
diff-only patches over a single base config. Builds **80** runs per finetune
epoch bucket (default 30).

Key points:
- No per-run configs are read; we patch a single base YAML.
- No warmup; ramp <= 15 only when specified.
- Adds small MMD-XY weights (0.01–0.10), class_weighting tweaks, LR=1e-4/5e-4.
- Names keep W1/KS/MMD/TR tokens for your parser; extra tokens are appended.
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

STATE_PATH = SWEEP_ROOT / "flowgen_training_v3_state.json"

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
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


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
    # float
    if v == 0:
        s = "0"
    elif v >= 1e-2 and v < 100:
        s = f"{v:.{prec}f}".rstrip("0").rstrip(".")
    else:
        s = f"{v:.0e}".replace("+0", "").replace("+", "")
    return f"{label}{s}"

def make_name_tag(hint: str, ov: Dict[str, Any], extra: str = "") -> str:
    toks = [hint]
    g = ov.get

    # Core tokens (your parser already understands these)
    toks += [
        _ftok("w1x", g("training.w1_x_weight")),
        _ftok("nx_",  g("training.w1_x_norm")),
        _ftok("sx_",  g("training.w1_x_softclip_s")),
        _ftok("cx_",  g("training.w1_x_clip_perdim")),
        _ftok("w1y",  g("training.w1_y_weight")),
        _ftok("ny_",  g("training.w1_y_norm")),
        _ftok("sy_",  g("training.w1_y_softclip_s")),
        _ftok("cy_",  g("training.w1_y_clip_perdim")),
        _ftok("ksx",  g("training.ks_x_weight")),
        _ftok("gx",   g("training.ks_grid_points_x")),
        _ftok("tx",   g("training.ks_tau_x")),
        _ftok("ksy",  g("training.ks_y_weight")),
        _ftok("gy",   g("training.ks_grid_points_y")),
        _ftok("ty",   g("training.ks_tau_y")),
        _ftok("mmdx", g("training.mmd_x_weight")),
        _ftok("mmdy", g("training.mmd_y_weight")),
        _ftok("tr",   g("training.realism_z_trunc")),
    ]

    # Extra tokens (safe for your parser to ignore)
    if ov.get("training.use_mmd_xy"):
        toks.append(_ftok("xy", g("training.mmd_xy_weight")))
    if "training.class_weighting" in ov:
        toks.append(_ftok("cw_", g("training.class_weighting")))
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
        "training.realism_stride_batches": 1,
        "training.realism_stride_epochs": 1,
        "training.realism_scale_mode": "keep_mean",
        "training.use_nll": True,
        "training.nll_weight": 1.0,
        "training.save_results": True,
        "training.save_states": False,
        "training.save_model": False,
    }

# -------------------------
# Spec builder (80 runs)
# -------------------------
def build_80_specs(finetune_epochs: int) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    common = common_finetune_overrides(finetune_epochs)

    # --- Anchors are the “diff-only” patches from your Top-10 (applied to base)
    A = {}  # FG66 (baseline-ish in your table)
    B = {"training.ks_tau_y": 0.05, "training.mmd_x_weight": 0.5, "training.mmd_y_weight": 1.5,
         "training.realism_z_trunc": 0.0, "training.w1_y_weight": 0.05}
    C = {"training.ks_y_weight": 0.02, "training.mmd_x_weight": 0.15, "training.mmd_y_weight": 0.5,
         "training.realism_z_trunc": 3.0, "training.w1_y_weight": 0.05}
    D = {}  # FG66_50 (only ft changed previously; we’ll unify with ft=30)
    E = {"training.ks_tau_y": 0.05, "training.ks_y_weight": 0.02, "training.mmd_y_weight": 1.5,
         "training.realism_z_trunc": 3.0, "training.w1_x_norm": "rms", "training.w1_y_weight": 0.05}
    F = {"training.ks_grid_points_y": 96, "training.ks_tau_y": 0.05, "training.ks_y_weight": 0.02,
         "training.mmd_y_weight": 1.5, "training.realism_z_trunc": 3.0, "training.w1_y_weight": 0.05}
    G = {"training.ks_tau_y": 0.05, "training.mmd_x_weight": 1.0, "training.mmd_y_weight": 2.0,
         "training.realism_z_trunc": 3.0, "training.w1_y_norm": "rms", "training.w1_y_weight": 0.05}
    H = {"training.ks_grid_points_y": 32, "training.ks_tau_y": 0.05, "training.ks_y_weight": 0.0,
         "training.mmd_x_weight": 0.5, "training.mmd_y_weight": 1.5, "training.realism_z_trunc": 3.0,
         "training.use_ks_y": False, "training.w1_y_weight": 0.05}
    I = {"training.ks_grid_points_x": 64, "training.ks_grid_points_y": 32, "training.ks_tau_y": 0.05,
         "training.ks_x_weight": 0.008, "training.ks_y_weight": 0.0, "training.mmd_x_weight": 0.15,
         "training.mmd_y_weight": 0.5, "training.realism_z_trunc": 3.0, "training.use_ks_x": True,
         "training.use_ks_y": False, "training.w1_y_weight": 0.05}
    J = {"training.ks_tau_y": 0.05, "training.mmd_y_weight": 1.5, "training.realism_z_trunc": 3.0,
         "training.w1_x_norm": "rms", "training.w1_y_weight": 0.05}

    GROUPS = {
        "A": (A, [
            {"training.learning_rate": 1e-4, "training.use_mmd_xy": True, "training.mmd_xy_weight": 0.02},
            {"training.w1_y_weight": 0.02, "training.use_mmd_xy": True, "training.mmd_xy_weight": 0.03},
            {"training.w1_y_weight": 0.03, "training.ks_y_weight": 0.005},
            {"training.realism_ramp_epochs": 10, "training.use_mmd_xy": True, "training.mmd_xy_weight": 0.05},
            {"training.class_weighting": "prior", "training.realism_ramp_epochs": 12},
            {"training.use_ks_x": True, "training.ks_x_weight": 0.006, "training.ks_grid_points_x": 64},
            {"training.w1_x_norm": "rms", "training.w1_y_weight": 0.03},
            {"training.realism_z_trunc": 3.0, "training.w1_y_weight": 0.03},
        ]),
        "B": (B, [
            {"training.use_mmd_xy": True, "training.mmd_xy_weight": 0.02},
            {"training.learning_rate": 1e-4, "training.realism_ramp_epochs": 8},
            {"training.w1_y_weight": 0.03},
            {"training.ks_y_weight": 0.0, "training.use_ks_y": False},
            {"training.class_weighting": "inverse", "training.use_mmd_xy": True, "training.mmd_xy_weight": 0.05},
            {"training.ks_tau_y": 0.04},
            {"training.w1_y_norm": "rms"},
            {"training.realism_z_trunc": 3.0},
        ]),
        "C": (C, [
            {"training.learning_rate": 1e-4},
            {"training.w1_y_weight": 0.03},
            {"training.use_mmd_xy": True, "training.mmd_xy_weight": 0.03, "training.mmd_xy_scales": [1.0, 2.0, 4.0]},
            {"training.class_weighting": "prior", "training.realism_ramp_epochs": 12},
            {"training.ks_tau_y": 0.03},
            {"training.ks_y_weight": 0.03},
            {"training.use_ks_x": True, "training.ks_x_weight": 0.004, "training.ks_grid_points_x": 48},
            {"training.w1_x_norm": "rms"},
        ]),
        "D": (D, [
            {"training.learning_rate": 1e-4},
            {"training.use_mmd_xy": True, "training.mmd_xy_weight": 0.02},
            {"training.realism_ramp_epochs": 15},
            {"training.w1_y_weight": 0.03},
            {"training.class_weighting": "prior", "training.use_mmd_xy": True, "training.mmd_xy_weight": 0.05},
            {"training.class_weighting": "inverse", "training.learning_rate": 5e-4},
            {"training.ks_tau_y": 0.03},
            {"training.use_ks_x": True, "training.ks_x_weight": 0.008, "training.ks_grid_points_x": 64},
        ]),
        "E": (E, [
            {"training.learning_rate": 1e-4},
            {"training.w1_y_weight": 0.03},
            {"training.use_mmd_xy": True, "training.mmd_xy_weight": 0.02},
            {"training.ks_y_weight": 0.015},
            {"training.ks_tau_y": 0.04},
            {"training.class_weighting": "prior", "training.realism_ramp_epochs": 10},
            {"training.ks_y_weight": 0.0, "training.use_ks_y": False},
            {"training.w1_y_norm": "rms", "training.w1_y_weight": 0.02},
        ]),
        "F": (F, [
            {"training.learning_rate": 1e-4},
            {"training.ks_grid_points_y": 80, "training.ks_tau_y": 0.04},
            {"training.ks_y_weight": 0.0, "training.use_ks_y": False},
            {"training.w1_y_weight": 0.03},
            {"training.use_mmd_xy": True, "training.mmd_xy_weight": 0.05},
            {"training.class_weighting": "prior", "training.realism_ramp_epochs": 12},
            {"training.w1_y_norm": "rms", "training.w1_y_weight": 0.02},
            {"training.ks_tau_y": 0.03, "training.ks_grid_points_y": 64},
        ]),
        "G": (G, [
            {"training.learning_rate": 1e-4},
            {"training.w1_y_weight": 0.03},
            {"training.mmd_y_weight": 1.75},
            {"training.mmd_x_weight": 0.8},
            {"training.ks_y_weight": 0.0, "training.use_ks_y": False},
            {"training.use_mmd_xy": True, "training.mmd_xy_weight": 0.02},
            {"training.class_weighting": "prior", "training.realism_ramp_epochs": 15},
            {"training.ks_tau_y": 0.04},
        ]),
        "H": (H, [
            {"training.learning_rate": 1e-4},
            {"training.use_ks_y": True, "training.ks_y_weight": 0.005},
            {"training.w1_y_weight": 0.03},
            {"training.use_mmd_xy": True, "training.mmd_xy_weight": 0.03},
            {"training.class_weighting": "prior", "training.realism_ramp_epochs": 12},
            {"training.ks_tau_y": 0.03},
            {"training.w1_x_norm": "rms"},
            {"training.mmd_x_weight": 0.6},
        ]),
        "I": (I, [
            {"training.learning_rate": 1e-4},
            {"training.w1_y_weight": 0.03},
            {"training.use_ks_y": True, "training.ks_y_weight": 0.005},
            {"training.use_mmd_xy": True, "training.mmd_xy_weight": 0.05},
            {"training.class_weighting": "prior", "training.realism_ramp_epochs": 10},
            {"training.ks_x_weight": 0.006, "training.ks_grid_points_x": 48},
            {"training.mmd_x_weight": 0.2, "training.mmd_y_weight": 0.6},
            {"training.ks_tau_y": 0.03},
        ]),
        "J": (J, [
            {"training.learning_rate": 1e-4},
            {"training.w1_y_weight": 0.03},
            {"training.use_mmd_xy": True, "training.mmd_xy_weight": 0.02},
            {"training.ks_y_weight": 0.0, "training.use_ks_y": False},
            {"training.class_weighting": "inverse", "training.realism_ramp_epochs": 15},
            {"training.ks_tau_y": 0.04},
            {"training.w1_y_norm": "rms", "training.w1_y_weight": 0.02},
            {"training.realism_z_trunc": 3.5, "training.w1_y_weight": 0.03},
        ]),
    }

    # Build specs
    for group_key, (anchor, variants) in GROUPS.items():
        for idx, extra in enumerate(variants, start=1):
            ov = {**anchor, **extra}
            # keep ramp cap
            if "training.realism_ramp_epochs" in ov:
                ov["training.realism_ramp_epochs"] = min(15, int(ov["training.realism_ramp_epochs"]))
            name_tag = make_name_tag(f"{group_key}{idx:02d}", ov)
            specs.append({"name_tag": name_tag, "overrides": {**common, **ov}})

    assert len(specs) == 80, f"Expected 80 specs, got {len(specs)}"
    return specs

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="FlowGen finetune-only sweep (80 combos).")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--base_config", type=str, default="flowgen.yaml",
                    help="YAML filename under ROOT_PATH/config to clone per run.")
    ap.add_argument("--pretrained_path", type=str, default="",
                    help="If omitted or placeholder-like, uses DEFAULT_PRETRAINED.")
    ap.add_argument("--finetune_epochs_list", type=str, default="30",
                    help="Comma-separated list of FT epochs to try, e.g. '30' or '25,30'.")
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
        specs = build_80_specs(finetune_epochs=fe)

        def make_base_name(tag: str) -> str:
            prefix = f"FT80_{fe}_"
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

    print(f"🧪 Planned total: {len(ft_buckets)}×80 = {len(ft_buckets)*80}")
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

                fe_match = re.search(r"FT80_(\d+)_", base_name)
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
    summary_csv = SWEEP_ROOT / f"sweep_flowgen_finetune_v2_{ts}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    total_elapsed = time.time() - t_start
    ok_ct = sum(1 for r in summary_rows if r["status"] == "ok")
    fail_ct = sum(1 for r in summary_rows if r["status"] != "ok")
    print(f"\n✅ Sweep finished. OK: {ok_ct}  |  Failed: {fail_ct}")
    print(f"📝 Summary CSV: {summary_csv}")
    print(f"⏲️ Total elapsed: {fmt_hms(total_elapsed)}")


if __name__ == "__main__":
    main()
