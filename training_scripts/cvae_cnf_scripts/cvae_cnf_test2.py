#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Second tanda: run 132 CVAE-CNF trainings focused on the highest-impact knobs.

What this does
--------------
- Writes *temporary* YAMLs directly into ROOT_PATH/config (filename only),
  passes just that filename to your trainer, then deletes the temp file.
- Measures per-run time and prints a running mean.
- Names every run with a user-provided prefix (default: "T2") so outputs are
  clearly separated from the first tanda.
- **Skips** runs that already finished earlier (existing dir starting with the
  run name AND containing a *_results.yaml).

Plan (counts)
-------------
Core grid (Batch F):                                    3 * 3 * 2 * 2 * 3 = 108
  - encoder.latent_dim                ∈ {32, 44, 64}
  - training.batch_size               ∈ {96, 128, 160}
  - schedules.mse_warmup_epochs       ∈ {0, 16}   (enable flag toggled accordingly)
  - schedules.kl_end_epoch            ∈ {240, 360}
  - loss.dec_logstd_reg_weight        ∈ {0.10, 0.12, 0.15}

Focus belt (Batch G):                                                   = 24
  Part A (18): bs=128, reg_w=0.12, warmup∈{16,20}, kl_end∈{240,300,360}, z∈{32,44,64}
  Part B ( 6): bs∈{96,160}, reg_w=0.12, warmup=16, kl_end=300,          z∈{32,44,64}

Total = 108 + 24 = 132.

Usage
-----
python training/sweep_cvae_cnf_t2_132.py \
  --dataset df_scaled_x_flowpre_fair_ystandard \
  --device cuda --seed 4321 \
  --run_prefix T2
"""

from __future__ import annotations

import argparse
import copy
import itertools
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

# --- project imports ---
from training.utils import ROOT_PATH, load_yaml_config
from training.train_cvae_cnf import train_cvae_cnf_model

# dataset helpers
from data.sets import (
    load_flowpre_scaled_set,
    load_or_create_scaled_sets,
    load_or_create_raw_splits,
)

CONFIG_DIR = Path(ROOT_PATH) / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Dataset loader utilities
# -------------------------
def _parse_flowpre_dataset_tag(tag: str) -> Tuple[str | None, str | None]:
    m = re.fullmatch(
        r"df_scaled_x_flowpre_(rrmse|mvn|fair)_y(standard|quantile|minmax)",
        tag.strip().lower(),
    )
    return (m.group(1), m.group(2)) if m else (None, None)


def load_scaled_dataset_or_fallback(dataset_name: str):
    x_variant, y_scaler = _parse_flowpre_dataset_tag(dataset_name)
    if x_variant is not None:
        print(f"📦 Detected FlowPre set → x_variant={x_variant}, y_scaler={y_scaler}")
        # Xtr, Xva, Xte, ytr, yva, yte, rtr, rva, rte, x_scaler_obj, y_scaler_obj
        return load_flowpre_scaled_set(x_variant, y_scaler)

    m = re.fullmatch(
        r"df_scaled_x(standard|robust|quantile)_y(standard|quantile|minmax)",
        dataset_name.strip().lower(),
    )
    if m:
        x_scaler_type, y_scaler_type = m.group(1), m.group(2)
        Xtr, Xva, Xte, ytr, yva, yte, rtr, rva, rte, *_ = load_or_create_scaled_sets(
            raw_df_name="df_input",
            scaled_df_name="df_scaled",
            condition_col="type",
            val_size=150,
            test_size=100,
            target="init",
            force=False,
            verbose=True,
            x_scaler_type=x_scaler_type,
            y_scaler_type=y_scaler_type,
            exclude_cols=["post_cleaning_index", "type"],
        )
        return Xtr, Xva, Xte, ytr, yva, yte, rtr, rva, rte

    print("ℹ️ Unknown dataset tag; falling back to raw splits.")
    return load_or_create_raw_splits(
        df_name="df_input",
        condition_col="type",
        val_size=150,
        test_size=100,
        target="init",
        verbose=True,
        force=False,
    )

# -------------------------
# Config helpers
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
    """Safe filename: [A-Za-z0-9_.-] only; collapse others to '_'."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def write_temp_config_in_config_dir(cfg_dict: dict, stem: str) -> tuple[str, Path]:
    """
    Write YAML directly into ROOT_PATH/config with a unique safe filename,
    return (filename_for_trainer, full_path).
    The trainer expects just a filename (no directories).
    """
    safe_stem = _sanitize_name(stem)
    fname = f"{safe_stem}.yaml"
    full_path = CONFIG_DIR / fname
    with open(full_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    if not full_path.exists():
        raise FileNotFoundError(f"Temp config was not created: {full_path}")
    return fname, full_path

# -------------------------
# Sweep definition (132 runs)
# -------------------------
def build_run_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    # ---------- Batch F (Core grid) → 108 runs ----------
    z_list   = [32, 44, 64]
    bs_list  = [96, 128, 160]
    mw_list  = [0, 16]             # mse warmup epochs
    kl_list  = [240, 360]
    rw_list  = [0.10, 0.12, 0.15]  # dec_logstd_reg_weight

    for z, bs, mw, kl, rw in itertools.product(z_list, bs_list, mw_list, kl_list, rw_list):
        ov = {
            "encoder.latent_dim": int(z),
            "training.batch_size": int(bs),
            "training.schedules.mse_warmup_epochs": int(mw),
            "training.schedules.enable_mse_warmup": bool(mw > 0),
            "training.schedules.kl_end_epoch": int(kl),
            "loss.dec_logstd_reg_weight": float(rw),
        }
        tag = f"F_z{z}_bs{bs}_m{mw}_kl{kl}_rw{rw}"
        specs.append({"batch": "F", "name_tag": tag, "overrides": ov})

    # ---------- Batch G (Focus belt) → 24 runs ----------
    # Part A: 18 runs (bs=128, rw=0.12, mw∈{16,20}, kl∈{240,300,360}, z∈{32,44,64})
    for z, mw, kl in itertools.product([32, 44, 64], [16, 20], [240, 300, 360]):
        ov = {
            "encoder.latent_dim": int(z),
            "training.batch_size": 128,
            "training.schedules.mse_warmup_epochs": int(mw),
            "training.schedules.enable_mse_warmup": True,
            "training.schedules.kl_end_epoch": int(kl),
            "loss.dec_logstd_reg_weight": 0.12,
        }
        tag = f"G_A_z{z}_bs128_m{mw}_kl{kl}_rw0.12"
        specs.append({"batch": "G", "name_tag": tag, "overrides": ov})

    # Part B: 6 runs (bs∈{96,160}, rw=0.12, mw=16, kl=300, z∈{32,44,64})
    for z, bs in itertools.product([32, 44, 64], [96, 160]):
        ov = {
            "encoder.latent_dim": int(z),
            "training.batch_size": int(bs),
            "training.schedules.mse_warmup_epochs": 16,
            "training.schedules.enable_mse_warmup": True,
            "training.schedules.kl_end_epoch": 300,
            "loss.dec_logstd_reg_weight": 0.12,
        }
        tag = f"G_B_z{z}_bs{bs}_m16_kl300_rw0.12"
        specs.append({"batch": "G", "name_tag": tag, "overrides": ov})

    # Sanity checks
    nF = 108
    nG = 24
    assert len([s for s in specs if s["batch"] == "F"]) == nF, f"Expected {nF} F-runs"
    assert len([s for s in specs if s["batch"] == "G"]) == nG, f"Expected {nG} G-runs"
    assert len(specs) == 132, f"Expected 132 specs, got {len(specs)}"
    return specs

# -------------------------
# Run utilities
# -------------------------
def outputs_root() -> Path:
    return Path(ROOT_PATH) / "outputs" / "models" / "cvae_cnf"

def find_latest_run_dir(base_name: str) -> Path | None:
    root = outputs_root()
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(base_name)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def run_already_finished(base_name: str) -> bool:
    """
    A run is considered 'finished' if there exists a directory whose name
    starts with base_name AND that directory contains any file matching '*_results.yaml'.
    """
    root = outputs_root()
    if not root.exists():
        return False
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith(base_name):
            continue
        # treat presence of a results YAML as 'finished'
        has_results = any(fp.name.endswith("_results.yaml") for fp in d.iterdir() if fp.is_file())
        if has_results:
            return True
    return False

def read_results_yaml_if_any(run_dir: Path) -> Dict[str, Any] | None:
    if run_dir is None:
        return None
    # pick any *_results.yaml in the dir
    yml_files = [p for p in run_dir.iterdir() if p.is_file() and p.name.endswith("_results.yaml")]
    if not yml_files:
        return None
    # prefer the newest
    yml_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    try:
        with open(yml_files[0], "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None

def fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}h {m:02d}m {s:02d}s"

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Second tanda: run 132 CVAE-CNF trainings (skips finished runs).")
    ap.add_argument("--dataset", type=str, required=True,
                    help="e.g. df_scaled_x_flowpre_fair_ystandard")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--base_config", type=str, default="cvae_cnf.yaml",
                    help="Base YAML filename under ROOT_PATH/config/")
    ap.add_argument("--override_epochs", type=int, default=None,
                    help="Optional global override of num_epochs.")
    ap.add_argument("--save_model", action="store_true", default=False)
    ap.add_argument("--save_states", action="store_true", default=False)
    ap.add_argument("--no_save_results", action="store_true", default=False)
    ap.add_argument("--run_prefix", type=str, default="T2",
                    help="Prefix added to each run's base_name (e.g., 'T2').")
    args = ap.parse_args()

    dataset = args.dataset
    device = args.device
    seed = args.seed
    run_prefix = (args.run_prefix or "").strip()
    save_results = not args.no_save_results

    # Load data once
    print(f"📦 Loading dataset: {dataset}")
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = load_scaled_dataset_or_fallback(dataset)

    # Load base config once
    print(f"⚙️  Loading base config: {args.base_config}")
    base_cfg = load_yaml_config(args.base_config)

    # Enforce flow active for this tanda (optional safety)
    if "flow" in base_cfg:
        base_cfg["flow"]["active"] = True

    # Global trainer flags that the sweep shouldn't change
    if args.override_epochs is not None:
        deep_set(base_cfg, "training.num_epochs", int(args.override_epochs))
    # If no override, keep whatever is in the YAML (now 500)
    deep_set(base_cfg, "training.save_model", bool(args.save_model))
    deep_set(base_cfg, "training.save_states", bool(args.save_states))
    deep_set(base_cfg, "training.save_results", bool(save_results))

    # Build run specs
    all_specs = build_run_specs()

    # Pre-compute run_base names to filter finished ones
    def make_run_base(spec: Dict[str, Any]) -> str:
        batch_tag = spec["batch"]
        name_tag = spec["name_tag"]
        prefix = f"{run_prefix}_" if run_prefix else ""
        return f"{prefix}cvae_{dataset}_{batch_tag}_{name_tag}_seed{seed}"

    planned = [(spec, make_run_base(spec)) for spec in all_specs]

    # Skip runs that already finished earlier
    skipped = []
    todo = []
    for spec, run_base in planned:
        if run_already_finished(run_base):
            skipped.append(run_base)
        else:
            todo.append((spec, run_base))

    print(f"🧪 Planned specs: {len(all_specs)}  |  🟢 To run: {len(todo)}  |  ⏭️ Skipped (finished): {len(skipped)}")
    if skipped:
        # Only print a few to keep logs tidy
        preview = "\n".join(f"   - {rb}" for rb in skipped[:10])
        more = "" if len(skipped) <= 10 else f"\n   ... and {len(skipped)-10} more"
        print(f"⏭️ Already finished:\n{preview}{more}")

    summary_rows: List[Dict[str, Any]] = []
    per_run_durations: List[float] = []
    sweep_start = time.time()

    try:
        for idx, (spec, run_base) in enumerate(todo, start=1):
            batch_tag = spec["batch"]
            name_tag = spec["name_tag"]
            overrides = spec["overrides"]

            # Build final config for this run
            run_cfg = apply_overrides(base_cfg, overrides)

            # Temp YAML stem (doesn't need the prefix)
            tmp_stem = f"{batch_tag}_{name_tag}_seed{seed}"

            print(f"\n[{idx:03d}/{len(todo)}] 🚀 {run_base}")

            # Safety: if it got finished after the earlier filter (e.g., another process),
            # re-check and skip.
            if run_already_finished(run_base):
                print(f"⏭️  Detected as finished just now. Skipping: {run_base}")
                status = "skipped_finished"
                error_msg = ""
                dt = 0.0
                per_run_durations.append(dt)
                avg_dt = sum(per_run_durations) / max(1, len(per_run_durations))
                print(f"⏱️  Run time: {fmt_hms(dt)}  |  Mean so far: {fmt_hms(avg_dt)}")
                summary_rows.append({
                    "dataset": dataset, "base_name": run_base,
                    "batch": batch_tag, "name_tag": name_tag,
                    "status": status, "error": error_msg,
                    "run_dir": "", "duration_sec": round(dt, 2),
                })
                continue

            # Write temp YAML directly under ROOT_PATH/config and pass ONLY filename
            cfg_filename_for_trainer, full_tmp_path = write_temp_config_in_config_dir(run_cfg, tmp_stem)

            # Time this run
            t0 = time.time()
            status = "ok"
            error_msg = ""
            try:
                _ = train_cvae_cnf_model(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val,     y_val=y_val,
                    X_test=X_test,   y_test=y_test,
                    condition_col="type",
                    index_col="post_cleaning_index",
                    context_cols=["type"],
                    config_filename=cfg_filename_for_trainer,  # filename only; no subdir
                    base_name=run_base,
                    device=device,
                    verbose=False,
                    seed=seed,
                )
            except Exception as e:
                status = "failed"
                error_msg = str(e)
                print(f"❌ Run failed: {run_base} — {error_msg}")
            finally:
                # Always remove temp config
                try:
                    if full_tmp_path.exists():
                        full_tmp_path.unlink()
                except Exception:
                    pass

            # Duration
            dt = time.time() - t0
            per_run_durations.append(dt)
            avg_dt = sum(per_run_durations) / len(per_run_durations)
            print(f"⏱️  Run time: {fmt_hms(dt)}  |  Mean so far: {fmt_hms(avg_dt)}")

            # Results
            run_dir = find_latest_run_dir(run_base) if status == "ok" else None
            res = read_results_yaml_if_any(run_dir) if status == "ok" else None

            row = {
                "dataset": dataset,
                "base_name": run_base,
                "batch": batch_tag,
                "name_tag": name_tag,
                "status": status,
                "error": error_msg,
                "run_dir": str(run_dir) if run_dir else "",
                "duration_sec": round(dt, 2),
            }
            if res:
                row.update({
                    "best_epoch": res.get("best_epoch", None),
                    "train_rrmse": res.get("train", {}).get("recon_rrmse", None),
                    "train_r2":    res.get("train", {}).get("r2", None),
                    "val_rrmse":   res.get("val", {}).get("recon_rrmse", None),
                    "val_r2":      res.get("val", {}).get("r2", None),
                    "test_rrmse":  res.get("test", {}).get("recon_rrmse", None),
                    "test_r2":     res.get("test", {}).get("r2", None),
                })
            summary_rows.append(row)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted — writing partial summary...")

    # Save sweep summary
    out_dir = Path(ROOT_PATH) / "outputs" / "sweeps" / "cvae_cnf"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"sweep_{dataset}_{ts}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    total_elapsed = time.time() - sweep_start
    print(f"\n✅ Sweep complete — attempted {len(summary_rows)} runs.")
    print(f"📝 Summary CSV: {summary_path}")
    print(f"⏲️  Total elapsed: {fmt_hms(total_elapsed)}")

    # Optional leaderboard
    try:
        df = pd.DataFrame(summary_rows)
        df_ok = df[df["status"] == "ok"].copy()
        if "val_rrmse" in df_ok.columns:
            df_ok = df_ok.sort_values("val_rrmse", ascending=True)
            topk = df_ok.head(12)[[
                "base_name", "batch", "name_tag",
                "val_rrmse", "val_r2", "duration_sec"
            ]]
            print("\n🏆 Top 12 by lowest val_rrmse:")
            print(topk.to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()

