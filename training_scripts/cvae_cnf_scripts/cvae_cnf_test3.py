#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
T3 (60 runs): smaller batches, larger latents, short MSE warmups, with three realism tiers.
Validation uses your trainer; this script only builds configs.

Grid
----
Core combos (18): latent ∈ {56,64,72} × batch ∈ {64,80,96} × mse_warmup ∈ {0,8}
For each core combo → 3 variants = 54 total:
  - baseline (realism OFF)
  - realism BOTH (balanced):   MMD=0.04, Corr=0.02, Mom=0.01, ramp [mw+1, mw+20], KL=0.10, var_reg=0.08
  - realism BOTH (squeezed):   MMD=0.06, Corr=0.025, Mom=0.015, ramp [mw+1, mw+25], KL=0.08, var_reg=0.06

Explorers (6): latent=72; batch ∈ {64,80,96}; mse_warmup=24; kl_end=360
  - realism BOTH (strong++):   MMD=0.08, Corr=0.03,  Mom=0.02,  ramp [25, 55], KL=0.08, var_reg=0.06, flow.n_repeat_blocks=5

Total planned = 60. Finished runs are auto-skipped.
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
        return load_flowpre_scaled_set(x_variant, y_scaler)

    m = re.fullmatch(
        r"df_scaled_x(standard|robust|quantile)_y(standard|quantile|minmax)",
        dataset_name.strip().lower(),
    )
    if m:
        x_scaler_type, y_scaler_type = m.group(1), m.group(2)
        Xtr, Xva, Xte, ytr, yva, yte, *_ = load_or_create_scaled_sets(
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
        return Xtr, Xva, Xte, ytr, yva, yte

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
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def write_temp_config_in_config_dir(cfg_dict: dict, stem: str) -> tuple[str, Path]:
    safe_stem = _sanitize_name(stem)
    fname = f"{safe_stem}.yaml"
    full_path = CONFIG_DIR / fname
    with open(full_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    if not full_path.exists():
        raise FileNotFoundError(f"Temp config was not created: {full_path}")
    return fname, full_path

# -------------------------
# Sweep definition (60 runs)
# -------------------------
def _baseline_block() -> Dict[str, Any]:
    return {
        "loss.enable_realism": False,
        "loss.realism_weight_mmd": 0.0,
        "loss.realism_weight_corr": 0.0,
        "loss.realism_weight_mom": 0.0,
        "loss.realism_ramp_start": 0,
        "loss.realism_ramp_end": 0,
        # keep base KL + var-reg
    }

def _realism_block(mw: int, *, w_mmd: float, w_corr: float, w_mom: float,
                   ramp_extra: int, kl_weight: float, var_reg: float) -> Dict[str, Any]:
    rs, re = (mw + 1, mw + ramp_extra)
    return {
        "loss.enable_realism": True,
        "loss.realism_mode": "both",          # match recon & prior
        "loss.realism_weight_mmd": float(w_mmd),
        "loss.realism_weight_corr": float(w_corr),
        "loss.realism_weight_mom": float(w_mom),
        "loss.realism_ramp_start": int(rs),
        "loss.realism_ramp_end": int(re),
        "loss.realism_use_per_class": True,
        "loss.realism_sigma": None,           # median heuristic
        # balance against realism
        "loss.kl_weight": float(kl_weight),
        "loss.dec_logstd_reg_weight": float(var_reg),
    }

def build_run_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    # Core: 3 (z) * 3 (bs) * 2 (mw) = 18 combos
    z_list  = [56, 64, 72]
    bs_list = [64, 80, 96]
    mw_list = [0, 8]  # we push mw=24 to explorers

    for z, bs, mw in itertools.product(z_list, bs_list, mw_list):
        common = {
            "encoder.latent_dim": int(z),
            "training.batch_size": int(bs),
            "training.schedules.mse_warmup_epochs": int(mw),
            "training.schedules.enable_mse_warmup": bool(mw > 0),
        }

        # 1) Baseline (OFF)
        ov_base = {**common, **_baseline_block()}
        specs.append({
            "group": "Core",
            "name_tag": f"Core_z{z}_bs{bs}_m{mw}_base",
            "overrides": ov_base
        })

        # 2) Balanced realism
        ov_bal = {
            **common,
            **_realism_block(mw, w_mmd=0.04, w_corr=0.02, w_mom=0.01,
                             ramp_extra=20, kl_weight=0.10, var_reg=0.08),
        }
        specs.append({
            "group": "Core",
            "name_tag": f"Core_z{z}_bs{bs}_m{mw}_realismBalanced",
            "overrides": ov_bal
        })

        # 3) Squeezed realism
        ov_sq = {
            **common,
            **_realism_block(mw, w_mmd=0.06, w_corr=0.025, w_mom=0.015,
                             ramp_extra=25, kl_weight=0.08, var_reg=0.06),
        }
        specs.append({
            "group": "Core",
            "name_tag": f"Core_z{z}_bs{bs}_m{mw}_realismSqueezed",
            "overrides": ov_sq
        })

    # Explorers: 6 runs (mw in {0,24}; stronger realism; deeper flow; longer KL anneal)
    for bs, mw in itertools.product([64, 80, 96], [0, 24]):
        ov = {
            "encoder.latent_dim": 72,
            "training.batch_size": int(bs),
            "training.schedules.mse_warmup_epochs": int(mw),
            "training.schedules.enable_mse_warmup": bool(mw > 0),
            "training.schedules.kl_end_epoch": 360,
            "flow.n_repeat_blocks": 5,
            **_realism_block(mw, w_mmd=0.08, w_corr=0.03, w_mom=0.02,
                             ramp_extra=30, kl_weight=0.08, var_reg=0.06),
        }
        specs.append({
            "group": "Explorers",
            "name_tag": f"Expl_z72_bs{bs}_m{mw}_kl360_realismStrongPP",
            "overrides": ov
        })

    assert len(specs) == 60, f"Expected 60 specs, got {len(specs)}"
    return specs

# -------------------------
# Outputs / run guards
# -------------------------
def outputs_root() -> Path:
    return Path(ROOT_PATH) / "outputs" / "models" / "cvae_cnf"

def find_latest_run_dir(base_name: str) -> Path | None:
    root = outputs_root()
    if not root.exists():
        return None
    cands = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(base_name)]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

def run_already_finished(base_name: str) -> bool:
    root = outputs_root()
    if not root.exists():
        return False
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith(base_name):
            if any(fp.name.endswith("_results.yaml") for fp in d.iterdir() if fp.is_file()):
                return True
    return False

def read_results_yaml_if_any(run_dir: Path) -> Dict[str, Any] | None:
    if run_dir is None:
        return None
    ymls = [p for p in run_dir.iterdir() if p.is_file() and p.name.endswith("_results.yaml")]
    if not ymls:
        return None
    ymls.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    try:
        with open(ymls[0], "r", encoding="utf-8") as f:
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
    ap = argparse.ArgumentParser(description="T3: 60-run CVAE-CNF sweep with three realism tiers.")
    ap.add_argument("--dataset", type=str, required=True,
                    help="e.g. df_scaled_x_flowpre_fair_ystandard")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--base_config", type=str, default="cvae_cnf.yaml",
                    help="Base YAML filename under ROOT_PATH/config/")
    ap.add_argument("--override_epochs", type=int, default=None)   # ← fixed
    ap.add_argument("--save_model", action="store_true", default=False)
    ap.add_argument("--save_states", action="store_true", default=False)
    ap.add_argument("--no_save_results", action="store_true", default=False)
    ap.add_argument("--run_prefix", type=str, default="T3",
                    help="Prefix added to each run's base_name.")
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

    # Enforce flow active for this tanda (safety)
    if "flow" in base_cfg:
        base_cfg["flow"]["active"] = True

    # Global trainer flags
    if args.override_epochs is not None:
        deep_set(base_cfg, "training.num_epochs", int(args.override_epochs))
    deep_set(base_cfg, "training.save_model", bool(args.save_model))
    deep_set(base_cfg, "training.save_states", bool(args.save_states))
    deep_set(base_cfg, "training.save_results", bool(save_results))

    # Build run specs
    specs = build_run_specs()

    def make_run_base(spec: Dict[str, Any]) -> str:
        prefix = f"{run_prefix}_" if run_prefix else ""
        return f"{prefix}cvae_{dataset}_{spec['group']}_{spec['name_tag']}_seed{seed}"

    planned = [(spec, make_run_base(spec)) for spec in specs]

    # Skip already-finished
    todo, skipped = [], []
    for spec, base in planned:
        if run_already_finished(base):
            skipped.append(base)
        else:
            todo.append((spec, base))

    print(f"🧪 Planned: {len(specs)}  |  🟢 To run: {len(todo)}  |  ⏭️ Skipped: {len(skipped)}")
    if skipped:
        preview = "\n".join(f"   - {rb}" for rb in skipped[:10])
        more = "" if len(skipped) <= 10 else f"\n   ... and {len(skipped)-10} more"
        print(f"⏭️ Already finished:\n{preview}{more}")

    summary_rows: List[Dict[str, Any]] = []
    per_run_durations: List[float] = []
    t_start = time.time()

    try:
        for idx, (spec, run_base) in enumerate(todo, start=1):
            name_tag = spec["name_tag"]
            overrides = spec["overrides"]

            # Build final config for this run
            run_cfg = apply_overrides(base_cfg, overrides)

            # Write temp YAML under ROOT_PATH/config and pass ONLY filename
            cfg_filename, full_tmp_path = write_temp_config_in_config_dir(run_cfg, name_tag)

            print(f"\n[{idx:03d}/{len(todo)}] 🚀 {run_base}")

            # Safety re-check
            if run_already_finished(run_base):
                print(f"⏭️  Detected as finished just now. Skipping.")
                status, error_msg, dt = "skipped_finished", "", 0.0
            else:
                t0 = time.time()
                status, error_msg = "ok", ""
                try:
                    _ = train_cvae_cnf_model(
                        X_train=X_train, y_train=y_train,
                        X_val=X_val,     y_val=y_val,
                        X_test=X_test,   y_test=y_test,
                        condition_col="type",
                        index_col="post_cleaning_index",
                        context_cols=["type"],
                        config_filename=cfg_filename,  # filename only
                        base_name=run_base,
                        device=device,
                        verbose=False,
                        seed=seed,
                    )
                except Exception as e:
                    status, error_msg = "failed", str(e)
                    print(f"❌ Run failed: {run_base} — {error_msg}")
                dt = time.time() - t0

            # Clean temp YAML
            try:
                if full_tmp_path.exists():
                    full_tmp_path.unlink()
            except Exception:
                pass

            per_run_durations.append(dt)
            mean_dt = sum(per_run_durations) / max(1, len(per_run_durations))
            print(f"⏱️  Run time: {fmt_hms(dt)}  |  Mean so far: {fmt_hms(mean_dt)}")

            run_dir = find_latest_run_dir(run_base) if status == "ok" else None
            res = read_results_yaml_if_any(run_dir) if status == "ok" else None

            row = {
                "dataset": dataset, "base_name": run_base, "group": spec["group"], "name_tag": name_tag,
                "status": status, "error": error_msg,
                "run_dir": str(run_dir) if run_dir else "", "duration_sec": round(dt, 2),
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
    summary_path = out_dir / f"sweep_{dataset}_T3_{ts}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    total_elapsed = time.time() - t_start
    print(f"\n✅ Sweep complete — attempted {len(summary_rows)} runs.")
    print(f"📝 Summary CSV: {summary_path}")
    print(f"⏲️  Total elapsed: {fmt_hms(total_elapsed)}")

    # Optional leaderboard (quick sanity)
    try:
        df = pd.DataFrame(summary_rows)
        df_ok = df[df["status"] == "ok"].copy()
        if "val_rrmse" in df_ok.columns:
            df_ok = df_ok.sort_values("val_rrmse", ascending=True)
            topk = df_ok.head(12)[["base_name", "group", "name_tag", "val_rrmse", "val_r2", "duration_sec"]]
            print("\n🏆 Top 12 by lowest val_rrmse:")
            print(topk.to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
