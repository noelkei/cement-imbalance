#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run 391 CVAE-CNF trainings.

Key fixes vs previous version:
- Temp YAMLs are written directly under ROOT_PATH/config with a safe unique filename
  (no directories), because the trainer's loader likely disallows subfolders.
- We pass ONLY the filename to train_cvae_cnf_model(config_filename=...).
- We verify the file exists before training and always delete it afterwards.
- Per-run timer + running mean timer.

Usage:
  python training/sweep_cvae_cnf_391.py --dataset df_scaled_x_flowpre_fair_ystandard --device cuda --seed 4321
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
    The trainer likely expects just a filename (no directories).
    """
    safe_stem = _sanitize_name(stem)
    fname = f"{safe_stem}.yaml"
    full_path = CONFIG_DIR / fname
    with open(full_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)

    # verify immediate presence on disk
    if not full_path.exists():
        raise FileNotFoundError(f"Temp config was not created: {full_path}")

    # The trainer should accept just the filename (no subdir)
    return fname, full_path


# -------------------------
# Sweep definition (391 runs)
# -------------------------
def build_run_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    # ---------- Batch A (256 runs) ----------
    flow_freeze_epochs = [0, 5, 10, 15]
    mse_warmup_epochs = [0, 8, 12, 16]
    reset_on_nll = [False, True]
    intercept_es = [False, True]
    warmup_sample = [False, True]
    reset_on_unfreeze = [False, True]

    for ff, mse, rst_nll, ies, sw, rst_unf in itertools.product(
        flow_freeze_epochs, mse_warmup_epochs, reset_on_nll, intercept_es, warmup_sample, reset_on_unfreeze
    ):
        ov = {
            "training.schedules.enable_flow_freeze": bool(ff > 0),
            "training.schedules.flow_freeze_epochs": int(ff),
            "training.schedules.enable_mse_warmup": bool(mse > 0),
            "training.schedules.mse_warmup_epochs": int(mse),
            "training.schedules.reset_scheduler_on_nll_start": bool(rst_nll),
            "training.schedules.reset_scheduler_on_unfreeze": bool(rst_unf),
            "training.schedules.intercept_early_stop_during_warmup": bool(ies),
            "training.schedules.enable_sample_posterior_after_warmup": True,
            "loss.sample_posterior": bool(sw),
        }
        tag = f"A_f{ff}_m{mse}_rn{int(rst_nll)}_ru{int(rst_unf)}_ies{int(ies)}_sW{int(sw)}"
        specs.append({"batch": "A", "name_tag": tag, "overrides": ov})

    # ---------- Batch B (48 runs) ----------
    n_repeat_blocks = [4, 6, 8]
    num_bins = [6, 10]
    flow_hidden = [128, 192]
    final_rq_layers = [1, 2]
    lulinear_finisher = [True, False]
    for nb, bins, hf, finl, luf in itertools.product(
        n_repeat_blocks, num_bins, flow_hidden, final_rq_layers, lulinear_finisher
    ):
        ov = {
            "flow.n_repeat_blocks": int(nb),
            "flow.num_bins": int(bins),
            "flow.hidden_features": int(hf),
            "flow.final_rq_layers": int(finl),
            "flow.lulinear_finisher": bool(luf),
        }
        tag = f"B_rb{nb}_bins{bins}_fh{hf}_frq{finl}_luf{int(luf)}"
        specs.append({"batch": "B", "name_tag": tag, "overrides": ov})

    # ---------- Batch C (36 runs) ----------
    clamp_ranges = [[-3, 1], [-3, 2], [-4, 2]]
    dec_targets = [-0.7, -1.0]
    reg_weights = [0.06, 0.08, 0.12]
    until_epochs = [100, 150]
    for cr, tgt, rw, ue in itertools.product(clamp_ranges, dec_targets, reg_weights, until_epochs):
        ov = {
            "loss.clamp_dec_logstd_range": [float(cr[0]), float(cr[1])],
            "loss.dec_logstd_target": float(tgt),
            "loss.dec_logstd_reg_weight": float(rw),
            "loss.dec_logstd_reg_until_epoch": int(ue),
        }
        tag = f"C_clmp{cr[0]}..{cr[1]}_t{tgt}_rw{rw}_u{ue}"
        specs.append({"batch": "C", "name_tag": tag, "overrides": ov})

    # ---------- Batch D (27 runs) ----------
    kl_end_epoch = [120, 180, 240]
    kl_weight = [0.10, 0.15, 0.20]
    free_bits = [0.00, 0.01, 0.02]
    for ke, kw, fb in itertools.product(kl_end_epoch, kl_weight, free_bits):
        ov = {
            "training.schedules.kl_end_epoch": int(ke),
            "loss.kl_weight": float(kw),
            "loss.free_bits_per_dim": float(fb),
        }
        tag = f"D_klend{ke}_klw{kw}_fb{fb}"
        specs.append({"batch": "D", "name_tag": tag, "overrides": ov})

    # ---------- Batch E (24 runs) ----------
    base_lr = [0.0009, 0.0012, 0.0015]
    batch_sizes = [128, 256]
    lr_patience = [8, 12]
    es_patience = [30, 45]
    for lr, bs, lrp, esp in itertools.product(base_lr, batch_sizes, lr_patience, es_patience):
        ov = {
            "training.schedules.enable_flow_freeze": True,
            "training.schedules.flow_freeze_epochs": 10,
            "training.schedules.enable_mse_warmup": True,
            "training.schedules.mse_warmup_epochs": 12,
            "training.schedules.reset_scheduler_on_nll_start": False,
            "training.schedules.reset_scheduler_on_unfreeze": False,
            "training.schedules.intercept_early_stop_during_warmup": True,
            "training.schedules.enable_sample_posterior_after_warmup": True,
            "loss.sample_posterior": False,
            "training.learning_rate": float(lr),
            "training.batch_size": int(bs),
            "training.lr_decay_patience": int(lrp),
            "training.early_stopping_patience": int(esp),
        }
        tag = f"E_lr{lr}_bs{bs}_lrp{lrp}_esp{esp}"
        specs.append({"batch": "E", "name_tag": tag, "overrides": ov})

    assert len(specs) == 391, f"Expected 391 specs, got {len(specs)}"
    return specs


# -------------------------
# Run utilities
# -------------------------
def find_latest_run_dir(base_name: str) -> Path | None:
    root = Path(ROOT_PATH) / "outputs" / "models" / "cvae_cnf"
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(base_name)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def read_results_yaml_if_any(run_dir: Path) -> Dict[str, Any] | None:
    if run_dir is None:
        return None
    run_name = run_dir.name
    yml = run_dir / f"{run_name}_results.yaml"
    if yml.exists():
        try:
            with open(yml, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return None
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
    ap = argparse.ArgumentParser(description="Run 391 CVAE-CNF trainings.")
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
    args = ap.parse_args()

    dataset = args.dataset
    device = args.device
    seed = args.seed
    save_results = not args.no_save_results

    # Load data once
    print(f"📦 Loading dataset: {dataset}")
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = load_scaled_dataset_or_fallback(dataset)

    # Load base config once
    print(f"⚙️  Loading base config: {args.base_config}")
    base_cfg = load_yaml_config(args.base_config)

    # Set global trainer flags that the sweep shouldn't change
    if args.override_epochs is not None:
        deep_set(base_cfg, "training.num_epochs", int(args.override_epochs))
    deep_set(base_cfg, "training.save_model", bool(args.save_model))
    deep_set(base_cfg, "training.save_states", bool(args.save_states))
    deep_set(base_cfg, "training.save_results", bool(save_results))

    # Build run specs
    run_specs = build_run_specs()
    total = len(run_specs)
    print(f"🧪 Prepared {total} run specs.")

    summary_rows: List[Dict[str, Any]] = []
    per_run_durations: List[float] = []
    sweep_start = time.time()

    try:
        for idx, spec in enumerate(run_specs, start=1):
            batch_tag = spec["batch"]
            name_tag = spec["name_tag"]
            overrides = spec["overrides"]

            # Build final config for this run
            run_cfg = apply_overrides(base_cfg, overrides)

            # Names
            run_base = f"cvae_{dataset}_{batch_tag}_{name_tag}_seed{seed}"
            tmp_stem = f"{batch_tag}_{name_tag}_seed{seed}"

            print(f"\n[{idx:03d}/{total}] 🚀 {run_base}")

            # Write temp YAML directly under ROOT_PATH/config and pass ONLY filename
            cfg_filename_for_trainer, full_tmp_path = write_temp_config_in_config_dir(run_cfg, tmp_stem)
            # Optional sanity print (uncomment if needed)
            # print(f"   → temp cfg: {full_tmp_path}")

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


