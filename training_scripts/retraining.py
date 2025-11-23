#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Retrain FlowPre models for the union of Top-20 configs (from previous results).

Enhancements:
- Prints config source directory and destination results directory.
- Prints the union set size and the full list of selected run_idx.
- Retries once per config on failure.
- Persists progress to outputs/retrained/_completed_runs.txt and resumes automatically.

Run:
  python tools/retrain_flowpre_topk.py --seed 123456 --device auto --k 20 --limit 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import re
import shutil
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set

# keep CPU phases well-behaved by default
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---- project paths ------------------------------------------------------------
try:
    from training.utils import ROOT_PATH
except Exception:
    ROOT_PATH = Path("../training").resolve()
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

from training.train_flow_pre import train_flowpre_pipeline  # your trainer

RESULTS_DIR = Path(ROOT_PATH) / "outputs" / "flow_pre"                 # where old runs/configs are
MODELS_OUT_DIR = Path(ROOT_PATH) / "outputs" / "models" / "flow_pre"   # trainer writes here first
RETRAINED_DIR = Path(ROOT_PATH) / "outputs" / "retrained"              # final destination
RETRAINED_DIR.mkdir(parents=True, exist_ok=True)

COMPLETED_FILE = RETRAINED_DIR / "_completed_runs.txt"                 # progress log (one line per run_dir_name)

# ------------------- helpers copied from your analysis logic -------------------

LR_MAP = {"0001": 1e-3, "00001": 1e-4, "000001": 1e-5}
BOOL_MAP = {"True": True, "False": False}

def parse_run_name(stem: str) -> dict:
    parts = stem.split("_")
    run_base = parts[0] if parts else None
    run_idx, tokens = None, []
    if len(parts) > 1 and parts[1].isdigit():
        run_idx = int(parts[1])
        tokens = parts[2:]
    else:
        tokens = parts[1:]

    hp = {"run_base": run_base, "run_idx": run_idx, "name_suffix": "_".join(tokens)}
    suffix = hp["name_suffix"]

    pats = {
        "hidden_features": r"hidden_features(\d+)",
        "num_layers": r"num_layers(\d+)",
        "final_rq_layers": r"final_rq_layers(\d+)",
        "learning_rate": r"learning_rate(\d+)",
        "affine_rq_ratio": r"affine_rq_ratio([\dx]+)",
        "use_mean_penalty": r"use_mean_penalty(True|False)",
        "use_std_penalty": r"use_std_penalty(True|False)",
        "use_skew_penalty": r"use_skew_penalty(True|False)",
        "use_kurtosis_penalty": r"use_kurtosis_penalty(True|False)",
    }
    for key, pat in pats.items():
        m = re.search(pat, suffix)
        if not m:
            continue
        val = m.group(1)
        if key in {"hidden_features", "num_layers", "final_rq_layers"}:
            hp[key] = int(val)
        elif key == "learning_rate":
            hp[key] = LR_MAP.get(val, np.nan)
            hp["learning_rate_token"] = val
        elif key == "affine_rq_ratio":
            try:
                hp[key] = [int(x) for x in val.split("x")]
                hp["affine_rq_ratio_str"] = val
            except Exception:
                hp[key] = val
                hp["affine_rq_ratio_str"] = str(val)
        else:
            hp[key] = BOOL_MAP.get(val, val)

    if "affine_rq_ratio_str" not in hp and "affine_rq_ratio" in hp and isinstance(hp["affine_rq_ratio"], (list, tuple)):
        hp["affine_rq_ratio_str"] = "x".join(map(str, hp["affine_rq_ratio"]))
    return hp

def read_one_results_yaml(yaml_path: Path) -> dict | None:
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    needed = {"best_epoch", "total_epochs", "train", "val"}
    if not needed.issubset(data.keys()):
        return None
    row = {
        "file": str(yaml_path),
        "run_dir_name": yaml_path.parent.name,
        "best_epoch": data.get("best_epoch"),
        "total_epochs": data.get("total_epochs"),
        "train_rrmse_mean": (data.get("train") or {}).get("rrmse_mean"),
        "train_rrmse_std":  (data.get("train") or {}).get("rrmse_std"),
        "val_rrmse_mean":   (data.get("val") or {}).get("rrmse_mean"),
        "val_rrmse_std":    (data.get("val") or {}).get("rrmse_std"),
    }
    row["gap_rrmse_mean"] = (
        row["val_rrmse_mean"] - row["train_rrmse_mean"]
        if row["val_rrmse_mean"] is not None and row["train_rrmse_mean"] is not None
        else np.nan
    )
    row["gap_rrmse_std"] = (
        row["val_rrmse_std"] - row["train_rrmse_std"]
        if row["val_rrmse_std"] is not None and row["train_rrmse_std"] is not None
        else np.nan
    )
    row.update(parse_run_name(row["run_dir_name"]))
    return row

def collect_results(root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        picked = None
        for yml in sorted(run_dir.glob("*_results.yaml")):
            row = read_one_results_yaml(yml)
            if row is not None:
                picked = row
                break
        if picked is None:
            for yml in sorted(run_dir.rglob("*_results.yaml")):
                row = read_one_results_yaml(yml)
                if row is not None:
                    picked = row
                    break
        if picked:
            rows.append(picked)
    return pd.DataFrame(rows)

def top_union_indices(df: pd.DataFrame, k: int = 20) -> list[int]:
    for c in [
        "best_epoch","total_epochs",
        "train_rrmse_mean","train_rrmse_std",
        "val_rrmse_mean","val_rrmse_std",
        "gap_rrmse_mean","gap_rrmse_std",
        "hidden_features","num_layers","final_rq_layers","learning_rate"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["val_mean_plus_std"] = df["val_rrmse_mean"] + df["val_rrmse_std"]
    df["sum_all_rrmse"] = (
        df["train_rrmse_mean"] + df["train_rrmse_std"] +
        df["val_rrmse_mean"]   + df["val_rrmse_std"]
    )
    df["abs_gap_mean"] = (df["val_rrmse_mean"] - df["train_rrmse_mean"]).abs()
    df["train_mean_plus_std"] = df["train_rrmse_mean"] + df["train_rrmse_std"]

    tab1 = df.sort_values(["val_rrmse_mean","val_rrmse_std","gap_rrmse_mean"])
    tab2 = df.sort_values(["val_rrmse_std","val_rrmse_mean","gap_rrmse_mean"])
    tab3 = df.sort_values(["val_mean_plus_std","gap_rrmse_mean"])
    tab4 = df.sort_values(["sum_all_rrmse","gap_rrmse_mean"])
    tab5 = df.sort_values(["abs_gap_mean","val_rrmse_mean","val_rrmse_std"])
    tab6 = df.sort_values(["train_rrmse_mean","train_rrmse_std","val_rrmse_mean"])
    tab7 = df.sort_values(["train_rrmse_std","train_rrmse_mean","val_rrmse_mean"])
    tab8 = df.sort_values(["train_mean_plus_std","val_rrmse_mean"])

    sets = [
        set(tab1["run_idx"].head(k)),
        set(tab2["run_idx"].head(k)),
        set(tab3["run_idx"].head(k)),
        set(tab4["run_idx"].head(k)),
        set(tab5["run_idx"].head(k)),
        set(tab6["run_idx"].head(k)),
        set(tab7["run_idx"].head(k)),
        set(tab8["run_idx"].head(k)),
    ]
    union = set().union(*sets)
    union = {int(i) for i in union if pd.notna(i)}
    return sorted(union)

# ---------------------- config + folder helpers for retrain --------------------

def find_config_in_run_dir(run_dir: Path) -> Path:
    run_name = run_dir.name
    exact = run_dir / f"{run_name}.yaml"
    if exact.exists():
        return exact
    candidates = [p for p in run_dir.glob("*.yaml") if not p.name.endswith("_results.yaml")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        for p in candidates:
            if p.stem == run_name:
                return p
        return candidates[0]
    raise FileNotFoundError(f"No config YAML found in {run_dir} (looked for {run_name}.yaml).")

def rt_name_from_vm(run_dir_name: str) -> str:
    if run_dir_name.startswith("VM"):
        return "RT" + run_dir_name[2:]
    return "RT_" + run_dir_name

def move_new_outdirs(prefix: str) -> list[Path]:
    moved = []
    if not MODELS_OUT_DIR.exists():
        return moved
    for d in MODELS_OUT_DIR.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith(prefix):
            dest = RETRAINED_DIR / d.name
            if dest.exists():
                i = 2
                while (RETRAINED_DIR / f"{d.name}_dup{i}").exists():
                    i += 1
                dest = RETRAINED_DIR / f"{d.name}_dup{i}"
            shutil.move(str(d), str(dest))
            moved.append(dest)
    return moved

# --------------------------- completion tracking -------------------------------

def read_completed() -> Set[str]:
    if not COMPLETED_FILE.exists():
        return set()
    try:
        with open(COMPLETED_FILE, "r", encoding="utf-8") as f:
            items = [ln.strip() for ln in f if ln.strip()]
        return set(items)
    except Exception:
        return set()

def append_completed(run_dir_name: str) -> None:
    with open(COMPLETED_FILE, "a", encoding="utf-8") as f:
        f.write(run_dir_name + "\n")

# ----------------------------------- main --------------------------------------

def main():
    import torch
    ap = argparse.ArgumentParser(description="Retrain FlowPre models for Top-K union (using on-disk configs).")
    ap.add_argument("--k", type=int, default=20, help="Top-K per ranking to union (default 20).")
    ap.add_argument("--limit", type=int, default=0, help="Cap number of retrains (0 = no cap).")
    ap.add_argument("--device", type=str, choices=["auto","cuda","cpu"], default="auto",
                    help="auto → cuda if available else cpu.")
    ap.add_argument("--condition-col", type=str, default="type", help="Condition column.")
    ap.add_argument("--seed", type=int, default=123456, help="Fixed seed for all retrains.")
    ap.add_argument("--cpu-threads", type=int, default=1, help="torch.set_num_threads (default 1).")
    ap.add_argument("--dry-run", action="store_true", help="List what would be retrained and exit.")

    ap.add_argument("--quiet", action="store_true", help="Suppress epoch-level logs (only high-level retrain status).")

    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.cpu_threads)))
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")

    # --- print key directories
    print(f"📂 Config source dir:   {RESULTS_DIR}")
    print(f"📦 Retrained dest dir:  {RETRAINED_DIR}")
    print(f"🗂  Trainer temp dir:    {MODELS_OUT_DIR}\n")

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results folder not found: {RESULTS_DIR}")

    df = collect_results(RESULTS_DIR)
    if df.empty:
        raise RuntimeError("No *_results.yaml files found under outputs/flow_pre.")

    selected = top_union_indices(df, k=args.k)
    print(f"🔢 Union size (Top-{args.k} across 8 ranks): {len(selected)}")
    print(f"🧮 run_idx in union (sorted): {selected}\n")

    pool = df[df["run_idx"].isin(selected)].copy().sort_values(
        ["val_rrmse_mean","val_rrmse_std","gap_rrmse_mean","run_idx"]
    )

    # resume: skip already completed run_dir_names
    completed = read_completed()
    if completed:
        pool = pool[~pool["run_dir_name"].isin(completed)]

    if args.limit and args.limit > 0:
        pool = pool.head(args.limit)

    print(f"✅ Will retrain {len(pool)} configs (after resume filter); device='{device}', seed={args.seed}\n")

    failures: list[str] = []

    for i, row in pool.reset_index(drop=True).iterrows():
        run_dir_name = row["run_dir_name"]           # e.g., VM_012_affine_rq_ratio1x3_...
        run_dir = RESULTS_DIR / run_dir_name
        base_name = rt_name_from_vm(run_dir_name)

        # already completed?
        if run_dir_name in completed:
            print(f"[skip] Already completed: {run_dir_name}")
            continue

        # locate config
        try:
            cfg_path = find_config_in_run_dir(run_dir)
        except Exception as e:
            print(f"[ERROR] Missing config in {run_dir}: {e}")
            failures.append(run_dir_name)
            continue

        print(f"[{i+1}/{len(pool)}] ▶ Retraining {run_dir_name}  →  {base_name}")
        print(f"   • using config: {cfg_path}")

        if args.dry_run:
            continue

        # retry-once loop
        success = False
        last_err = None
        for attempt in (1, 2):
            try:
                _ = train_flowpre_pipeline(
                    condition_col=args.condition_col,
                    cols_to_exclude=["post_cleaning_index"],
                    config_filename=str(cfg_path),  # on-disk config from the run folder
                    base_name=base_name,            # renamed prefix RT_
                    device=device,
                    seed=int(args.seed),
                    verbose=not args.quiet
                )
                # move outputs to retrained folder
                moved = move_new_outdirs(base_name)
                if moved:
                    for dest in moved:
                        print(f"   • moved outputs to: {dest}")
                else:
                    print("   • warning: no output directory found to move (did the trainer write elsewhere?).")

                # mark completion
                append_completed(run_dir_name)
                completed.add(run_dir_name)
                success = True
                break

            except Exception as e:
                last_err = e
                print(f"   ✖ attempt {attempt} failed: {e}")
                if attempt == 1:
                    print("   ↻ retrying once...")
        if not success:
            print(f"   ⛔ giving up on {run_dir_name}")
            failures.append(run_dir_name)

    if args.dry_run:
        print("\n(dry-run) Nothing retrained. Remove --dry-run to start retraining.")
    else:
        print("\n🎉 Retraining pass finished.")
        if failures:
            print(f"⚠️  Failures ({len(failures)}): {failures}")
        done_now = read_completed()
        print(f"✅ Completed runs so far: {len(done_now)} (tracked in {COMPLETED_FILE})")

if __name__ == "__main__":
    main()
