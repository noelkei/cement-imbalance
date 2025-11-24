# retrain_from_meta_union_v2.py
from __future__ import annotations

import argparse
import os
import sys
import re
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, List, Set
import yaml
import numpy as np
import pandas as pd
from collections import defaultdict

# keep CPU phases well-behaved by default
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---- project paths ------------------------------------------------------------
try:
    from training.utils import ROOT_PATH
except Exception:
    ROOT_PATH = Path("../../training").resolve()
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

from training.train_flow_pre import train_flowpre_pipeline  # your trainer

# === Paths ===
PREV_RETRAIN_DIR = Path(ROOT_PATH) / "outputs" / "retrained"            # SOURCE: previous retrains (meta selection base)
MODELS_OUT_DIR   = Path(ROOT_PATH) / "outputs" / "models" / "flow_pre"  # trainer writes here first (UNCHANGED)
RETRAINED_V2_DIR = Path(ROOT_PATH) / "outputs" / "retrained_v2"         # DEST: per-seed subfolders live here
RETRAINED_V2_DIR.mkdir(parents=True, exist_ok=True)

COMPLETED_FILE   = RETRAINED_V2_DIR / "_completed_runs.txt"             # progress log: one line per "seed:run_dir_name"

# ---------------------- helpers (match your notebook logic) --------------------

def robust_z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    med = np.nanmedian(s); mad = np.nanmedian(np.abs(s - med))
    if mad == 0 or np.isnan(mad):
        std = np.nanstd(s)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros_like(s), index=series.index)
        return (s - med) / (std + 1e-8)
    return 0.6745 * (s - med) / (mad + 1e-8)

def pick_rrmse(split_dict: Dict[str, Any]) -> Tuple[float, float]:
    m = (split_dict or {}).get("rrmse_mean_whole", (split_dict or {}).get("rrmse_mean", np.nan))
    s = (split_dict or {}).get("rrmse_std_whole",  (split_dict or {}).get("rrmse_std",  np.nan))
    try:
        return float(m), float(s)
    except Exception:
        return np.nan, np.nan

def perclass_stats(pc: Dict[str, Any]) -> Tuple[float, float, float, float]:
    if not isinstance(pc, dict) or not pc:
        return (np.nan, np.nan, np.nan, np.nan)
    rr_m, rr_s, ns = [], [], []
    for _, v in pc.items():
        rr_m.append(v.get("rrmse_mean", np.nan))
        rr_s.append(v.get("rrmse_std", np.nan))
        ns.append(v.get("n", np.nan))
    rr_m = np.array(rr_m, dtype=float); rr_s = np.array(rr_s, dtype=float); ns = np.array(ns, dtype=float)
    worst_m = float(np.nanmax(rr_m)) if rr_m.size else np.nan
    worst_s = float(np.nanmax(rr_s)) if rr_s.size else np.nan
    mask_m = (~np.isnan(rr_m)) & (~np.isnan(ns))
    wavg_m = float(np.average(rr_m[mask_m], weights=ns[mask_m])) if mask_m.any() else np.nan
    mask_s = (~np.isnan(rr_s)) & (~np.isnan(ns))
    wavg_s = float(np.average(rr_s[mask_s], weights=ns[mask_s])) if mask_s.any() else np.nan
    return worst_m, wavg_m, worst_s, wavg_s

def infer_run_idx(run_name: str) -> float:
    m = re.search(r'(?:^|[_-])(\d{1,4})(?:[_-]|$)', run_name)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return float('nan')
    return float('nan')

def read_one_results(yaml_path: Path) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    row: Dict[str, Any] = {}
    row["file"] = str(yaml_path)
    row["run_name"] = yaml_path.parent.name
    row["run_idx"]  = infer_run_idx(row["run_name"])

    row["best_epoch"]   = data.get("best_epoch", np.nan)
    row["total_epochs"] = data.get("total_epochs", np.nan)
    row["seed"]         = data.get("seed", np.nan)

    for split in ("train","val","test"):
        s = data.get(split, {}) or {}
        m, sd = pick_rrmse(s)
        row[f"{split}_rrmse_mean"] = m
        row[f"{split}_rrmse_std"]  = sd
        row[f"{split}_mean_plus_std"] = m + sd if (np.isfinite(m) and np.isfinite(sd)) else np.nan

        iso = s.get("isotropy_stats", {}) or {}
        row[f"{split}_skew_abs"]  = abs(iso.get("skewness_mean", np.nan)) if pd.notna(iso.get("skewness_mean", np.nan)) else np.nan
        kurt = iso.get("kurtosis_mean", np.nan)
        row[f"{split}_kurt_excess_abs"] = abs(float(kurt) - 3.0) if pd.notna(kurt) else np.nan
        row[f"{split}_eigstd"]   = iso.get("eigval_std", np.nan)
        row[f"{split}_mahal_mu"] = iso.get("mahalanobis_mean", np.nan)
        row[f"{split}_mahal_md"] = iso.get("mahalanobis_median", np.nan)

        pc = s.get("per_class_iso_rrmse", {}) or {}
        wmean, wavgm, wstd, wavgs = perclass_stats(pc)
        row[f"{split}_pc_worst_mean"] = wmean
        row[f"{split}_pc_wavg_mean"]  = wavgm
        row[f"{split}_pc_worst_std"]  = wstd
        row[f"{split}_pc_wavg_std"]   = wavgs

    # blends/gaps
    row["gap_val_train_mean"] = row["val_rrmse_mean"] - row["train_rrmse_mean"]
    row["gap_val_train_sum"]  = (row["val_rrmse_mean"] + row["val_rrmse_std"]) - (row["train_rrmse_mean"] + row["train_rrmse_std"])
    row["train_val_mean_plus_std"] = (
        (row["train_rrmse_mean"] + row["train_rrmse_std"]) +
        (row["val_rrmse_mean"]   + row["val_rrmse_std"])
    ) if (pd.notna(row["train_rrmse_mean"]) and pd.notna(row["train_rrmse_std"])
          and pd.notna(row["val_rrmse_mean"]) and pd.notna(row["val_rrmse_std"])) else np.nan

    return row

def gather_results(base_dir: Path) -> pd.DataFrame:
    patterns = ["**/*_results.yaml", "**/results.yaml", "**/*results.yml"]
    files = set()
    for pat in patterns:
        files.update(base_dir.glob(pat))
    if not files:
        files.update([p for p in base_dir.glob("**/*.yaml") if "result" in p.name.lower()])
    rows = []
    for p in sorted(files):
        try:
            rows.append(read_one_results(p))
        except Exception as e:
            print(f"[WARN] Could not parse {p}: {e}")
    return pd.DataFrame(rows)

# ---------- compute union of Top-N across six meta tables (rank & z) -----------

def build_meta_union_runidx(df: pd.DataFrame, top_n: int = 10) -> List[int]:
    df = df.copy()
    df["val_mean_plus_std"]   = df["val_rrmse_mean"]   + df["val_rrmse_std"]
    df["train_mean_plus_std"] = df["train_rrmse_mean"] + df["train_rrmse_std"]

    # isotropy & fairness composites (validation)
    iso_terms_val = pd.DataFrame({
        "skew": df["val_skew_abs"],
        "kurt": df["val_kurt_excess_abs"],
        "eig":  df["val_eigstd"],
        "mu":   df["val_mahal_mu"],
        "md":   df["val_mahal_md"],
    })
    iso_w = pd.Series({"skew": 1.0, "kurt": 1.0, "eig": 1.2, "mu": 0.3, "md": 0.3})
    # --- Absolute deviations from theory instead of z-scored ---
    # target values: skew→0, kurt→3, eigstd→0, mahalanobis→sqrt(D-0.5)
    D = 43  # latent dim
    target_mu = np.sqrt(D - 0.5)
    target_md = np.sqrt(D - 0.5)

    iso_dev = pd.DataFrame({
        "skew": df["val_skew_abs"],  # already |skew|
        "kurt": df["val_kurt_excess_abs"],  # already |kurt-3|
        "eig": df["val_eigstd"],  # already deviation
        "mu": abs(df["val_mahal_mu"] - target_mu),
        "md": abs(df["val_mahal_md"] - target_md),
    })
    df["val_isotropy_score"] = (iso_dev * iso_w).sum(axis=1) / max(iso_w.sum(), 1e-8)

    fair_terms_val = pd.DataFrame({
        "pc_worst_mean": df["val_pc_worst_mean"],
        "pc_worst_std":  df["val_pc_worst_std"],
        "pc_wavg_mean":  df["val_pc_wavg_mean"],
    })
    fair_z = fair_terms_val.apply(robust_z, axis=0)
    df["val_fairness_score"] = fair_z.mean(axis=1)

    rankings: Dict[str, Dict[str, Any]] = {
        # Validation reconstruction
        "Val RRMSE — mean":       {"sort_cols": ["val_rrmse_mean","val_rrmse_std","val_mean_plus_std"], "asc": [True,True,True]},
        "Val RRMSE — std":        {"sort_cols": ["val_rrmse_std","val_rrmse_mean","val_mean_plus_std"], "asc": [True,True,True]},
        "Val RRMSE — mean+std":   {"sort_cols": ["val_mean_plus_std","val_rrmse_mean","val_rrmse_std"], "asc": [True,True,True]},
        # Train reconstruction
        "Train RRMSE — mean":     {"sort_cols": ["train_rrmse_mean","train_rrmse_std","val_rrmse_mean"], "asc": [True,True,True]},
        "Train RRMSE — std":      {"sort_cols": ["train_rrmse_std","train_rrmse_mean","val_rrmse_mean"], "asc": [True,True,True]},
        "Train RRMSE — mean+std": {"sort_cols": ["train_mean_plus_std","val_rrmse_mean"], "asc": [True,True]},
        # Gaps
        "Small gap (Val-Train) mean":     {"sort_cols": ["gap_val_train_mean","val_mean_plus_std"], "asc": [True,True]},
        "Small gap (Val-Train) mean+std": {"sort_cols": ["gap_val_train_sum","val_mean_plus_std"],  "asc": [True,True]},
        # Isotropy
        "Isotropy composite (Val)": {"sort_cols": ["val_isotropy_score","val_eigstd","val_kurt_excess_abs","val_skew_abs"], "asc": [True,True,True,True]},
        "Low eigstd then kurt diff": {"sort_cols": ["val_eigstd","val_kurt_excess_abs","val_skew_abs"], "asc": [True,True,True]},
        "Low |skew|":                {"sort_cols": ["val_skew_abs","val_eigstd","val_kurt_excess_abs"], "asc": [True,True,True]},
        "Low |kurt-3|":              {"sort_cols": ["val_kurt_excess_abs","val_eigstd","val_skew_abs"], "asc": [True,True,True]},
        # Fairness
        "Per-class fairness (Val)": {"sort_cols": ["val_fairness_score","val_pc_worst_mean","val_pc_worst_std"], "asc": [True,True,True]},
        "Low worst per-class mean": {"sort_cols": ["val_pc_worst_mean","val_pc_worst_std","val_pc_wavg_mean"], "asc": [True,True,True]},
        # Composites
        "Composite (Recon+Iso+Fair)": {"sort_cols": ["val_fairness_score","val_isotropy_score","val_mean_plus_std"], "asc": [True,True,True]},
        "Balanced (recon+eigstd+kurt+pcwm)": {"sort_cols": ["val_mean_plus_std","val_eigstd","val_kurt_excess_abs","val_pc_worst_mean"], "asc": [True,True,True,True]},
        # Train+Val blends
        "Train+Val mean+std (sum of sums)": {"sort_cols": ["train_val_mean_plus_std","val_mean_plus_std"], "asc": [True,True]},
        "Train+Val recon score (z)":        {"sort_cols": ["train_val_mean_plus_std","val_mean_plus_std"], "asc": [True,True]},
    }

    # normalized positional ranks per leaderboard
    table_ranks: Dict[str, pd.Series] = {}
    for name, spec in rankings.items():
        df_sorted = df.sort_values(spec["sort_cols"], ascending=spec["asc"], na_position="last")
        ordered = df_sorted["run_idx"].tolist()
        n = len(ordered)
        pos = pd.Series(range(n), index=ordered, dtype=float)
        norm = pos / max(n - 1, 1)
        norm = norm[pd.to_numeric(pd.Index(norm.index), errors="coerce").notna()]
        norm.index = norm.index.astype(float).astype(int)
        table_ranks[name] = norm

    all_run_idxs = sorted(set(int(i) for i in df["run_idx"].dropna().astype(int)))
    ranks_frame = pd.DataFrame(index=all_run_idxs)
    for t, s in table_ranks.items():
        col = pd.Series(1.0, index=ranks_frame.index, dtype=float)
        idx = s.index.intersection(ranks_frame.index)
        col.loc[idx] = s.loc[idx].values
        ranks_frame[t] = col

    z_frame = ranks_frame.apply(robust_z, axis=0)

    META_SPECS: Dict[str, Dict[str, float]] = {
        "Meta: Reconstruction/Fit": {
            "Val RRMSE — mean": 0.22, "Val RRMSE — std": 0.18, "Val RRMSE — mean+std": 0.20,
            "Small gap (Val-Train) mean": 0.10, "Small gap (Val-Train) mean+std": 0.10,
            "Train+Val mean+std (sum of sums)": 0.10, "Train+Val recon score (z)": 0.10,
        },
        "Meta: Isotropy/Gaussianity": {
            "Isotropy composite (Val)": 0.40, "Low eigstd then kurt diff": 0.30,
            "Low |kurt-3|": 0.20, "Low |skew|": 0.10,
        },
        "Meta: Fairness/Balance": {
            "Low worst per-class mean": 0.50, "Per-class fairness (Val)": 0.30,
            "Composite (Recon+Iso+Fair)": 0.12, "Balanced (recon+eigstd+kurt+pcwm)": 0.08,
        },
    }
    def norm_w(d: Dict[str, float]) -> pd.Series:
        s = sum(d.values());
        return pd.Series({k: (v / s if s > 0 else 0.0) for k, v in d.items()})

    meta_top_sets = []
    for _, weights in META_SPECS.items():
        w = norm_w(weights); cols = [c for c in w.index if c in ranks_frame.columns]
        if not cols:
            continue
        meta_r = ranks_frame[cols].mul(w[cols], axis=1).sum(axis=1)
        meta_r_top = meta_r.sort_values().index[:top_n]
        meta_top_sets.append(set(int(i) for i in meta_r_top))
        meta_z = z_frame[cols].mul(w[cols], axis=1).sum(axis=1)
        meta_z_top = meta_z.sort_values().index[:top_n]
        meta_top_sets.append(set(int(i) for i in meta_z_top))

    if not meta_top_sets:
        return []
    union_set = set().union(*meta_top_sets)
    return sorted(union_set)

# ---------------------- file ops: moving / copying -----------------------------

def move_new_outdirs(prefix: str, seed_dest_dir: Path) -> List[Path]:
    """
    Move newly created model output folders from MODELS_OUT_DIR whose name starts with 'prefix'
    into 'seed_dest_dir' (which is outputs/retrained_v2/seed_<seed>).
    """
    moved = []
    seed_dest_dir.mkdir(parents=True, exist_ok=True)
    if not MODELS_OUT_DIR.exists():
        return moved
    for d in list(MODELS_OUT_DIR.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith(prefix):
            dest = seed_dest_dir / d.name
            if dest.exists():
                i = 2
                while (seed_dest_dir / f"{d.name}_dup{i}").exists():
                    i += 1
                dest = seed_dest_dir / f"{d.name}_dup{i}"
            shutil.move(str(d), str(dest))
            moved.append(dest)
    return moved

def ensure_seed_dir(seed: int) -> Path:
    p = RETRAINED_V2_DIR / f"seed_{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_copy_seed1234(selected_runidx: List[int], run_idx_to_dir: Dict[int, Path]) -> None:
    """
    For seed 1234 ONLY: copy selected runs from PREV_RETRAIN_DIR into retrained_v2/seed_1234
    if the run folder is missing OR exists but lacks a config .yaml and/or a *_results.yaml.
    No renaming of inner files; we keep the original filenames.
    """
    seed_dir = ensure_seed_dir(1234)
    for ridx in selected_runidx:
        src_dir = run_idx_to_dir.get(ridx)
        if src_dir is None:
            print(f"[seed 1234] WARN: missing source for run_idx={ridx} → skip copy.")
            continue
        dest_dir = seed_dir / f"seed1234_{src_dir.name}"

        def has_required_files(path: Path) -> bool:
            if not path.exists() or not path.is_dir():
                return False
            ymls = [p for p in path.glob("*.yaml") if not p.name.endswith("_results.yaml")]
            res  = list(path.glob("*_results.yaml"))
            return bool(ymls) and bool(res)

        if not dest_dir.exists():
            print(f"[seed 1234] copy → {dest_dir.name}")
            shutil.copytree(src_dir, dest_dir)
        else:
            if not has_required_files(dest_dir):
                print(f"[seed 1234] fixing missing files in {dest_dir.name}")
                # copy missing critical files only
                src_ymls = [p for p in src_dir.glob("*.yaml") if not p.name.endswith("_results.yaml")]
                src_res  = list(src_dir.glob("*_results.yaml"))
                for p in src_ymls + src_res:
                    target = dest_dir / p.name
                    if not target.exists():
                        shutil.copy2(p, target)

# --------------------------- completion tracking --------------------------------

def read_completed() -> Set[str]:
    if not COMPLETED_FILE.exists():
        return set()
    try:
        with open(COMPLETED_FILE, "r", encoding="utf-8") as f:
            items = [ln.strip() for ln in f if ln.strip()]
        return set(items)
    except Exception:
        return set()

def append_completed(key: str) -> None:
    with open(COMPLETED_FILE, "a", encoding="utf-8") as f:
        f.write(key + "\n")

# ----------------------------------- main --------------------------------------

def main():
    import torch
    ap = argparse.ArgumentParser(description="v2: Multi-seed retrain from union of Top-N across 6 meta tables, with per-seed folders; seed 1234 copied only.")
    ap.add_argument("--meta-top", type=int, default=10, help="Top-N per meta table to union (default 10).")
    ap.add_argument("--device", type=str, choices=["auto","cuda","cpu"], default="auto",
                    help="auto → cuda if available else cpu.")
    ap.add_argument("--condition-col", type=str, default="type", help="Condition column.")
    # Three retrain seeds; seed 1234 will not retrain (copied only)
    ap.add_argument("--seeds", type=str, default="4321,5678,9101",
                    help="Comma-separated list of THREE retrain seeds (1234 is handled by copy-only).")
    ap.add_argument("--cpu-threads", type=int, default=1, help="torch.set_num_threads (default 1).")
    ap.add_argument("--limit", type=int, default=0, help="Cap number of selected run_idx to process (0 = no cap).")
    ap.add_argument("--dry-run", action="store_true", help="List what would be retrained and exit.")
    ap.add_argument("--quiet", action="store_true", help="Suppress epoch logs (only high-level status).")

    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.cpu_threads)))
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")

    print(f"📂 Selection source dir: {PREV_RETRAIN_DIR}")
    print(f"📦 New retrain dest dir: {RETRAINED_V2_DIR}")
    print(f"🗂  Trainer temp dir:    {MODELS_OUT_DIR}\n")

    if not PREV_RETRAIN_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {PREV_RETRAIN_DIR}")

    df = gather_results(PREV_RETRAIN_DIR)
    if df.empty:
        raise RuntimeError(f"No enriched results found under {PREV_RETRAIN_DIR}.")

    # ---- compute union of Top-N across 6 meta tables
    selected_runidx = build_meta_union_runidx(df, top_n=int(args.meta_top))
    if args.limit and args.limit > 0:
        selected_runidx = selected_runidx[: int(args.limit)]
    print(f"🔢 Selected by meta-union (Top-{args.meta_top} × 6 tables): {len(selected_runidx)}")
    print(f"🧮 run_idx (sorted): {selected_runidx}\n")

    # Map run_idx -> original run_dir name (in PREV_RETRAIN_DIR)
    run_idx_to_dir: Dict[int, Path] = {}
    for run_dir in sorted(PREV_RETRAIN_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        ridx = infer_run_idx(run_dir.name)
        if np.isnan(ridx):
            continue
        if int(ridx) in selected_runidx:
            run_idx_to_dir[int(ridx)] = run_dir

    # --- seed 1234: copy-only (no retrain)
    ensure_copy_seed1234(selected_runidx, run_idx_to_dir)

    # --- seeds to retrain (not including 1234)
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    except Exception:
        raise ValueError("--seeds must be a comma-separated list of integers, e.g. '4321,5678,9101'")
    if len(seeds) != 3:
        raise ValueError("Please provide exactly THREE seeds via --seeds (default is '4321,5678,9101').")

    # completion key format: f"{seed}:{run_dir_name}"
    completed = read_completed()

    # planned jobs (per-seed), no pre-copy
    jobs: List[Tuple[int, int, Path]] = []  # (run_idx, seed, src_dir)
    for ridx in selected_runidx:
        src_dir = run_idx_to_dir.get(ridx)
        if src_dir is None:
            print(f"[WARN] Missing source folder for run_idx={ridx} under {PREV_RETRAIN_DIR}; skipping.")
            continue
        for seed in seeds:
            key = f"{seed}:{src_dir.name}"
            # check both completed-file AND retrained_v2/seed_<seed> actual folder
            seed_dir = ensure_seed_dir(seed)
            expected = seed_dir / f"seed{seed}_{src_dir.name}"
            if key in completed or expected.exists():
                print(f"[skip] Already done: {key} (folder exists or tracked)")
                continue
            jobs.append((ridx, seed, src_dir))

    total_runs = len(selected_runidx)
    total_jobs = len(jobs)
    print(f"\n📋 Planned retrains (excluding seed 1234): {total_jobs}  (run_idx × {len(seeds)} seeds)")

    if args.dry_run:
        for i, (ridx, seed, src) in enumerate(jobs, 1):
            print(f"DRY-RUN  {i}/{total_jobs}  seed={seed}  run_idx={ridx}  src={src.name}")
        print("\n(dry-run) No training performed.")
        return

    # execute: iterate by seed to show per-seed counters
    global_counter = 0
    failures: list[str] = []

    G = total_runs * len(seeds)  # global total (only retrain seeds)
    # run per seed to print s/S as requested
    for seed in seeds:
        seed_dest_dir = ensure_seed_dir(seed)
        seed_jobs = [(ridx, sd, src) for (ridx, sd, src) in jobs if sd == seed]
        S = len(seed_jobs)
        if S == 0:
            continue
        s = 0

        for ridx, sd, src_dir in seed_jobs:
            s += 1
            global_counter += 1
            print(f"\n[{global_counter}/{G}]  (seed {seed}: {s}/{S})  ▶ Retraining run_idx={ridx}")

            # base_name includes seed to keep MODELS_OUT_DIR unique; outputs will be moved into seed folder
            base_name = f"seed{seed}_{src_dir.name}"

            # locate config within SOURCE run folder (no pre-copy)
            cfg_path = None
            exact = src_dir / f"{src_dir.name}.yaml"
            if exact.exists():
                cfg_path = exact
            else:
                cands = [p for p in src_dir.glob("*.yaml") if not p.name.endswith("_results.yaml")]
                if len(cands) == 1:
                    cfg_path = cands[0]
                elif len(cands) > 1:
                    m = [p for p in cands if p.stem == src_dir.name]
                    cfg_path = m[0] if m else cands[0]

            if cfg_path is None or not cfg_path.exists():
                msg = f"[ERROR] Missing config in {src_dir}"
                print(msg); failures.append(f"{seed}:{src_dir.name}"); continue

            print(f"   • config:  {cfg_path}")
            print(f"   • base_name: {base_name}")

            # choose device
            use_device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")

            # retry-once loop
            success = False
            for attempt in (1, 2):
                try:
                    _ = train_flowpre_pipeline(
                        condition_col=args.condition_col,
                        cols_to_exclude=["post_cleaning_index"],
                        config_filename=str(cfg_path),
                        base_name=base_name,
                        device=use_device,
                        seed=int(seed),
                        verbose=not args.quiet
                    )
                    # Move outputs from MODELS_OUT_DIR to retrained_v2/seed_<seed>/ after finishing
                    moved = move_new_outdirs(base_name, seed_dest_dir)
                    if moved:
                        for dest in moved:
                            print(f"   • moved outputs to: {dest}")
                    else:
                        print("   • warning: no output directory found to move (did the trainer write elsewhere?).")

                    append_completed(f"{seed}:{src_dir.name}")
                    success = True
                    break
                except Exception as e:
                    print(f"   ✖ attempt {attempt} failed: {e}")
                    if attempt == 1:
                        print("   ↻ retrying once...")

            if not success:
                print(f"   ⛔ giving up on seed={seed}, run_idx={ridx}")
                failures.append(f"{seed}:{src_dir.name}")

    print("\n🎉 v2 multi-seed retraining pass finished.")
    if failures:
        print(f"⚠️  Failures ({len(failures)}): {failures}")
    done_now = read_completed()
    print(f"✅ Completed (tracked in {COMPLETED_FILE}): {len(done_now)}")

if __name__ == "__main__":
    main()
