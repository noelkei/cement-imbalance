#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_from_union_auto.py  (reseed runner: seed-major order)

What it does:
  For each SEED in SEEDS (outer loop), and each MODEL in MODELS (inner loop):
    • load the model's saved config from outputs/models/flowgen/<model_name>/
    • copy it, set seed (top-level `seed` + `training.seed`)
    • derive a new base_name by:
        - removing a trailing '_v1' (only if it's at the very end)
        - replacing/adding `_seed{SEED}`
    • apply optional overrides:
        - --ft_epochs -> training.finetune_num_epochs
        - --realism_bootstrap -> training.realism_bootstrap
        - --realism_rvr_bootstrap -> training.realism_rvr_bootstrap
    • run train_flowgen_pipeline with that temp config

If --models / --seeds are omitted, it falls back to DEFAULT_MODEL_NAMES / DEFAULT_SEEDS.
"""

from __future__ import annotations

import argparse
import copy
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml

from training.utils import ROOT_PATH
from training.train_flowgen import train_flowgen_pipeline

# ──────────────────────────────────────────────────────────────────────────────
# Defaults — edit these to your usual seeds/models
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAMES: List[str] = [
    "PS_100_flowgen_base_2_v1__FT180_50__E18_seed96024_v1",
    "PS_100_flowgen_base_2_v1__FT180_50__E32_seed96024_v1",
    "PS_100_flowgen_base_2_v1__UNKNOWN__T3_seed96024_v1",
    "PS_100_flowgen_base_1_v6__FT180_50__E03_seed96024_v1",
    "PS_100_flowgen_base_1_v6__FT180_50__E32_seed96024_v1",
    "PS_100_flowgen_base_4_v2__FT180_50__E32_seed96024_v1",
    "PS_100_flowgen_base_4_v2__FT180_50__A45_seed96024_v1",
]
DEFAULT_SEEDS: List[int] = [5270, 3901, 2887, 6479]

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT_PATH = Path(ROOT_PATH)
OUTPUTS_ROOT = ROOT_PATH / "outputs" / "models" / "flowgen"
CONFIG_DIR   = ROOT_PATH / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def read_yaml_any(p: Path) -> Dict[str, Any]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

def find_saved_config(run_dir: Path) -> Tuple[Dict[str, Any], Path]:
    """
    Prefer <run_dir>/<run_dir.name>.yaml, else first *.yaml in dir,
    else first *.yaml in any first-level subdir.
    """
    preferred = run_dir / f"{run_dir.name}.yaml"
    if preferred.exists():
        d = read_yaml_any(preferred)
        if d:
            return d, preferred

    for y in sorted(run_dir.glob("*.yaml")):
        d = read_yaml_any(y)
        if d:
            return d, y

    for sub in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        for y in sorted(sub.glob("*.yaml")):
            d = read_yaml_any(y)
            if d:
                return d, y

    raise FileNotFoundError(f"No YAML config found under {run_dir}")

def deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def write_temp_config(cfg_dict: dict, stem: str) -> Tuple[str, Path]:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    path = CONFIG_DIR / f"{safe}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    return path.name, path

def replace_or_append_seed_tag(name: str, seed: int | str) -> str:
    """
    If `name` contains `_seed<digits>`, replace the digits.
    Otherwise, append `_seed{seed}`.
    """
    s = str(seed)
    if re.search(r"_seed\d+", name):
        return re.sub(r"_seed\d+", f"_seed{s}", name)
    return f"{name}_seed{s}"

def remove_trailing_v1(name: str) -> str:
    """
    Remove exactly one trailing '_v1' if and only if it appears at the very end.
    Internal '_v1' tokens are left untouched.
    """
    return re.sub(r"_v1$", "", name)

def fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}h {m:02d}m {s:02d}s"

# ──────────────────────────────────────────────────────────────────────────────
# Runner (seed-major order)
# ──────────────────────────────────────────────────────────────────────────────
def run_reseeds(
    model_names: List[str],
    seeds: List[int],
    *,
    device: str = "cuda",
    condition_col: str = "type",
    ft_epochs: Optional[int] = None,               # override training.finetune_num_epochs
    realism_bootstrap: Optional[int] = None,       # override training.realism_bootstrap
    realism_rvr_bootstrap: Optional[int] = None,   # override training.realism_rvr_bootstrap
) -> None:
    """
    Seed-major execution order:
      for seed in seeds:
        for model in models:
            train(model, seed)
    """
    if not model_names:
        print("No model names provided. Nothing to do.")
        return
    if not seeds:
        print("No seeds provided. Nothing to do.")
        return

    # Preload each model's saved config once (from original names, not sanitized)
    model_cfg: Dict[str, Dict[str, Any]] = {}
    for model_name in model_names:
        run_dir = OUTPUTS_ROOT / model_name
        if not run_dir.exists():
            model_cfg[model_name] = {"ok": False, "err": f"Run dir not found: {run_dir}"}
            continue
        try:
            cfg, cfg_path_src = find_saved_config(run_dir)
            model_cfg[model_name] = {"ok": True, "cfg": cfg, "src": cfg_path_src}
        except Exception as e:
            model_cfg[model_name] = {"ok": False, "err": f"{type(e).__name__}: {e}"}

    # Planned runs (seed-major), showing sanitized target names
    print("—" * 100)
    print("Planned runs (seed-major):")
    for s in seeds:
        for m in model_names:
            m_sanit = remove_trailing_v1(m)
            planned = replace_or_append_seed_tag(m_sanit, s)
            print(f"  • seed={s}  |  {m}  →  {planned}")

    t_all = time.time()
    ok, fail = 0, 0

    total_runs = len(seeds) * len(model_names)
    run_idx = 0

    # Execute in seed-major order
    for seed in seeds:
        print("\n" + "=" * 100)
        print(f"▶️  SEED {seed}")
        for model_name in model_names:

            info = model_cfg[model_name]
            if not info.get("ok", False):
                print(f"❌ Skipping {model_name} (no usable config): {info.get('err','unknown error')}")
                fail += 1
                continue

            cfg_src = info["cfg"]
            cfg_src_path: Path = info["src"]

            # SANITIZE base name (remove trailing _v1 only) then apply/replace seed tag
            sanitized = remove_trailing_v1(model_name)
            new_base = replace_or_append_seed_tag(sanitized, seed)

            # Deep-copy config and apply seed + optional overrides
            cfg_copy = copy.deepcopy(cfg_src)
            cfg_copy["seed"] = int(seed)
            deep_set(cfg_copy, "training.seed", int(seed))
            if ft_epochs is not None:
                deep_set(cfg_copy, "training.finetune_num_epochs", int(ft_epochs))
            if realism_bootstrap is not None:
                deep_set(cfg_copy, "training.realism_bootstrap", int(realism_bootstrap))
            if realism_rvr_bootstrap is not None:
                deep_set(cfg_copy, "training.realism_rvr_bootstrap", int(realism_rvr_bootstrap))

            # write temp config
            temp_cfg_name, temp_cfg_path = write_temp_config(cfg_copy, stem=f"{new_base}")

            run_idx += 1
            print("\n" + "—" * 100)
            print(f"[{run_idx}/{total_runs}] ▶️  Training: {new_base}")
            print(f"    from config: {cfg_src_path.name}")
            print(f"    temp config: {temp_cfg_name}")
            print(f"    seed: {seed} | device: {device}")
            if ft_epochs is not None:
                print(f"    override: training.finetune_num_epochs = {ft_epochs}")
            if realism_bootstrap is not None:
                print(f"    override: training.realism_bootstrap = {realism_bootstrap}")
            if realism_rvr_bootstrap is not None:
                print(f"    override: training.realism_rvr_bootstrap = {realism_rvr_bootstrap}")

            t0 = time.time()
            try:
                _ = train_flowgen_pipeline(
                    condition_col=condition_col,
                    config_filename=temp_cfg_name,
                    base_name=new_base,
                    device=device,
                    seed=int(seed),
                    verbose=False,
                )
                dt = time.time() - t0
                print(f"✅ Done: {new_base}  |  duration: {fmt_hms(dt)}")
                ok += 1
            except Exception as e:
                dt = time.time() - t0
                print(f"❌ Failed: {new_base}  |  {type(e).__name__}: {e}  |  duration: {fmt_hms(dt)}")
                fail += 1

    print("\n" + "—" * 100)
    print(f"All done. OK: {ok}  |  Failed: {fail}  |  Total time: {fmt_hms(time.time() - t_all)}")

# ──────────────────────────────────────────────────────────────────────────────
# Minimal CLI (optional)
# ──────────────────────────────────────────────────────────────────────────────
def _parse_cli_list(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]

def _parse_cli_int_list(s: str) -> List[int]:
    out = []
    for t in s.split(","):
        t = t.strip()
        if not t: continue
        out.append(int(t))
    return out

def main():
    ap = argparse.ArgumentParser(description="Reseed previously trained FlowGen models (seed-major order).")
    ap.add_argument("--models", type=str, default=None,
                    help="Comma-separated run names under outputs/models/flowgen/. If omitted, uses DEFAULT_MODEL_NAMES.")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated integers. If omitted, uses DEFAULT_SEEDS.")
    ap.add_argument("--ft_epochs", type=int, default=150,
                    help="Override training.finetune_num_epochs in each run's config.")
    ap.add_argument("--realism_bootstrap", type=int, default=None,
                    help="Override training.realism_bootstrap in each run's config.")
    ap.add_argument("--realism_rvr_bootstrap", type=int, default=None,
                    help="Override training.realism_rvr_bootstrap in each run's config.")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--condition_col", type=str, default="type")
    args = ap.parse_args()

    model_names = DEFAULT_MODEL_NAMES if not args.models else _parse_cli_list(args.models)
    seeds = DEFAULT_SEEDS if not args.seeds else _parse_cli_int_list(args.seeds)

    print("Using models:", model_names)
    print("Using seeds:", seeds)
    if args.ft_epochs is not None:
        print("Override: training.finetune_num_epochs =", int(args.ft_epochs))
    if args.realism_bootstrap is not None:
        print("Override: training.realism_bootstrap =", int(args.realism_bootstrap))
    if args.realism_rvr_bootstrap is not None:
        print("Override: training.realism_rvr_bootstrap =", int(args.realism_rvr_bootstrap))

    run_reseeds(
        model_names,
        seeds,
        device=args.device,
        condition_col=args.condition_col,
        ft_epochs=args.ft_epochs,
        realism_bootstrap=args.realism_bootstrap,
        realism_rvr_bootstrap=args.realism_rvr_bootstrap,
    )

if __name__ == "__main__":
    main()
