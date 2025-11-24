#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_from_union_auto.py

Pipeline completo:
  1) Escanear runs → cargar YAMLs → calcular scores (classic + banded).
  2) Excluir por substrings (opcional) y deduplicar por igualdad de TRAIN cfg
     en claves compartidas (ignorando claves configurables) quedándose con
     el mayor banded_score en cada grupo.
  3) Mostrar reporte de limpieza y tablas (Classic Top-15; y por métrica,
     Top-15 BLENDED = 0.5*inv-banded + 0.5*agg(métrica)) — TODO ya LIMPIO.
  4) UNION = Classic Top-15 ∪ (4×BLENDED Top-15).
  5) Detectar bases (.pt) y **imprimir** combinaciones por base (sin seed),
     conteo por base y total.
  6) Entrenar **por sweep→base**:
        para cada run del UNION: para cada base → ejecutar entrenamiento.

Overrides globales para TODOS los entrenos:
  training.finetune_num_epochs     = 100
  training.realism_bootstrap       = 100
  training.realism_rvr_bootstrap   = 100
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
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

# Tu stack del proyecto
from training.utils import ROOT_PATH, load_yaml_config
from training.train_flowgen import train_flowgen_pipeline

# ──────────────────────────────────────────────────────────────────────────────
# Ajustes globales
# ──────────────────────────────────────────────────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)

ROOT_PATH = Path(ROOT_PATH)
CONFIG_DIR = ROOT_PATH / "config"
OUTPUTS_ROOT = ROOT_PATH / "outputs" / "models" / "flowgen"
SWEEP_ROOT = ROOT_PATH / "outputs" / "sweeps" / "flowgen"
STATE_PATH = SWEEP_ROOT / "train_from_union_state.json"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

CLASSES = [0, 1, 2]
SPLITS = ["train", "val"]
SCOPES = ["overall", "perclass"]
COMPONENTS = ["xy", "x", "y"]

# Pesos para agregaciones
W_SPLIT = {"train": 0.5, "val": 0.5}
W_SCOPE = {"overall": 0.2, "perclass": 0.8}
W_COMPONENT = {"xy": 1.0, "x": 1.0, "y": 1.0}
W_CLASS = {k: 1.0 for k in CLASSES}

# Métricas (menor es mejor)
METRICS6 = ["ks_mean", "ks_median", "w1_mean", "w1_median", "mmd2_rvs", "mmd2_gap"]
METRICS4 = ["ks_mean", "ks_median", "w1_mean", "w1_median"]

# Bandas por métrica
BAND_CUTS = {
    "ks_mean":    [0.020, 0.060, 0.10, 0.150, 0.20, 0.250],
    "ks_median":  [0.01, 0.040, 0.080, 0.10, 0.150, 0.20],
    "w1_mean":    [0.025, 0.040, 0.050, 0.060, 0.070, 0.090],
    "w1_median":  [0.020, 0.032, 0.040, 0.050, 0.060, 0.080],
    "mmd2_rvs":   [0.0030, 0.0050, 0.0060, 0.0080, 0.0100, 0.0150],
    "mmd2_gap":   [0.004, 0.007, 0.010, 0.015, 0.020, 0.030],
}

RANKS = ["unusable", "very bad", "bad", "mediocre", "good", "very good", "perfect"]
RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}
_LABELS_BEST_FIRST = ["perfect", "very good", "good", "mediocre", "bad", "very bad", "unusable"]
_LABEL_TO_BOUNDS_INDEX = {lab: i for i, lab in enumerate(_LABELS_BEST_FIRST)}
EXPONENT_P = 1.35  # penaliza más al bajar de rango

METRIC_GROUPS = {
    "ks":  ["ks_mean", "ks_median"],
    "w1":  ["w1_mean", "w1_median"],
    "mmd": ["mmd2_rvs", "mmd2_gap"],
}
GROUP_WEIGHTS = {"ks": 0.30, "w1": 0.45, "mmd": 0.25}
_METRIC_TO_GROUP = {m: g for g, ms in METRIC_GROUPS.items() for m in ms}
METRIC_WEIGHT = {m: GROUP_WEIGHTS[_METRIC_TO_GROUP[m]] / len(METRIC_GROUPS[_METRIC_TO_GROUP[m]])
                 for m in METRICS6}
assert np.isclose(sum(GROUP_WEIGHTS.values()), 1.0)
assert np.isclose(sum(METRIC_WEIGHT[m] for m in METRICS6), 1.0)

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────
def deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}h {m:02d}m {s:02d}s"

def nested_get(d: Dict[str, Any], dotted: str):
    cur = d
    for tok in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        if tok in cur:
            cur = cur[tok]; continue
        try:
            itok = int(tok)
            if itok in cur:
                cur = cur[itok]; continue
        except Exception:
            pass
        return None
    return cur

def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def looks_like_results(d: Dict[str, Any]) -> bool:
    return any(nested_get(d, f"{split}.realism.overall.ks_mean") is not None
               for split in ("val", "train", "test"))

def read_yaml_any(p: Path) -> Dict[str, Any]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

def collect_run_yaml(run_dir: Path) -> Optional[Dict[str, Any]]:
    # .yaml en el dir primero
    for y in sorted(run_dir.glob("*.yaml")):
        d = read_yaml_any(y)
        if isinstance(d, dict) and looks_like_results(d):
            return d
    # luego subdirs de primer nivel
    for sub in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        for y in sorted(sub.glob("*.yaml")):
            d = read_yaml_any(y)
            if isinstance(d, dict) and looks_like_results(d):
                return d
    return None

def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out

def _norm_val(v: Any):
    if isinstance(v, float):
        return float(round(v, 12))
    if isinstance(v, (list, tuple)):
        return tuple(_norm_val(x) for x in v)
    if isinstance(v, np.generic):
        return _norm_val(np.asarray(v).item())
    return v

def z_lower(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def minmax01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)

def points_for_rank_idx(idx: int, p: float = EXPONENT_P) -> float:
    return 100.0 * ((idx / (len(RANKS) - 1)) ** p)

def band_label(metric: str, value: float) -> str:
    if metric not in BAND_CUTS or not np.isfinite(value):
        return "unusable"
    cuts = BAND_CUTS[metric]
    for i, ub in enumerate(cuts):  # perfect..very bad
        if value <= ub:
            return _LABELS_BEST_FIRST[i]
    return "unusable"

def continuous_points(metric: str, value: float, obs_max: float) -> Tuple[str, float]:
    if not np.isfinite(value):
        return "—", np.nan
    label = band_label(metric, value)
    cuts = BAND_CUTS[metric]
    bounds = [0.0] + list(cuts) + [float("inf")]
    bi = _LABEL_TO_BOUNDS_INDEX[label]
    L = bounds[bi]; U = bounds[bi + 1]
    if label == "unusable":
        U = obs_max if (np.isfinite(obs_max) and obs_max > L) else (L * 1.25 if L > 0 else 1.0)
    if not np.isfinite(L) or not np.isfinite(U) or U <= L:
        idx_cur = RANK_TO_IDX[label]
        pts = points_for_rank_idx(idx_cur)
        return f"{label} ({pts:.1f})", pts
    idx_cur = RANK_TO_IDX[label]
    idx_worse = max(0, idx_cur - 1)
    P_cur = points_for_rank_idx(idx_cur)
    P_worse = points_for_rank_idx(idx_worse)
    t = float(np.clip((U - value) / (U - L), 0.0, 1.0))
    pts = P_worse + t * (P_cur - P_worse)
    return f"{label} ({pts:.1f})", pts

def _norm_class_weights(wcls: Dict[int, float]) -> Dict[int, float]:
    s = sum(max(0.0, float(v)) for v in wcls.values()) or 1.0
    return {int(k): max(0.0, float(v)) / s for k, v in wcls.items()}

W_CLASS_NORM = _norm_class_weights(W_CLASS)

def _get_yaml_value(data: Dict[str, Any], split: str, scope: str, component: str,
                    metric: str, cls: Optional[int] = None) -> float:
    comp_key = "overall" if component == "xy" else component
    if scope == "overall":
        base = f"{split}.realism.{comp_key}"
    else:
        if cls is None:
            return np.nan
        base = f"{split}.realism.per_class.{int(cls)}.{comp_key}"
    if metric == "mmd2_gap":
        rvs = to_float(nested_get(data, f"{base}.mmd2_rvs"))
        rvr = to_float(nested_get(data, f"{base}.mmd2_rvr_med"))
        return float(max(rvs - rvr, 0.0)) if (np.isfinite(rvs) and np.isfinite(rvr)) else np.nan
    return to_float(nested_get(data, f"{base}.{metric}"))

# Score clásico (menor es mejor)
def _cell_sum_metrics(data: Dict[str, Any], split: str, scope: str, comp: str, cls: Optional[int]) -> float:
    vals = np.array([_get_yaml_value(data, split, scope, comp, m, cls) for m in METRICS6], dtype=float)
    if not np.isfinite(vals).any():
        return np.nan
    present = np.isfinite(vals)
    mean_present = np.nanmean(vals) if present.any() else np.nan
    return float(mean_present * len(METRICS6)) if np.isfinite(mean_present) else np.nan

def classic_score_for_run(data: Dict[str, Any]) -> float:
    num = 0.0; den = 0.0
    for split in SPLITS:
        ws = float(W_SPLIT.get(split, 0.0))
        w_over = float(W_SCOPE.get("overall", 0.0))
        for comp in COMPONENTS:
            wc = float(W_COMPONENT.get(comp, 0.0))
            csum = _cell_sum_metrics(data, split, "overall", comp, None)
            if np.isfinite(csum):
                w = ws * w_over * wc; num += w * csum; den += w
        w_pc = float(W_SCOPE.get("perclass", 0.0))
        for cls in CLASSES:
            wcls = float(W_CLASS_NORM.get(int(cls), 0.0))
            for comp in COMPONENTS:
                wc = float(W_COMPONENT.get(comp, 0.0))
                csum = _cell_sum_metrics(data, split, "perclass", comp, cls)
                if np.isfinite(csum):
                    w = ws * w_pc * wcls * wc; num += w * csum; den += w
    return num / den if den > 0 else np.nan

def aggregate_metric_for_run(data: Dict[str, Any], metric: str) -> float:
    num = 0.0; den = 0.0
    for split in SPLITS:
        ws = float(W_SPLIT.get(split, 0.0))
        w_over = float(W_SCOPE.get("overall", 0.0))
        for comp in COMPONENTS:
            wc = float(W_COMPONENT.get(comp, 0.0))
            v = _get_yaml_value(data, split, "overall", comp, metric, None)
            if np.isfinite(v):
                w = ws * w_over * wc; num += w * v; den += w
        w_pc = float(W_SCOPE.get("perclass", 0.0))
        for cls in CLASSES:
            wcls = float(W_CLASS_NORM.get(int(cls), 0.0))
            for comp in COMPONENTS:
                wc = float(W_COMPONENT.get(comp, 0.0))
                v = _get_yaml_value(data, split, "perclass", comp, metric, cls)
                if np.isfinite(v):
                    w = ws * w_pc * wcls * wc; num += w * v; den += w
    return num / den if den > 0 else np.nan

def compute_banded_view(run_names: List[str], OBS_MAX: Dict[str, float],
                        yaml_by_run: Dict[str, Any], df_meta: pd.DataFrame) -> pd.DataFrame:
    rows_out: List[Dict[str, Any]] = []
    for rn in run_names:
        data = yaml_by_run.get(rn);
        if data is None:
            continue
        meta = df_meta.loc[df_meta["run_name"] == rn].iloc[0]
        disp_row: Dict[str, Any] = {
            "run_name": rn,
            "family": meta.get("family"),
            "ft_epochs": int(meta.get("ft_epochs") or nested_get(data, "finetune.total_epochs") or 0),
        }
        num, den = 0.0, 0.0
        for split in SPLITS:
            ws = float(W_SPLIT.get(split, 0.0))
            w_over = float(W_SCOPE.get("overall", 0.0))
            for comp in COMPONENTS:
                wc = float(W_COMPONENT.get(comp, 0.0))
                base_w = ws * w_over * wc
                for m in METRICS6:
                    v = _get_yaml_value(data, split, "overall", comp, m, None)
                    _, pts = continuous_points(m, v, obs_max=OBS_MAX[m])
                    if np.isfinite(pts):
                        w = base_w * METRIC_WEIGHT[m]
                        num += w * pts; den += w
            w_pc = float(W_SCOPE.get("perclass", 0.0))
            for cls in CLASSES:
                wcls = float(W_CLASS_NORM.get(int(cls), 0.0))
                for comp in COMPONENTS:
                    wc = float(W_COMPONENT.get(comp, 0.0))
                    base_w = ws * w_pc * wcls * wc
                    for m in METRICS6:
                        v = _get_yaml_value(data, split, "perclass", comp, m, cls)
                        _, pts = continuous_points(m, v, obs_max=OBS_MAX[m])
                        if np.isfinite(pts):
                            w = base_w * METRIC_WEIGHT[m]
                            num += w * pts; den += w
        disp_row["banded_score"] = round(num / den, 3) if den > 0 else np.nan
        rows_out.append(disp_row)
    out = pd.DataFrame(rows_out)
    if out.empty: return out
    out = out.sort_values(["banded_score", "run_name"], ascending=[False, True]).reset_index(drop=True)
    return out[["run_name", "family", "ft_epochs", "banded_score"]]

def detect_family(run_name: str) -> str:
    m = re.match(r"(FT\d+(_\d+)?)", run_name)
    return m.group(1) if m else "UNKNOWN"

def identifier_token(run_name: str) -> str:
    m = re.search(r"flowgen_(.+)", run_name)
    if not m: return "IDUNK"
    first = m.group(1).split("_", 1)[0]
    m2 = re.match(r"([A-Za-z]+\d+)", first)
    return m2.group(1) if m2 else first

def find_yaml_with_keys(root: Path, required_keys: List[str]) -> Optional[Dict[str, Any]]:
    pref = root / f"{root.name}.yaml"
    if pref.exists():
        d = read_yaml_any(pref)
        if all(k in d for k in required_keys): return d
    for y in sorted(root.glob("*.yaml")):
        d = read_yaml_any(y)
        if all(k in d for k in required_keys): return d
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        for y in sorted(sub.glob("*.yaml")):
            d = read_yaml_any(y)
            if all(k in d for k in required_keys): return d
    return None

def find_base_model_yaml(base_dir: Path) -> Dict[str, Any]:
    d = find_yaml_with_keys(base_dir, ["model"])
    if not d or "model" not in d:
        raise FileNotFoundError(f"[BASE] model YAML no encontrado: {base_dir}")
    return {"model": d["model"]}

def find_training_yaml_for_run(run_dir: Path) -> Dict[str, Any]:
    d = find_yaml_with_keys(run_dir, ["training"])
    if not d or "training" not in d:
        raise FileNotFoundError(f"[RUN] training YAML no encontrado: {run_dir}")
    return {"training": d["training"]}

def find_pretrained_pt(base_dir: Path) -> str:
    cands = list((base_dir / "snapshots").glob("*.pt")) + list(base_dir.glob("*.pt"))
    if not cands:
        raise FileNotFoundError(f"No .pt en base: {base_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0])

def discover_bases(flowgen_root: Path) -> List[Path]:
    bases: List[Path] = []
    for d in sorted([p for p in flowgen_root.iterdir() if p.is_dir()]):
        if list((d / "snapshots").glob("*.pt")) or list(d.glob("*.pt")):
            bases.append(d)
    return bases

def find_latest_run_dir(prefix: str) -> Optional[Path]:
    cands = [p for p in OUTPUTS_ROOT.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not cands: return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

def has_results_yaml(run_dir: Optional[Path]) -> bool:
    if not run_dir or not run_dir.exists(): return False
    return any(p.name.endswith("_results.yaml") for p in run_dir.iterdir() if p.is_file())

def write_temp_config(cfg_dict: dict, stem: str) -> Tuple[str, Path]:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    path = CONFIG_DIR / f"{safe}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    return path.name, path

def read_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"statuses": {}, "attempts": {}}

def write_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# Lógica de igualdad de TRAIN cfg (con exclusiones)
# ──────────────────────────────────────────────────────────────────────────────
def equal_on_shared_training(a_train: Dict[str, Any], b_train: Dict[str, Any],
                             exclude_train_keys: List[str], tol: float = 1e-9) -> bool:
    def skip(k: str) -> bool:
        from fnmatch import fnmatchcase
        return any(fnmatchcase(k, pat) or fnmatchcase(f"training.{k}", pat)
                   for pat in exclude_train_keys)

    def flat_clean(d: Dict[str, Any]) -> Dict[str, Any]:
        f = {k.split("training.",1)[-1]: _norm_val(v) for k, v in flatten_dict(d).items()}
        return {k: v for k, v in f.items() if not skip(k)}

    af = flat_clean(a_train); bf = flat_clean(b_train)
    shared = sorted(set(af.keys()) & set(bf.keys()))
    if not shared: return False
    for k in shared:
        va, vb = af[k], bf[k]
        if isinstance(va, float) or isinstance(vb, float):
            try:
                fa, fb = float(va), float(vb)
                if not math.isclose(fa, fb, rel_tol=tol, abs_tol=tol): return False
            except Exception:
                if va != vb: return False
        else:
            if va != vb: return False
    return True

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n)); self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]: self.p[ra] = rb
        elif self.r[ra] > self.r[rb]: self.p[rb] = ra
        else: self.p[rb] = ra; self.r[ra] += 1

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Train combos from auto-built UNION (sweep→base).")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--exclude", type=str, default="",
                    help="Substrings separadas por coma para excluir por nombre de run.")
    ap.add_argument("--ignore_train_keys", type=str, default="finetune_num_epochs,early_stopping_patience,lr_decay_patience,lr_decay_factor,min_improvement,min_improvement_floor,lr_patience_factor,save_states,log_training,save_results,save_model,realism_bootstrap,realism_rvr_bootstrap",
                    help="Claves de TRAIN (sin 'training.') a ignorar en igualdad; admitir comodines con fnmatch. Coma-separadas.")
    ap.add_argument("--topk_classic", type=int, default=10)
    ap.add_argument("--topk_metric", type=int, default=10)
    ap.add_argument("--ft_epochs", type=int, default=100)
    ap.add_argument("--realism_bootstrap", type=int, default=100)
    ap.add_argument("--realism_rvr_bootstrap", type=int, default=100)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--sleep_on_fail_sec", type=int, default=60)
    args = ap.parse_args()

    device = args.device
    seed = int(args.seed)
    EXCLUDE_SUBSTRS = [s.strip() for s in args.exclude.split(",") if s.strip()]
    EXCLUDE_TRAIN_KEYS = [s.strip() for s in args.ignore_train_keys.split(",") if s.strip()]

    # ── 1) Escanear runs y YAMLs
    run_dirs = [d for d in sorted(OUTPUTS_ROOT.iterdir(), key=lambda p: p.name) if d.is_dir()]
    rows = []
    yaml_by_run: Dict[str, Any] = {}
    for rd in run_dirs:
        data = collect_run_yaml(rd)
        if data is None:
            continue
        yaml_by_run[rd.name] = data
        rows.append({
            "run_name": rd.name,
            "family": detect_family(rd.name),
            "seed": data.get("seed", None),
            "ft_epochs": int(nested_get(data, "finetune.total_epochs") or 0),
        })
    df_meta = pd.DataFrame(rows)
    if df_meta.empty:
        print(f"No hay YAMLs utilizables bajo {OUTPUTS_ROOT}")
        return

    # ── OBS_MAX para interpolar 'unusable'
    OBS_MAX = {m: 0.0 for m in METRICS6}
    for rn, data in yaml_by_run.items():
        for split in SPLITS:
            for comp in COMPONENTS:
                for m in METRICS6:
                    v = _get_yaml_value(data, split, "overall", comp, m, None)
                    if np.isfinite(v): OBS_MAX[m] = max(OBS_MAX[m], v)
                for cls in CLASSES:
                    for m in METRICS6:
                        v = _get_yaml_value(data, split, "perclass", comp, m, cls)
                        if np.isfinite(v): OBS_MAX[m] = max(OBS_MAX[m], v)

    # ── Scores banded para TODOS (para lookup rápido)
    banded_all = compute_banded_view(df_meta["run_name"].tolist(), OBS_MAX, yaml_by_run, df_meta)
    banded_scores_all = {rn: float(banded_all.loc[banded_all["run_name"] == rn, "banded_score"].values[0])
                         for rn in df_meta["run_name"] if rn in set(banded_all["run_name"])}

    # ── 2) CLEAN: excluir por nombre + cargar TRAIN cfgs + deduplicar por TRAIN igualdad
    all_runs = sorted(df_meta["run_name"].tolist())
    excluded_by_name = [rn for rn in all_runs if any(sub in rn for sub in EXCLUDE_SUBSTRS)]
    remaining = [rn for rn in all_runs if rn not in excluded_by_name]

    # cargar TRAIN cfg desde el YAML de cada run (buscamos un YAML con 'model' y 'training')
    def find_model_yaml(run_dir: Path) -> Optional[Dict[str, Any]]:
        pref = [run_dir / f"{run_dir.name}.yaml"]
        cands = pref + sorted(run_dir.rglob("*.yaml"))
        for y in cands:
            d = read_yaml_any(y)
            if isinstance(d, dict) and "model" in d and "training" in d and isinstance(d["training"], dict):
                return d
        return None

    train_cfg: Dict[str, Dict[str, Any]] = {}
    for rn in remaining:
        rd = OUTPUTS_ROOT / rn
        y = find_model_yaml(rd)
        if y:
            train_cfg[rn] = y.get("training", {}) or {}
    remaining = [rn for rn in remaining if rn in train_cfg]

    # DSU por igualdad de TRAIN (en claves compartidas, ignorando EXCLUDE_TRAIN_KEYS)
    names = remaining
    idx = {rn: i for i, rn in enumerate(names)}
    dsu = DSU(len(names))
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            if equal_on_shared_training(train_cfg[a], train_cfg[b], EXCLUDE_TRAIN_KEYS):
                dsu.union(i, j)

    comps: Dict[int, List[str]] = {}
    for rn in names:
        root = dsu.find(idx[rn]); comps.setdefault(root, []).append(rn)

    kept_after_dedup: List[str] = []
    dup_report: List[Tuple[List[str], str]] = []
    for members in comps.values():
        if len(members) == 1:
            kept_after_dedup.append(members[0]); continue
        members_sorted = sorted(members, key=lambda r: (-banded_scores_all.get(r, -np.inf), r))
        winner = members_sorted[0]
        kept_after_dedup.append(winner)
        dup_report.append((members_sorted, winner))
    final_kept = sorted(kept_after_dedup)

    # ── 3) Reporte de limpieza
    print("\n" + "—"*100)
    print("CLEANING — Excluidos por substring:")
    if excluded_by_name:
        for rn in excluded_by_name: print("  •", rn)
    else:
        print("  (none)")
    print("Claves TRAIN ignoradas (wildcards ok):", EXCLUDE_TRAIN_KEYS)

    print("\nGrupos duplicados por TRAIN cfg — se mantiene el mejor por BANDED:")
    if dup_report:
        for gi, (members_sorted, winner) in enumerate(dup_report, 1):
            print(f"  Grupo {gi}:")
            for rn in members_sorted:
                tag = "  → keep" if rn == winner else "    drop"
                bs = banded_scores_all.get(rn, float("nan"))
                print(f"    {tag}  {rn}   banded_score={bs:.3f}")
    else:
        print("  (no duplicates)")

    print("\n" + "—"*100)
    print(f"FINAL KEPT (tras limpieza)  n={len(final_kept)}")
    for rn in final_kept:
        fam = detect_family(rn); ident = identifier_token(rn)
        bs = banded_scores_all.get(rn, float('nan'))
        print(f"  • {rn}  | family={fam}  id={ident}  banded_score={bs:.3f}")

    if not final_kept:
        print("\nNo quedan modelos tras limpieza. Abortando.")
        return

    # ── 4) Tablas tras limpieza
    # Classic Top-15
    classic_scores = []
    for rn in final_kept:
        data = yaml_by_run.get(rn)
        classic_scores.append(classic_score_for_run(data) if isinstance(data, dict) else np.nan)

    df_clean = (df_meta[df_meta["run_name"].isin(final_kept)].copy()
                .assign(classic_score = classic_scores))
    df_clean = df_clean[np.isfinite(df_clean["classic_score"])].copy()
    df_clean["_ord_classic"] = df_clean["classic_score"].replace({np.nan: np.inf})
    df_clean["classic_rank"] = df_clean["_ord_classic"].rank(method="min", ascending=True)
    df_clean["z_classic"] = z_lower(df_clean["classic_score"])
    df_clean.sort_values(["_ord_classic", "run_name"], inplace=True, ignore_index=True)

    print("\n" + "="*100)
    print("Leader — NEW classic score (Top 15) AFTER CLEANING")
    print(df_clean.head(args.topk_classic)[["classic_rank","run_name","family","seed","ft_epochs","classic_score","z_classic"]].to_string(index=False))

    top15_classic = set(df_clean.head(args.topk_classic)["run_name"].tolist())

    # banded y mm(inv) sobre limpio
    banded_clean = compute_banded_view(df_clean["run_name"].tolist(), OBS_MAX, yaml_by_run, df_meta)
    banded_map = {rn: float(banded_clean.loc[banded_clean["run_name"] == rn, "banded_score"].values[0])
                  for rn in df_clean["run_name"]}
    mm_banded_inv = minmax01(pd.Series([-banded_map[rn] for rn in df_clean["run_name"]], index=df_clean.index))

    union_after_clean: set[str] = set()

    for m in METRICS4:
        s_metric_vals = []
        for rn in df_clean["run_name"]:
            y = yaml_by_run.get(rn)
            s_metric_vals.append(aggregate_metric_for_run(y, m) if isinstance(y, dict) else np.nan)
        s_metric = pd.Series(s_metric_vals, index=df_clean.index, dtype=float)
        mask = s_metric.notna()
        metric_mm = minmax01(s_metric[mask])
        blended = 0.5 * mm_banded_inv[mask].values + 0.5 * metric_mm.values  # menor=mejor

        out = (df_clean.loc[mask, ["run_name","family","seed","ft_epochs","classic_score"]]
               .assign(banded_score = df_clean.loc[mask,"run_name"].map(banded_map).values,
                       metric_agg   = s_metric[mask].values,
                       blended      = blended)
               .sort_values(["blended","run_name"], ascending=[True, True])
               .reset_index(drop=True))

        print("\n" + "="*100)
        print(f"Top-15 — BLENDED (0.5×inv-banded + 0.5×{m}) AFTER CLEANING")
        print(out.head(args.topk_metric)[["run_name","family","seed","ft_epochs","blended","banded_score","classic_score","metric_agg"]].to_string(index=False))

        union_after_clean.update(out.head(args.topk_metric)["run_name"].tolist())

    union_list_final = sorted(set(list(top15_classic) + list(union_after_clean)))
    print("\n" + "—"*100)
    print(f"UNION (Classic Top-15  ∪  4×BLENDED Top-15) — size = {len(union_list_final)}")
    print(union_list_final)

    # ── 5) Bases + imprimir combinaciones (sin seed) y conteos
    base_dirs = discover_bases(OUTPUTS_ROOT)
    if not base_dirs:
        print("\nNo se han encontrado bases (carpetas con .pt). Abortando.")
        return

    print("\n" + "—"*100)
    print("Combinations per base (NO seed) — built from UNION (prefix 'PS_'):")

    total_unique_per_base = 0
    for b in base_dirs:
        combos = sorted({f"PS_{b.name}__{detect_family(rn)}__{identifier_token(rn)}" for rn in union_list_final})
        total_unique_per_base = len(combos)  # todas las bases comparten el mismo set
        print(f"\nBase: {b.name}  |  #combos: {len(combos)}")
        for c in combos:
            print(f"  - {c}")

    print("\n" + "—"*100)
    print("Summary:")
    print(f"  Total runs escaneados: {len(all_runs)}")
    print(f"  Excluidos por substrings: {len(excluded_by_name)}")
    dup_groups_ct = sum(1 for members in comps.values() if len(members) > 1)
    print(f"  Grupos duplicados (TRAIN cfg iguales): {dup_groups_ct}")
    print(f"  Final kept tras limpieza: {len(final_kept)}")
    print(f"  Bases: {len(base_dirs)} → {[b.name for b in base_dirs]}")
    print(f"  UNION size: {len(union_list_final)}")
    print(f"  Combos por base: {total_unique_per_base}")
    print(f"  TOTAL combos (bases × union): {len(base_dirs) * total_unique_per_base}")

    # ── 6) Entrenar (orden: sweep→base)
    # interpretación común desde flowgen.yaml
    base_cfg_for_interp = load_yaml_config("flowgen.yaml")
    interp_block = {"interpretability": base_cfg_for_interp.get("interpretability", {})}

    # plan = [(run_name, base_dir), ...] con orden por sweep→base
    plan: List[Tuple[str, Path]] = []
    for rn in union_list_final:                # sweep primero
        for b in base_dirs:                    # luego todas las bases
            plan.append((rn, b))

    state = read_state()
    statuses = state.get("statuses", {})
    attempts = state.get("attempts", {})

    def assemble_config(base_dir: Path, run_name: str) -> Tuple[Dict[str, Any], str]:
        base_model = find_base_model_yaml(base_dir)
        run_training = find_training_yaml_for_run(OUTPUTS_ROOT / run_name)
        cfg = {
            "model": copy.deepcopy(base_model["model"]),
            "training": copy.deepcopy(run_training["training"]),
            "interpretability": copy.deepcopy(interp_block["interpretability"]),
        }
        # overrides globales
        deep_set(cfg, "training.finetune_num_epochs", int(args.ft_epochs))
        deep_set(cfg, "training.realism_bootstrap", int(args.realism_bootstrap))
        deep_set(cfg, "training.realism_rvr_bootstrap", int(args.realism_rvr_bootstrap))
        pt_path = find_pretrained_pt(base_dir)
        return cfg, pt_path

    def run_already_done(base_name: str) -> bool:
        rd = find_latest_run_dir(base_name)
        return has_results_yaml(rd)

    t_all = time.time()
    summary_rows: List[Dict[str, Any]] = []
    cycle = 0
    pending: List[Tuple[str, Path]] = list(plan)

    print("\n" + "="*100)
    print(f"🚀 Empezando entrenamiento — combos: {len(plan)}  (orden sweep→base)")
    try:
        while pending:
            cycle += 1
            print(f"\n🔁 Ciclo #{cycle} — pendientes: {len(pending)}")
            next_pending: List[Tuple[str, Path]] = []

            for idx, (rn, base_dir) in enumerate(pending, start=1):
                fam = detect_family(rn)
                ident = identifier_token(rn)
                out_prefix = f"PS_100_{base_dir.name}__{fam}__{ident}_seed{seed}"

                a = attempts.get(out_prefix, 0)
                if a >= args.max_retries:
                    print(f"⛔ Max retries: {out_prefix}")
                    statuses[out_prefix] = {"status": "failed", "error": "max_retries", "attempts": a}
                    continue

                if run_already_done(out_prefix):
                    statuses[out_prefix] = {"status": "ok", "error": "", "attempts": a}
                    print(f"✅ Ya finalizado: {out_prefix}")
                    continue

                # preparar config y entrenar
                try:
                    cfg, pt_path = assemble_config(base_dir, rn)
                    cfg_name, cfg_path = write_temp_config(cfg, stem=out_prefix)
                except Exception as e:
                    err = f"prep_failed: {type(e).__name__}: {e}"
                    print(f"❌ Prep falló: {out_prefix}\n    → {err}")
                    attempts[out_prefix] = a + 1
                    statuses[out_prefix] = {"status": "failed", "error": err, "attempts": attempts[out_prefix]}
                    state["attempts"] = attempts; state["statuses"] = statuses; write_state(state)
                    next_pending.append((rn, base_dir))
                    time.sleep(min(args.sleep_on_fail_sec, 30))
                    continue

                print(f"\n[{idx}/{len(pending)}] 🏃 {out_prefix}")
                t0 = time.time()
                status, error_msg = "ok", ""
                try:
                    print(out_prefix, device, seed, pt_path)

                    _ = train_flowgen_pipeline(
                        condition_col="type",
                        config_filename=cfg_name,
                        base_name=out_prefix,
                        device=device,
                        seed=seed,
                        verbose=False,
                        skip_phase1=True,
                        pretrained_path=pt_path,
                    )
                except Exception as e:
                    status, error_msg = "failed", f"{type(e).__name__}: {e}"
                    print(f"❌ Entreno falló: {out_prefix}\n    → {error_msg}")

                dt = time.time() - t0
                print(f"⏱️ Duración: {fmt_hms(dt)}")

                # limpiar cfg temporal
                try:
                    if cfg_path.exists(): cfg_path.unlink()
                except Exception:
                    pass

                success = (status == "ok") and has_results_yaml(find_latest_run_dir(out_prefix))
                attempts[out_prefix] = a + 1
                statuses[out_prefix] = {"status": "ok" if success else "failed",
                                        "error": "" if success else error_msg,
                                        "attempts": attempts[out_prefix]}
                state["attempts"] = attempts; state["statuses"] = statuses; write_state(state)

                summary_rows.append({
                    "output_name": out_prefix,
                    "base": base_dir.name,
                    "run_name": rn,
                    "status": "ok" if success else "failed",
                    "error": "" if success else error_msg,
                    "attempts": attempts[out_prefix],
                    "duration_sec": round(dt, 2),
                })

                if not success and attempts[out_prefix] < args.max_retries:
                    k = attempts[out_prefix]
                    sleep_s = min(args.sleep_on_fail_sec * (2 ** max(0, k - 1)), args.sleep_on_fail_sec * 10)
                    print(f"🕒 Backoff {sleep_s}s …")
                    time.sleep(sleep_s)
                    next_pending.append((rn, base_dir))

            pending = next_pending

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido — guardando resumen parcial…")

    # Resumen final
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_csv = SWEEP_ROOT / f"sweep_train_from_union_{ts}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    ok_ct = sum(1 for r in summary_rows if r["status"] == "ok")
    fail_ct = sum(1 for r in summary_rows if r["status"] != "ok")
    print("\n" + "—"*100)
    print(f"✅ Terminado. OK: {ok_ct}  |  Failed: {fail_ct}")
    print(f"📝 Summary CSV: {summary_csv}")
    print(f"⏲️ Elapsed total: {fmt_hms(time.time() - t_all)}")

if __name__ == "__main__":
    main()
