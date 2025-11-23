from training.train_flow_pre import build_flow_pre_model, filter_flowpre_columns, transform_to_latent_with_flowpre
import torch
from training.utils import load_yaml_config
import umap.umap_ as umap
from training.utils import ROOT_PATH
from training.utils import flowpre_log
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import cosine
import warnings
import json
from pathlib import Path
import re
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, kstest, chi2
from sklearn.decomposition import PCA
import seaborn as sns
from IPython.display import display
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr

from typing import Any, Dict, List, Tuple, Optional, Literal
import random
from models.flowgen import FlowGen
# reuse your factory (ensures arch parity with training)
from training.train_flowgen import build_flowgen_model  # same signature you shared

_HAS_UMAP = True

def densify_frames(frames: list[np.ndarray], extra_per_step: int = 3) -> list[np.ndarray]:
    """
    Insert `extra_per_step` linear-interpolated frames between every consecutive pair.
    If frames are [F0, F1, F2], output is:
      F0, (F0→F1)_1, ..., (F0→F1)_k, F1, (F1→F2)_1, ..., (F1→F2)_k, F2
    Works for 2D coords (X/XY) and 1D arrays (Y).
    """
    if len(frames) <= 1 or extra_per_step <= 0:
        return frames
    out = [frames[0]]
    for A, B in zip(frames[:-1], frames[1:]):
        for j in range(extra_per_step):
            t = (j + 1) / (extra_per_step + 1)
            out.append((1.0 - t) * A + t * B)
        out.append(B)
    return out

def _sample_real_by_scale_per_class(c: torch.Tensor, scale: float, *, seed: int | None = None) -> torch.Tensor:
    """
    Return indices that keep `scale` fraction of each class in `c`, sampled without replacement.
    If scale >= 1 → keep all. If scale <= 0 → keep none.
    """
    if scale >= 1.0:
        return torch.arange(c.numel(), device=c.device)
    if scale <= 0.0:
        return torch.empty(0, dtype=torch.long, device=c.device)

    import numpy as np
    rng = np.random.default_rng(None if seed is None else int(seed))

    idx_list = []
    for cls in torch.unique(c).tolist():
        cls_idx = torch.nonzero(c == cls, as_tuple=False).squeeze(1).cpu().numpy()
        n_keep = int(round(scale * len(cls_idx)))
        if n_keep <= 0:
            continue
        # sample without replacement; if rounding overshoots, clip to available
        n_keep = min(n_keep, len(cls_idx))
        pick = rng.choice(cls_idx, size=n_keep, replace=False)
        idx_list.append(torch.from_numpy(pick))

    if not idx_list:
        return torch.empty(0, dtype=torch.long, device=c.device)
    return torch.cat(idx_list, dim=0).to(c.device)

def _load_train_cXy(condition_col: str = "type") -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    base = Path(ROOT_PATH) / "data" / "sets" / "df_input"
    Xdf = pd.read_csv(base / "X" / "df_input_X_train.csv").sort_values("post_cleaning_index").reset_index(drop=True)
    ydf = pd.read_csv(base / "y" / "df_input_y_train.csv").sort_values("post_cleaning_index").reset_index(drop=True)
    df  = pd.merge(Xdf, ydf, on="post_cleaning_index", how="inner", validate="one_to_one")

    x_cols = [c for c in Xdf.columns if c not in ("post_cleaning_index", condition_col)]
    y_cols = [c for c in ydf.columns if c != "post_cleaning_index"]
    if len(y_cols) != 1:
        raise ValueError(f"Expected single target column in y_train.csv, got {y_cols}")
    X = df[x_cols].to_numpy(np.float32)
    y = df[[y_cols[0]]].to_numpy(np.float32)
    c = df[condition_col].to_numpy(np.int64)
    return X, y, c, x_cols, y_cols

# ───────────────────────────────────────────────────────────────────────────────
# Snapshots & configs
# ───────────────────────────────────────────────────────────────────────────────

def discover_ordered_snapshots(model_name: str) -> List[Path]:
    snap_dir = Path(ROOT_PATH) / "outputs" / "models" / "flowgen" / model_name / "snapshots"
    if not snap_dir.exists():
        raise FileNotFoundError(f"Snapshots folder not found: {snap_dir}")
    pts = [p for p in snap_dir.glob("*.pt") if "best" not in p.name.lower()]
    if not pts:
        raise FileNotFoundError(f"No snapshot .pt files (excluding 'best') in: {snap_dir}")

    def _key(p: Path) -> Tuple[int, str]:
        m = re.search(r"_epoch(\d+)_", p.name)
        return (int(m.group(1)), p.name) if m else (10**9, p.name)

    pts_sorted = sorted(pts, key=_key)
    print(f"[snapshots] {model_name}:")
    for p in pts_sorted: print("  -", p.name)
    return pts_sorted

def _device_str(pref: str = "auto") -> str:
    pref = (pref or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available(): return "cuda"
        try:
            if torch.backends.mps.is_available(): return "mps"   # type: ignore[attr-defined]
        except Exception:
            pass
        return "cpu"
    return pref

def _load_model_cfg(model_name: str) -> Dict[str, Any]:
    base = Path(ROOT_PATH) / "outputs" / "models" / "flowgen" / model_name
    for nm in (f"{model_name}.yaml", f"{model_name}.yml", "config.yaml", "config.yml"):
        p = base / nm
        if p.exists():
            import yaml
            return yaml.safe_load(p.read_text()) or {}
    return {}

def _build_model(model_name: str, x_dim: int, y_dim: int, num_classes: int, device: str) -> FlowGen:
    cfg = _load_model_cfg(model_name)
    model_cfg = cfg.get("model", cfg)
    return build_flowgen_model(model_cfg, x_dim=x_dim, y_dim=y_dim, num_classes=num_classes, device=device)

def _load_weights(model: FlowGen, ckpt: Path, device: torch.device) -> None:
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        state = sd["state_dict"]
    elif isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values()):
        state = sd
    elif hasattr(sd, "state_dict"):
        state = sd.state_dict()
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt}")
    model.load_state_dict(state, strict=False)


# ───────────────────────────────────────────────────────────────────────────────
# Sampling plan (fixed identities across all snapshots & models)
# ───────────────────────────────────────────────────────────────────────────────

def _real_counts(c: np.ndarray, class_ids: Tuple[int, ...]) -> Dict[int, int]:
    vals, cnts = np.unique(c, return_counts=True)
    mp = dict(zip(vals.tolist(), cnts.tolist()))
    return {ci: int(mp.get(ci, 0)) for ci in class_ids}

def make_sample_plan(c_real: np.ndarray, class_ids: Tuple[int, ...], scale: float) -> Dict[int, int]:
    base = _real_counts(c_real, class_ids)
    plan = {cls: int(round(base.get(cls, 0) * float(scale))) for cls in class_ids}
    print("[data] per-class real counts:", base)
    print("[data] sampling scale:", scale, "→ per-snapshot samples:", plan)
    return plan

def fixed_latents_by_class(n_by_class: Dict[int, int], dxy: int, *, seed: int = 123) -> Dict[int, torch.Tensor]:
    g = torch.Generator(device="cpu"); g.manual_seed(int(seed))
    out = {}
    for cls, n in n_by_class.items():
        out[cls] = torch.randn(n, dxy, generator=g, dtype=torch.float32, device="cpu") if n > 0 else \
                   torch.empty((0, dxy), dtype=torch.float32)
    return out

def synth_from_z_by_class(model: FlowGen, z_by_class: Dict[int, torch.Tensor], device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, Ys, Cs = [], [], []
    model.eval()
    with torch.no_grad():
        for cls, z in z_by_class.items():
            if z.numel() == 0: continue
            cls_vec = torch.full((z.shape[0],), int(cls), dtype=torch.long, device=device)
            (x_c, y_c), _ = model.inverse_xy(z.to(device), cls_vec)
            Xs.append(x_c.detach().cpu().numpy())
            Ys.append(y_c.detach().cpu().numpy())
            Cs.append(np.full((z.shape[0],), int(cls), dtype=np.int64))
    if not Xs:
        return np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32), np.zeros((0,), np.int64)
    return np.concatenate(Xs, 0), np.concatenate(Ys, 0), np.concatenate(Cs, 0)


# ───────────────────────────────────────────────────────────────────────────────
# Embedding (fit ONCE on real; real cloud stays static)
# ───────────────────────────────────────────────────────────────────────────────

class _Embedder:
    def __init__(self, n_components: int = 2):
        self.reducer = umap.UMAP(n_components=int(n_components), random_state=42, n_neighbors=30, min_dist=0.1)
    def fit(self, X: np.ndarray) -> "_Embedder":
        self.reducer.fit(X); return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.reducer.transform(X)

def fit_embedders(X_real: np.ndarray, y_real: np.ndarray) -> Tuple[_Embedder, _Embedder, np.ndarray, np.ndarray]:
    emb_x = _Embedder(n_components=2).fit(X_real)
    emb_xy = _Embedder(n_components=2).fit(np.concatenate([X_real, y_real], axis=1))
    return emb_x, emb_xy, emb_x.transform(X_real), emb_xy.transform(np.concatenate([X_real, y_real], axis=1))


# ───────────────────────────────────────────────────────────────────────────────
# Robust per-frame scaling (so syn & real share a canvas)
# ───────────────────────────────────────────────────────────────────────────────

def _robust_radius_2d(P: np.ndarray) -> float:
    if P.size == 0: return 1.0
    med = np.median(P, axis=0)
    r = np.linalg.norm(P - med, axis=1)
    return float(np.median(r) + 1e-8)

def _robust_scale_to_real(real_2d: np.ndarray, syn_2d: np.ndarray) -> float:
    rs = _robust_radius_2d(real_2d)
    ss = _robust_radius_2d(syn_2d)
    return 1.0 if ss <= 1e-8 else (rs / ss)

def _apply_iso_scale(P: np.ndarray, s: float) -> np.ndarray:
    if P.size == 0: return P
    center = np.median(P, axis=0)
    return center + s * (P - center)

def scale_frames_to_real(real_2d: np.ndarray, frames_2d: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
    scaled, scales = [], []
    for F in frames_2d:
        s = _robust_scale_to_real(real_2d, F) if F.size else 1.0
        scales.append(float(s))
        scaled.append(_apply_iso_scale(F, s))
    return scaled, scales

def scale_y_frames_to_real(real_y: np.ndarray, frames_y: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
    q25, q75 = np.quantile(real_y[:, 0], 0.25), np.quantile(real_y[:, 0], 0.75)
    real_iqr = max(q75 - q25, 1e-8)
    scaled, scales = [], []
    for Y in frames_y:
        if Y.size == 0:
            scaled.append(Y); scales.append(1.0); continue
        q25s, q75s = np.quantile(Y[:, 0], 0.25), np.quantile(Y[:, 0], 0.75)
        syn_iqr = max(q75s - q25s, 1e-8)
        s = real_iqr / syn_iqr
        y_med = np.median(Y[:, 0])
        Y_sc = (Y[:, 0] - y_med) * s + y_med
        scaled.append(Y_sc.reshape(-1, 1).astype(np.float32))
        scales.append(float(s))
    return scaled, scales


# ───────────────────────────────────────────────────────────────────────────────
# Build frames for ONE model (then we’ll concat two models)
# ───────────────────────────────────────────────────────────────────────────────

def _last_snapshot(model_name: str) -> Path:
    snaps = discover_ordered_snapshots(model_name)
    return snaps[-1]

def latent_seeds_from_final(
    reference_model: str,
    X_sel: np.ndarray,
    y_sel: np.ndarray,
    c_sel: np.ndarray,
    device_pref: str = "auto",
) -> torch.Tensor:
    """
    Use the LAST .pt of `reference_model` to map real (X_sel, y_sel, c_sel) → latent seeds z*.
    Returns z seeds on CPU with shape (N, Dx+Dy). Order matches inputs.
    """
    dev_str = _device_str(device_pref)
    device = torch.device("cuda" if dev_str == "cuda" and torch.cuda.is_available()
                          else ("mps" if dev_str == "mps" else "cpu"))
    Dx, Dy = X_sel.shape[1], y_sel.shape[1]
    model = _build_model(reference_model, x_dim=Dx, y_dim=Dy, num_classes=int(c_sel.max()) + 1, device=str(device))
    last_ckpt = _last_snapshot(reference_model)
    _load_weights(model, last_ckpt, device)

    x_t = torch.from_numpy(X_sel).to(device)
    y_t = torch.from_numpy(y_sel).to(device)
    c_t = torch.from_numpy(c_sel).long().to(device)

    model.eval()
    with torch.no_grad():
        # If your API expects concatenated [x|y], change accordingly:
        # z_t, _ = model.forward_xy(torch.cat([x_t, y_t], dim=1), c_t)
        z_t, _ = model.forward_xy(x_t, y_t, c_t)
    return z_t.detach().cpu()

def random_latent_seeds(
    N: int,
    dxy: int,
    seed: int = 123,
) -> torch.Tensor:
    """
    Draw one set of standard-normal latent seeds z ~ N(0, I) for ALL points (N, dxy).
    Deterministic via `seed`. Returned on CPU.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randn(N, dxy, generator=g, dtype=torch.float32, device="cpu")

def _per_pair_avg_move_2d(frames: List[np.ndarray], mask: Optional[np.ndarray] = None) -> List[float]:
    """Mean pointwise Euclidean movement between consecutive 2D frames."""
    if len(frames) < 2: return []
    avgs = []
    for A, B in zip(frames[:-1], frames[1:]):
        if A.size == 0 or B.size == 0:
            avgs.append(0.0); continue
        if mask is not None:
            A = A[mask]
            B = B[mask]
        d = np.linalg.norm(B - A, axis=1)
        avgs.append(float(np.mean(d)))
    return avgs

def _per_pair_avg_move_hd(frames_hd: List[np.ndarray], mask: Optional[np.ndarray] = None) -> List[float]:
    """Mean pointwise Euclidean movement between consecutive high-dim XY frames."""
    if len(frames_hd) < 2: return []
    avgs = []
    for A, B in zip(frames_hd[:-1], frames_hd[1:]):
        if A.size == 0 or B.size == 0:
            avgs.append(0.0); continue
        if mask is not None:
            A = A[mask]; B = B[mask]
        d = np.linalg.norm(B - A, axis=1)
        avgs.append(float(np.mean(d)))
    return avgs

def _densify_adaptive_2d(frames: List[np.ndarray], policy: Optional[List[float]]) -> List[np.ndarray]:
    """
    Adaptive interpolation for 2D frames.
    policy = [scale, min_clamp, max_clamp]; steps_i = clamp(round(scale * (avg_i / median_avg)))
    """
    if len(frames) <= 1 or not policy:
        return frames
    scale, min_c, max_c = float(policy[0]), int(policy[1]), int(policy[2])
    # robust reference
    avgs = _per_pair_avg_move_2d(frames)
    ref = np.median(avgs) if avgs else 0.0
    out = [frames[0]]
    for i, (A, B) in enumerate(zip(frames[:-1], frames[1:])):
        avg = avgs[i] if avgs else 0.0
        steps = 0 if ref <= 1e-12 else int(np.clip(round(scale * (avg / (ref + 1e-12))), min_c, max_c))
        for j in range(steps):
            t = (j + 1) / (steps + 1)
            out.append((1.0 - t) * A + t * B)
        out.append(B)
    return out

def _densify_adaptive_1d(frames_1d: List[np.ndarray], policy: Optional[List[float]]) -> List[np.ndarray]:
    """
    Adaptive interpolation for 1D arrays (N,1) — used by Y histogram.
    Uses mean |Δ| per pair with same policy.
    """
    if len(frames_1d) <= 1 or not policy:
        return frames_1d
    # compute mean |Δ| per pair
    avgs = []
    for A, B in zip(frames_1d[:-1], frames_1d[1:]):
        if A.size == 0 or B.size == 0:
            avgs.append(0.0); continue
        d = np.abs(B[:, 0] - A[:, 0])
        avgs.append(float(np.mean(d)))
    ref = np.median(avgs) if avgs else 0.0
    scale, min_c, max_c = float(policy[0]), int(policy[1]), int(policy[2])
    out = [frames_1d[0]]
    for i, (A, B) in enumerate(zip(frames_1d[:-1], frames_1d[1:])):
        avg = avgs[i] if avgs else 0.0
        steps = 0 if ref <= 1e-12 else int(np.clip(round(scale * (avg / (ref + 1e-12))), min_c, max_c))
        for j in range(steps):
            t = (j + 1) / (steps + 1)
            out.append(((1.0 - t) * A + t * B))
        out.append(B)
    return out

def _flatten_point_deltas_2d(frames: List[np.ndarray], mask: Optional[np.ndarray] = None) -> np.ndarray:
    """All per-point Euclidean deltas across consecutive 2D frames → (M,)"""
    if len(frames) < 2: return np.zeros((0,), np.float32)
    all_d = []
    for A, B in zip(frames[:-1], frames[1:]):
        if A.size == 0 or B.size == 0:
            continue
        if mask is not None:
            A = A[mask]; B = B[mask]
        d = np.linalg.norm(B - A, axis=1)
        all_d.append(d)
    return np.concatenate(all_d, axis=0) if all_d else np.zeros((0,), np.float32)

def _flatten_point_deltas_hd(frames_hd: List[np.ndarray], mask: Optional[np.ndarray] = None) -> np.ndarray:
    """All per-point Euclidean deltas across consecutive high-dim frames → (M,)"""
    if len(frames_hd) < 2: return np.zeros((0,), np.float32)
    all_d = []
    for A, B in zip(frames_hd[:-1], frames_hd[1:]):
        if A.size == 0 or B.size == 0:
            continue
        if mask is not None:
            A = A[mask]; B = B[mask]
        d = np.linalg.norm(B - A, axis=1)
        all_d.append(d)
    return np.concatenate(all_d, axis=0) if all_d else np.zeros((0,), np.float32)

def _summarize_deltas(d: np.ndarray) -> Dict[str, float]:
    if d.size == 0:
        return {k: 0.0 for k in ["min","max","mean","median","p01","p05","p95","p99"]}
    return {
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "p01": float(np.percentile(d, 1)),
        "p05": float(np.percentile(d, 5)),
        "p95": float(np.percentile(d, 95)),
        "p99": float(np.percentile(d, 99)),
    }

def _format_stats_lines(tag: str, stats: Dict[str, float]) -> str:
    return (f"{tag} — mean {stats['mean']:.4f}, med {stats['median']:.4f}, "
            f"min {stats['min']:.4f}, p1 {stats['p01']:.4f}, p5 {stats['p05']:.4f}, "
            f"p95 {stats['p95']:.4f}, p99 {stats['p99']:.4f}, max {stats['max']:.4f}")

def _steps_per_pair_2d(frames_2d: List[np.ndarray], policy: Optional[List[float]]) -> List[int]:
    """Return number of interp steps for each consecutive pair, using the same logic as _densify_adaptive_2d."""
    if len(frames_2d) <= 1 or not policy:
        return [0] * max(0, len(frames_2d) - 1)
    scale, min_c, max_c = float(policy[0]), int(policy[1]), int(policy[2])
    avgs = _per_pair_avg_move_2d(frames_2d)
    ref = np.median(avgs) if avgs else 0.0
    steps = []
    for avg in avgs:
        s = 0 if ref <= 1e-12 else int(np.clip(round(scale * (avg / (ref + 1e-12))), min_c, max_c))
        steps.append(s)
    return steps

def _densify_hd_with_steps(frames_hd: List[np.ndarray], steps_per_pair: List[int]) -> List[np.ndarray]:
    """Densify high-dim frames to match a given steps-per-pair schedule."""
    if len(frames_hd) <= 1:
        return frames_hd
    out = [frames_hd[0]]
    for (A, B), k in zip(zip(frames_hd[:-1], frames_hd[1:]), steps_per_pair):
        for j in range(k):
            t = (j + 1) / (k + 1)
            out.append((1.0 - t) * A + t * B)
        out.append(B)
    return out

def frames_for_model(
    model_name: str,
    *,
    X_real: np.ndarray,
    y_real: np.ndarray,
    c_real: np.ndarray,
    emb_x: Optional[_Embedder],
    emb_xy: Optional[_Embedder],
    z_seeds_cpu: torch.Tensor,     # (N, Dx+Dy) fixed seeds on CPU
    c_seeds: np.ndarray,           # (N,) classes matching seeds order
    device_pref: str = "auto",
    seed_init: Optional[int] = None,
) -> Dict[str, Any]:
    snaps = discover_ordered_snapshots(model_name)
    dev_str = _device_str(device_pref)
    device = torch.device("cuda" if dev_str == "cuda" and torch.cuda.is_available()
                          else ("mps" if dev_str == "mps" else "cpu"))
    Dx, Dy = X_real.shape[1], y_real.shape[1]

    # Seed & build INIT model (untrained)
    if seed_init is not None:
        torch.manual_seed(int(seed_init))
        np.random.seed(int(seed_init) & 0xFFFFFFFF)
        random.seed(int(seed_init))
    model = _build_model(model_name, x_dim=Dx, y_dim=Dy, num_classes=int(c_real.max()) + 1, device=str(device))

    # Fixed identities across all frames
    z_fix = z_seeds_cpu.to(device)  # (N, Dx+Dy)
    c_fix = torch.as_tensor(c_seeds, dtype=torch.long, device=device)

    frames_X_raw, frames_XY_raw, frames_y_raw = [], [], []
    frames_XY_hd_raw = []
    frames_c: np.ndarray = c_seeds.copy()

    def _gen_one_frame() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            (x_c, y_c), _ = model.inverse_xy(z_fix, c_fix)
        Xs = x_c.detach().cpu().numpy()
        Ys = y_c.detach().cpu().numpy()
        XY = np.concatenate([Xs, Ys], axis=1)
        return Xs, Ys, XY

    # INIT frame
    X0, Y0, XY0 = _gen_one_frame()
    frames_XY_hd_raw.append(XY0)
    if emb_x is not None:
        frames_X_raw.append(emb_x.transform(X0) if X0.size else np.zeros((0, 2), np.float32))
    if emb_xy is not None:
        frames_XY_raw.append(emb_xy.transform(XY0) if XY0.size else np.zeros((0, 2), np.float32))
    frames_y_raw.append(Y0.copy())

    # Trained snapshots
    for ckpt in snaps:
        _load_weights(model, ckpt, device)
        Xs, Ys, XY = _gen_one_frame()
        frames_XY_hd_raw.append(XY)
        if emb_x is not None:
            frames_X_raw.append(emb_x.transform(Xs) if Xs.size else np.zeros((0, 2), np.float32))
        if emb_xy is not None:
            frames_XY_raw.append(emb_xy.transform(XY) if XY.size else np.zeros((0, 2), np.float32))
        frames_y_raw.append(Ys.copy())

    # Overall reals (if embedders provided)
    X_real_2d = emb_x.transform(X_real) if emb_x is not None else None
    XY_real_2d = emb_xy.transform(np.concatenate([X_real, y_real], axis=1)) if emb_xy is not None else None

    # scales (left free)
    sX  = [1.0] * len(frames_X_raw)
    sXY = [1.0] * len(frames_XY_raw)
    sY  = [1.0] * len(frames_y_raw)

    init_tag = Path(f"__INIT_seed{seed_init if seed_init is not None else 0}__")
    snaps_with_init = [init_tag] + snaps

    return dict(
        snapshots=snaps_with_init,
        C=frames_c,
        X_real_2d=X_real_2d, XY_real_2d=XY_real_2d, y_real=y_real,
        frames_X=frames_X_raw,  frames_X_scale=sX,
        frames_XY=frames_XY_raw, frames_XY_scale=sXY,
        frames_y=frames_y_raw,  frames_y_scale=sY,
        frames_XY_hd=frames_XY_hd_raw,
    )

def _drop_init_frame_inplace(fr: Dict[str, Any]) -> None:
    """Remove the leading __INIT_* frame from a frames_for_model() payload (snapshots + all frame lists)."""
    snaps = fr.get("snapshots", [])
    if not snaps:
        return
    first_name = snaps[0].name if hasattr(snaps[0], "name") else str(snaps[0])
    if first_name.startswith("__INIT_"):
        fr["snapshots"] = snaps[1:]
        for k in ("frames_X", "frames_XY", "frames_y", "frames_XY_hd",
                  "frames_X_scale", "frames_XY_scale", "frames_y_scale"):
            if k in fr and isinstance(fr[k], list) and fr[k]:
                fr[k] = fr[k][1:]



# ───────────────────────────────────────────────────────────────────────────────
# Plotly helpers (slider-only scrubber). Key fix: frames include real+synthetic.
# ───────────────────────────────────────────────────────────────────────────────

_COLORS = {0: "#377eb8", 1: "#e41a1c", 2: "#4daf4a"}  # classes 0/1/2

def _axis_off(fig: go.Figure):
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, visible=False)
    fig.update_layout(margin=dict(l=10, r=110, t=40, b=10))

def _scale_annot(scale: float) -> Dict[str, Any]:
    return dict(x=1.02, y=0.5, xref="paper", yref="paper",
                text=f"<b>scale × {scale:.3f}</b>", showarrow=False, align="left")

def _static_real_traces_2d(coords: np.ndarray, c: np.ndarray, class_ids: Tuple[int, ...]) -> List[go.Scatter]:
    out = []
    for cls in class_ids:
        m = (c == cls)
        out.append(go.Scatter(
            x=(coords[m, 0] if m.any() else []),
            y=(coords[m, 1] if m.any() else []),
            mode="markers",
            name=f"real c={cls}",
            marker=dict(size=6, symbol="circle", color=_COLORS.get(cls, "gray"), opacity=0.28),
            hoverinfo="skip",
        ))
    return out


def _synthetic_traces_2d(coords: np.ndarray, c: np.ndarray, class_ids: Tuple[int, ...]) -> List[go.Scatter]:
    out = []
    for cls in class_ids:
        m = (c == cls)
        out.append(go.Scatter(
            x=(coords[m, 0] if coords.size else []),
            y=(coords[m, 1] if coords.size else []),
            mode="markers",
            name=f"syn c={cls}",
            marker=dict(size=6, symbol="x", color=_COLORS.get(cls, "gray"), opacity=0.95),
            hoverinfo="skip",
        ))
    return out


def _static_real_hist_traces(y: np.ndarray, c: np.ndarray, class_ids: Tuple[int, ...],
                             bins: Tuple[float, float, float]) -> List[go.Histogram]:
    out = []
    for cls in class_ids:
        m = (c == cls)
        out.append(go.Histogram(
            x=(y[m, 0] if m.any() else []),
            name=f"real c={cls}",
            marker_color=_COLORS.get(cls, "gray"),
            opacity=0.25,
            xbins=dict(start=bins[0], end=bins[1], size=bins[2]),
            autobinx=False,
            histnorm="",
        ))
    return out

def _synthetic_hist_traces(y: np.ndarray, c: np.ndarray, class_ids: Tuple[int, ...],
                           bins: Tuple[float, float, float]) -> List[go.Histogram]:
    out = []
    for cls in class_ids:
        m = (c == cls)
        out.append(go.Histogram(
            x=(y[m, 0] if y.size else []),
            name=f"syn c={cls}",
            marker_color=_COLORS.get(cls, "gray"),
            opacity=0.85,
            xbins=dict(start=bins[0], end=bins[1], size=bins[2]),
            autobinx=False,
            histnorm="",
        ))
    return out

def make_scrubber_X_or_XY(
    *, title: str, real_2d: np.ndarray, real_c: np.ndarray,
    frames_2d: List[np.ndarray], frames_c: np.ndarray, scales: List[float],
    frame_labels: List[str], class_ids: Tuple[int, ...] = (0,1,2),
    legend_notes: Optional[List[str]] = None,
) -> go.Figure:
    real_tr = _static_real_traces_2d(real_2d, real_c, class_ids)
    syn_tr0 = _synthetic_traces_2d(frames_2d[0], frames_c, class_ids) if frames_2d else _synthetic_traces_2d(np.zeros((0,2)), frames_c, class_ids)
    data0 = real_tr + syn_tr0
    if legend_notes:
        for note in legend_notes:
            data0.append(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=0.1, opacity=0.0),
                                    name=note, showlegend=True, hoverinfo="skip"))

    n_real = len(class_ids)
    n_syn  = len(class_ids)
    syn_trace_indices = list(range(n_real, n_real + n_syn))

    frames = []
    for i, F in enumerate(frames_2d):
        syn_only = _synthetic_traces_2d(F, frames_c, class_ids)
        ann = [_scale_annot(scales[i])]
        if legend_notes and i < len(legend_notes):
            ann.append(_stats_annot(legend_notes[i]))
        frames.append(go.Frame(
            data=syn_only, name=f"f{i}", traces=syn_trace_indices,
            layout=go.Layout(annotations=ann)
        ))

    # --- NEW: ensure labels match number of frames
    if (not frame_labels) or (len(frame_labels) != len(frames)):
        frame_labels = [f"{i+1}" for i in range(len(frames))]

    steps = [{
        "method": "animate",
        "label": frame_labels[i],
        "args": [[f"f{i}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
    } for i in range(len(frames))]

    fig = go.Figure(data=data0, frames=frames)
    init_ann = [_scale_annot(scales[0] if scales else 1.0)]
    if legend_notes and len(legend_notes) > 0:
        init_ann.append(_stats_annot(legend_notes[0]))

    fig.update_layout(
        title=title,
        sliders=[{
            "active": 0, "y": -0.06, "len": 0.9,
            "currentvalue": {"prefix": "frame: ", "visible": True},
            "steps": steps,
        }],
        annotations=init_ann,
        showlegend=True,
        updatemenus=[],  # slider-only
    )

    _axis_off(fig)
    return fig




def make_scrubber_hist_y(
    *, title: str, real_y: np.ndarray, real_c: np.ndarray,
    frames_y: List[np.ndarray], frames_c: np.ndarray, scales: List[float],
    frame_labels: List[str], class_ids: Tuple[int, ...] = (0,1,2), n_bins: int = 80,
) -> go.Figure:
    y_min, y_max = float(np.min(real_y)), float(np.max(real_y))
    if y_max <= y_min: y_max = y_min + 1.0
    bin_size = (y_max - y_min) / max(5, int(n_bins))
    bins = (y_min, y_max, bin_size)

    real_hist = _static_real_hist_traces(real_y, real_c, class_ids, bins)
    syn_hist0 = _synthetic_hist_traces(frames_y[0], frames_c, class_ids, bins) if frames_y else _synthetic_hist_traces(np.zeros((0,1)), frames_c, class_ids, bins)
    data0 = real_hist + syn_hist0

    n_real = len(class_ids)
    n_syn  = len(class_ids)
    syn_trace_indices = list(range(n_real, n_real + n_syn))

    frames = []
    for i, Y in enumerate(frames_y):
        syn_only = _synthetic_hist_traces(Y, frames_c, class_ids, bins)
        frames.append(go.Frame(
            data=syn_only, name=f"fy{i}", traces=syn_trace_indices,
            layout=go.Layout(annotations=[_scale_annot(scales[i])])
        ))

    # --- NEW: ensure labels match number of frames
    if (not frame_labels) or (len(frame_labels) != len(frames)):
        frame_labels = [f"{i+1}" for i in range(len(frames))]

    steps = [{
        "method": "animate",
        "label": frame_labels[i],
        "args": [[f"fy{i}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
    } for i in range(len(frames))]

    fig = go.Figure(data=data0, frames=frames)
    fig.update_layout(
        title=title,
        barmode="overlay",
        sliders=[{
            "active": 0, "y": -0.06, "len": 0.9,
            "currentvalue": {"prefix": "frame: ", "visible": True},
            "steps": steps,
        }],
        annotations=[_scale_annot(scales[0] if scales else 1.0)],
        showlegend=True,
    )
    return fig

# ── Legend / distance helpers ──────────────────────────────────────────────────

def _stats_annot(text: str) -> Dict[str, Any]:
    # right side (below the scale annotation), multiline HTML ok
    return dict(
        x=1.02, y=0.05, xref="paper", yref="paper",
        xanchor="left", yanchor="bottom",
        text=text, showarrow=False, align="left",
        font=dict(size=11)
    )

def _percentiles(v: np.ndarray, ps=(1, 5, 50, 95, 99)) -> Dict[int, float]:
    return {p: float(np.percentile(v, p)) for p in ps}

def _dist_stats(v: np.ndarray) -> Dict[str, float]:
    if v.size == 0:
        return dict(min=0.0, max=0.0, mean=0.0, median=0.0, p01=0.0, p05=0.0, p95=0.0, p99=0.0)
    qs = _percentiles(v, (1, 5, 50, 95, 99))
    return dict(
        min=float(v.min()), max=float(v.max()),
        mean=float(v.mean()), median=float(qs[50]),
        p01=float(qs[1]), p05=float(qs[5]), p95=float(qs[95]), p99=float(qs[99])
    )

def _fmt_stats(prefix: str, s: Dict[str, float]) -> str:
    # compact multiline; tuned for ~80 chars wide annotation
    return (
        f"<b>{prefix}</b> min {s['min']:.3f} | max {s['max']:.3f}<br>"
        f"mean {s['mean']:.3f} | med {s['median']:.3f}<br>"
        f"p01 {s['p01']:.3f} | p05 {s['p05']:.3f} | p95 {s['p95']:.3f} | p99 {s['p99']:.3f}"
    )

def build_legend_notes(
    frames_2d: List[np.ndarray],
    frames_hd: Optional[List[np.ndarray]] = None,
    hd_slice: Optional[slice] = None,
    mask: Optional[np.ndarray] = None,
    label_2d: str = "Δ2D",
    label_hd: str = "ΔHD",
) -> List[str]:
    """
    Per frame, compute movement stats vs previous frame:
    - always uses 2D (UMAP coords) distances
    - optionally also high-dim distances (frames_hd[:, hd_slice])
    - mask can restrict to a class for per_class view
    Returns a list[str] with same length as frames_2d.
    """
    T = len(frames_2d)
    notes: List[str] = []
    use_hd = frames_hd is not None and len(frames_hd) == T

    for t in range(T):
        if t == 0:
            notes.append(f"<b>{label_2d}</b> —")  # first frame has no delta
            continue

        A2, B2 = frames_2d[t-1], frames_2d[t]
        if mask is not None:
            A2 = A2[mask]; B2 = B2[mask]
        d2 = np.linalg.norm(B2 - A2, axis=1) if A2.size else np.array([], dtype=float)
        s2 = _dist_stats(d2)
        text = _fmt_stats(label_2d, s2)

        if use_hd:
            Ah, Bh = frames_hd[t-1], frames_hd[t]
            if hd_slice is not None:
                Ah = Ah[:, hd_slice]; Bh = Bh[:, hd_slice]
            if mask is not None:
                Ah = Ah[mask]; Bh = Bh[mask]
            dh = np.linalg.norm(Bh - Ah, axis=1) if Ah.size else np.array([], dtype=float)
            sh = _dist_stats(dh)
            text = text + "<br>" + _fmt_stats(label_hd, sh)

        notes.append(text)

    return notes


def generate_flowgen_evolution_figs_combined(
    *,
    standard_model: str,
    realism_model: str,
    condition_col: str = "type",
    class_ids: Tuple[int, ...] = (0, 1, 2),
    scale: float = 0.05,
    device: str = "auto",
    seed: int = 123,                        # seeds INIT and random latents (if chosen) and real-subset sampling
    interp_policy: Optional[List[float]] = None,  # ← [scale, min_clamp, max_clamp], e.g., [1.5, 0, 30]
    cluster_features: int | None = None,
    y_bins: int = 80,
    view: Literal["overall", "per_class"] = "overall",
    latent_source: Literal["original", "random"] = "original",
    latent_reference_model: Optional[str] = None,
) -> Tuple[Dict[str, go.Figure], Dict[str, Any]]:

    # 1) Real data and a SINGLE per-class subset (for plotting & seeding)
    X_real, y_real, c_real, x_cols, y_cols = _load_train_cXy(condition_col=condition_col)
    keep_idx_t = _sample_real_by_scale_per_class(torch.as_tensor(c_real), scale, seed=seed)
    keep_idx = keep_idx_t.cpu().numpy()
    Xr_plot, yr_plot, cr_plot = X_real[keep_idx], y_real[keep_idx], c_real[keep_idx]

    Dx, Dy = Xr_plot.shape[1], yr_plot.shape[1]
    Dxy = Dx + Dy
    N   = Xr_plot.shape[0]

    # 2) Fixed latent seeds ONCE (and reuse everywhere)
    ref_model = latent_reference_model or realism_model
    if latent_source == "original":
        # Seeds are the latents of THESE exact real points under the LAST snapshot → 1:1 matching
        z_seeds_cpu = latent_seeds_from_final(
            reference_model=ref_model,
            X_sel=Xr_plot, y_sel=yr_plot, c_sel=cr_plot, device_pref=device
        )
        if z_seeds_cpu.shape != (N, Dxy):
            raise ValueError(f"latent_seeds_from_final returned {tuple(z_seeds_cpu.shape)}; expected {(N, Dxy)}")
        c_seeds = cr_plot.copy()
    else:
        z_seeds_cpu = random_latent_seeds(N=N, dxy=Dxy, seed=seed)
        c_seeds = cr_plot.copy()

    # 3) Overall embedders only for view="overall"
    emb_x_overall = None
    emb_xy_overall = None
    if view == "overall":
        emb_x_overall, emb_xy_overall, _, _ = fit_embedders(Xr_plot, yr_plot)

    # 4) Build frames for both models with SAME seeds; prepend INIT (seeded)
    std = frames_for_model(
        standard_model,
        X_real=Xr_plot, y_real=yr_plot, c_real=cr_plot,
        emb_x=emb_x_overall, emb_xy=emb_xy_overall,
        z_seeds_cpu=z_seeds_cpu, c_seeds=c_seeds, device_pref=device, seed_init=seed
    )
    rlm = frames_for_model(
        realism_model,
        X_real=Xr_plot, y_real=yr_plot, c_real=cr_plot,
        emb_x=emb_x_overall, emb_xy=emb_xy_overall,
        z_seeds_cpu=z_seeds_cpu, c_seeds=c_seeds, device_pref=device, seed_init=seed
    )

    # Keep INIT only for the standard stream; drop it from realism
    _drop_init_frame_inplace(rlm)

    # 5) Concatenate INIT+snapshots (std → rlm)
    frames_X_overall_raw  = (std["frames_X"]  + rlm["frames_X"])   if view == "overall" else []
    frames_XY_overall_raw = (std["frames_XY"] + rlm["frames_XY"])  if view == "overall" else []
    frames_XY_hd_all      = std["frames_XY_hd"] + rlm["frames_XY_hd"]   # always available (INIT+all)

    # labels
    frame_labels = [f"STD {i+1}" for i in range(len(std["frames_XY_hd"]))] + \
                   [f"RL  {i+1}" for i in range(len(rlm["frames_XY_hd"]))]

    print("\n[snapshot order] standard:", [p.name for p in std["snapshots"]])
    print("[snapshot order] realism :",  [p.name for p in rlm["snapshots"]])

    figs: Dict[str, go.Figure] = {}

    # ───────────── VIEW = OVERALL ─────────────
    if view == "overall":
        # Legend stats computed on raw (non-densified) 2D UMAP deltas and HD deltas
        deltas2d_X  = _flatten_point_deltas_2d(frames_X_overall_raw)
        deltas2d_XY = _flatten_point_deltas_2d(frames_XY_overall_raw)
        deltas_hd   = _flatten_point_deltas_hd(frames_XY_hd_all)

        stats_X_2d   = _summarize_deltas(deltas2d_X)
        stats_XY_2d  = _summarize_deltas(deltas2d_XY)
        stats_HD     = _summarize_deltas(deltas_hd)

        notes_X  = [_format_stats_lines("Δ2D (X UMAP)",  stats_X_2d),
                    _format_stats_lines("ΔHD (X+Y space)", stats_HD)]
        notes_XY = [_format_stats_lines("Δ2D (X+Y UMAP)", stats_XY_2d),
                    _format_stats_lines("ΔHD (X+Y space)", stats_HD)]

        # Adaptive interpolation
        frames_X  = _densify_adaptive_2d(frames_X_overall_raw,  interp_policy)
        frames_XY = _densify_adaptive_2d(frames_XY_overall_raw, interp_policy)
        frames_y  = _densify_adaptive_1d([F[:, Dx:Dx+Dy] for F in frames_XY_hd_all], interp_policy)

        # scales fixed to 1
        scales_X  = [1.0] * len(frames_X)
        scales_XY = [1.0] * len(frames_XY)
        scales_y  = [1.0] * len(frames_y)

        # simple numeric labels after densify
        frame_labels_out = [f"{i+1}" for i in range(len(frames_X))]

        C_frames_all = std["C"]

        figs.update({
            "X": make_scrubber_X_or_XY(
                title=f"FlowGen evolution — X only (UMAP) — {standard_model} → {realism_model}",
                real_2d=std["X_real_2d"], real_c=cr_plot,
                frames_2d=frames_X, frames_c=C_frames_all,
                scales=scales_X, frame_labels=frame_labels_out, class_ids=class_ids,
                legend_notes=notes_X
            ),
            "XY": make_scrubber_X_or_XY(
                title=f"FlowGen evolution — X+Y (UMAP) — {standard_model} → {realism_model}",
                real_2d=std["XY_real_2d"], real_c=cr_plot,
                frames_2d=frames_XY, frames_c=C_frames_all,
                scales=scales_XY, frame_labels=frame_labels_out, class_ids=class_ids,
                legend_notes=notes_XY
            ),
            "Y": make_scrubber_hist_y(
                title=f"FlowGen evolution — Y histogram — {standard_model} → {realism_model}",
                real_y=yr_plot, real_c=cr_plot,
                frames_y=frames_y, frames_c=C_frames_all,
                scales=scales_y, frame_labels=frame_labels_out,
                class_ids=class_ids, n_bins=y_bins
            ),
        })

    # ─────────── VIEW = PER_CLASS ────────────
    else:
        for cls in class_ids:
            m = (cr_plot == cls)
            if not np.any(m):
                continue

            # Per-class UMAPs trained on REAL of that class
            emb_x_c  = _Embedder(n_components=2).fit(Xr_plot[m])
            emb_xy_c = _Embedder(n_components=2).fit(np.concatenate([Xr_plot[m], yr_plot[m]], axis=1))
            real_x2d_c  = emb_x_c.transform(Xr_plot[m])
            real_xy2d_c = emb_xy_c.transform(np.concatenate([Xr_plot[m], yr_plot[m]], axis=1))

            # Build per-class 2D frames (RAW, then densify)
            frames_X_c_raw, frames_XY_c_raw, frames_y_c_raw = [], [], []
            for XY in frames_XY_hd_all:
                Xf = XY[m, :Dx]
                Yf = XY[m, Dx:Dx+Dy]
                frames_X_c_raw.append(emb_x_c.transform(Xf) if Xf.size else np.zeros((0, 2), np.float32))
                frames_XY_c_raw.append(emb_xy_c.transform(np.concatenate([Xf, Yf], axis=1)) if Xf.size else np.zeros((0, 2), np.float32))
                frames_y_c_raw.append(Yf.copy())

            # Legend stats (per-class)
            deltas2d_X_c  = _flatten_point_deltas_2d(frames_X_c_raw)
            deltas2d_XY_c = _flatten_point_deltas_2d(frames_XY_c_raw)
            deltas_hd_c   = _flatten_point_deltas_hd([F[m] for F in frames_XY_hd_all], mask=None)  # already masked above

            stats_X_2d_c  = _summarize_deltas(deltas2d_X_c)
            stats_XY_2d_c = _summarize_deltas(deltas2d_XY_c)
            stats_HD_c    = _summarize_deltas(deltas_hd_c)

            notes_X_c  = [_format_stats_lines("Δ2D (X UMAP)",  stats_X_2d_c),
                          _format_stats_lines("ΔHD (X+Y space)", stats_HD_c)]
            notes_XY_c = [_format_stats_lines("Δ2D (X+Y UMAP)", stats_XY_2d_c),
                          _format_stats_lines("ΔHD (X+Y space)",  stats_HD_c)]

            # Adaptive densify
            frames_X_c  = _densify_adaptive_2d(frames_X_c_raw,  interp_policy)
            frames_XY_c = _densify_adaptive_2d(frames_XY_c_raw, interp_policy)
            frames_y_c  = _densify_adaptive_1d(frames_y_c_raw,  interp_policy)

            scales_X_c  = [1.0] * len(frames_X_c)
            scales_XY_c = [1.0] * len(frames_XY_c)
            scales_y_c  = [1.0] * len(frames_y_c)
            labels_X_c = [f"{i + 1}" for i in range(len(frames_X_c))]
            labels_XY_c = [f"{i + 1}" for i in range(len(frames_XY_c))]
            labels_Y_c = [f"{i + 1}" for i in range(len(frames_y_c))]
            C_frames_c = cr_plot[m]

            figs.update({
                f"X_c{cls}": make_scrubber_X_or_XY(
                    title=f"[Class {cls}] FlowGen evolution — X only (UMAP) — {standard_model} → {realism_model}",
                    real_2d=real_x2d_c, real_c=C_frames_c,
                    frames_2d=frames_X_c, frames_c=C_frames_c,
                    scales=scales_X_c, frame_labels=labels_X_c, class_ids=(cls,),
                    legend_notes=notes_X_c
                ),
                f"XY_c{cls}": make_scrubber_X_or_XY(
                    title=f"[Class {cls}] FlowGen evolution — X+Y (UMAP) — {standard_model} → {realism_model}",
                    real_2d=real_xy2d_c, real_c=C_frames_c,
                    frames_2d=frames_XY_c, frames_c=C_frames_c,
                    scales=scales_XY_c, frame_labels=labels_XY_c, class_ids=(cls,),
                    legend_notes=notes_XY_c
                ),
                f"Y_c{cls}": make_scrubber_hist_y(
                    title=f"[Class {cls}] FlowGen evolution — Y histogram — {standard_model} → {realism_model}",
                    real_y=yr_plot[m], real_c=C_frames_c,
                    frames_y=frames_y_c, frames_c=C_frames_c,
                    scales=scales_y_c, frame_labels=labels_Y_c, class_ids=(cls,), n_bins=y_bins
                ),
            })

    # 6) Optionally: chunk UMAPs (reuse SAME seeds; adaptive densify; overall)
    if cluster_features and cluster_features > 0:
        chunk_figs: Dict[str, go.Figure] = {}
        for s in range(0, Dx, int(cluster_features)):
            e = min(s + int(cluster_features), Dx)

            XY_real_chunk = np.concatenate([Xr_plot[:, s:e], yr_plot], axis=1)
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
            reducer.fit(XY_real_chunk)
            real_2d_chunk = reducer.transform(XY_real_chunk)

            frames_chunk_raw = []
            for XYhd in frames_XY_hd_all:
                XY_chunk = np.concatenate([XYhd[:, s:e], XYhd[:, Dx:Dx+Dy]], axis=1)
                frames_chunk_raw.append(reducer.transform(XY_chunk))

            # Legend stats per chunk (overall deltas)
            deltas2d_chunk = _flatten_point_deltas_2d(frames_chunk_raw)
            deltas_hd_all  = _flatten_point_deltas_hd(frames_XY_hd_all)  # same HD stats as others
            notes_chunk = [
                _format_stats_lines(f"Δ2D (X[{s}:{e})+Y UMAP)", _summarize_deltas(deltas2d_chunk)),
                _format_stats_lines("ΔHD (X+Y space)",          _summarize_deltas(deltas_hd_all)),
            ]

            frames_chunk = _densify_adaptive_2d(frames_chunk_raw, interp_policy)
            scales_chunk = [1.0] * len(frames_chunk)
            labels_chunk = [f"{i+1}" for i in range(len(frames_chunk))]
            key = f"XY_chunk_{s}_{e-1}"

            chunk_figs[key] = make_scrubber_X_or_XY(
                title=f"FlowGen evolution — UMAP on X[{s}:{e})+Y — {standard_model} → {realism_model}",
                real_2d=real_2d_chunk, real_c=cr_plot,
                frames_2d=frames_chunk, frames_c=cr_plot,
                scales=scales_chunk, frame_labels=labels_chunk, class_ids=class_ids,
                legend_notes=notes_chunk
            )
        figs.update(chunk_figs)

    # 7) meta
    meta = {
        "x_cols": x_cols, "y_cols": y_cols,
        "standard_snapshots": [p.name for p in std["snapshots"]],
        "realism_snapshots":  [p.name for p in rlm["snapshots"]],
        "view": view,
        "latent_source": latent_source,
        "latent_reference_model": ref_model,
        "seed_init": seed,
        "N_points": int(N),
        "Dx": int(Dx), "Dy": int(Dy),
        "interp_policy": interp_policy,
    }

    return figs, meta



def plot_umap_comparison(df_input, df_latent, condition_col="type"):
    """
    Plots UMAP projections of input and latent DataFrames side-by-side:
    - Whole dataset
    - One plot per class
    """
    assert condition_col in df_input.columns and condition_col in df_latent.columns, \
        f"Column '{condition_col}' must exist in both DataFrames"

    # Drop non-feature columns
    def extract_features(df):
        return df.drop(columns=["post_cleaning_index", condition_col], errors="ignore")

    X_input = extract_features(df_input).values
    X_latent = extract_features(df_latent).values
    y = df_input[condition_col].values  # Same order as df_latent

    reducer = umap.UMAP(random_state=42)
    X_input_umap = reducer.fit_transform(X_input)
    X_latent_umap = reducer.fit_transform(X_latent)

    # Create side-by-side plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    import numpy as np

    unique_classes = np.unique(y)
    titles = ["All"] + [f"Class {cls}" for cls in unique_classes]


    for i, ax in enumerate(axes[0]):
        if i == 0:
            sns.scatterplot(x=X_input_umap[:, 0], y=X_input_umap[:, 1], hue=y, ax=ax, palette="tab10", s=10, legend=False)
        else:
            class_mask = y == unique_classes[i - 1]
            sns.scatterplot(x=X_input_umap[class_mask, 0], y=X_input_umap[class_mask, 1], ax=ax, s=10)
        ax.set_title(f"Input UMAP — {titles[i]}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom row: latent space
    for i, ax in enumerate(axes[1]):
        if i == 0:
            sns.scatterplot(x=X_latent_umap[:, 0], y=X_latent_umap[:, 1], hue=y, ax=ax, palette="tab10", s=10, legend=False)
        else:
            class_mask = y == unique_classes[i - 1]
            sns.scatterplot(x=X_latent_umap[class_mask, 0], y=X_latent_umap[class_mask, 1], ax=ax, s=10)
        ax.set_title(f"Latent UMAP — {titles[i]}")
        ax.set_xticks([])
        ax.set_yticks([])


    plt.tight_layout()
    plt.show()

def generate_umap_interpolated_snapshots(
    df: pd.DataFrame,
    condition_col: str,
    cols_to_exclude: list[str],
    model_name: str,
    verbose: bool = True
) -> list[np.ndarray]:
    """
    Generates interpolated UMAP coordinates across FlowPre snapshots.

    Returns:
        List of np.ndarray: interpolated 2D UMAP frames (n_steps x n_points x 2).
    """
    assert re.match(r".+_v\d+$", model_name), f"model_name must end with '_vX'"

    # Prepare paths
    model_dir = ROOT_PATH / "outputs" / "models" / "flow_pre" / model_name
    config = load_yaml_config(model_dir / f"{model_name}.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flowpre_log(f"Generating UMAP interpolation for {model_name}", log_training=False, verbose=verbose)
    flowpre_log(f"Device: {device}", log_training=False, verbose=verbose)

    # Prepare data
    df_filtered = filter_flowpre_columns(df, cols_to_exclude, condition_col)
    x = torch.tensor(df_filtered.drop(columns=[condition_col]).values, dtype=torch.float32).to(device)
    c = torch.tensor(df_filtered[condition_col].values, dtype=torch.long).to(device)

    # Load all snapshots sorted by epoch
    snapshots_dir = model_dir / "snapshots"
    snapshot_paths = sorted(
        list(snapshots_dir.glob(f"{model_name}_epoch*_loss*.pt")),
        key=lambda p: int(re.search(r"epoch(\d+)", p.name).group(1))
    )

    flowpre_log(f"Found {len(snapshot_paths)} snapshots", log_training=False, verbose=verbose)

    if not snapshot_paths:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")

    # Rebuild model
    model_cfg = config["model"]
    model = build_flow_pre_model(model_cfg, input_dim=x.shape[1], num_classes=c.max().item() + 1, device=device)

    # Transform using each snapshot
    latent_list = []
    for path in snapshot_paths:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        with torch.no_grad():
            z, _ = model.forward(x, c)
        latent_list.append(z.cpu().numpy())

    # UMAP on each latent
    reducer = umap.UMAP(n_components=2, random_state=42)
    original_data = x.cpu().numpy()
    reducer.fit(original_data)  # Fit on RAW input data (not latent)

    # Initial UMAP frame from raw
    original_coords = reducer.transform(original_data)
    umap_coords = [original_coords]

    # Then transform latents
    for latent in latent_list:
        coords = reducer.transform(latent)
        umap_coords.append(coords)

    umap_coords = np.array(umap_coords)  # shape: (n_snapshots + 1, n_points, 2)

    # Interpolation
    min_steps = 10
    interp_frames = []

    for i in range(len(umap_coords) - 1):
        start = umap_coords[i]
        end = umap_coords[i + 1]

        # Compute average movement between all points
        avg_movement = np.linalg.norm(end - start, axis=1).mean()

        # Dynamically determine number of interpolation steps
        # Increase more steps if points move more
        dynamic_steps = int(np.clip(avg_movement * 10, min_steps, 100))

        for alpha in np.linspace(0, 1, dynamic_steps, endpoint=False):
            interp = start + (end - start) * alpha
            interp_frames.append(interp)

    # Append final frame
    interp_frames.append(umap_coords[-1])

    return interp_frames  # list of np.ndarray, each of shape (n_points, 2)

def generate_umap_interpolated_snapshots_aligned(
    df: pd.DataFrame,
    condition_col: str,
    cols_to_exclude: list[str],
    model_name: str,
    verbose: bool = True
) -> list[np.ndarray]:
    """
    Generates aligned and interpolated UMAP coordinates across FlowPre snapshots.
    Each latent gets its own UMAP fit. Coordinates are aligned via Procrustes.

    Returns:
        List of np.ndarray: interpolated 2D UMAP frames (n_steps x n_points x 2).
    """
    assert re.match(r".+_v\d+$", model_name), f"model_name must end with '_vX'"

    model_dir = ROOT_PATH / "outputs" / "models" / "flow_pre" / model_name
    config = load_yaml_config(model_dir / f"{model_name}.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flowpre_log(f"Generating aligned UMAP interpolation for {model_name}", log_training=False, verbose=verbose)
    df_filtered = filter_flowpre_columns(df, cols_to_exclude, condition_col)

    x = torch.tensor(df_filtered.drop(columns=[condition_col]).values, dtype=torch.float32).to(device)
    c = torch.tensor(df_filtered[condition_col].values, dtype=torch.long).to(device)

    # Load model snapshots
    snapshots_dir = model_dir / "snapshots"
    snapshot_paths = sorted(
        list(snapshots_dir.glob(f"{model_name}_epoch*_loss*.pt")),
        key=lambda p: float(re.search(r"loss-?([0-9.]+)\.pt", p.name).group(1)),
        reverse=True
    )
    if not snapshot_paths:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
    flowpre_log(f"Found {len(snapshot_paths)} snapshots", log_training=False, verbose=verbose)

    # Build model
    model_cfg = config["model"]
    model = build_flow_pre_model(model_cfg, input_dim=x.shape[1], num_classes=c.max().item() + 1, device=device)

    # Compute latent projections
    latent_list = []
    for path in snapshot_paths:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        with torch.no_grad():
            z, _ = model.forward(x, c)
        latent_list.append(z.cpu().numpy())

    # --- UMAP per latent + alignment ---
    aligned_coords = []
    ref_coords = None

    reducer = umap.UMAP(n_components=2, random_state=42)
    ref_coords = reducer.fit_transform(latent_list[-1])  # Fit UMAP only on the final snapshot

    aligned_coords = []
    for i, latent in enumerate(latent_list):
        coords = reducer.transform(latent)  # Project all previous latents using the same reducer
        aligned_coords.append(coords)

    # --- Interpolation ---
    interp_frames = []
    min_steps = 10

    for i in range(len(aligned_coords) - 1):
        start = aligned_coords[i]
        end = aligned_coords[i + 1]
        avg_movement = np.linalg.norm(end - start, axis=1).mean()
        dynamic_steps = max(int(np.clip(avg_movement * 10, min_steps, 100)) // 2, 1)
        for alpha in np.linspace(0, 1, dynamic_steps, endpoint=False):
            interp = start + (end - start) * alpha
            interp_frames.append(interp)

    interp_frames.append(aligned_coords[-1])
    return interp_frames  # List of np.ndarray

def plot_umap_evolution_interactive(
    df: pd.DataFrame,
    condition_col: str,
    cols_to_exclude: list[str],
    model_name: str,
    verbose: bool = True
):
    """
    Plots an interactive UMAP evolution over FlowPre training snapshots.

    Args:
        df (pd.DataFrame): Original input data.
        condition_col (str): Conditioning column used during training.
        cols_to_exclude (list[str]): Columns to exclude before transformation.
        config_filename (str): YAML config filename.
        model_name (str): Name of the model directory (must end in _vX).
        use_last_version (bool): Whether to fallback to latest version.
    """
    umap_frames = generate_umap_interpolated_snapshots_aligned(
        df=df,
        condition_col=condition_col,
        cols_to_exclude=cols_to_exclude,
        model_name=model_name,
        verbose=verbose
    )

    condition_values = df[condition_col].values
    unique_classes = np.unique(condition_values)

    # Create a color map
    import plotly.express as px
    color_map = px.colors.qualitative.Set1
    color_dict = {cls: color_map[i % len(color_map)] for i, cls in enumerate(unique_classes)}
    colors = [color_dict[val] for val in condition_values]

    fig = go.Figure()

    for i, coords in enumerate(umap_frames):
        scatter = go.Scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode='markers',
            marker=dict(color=colors, size=6, opacity=0.8),
            text=[f"{condition_col}: {v}" for v in condition_values],
            hoverinfo="text",
            visible=(i == 0),
            name=f"Step {i+1}"
        )
        fig.add_trace(scatter)

    # Create slider steps
    steps = []
    for i in range(len(umap_frames)):
        step = dict(
            method="update",
            args=[
                {"visible": [j == i for j in range(len(umap_frames))]},
                {"title": f"UMAP Frame {i + 1}"},
            ],
            label=f"{i+1}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Step: "},
        pad={"t": 40},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title="UMAP Evolution over FlowPre Snapshots",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        height=600,
        showlegend=False
    )

    fig.show()

def compute_mi_matrix(df):
    cols = df.columns
    mi_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for i, col in enumerate(cols):
        mi = mutual_info_regression(df.drop(columns=[col]), df[col], discrete_features=False)
        mi_matrix.loc[col, df.drop(columns=[col]).columns] = mi
    return mi_matrix

def rank_similarity(corr1, corr2):
    np.fill_diagonal(corr1.values, 0)
    np.fill_diagonal(corr2.values, 0)
    top_corr1 = corr1.abs().unstack().sort_values(ascending=False).drop_duplicates().head(10)
    top_corr2 = corr2.abs().unstack().sort_values(ascending=False).drop_duplicates().head(10)
    common = set(top_corr1.index).intersection(set(top_corr2.index))
    return len(common) / 10  # fraction match

def analyze_and_plot(df_raw, df_scaled):
    epsilon = 1e-8

    # Correlation and Covariance
    corr_raw = df_raw.corr()
    corr_scaled = df_scaled.corr()
    cov_raw = df_raw.cov()
    cov_scaled = df_scaled.cov()

    log_cov_raw = np.log10(np.abs(cov_raw) + epsilon)
    log_cov_scaled = np.log10(np.abs(cov_scaled) + epsilon)

    # Mutual Information
    mi_raw = compute_mi_matrix(df_raw)
    mi_scaled = compute_mi_matrix(df_scaled)

    # PCA
    pca_raw = PCA().fit(df_raw)
    pca_scaled = PCA().fit(df_scaled)
    cos_sim = 1 - cosine(pca_raw.explained_variance_ratio_, pca_scaled.explained_variance_ratio_)

    # Rank similarity
    rank_similarity_score = rank_similarity(corr_raw.copy(), corr_scaled.copy())

    # Print summary
    print("=== CORRELATION ===")
    print(f"Total abs diff: {np.abs(corr_raw - corr_scaled).values.sum():.4f}")
    print(f"Mean % diff: {np.mean((np.abs(corr_raw - corr_scaled) / (np.abs(corr_raw) + epsilon)).values) * 100:.2f}%")

    print("\n=== COVARIANCE ===")
    print(f"Total abs diff: {np.abs(cov_raw - cov_scaled).values.sum():.4f}")
    print(f"Mean % diff: {np.mean((np.abs(cov_raw - cov_scaled) / (np.abs(cov_raw) + epsilon)).values) * 100:.2f}%")

    print("\n=== MUTUAL INFORMATION ===")
    print(f"Total abs diff: {np.abs(mi_raw - mi_scaled).values.sum():.4f}")
    print(f"Mean % diff: {np.mean((np.abs(mi_raw - mi_scaled) / (np.abs(mi_raw) + epsilon)).values) * 100:.2f}%")

    print("\n=== PCA ===")
    print(f"Cosine similarity of explained variance: {cos_sim:.4f}")

    print("\n=== RANK ORDER SIMILARITY ===")
    print(f"Top 10 correlation pairs match: {rank_similarity_score * 100:.1f}%")

    # Visuals
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    sns.heatmap(corr_raw, ax=axes[0, 0], cmap='coolwarm', center=0)
    axes[0, 0].set_title("Correlation (Original)")

    sns.heatmap(corr_scaled, ax=axes[0, 1], cmap='coolwarm', center=0)
    axes[0, 1].set_title("Correlation (Scaled)")

    sns.heatmap(log_cov_raw, ax=axes[1, 0], cmap='magma')
    axes[1, 0].set_title("Log₁₀ |Covariance| (Original)")

    sns.heatmap(log_cov_scaled, ax=axes[1, 1], cmap='magma')
    axes[1, 1].set_title("Log₁₀ |Covariance| (Scaled)")

    sns.heatmap(mi_raw, ax=axes[2, 0], cmap='viridis')
    axes[2, 0].set_title("Mutual Information (Original)")

    sns.heatmap(mi_scaled, ax=axes[2, 1], cmap='viridis')
    axes[2, 1].set_title("Mutual Information (Scaled)")

    plt.tight_layout()
    plt.show()

    # PCA curve
    plt.figure(figsize=(8, 4))
    plt.plot(pca_raw.explained_variance_ratio_, label="Original")
    plt.plot(pca_scaled.explained_variance_ratio_, label="Scaled")
    plt.title("PCA Explained Variance Ratio")
    plt.xlabel("Component")
    plt.ylabel("Explained Variance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def custom_scale(
        df,
        method="standard",
        exclude_columns=None,
        standard_cols=None,
        robust_cols=None,
        minmax_cols=None
):
    """
    Scale a DataFrame using global or per-column method.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        method (str): Global method: 'standard', 'robust', 'minmax', or 'own'.
        exclude_columns (list[str]): Columns to skip scaling but include in result.
        standard_cols, robust_cols, minmax_cols (list[str]): Per-column overrides when method='own'.

    Returns:
        pd.DataFrame: Same columns and order as original, with selected ones scaled.
    """
    df_t = df.copy()
    all_cols = df_t.columns.tolist()

    # Sanitize inputs
    exclude_columns = set(exclude_columns or [])
    standard_cols = set(standard_cols or [])
    robust_cols = set(robust_cols or [])
    minmax_cols = set(minmax_cols or [])

    # Resolve overlaps → robust takes priority
    overlapping = (standard_cols & robust_cols) | (standard_cols & minmax_cols) | (robust_cols & minmax_cols)
    if overlapping:
        print(f"[INFO] Overlapping columns defaulted to 'robust': {overlapping}")
        robust_cols |= overlapping
        standard_cols -= overlapping
        minmax_cols -= overlapping

    # Determine columns to scale based on method
    if method != "own":
        # Scale everything except excluded
        cols_to_scale = [col for col in all_cols if col not in exclude_columns]
        df_scaled = df_t.copy()

        if method == "standard":
            df_scaled[cols_to_scale] = (df_t[cols_to_scale] - df_t[cols_to_scale].mean()) / (
                        df_t[cols_to_scale].std(ddof=0) + 1e-8)

        elif method == "robust":
            q1 = df_t[cols_to_scale].quantile(0.25)
            q3 = df_t[cols_to_scale].quantile(0.75)
            iqr = q3 - q1
            df_scaled[cols_to_scale] = (df_t[cols_to_scale] - df_t[cols_to_scale].median()) / (iqr + 1e-8)

        elif method == "minmax":
            df_scaled[cols_to_scale] = (df_t[cols_to_scale] - df_t[cols_to_scale].min()) / (
                        df_t[cols_to_scale].max() - df_t[cols_to_scale].min() + 1e-8)

        else:
            raise ValueError("Invalid method: choose 'standard', 'robust', 'minmax', or 'own'.")

        return df_scaled[all_cols]

    # === OWN SCALING ===
    df_result = df.copy()

    # Standard scaling
    for col in standard_cols:
        mean = df[col].mean()
        std = df[col].std(ddof=0) + 1e-8
        df_result[col] = (df[col] - mean) / std

    # Robust scaling
    for col in robust_cols:
        median = df[col].median()
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25) + 1e-8
        df_result[col] = (df[col] - median) / iqr

    # Min-max scaling
    for col in minmax_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        df_result[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)

    # Everything else remains untouched
    return df_result[all_cols]

def run_analysis_pipeline(
        df,
        method="standard",
        exclude_columns=None,
        standard_cols=None,
        robust_cols=None,
        minmax_cols=None
):
    """
    Full analysis pipeline:
    - Scales data using custom_scale
    - Removes excluded columns for analysis
    - Runs correlation, covariance, MI, PCA, rank similarity comparisons

    Parameters:
        df (pd.DataFrame): Full input DataFrame
        method (str): 'standard', 'robust', 'minmax', or 'own'
        exclude_columns (list[str]): Columns to skip from scaling and analysis
        standard_cols, robust_cols, minmax_cols: Used if method='own'

    Returns:
        df_scaled (pd.DataFrame): The fully scaled DataFrame (including excluded columns)
    """
    # Step 1: Scale full DataFrame (with excluded columns untouched)
    df_scaled = custom_scale(
        df,
        method=method,
        exclude_columns=exclude_columns,
        standard_cols=standard_cols,
        robust_cols=robust_cols,
        minmax_cols=minmax_cols
    )

    # Step 2: Remove excluded columns for analysis
    analysis_cols = [col for col in df.columns if col not in (exclude_columns or [])]
    df_raw = df[analysis_cols]
    df_scaled_sub = df_scaled[analysis_cols]

    # Step 3: Run analysis
    analyze_and_plot(df_raw, df_scaled_sub)

    # Return the full scaled DataFrame
    return df_scaled

def compute_rrmse(arr, target, divisor: bool = True):
    nominator = np.sqrt(np.mean((arr - target) ** 2))
    if divisor:
        return nominator / (np.abs(target) + 1e-8)
    return nominator

def compute_metrics(df_latent, drop_cols=["post_cleaning_index", "type"]):
    z = df_latent.drop(columns=drop_cols).values
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    skewness_vals = skew(z, axis=0)
    kurt_vals = kurtosis(z, axis=0, fisher=False)  # Not using Fisher to get total kurtosis

    # Mahalanobis distances
    z_centered = z - mean
    cov = np.cov(z_centered, rowvar=False)
    cov_inv = np.linalg.inv(cov)
    mahalanobis_sq = np.sum(z_centered @ cov_inv * z_centered, axis=1)
    mahalanobis_mean = mahalanobis_sq.mean()
    ks_stat, ks_p = kstest(mahalanobis_sq, chi2(df=z.shape[1]).cdf)

    # PCA-based isotropy (ideal = 1 for all eigenvalues)
    pca = PCA()
    pca.fit(z)
    eig_std = np.std(pca.explained_variance_)

    return {
        "z_mean_mean": mean.mean(),
        "z_std_mean": std.mean(),
        "rrmse_mean": compute_rrmse(mean, 0.0, False),
        "rrmse_std": compute_rrmse(std, 1.0),
        "skewness_mean": np.abs(skewness_vals).mean(),
        "kurtosis_mean": np.abs(kurt_vals).mean(),
        "mahalanobis_mean": mahalanobis_mean,
        "mahalanobis_ks_p": ks_p,
        "eigval_std": eig_std
    }

def plot_umap(df_latent, condition_col="type", ax=None):
    z = df_latent.drop(columns=[condition_col, "post_cleaning_index"])
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(z)

    if ax is None:
        _, ax = plt.subplots()

    if condition_col in df_latent.columns:
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=df_latent[condition_col],
            palette="Set2",
            ax=ax,
            s=10,
            alpha=0.8,
            legend=False,
        )
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.8)

    ax.set_xticks([])
    ax.set_yticks([])

def plot_multivariate_qq(df_latent, title="Q-Q Plot", ax=None):
    from scipy.stats import chi2
    from sklearn.covariance import EmpiricalCovariance

    X = df_latent.values
    cov = EmpiricalCovariance().fit(X)
    mahal = cov.mahalanobis(X)
    sorted_mahal = np.sort(mahal)
    quantiles = chi2.ppf((np.arange(1, len(X) + 1) - 0.5) / len(X), df=X.shape[1])

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(quantiles, sorted_mahal, 'o', markersize=2, label="Empirical vs. Theoretical")
    ax.plot(quantiles, quantiles, 'r--', label="Ideal fit")
    ax.set_title(title)
    ax.set_xlabel("Theoretical Quantiles (Chi²)")
    ax.set_ylabel("Empirical Mahalanobis Distances")
    ax.legend()

def evaluate_flowpre_models(
    model_versions,
    base_model_name,
    df_flowpre_input,
    condition_col="type",
    cols_to_exclude=None,
    config_filename="flow_pre.yaml",
    use_last_version=False,
    root_path=ROOT_PATH,
    verbose=False,
):
    """
    Evaluates a list of FlowPre models by transforming input data to latent space,
    computing metrics, plotting UMAP and Q-Q plots, and visualizing feature influence.

    Returns:
        metrics_list (list of dict): Evaluation metrics for each model.
    """
    if cols_to_exclude is None:
        cols_to_exclude = []

    metrics_list = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for version in model_versions:
            model_name = f"{base_model_name}_{version}"

            # --- Latent space extraction ---
            df_latent = transform_to_latent_with_flowpre(
                df=df_flowpre_input,
                condition_col=condition_col,
                cols_to_exclude=cols_to_exclude,
                config_filename=config_filename,
                model_name=model_name,
                use_last_version=use_last_version,
                verbose=verbose,
            )

            # --- Compute metrics ---
            metrics = compute_metrics(df_latent)
            metrics["model"] = model_name
            metrics_list.append(metrics)

            # --- Plot UMAP + Q-Q ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(model_name)

            plot_umap(df_latent, condition_col=condition_col, ax=axes[0])
            axes[0].set_title("UMAP (Latent Space)")

            plot_multivariate_qq(
                df_latent.drop(columns=[condition_col, "post_cleaning_index"], errors="ignore"),
                title="Q-Q Plot (Latent Space)",
                ax=axes[1],
            )
            axes[1].set_title("Q-Q Plot (Latent Space)")

            plt.tight_layout()
            plt.show()

            # --- Load Influence JSON ---
            file_path = Path(root_path) / "outputs" / "models" / "flow_pre" / model_name / f"{model_name}_influence.json"
            with open(file_path, "r") as f:
                influence_dict = json.load(f)

            # --- Parse influence to DataFrame ---
            records = []
            for z_name, features in influence_dict.items():
                for feat, (raw, norm) in features.items():
                    records.append({
                        "latent_feature": z_name,
                        "input_feature": feat,
                        "raw_influence": raw,
                        "normalized_influence": norm
                    })

            df_influence = pd.DataFrame.from_records(records)

            # --- Plot Influence ---
            plt.figure(figsize=(14, 6))
            sns.boxplot(data=df_influence, x="input_feature", y="normalized_influence", palette="viridis")
            plt.xticks(rotation=45, ha="right")
            plt.title("Distribution of Normalized Influence per Input Feature")
            plt.tight_layout()
            plt.show()

    return metrics_list

def summarize_flowpre_metrics(
    metrics_list: list[dict],
    display_tables: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Summarizes and sorts FlowPre evaluation metrics.

    Args:
        metrics_list (list of dict): Metrics for each model (output of evaluate_flowpre_models).
        display_tables (bool): Whether to display tables using .style in a notebook.

    Returns:
        summary_tables (dict): Dictionary of DataFrames sorted by each metric.
    """

    def sort_by_distance(df, column, target):
        return df.copy().assign(
            _distance=(df[column] - target).abs()
        ).sort_values("_distance").drop(columns="_distance").reset_index(drop=True)

    df_summary = pd.DataFrame(metrics_list)
    summary_tables = {}

    # Metrics where lower is better
    for metric in ["rrmse_mean", "rrmse_std", "eigval_std"]:
        sorted_df = df_summary.sort_values(by=metric).reset_index(drop=True)
        summary_tables[metric] = sorted_df
        if display_tables:
            display(sorted_df.style.set_caption(f"Sorted by {metric} (lower is better)"))

    # Metrics where closeness to target is better
    distance_targets = {
        "z_mean_mean": 0.0,
        "z_std_mean": 1.0,
        "skewness_mean": 0.0,
        "kurtosis_mean": 3.0,
        "mahalanobis_mean": df_summary.shape[1] - 1,  # dim ≈ mean of χ²_d
    }

    for metric, target in distance_targets.items():
        sorted_df = sort_by_distance(df_summary, metric, target)
        summary_tables[metric] = sorted_df
        if display_tables:
            display(sorted_df.style.set_caption(f"Sorted by closeness to {target} for {metric}"))

    # Mahalanobis KS p-value: higher is better
    if "mahalanobis_ks_p" in df_summary.columns:
        sorted_df = df_summary.sort_values(by="mahalanobis_ks_p", ascending=False).reset_index(drop=True)
        summary_tables["mahalanobis_ks_p"] = sorted_df
        if display_tables:
            display(sorted_df.style.set_caption("Sorted by Mahalanobis KS p-value (higher is better)"))

    return summary_tables

def plot_correlation_matrix(df_latent, title = None):
    plt.figure(figsize=(12, 10))
    corr = df_latent.corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, square=True, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_covariance_matrix(df_latent, title = None):
    plt.figure(figsize=(12, 10))
    cov = df_latent.cov()
    sns.heatmap(cov, annot=False, cmap="coolwarm", center=0, square=True, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def describe_latent(df_latent, condition_col="type"):
    features = df_latent.drop(columns=["post_cleaning_index", condition_col], errors="ignore")
    return features.describe()

def compare_pairwise_distances(df_before, df_after, condition_col="type"):
    X1 = df_before.drop(columns=["post_cleaning_index", condition_col], errors="ignore").values
    X2 = df_after.drop(columns=["post_cleaning_index", condition_col], errors="ignore").values

    D1 = pairwise_distances(X1)
    D2 = pairwise_distances(X2)

    corr, _ = pearsonr(D1.ravel(), D2.ravel())
    return corr

def cosine_structure_similarity(df1, df2, condition_col="type"):
    X1 = df1.drop(columns=["post_cleaning_index", condition_col], errors="ignore").values
    X2 = df2.drop(columns=["post_cleaning_index", condition_col], errors="ignore").values

    sim1 = cosine_similarity(X1)
    sim2 = cosine_similarity(X2)

    corr, _ = pearsonr(sim1.ravel(), sim2.ravel())
    return corr

def plot_umap_per_set(X_train, X_val, X_test, n_neighbors=15, min_dist=0.1, random_state=42):
    def clean_input(X):
        return X.drop(columns=["post_cleaning_index", "type"], errors="ignore")

    # Step 1: Fit UMAP on all combined sets to ensure shared projection space
    X_all = pd.concat([clean_input(X_train), clean_input(X_val), clean_input(X_test)], axis=0)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    reducer.fit(X_all)

    # Step 2: Transform each set
    sets = {
        "Train": (X_train, reducer.transform(clean_input(X_train))),
        "Validation": (X_val, reducer.transform(clean_input(X_val))),
        "Test": (X_test, reducer.transform(clean_input(X_test)))
    }

    # Step 3: Plot UMAP per set
    for name, (X_orig, embedding) in sets.items():
        plt.figure(figsize=(7, 5))
        scatter = plt.scatter(
            embedding[:, 0], embedding[:, 1],
            c=X_orig["type"], cmap="tab10", s=20, alpha=0.7
        )
        plt.colorbar(scatter, label="Type")
        plt.title(f"UMAP - {name} Set")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


