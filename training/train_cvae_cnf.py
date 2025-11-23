import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from models.encoder import Encoder
from models.decoder import Decoder
from models.conditional_flow import ConditionalFlow, NoFlowPrior
from models.cvae_cnf_wrapper import CVAECNF
from typing import Optional, Dict, Any, List, Tuple
from training.utils import (
    load_yaml_config,
    flowpre_log,
    setup_training_logs_and_dirs,
    log_epoch_diagnostics,   # ← mirror FlowPre
    ROOT_PATH,               # ← mirror FlowPre
)
import shutil
import json
import math
import numpy as np
from copy import deepcopy
import yaml

from losses.cvae_cnf_loss import flexible_cvae_cnf_loss_from_model

import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple, Optional, Union, List
import os, random, secrets

# --- Determinism helper (no-op if seed is None) ---
def _maybe_set_seed(seed: int | None) -> int:
    """
    If seed is None, sample a 64-bit seed and return it.
    If seed is provided, use it deterministically and return it.
    Torch allows up to 2**63-1; NumPy legacy seeding wants < 2**32.
    """

    # 1) Choose a seed
    if seed is None:
        # 64-bit random; keep within torch's recommended bound
        seed = secrets.randbits(64) % (2**63 - 1)

    # 2) Apply to Python/NumPy/Torch
    random.seed(seed)

    # NumPy legacy RNG expects 32-bit
    np_seed = int(seed % (2**32 - 1))  # keep in [0, 2**32-2]
    np.random.seed(np_seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    return int(seed)

def _make_baseline_x(
    features: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Baseline input x0: mean over rows (optionally restricted by mask).
    Returns shape (1, Dx).
    """
    if mask is not None:
        x0 = features[mask].mean(dim=0, keepdim=True)
    else:
        x0 = features.mean(dim=0, keepdim=True)
    return x0

def _repeat_context(
    context_row: torch.Tensor,
    n: int
) -> torch.Tensor:
    """
    Repeat a 1-row context tensor n times. Accepts 1D (labels) or 2D (multi-col).
    """
    if context_row.dim() == 0:
        # scalar label -> shape (1,)
        context_row = context_row.unsqueeze(0)
    if context_row.dim() == 1:
        # shape (1,) or (C,) -> repeat to (n,) or (n, C)
        return context_row.repeat(n, *([1] if context_row.numel() > 1 else []))
    elif context_row.dim() == 2:
        return context_row.repeat(n, 1)
    else:
        raise ValueError("context_row must be 0D, 1D or 2D tensor.")

def _encode_mean_latent(model, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Deterministic latent path: z = mu_q(x, c).
    Uses torch.no_grad() to avoid tracking grads w.r.t. model parameters.
    Gradients will still flow back to x if x.requires_grad is True.
    """
    with torch.no_grad():
        mu, _ = model.encode_params(x, c)  # (B, Dz)
    return mu

def compute_cvae_feature_influence(
    model,
    features: torch.Tensor,            # (N, Dx) on same device as model or will be moved
    context: torch.Tensor,             # (N,) int labels OR (N, Cctx)
    feature_names: List[str],
    per_class: bool = True,
    context_baseline: Optional[torch.Tensor] = None,  # explicit context to evaluate at
    use_classwise_x0: bool = True,     # baseline x0 is classwise mean if per_class
    abs_value: bool = True,
    normalize_by_input: bool = True,
) -> Union[
        Dict[Union[int, str], Dict[str, Dict[str, Tuple[float, float]]]],
        Dict[str, Dict[str, Tuple[float, float]]]
    ]:
    """
    Compute feature→feature influence matrix at a baseline (x0, c), in the *original* space:
        J = d x_hat / d x  (via encoder mean -> decoder)

    Returns either:
      - per-class dict: { class_value: { "matrix": <numpy Dx x Dx>, "by_input": {...} } }
      - single dict if per_class=False: { "matrix": ..., "by_input": {...} }

    "by_input" is a readable structure:
       {
         input_feat_k: {
            "total_raw": float,
            "to_outputs": { output_feat_j: (raw, norm_share), ... }
         },
         ...
       }

    Args:
      model: CVAECNF model (uses encoder mean-path, decoder)
      features: (N, Dx) tensor (used to define baseline x0)
      context:  (N,) labels OR (N, Cctx) tensor
      feature_names: list of length Dx
      per_class: if True and context is 1-D labels, compute one matrix per class
      context_baseline: optional explicit single-row context to evaluate at
      use_classwise_x0: when per_class=True, baseline x0 is the mean of features for that class
      abs_value: take absolute value of Jacobian entries (recommended)
      normalize_by_input: also report per-input normalized shares (columns normalized)

    Notes:
      - Sets model.eval(), no dropout etc. Gradients are enabled only for x0.
      - Uses encoder *mean* path (z = mu_q(x,c)) to avoid sampling noise.
    """
    model.eval()
    device = model.device
    features = features.to(device)
    context  = context.to(device)

    Dx = features.shape[1]
    assert len(feature_names) == Dx, "feature_names length must match number of columns in features."

    # Helper to compute a single (Dx_out x Dx_in) Jacobian at (x0, c0)
    def _jacobian_at(x0: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
        # x0: (1, Dx), c0: (1,) or (1, Cctx)
        x0 = x0.clone().detach().to(device).requires_grad_(True)

        # z = mu_q(x0, c0)
        z = _encode_mean_latent(model, x0, c0)
        # recon mean (1, Dx)
        xhat = model.decode(z, c0)

        # Build Jacobian J[j, k] = d xhat_j / d x0_k
        J = torch.zeros(Dx, Dx, device=device, dtype=xhat.dtype)
        for j in range(Dx):
            model.zero_grad(set_to_none=True)
            if x0.grad is not None:
                x0.grad.zero_()
            # scalar output
            y = xhat[0, j]
            y.backward(retain_graph=True)
            J[j, :] = x0.grad.detach().clone()  # (Dx,)
        if abs_value:
            J = J.abs()
        return J  # (Dx, Dx)

    # Convert a Jacobian to the requested dict format
    def _format_output(J: torch.Tensor):
        J_np = J.detach().cpu().numpy()
        by_input = OrderedDict()
        col_sums = np.sum(np.abs(J_np), axis=0) + 1e-12  # sums over outputs for each input
        for k, in_name in enumerate(feature_names):
            per_out = OrderedDict()
            for j, out_name in enumerate(feature_names):
                raw = float(J_np[j, k])
                if normalize_by_input:
                    norm_share = float(np.abs(J_np[j, k]) / col_sums[k])
                else:
                    norm_share = 0.0
                per_out[out_name] = (round(raw, 6), round(norm_share, 6))
            by_input[in_name] = {
                "total_raw": round(float(col_sums[k]), 6),
                "to_outputs": per_out
            }
        return {"matrix": J_np, "by_input": by_input}

    # If an explicit context baseline is provided, just use that (single result)
    if context_baseline is not None:
        # Ensure 1-row
        if context_baseline.dim() == 0:
            c0 = context_baseline.unsqueeze(0)
        elif context_baseline.dim() == 1:
            c0 = context_baseline.unsqueeze(0)
        else:
            c0 = context_baseline[:1]
        x0 = _make_baseline_x(features, mask=None)
        c0 = _repeat_context(c0, n=1)  # (1,) or (1, Cctx)

        J = _jacobian_at(x0, c0)
        return _format_output(J)

    # Otherwise, either per-class or single (global) baseline
    # Case A: per-class (only if 1-D integer labels)
    if per_class and context.dim() == 1 and context.dtype in (torch.long, torch.int32, torch.int64):
        out = {}
        classes = torch.unique(context).tolist()
        for cls in classes:
            mask = (context == cls)
            # defensive: if tiny class, fallback to global mean for x0
            x0 = _make_baseline_x(features, mask=mask if (use_classwise_x0 and mask.sum() > 1) else None)
            c0 = torch.tensor([cls], device=device, dtype=context.dtype)
            c0 = _repeat_context(c0, n=1)

            J = _jacobian_at(x0, c0)
            out[int(cls)] = _format_output(J)
        return out

    # Case B: single global baseline with a representative context
    # If context is 1-D labels: pick the majority (mode). If multi-col: use the first row.
    x0 = _make_baseline_x(features)
    if context.dim() == 1 and context.dtype in (torch.long, torch.int32, torch.int64):
        # mode class
        vals, counts = torch.unique(context, return_counts=True)
        c_mode = vals[counts.argmax()]
        c0 = _repeat_context(c_mode, n=1)
    else:
        # multi-col: take first row as baseline context
        c0 = context[:1].clone()

    J = _jacobian_at(x0, c0)
    return _format_output(J)


def compute_rrmse_r2(x_true: torch.Tensor, x_hat: torch.Tensor):
    """
    Compute Relative RMSE and R² score between true and reconstructed features.

    Args:
        x_true (Tensor): ground-truth features, shape (B, D)
        x_hat  (Tensor): reconstructed features, shape (B, D)

    Returns:
        tuple (rrmse, r2) as floats
    """
    with torch.no_grad():
        rmse_val = torch.sqrt(torch.mean((x_hat - x_true) ** 2))
        denom    = torch.sqrt(torch.mean(x_true ** 2)) + 1e-8
        rrmse    = (rmse_val / denom).item()

        ss_res = torch.sum((x_true - x_hat) ** 2)
        ss_tot = torch.sum((x_true - x_true.mean(dim=0)) ** 2) + 1e-8
        r2_val = (1.0 - ss_res / ss_tot).item()

    return rrmse, r2_val


def prepare_dataset(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    index_col: str = "post_cleaning_index",
    context_cols: list = None
):
    """
    Align and split the dataset into context and features tensors for CVAE-CNF training.
    - If context_cols has ONE column (e.g. 'type'), it is mapped to integer class IDs 0..K-1.
    - If context_cols has >1 columns, they are treated as continuous and returned as float32 (NOTE: your Encoder/Flow expect embeddings -> single int column is recommended).
    """
    import pandas as pd
    import numpy as np
    import pandas.api.types as ptypes
    import torch

    context_cols = context_cols or ["type"]
    assert isinstance(X_df, pd.DataFrame) and isinstance(y_df, pd.DataFrame)
    assert index_col in X_df.columns and index_col in y_df.columns
    for col in context_cols:
        assert col in X_df.columns, f"Context column '{col}' not found in X_df."

    # align by index_col
    X_df = X_df.copy().set_index(index_col)
    y_df = y_df.copy().set_index(index_col)
    X_df, y_df = X_df.align(y_df, join="inner", axis=0)
    X_df = X_df.reset_index().copy()
    y_df = y_df.reset_index().copy()

    # extract context and feature blocks
    context = X_df[context_cols].copy()
    feature_cols = [c for c in X_df.columns if c not in context_cols + [index_col]]
    X_no_context = X_df[feature_cols].copy()
    X_no_context = X_no_context.drop(columns=[index_col], errors="ignore")
    y = y_df.drop(columns=[index_col], errors="ignore").copy()

    # ===== sanitize context dtype(s) =====
    if len(context_cols) == 1:
        # single col -> force to class IDs 0..K-1 (int64)
        ctx_ser = context[context_cols[0]]
        # If not integer dtype, or integers but not guaranteed 0..K-1, remap stably:
        if not ptypes.is_integer_dtype(ctx_ser.dtype):
            codes = pd.Categorical(ctx_ser).codes.astype("int64")
        else:
            # ensure contiguous 0..K-1
            uniq, inv = np.unique(ctx_ser.values, return_inverse=True)
            codes = inv.astype("int64")
        context_tensor = torch.as_tensor(codes, dtype=torch.long)
    else:
        # multi-column context (continuous) -> float32 tensor
        context_tensor = torch.as_tensor(context.values, dtype=torch.float32)

    # ===== build features tensor (X + y concatenated) =====
    features_tensor = torch.as_tensor(
        pd.concat([X_no_context, y], axis=1).values,
        dtype=torch.float32
    )

    context_names = context_cols
    feature_names = feature_cols + list(y.columns)

    return context_tensor, features_tensor, context_names, feature_names



def get_dataloader(
    context_tensor,
    features_tensor,
    batch_size=128,
    shuffle=True,
    seed: Optional[int] = None,
):
    """
    Wrap context and features tensors into a DataLoader.
    Pass a per-run CPU generator for deterministic shuffling.
    """
    dataset = TensorDataset(features_tensor, context_tensor)

    # Always CPU for DataLoader generator (PyTorch requirement)
    g = torch.Generator(device="cpu")
    # Use the explicit seed if provided; otherwise use torch.initial_seed()
    gen_seed = int(seed if seed is not None else torch.initial_seed())
    g.manual_seed(gen_seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=0,
    )
    return dataset, loader


def build_encoder(cfg: Dict[str, Any], input_dim: int, num_classes: int, device: str = "cuda") -> Encoder:
    encoder_cfg = cfg["encoder"]
    return Encoder(
        input_dim=input_dim,
        latent_dim=encoder_cfg["latent_dim"],
        embedding_dim=encoder_cfg["embedding_dim"],
        hidden_features=encoder_cfg["hidden_features"],
        num_layers=encoder_cfg["num_layers"],
        dropout=encoder_cfg["dropout"],
        num_classes=num_classes,
        device=device,
    )


def build_decoder(cfg: Dict[str, Any], output_dim: int, num_classes: int, latent_dim: int, device: str = "cuda") -> Decoder:
    decoder_cfg = cfg["decoder"]
    return Decoder(
        output_dim=output_dim,
        latent_dim=latent_dim,
        embedding_dim=decoder_cfg["embedding_dim"],
        hidden_features=decoder_cfg["hidden_features"],
        num_layers=decoder_cfg["num_layers"],
        dropout=decoder_cfg["dropout"],
        num_classes=num_classes,
        use_variance_head=decoder_cfg.get("use_variance_head", False),  # ← add this
        device=device,
    )

def build_flow(cfg: Dict[str, Any], latent_dim: int, num_classes: int, device: str = "cuda"):
    flow_cfg = cfg["flow"]
    active = bool(flow_cfg.get("active", True))
    if not active:
        # Plain CVAE prior N(0, I)
        return NoFlowPrior(latent_dim=latent_dim, num_classes=num_classes, device=device)

    # Otherwise build the conditional flow as before
    return ConditionalFlow(
        latent_dim=latent_dim,
        num_classes=num_classes,
        embedding_dim=flow_cfg["embedding_dim"],
        hidden_features=flow_cfg["hidden_features"],
        num_layers=flow_cfg["num_layers"],
        use_actnorm=flow_cfg["use_actnorm"],
        use_learnable_permutations=flow_cfg["use_learnable_permutations"],
        num_bins=flow_cfg["num_bins"],
        tail_bound=flow_cfg["tail_bound"],
        initial_affine_layers=flow_cfg["initial_affine_layers"],
        affine_rq_ratio=tuple(flow_cfg["affine_rq_ratio"]),
        n_repeat_blocks=flow_cfg["n_repeat_blocks"],
        final_rq_layers=flow_cfg["final_rq_layers"],
        lulinear_finisher=flow_cfg["lulinear_finisher"],
        apply_weight_norm_hidden=flow_cfg.get("apply_weight_norm_hidden", True),
        zero_init_last=flow_cfg.get("zero_init_last", True),
        last_bias_init=float(flow_cfg.get("last_bias_init", 0.0)),
        device=device,
    )




def build_cvae_cnf_wrapper(cfg: Dict[str, Any], input_dim: int, output_dim: int, num_classes: int, device: str = "cuda") -> CVAECNF:
    latent_dim = cfg["encoder"]["latent_dim"]
    encoder = build_encoder(cfg, input_dim=input_dim, num_classes=num_classes, device=device)
    decoder = build_decoder(cfg, output_dim=output_dim, num_classes=num_classes, latent_dim=latent_dim, device=device)
    flow = build_flow(cfg, latent_dim=latent_dim, num_classes=num_classes, device=device)

    return CVAECNF(encoder=encoder, decoder=decoder, flow=flow, device=device)

def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def _safe_quantiles(a: np.ndarray, qs: np.ndarray) -> np.ndarray:
    # robust 1D or 2D: if 2D (N,D) → quantile along axis=0 → (len(qs), D)
    return np.quantile(a, qs, axis=0, method="linear")

def _per_feature_rrmse(x_true: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    # shape (N,D)
    num = np.sqrt(np.mean((x_hat - x_true) ** 2, axis=0))
    denom = np.sqrt(np.mean(x_true ** 2, axis=0)) + 1e-12
    return (num / denom)

def _per_feature_mae_mse(x_true: np.ndarray, x_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mae = np.mean(np.abs(x_hat - x_true), axis=0)
    mse = np.mean((x_hat - x_true) ** 2, axis=0)
    return mae, mse

def _summarize_vector(v: np.ndarray) -> Dict[str, float]:
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"mean": np.nan, "median": np.nan, "p90": np.nan, "p95": np.nan, "max": np.nan, "min": np.nan}
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p90": float(np.quantile(v, 0.90)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)),
        "min": float(np.min(v)),
    }

def _empirical_cdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # values shape (N,), grid shape (G,)
    # returns ECDF(grid) for each column if values is 2D; we vectorize per feature
    # Here we assume 1D; for 2D we loop column-wise (fast enough for D up to few hundred)
    idx = np.searchsorted(np.sort(values), grid, side="right")
    return idx / values.shape[0]

def _ks_1d(a: np.ndarray, b: np.ndarray) -> float:
    # exact KS distance using merged grid
    grid = np.sort(np.unique(np.concatenate([a, b], axis=0)))
    # guard degenerate case
    if grid.size == 0:
        return 0.0
    Fa = _empirical_cdf(a, grid)
    Fb = _empirical_cdf(b, grid)
    return float(np.max(np.abs(Fa - Fb)))

def _ks_all_features(X_real: np.ndarray, X_synth: np.ndarray) -> np.ndarray:
    D = X_real.shape[1]
    ks = np.zeros(D, dtype=np.float64)
    for j in range(D):
        ks[j] = _ks_1d(X_real[:, j], X_synth[:, j])
    return ks

def _w1_quantile_all_features(X_real: np.ndarray, X_synth: np.ndarray, qgrid: np.ndarray=None) -> np.ndarray:
    # 1D Wasserstein via quantile grid (fast, stable)
    if qgrid is None:
        qgrid = np.linspace(0.01, 0.99, 99)
    Qr = _safe_quantiles(X_real, qgrid)   # (Q, D)
    Qs = _safe_quantiles(X_synth, qgrid)  # (Q, D)
    return np.mean(np.abs(Qr - Qs), axis=0)  # (D,)

def _moments(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # returns (mean, std, skew, kurt_excess) featurewise
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0) + 1e-12
    z  = (x - mu) / sd
    skew = np.mean(z**3, axis=0)
    kurt_excess = np.mean(z**4, axis=0) - 3.0
    return mu, sd, skew, kurt_excess

def _moment_deltas(X_real: np.ndarray, X_synth: np.ndarray) -> Dict[str, np.ndarray]:
    mu_r, sd_r, sk_r, ku_r = _moments(X_real)
    mu_s, sd_s, sk_s, ku_s = _moments(X_synth)
    mean_abs_diff = np.abs(mu_s - mu_r)
    std_rel_err   = np.abs(sd_s - sd_r) / (sd_r + 1e-12)
    skew_abs_diff = np.abs(sk_s - sk_r)
    kurt_abs_diff = np.abs(ku_s - ku_r)
    return {
        "mean_abs_diff": mean_abs_diff,
        "std_rel_err": std_rel_err,
        "skew_abs_diff": skew_abs_diff,
        "kurt_abs_diff": kurt_abs_diff,
    }

def _corr_matrix(X: np.ndarray, rank: bool=False) -> np.ndarray:
    if rank:
        # Spearman: rank transform per column (average ranks for ties via pandas)
        df = pd.DataFrame(X)
        R = df.rank(method="average").to_numpy()
        return np.corrcoef(R, rowvar=False)
    return np.corrcoef(X, rowvar=False)

def _corr_deltas(X_real: np.ndarray, X_synth: np.ndarray) -> Dict[str, float]:
    Cpr = _corr_matrix(X_real, rank=False)
    Cps = _corr_matrix(X_synth, rank=False)
    Csr = _corr_matrix(X_real, rank=True)
    Css = _corr_matrix(X_synth, rank=True)

    def _delta(Cr, Cs):
        diff = Cr - Cs
        fro  = np.linalg.norm(diff, ord="fro")
        fro_rel = fro / (np.linalg.norm(Cr, ord="fro") + 1e-12)
        mad  = float(np.mean(np.abs(diff)))
        return float(fro), float(fro_rel), mad

    fro_p, fro_rel_p, mad_p = _delta(Cpr, Cps)
    fro_s, fro_rel_s, mad_s = _delta(Csr, Css)
    return {
        "pearson_fro": fro_p,
        "pearson_fro_rel": fro_rel_p,
        "pearson_mean_abs_diff": mad_p,
        "spearman_fro": fro_s,
        "spearman_fro_rel": fro_rel_s,
        "spearman_mean_abs_diff": mad_s,
    }

def _subsample_rows(X: torch.Tensor, n: int) -> torch.Tensor:
    n = min(n, X.shape[0])
    if n <= 0:
        return X
    idx = torch.randperm(X.shape[0], device=X.device)[:n]
    return X.index_select(0, idx)

def _pairwise_sq_dists(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # returns ||a-b||^2 matrix
    # (N,d) and (M,d) → (N,M)
    a2 = (A*A).sum(dim=1, keepdim=True)
    b2 = (B*B).sum(dim=1, keepdim=True).T
    return a2 + b2 - 2.0 * (A @ B.T)

def _median_pairwise_dist(X: torch.Tensor, max_samples: int = 1024) -> float:
    Xs = _subsample_rows(X, max_samples)
    D2 = _pairwise_sq_dists(Xs, Xs)
    D = torch.sqrt(torch.clamp(D2, min=0.0))
    # take upper triangle without diag
    triu = D[torch.triu_indices(D.shape[0], D.shape[1], offset=1).unbind()]
    if triu.numel() == 0:
        return 1.0
    return float(triu.median().item())

def _mmd_rbf_unbiased(X: torch.Tensor, Y: torch.Tensor, sigma: Optional[float] = None, max_samples: int = 1024) -> Dict[str, float]:
    # Subsample for stability & memory
    Xs = _subsample_rows(X, max_samples)
    Ys = _subsample_rows(Y, max_samples)
    n = Xs.shape[0]
    m = Ys.shape[0]
    if n < 2 or m < 2:
        return {"mmd2": float("nan"), "sigma": float("nan"), "n_used": int(n), "m_used": int(m)}
    if sigma is None:
        sigma = _median_pairwise_dist(torch.cat([Xs, Ys], dim=0), max_samples=max_samples)
        sigma = max(sigma, 1e-3)

    def _rbf(d2):
        return torch.exp(-d2 / (2.0 * (sigma ** 2)))

    Dxx = _pairwise_sq_dists(Xs, Xs)
    Dyy = _pairwise_sq_dists(Ys, Ys)
    Dxy = _pairwise_sq_dists(Xs, Ys)

    # Unbiased estimates: exclude diagonals
    Kxx = _rbf(Dxx)
    Kyy = _rbf(Dyy)
    Kxy = _rbf(Dxy)

    sum_Kxx = (Kxx.sum() - torch.diagonal(Kxx).sum()) / (n * (n - 1))
    sum_Kyy = (Kyy.sum() - torch.diagonal(Kyy).sum()) / (m * (m - 1))
    mean_Kxy = Kxy.mean()

    mmd2 = float((sum_Kxx + sum_Kyy - 2.0 * mean_Kxy).item())
    return {"mmd2": mmd2, "sigma": float(sigma), "n_used": int(n), "m_used": int(m)}

def _grade_realism(rrmse: float, r2: float) -> str:
    # Your pragmatic bands for recon quality → single label
    if rrmse <= 0.02 and r2 >= 0.98: return "perfect"
    if 0.02 < rrmse <= 0.05 and r2 >= 0.95: return "very good"
    if 0.05 < rrmse <= 0.10 and r2 >= 0.90: return "good"
    if 0.10 < rrmse <= 0.20 and r2 >= 0.75: return "mediocre"
    if 0.20 < rrmse <= 0.35 and r2 >= 0.50: return "bad"
    return "very bad"

def _save_csv(df: pd.DataFrame, path: os.PathLike) -> str:
    df.to_csv(path, index=False)
    return str(path)

def train_cvae_cnf_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.DataFrame] = None,
    condition_col: str = "type",
    index_col: str = "post_cleaning_index",
    context_cols: list = None,
    config_filename: str = "cvae_cnf.yaml",
    base_name: str = "cvae_cnf",
    device: str = "cuda",
    verbose: bool = True,
    seed: Optional[int] = None,
):
    # Ensure determinism
    seed = _maybe_set_seed(seed)

    config = load_yaml_config(config_filename)
    enc_cfg  = config["encoder"]
    dec_cfg  = config["decoder"]
    flow_cfg = config["flow"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    interp_cfg = config.get("interpretability", {})
    log_training = train_cfg.get("log_training", True)

    # ---- optional schedules ----
    sched_cfg = train_cfg.get("schedules", {})
    FLOW_ACTIVE = bool(flow_cfg.get("active", True))
    ENABLE_MSE_WARMUP = bool(sched_cfg.get("enable_mse_warmup", False))
    MSE_WARMUP_EPOCHS = int(sched_cfg.get("mse_warmup_epochs", 0))
    ENABLE_FLOW_FREEZE = bool(sched_cfg.get("enable_flow_freeze", False))
    FLOW_FREEZE_EPOCHS = int(sched_cfg.get("flow_freeze_epochs", 0))
    ENABLE_KL_ANNEAL = bool(sched_cfg.get("enable_kl_anneal", False))
    KL_START_EPOCH = int(sched_cfg.get("kl_start_epoch", 1))
    KL_END_EPOCH = int(sched_cfg.get("kl_end_epoch", 1))
    ENABLE_SAMPLE_POSTERIOR_AFTER_WARMUP = bool(sched_cfg.get("enable_sample_posterior_after_warmup", False))
    ENABLE_DEC_LOGSTD_TIGHTEN = bool(sched_cfg.get("enable_dec_logstd_tighten", False))
    DEC_LOGSTD_TIGHT_RANGE = tuple(sched_cfg.get("dec_logstd_tight_range", [-3.0, -1.0]))
    DEC_LOGSTD_TIGHT_EPOCHS = int(sched_cfg.get("dec_logstd_tight_epochs", 0))

    device = torch.device("cuda" if device.lower() == "cuda" and torch.cuda.is_available() else "cpu")

    # ── data
    context_cols = context_cols or [condition_col]
    context_train, features_train, context_names, feature_names = prepare_dataset(
        X_train, y_train, index_col=index_col, context_cols=context_cols
    )
    context_val, features_val, _, _ = prepare_dataset(
        X_val, y_val, index_col=index_col, context_cols=context_cols
    )

    context_train = context_train.to(device)
    context_val   = context_val.to(device)
    features_train = features_train.to(device)
    features_val   = features_val.to(device)

    features_test = None
    context_test = None
    if X_test is not None and y_test is not None:
        context_test, features_test, _, _ = prepare_dataset(
            X_test, y_test, index_col=index_col, context_cols=context_cols
        )
        context_test = context_test.to(device)
        features_test = features_test.to(device)

    input_dim  = features_train.shape[1]
    output_dim = input_dim

    # Ensure 1-D int labels for embeddings
    assert context_train.dim() == 1, "Context must be a single integer label per row for the embedding."
    context_train = context_train.long()
    context_val = context_val.long()
    if 'context_test' in locals() and context_test is not None:
        context_test = context_test.long()

    num_classes = int(context_train.max().item()) + 1
    latent_dim  = enc_cfg["latent_dim"]

    # ---- realism scaling (compute once on training set) ----
    # Use IQR; fallback to std if IQR is tiny. Works on your standardized data too.
    with torch.no_grad():
        qt = torch.quantile(features_train, torch.tensor([0.25, 0.75], device=features_train.device), dim=0)
        iqr = (qt[1] - qt[0]).clamp_min(1e-6)
        std = features_train.std(dim=0, unbiased=False).clamp_min(1e-6)
        realism_scale = torch.where(iqr > 1e-5, iqr, std)  # (D,)

    # ---- stable MMD bandwidth sigma (compute once on a subsample) ----
    def _pairwise_sq_dists(a):
        a2 = (a * a).sum(1, keepdim=True)
        return (a2 + a2.t() - 2 * a @ a.t()).clamp_min(0.0)

    with torch.no_grad():
        # sample up to ~2048 training points for a robust median
        S = min(2048, features_train.shape[0])
        idx = torch.randperm(features_train.shape[0], device=features_train.device)[:S]
        Xs = features_train[idx]
        d2 = _pairwise_sq_dists(Xs)
        iu, ju = torch.triu_indices(d2.size(0), d2.size(1), offset=1)
        med = torch.median(d2[iu, ju])
        realism_sigma_fixed = torch.sqrt(med.clamp_min(1e-12)).item()  # float is fine

    # ── model
    encoder = build_encoder(cfg=config, input_dim=input_dim, num_classes=num_classes, device=device)
    decoder = build_decoder(cfg=config, output_dim=output_dim, num_classes=num_classes, latent_dim=latent_dim, device=device)
    flow    = build_flow(cfg=config, latent_dim=latent_dim, num_classes=num_classes, device=device)
    model   = CVAECNF(encoder=encoder, decoder=decoder, flow=flow, device=device)

    # optional actnorm warmup pass
    if FLOW_ACTIVE and flow_cfg.get("use_actnorm", False):
        model.eval()
        with torch.no_grad():
            try:
                _ = flexible_cvae_cnf_loss_from_model(model, features_train[:64], context_train[:64], **loss_cfg)
            except Exception:
                pass
        model.train()

    # optimizer (no wd/clipping here per your current plan)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    _, dataloader = get_dataloader(
        context_train, features_train,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        seed=seed
    )

    # ── training state
    best_val_loss = float("inf")
    best_model_state = None
    state_loss_epoch = None
    best_metrics = {
        "train_recon_rrmse": None,
        "val_recon_rrmse": None,
        "train_r2": None,
        "val_r2": None,
    }

    early_stopping_patience = train_cfg.get("early_stopping_patience", 40)
    lr_decay_patience       = train_cfg.get("lr_decay_patience", 16)
    min_improvement         = train_cfg.get("min_improvement", 0.04)
    min_improvement_floor   = train_cfg.get("min_improvement_floor", 0.0025)
    lr_decay_factor         = train_cfg.get("lr_decay_factor", 0.5)
    lr_patience_factor      = train_cfg.get("lr_patience_factor", 0.8)

    epochs_no_improve = 0
    lr_decay_wait     = 0
    lr                = train_cfg["learning_rate"]
    lr_factor         = 1.0
    patience_factor   = 1.0
    initial_lr_decay_patience = lr_decay_patience
    total_epochs = train_cfg["num_epochs"]

    # originals for resets
    ORIG_EARLY_STOP_PATIENCE = early_stopping_patience
    ORIG_LR_DECAY_PATIENCE = lr_decay_patience
    ORIG_MIN_IMPROVEMENT = min_improvement
    ORIG_LR = lr

    ORIG_KL_START = KL_START_EPOCH
    ORIG_KL_END = KL_END_EPOCH
    KL_DURATION = max(0, ORIG_KL_END - ORIG_KL_START)

    RESET_ON_NLL_START = bool(sched_cfg.get("reset_scheduler_on_nll_start", False))
    INTERCEPT_ES_IN_WARMUP = bool(sched_cfg.get("intercept_early_stop_during_warmup", False))
    RESET_PHASE_BASELINE = bool(sched_cfg.get("reset_phase_baseline_on_nll_start", True))

    # track transition to NLL
    prev_used_nll = False if (ENABLE_MSE_WARMUP and MSE_WARMUP_EPOCHS > 0) else True
    nll_started_epoch = None
    archived_phase_bests = {}

    # logs/dirs
    run_dir, run_name, log_file_path, snapshots_dir = setup_training_logs_and_dirs(
        base_name=base_name,
        config_filename=config_filename,
        config=config,
        verbose=verbose,
        should_save_states=train_cfg.get("save_states", False),
        log_training=log_training,
        subdir="cvae_cnf",
    )
    flowpre_log(f"Using device: {device}", log_training=log_training, filename_or_path=log_file_path, verbose=verbose)
    flowpre_log(f"🎲 Using seed: {seed}", log_training=log_training, filename_or_path=log_file_path, verbose=verbose)
    flowpre_log(f"🌊 Using flow: {FLOW_ACTIVE}", log_training=log_training, filename_or_path=log_file_path, verbose=verbose)

    def _set_flow_requires_grad(model, requires_grad: bool):
        for p in model.flow.parameters():
            p.requires_grad = requires_grad

    def _reset_scheduler_state():
        nonlocal lr, lr_factor, patience_factor, lr_decay_patience
        nonlocal epochs_no_improve, lr_decay_wait, min_improvement, early_stopping_patience
        nonlocal initial_lr_decay_patience
        lr = ORIG_LR
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        lr_factor = 1.0
        patience_factor = 1.0
        lr_decay_patience = ORIG_LR_DECAY_PATIENCE
        initial_lr_decay_patience = ORIG_LR_DECAY_PATIENCE
        min_improvement = ORIG_MIN_IMPROVEMENT
        early_stopping_patience = ORIG_EARLY_STOP_PATIENCE
        epochs_no_improve = 0
        lr_decay_wait = 0
        flowpre_log(
            f"🔁 Reset scheduler: lr={lr:.6f}, lr_decay_patience={lr_decay_patience}, "
            f"min_improvement={min_improvement:.4f}, early_stop_patience={early_stopping_patience}",
            filename_or_path=log_file_path, verbose=verbose
        )

    def _reset_phase_baseline(label: str):
        nonlocal best_val_loss, state_loss_epoch
        archived_phase_bests[label] = {"loss": best_val_loss, "epoch": state_loss_epoch}
        best_val_loss = float("inf")
        state_loss_epoch = None
        flowpre_log("🧼 Reset phase baseline: best_val_loss := +inf (improvements re-based).",
                    filename_or_path=log_file_path, verbose=verbose)

    def _force_switch_to_nll(next_epoch_idx: int):
        nonlocal MSE_WARMUP_EPOCHS, KL_START_EPOCH, KL_END_EPOCH, prev_used_nll
        MSE_WARMUP_EPOCHS = next_epoch_idx - 1
        _set_flow_requires_grad(model, True)
        KL_START_EPOCH = next_epoch_idx
        KL_END_EPOCH = KL_START_EPOCH + KL_DURATION
        prev_used_nll = False

    # ── loop
    model.train()
    for epoch in range(total_epochs):
        model.train()
        E = epoch + 1

        # flow freeze schedule
        if FLOW_ACTIVE and ENABLE_FLOW_FREEZE:
            if E == 1:
                _set_flow_requires_grad(model, False)
                flow_frozen = True
                flowpre_log("🧊 Flow frozen for warmup", filename_or_path=log_file_path, verbose=verbose)
            if E == FLOW_FREEZE_EPOCHS + 1:
                _set_flow_requires_grad(model, True)
                flow_frozen = False
                flowpre_log("🔥 Flow unfrozen", filename_or_path=log_file_path, verbose=verbose)
        else:
            flow_frozen = False

        # effective loss cfg
        effective_loss_cfg = dict(loss_cfg)
        use_recon_nll = bool(effective_loss_cfg.get("use_recon_nll", True))
        sample_posterior_eff = bool(effective_loss_cfg.get("sample_posterior", True))
        if ENABLE_MSE_WARMUP and E <= MSE_WARMUP_EPOCHS:
            use_recon_nll = False
            sample_posterior_eff = False

        kl_target = float(effective_loss_cfg.get("kl_weight", 0.0))
        kl_weight_eff = kl_target
        if ENABLE_KL_ANNEAL:
            if E < KL_START_EPOCH:
                kl_weight_eff = 0.0
            else:
                t = min(1.0, (E - KL_START_EPOCH) / max(1, KL_END_EPOCH - KL_START_EPOCH))
                kl_weight_eff = kl_target * t

        if ENABLE_SAMPLE_POSTERIOR_AFTER_WARMUP and (not (ENABLE_MSE_WARMUP and E <= MSE_WARMUP_EPOCHS)):
            sample_posterior_eff = bool(loss_cfg.get("sample_posterior", True))

        clamp_dec_range_eff = effective_loss_cfg.get("clamp_dec_logstd_range", None)
        if ENABLE_DEC_LOGSTD_TIGHTEN:
            nll_start_epoch = (MSE_WARMUP_EPOCHS if ENABLE_MSE_WARMUP else 0) + 1
            if use_recon_nll and E < nll_start_epoch + DEC_LOGSTD_TIGHT_EPOCHS:
                clamp_dec_range_eff = DEC_LOGSTD_TIGHT_RANGE

        this_used_nll = bool(use_recon_nll)
        if this_used_nll and not prev_used_nll:
            nll_started_epoch = E
            flowpre_log(f"🚦 Entering NLL phase at epoch {E}", filename_or_path=log_file_path, verbose=verbose)
            if RESET_PHASE_BASELINE:
                _reset_phase_baseline(label="warmup")
            if RESET_ON_NLL_START:
                _reset_scheduler_state()
        prev_used_nll = this_used_nll

        effective_loss_cfg.update({
            "use_recon_nll": use_recon_nll,
            "kl_weight": kl_weight_eff,
            "sample_posterior": sample_posterior_eff,
            "clamp_dec_logstd_range": clamp_dec_range_eff,
        })

        total_loss = 0.0
        epoch_rrmse_list = []
        diagnostics_accum = {}

        for features, context in dataloader:
            features, context = features.to(device), context.to(device)
            if context.dim() > 1:
                context = context.view(-1)
            context = context.long()
            loss_call_kwargs = {
                **effective_loss_cfg,
                "current_epoch": E,
                "realism_scale": realism_scale,
                "realism_sigma": realism_sigma_fixed,
                "realism_subsample": effective_loss_cfg.get("realism_subsample", 256),
                "realism_average_over_classes": True,
            }

            loss, diagnostics, rrmse_recon = flexible_cvae_cnf_loss_from_model(
                model, features, context, **loss_call_kwargs
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            epoch_rrmse_list.append(rrmse_recon)
            for k, v in diagnostics.items():
                diagnostics_accum.setdefault(k, []).append(v)

        avg_train_loss   = total_loss / len(dataloader.dataset)
        avg_rrmse_recon  = float(sum(epoch_rrmse_list) / len(epoch_rrmse_list)) if epoch_rrmse_list else float("nan")

        # validation step
        model.eval()
        with torch.no_grad():
            val_loss, val_diagnostics, val_rrmse_recon = flexible_cvae_cnf_loss_from_model(
                model, features_val, context_val,
                **{**loss_call_kwargs, "enable_dec_logstd_reg": False, "enable_realism": False}
            )

        # R² via deterministic mean latents
        with torch.no_grad():
            recon_train = model.reconstruct(features_train, context_train, use_mean_latent=True)
            recon_val   = model.reconstruct(features_val,   context_val,   use_mean_latent=True)
            train_rrmse_epoch, train_r2_epoch = compute_rrmse_r2(features_train, recon_train)
            val_rrmse_epoch,   val_r2_epoch   = compute_rrmse_r2(features_val,   recon_val)

        numeric_diagnostics_accum = {
            k: v for k, v in diagnostics_accum.items()
            if v and all(isinstance(x, (int, float, bool)) for x in v)
        }
        log_epoch_diagnostics(epoch, numeric_diagnostics_accum, log_file_path, verbose)

        flowpre_log(
            f"📉 Epoch {epoch + 1}/{total_epochs} — Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}",
            filename_or_path=log_file_path, verbose=verbose
        )
        flowpre_log(
            f"🔍 Recon RRMSE (Train/Val): {avg_rrmse_recon:.4f} / {val_rrmse_recon:.4f}",
            filename_or_path=log_file_path, verbose=verbose
        )
        flowpre_log(
            f"📈 R² (Train/Val): {train_r2_epoch:.4f} / {val_r2_epoch:.4f}",
            filename_or_path=log_file_path, verbose=verbose
        )

        improvement = (best_val_loss - val_loss) / (abs(best_val_loss) + 1e-8) if best_val_loss < float("inf") else float("inf")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_loss_epoch = epoch + 1
            best_model_state = deepcopy(model.state_dict())
            best_metrics["train_recon_rrmse"] = avg_rrmse_recon
            best_metrics["val_recon_rrmse"]   = float(val_rrmse_recon)
            best_metrics["train_r2"]          = float(train_r2_epoch)
            best_metrics["val_r2"]            = float(val_r2_epoch)
            if improvement >= min_improvement:
                epochs_no_improve = 0
                lr_decay_wait = 0
                if train_cfg.get("save_states", False):
                    snapshot_path = snapshots_dir / f"{run_name}_epoch{epoch + 1}_valloss{val_loss:.2f}.pt"
                    torch.save(best_model_state, snapshot_path)
                    flowpre_log(f"💾 Saved snapshot: {snapshot_path.name}", filename_or_path=log_file_path, verbose=verbose)
        else:
            epochs_no_improve += 1
            lr_decay_wait += 1
            if lr_decay_wait >= lr_decay_patience:
                last_lr = lr
                lr_factor *= lr_decay_factor
                lr = train_cfg["learning_rate"] * lr_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                flowpre_log(f"🔽 LR decay: {last_lr:.6f} → {lr:.6f}", filename_or_path=log_file_path, verbose=verbose)

                old_improvement = min_improvement
                min_improvement = max(min_improvement_floor, min_improvement * lr_decay_factor)
                old_lr_decay_patience = lr_decay_patience
                patience_factor *= lr_patience_factor
                lr_decay_patience = max(5, int(math.ceil(initial_lr_decay_patience * patience_factor)))
                if early_stopping_patience > 5:
                    epochs_no_improve = 0
                lr_decay_wait = 0
                if old_improvement * lr_decay_factor < min_improvement:
                    early_stopping_patience = max(5, early_stopping_patience * 0.5)

            if epochs_no_improve >= early_stopping_patience:
                in_freeze_window = FLOW_ACTIVE and ENABLE_FLOW_FREEZE and (E <= FLOW_FREEZE_EPOCHS)
                not_yet_nll = not this_used_nll
                if INTERCEPT_ES_IN_WARMUP and (in_freeze_window or not_yet_nll):
                    flowpre_log("⏭️ Intercepted ES during warmup/freeze — forcing NLL and resetting.", filename_or_path=log_file_path, verbose=verbose)
                    _force_switch_to_nll(next_epoch_idx=E + 1)
                    if RESET_PHASE_BASELINE: _reset_phase_baseline(label="warmup")
                    _reset_scheduler_state()
                    continue
                else:
                    flowpre_log(f"🛌 Early stopping at epoch {epoch + 1}", filename_or_path=log_file_path, verbose=verbose)
                    break

    flowpre_log(f"✅ Best validation loss: {best_val_loss:.4f} (epoch {state_loss_epoch})",
                filename_or_path=log_file_path, verbose=verbose)
    if "warmup" in archived_phase_bests:
        b = archived_phase_bests["warmup"]
        flowpre_log(f"ℹ️ Warmup-best (MSE phase): loss={b['loss']:.4f} @ epoch {b['epoch']}",
                    filename_or_path=log_file_path, verbose=verbose)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if train_cfg.get("save_states", False):
        final_path = snapshots_dir / f"{run_name}_best_epoch{state_loss_epoch}_valloss{best_val_loss:.2f}.pt"
        torch.save(best_model_state, final_path)

    # === Final recon metrics (deterministic mean z) ===
    with torch.no_grad():
        recon_train = model.reconstruct(features_train, context_train, use_mean_latent=True)
        recon_val   = model.reconstruct(features_val,   context_val,   use_mean_latent=True)
        rrmse_train_final, r2_train_final = compute_rrmse_r2(features_train, recon_train)
        rrmse_val_final,   r2_val_final   = compute_rrmse_r2(features_val,   recon_val)

    flowpre_log(f"🏁 Final Train  — RRMSE: {rrmse_train_final:.4f}, R²: {r2_train_final:.4f}",
                filename_or_path=log_file_path, verbose=verbose)
    flowpre_log(f"🏁 Final Val    — RRMSE: {rrmse_val_final:.4f}, R²: {r2_val_final:.4f}",
                filename_or_path=log_file_path, verbose=verbose)

    # === Optional TEST metrics (if provided) ===
    rrmse_test_final = None
    r2_test_final = None
    if 'features_test' in locals() and features_test is not None:
        with torch.no_grad():
            recon_test = model.reconstruct(features_test, context_test, use_mean_latent=True)
            rrmse_test_final, r2_test_final = compute_rrmse_r2(features_test, recon_test)
        flowpre_log(f"🏁 Final Test   — RRMSE: {rrmse_test_final:.4f}, R²: {r2_test_final:.4f}",
                    filename_or_path=log_file_path, verbose=verbose)

    # === Synthetic samples (match val class labels) ===
    model.eval()
    with torch.no_grad():
        BATCH = 2048
        synth_chunks = []
        for i in range(0, context_val.shape[0], BATCH):
            c_batch = context_val[i:i + BATCH]
            x_s = model.sample(num_samples=c_batch.shape[0], class_labels=c_batch)
            if isinstance(x_s, (list, tuple)):  # (mean, logvar)
                x_s = x_s[0]
            synth_chunks.append(x_s)
        synth_val = torch.cat(synth_chunks, dim=0) if synth_chunks else torch.empty_like(features_val)

    # === Build realism suite (global + per-class) ===
    Xr_val = _to_np(features_val)
    Xh_val = _to_np(recon_val)
    Xs_val = _to_np(synth_val)
    ctx_val_np = _to_np(context_val)

    # Work out input/target split if y was appended to X in prepare_dataset
    y_cols = list(y_val.drop(columns=[index_col], errors="ignore").columns)
    n_targets = len(y_cols)
    D = Xr_val.shape[1]
    idx_inputs  = np.arange(0, D - n_targets) if n_targets > 0 else np.arange(D)
    idx_targets = np.arange(D - n_targets, D)  if n_targets > 0 else np.array([], dtype=int)

    # 1) Per-feature reconstruction errors (val) — GLOBAL
    rrmse_feat = _per_feature_rrmse(Xr_val, Xh_val)
    mae_feat, mse_feat = _per_feature_mae_mse(Xr_val, Xh_val)

    df_recon_feat = pd.DataFrame({
        "feature": feature_names,
        "rrmse": rrmse_feat,
        "mae": mae_feat,
        "mse": mse_feat,
    })
    recon_feat_csv = _save_csv(df_recon_feat, run_dir / f"{run_name}_recon_per_feature.csv")

    recon_summary_all = _summarize_vector(rrmse_feat)
    recon_summary_inputs  = _summarize_vector(rrmse_feat[idx_inputs])  if idx_inputs.size  else {}
    recon_summary_targets = _summarize_vector(rrmse_feat[idx_targets]) if idx_targets.size else {}

    # 2) Synthetic vs Real (val) — GLOBAL distribution distances
    ks_feat  = _ks_all_features(Xr_val, Xs_val)                      # KS per feature
    w1_feat  = _w1_quantile_all_features(Xr_val, Xs_val)             # approx W1 via quantiles
    mom_d    = _moment_deltas(Xr_val, Xs_val)                        # moment deltas (vectors)
    corr_d   = _corr_deltas(Xr_val, Xs_val)                          # corr matrix deltas (scalars)
    mmd      = _mmd_rbf_unbiased(torch.as_tensor(Xr_val, device=device),
                                 torch.as_tensor(Xs_val, device=device),
                                 sigma=None, max_samples=1024)       # kernel MMD^2

    df_synth_feat = pd.DataFrame({
        "feature": feature_names,
        "ks": ks_feat,
        "w1_quantile": w1_feat,
        "mean_abs_diff": mom_d["mean_abs_diff"],
        "std_rel_err": mom_d["std_rel_err"],
        "skew_abs_diff": mom_d["skew_abs_diff"],
        "kurt_abs_diff": mom_d["kurt_abs_diff"],
    })
    synth_feat_csv = _save_csv(df_synth_feat, run_dir / f"{run_name}_synth_vs_real_per_feature.csv")

    ks_summary_all = _summarize_vector(ks_feat)
    w1_summary_all = _summarize_vector(w1_feat)
    ks_frac_010 = float(np.mean(ks_feat <= 0.10))
    ks_frac_050 = float(np.mean(ks_feat <= 0.50))

    # 3) Per-CLASS metrics (both recon & synth-vs-real)
    classes = np.unique(ctx_val_np)
    rows_recon_by_class = []
    rows_synth_feat_by_class = []
    recon_per_class_summary = {}
    synth_vs_real_per_class = {}

    # prepare tensors for classwise RRMSE/R2
    features_val_t = features_val
    recon_val_t    = recon_val
    context_val_t  = context_val

    for cls in classes:
        mask_np = (ctx_val_np == cls)
        if mask_np.sum() == 0:
            continue
        # --- Recon per-feature (class) ---
        Xr_c = Xr_val[mask_np]
        Xh_c = Xh_val[mask_np]
        rrmse_feat_c = _per_feature_rrmse(Xr_c, Xh_c)
        mae_feat_c, mse_feat_c = _per_feature_mae_mse(Xr_c, Xh_c)
        # accumulate featurewise rows
        for f_idx, fname in enumerate(feature_names):
            rows_recon_by_class.append({
                "class": int(cls),
                "feature": fname,
                "rrmse": float(rrmse_feat_c[f_idx]),
                "mae": float(mae_feat_c[f_idx]),
                "mse": float(mse_feat_c[f_idx]),
            })
        # scalar recon metrics (RRMSE/R² over all dims for this class)
        mask_t = (context_val_t == int(cls))
        x_true_c = features_val_t[mask_t]
        x_hat_c  = recon_val_t[mask_t]
        rrmse_c, r2_c = compute_rrmse_r2(x_true_c, x_hat_c)
        recon_per_class_summary[int(cls)] = {
            "rrmse": float(rrmse_c),
            "r2": float(r2_c),
            "grade_from_recon": _grade_realism(rrmse_c, r2_c),
            "rrmse_feature_summary": _summarize_vector(rrmse_feat_c),
            "rrmse_inputs_summary":  _summarize_vector(rrmse_feat_c[idx_inputs])  if idx_inputs.size  else {},
            "rrmse_targets_summary": _summarize_vector(rrmse_feat_c[idx_targets]) if idx_targets.size else {},
        }

        # --- Synth vs Real (class) ---
        Xs_c = Xs_val[mask_np]
        ks_c  = _ks_all_features(Xr_c, Xs_c)
        w1_c  = _w1_quantile_all_features(Xr_c, Xs_c)
        mom_c = _moment_deltas(Xr_c, Xs_c)
        corr_c = _corr_deltas(Xr_c, Xs_c)
        mmd_c  = _mmd_rbf_unbiased(torch.as_tensor(Xr_c, device=device),
                                   torch.as_tensor(Xs_c, device=device),
                                   sigma=None, max_samples=1024)
        # per-feature rows
        for f_idx, fname in enumerate(feature_names):
            rows_synth_feat_by_class.append({
                "class": int(cls),
                "feature": fname,
                "ks": float(ks_c[f_idx]),
                "w1_quantile": float(w1_c[f_idx]),
                "mean_abs_diff": float(mom_c["mean_abs_diff"][f_idx]),
                "std_rel_err": float(mom_c["std_rel_err"][f_idx]),
                "skew_abs_diff": float(mom_c["skew_abs_diff"][f_idx]),
                "kurt_abs_diff": float(mom_c["kurt_abs_diff"][f_idx]),
            })

        synth_vs_real_per_class[int(cls)] = {
            "ks_feature_summary": _summarize_vector(ks_c),
            "w1_feature_summary": _summarize_vector(w1_c),
            "ks_fraction_below_0p10": float(np.mean(ks_c <= 0.10)),
            "ks_fraction_below_0p50": float(np.mean(ks_c <= 0.50)),
            "moments_feature_summaries": {
                "mean_abs_diff": _summarize_vector(mom_c["mean_abs_diff"]),
                "std_rel_err":   _summarize_vector(mom_c["std_rel_err"]),
                "skew_abs_diff": _summarize_vector(mom_c["skew_abs_diff"]),
                "kurt_abs_diff": _summarize_vector(mom_c["kurt_abs_diff"]),
            },
            "correlation_deltas": corr_c,   # pearson/spearman fro & rel
            "mmd_rbf_unbiased": mmd_c,      # {'mmd2','sigma','n_used','m_used'}
        }

    # dump by-class CSVs
    df_recon_feat_by_class = pd.DataFrame(rows_recon_by_class)
    recon_feat_by_class_csv = _save_csv(df_recon_feat_by_class, run_dir / f"{run_name}_recon_per_feature_by_class.csv")

    df_synth_feat_by_class = pd.DataFrame(rows_synth_feat_by_class)
    synth_feat_by_class_csv = _save_csv(df_synth_feat_by_class, run_dir / f"{run_name}_synth_vs_real_per_feature_by_class.csv")

    # Realism grade (global) from recon
    realism_grade = _grade_realism(rrmse_val_final, r2_val_final)

    # === Optional interpretability (unchanged) ===
    if interp_cfg.get("save_influence", False):
        try:
            influence = compute_cvae_feature_influence(
                model=model,
                features=features_train,
                context=context_train,
                feature_names=feature_names,
                per_class=True,
                use_classwise_x0=True,
                abs_value=True,
                normalize_by_input=True,
            )
            with open(run_dir / f"{run_name}_influence.json", "w") as f:
                json.dump(influence, f, indent=2)
            flowpre_log(f"🧠 Influence saved: {run_name}_influence.json",
                        filename_or_path=log_file_path, verbose=verbose)
        except Exception as e:
            flowpre_log(f"ℹ️ Skipping influence (error): {e}",
                        filename_or_path=log_file_path, verbose=verbose)

    # === Save results (now includes per-class blocks) ===
    if train_cfg.get("save_results", False):
        results = {
            "best_epoch": state_loss_epoch,
            "total_epochs": epoch + 1,
            "seed": int(seed),
            "train": {
                "recon_rrmse": round(rrmse_train_final, 6),
                "r2": round(r2_train_final, 6),
            },
            "val": {
                "recon_rrmse": round(rrmse_val_final, 6),
                "r2": round(r2_val_final, 6),
            },
            "realism": {
                "grade_from_recon": realism_grade,
                "recon_per_feature_csv": recon_feat_csv,
                "recon_per_feature_by_class_csv": recon_feat_by_class_csv,
                "synth_vs_real_per_feature_csv": synth_feat_csv,
                "synth_vs_real_per_feature_by_class_csv": synth_feat_by_class_csv,
                "recon_rrmse_feature_summary": {
                    "all": recon_summary_all,
                    "inputs": recon_summary_inputs,
                    "targets": recon_summary_targets,
                },
                "recon_per_class": recon_per_class_summary,   # includes rrmse/r2/grade + summaries
                "synth_vs_real": {
                    "ks_feature_summary": ks_summary_all,
                    "w1_feature_summary": w1_summary_all,
                    "ks_fraction_below_0p10": ks_frac_010,
                    "ks_fraction_below_0p50": ks_frac_050,
                    "moments_feature_summaries": {
                        "mean_abs_diff": _summarize_vector(mom_d["mean_abs_diff"]),
                        "std_rel_err":   _summarize_vector(mom_d["std_rel_err"]),
                        "skew_abs_diff": _summarize_vector(mom_d["skew_abs_diff"]),
                        "kurt_abs_diff": _summarize_vector(mom_d["kurt_abs_diff"]),
                    },
                    "correlation_deltas": corr_d,
                    "mmd_rbf_unbiased": mmd,
                    "per_class": synth_vs_real_per_class,     # full per-class suite
                },
            },
            "files": {
                "config": str(run_dir / f"{run_name}.yaml"),
                "recon_per_feature_csv": recon_feat_csv,
                "recon_per_feature_by_class_csv": recon_feat_by_class_csv,
                "synth_vs_real_per_feature_csv": synth_feat_csv,
                "synth_vs_real_per_feature_by_class_csv": synth_feat_by_class_csv,
            }
        }

        if rrmse_test_final is not None:
            results["test"] = {
                "recon_rrmse": round(rrmse_test_final, 6),
                "r2": round(r2_test_final, 6),
            }

        results_path = run_dir / f"{run_name}_results.yaml"
        with open(results_path, "w") as f:
            yaml.dump(results, f, sort_keys=False)
        flowpre_log(f"📝 Saved results to: {results_path}",
                    filename_or_path=log_file_path, verbose=verbose)

    # copy config
    shutil.copy(ROOT_PATH / "config" / config_filename, run_dir / f"{run_name}.yaml")
    flowpre_log(f"✅ Config saved under {run_dir}",
                filename_or_path=log_file_path, verbose=verbose)

    # save model if enabled
    if train_cfg.get("save_model", False):
        torch.save(model.state_dict(), run_dir / f"{run_name}.pt")
        flowpre_log(f"💾 Model saved: {run_name}.pt",
                    filename_or_path=log_file_path, verbose=verbose)

    return model
