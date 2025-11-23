# losses/flowgen_loss.py

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import contextlib

from torch.cuda.amp import autocast
EPS = 1e-12

def _autocast_ctx(device: torch.device, *, enabled: bool = True):
    """
    Device-agnostic autocast:
      - CUDA: bf16 if supported, else fp16
      - MPS:  fp16
      - CPU/other: disabled
    """
    dev_type = device.type
    if not enabled or dev_type not in ("cuda", "mps"):
        return torch.autocast(device_type="cpu", enabled=False)  # no-op

    if dev_type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:  # "mps"
        return torch.autocast(device_type="mps", dtype=torch.float16)


# ----------------------------- small helpers (torch) -----------------------------

def _sqnorm(x: torch.Tensor) -> torch.Tensor:
    return (x * x).sum()

def _median_heuristic_sigma(xa: torch.Tensor, xb: torch.Tensor) -> float:
    x = torch.cat([xa, xb], dim=0)
    xx = (x * x).sum(dim=1, keepdim=True)
    d2 = (xx + xx.t() - 2.0 * (x @ x.t())).clamp_min_(0.0)
    iu, ju = torch.triu_indices(d2.size(0), d2.size(1), offset=1)
    med = torch.median(d2[iu, ju])
    return float(torch.sqrt(med.clamp_min(1e-12)).item())

def _mmd_rbf_biased(xa: torch.Tensor, xb: torch.Tensor, sigma: Optional[float] = None):
    if sigma is None:
        sigma = _median_heuristic_sigma(xa, xb)
    gamma = 1.0 / (2.0 * (sigma ** 2) + 1e-12)

    def _kmat(z):
        zz = (z * z).sum(dim=1, keepdim=True)
        d2 = (zz + zz.t() - 2.0 * (z @ z.t())).clamp_min_(0.0)
        return torch.exp(-gamma * d2)

    Ka = _kmat(xa)
    Kb = _kmat(xb)
    Kab = torch.exp(-gamma * ((xa * xa).sum(dim=1, keepdim=True)
                              + (xb * xb).sum(dim=1, keepdim=True).t()
                              - 2.0 * (xa @ xb.t())).clamp_min_(0.0))
    mmd2 = Ka.mean() + Kb.mean() - 2.0 * Kab.mean()
    return mmd2, sigma

def _center_and_std(x: torch.Tensor):
    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    sd = torch.sqrt((xc * xc).mean(dim=0) + EPS)
    return mu, xc, sd

def _pearson_corr(x: torch.Tensor) -> torch.Tensor:
    _, xc, sd = _center_and_std(x)
    z = xc / (sd.unsqueeze(0) + EPS)
    C = (z.t() @ z) / x.size(0)
    return C.clamp_(-1.0, 1.0)

def _ranks(x: torch.Tensor) -> torch.Tensor:
    N, D = x.shape
    idx = torch.argsort(x, dim=0)
    ranks = torch.empty_like(x, dtype=torch.float32)
    arange = torch.arange(1, N + 1, device=x.device, dtype=torch.float32).unsqueeze(1).expand_as(idx)
    ranks.scatter_(0, idx, arange)
    return ranks

def _spearman_corr(x: torch.Tensor) -> torch.Tensor:
    return _pearson_corr(_ranks(x))

def _fro_rel(Cr: torch.Tensor, Cs: torch.Tensor):
    diff = Cr - Cs
    fro_abs = torch.linalg.norm(diff, ord='fro')
    fro_rel = fro_abs / (torch.linalg.norm(Cr, ord='fro') + EPS)
    return fro_abs, fro_rel

def _iqr(x: torch.Tensor) -> torch.Tensor:
    q25 = x.quantile(0.25, dim=0)
    q75 = x.quantile(0.75, dim=0)
    return (q75 - q25).clamp_min(1e-8)

def _w1_1d_sorted(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact 1D Wasserstein distance for equal-weight samples via sorted L1.
    Works in float32 and is stable even with AMP outside the call site."""
    n = min(a.numel(), b.numel())
    if n == 0:
        return torch.zeros((), device=a.device, dtype=torch.float32)
    a = torch.sort(a.to(torch.float32)[:n]).values
    b = torch.sort(b.to(torch.float32)[:n]).values
    return torch.mean(torch.abs(a - b))


def _ks_soft_1d(real: torch.Tensor, synth: torch.Tensor, grid_points: int = 64, tau: float = 0.05) -> torch.Tensor:
    """Soft KS on a robust (1%, 99%) quantile grid in float32 to reduce outlier sensitivity."""
    x = torch.cat([real, synth], dim=0).to(torch.float32)
    gmin = torch.quantile(x, 0.01)
    gmax = torch.quantile(x, 0.99)
    if (not torch.isfinite(gmin)) or (not torch.isfinite(gmax)) or (gmax <= gmin):
        return torch.zeros((), device=x.device, dtype=torch.float32)
    grid = torch.linspace(gmin, gmax, grid_points, device=x.device, dtype=torch.float32)
    Fr = torch.sigmoid((grid.unsqueeze(1) - real.to(torch.float32).unsqueeze(0)) / tau).mean(dim=1)
    Fs = torch.sigmoid((grid.unsqueeze(1) - synth.to(torch.float32).unsqueeze(0)) / tau).mean(dim=1)
    return torch.max(torch.abs(Fr - Fs))


def _soft_ecdf(x: torch.Tensor, grid: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    return torch.sigmoid((grid.unsqueeze(1) - x.unsqueeze(0)) / tau).mean(dim=1)

def _ks_w1_soft(real: torch.Tensor, synth: torch.Tensor, grid_points: int = 64, tau: float = 0.05):
    # Robust, FP32, autocast-safe implementation.
    # Same return values and arguments as before.
    with torch.autocast(device_type=real.device.type, enabled=False):
        real32  = real.to(torch.float32)
        synth32 = synth.to(torch.float32)

        # Robust grid from combined quantiles + modest margin
        both = torch.cat([real32, synth32], dim=0)
        q_lo = torch.quantile(both, 0.01)
        q_hi = torch.quantile(both, 0.99)
        iqr  = (torch.quantile(both, 0.75) - torch.quantile(both, 0.25)).clamp_min(1e-6)
        # small margin (10% of IQR), but don’t let it be zero
        margin = 0.1 * iqr

        gmin = (q_lo - margin).item()
        gmax = (q_hi + margin).item()
        # if degenerate, widen a touch to avoid zero-length grid
        if not (gmax > gmin):
            gmax = gmin + 1.0

        grid = torch.linspace(gmin, gmax, int(max(4, grid_points)), device=real32.device, dtype=torch.float32)

        # clamp values to grid support to avoid exploding tails
        real32  = real32.clamp(min=gmin, max=gmax)
        synth32 = synth32.clamp(min=gmin, max=gmax)

        Fr = _soft_ecdf(real32,  grid, tau=float(tau))
        Fs = _soft_ecdf(synth32, grid, tau=float(tau))

        diff = (Fr - Fs).abs()
        ks = diff.max()
        w1 = torch.trapz(diff, grid)

        return ks.to(real.dtype), w1.to(real.dtype)

def _ks_w1_matrix(
    real: torch.Tensor,
    synth: torch.Tensor,
    grid_points: int = 64,
    tau: float = 0.05,
    norm: str = "iqr",
    denom_override: Optional[torch.Tensor] = None,
):
    """Per-feature KS (soft) and W1, in float32. W1 is normalized by a provided
    global per-feature scale when denom_override is given; otherwise uses a local
    scale from 'real'. Returned: (ks_mean, ks_median, w1_mean, w1_median)."""
    real = real.to(torch.float32)
    synth = synth.to(torch.float32)

    D = real.shape[1]
    if D == 0:
        z = torch.zeros(1, device=real.device, dtype=torch.float32)
        return z.mean(), z.median(), z.mean(), z.median()

    if denom_override is None:
        denom = _iqr(real) if (norm == "iqr") else torch.sqrt((real ** 2).mean(dim=0) + EPS)
    else:
        denom = denom_override.to(torch.float32)

    denom = denom.clamp_min(1e-6)

    ks_vals = []
    w1_vals = []
    for j in range(D):
        ks = _ks_soft_1d(real[:, j], synth[:, j], grid_points=grid_points, tau=tau)
        w1 = _w1_1d_sorted(real[:, j], synth[:, j]) / denom[j]
        ks_vals.append(ks)
        w1_vals.append(w1)

    ks_t = torch.stack(ks_vals)
    w1_t = torch.stack(w1_vals)
    return ks_t.mean(), ks_t.median(), w1_t.mean(), w1_t.median()

def _perdim_w1_normed(
    real: torch.Tensor,
    synth: torch.Tensor,
    *,
    norm: str,
    denom_override: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      w1_normed: [D] per-dimension W1 normalized (iqr or rms)
      denom:     [D] denominators actually used (for logging)
    """
    real = real.to(torch.float32); synth = synth.to(torch.float32)
    if denom_override is None:
        denom = (_iqr(real) if (norm == "iqr")
                 else torch.sqrt((real ** 2).mean(dim=0) + EPS)).to(torch.float32)
    else:
        denom = denom_override.to(torch.float32)
    denom = denom.clamp_min(1e-6)

    D = real.shape[1]
    if D == 0:
        return torch.zeros(0, device=real.device, dtype=torch.float32), denom

    w1 = []
    for j in range(D):
        w1_j = _w1_1d_sorted(real[:, j], synth[:, j])
        w1.append(w1_j / denom[j])
    return torch.stack(w1), denom

def _softclip_asinh(x: torch.Tensor, s: float) -> torch.Tensor:
    """
    Smooth clamp: ~linear when |x|<<s, ~logarithmic growth when |x|>>s.
    Keeps gradients well-behaved. s>0 is the “knee”.
    """
    if s <= 0:
        return x
    return s * torch.asinh(x / s)

# ===================== (A) joint realism, fully differentiable =====================

def _normalize_like_real_xy(model,
                            xr: torch.Tensor, yr: torch.Tensor,
                            xs: torch.Tensor, ys: torch.Tensor,
                            *, norm: str | None = "iqr") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardize concatenated [X;Y] using denominators computed from **real**.
    Returns (zr_std, zs_std, denom) in fp32. Differentiable w.r.t. xs, ys.
    """
    with torch.autocast(device_type=xr.device.type, enabled=False):
        zr = model._concat_xy(xr, yr).to(torch.float32)
        zs = model._concat_xy(xs, ys).to(torch.float32)

        if norm is None:
            denom = torch.ones(zr.shape[1], device=zr.device, dtype=torch.float32)
        elif norm == "iqr":
            denom = _iqr(zr).clamp_min(1e-6)
        elif norm == "rms":
            denom = torch.sqrt((zr ** 2).mean(dim=0) + EPS).clamp_min(1e-6)
        else:
            raise ValueError("norm must be None, 'iqr', or 'rms'.")

        return zr / denom, zs / denom, denom


def _mmd_joint_xy_ms(model,
                     xr: torch.Tensor, yr: torch.Tensor,
                     xs: torch.Tensor, ys: torch.Tensor,
                     *,
                     norm: str | None = "iqr",
                     sigma: float | None = None,
                     scales: tuple[float, ...] = (0.5, 1.0, 2.0)) -> tuple[torch.Tensor, float]:
    """
    Multi-scale RBF MMD^2 on concatenated [X;Y] (standardized by real).
    Returns (mmd2, base_sigma). 100% differentiable (no .detach/.item on loss).
    """
    with torch.autocast(device_type=xr.device.type, enabled=False):
        zr, zs, _ = _normalize_like_real_xy(model, xr, yr, xs, ys, norm=norm)

        if sigma is None:
            sigma = _median_heuristic_sigma(zr, zs)

        # pairwise squared distances (computed once)
        aa = (zr * zr).sum(dim=1, keepdim=True)
        bb = (zs * zs).sum(dim=1, keepdim=True)
        d2_aa = (aa + aa.t() - 2.0 * (zr @ zr.t())).clamp_min_(0.0)
        d2_bb = (bb + bb.t() - 2.0 * (zs @ zs.t())).clamp_min_(0.0)
        d2_ab = (aa + bb.t() - 2.0 * (zr @ zs.t())).clamp_min_(0.0)

        Kaa = 0.0; Kbb = 0.0; Kab = 0.0
        for s in scales:
            sig = max(float(sigma) * float(s), 1e-6)
            gamma = 1.0 / (2.0 * (sig ** 2) + 1e-12)
            Kaa = Kaa + torch.exp(-gamma * d2_aa)
            Kbb = Kbb + torch.exp(-gamma * d2_bb)
            Kab = Kab + torch.exp(-gamma * d2_ab)

        Kaa = Kaa / len(scales); Kbb = Kbb / len(scales); Kab = Kab / len(scales)
        mmd2 = Kaa.mean() + Kbb.mean() - 2.0 * Kab.mean()
        return mmd2, float(sigma)


# ===================== (E) XY-block correlation monitoring =====================

def _pearson_xyblock_fro_gap(model,
                             xr: torch.Tensor, yr: torch.Tensor,
                             xs: torch.Tensor, ys: torch.Tensor,
                             *, relative: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Frobenius gap between **cross** blocks Corr(X,Y) real vs synth using Pearson.
    Returns (fro_abs, fro_rel). Differentiable; useful as a metric or loss.
    """
    with torch.autocast(device_type=xr.device.type, enabled=False):
        zr = model._concat_xy(xr, yr).to(torch.float32)
        zs = model._concat_xy(xs, ys).to(torch.float32)
        Dx = xr.shape[1]

        Cr = _pearson_corr(zr)
        Cs = _pearson_corr(zs)

        Cr_xy = Cr[:Dx, Dx:]
        Cs_xy = Cs[:Dx, Dx:]
        diff = Cr_xy - Cs_xy
        fro_abs = torch.linalg.norm(diff, ord='fro')
        fro_rel = fro_abs / (torch.linalg.norm(Cr_xy, ord='fro') + EPS) if relative else fro_abs
        return fro_abs, fro_rel


def _soft_ranks(x: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """
    Differentiable 'soft' ranks per feature via pairwise sigmoids.
    x: [N, D] -> ranks in [1, N] (fp32), gradient flows to x.
    """
    x = x.to(torch.float32)
    N, D = x.shape
    ranks = []
    for j in range(D):
        col = x[:, j].unsqueeze(1)                 # [N,1]
        P = torch.sigmoid((col - col.t()) / float(tau))  # [N,N], P[i,k]≈1 if col[i]>col[k]
        ranks.append(1.0 + P.sum(dim=1))           # expected rank
    return torch.stack(ranks, dim=1)               # [N,D]


def _softspearman_xyblock_fro_gap(model,
                                  xr: torch.Tensor, yr: torch.Tensor,
                                  xs: torch.Tensor, ys: torch.Tensor,
                                  *, tau: float = 0.05, relative: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Spearman-style XY-block Frobenius gap using **soft** ranks (differentiable).
    Returns (fro_abs, fro_rel). Use for logging or as a gentle penalty.
    """
    with torch.autocast(device_type=xr.device.type, enabled=False):
        zr = model._concat_xy(xr, yr).to(torch.float32)
        zs = model._concat_xy(xs, ys).to(torch.float32)
        Dx = xr.shape[1]

        Cr = _pearson_corr(_soft_ranks(zr, tau))
        Cs = _pearson_corr(_soft_ranks(zs, tau))

        Cr_xy = Cr[:Dx, Dx:]
        Cs_xy = Cs[:Dx, Dx:]
        diff = Cr_xy - Cs_xy
        fro_abs = torch.linalg.norm(diff, ord='fro')
        fro_rel = fro_abs / (torch.linalg.norm(Cr_xy, ord='fro') + EPS) if relative else fro_abs
        return fro_abs, fro_rel
def _st_clamp(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """
    Per-dimension clamp with straight-through gradient:
    forward clamps to [lo, hi], backward passes gradients from unclamped x.
    lo/hi must be broadcastable to x's shape.
    """
    clamped = torch.max(torch.min(x, hi), lo)
    return x + (clamped - x).detach()

def _mmd_penalty(mmd2_rvs, mmd2_rvr, *, use_ratio, mode, eps):
    if use_ratio:
        ratio = (mmd2_rvs + eps) / (mmd2_rvr.detach() + eps)
        return (torch.log(ratio)**2) if (mode == "logsq") else (ratio - 1.0)**2
    else:
        # choose one of these:
        return mmd2_rvs                           # classic MMD^2 → 0
        # or a parity target without division:
        # return torch.relu(mmd2_rvs - mmd2_rvr.detach())


# ----------------------------- main loss -----------------------------

def flowgen_loss(
    model,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_c: torch.Tensor,
    *,
    epoch: int = 0,

    # reference data (for realism)
    x_ref_all: Optional[torch.Tensor] = None,
    y_ref_all: Optional[torch.Tensor] = None,
    c_ref_all: Optional[torch.Tensor] = None,

    use_full_ref: bool = False,
    ref_min: int = 100,
    syn_min: int = 100,
    class_weighting: str = "prior",        # "prior" | "uniform" | "inverse"
    min_per_class: int = 1,

    # base flow terms (joint)
    use_nll: bool = True,
    nll_weight: float = 1.0,

    use_logdet_penalty: bool = False,
    logdet_penalty_weight: float = 0.1,
    use_logdet_sq: bool = True,
    use_logdet_abs: bool = True,
    clamp_logabsdet_range: Tuple[float, float] = (-250.0, 250.0),

    use_logpz_centering: bool = False,
    logpz_centering_weight: float = 0.1,
    logpz_target: Optional[float] = None,

    use_latent_mean_penalty: bool = False,
    latent_mean_weight: float = 0.1,
    use_latent_mean_sq: bool = True,
    use_latent_mean_abs: bool = True,

    use_latent_std_penalty: bool = False,
    latent_std_weight: float = 0.1,
    use_latent_std_sq: bool = True,
    use_latent_std_abs: bool = True,

    use_latent_skew_penalty: bool = False,
    latent_skew_weight: float = 0.1,
    use_latent_skew_sq: bool = True,
    use_latent_skew_abs: bool = True,

    use_latent_kurtosis_penalty: bool = False,
    latent_kurtosis_weight: float = 0.1,
    use_latent_kurtosis_sq: bool = True,
    use_latent_kurtosis_abs: bool = True,

    # ---------------- NEW: joint XY realism ----------------
    use_mmd_xy: bool = False,
    mmd_xy_weight: float = 0.0,
    mmd_xy_norm: Optional[str] = "iqr",      # None | "iqr" | "rms" (normalizes [X;Y] by real stats)
    mmd_xy_scales: Tuple[float, ...] = (0.5, 1.0, 2.0),

    use_corr_xy_pearson: bool = False,
    corr_xy_pearson_weight: float = 0.0,
    corr_xy_use_relative: bool = True,       # Fro gap normalized by ||Corr_real||

    use_corr_xy_spearman: bool = False,
    corr_xy_spearman_weight: float = 0.0,
    corr_xy_tau: float = 0.05,               # soft-rank temperature for differentiable Spearman
    # -------------------------------------------------------

    # realism X side (marginals)
    use_mmd_x: bool = False,
    mmd_x_weight: float = 0.0,
    use_corr_pearson_x: bool = False,
    corr_pearson_x_weight: float = 0.0,
    corr_pearson_use_relative_x: bool = True,
    use_corr_spearman_x: bool = False,
    corr_spearman_x_weight: float = 0.0,
    corr_spearman_use_relative_x: bool = True,
    use_ks_x: bool = False,
    ks_x_weight: float = 0.0,
    use_w1_x: bool = False,
    w1_x_weight: float = 0.0,
    w1_x_norm: str = "iqr",
    ks_grid_points_x: int = 64,
    ks_tau_x: float = 0.05,

    # realism y side (marginals)
    use_mmd_y: bool = False,
    mmd_y_weight: float = 0.0,
    use_corr_pearson_y: bool = False,
    corr_pearson_y_weight: float = 0.0,
    corr_pearson_use_relative_y: bool = True,
    use_corr_spearman_y: bool = False,
    corr_spearman_y_weight: float = 0.0,
    corr_spearman_use_relative_y: bool = True,
    use_ks_y: bool = False,
    ks_y_weight: float = 0.0,
    use_w1_y: bool = False,
    w1_y_weight: float = 0.0,
    w1_y_norm: str = "iqr",
    ks_grid_points_y: int = 64,
    ks_tau_y: float = 0.05,

    realism_stride_batches: int = 1,   # compute realism every K batches
    realism_stride_epochs: int = 1,    # and/or every K epochs (can leave at 1)
    batch_index: int = 0,              # pass from trainer
    realism_scale_mode: str = "keep_mean",  # "keep_mean" or "none"
    realism_warmup_epochs: int = 200,   # epochs with zero realism push
    realism_ramp_epochs: int = 50,      # epochs to ramp from 0 -> 1

    enforce_realism: bool = False,

    # W1 stability knobs
    w1_x_softclip_s: float = 1.0,
    w1_y_softclip_s: float = 1.0,
    w1_x_clip_perdim: float = 0.0,
    w1_y_clip_perdim: float = 0.0,
    realism_z_trunc: float = 0.0,
    w1_x_agg_softcap: float = 2.0,
    w1_y_agg_softcap: float = 2.0,

    # --- MMD ratio mode (use RvS/RvR -> 1 instead of raw MMD2 -> 0)
    use_mmd_as_ratio: bool = False,  # set True to switch to ratio objective
    mmd_ratio_eps: float = 1e-6,  # stabilizer to avoid div-by-zero
    mmd_ratio_mode: str = "logsq",  # "logsq" (recommended) or "sq"

        **unused_kwargs,
) -> Tuple[torch.Tensor, Dict[str, float], float, float, float, float]:
    """
    Returns:
        total_loss, diagnostics,
        rrmse_x_mean, rrmse_x_std, rrmse_y_mean, rrmse_y_std
    """
    device = batch_x.device
    N, Dx = batch_x.shape
    Dy = batch_y.shape[1]
    Dxy = Dx + Dy

    # ---------- Forward / Log-probs ----------
    z, logabsdet = model.forward_xy(batch_x, batch_y, batch_c)
    logabsdet_clamped = logabsdet.clamp(min=clamp_logabsdet_range[0],
                                        max=clamp_logabsdet_range[1])
    logp_xy = model.log_prob_xy(batch_x, batch_y, batch_c)
    logpz = model.flow._distribution.log_prob(z)

    total_loss = torch.zeros((), device=device)
    diagnostics: Dict[str, float] = {}
    ref_nc: Dict[int, int] = {}
    syn_nc: Dict[int, int] = {}
    class_w: Dict[int, float] = {}

    # ---------- Base terms ----------
    if use_nll:
        nll = -logp_xy.mean()
        total_loss = total_loss + nll_weight * nll
        diagnostics["loss_nll"] = float(nll.detach())

    if use_logdet_penalty:
        term_abs = logabsdet_clamped.abs().mean() if use_logdet_abs else 0.0
        term_sq  = (logabsdet_clamped ** 2).mean() if use_logdet_sq else 0.0
        reg = logdet_penalty_weight * (term_abs + term_sq)
        total_loss = total_loss + reg
        diagnostics.update({
            "loss_logdet_penalty": float(reg.detach()),
            "loss_logdet_abs": float((logdet_penalty_weight * term_abs).detach() if use_logdet_abs else 0.0),
            "loss_logdet_sq": float((logdet_penalty_weight * term_sq).detach() if use_logdet_sq else 0.0),
        })

    if use_logpz_centering:
        if logpz_target is None:
            logpz_target = -0.5 * Dxy * (1.0 + math.log(2.0 * math.pi))
        delta = logpz - float(logpz_target)
        pen = logpz_centering_weight * (delta.abs().mean() + (delta ** 2).mean())
        total_loss = total_loss + pen
        diagnostics["loss_logpz_centering"] = float(pen.detach())

    # ---------- Latent isotropy (joint) ----------
    with torch.no_grad():
        ref_scale = logpz.abs().mean().clamp_min(1.0)

    z_mean = z.mean(dim=0)
    z_std  = z.std(dim=0)

    if use_latent_mean_penalty:
        m_abs = z_mean.abs().mean() if use_latent_mean_abs else 0.0
        m_sq  = (z_mean ** 2).mean() if use_latent_mean_sq else 0.0
        pen_m = (latent_mean_weight * ref_scale) * (m_abs + m_sq)
        total_loss = total_loss + pen_m
        diagnostics["loss_latent_mean"] = float(pen_m.detach())

    if use_latent_std_penalty:
        d = z_std - 1.0
        s_abs = d.abs().mean() if use_latent_std_abs else 0.0
        s_sq  = (d ** 2).mean() if use_latent_std_sq else 0.0
        pen_s = (latent_std_weight * ref_scale) * (s_abs + s_sq)
        total_loss = total_loss + pen_s
        diagnostics["loss_latent_std"] = float(pen_s.detach())

    if use_latent_skew_penalty or use_latent_kurtosis_penalty:
        xc = z - z.mean(dim=0, keepdim=True)
        var = (xc ** 2).mean(dim=0) + EPS
        std = torch.sqrt(var)
        if use_latent_skew_penalty:
            m3 = (xc ** 3).mean(dim=0)
            skew = m3 / (std ** 3 + EPS)
            s_abs = skew.abs().sum() if use_latent_skew_abs else 0.0
            s_sq  = (skew ** 2).sum() if use_latent_skew_sq else 0.0
            pen   = (latent_skew_weight * ref_scale) * (s_abs + s_sq)
            total_loss = total_loss + pen
            diagnostics["loss_latent_skew"] = float(pen.detach())
        if use_latent_kurtosis_penalty:
            m4 = (xc ** 4).mean(dim=0)
            kurt = m4 / (var ** 2 + EPS)
            dk = kurt - 3.0
            k_abs = dk.abs().sum() if use_latent_kurtosis_abs else 0.0
            k_sq  = (dk ** 2).sum() if use_latent_kurtosis_sq else 0.0
            pen   = (latent_kurtosis_weight * ref_scale) * (k_abs + k_sq)
            total_loss = total_loss + pen
            diagnostics["loss_latent_kurtosis"] = float(pen.detach())

    # ---------- Isotropy-based RRMSEs (overall + split X/Y) ----------
    mean_rrmse_overall = math.sqrt((z_mean.pow(2).mean().item()))
    std_rrmse_overall  = math.sqrt(((z_std - 1.0).pow(2).mean().item()))

    z_mean_x = z_mean[:Dx]; z_std_x = z_std[:Dx]
    z_mean_y = z_mean[Dx:]; z_std_y = z_std[Dx:]

    rrmse_x_mean = math.sqrt((z_mean_x.pow(2).mean().item()))
    rrmse_x_std  = math.sqrt(((z_std_x - 1.0).pow(2).mean().item()))
    rrmse_y_mean = math.sqrt((z_mean_y.pow(2).mean().item()))
    rrmse_y_std  = math.sqrt(((z_std_y - 1.0).pow(2).mean().item()))

    # ---------- Realism gating ----------
    def _ramp_factor(epoch: int, warmup: int, ramp: int) -> float:
        if epoch < warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        t = (epoch - warmup) / float(ramp)
        t = max(0.0, min(1.0, t))
        return 0.5 - 0.5 * math.cos(math.pi * t)

    any_realism_enabled = any([
        use_mmd_xy and (mmd_xy_weight > 0.0),
        use_corr_xy_pearson and (corr_xy_pearson_weight > 0.0),
        use_corr_xy_spearman and (corr_xy_spearman_weight > 0.0),

        use_mmd_x and (mmd_x_weight > 0.0),
        use_corr_pearson_x and (corr_pearson_x_weight > 0.0),
        use_corr_spearman_x and (corr_spearman_x_weight > 0.0),
        use_ks_x and (ks_x_weight > 0.0),
        use_w1_x and (w1_x_weight > 0.0),

        use_mmd_y and (mmd_y_weight > 0.0),
        use_corr_pearson_y and (corr_pearson_y_weight > 0.0),
        use_corr_spearman_y and (corr_spearman_y_weight > 0.0),
        use_ks_y and (ks_y_weight > 0.0),
        use_w1_y and (w1_y_weight > 0.0),
    ])

    have_refs = (x_ref_all is not None) and (y_ref_all is not None) and (c_ref_all is not None)
    stride_ok = ((epoch % max(realism_stride_epochs, 1)) == 0) and ((batch_index % max(realism_stride_batches, 1)) == 0)
    phase = _ramp_factor(epoch, realism_warmup_epochs, realism_ramp_epochs)

    if enforce_realism:
        stride_ok = True


    do_realism = any_realism_enabled and have_refs and stride_ok and (phase > 0.0)

    if do_realism and (realism_scale_mode == "keep_mean"):
        realism_weight_multiplier = float(max(realism_stride_batches, 1)) * float(max(realism_stride_epochs, 1))
    else:
        realism_weight_multiplier = 1.0
    scale = float(realism_weight_multiplier) * float(phase) if do_realism else 0.0
    diagnostics["realism_phase"] = float(phase)
    diagnostics["realism_scale"] = float(scale)

    # --- Global per-feature denominators for W1 normalization (marginals) ---
    denom_x_global = None
    denom_y_global = None
    if do_realism:
        if use_w1_x:
            denom_x_global = (_iqr(x_ref_all) if (w1_x_norm == "iqr")
                              else torch.sqrt((x_ref_all ** 2).mean(dim=0) + EPS)).to(torch.float32).clamp_min(1e-6).detach()
        if use_w1_y:
            denom_y_global = (_iqr(y_ref_all) if (w1_y_norm == "iqr")
                              else torch.sqrt((y_ref_all ** 2).mean(dim=0) + EPS)).to(torch.float32).clamp_min(1e-6).detach()

    if denom_x_global is not None:
        diagnostics["w1_x_denom_min"] = float(denom_x_global.min().item())
        diagnostics["w1_x_denom_med"] = float(denom_x_global.median().item())
    if denom_y_global is not None:
        diagnostics["w1_y_denom_min"] = float(denom_y_global.min().item())
        diagnostics["w1_y_denom_med"] = float(denom_y_global.median().item())

    if do_realism:
        with _autocast_ctx(device):
            classes = torch.unique(c_ref_all).tolist()
            counts = {int(c): int((c_ref_all == c).sum().item()) for c in classes}
            total_n = sum(counts.values())
            sampling_priors = {k: counts[k] / max(total_n, 1) for k in counts}

            if class_weighting == "uniform":
                class_w = {k: 1.0 / max(len(classes), 1) for k in counts}
            elif class_weighting == "inverse":
                inv = {k: 1.0 / max(sampling_priors[k], 1e-6) for k in counts}
                s = sum(inv.values()) + EPS
                class_w = {k: inv[k] / s for k in counts}
            else:
                class_w = sampling_priors

            def _alloc_by_min(min_target: int) -> Dict[int, int]:
                min_count = max(min(counts.values()), 1)
                base = max(int(min_target), 0)
                desired = {k: int(round(base * (counts[k] / min_count))) for k in counts}
                alloc = {}
                for k in counts:
                    if use_full_ref:
                        alloc[k] = counts[k]
                    else:
                        alloc[k] = max(min_per_class, min(counts[k], desired[k]))
                return alloc

            ref_nc = _alloc_by_min(ref_min)
            syn_nc = _alloc_by_min(syn_min)

            # ----- accumulators -----
            loss_mmd_xy = torch.zeros((), device=device)
            loss_corr_xy_p = torch.zeros((), device=device)
            loss_corr_xy_s = torch.zeros((), device=device)

            loss_mmd_x = torch.zeros((), device=device)
            loss_mmd_y = torch.zeros((), device=device)
            loss_pcorr_x = torch.zeros((), device=device)
            loss_pcorr_y = torch.zeros((), device=device)
            loss_scorr_x = torch.zeros((), device=device)
            loss_scorr_y = torch.zeros((), device=device)
            loss_ks_x = torch.zeros((), device=device)
            loss_ks_y = torch.zeros((), device=device)
            loss_w1_x = torch.zeros((), device=device)
            loss_w1_y = torch.zeros((), device=device)

            diag_acc = {
                "MMD2_XY": 0.0,   # NEW joint
                "CorrXY_P_abs": 0.0, "CorrXY_P_rel": 0.0,
                "CorrXY_S_abs": 0.0, "CorrXY_S_rel": 0.0,
                "MMD2_XY_RVR": 0.0, "MMD2_XY_ratio": 0.0,

                "MMD2_X": 0.0, "MMD2_Y": 0.0,
                "KS_X_mean": 0.0, "KS_X_median": 0.0, "W1_X_mean": 0.0, "W1_X_median": 0.0,
                "KS_Y_mean": 0.0, "KS_Y_median": 0.0, "W1_Y_mean": 0.0, "W1_Y_median": 0.0,
                "MMD2_X_RVR": 0.0, "MMD2_X_ratio": 0.0,

                "Pearson_X_abs": 0.0, "Pearson_X_rel": 0.0,
                "Pearson_Y_abs": 0.0, "Pearson_Y_rel": 0.0,
                "Spearman_X_abs": 0.0, "Spearman_X_rel": 0.0,
                "Spearman_Y_abs": 0.0, "Spearman_Y_rel": 0.0,

                "W1_X_dim_max": 0.0, "W1_X_dim_max_j": -1.0,
                "W1_Y_dim_max": 0.0, "W1_Y_dim_max_j": -1.0,
                "Y_maxdim_real_mean": 0.0, "Y_maxdim_real_std": 0.0,
                "Y_maxdim_synth_mean": 0.0, "Y_maxdim_synth_std": 0.0,
                "Y_maxdim_real_min": 0.0, "Y_maxdim_real_max": 0.0,
                "Y_maxdim_synth_min": 0.0, "Y_maxdim_synth_max": 0.0,
                "X_synth_isfinite_frac": 0.0, "Y_synth_isfinite_frac": 0.0,
                "Y_synth_outlier_frac": 0.0,

                "MMD2_Y_RVR": 0.0, "MMD2_Y_ratio": 0.0,
            }

            def _downsample_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                """Randomly match counts: returns (a_k, b_k) with k = min(len(a), len(b))."""
                na, nb = a.size(0), b.size(0)
                k = min(int(na), int(nb))
                if k <= 0:
                    return a[:0], b[:0]
                if na > k:
                    ia = torch.randperm(na, device=a.device)[:k]
                    a = a.index_select(0, ia)
                if nb > k:
                    ib = torch.randperm(nb, device=b.device)[:k]
                    b = b.index_select(0, ib)
                return a, b

            for cls in classes:
                mref = (c_ref_all == cls)
                xr_c = x_ref_all[mref]
                yr_c = y_ref_all[mref]

                nr = int(ref_nc[cls])
                if not use_full_ref and xr_c.size(0) > nr:
                    idx = torch.randperm(xr_c.size(0), device=device)[:nr]
                    xr_c = xr_c.index_select(0, idx)
                    yr_c = yr_c.index_select(0, idx)

                ns = int(syn_nc[cls])
                z_s = torch.randn(ns, Dxy, device=device, dtype=torch.float32)
                if realism_z_trunc and realism_z_trunc > 0:
                    z_s = z_s.clamp_(min=-float(realism_z_trunc), max=float(realism_z_trunc))

                c_s = torch.full((ns,), cls, dtype=torch.long, device=device)
                # --- inverse in FP32 for stability ---
                with torch.autocast(device_type=device.type, enabled=False):
                    (xs_c, ys_c), _ = model.inverse_xy(z_s.to(torch.float32), c_s)

                # --- drop any non-finite synth rows early ---
                finite = torch.isfinite(xs_c).all(dim=1) & torch.isfinite(ys_c).all(dim=1)
                if finite.sum().item() < 1:
                    continue
                xs_c, ys_c = xs_c[finite], ys_c[finite]

                # --- per-class robust bounds from REAL data (5*IQR) and straight-through clamp ---
                q25x = xr_c.quantile(0.25, dim=0);
                q75x = xr_c.quantile(0.75, dim=0)
                iqr_x = (q75x - q25x).clamp_min(1e-6)
                lo_x = q25x - 5.0 * iqr_x
                hi_x = q75x + 5.0 * iqr_x

                q25y = yr_c.quantile(0.25, dim=0);
                q75y = yr_c.quantile(0.75, dim=0)
                iqr_y = (q75y - q25y).clamp_min(1e-6)
                lo_y = q25y - 5.0 * iqr_y
                hi_y = q75y + 5.0 * iqr_y

                xs_c = _st_clamp(xs_c, lo_x, hi_x)
                ys_c = _st_clamp(ys_c, lo_y, hi_y)

                w = float(class_w[cls])

                # ---- (A) Joint XY MMD ----
                if use_mmd_xy:
                    # Match counts for fairness
                    (xr_xy, yr_xy), (xs_xy, ys_xy) = (xr_c, yr_c), (xs_c, ys_c)
                    xr_xy, xs_xy = _downsample_pair(xr_xy, xs_xy)
                    yr_xy, ys_xy = _downsample_pair(yr_xy, ys_xy)

                    # Numerator: RvS with sigma determined on (R,S)
                    mmd2_rvs, sigma_xy = _mmd_joint_xy_ms(
                        model, xr_xy, yr_xy, xs_xy, ys_xy,
                        norm=mmd_xy_norm, scales=mmd_xy_scales
                    )

                    # Denominator: RvR using the SAME sigma and norm
                    # Split real into two disjoint subsamples of matched size
                    xr1, xr2 = _downsample_pair(xr_xy, xr_xy)
                    yr1, yr2 = _downsample_pair(yr_xy, yr_xy)
                    mmd2_rvr, _ = _mmd_joint_xy_ms(
                        model, xr1, yr1, xr2, yr2,
                        norm=mmd_xy_norm, sigma=sigma_xy, scales=mmd_xy_scales
                    )
                    # Detach denominator (no grads through real-real)
                    denom = (mmd2_rvr.detach() + mmd_ratio_eps)
                    ratio_xy = (mmd2_rvs + mmd_ratio_eps) / denom
                    pen_xy = _mmd_penalty(mmd2_rvs, mmd2_rvr,
                                          use_ratio=use_mmd_as_ratio,
                                          mode=mmd_ratio_mode,
                                          eps=mmd_ratio_eps)

                    loss_mmd_xy = loss_mmd_xy + w * pen_xy
                    diag_acc["MMD2_XY"] += w * float(mmd2_rvs.detach())
                    diag_acc["MMD2_XY_RVR"] += w * float(mmd2_rvr.detach())
                    diag_acc["MMD2_XY_ratio"] += w * float(ratio_xy.detach())

                # ---- (E) XY correlation block gaps ----
                if use_corr_xy_pearson:
                    p_abs, p_rel = _pearson_xyblock_fro_gap(
                        model, xr_c, yr_c, xs_c, ys_c, relative=corr_xy_use_relative
                    )
                    loss_corr_xy_p = loss_corr_xy_p + w * (p_rel if corr_xy_use_relative else p_abs)
                    diag_acc["CorrXY_P_abs"] += w * float(p_abs.detach())
                    diag_acc["CorrXY_P_rel"] += w * float(p_rel.detach())

                if use_corr_xy_spearman:
                    s_abs, s_rel = _softspearman_xyblock_fro_gap(
                        model, xr_c, yr_c, xs_c, ys_c, tau=corr_xy_tau, relative=corr_xy_use_relative
                    )
                    loss_corr_xy_s = loss_corr_xy_s + w * (s_rel if corr_xy_use_relative else s_abs)
                    diag_acc["CorrXY_S_abs"] += w * float(s_abs.detach())
                    diag_acc["CorrXY_S_rel"] += w * float(s_rel.detach())

                # ---- Marginal pushes (X) ----
                if use_mmd_x:
                    xr_m, xs_m = _downsample_pair(xr_c, xs_c)
                    mmd2_x_rvs, sig_x = _mmd_rbf_biased(xr_m, xs_m)  # numerator (RvsS)
                    xr1, xr2 = _downsample_pair(xr_m, xr_m)  # two real splits
                    mmd2_x_rvr, _ = _mmd_rbf_biased(xr1, xr2, sigma=sig_x)  # denominator (RvsR) with SAME sigma
                    ratio_x = (mmd2_x_rvs + mmd_ratio_eps) / (mmd2_x_rvr.detach() + mmd_ratio_eps)
                    pen_x = _mmd_penalty(mmd2_x_rvs, mmd2_x_rvr,
                                          use_ratio=use_mmd_as_ratio,
                                          mode=mmd_ratio_mode,
                                          eps=mmd_ratio_eps)
                    loss_mmd_x = loss_mmd_x + w * pen_x
                    diag_acc["MMD2_X"] += w * float(mmd2_x_rvs.detach())
                    diag_acc["MMD2_X_RVR"] += w * float(mmd2_x_rvr.detach())
                    diag_acc["MMD2_X_ratio"] += w * float(ratio_x.detach())

                if use_corr_pearson_x and xr_c.shape[1] >= 2:
                    Cr = _pearson_corr(xr_c); Cs = _pearson_corr(xs_c)
                    fro_abs, fro_rel = _fro_rel(Cr, Cs)
                    loss_pcorr_x = loss_pcorr_x + w * (fro_rel if corr_pearson_use_relative_x else fro_abs)
                    diag_acc["Pearson_X_abs"] += w * float(fro_abs.detach())
                    diag_acc["Pearson_X_rel"] += w * float(fro_rel.detach())

                if use_corr_spearman_x and xr_c.shape[1] >= 2:
                    Cr = _spearman_corr(xr_c); Cs = _spearman_corr(xs_c)
                    fro_abs, fro_rel = _fro_rel(Cr, Cs)
                    loss_scorr_x = loss_scorr_x + w * (fro_rel if corr_spearman_use_relative_x else fro_abs)
                    diag_acc["Spearman_X_abs"] += w * float(fro_abs.detach())
                    diag_acc["Spearman_X_rel"] += w * float(fro_rel.detach())

                if use_ks_x or use_w1_x:
                    ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                        xr_c, xs_c, grid_points=ks_grid_points_x, tau=ks_tau_x, norm=w1_x_norm,
                        denom_override=denom_x_global
                    )
                    if use_w1_x:
                        w1x_perdim, _ = _perdim_w1_normed(xr_c, xs_c, norm=w1_x_norm, denom_override=denom_x_global)
                        if w1_x_softclip_s and w1_x_softclip_s > 0:
                            w1x_perdim = _softclip_asinh(w1x_perdim, w1_x_softclip_s)
                        if w1_x_clip_perdim and w1_x_clip_perdim > 0:
                            w1x_perdim = torch.clamp(w1x_perdim, max=float(w1_x_clip_perdim))
                        w1_mean = w1x_perdim.mean(); w1_med = w1x_perdim.median()
                        maxv, maxj = torch.max(w1x_perdim, dim=0)
                        maxv = float(maxv.detach()); maxj = int(maxj.detach())
                        if maxv * w > diag_acc["W1_X_dim_max"]:
                            diag_acc["W1_X_dim_max"] = maxv * w
                            diag_acc["W1_X_dim_max_j"] = float(maxj)

                    if use_ks_x:  loss_ks_x  = loss_ks_x  + w * ks_mean
                    if use_w1_x: loss_w1_x = loss_w1_x + w * w1_mean
                    diag_acc["KS_X_mean"] += w * float(ks_mean.detach())
                    diag_acc["KS_X_median"] += w * float(ks_med.detach())
                    diag_acc["W1_X_mean"] += w * float(w1_mean.detach())
                    diag_acc["W1_X_median"] += w * float(w1_med.detach())

                # ---- Marginal pushes (Y) ----
                if use_mmd_y:
                    yr_m, ys_m = _downsample_pair(yr_c, ys_c)
                    mmd2_y_rvs, sig_y = _mmd_rbf_biased(yr_m, ys_m)  # numerator (RvsS)
                    yr1, yr2 = _downsample_pair(yr_m, yr_m)  # two real splits
                    mmd2_y_rvr, _ = _mmd_rbf_biased(yr1, yr2, sigma=sig_y)  # denominator (RvsR) with SAME sigma
                    ratio_y = (mmd2_y_rvs + mmd_ratio_eps) / (mmd2_y_rvr.detach() + mmd_ratio_eps)
                    pen_y = _mmd_penalty(mmd2_y_rvs, mmd2_y_rvr,
                                          use_ratio=use_mmd_as_ratio,
                                          mode=mmd_ratio_mode,
                                          eps=mmd_ratio_eps)
                    loss_mmd_y = loss_mmd_y + w * pen_y
                    diag_acc["MMD2_Y"] += w * float(mmd2_y_rvs.detach())
                    diag_acc["MMD2_Y_RVR"] += w * float(mmd2_y_rvr.detach())
                    diag_acc["MMD2_Y_ratio"] += w * float(ratio_y.detach())

                if use_corr_pearson_y and yr_c.shape[1] >= 2:
                    Cr = _pearson_corr(yr_c); Cs = _pearson_corr(ys_c)
                    fro_abs, fro_rel = _fro_rel(Cr, Cs)
                    loss_pcorr_y = loss_pcorr_y + w * (fro_rel if corr_pearson_use_relative_y else fro_abs)
                    diag_acc["Pearson_Y_abs"] += w * float(fro_abs.detach())
                    diag_acc["Pearson_Y_rel"] += w * float(fro_rel.detach())

                if use_corr_spearman_y and yr_c.shape[1] >= 2:
                    Cr = _spearman_corr(yr_c); Cs = _spearman_corr(ys_c)
                    fro_abs, fro_rel = _fro_rel(Cr, Cs)
                    loss_scorr_y = loss_scorr_y + w * (fro_rel if corr_spearman_use_relative_y else fro_abs)
                    diag_acc["Spearman_Y_abs"] += w * float(fro_abs.detach())
                    diag_acc["Spearman_Y_rel"] += w * float(fro_rel.detach())

                if use_ks_y or use_w1_y:
                    ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                        yr_c, ys_c, grid_points=ks_grid_points_y, tau=ks_tau_y, norm=w1_y_norm,
                        denom_override=denom_y_global
                    )
                    if use_w1_y:
                        w1y_perdim, _ = _perdim_w1_normed(yr_c, ys_c, norm=w1_y_norm, denom_override=denom_y_global)
                        if w1_y_softclip_s and w1_y_softclip_s > 0:
                            w1y_perdim = _softclip_asinh(w1y_perdim, w1_y_softclip_s)
                        if w1_y_clip_perdim and w1_y_clip_perdim > 0:
                            w1y_perdim = torch.clamp(w1y_perdim, max=float(w1_y_clip_perdim))
                        w1_mean = w1y_perdim.mean(); w1_med = w1y_perdim.median()
                        maxv, maxj = torch.max(w1y_perdim, dim=0)
                        maxv = float(maxv.detach()); maxj = int(maxj.detach())
                        if maxv * w > diag_acc["W1_Y_dim_max"]:
                            diag_acc["W1_Y_dim_max"] = maxv * w
                            diag_acc["W1_Y_dim_max_j"] = float(maxj)

                            yrj = yr_c[:, maxj].to(torch.float32)
                            ysj = ys_c[:, maxj].to(torch.float32)
                            diag_acc["Y_maxdim_real_mean"] = float(yrj.mean().detach())
                            diag_acc["Y_maxdim_real_std"] = float(yrj.std().detach())
                            diag_acc["Y_maxdim_synth_mean"] = float(ysj.mean().detach())
                            diag_acc["Y_maxdim_synth_std"] = float(ysj.std().detach())
                            diag_acc["Y_maxdim_real_min"] = float(yrj.min().detach())
                            diag_acc["Y_maxdim_real_max"] = float(yrj.max().detach())
                            diag_acc["Y_maxdim_synth_min"] = float(ysj.min().detach())
                            diag_acc["Y_maxdim_synth_max"] = float(ysj.max().detach())

                        y_isfinite = torch.isfinite(ys_c).to(torch.float32)
                        x_isfinite = torch.isfinite(xs_c).to(torch.float32)
                        diag_acc["Y_synth_isfinite_frac"] += w * float(y_isfinite.mean().detach())
                        diag_acc["X_synth_isfinite_frac"] += w * float(x_isfinite.mean().detach())

                        q25 = torch.quantile(yr_c, 0.25, dim=0)
                        q75 = torch.quantile(yr_c, 0.75, dim=0)
                        iqr = (q75 - q25).clamp_min(1e-6)
                        low = q25 - 1.5 * iqr
                        high = q75 + 1.5 * iqr
                        outside = ((ys_c < low) | (ys_c > high)).to(torch.float32).mean()
                        diag_acc["Y_synth_outlier_frac"] += w * float(outside.detach())

                    if use_ks_y:  loss_ks_y  = loss_ks_y  + w * ks_mean
                    if use_w1_y: loss_w1_y = loss_w1_y + w * w1_mean
                    diag_acc["KS_Y_mean"] += w * float(ks_mean.detach())
                    diag_acc["KS_Y_median"] += w * float(ks_med.detach())
                    diag_acc["W1_Y_mean"] += w * float(w1_mean.detach())
                    diag_acc["W1_Y_median"] += w * float(w1_med.detach())

            # ----- add to loss with current scale -----
            if use_mmd_xy:
                total_loss = total_loss + scale * mmd_xy_weight * loss_mmd_xy
                diagnostics["loss_mmd2_xy"] = float((scale * mmd_xy_weight * loss_mmd_xy).detach())
            if use_corr_xy_pearson:
                total_loss = total_loss + scale * corr_xy_pearson_weight * loss_corr_xy_p
                diagnostics["loss_corr_xy_pearson"] = float((scale * corr_xy_pearson_weight * loss_corr_xy_p).detach())
            if use_corr_xy_spearman:
                total_loss = total_loss + scale * corr_xy_spearman_weight * loss_corr_xy_s
                diagnostics["loss_corr_xy_spearman"] = float((scale * corr_xy_spearman_weight * loss_corr_xy_s).detach())

            if use_mmd_x:
                total_loss = total_loss + scale * mmd_x_weight * loss_mmd_x
                diagnostics["loss_mmd2_x"] = float((scale * mmd_x_weight * loss_mmd_x).detach())
            if use_mmd_y:
                total_loss = total_loss + scale * mmd_y_weight * loss_mmd_y
                diagnostics["loss_mmd2_y"] = float((scale * mmd_y_weight * loss_mmd_y).detach())

            if use_corr_pearson_x:
                total_loss = total_loss + scale * corr_pearson_x_weight * loss_pcorr_x
                diagnostics["loss_corr_pearson_x"] = float((scale * corr_pearson_x_weight * loss_pcorr_x).detach())
            if use_corr_pearson_y:
                total_loss = total_loss + scale * corr_pearson_y_weight * loss_pcorr_y
                diagnostics["loss_corr_pearson_y"] = float((scale * corr_pearson_y_weight * loss_pcorr_y).detach())

            if use_corr_spearman_x:
                total_loss = total_loss + scale * corr_spearman_x_weight * loss_scorr_x
                diagnostics["loss_corr_spearman_x"] = float((scale * corr_spearman_x_weight * loss_scorr_x).detach())
            if use_corr_spearman_y:
                total_loss = total_loss + scale * corr_spearman_y_weight * loss_scorr_y
                diagnostics["loss_corr_spearman_y"] = float((scale * corr_spearman_y_weight * loss_scorr_y).detach())

            if use_ks_x:
                total_loss = total_loss + scale * ks_x_weight * loss_ks_x
                diagnostics["loss_ks_x"] = float((scale * ks_x_weight * loss_ks_x).detach())
            if use_ks_y:
                total_loss = total_loss + scale * ks_y_weight * loss_ks_y
                diagnostics["loss_ks_y"] = float((scale * ks_y_weight * loss_ks_y).detach())

            if use_w1_x:
                loss_w1_x = _softclip_asinh(loss_w1_x, s=float(w1_x_agg_softcap))
            if use_w1_y:
                loss_w1_y = _softclip_asinh(loss_w1_y, s=float(w1_y_agg_softcap))
            if use_w1_x:
                total_loss = total_loss + scale * w1_x_weight * loss_w1_x
                diagnostics["loss_w1_x"] = float((scale * w1_x_weight * loss_w1_x).detach())
            if use_w1_y:
                total_loss = total_loss + scale * w1_y_weight * loss_w1_y
                diagnostics["loss_w1_y"] = float((scale * w1_y_weight * loss_w1_y).detach())

            diagnostics.update({
                "realism_MMD2_XY": diag_acc["MMD2_XY"],
                "realism_MMD2_XY_RVR": diag_acc["MMD2_XY_RVR"],
                "realism_MMD2_XY_ratio": diag_acc["MMD2_XY_ratio"],
                "realism_CorrXY_P_abs": diag_acc["CorrXY_P_abs"],
                "realism_CorrXY_P_rel": diag_acc["CorrXY_P_rel"],
                "realism_CorrXY_S_abs": diag_acc["CorrXY_S_abs"],
                "realism_CorrXY_S_rel": diag_acc["CorrXY_S_rel"],

                "realism_MMD2_X": diag_acc["MMD2_X"],
                "realism_MMD2_Y": diag_acc["MMD2_Y"],
                "realism_MMD2_X_RVR": diag_acc["MMD2_X_RVR"],
                "realism_MMD2_X_ratio": diag_acc["MMD2_X_ratio"],
                "realism_MMD2_Y_RVR": diag_acc["MMD2_Y_RVR"],
                "realism_MMD2_Y_ratio": diag_acc["MMD2_Y_ratio"],
                "realism_KS_X_mean": diag_acc["KS_X_mean"],
                "realism_KS_X_median": diag_acc["KS_X_median"],
                "realism_W1_X_mean": diag_acc["W1_X_mean"],
                "realism_W1_X_median": diag_acc["W1_X_median"],
                "realism_KS_Y_mean": diag_acc["KS_Y_mean"],
                "realism_KS_Y_median": diag_acc["KS_Y_median"],
                "realism_W1_Y_mean": diag_acc["W1_Y_mean"],
                "realism_W1_Y_median": diag_acc["W1_Y_median"],

                "realism_Pearson_X_abs": diag_acc["Pearson_X_abs"],
                "realism_Pearson_X_rel": diag_acc["Pearson_X_rel"],
                "realism_Pearson_Y_abs": diag_acc["Pearson_Y_abs"],
                "realism_Pearson_Y_rel": diag_acc["Pearson_Y_rel"],
                "realism_Spearman_X_abs": diag_acc["Spearman_X_abs"],
                "realism_Spearman_X_rel": diag_acc["Spearman_X_rel"],
                "realism_Spearman_Y_abs": diag_acc["Spearman_Y_abs"],
                "realism_Spearman_Y_rel": diag_acc["Spearman_Y_rel"],

                "realism_W1_X_dim_max": diag_acc["W1_X_dim_max"],
                "realism_W1_X_dim_max_j": diag_acc["W1_X_dim_max_j"],
                "realism_W1_Y_dim_max": diag_acc["W1_Y_dim_max"],
                "realism_W1_Y_dim_max_j": diag_acc["W1_Y_dim_max_j"],

                "realism_Y_maxdim_real_mean": diag_acc["Y_maxdim_real_mean"],
                "realism_Y_maxdim_real_std": diag_acc["Y_maxdim_real_std"],
                "realism_Y_maxdim_synth_mean": diag_acc["Y_maxdim_synth_mean"],
                "realism_Y_maxdim_synth_std": diag_acc["Y_maxdim_synth_std"],
                "realism_Y_maxdim_real_min": diag_acc["Y_maxdim_real_min"],
                "realism_Y_maxdim_real_max": diag_acc["Y_maxdim_real_max"],
                "realism_Y_maxdim_synth_min": diag_acc["Y_maxdim_synth_min"],
                "realism_Y_maxdim_synth_max": diag_acc["Y_maxdim_synth_max"],

                "realism_X_synth_isfinite_frac": diag_acc["X_synth_isfinite_frac"],
                "realism_Y_synth_isfinite_frac": diag_acc["Y_synth_isfinite_frac"],
                "realism_Y_synth_outlier_frac": diag_acc["Y_synth_outlier_frac"],
            })

    # ---------- Final diagnostics ----------
    diagnostics.update({
        "loss_total": float(total_loss.detach().item()),
        "logpz_mean": float(logpz.mean().detach().item()),
        "logabsdet_mean": float(logabsdet.mean().detach().item()),
        "logpz_min": float(logpz.min().detach().item()),
        "logpz_max": float(logpz.max().detach().item()),
        "logdet_min": float(logabsdet.min().detach().item()),
        "logdet_max": float(logabsdet.max().detach().item()),
        "z_mean_mean": float(z_mean.mean().detach().item()),
        "z_mean_min": float(z_mean.min().detach().item()),
        "z_mean_max": float(z_mean.max().detach().item()),
        "z_std_mean": float(z_std.mean().detach().item()),
        "z_std_min": float(z_std.min().detach().item()),
        "z_std_max": float(z_std.max().detach().item()),
        "rrmse_mean": float(mean_rrmse_overall),
        "rrmse_std":  float(std_rrmse_overall),
    })
    if ref_nc:
        for k, v in ref_nc.items():
            diagnostics[f"alloc_ref_c{int(k)}"] = float(v)
    if syn_nc:
        for k, v in syn_nc.items():
            diagnostics[f"alloc_syn_c{int(k)}"] = float(v)
    if class_w:
        for k, v in class_w.items():
            diagnostics[f"class_w_c{int(k)}"] = float(v)

    diagnostics["realism_do"] = float(do_realism)
    diagnostics["realism_have_refs"] = float(1.0 if have_refs else 0.0)
    diagnostics["realism_any_enabled"] = float(1.0 if any_realism_enabled else 0.0)
    diagnostics["realism_stride_ok"] = float(1.0 if stride_ok else 0.0)
    diagnostics["realism_enforced"] = float(1.0 if enforce_realism else 0.0)

    return total_loss, diagnostics, rrmse_x_mean, rrmse_x_std, rrmse_y_mean, rrmse_y_std
