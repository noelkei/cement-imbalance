# Loss for CVAE+Conditional Flow with optional "realism" penalties
# ---------------------------------------------------------------
import math
import torch
import torch.nn.functional as F

# ---------- reconstruction helpers ----------
def _log_diag_gaussian(x, mean, logvar):
    var = (logvar.exp()).clamp_min(1e-8)
    return -0.5 * (math.log(2 * math.pi) + logvar + (x - mean) ** 2 / var)

def _gaussian_nll(x, mean, logvar):
    log_prob = _log_diag_gaussian(x, mean, logvar)          # (B, D)
    return -(log_prob.sum(dim=1)).mean()                    # scalar

def _mse_recon(x, mean):
    return F.mse_loss(mean, x, reduction="mean")

def _rrmse(x, xhat):
    rmse  = torch.sqrt(torch.mean((xhat - x) ** 2))
    denom = torch.sqrt(torch.mean(x ** 2)) + 1e-8
    return float((rmse / denom).item())

# ---------- differentiable realism surrogates ----------
def _pairwise_sq_dists(a, b):
    a2 = (a * a).sum(dim=1, keepdim=True)          # (Na,1)
    b2 = (b * b).sum(dim=1, keepdim=True).t()      # (1,Nb)
    return (a2 + b2 - 2.0 * a @ b.t()).clamp_min(0.0)

def _mmd_rbf(a, b, sigma=None):
    """
    Unbiased RBF MMD^2(a,b). a:(Na,D), b:(Nb,D). Differentiable.
    If sigma is None, use the median heuristic on a∪b.
    """
    Na, Nb = a.shape[0], b.shape[0]
    if Na < 2 or Nb < 2:
        return a.new_tensor(0.0)

    if sigma is None:
        with torch.no_grad():
            X  = torch.cat([a, b], dim=0)
            d2 = _pairwise_sq_dists(X, X)
            iu, ju = torch.triu_indices(d2.size(0), d2.size(1), offset=1)
            med = torch.median(d2[iu, ju])
            sigma = torch.sqrt(med.clamp_min(1e-12))
    gamma = 1.0 / (2.0 * (sigma ** 2) + 1e-12)

    Ka  = torch.exp(-gamma * _pairwise_sq_dists(a, a))
    Kb  = torch.exp(-gamma * _pairwise_sq_dists(b, b))
    Kab = torch.exp(-gamma * _pairwise_sq_dists(a, b))

    mmd2 = (Ka.sum() - Ka.diag().sum()) / (Na * (Na - 1) + 1e-12) \
         + (Kb.sum() - Kb.diag().sum()) / (Nb * (Nb - 1) + 1e-12) \
         - 2.0 * Kab.mean()
    return mmd2

def _corr_matrix(X, eps=1e-8):
    Xc  = X - X.mean(dim=0, keepdim=True)
    cov = (Xc.t() @ Xc) / (X.shape[0] - 1 + eps)          # (D,D)
    std = torch.sqrt(torch.diag(cov).clamp_min(eps))      # (D,)
    return cov / (std[:, None] * std[None, :] + eps)

def _corr_fro_penalty(real, fake):
    Cr = _corr_matrix(real)
    Cf = _corr_matrix(fake)
    return torch.norm(Cr - Cf, p='fro') / (Cr.numel() ** 0.5)

def _moment_penalty(real, fake, use_mean=True, use_std=True):
    losses = []
    if use_mean:
        losses.append(F.l1_loss(fake.mean(dim=0), real.mean(dim=0)))
    if use_std:
        rstd = real.std(dim=0, unbiased=False).clamp_min(1e-8)
        fstd = fake.std(dim=0, unbiased=False).clamp_min(1e-8)
        losses.append(F.l1_loss(fstd, rstd))
    return sum(losses) if losses else real.new_tensor(0.0)

# ---------- main loss ----------
def flexible_cvae_cnf_loss_from_model(
    model,
    batch_x,
    batch_c,

    # --- Reconstruction ---
    use_recon_nll: bool = True,              # if False → MSE
    recon_weight: float = 1.0,               # α
    clamp_dec_logstd_range = (-8.0, 8.0),    # (min_logstd, max_logstd) or None

    # --- KL / regularization ---
    use_kl: bool = True,
    kl_weight: float = 1.0,                  # β
    free_bits_per_dim: float = 0.0,          # per-dim nats floor
    sum_kl_over_dims: bool = True,           # API parity only (KL already summed here)

    # --- Posterior sampling ---
    sample_posterior: bool = True,

    # --- Encoder stability ---
    clamp_enc_logstd_range = None,           # None ⇒ reuse decoder clamp

    # --- Diagnostics helpers ---
    return_recon_tensor: bool = False,

    # --- Optional decoder logstd regularizer (for NLL path) ---
    enable_dec_logstd_reg: bool = False,
    dec_logstd_target: float = 0.0,
    dec_logstd_reg_weight: float = 0.0,
    dec_logstd_reg_until_epoch: int = 0,     # 0 ⇒ no cutoff
    current_epoch: int | None = None,

    # ========== NEW: optional realism terms ==========
    enable_realism: bool = False,
    realism_mode: str = "prior",             # "prior" | "recon" | "both"
    realism_weight_mmd: float = 0.0,         # λ_mmd
    realism_weight_corr: float = 0.0,        # λ_corr
    realism_weight_mom: float = 0.0,         # λ_moments
    realism_sigma: float | None = None,      # RBF sigma (None ⇒ median heuristic)
    realism_subsample: int = 512,            # cap per set for O(B^2) terms
    realism_ramp_start: int = 0,             # start epoch (inclusive)
    realism_ramp_end: int = 0,               # end epoch (inclusive), 0 ⇒ no ramp
    realism_use_per_class: bool = True,      # match within each class in batch
    realism_use_mean: bool = True,           # moments: mean
    realism_use_std: bool = True,            # moments: std

    # --- NEW knobs we pass from trainer ---
    realism_scale: torch.Tensor | None = None,       # (D,) scale to divide real & fake
    realism_average_over_classes: bool = True,       # average classwise penalties
):
    """
    Total loss:
      L = α * Recon + β * KL + λ_mmd * MMD^2(real, gen) + λ_corr * ||Corr(real)-Corr(gen)||_F + λ_mom * moments_L1
    with all realism terms optional/ramped. When all realism weights=0, behavior matches plain ELBO-style loss.
    """
    x = batch_x
    c = batch_c

    # ----- q(z|x,c) -----
    z, mu, logvar = model.encode(x, c, sample=sample_posterior)

    # clamp encoder logvar (use log-STD bounds)
    enc_minmax = clamp_dec_logstd_range if clamp_enc_logstd_range is None else clamp_enc_logstd_range
    if enc_minmax is not None:
        a, b = enc_minmax
        logvar = torch.clamp(logvar, min=2 * a, max=2 * b)

    # ----- decoder -----
    dec_out = model.decode(z, c)
    if isinstance(dec_out, (list, tuple)):
        x_mean = dec_out[0]
        x_logvar = dec_out[1] if len(dec_out) >= 2 else None
    elif isinstance(dec_out, dict):
        x_mean = dec_out.get("mean", None)
        x_logvar = dec_out.get("logvar", None)
        if x_mean is None:
            raise RuntimeError("Decoder dict must include key 'mean'.")
    else:
        x_mean = dec_out
        x_logvar = None

    if x_mean.shape != x.shape:
        raise RuntimeError(f"Decoder mean shape {tuple(x_mean.shape)} != input shape {tuple(x.shape)}")

    if x_logvar is not None and clamp_dec_logstd_range is not None:
        a, b = clamp_dec_logstd_range
        x_logvar = torch.clamp(x_logvar, min=2 * a, max=2 * b)

    # ----- reconstruction term -----
    used_nll = False
    if use_recon_nll and x_logvar is not None:
        recon_term = _gaussian_nll(x, x_mean, x_logvar)
        recon_key  = "loss_recon_nll"
        used_nll   = True
    elif use_recon_nll and x_logvar is None:
        recon_term = _mse_recon(x, x_mean)
        recon_key  = "loss_recon_mse_fallback"
    else:
        recon_term = _mse_recon(x, x_mean)
        recon_key  = "loss_recon_mse"

    # ----- optional decoder log-std regularizer -----
    loss_dec_logstd_reg = x.new_tensor(0.0)
    apply_reg = (
        enable_dec_logstd_reg and used_nll and (x_logvar is not None) and
        (dec_logstd_reg_weight > 0.0) and
        (dec_logstd_reg_until_epoch <= 0 or (current_epoch is None) or (current_epoch <= dec_logstd_reg_until_epoch))
    )
    if apply_reg:
        x_logstd = 0.5 * x_logvar
        loss_dec_logstd_reg = dec_logstd_reg_weight * torch.mean((x_logstd - dec_logstd_target) ** 2)

    # ----- KL[q || p(z|c)] from the flow prior -----
    if use_kl:
        log_q = _log_diag_gaussian(z, mu, logvar).sum(dim=1)  # (B,)
        log_p = model.prior_log_prob(z, c)                    # (B,)
        kl_per_ex = (log_q - log_p)
        if free_bits_per_dim > 0.0:
            kl_per_ex = torch.clamp(kl_per_ex, min=free_bits_per_dim * z.shape[1])
        kl = kl_per_ex.mean()
    else:
        kl = x.new_tensor(0.0)
        log_q = x.new_tensor(0.0)
        log_p = x.new_tensor(0.0)

    # ----- realism penalties (optional & ramped) -----
    realism_loss = x.new_tensor(0.0)
    realism_diag = {"realism_mode": "off", "realism_weight_effective": 0.0, "realism_loss": 0.0}

    if enable_realism and (realism_weight_mmd > 0.0 or realism_weight_corr > 0.0 or realism_weight_mom > 0.0):
        # ramp factor in [0,1]
        if (realism_ramp_start > 0 or realism_ramp_end > 0) and (current_epoch is not None):
            if current_epoch < realism_ramp_start:
                ramp = 0.0
            elif realism_ramp_end > 0 and current_epoch > realism_ramp_end:
                ramp = 1.0
            else:
                denom = max(1, realism_ramp_end - realism_ramp_start)
                ramp = float(max(0, min(1, (current_epoch - realism_ramp_start) / denom)))
        else:
            ramp = 1.0

        gens = []
        if realism_mode in ("prior", "both"):
            # model.sample returns decoded x̂ (or (mean,logvar))
            x_prior = model.sample(num_samples=x.shape[0], class_labels=c)
            if isinstance(x_prior, (list, tuple)):
                x_prior = x_prior[0]
            gens.append(("prior", x_prior))
        if realism_mode in ("recon", "both"):
            gens.append(("recon", x_mean))

        def _sub(t):
            if t.shape[0] > realism_subsample:
                idx = torch.randperm(t.shape[0], device=t.device)[:realism_subsample]
                return t[idx]
            return t

        def _apply_scale(t):
            if realism_scale is None:
                return t
            return t / (realism_scale.view(1, -1) + 1e-8)

        def _accumulate(real, fake):
            real_s = _apply_scale(real)
            fake_s = _apply_scale(fake)
            L = real.new_tensor(0.0)
            if realism_weight_mmd > 0.0:
                L = L + realism_weight_mmd * _mmd_rbf(_sub(real_s), _sub(fake_s), sigma=realism_sigma)
            if realism_weight_corr > 0.0:
                L = L + realism_weight_corr * _corr_fro_penalty(real_s, fake_s)
            if realism_weight_mom > 0.0:
                L = L + realism_weight_mom * _moment_penalty(real_s, fake_s,
                                                             use_mean=realism_use_mean,
                                                             use_std=realism_use_std)
            return L

        L_total = x.new_tensor(0.0)
        n_terms = 0
        if realism_use_per_class:
            classes = torch.unique(c)
            for _, xg in gens:
                for cls in classes:
                    m = (c == cls)
                    if m.sum() < 4:  # avoid unstable corr on tiny groups
                        continue
                    L_total = L_total + _accumulate(x[m], xg[m])
                    n_terms += 1
        else:
            for _, xg in gens:
                L_total = L_total + _accumulate(x, xg)
                n_terms += 1

        if realism_average_over_classes and n_terms > 0:
            L_total = L_total / float(n_terms)

        realism_loss = ramp * L_total
        realism_diag.update({
            "realism_mode": realism_mode,
            "realism_weight_effective": float(ramp),
            "realism_loss": float(realism_loss.item()),
        })

    # ----- total loss -----
    total_loss = recon_weight * recon_term + kl_weight * kl + loss_dec_logstd_reg + realism_loss
    total_loss = torch.nan_to_num(total_loss, nan=1e6, posinf=1e6, neginf=1e6)

    # ----- friendly metrics -----
    rrmse_recon = _rrmse(x, x_mean)

    diagnostics = {
        "loss_total": float(total_loss.item()),
        recon_key: float(recon_term.item()),
        "loss_kl": float(kl.item()) if use_kl else 0.0,
        "kl_weight": float(kl_weight),
        "recon_weight": float(recon_weight),
        "use_recon_nll": bool(use_recon_nll),
        "use_kl": bool(use_kl),
        "free_bits_per_dim": float(free_bits_per_dim),
        "logq_mean": float(log_q.mean().item()) if torch.is_tensor(log_q) else 0.0,
        "logp_mean": float(log_p.mean().item()) if torch.is_tensor(log_p) else 0.0,
        "rrmse_recon": float(rrmse_recon),
        "loss_dec_logstd_reg": float(loss_dec_logstd_reg.item()),
        "recon_mode": ("nll" if (use_recon_nll and x_logvar is not None)
                       else "mse_fallback" if (use_recon_nll and x_logvar is None)
                       else "mse"),
        "has_variance_head": bool(x_logvar is not None),
        "posterior_sampled": bool(sample_posterior),
    }

    if x_logvar is not None:
        logstd = 0.5 * x_logvar
        diagnostics.update({
            "dec_logstd_mean": float(logstd.mean().item()),
            "dec_logstd_min": float(logstd.min().item()),
            "dec_logstd_max": float(logstd.max().item()),
            "dec_logstd_frac_clamped": float((
                (logstd <= (clamp_dec_logstd_range[0] if clamp_dec_logstd_range else -1e9)) |
                (logstd >= (clamp_dec_logstd_range[1] if clamp_dec_logstd_range else  1e9))
            ).float().mean().item()) if clamp_dec_logstd_range is not None else 0.0
        })

    diagnostics.update(realism_diag)

    if return_recon_tensor:
        diagnostics["_recon_mean_tensor"] = x_mean
    return total_loss, diagnostics, rrmse_recon


