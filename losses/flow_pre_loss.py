import math
import torch
from scipy.stats import skew, kurtosis

def flexible_flow_loss_from_model(
    model,
    batch_x,
    batch_c,

    use_nll=True,

    use_logdet_penalty=False,
    logdet_penalty_weight=0.1,
    use_logdet_sq=True,
    use_logdet_abs=True,
    ref_logdet=5.0,
    logdet_scale_factor=True,

    use_mean_penalty=False,
    mean_penalty_weight=0.1,
    use_mean_sq=True,
    use_mean_abs=True,

    use_std_penalty=False,
    std_penalty_weight=0.1,
    use_std_sq=True,
    use_std_abs=True,

    use_logpz_centering=False,
    logpz_centering_weight=0.1,
    logpz_target=None,

    use_skew_penalty=False,
    skew_penalty_weight=0.1,
    use_skew_sq=True,
    use_skew_abs=True,

    use_kurtosis_penalty=False,
    kurtosis_penalty_weight=0.1,
    use_kurtosis_sq=True,
    use_kurtosis_abs=True,

    clamp_logabsdet_range=(-250.0, 250.0),
):
    context = model.get_context(batch_c)
    z, logabsdet = model.flow._transform.forward(batch_x, context)
    input_dim = batch_x.shape[1]
    logpz = model.flow._distribution.log_prob(z)

    if clamp_logabsdet_range is not None:
        logabsdet_clamped = torch.clamp(logabsdet, *clamp_logabsdet_range)
    else:
        logabsdet_clamped = logabsdet

    total_loss = 0.0
    diagnostics = {}

    if use_nll:
        nll = -logpz.mean() - logabsdet_clamped.mean()
        total_loss += nll
        diagnostics["loss_nll"] = nll.item()

    if use_logdet_penalty:
        if logdet_scale_factor:
            scale_factor = torch.clamp(logabsdet_clamped.abs().mean() / ref_logdet, min=1.0, max=10.0)
            reg_weight = logdet_penalty_weight * scale_factor
        else:
            reg_weight = logdet_penalty_weight

        loss_logdet_abs = logabsdet_clamped.abs().mean() if use_logdet_abs else 0.0
        loss_logdet_sq = logabsdet_clamped.pow(2).mean() if use_logdet_sq else 0.0
        loss_logdet = reg_weight * (loss_logdet_abs + loss_logdet_sq)
        total_loss += loss_logdet

        diagnostics.update({
            "loss_logdet_penalty": loss_logdet.item(),
            "loss_logdet_abs": (reg_weight * loss_logdet_abs).item() if use_logdet_abs else 0.0,
            "loss_logdet_sq": (reg_weight * loss_logdet_sq).item() if use_logdet_sq else 0.0,
            "reg_weight": reg_weight.item(),
        })

    if use_logpz_centering:
        if logpz_target is None:
            logpz_target = -0.5 * input_dim * (1 + math.log(2 * math.pi))
        delta_logpz = logpz - logpz_target
        logpz_abs = delta_logpz.abs().mean()
        logpz_sq = delta_logpz.pow(2).mean()
        logpz_penalty = logpz_centering_weight * (logpz_abs + logpz_sq)
        total_loss += logpz_penalty

        diagnostics.update({
            "loss_logpz_centering": logpz_penalty.item(),
            "loss_logpz_abs": logpz_abs.item(),
            "loss_logpz_sq": logpz_sq.item(),
        })

    z_mean = z.mean(dim=0)
    z_std = z.std(dim=0)

    # Dynamic scaling reference
    with torch.no_grad():
        reference_scale = logpz.abs().mean().detach().clamp(min=1.0)

    if use_mean_penalty:
        mean_abs = z_mean.abs().mean() if use_mean_abs else 0.0
        mean_sq = z_mean.pow(2).mean() if use_mean_sq else 0.0
        mean_reg_weight = mean_penalty_weight * reference_scale
        mean_penalty = mean_reg_weight * (mean_abs + mean_sq)
        total_loss += mean_penalty

        diagnostics.update({
            "loss_mean_penalty": mean_penalty.item(),
            "loss_mean_abs": (mean_reg_weight * mean_abs).item() if use_mean_abs else 0.0,
            "loss_mean_sq": (mean_reg_weight * mean_sq).item() if use_mean_sq else 0.0,
            "mean_reg_weight": mean_reg_weight.item(),
        })

    if use_std_penalty:
        delta_std = z_std - 1.0
        std_abs = delta_std.abs().mean() if use_std_abs else 0.0
        std_sq = delta_std.pow(2).mean() if use_std_sq else 0.0
        std_reg_weight = std_penalty_weight * reference_scale
        std_penalty = std_reg_weight * (std_abs + std_sq)
        total_loss += std_penalty

        diagnostics.update({
            "loss_std_penalty": std_penalty.item(),
            "loss_std_abs": (std_reg_weight * std_abs).item() if use_std_abs else 0.0,
            "loss_std_sq": (std_reg_weight * std_sq).item() if use_std_sq else 0.0,
            "std_reg_weight": std_reg_weight.item(),
        })

    if use_skew_penalty or use_kurtosis_penalty:
        from scipy.stats import skew, kurtosis

        z_np = z.detach().cpu().numpy()
        skew_vec = torch.tensor(skew(z_np, axis=0), device=z.device)
        kurt_vec = torch.tensor(kurtosis(z_np, axis=0, fisher=False), device=z.device)  # Normal kurtosis = 3

    if use_skew_penalty:
        delta_skew = skew_vec  # Target = 0
        skew_abs = delta_skew.abs() if use_skew_abs else 0.0
        skew_sq = delta_skew.pow(2) if use_skew_sq else 0.0

        skew_component = 0.0
        if use_skew_abs:
            skew_component += skew_abs.sum()
        if use_skew_sq:
            skew_component += skew_sq.sum()

        skew_reg_weight = skew_penalty_weight * reference_scale
        skew_penalty = skew_reg_weight * skew_component
        total_loss += skew_penalty

        diagnostics.update({
            "loss_skew_penalty": skew_penalty.item(),
            "loss_skew_abs_sum": (skew_reg_weight * skew_abs.sum()).item() if use_skew_abs else 0.0,
            "loss_skew_sq_sum": (skew_reg_weight * skew_sq.sum()).item() if use_skew_sq else 0.0,
            "skew_reg_weight": skew_reg_weight.item(),
        })

    if use_kurtosis_penalty:
        delta_kurt = kurt_vec - 3.0  # Target = 3
        kurt_abs = delta_kurt.abs() if use_kurtosis_abs else 0.0
        kurt_sq = delta_kurt.pow(2) if use_kurtosis_sq else 0.0

        kurt_component = 0.0
        if use_kurtosis_abs:
            kurt_component += kurt_abs.sum()
        if use_kurtosis_sq:
            kurt_component += kurt_sq.sum()

        kurt_reg_weight = kurtosis_penalty_weight * reference_scale
        kurtosis_penalty = kurt_reg_weight * kurt_component
        total_loss += kurtosis_penalty

        diagnostics.update({
            "loss_kurtosis_penalty": kurtosis_penalty.item(),
            "loss_kurtosis_abs_sum": (kurt_reg_weight * kurt_abs.sum()).item() if use_kurtosis_abs else 0.0,
            "loss_kurtosis_sq_sum": (kurt_reg_weight * kurt_sq.sum()).item() if use_kurtosis_sq else 0.0,
            "kurtosis_reg_weight": kurt_reg_weight.item(),
        })

    # Additional metrics
    mean_rrmse = math.sqrt(z_mean.pow(2).mean().item()) / (1e-8 + 1.0)  # target = 0
    std_rrmse = math.sqrt((z_std - 1.0).pow(2).mean().item()) / (1e-8 + 1.0)  # target = 1

    diagnostics.update({
        "loss_total": total_loss.item(),
        "logpz_mean": logpz.mean().item(),
        "logabsdet_mean": logabsdet.mean().item(),
        "logpz_min": logpz.min().item(),
        "logpz_max": logpz.max().item(),
        "logdet_min": logabsdet.min().item(),
        "logdet_max": logabsdet.max().item(),
        "z_mean_mean": z_mean.mean().item(),
        "z_mean_min": z_mean.min().item(),
        "z_mean_max": z_mean.max().item(),
        "z_std_mean": z_std.mean().item(),
        "z_std_min": z_std.min().item(),
        "z_std_max": z_std.max().item(),
        "rrmse_mean": mean_rrmse,
        "rrmse_std": std_rrmse,
    })

    return total_loss, diagnostics, (mean_rrmse, std_rrmse)









