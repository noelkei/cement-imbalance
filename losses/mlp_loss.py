# losses/mlp_loss.py
from __future__ import annotations

from typing import Literal, Optional, Dict

import torch
import torch.nn as nn


@torch.no_grad()
def _unique_classes_in_batch(c: torch.Tensor) -> torch.Tensor:
    # sorted unique classes present in *this* batch
    return torch.unique(c.detach())


def _reduce_by_group(
    per_sample_loss: torch.Tensor,  # shape [N]
    c: torch.Tensor,                # shape [N] (context labels)
    mode: Literal["overall", "per_class_equal", "per_class_weighted"] = "overall",
    *,
    group_metric: Literal["mean", "rmse", "rrmse"] = "mean",
    y: Optional[torch.Tensor] = None,   # needed for rrmse
) -> torch.Tensor:
    """
    Generic reducer:
      - overall:           mean over all samples (== count-weighted over groups)
      - per_class_equal:   average of group metrics with equal weight per *present* class
      - per_class_weighted: count-weighted average of group metrics (≈ overall for 'mean')
    For regression:
      • group_metric 'mean'  -> average per-sample value in group (for MSE/MAE/etc.)
      • group_metric 'rmse'  -> sqrt(mean(per-sample MSE in group))
      • group_metric 'rrmse' -> rmse / sqrt(mean(y^2) in group)
    """
    if mode == "overall":
        return per_sample_loss.mean()

    classes = _unique_classes_in_batch(c)
    group_vals = []
    weights = []

    for cls in classes:
        m = (c == cls)
        if not torch.any(m):
            continue

        if group_metric == "mean":
            g = per_sample_loss[m].mean()
        elif group_metric in ("rmse", "rrmse"):
            # For these, per_sample_loss should carry MSE per sample (not already sqrt).
            mse_g = per_sample_loss[m].mean()
            rmse_g = torch.sqrt(mse_g + 1e-12)
            if group_metric == "rmse":
                g = rmse_g
            else:
                assert y is not None, "RRmse reduction needs 'y' to be provided."
                denom = torch.sqrt((y[m] ** 2).mean() + 1e-12)
                g = rmse_g / (denom + 1e-12)
        else:
            raise ValueError(f"Unsupported group_metric '{group_metric}'")

        group_vals.append(g)
        weights.append(m.float().sum())

    if len(group_vals) == 0:
        return per_sample_loss.mean()  # fallback

    group_stack = torch.stack(group_vals)  # [G]
    w = torch.stack(weights)               # [G]

    if mode == "per_class_equal":
        return group_stack.mean()
    elif mode == "per_class_weighted":
        wsum = w.sum().clamp_min(1.0)
        return (group_stack * w).sum() / wsum
    else:
        raise ValueError(f"Unsupported reduction mode '{mode}'")


def mlp_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    task: Literal["regression", "classification"] = "regression",
    reduction_mode: Literal["overall", "per_class_equal", "per_class_weighted"] = "overall",
    # --- Regression knobs ---
    regression_group_metric: Literal["mse", "rmse", "rrmse"] = "mse",
    # --- Classification knobs ---
    class_weights: Optional[torch.Tensor] = None,  # for CE/BCE weighting (on device)
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns:
        loss: scalar (torch.Tensor)
        diagnostics: dict of floats (safe for your epoch logger)
    Notes:
      • For regression, per-sample base is MSE over targets:
            mse_i = mean_j (y_hat[i,j] - y[i,j])^2
        You can reduce with:
          - overall                : mean over all i
          - per_class_equal        : equal-weight mean of group metrics across present classes in c
          - per_class_weighted     : count-weighted mean of group metrics across present classes
        And choose group metric:
          - 'mse'   : group mean(MSE_i)
          - 'rmse'  : sqrt(mean(MSE_i)) per group, then reduce
          - 'rrmse' : rmse / sqrt(mean(y^2)) per group, then reduce
      • For classification:
        - If model output dim == 1 ⇒ binary (BCE/ BCEWithLogits).
        - If model output dim  > 1 ⇒ multiclass (CrossEntropy).
        Grouping by 'c' uses the *context* label, not the classification target.
    """
    device = x.device
    model.train()  # the caller usually sets train/eval, but this keeps BN/Dropout consistent

    y_hat = model(x, c)  # [N, Dy]

    diagnostics: Dict[str, float] = {}

    if task == "regression":
        # Per-sample MSE over targets
        mse_per_sample = torch.mean((y_hat - y) ** 2, dim=1)  # [N]

        if regression_group_metric == "mse":
            loss = _reduce_by_group(
                mse_per_sample, c, mode=reduction_mode, group_metric="mean"
            )
        elif regression_group_metric == "rmse":
            loss = _reduce_by_group(
                mse_per_sample, c, mode=reduction_mode, group_metric="rmse"
            )
        elif regression_group_metric == "rrmse":
            loss = _reduce_by_group(
                mse_per_sample, c, mode=reduction_mode, group_metric="rrmse", y=y
            )
        else:
            raise ValueError(f"Unsupported regression_group_metric '{regression_group_metric}'")

        # Simple overall diagnostics (independent of chosen reduction)
        mse_overall = mse_per_sample.mean()
        rmse_overall = torch.sqrt(mse_overall + 1e-12)
        rrmse_overall = rmse_overall / (torch.sqrt((y ** 2).mean() + 1e-12) + 1e-12)

        # existing:
        mse_overall = mse_per_sample.mean()
        rmse_overall = torch.sqrt(mse_overall + 1e-12)
        rrmse_overall = rmse_overall / (torch.sqrt((y ** 2).mean() + 1e-12) + 1e-12)

        # NEW:
        abs_err = torch.abs(y_hat - y)
        mae_overall = abs_err.mean()

        mape_overall = (abs_err / (torch.clamp(torch.abs(y), min=1e-8))).mean()

        y_mu = y.mean(dim=0)
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y_mu) ** 2).sum()
        r2_overall = 1.0 - (ss_res / (ss_tot + 1e-12))

        # add to diagnostics:
        diagnostics["mse_overall"] = float(mse_overall.detach().cpu())
        diagnostics["rmse_overall"] = float(rmse_overall.detach().cpu())
        diagnostics["rrmse_overall"] = float(rrmse_overall.detach().cpu())
        diagnostics["mae_overall"] = float(mae_overall.detach().cpu())
        diagnostics["mape_overall"] = float(mape_overall.detach().cpu())
        diagnostics["r2_overall"] = float(r2_overall.detach().cpu())
        diagnostics["loss_final"] = float(loss.detach().cpu())

        return loss, diagnostics

    # ---------- classification ----------
    Dy = y_hat.shape[1]
    if Dy == 1:
        # Binary classification
        # Decide between BCE and BCEWithLogits based on model's final activation
        use_logits = not isinstance(getattr(model, "out_act", None), nn.Sigmoid)
        target = y.view(-1).float()

        if use_logits:
            criterion = nn.BCEWithLogitsLoss(weight=class_weights, reduction="none")
            per_sample = criterion(y_hat.view(-1), target)  # [N]
            probs = torch.sigmoid(y_hat.view(-1))
        else:
            criterion = nn.BCELoss(weight=class_weights, reduction="none")
            probs = y_hat.view(-1).clamp(1e-6, 1 - 1e-6)
            per_sample = criterion(probs, target)

        # Group reduction on per-sample loss using 'mean' metric
        loss = _reduce_by_group(per_sample, c, mode=reduction_mode, group_metric="mean")

        # Diagnostics: accuracy
        preds = (probs >= 0.5).long()
        acc = (preds == target.long()).float().mean()

        diagnostics["bce_overall"] = float(per_sample.mean().detach().cpu())
        diagnostics["acc"] = float(acc.detach().cpu())
        diagnostics["loss_final"] = float(loss.detach().cpu())
        return loss, diagnostics

    else:
        # Multiclass classification (CrossEntropy)
        # y expected as class indices [N] or [N,1]
        target_idx = y.view(-1).long()
        ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        per_sample = ce(y_hat, target_idx)  # [N]

        # Group reduction
        loss = _reduce_by_group(per_sample, c, mode=reduction_mode, group_metric="mean")

        # Diagnostics: accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == target_idx).float().mean()

        diagnostics["ce_overall"] = float(per_sample.mean().detach().cpu())
        diagnostics["acc"] = float(acc.detach().cpu())
        diagnostics["loss_final"] = float(loss.detach().cpu())
        return loss, diagnostics
