from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2, kstest, kurtosis, skew

from losses.flow_pre_loss import flexible_flow_loss_from_model
from losses.flowgen_loss import flowgen_loss
from losses.mlp_loss import _reduce_by_group


DEFAULT_QUANTILE_RANGES: dict[str, tuple[float, float]] = {
    "0_50": (0.0, 0.5),
    "0_75": (0.0, 0.75),
    "25_75": (0.25, 0.75),
    "10_90": (0.10, 0.90),
    "5_95": (0.05, 0.95),
    "90_100": (0.90, 1.0),
}


def _ensure_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Model has no parameters to infer device from.") from exc


def _prepare_mlp_eval_tensors(
    *,
    X: pd.DataFrame,
    y: pd.DataFrame,
    condition_col: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    drop_cols = ["post_cleaning_index", condition_col]
    if "is_synth" in X.columns:
        drop_cols.append("is_synth")

    x = torch.tensor(
        X.drop(columns=drop_cols).values,
        dtype=torch.float32,
        device=device,
    )
    c = torch.tensor(
        X[condition_col].values,
        dtype=torch.long,
        device=device,
    )
    y_tensor = torch.tensor(
        y.drop(columns=["post_cleaning_index"]).values,
        dtype=torch.float32,
        device=device,
    )
    return x, y_tensor, c


def inverse_transform_tensor(tensor: torch.Tensor, scaler, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        scaler.inverse_transform(tensor.detach().cpu().numpy()),
        dtype=tensor.dtype,
        device=device,
    )


def _regression_quantiles(
    err: torch.Tensor,
    y: torch.Tensor,
    quantile_ranges: Mapping[str, tuple[float, float]],
) -> dict[str, dict[str, float | int]]:
    abs_err = torch.abs(err)
    out: dict[str, dict[str, float | int]] = {}

    for qname, (qlo, qhi) in quantile_ranges.items():
        lo = torch.quantile(abs_err, qlo)
        hi = torch.quantile(abs_err, qhi)
        mask = (abs_err >= lo) & (abs_err <= hi)
        if not torch.any(mask):
            continue

        e = err[mask]
        yt = y[mask]
        mse = float(torch.mean(e ** 2))
        rmse = float(torch.sqrt(torch.mean(e ** 2) + 1e-12))
        rrmse = float(
            (torch.sqrt(torch.mean(e ** 2)) / (torch.sqrt(torch.mean(yt ** 2)) + 1e-12)).item()
        )
        mae = float(torch.mean(torch.abs(e)))
        medae = float(torch.median(torch.abs(e)))
        mape = float(torch.mean(torch.abs(e) / torch.clamp(torch.abs(yt), min=1e-8)).item())

        yt_mu = yt.mean(dim=0)
        ss_res = torch.sum(e ** 2)
        ss_tot = torch.sum((yt - yt_mu) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

        out[qname] = {
            "mse": mse,
            "rmse": rmse,
            "rrmse": rrmse,
            "mae": mae,
            "medae": medae,
            "mape": mape,
            "r2": r2,
            "n": int(mask.sum().item()),
        }

    return out


def _derive_macro_and_worst(per_class: Mapping[int, Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    if not per_class:
        return {}, {}

    metric_names = [
        name
        for name in next(iter(per_class.values())).keys()
        if name not in {"quantiles", "n"}
    ]
    macro: dict[str, float] = {}
    worst: dict[str, float] = {}

    for metric_name in metric_names:
        vals = [float(per_class[cls][metric_name]) for cls in per_class if metric_name in per_class[cls]]
        if not vals:
            continue
        macro[metric_name] = float(np.mean(vals))
        if metric_name == "r2":
            worst[metric_name] = float(np.min(vals))
        else:
            worst[metric_name] = float(np.max(vals))

    macro["n"] = int(sum(int(per_class[cls].get("n", 0)) for cls in per_class))
    worst["n"] = macro["n"]
    return macro, worst


def compute_regression_metrics_from_preds(
    *,
    y_hat: torch.Tensor,
    y: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    quantile_ranges: Mapping[str, tuple[float, float]] = DEFAULT_QUANTILE_RANGES,
) -> dict[str, Any]:
    err = y_hat - y
    mse_tensor = torch.mean(err ** 2)
    mse = float(mse_tensor)
    rmse = float(torch.sqrt(mse_tensor + 1e-12))
    rrmse = float(
        (torch.sqrt(mse_tensor) / (torch.sqrt(torch.mean(y ** 2)) + 1e-12)).item()
    )
    mae = float(torch.mean(torch.abs(err)))
    medae = float(torch.median(torch.abs(err)))
    mape = float(torch.mean(torch.abs(err) / torch.clamp(torch.abs(y), min=1e-8)).item())

    y_mu = y.mean(dim=0)
    ss_res = torch.sum((y - y_hat) ** 2)
    ss_tot = torch.sum((y - y_mu) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    quantiles_overall = _regression_quantiles(err, y, quantile_ranges)
    overall = {
        "mse": mse,
        "rmse": rmse,
        "rrmse": rrmse,
        "mae": mae,
        "medae": medae,
        "mape": mape,
        "r2": r2,
        "n": int(y.shape[0]),
        "quantiles": quantiles_overall,
    }

    out_pc: dict[int, dict[str, Any]] = {}
    if c is not None:
        for cls in torch.unique(c):
            mask = (c == cls)
            if not torch.any(mask):
                continue

            yh = y_hat[mask]
            yt = y[mask]
            e = yh - yt
            mse_c = float(torch.mean(e ** 2))
            rmse_c = float(torch.sqrt(torch.mean(e ** 2) + 1e-12))
            rrmse_c = float(
                (torch.sqrt(torch.mean(e ** 2)) / (torch.sqrt(torch.mean(yt ** 2)) + 1e-12)).item()
            )
            mae_c = float(torch.mean(torch.abs(e)))
            medae_c = float(torch.median(torch.abs(e)))
            mape_c = float(torch.mean(torch.abs(e) / torch.clamp(torch.abs(yt), min=1e-8)).item())

            yt_mu = yt.mean(dim=0)
            ss_res_c = torch.sum((yt - yh) ** 2)
            ss_tot_c = torch.sum((yt - yt_mu) ** 2)
            r2_c = float(1.0 - ss_res_c / (ss_tot_c + 1e-12))

            out_pc[int(cls.item())] = {
                "mse": mse_c,
                "rmse": rmse_c,
                "rrmse": rrmse_c,
                "mae": mae_c,
                "medae": medae_c,
                "mape": mape_c,
                "r2": r2_c,
                "n": int(mask.sum().item()),
                "quantiles": _regression_quantiles(e, yt, quantile_ranges),
            }

    macro, worst = _derive_macro_and_worst(out_pc)

    return {
        "overall": overall,
        "macro": macro,
        "worst_class": worst,
        "per_class": out_pc,
    }


def compute_mlp_metrics(
    *,
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    condition_col: str,
    y_scaler=None,
    quantile_ranges: Mapping[str, tuple[float, float]] = DEFAULT_QUANTILE_RANGES,
) -> dict[str, Any]:
    device = _ensure_device(model)
    model.eval()
    x, y_tensor, c = _prepare_mlp_eval_tensors(X=X, y=y, condition_col=condition_col, device=device)

    with torch.no_grad():
        y_hat = model(x, c)

    if y_scaler is not None:
        y_hat = inverse_transform_tensor(y_hat, y_scaler, device=device)
        y_tensor = inverse_transform_tensor(y_tensor, y_scaler, device=device)

    return compute_regression_metrics_from_preds(
        y_hat=y_hat,
        y=y_tensor,
        c=c,
        quantile_ranges=quantile_ranges,
    )


def compute_mlp_grouped_loss(
    *,
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    condition_col: str,
    reduction_mode: str,
    regression_group_metric: str,
    y_scaler=None,
) -> float:
    device = _ensure_device(model)
    model.eval()
    x, y_tensor, c = _prepare_mlp_eval_tensors(X=X, y=y, condition_col=condition_col, device=device)

    with torch.no_grad():
        y_hat = model(x, c)

    if y_scaler is not None:
        y_hat = inverse_transform_tensor(y_hat, y_scaler, device=device)
        y_tensor = inverse_transform_tensor(y_tensor, y_scaler, device=device)

    mse_per_sample = torch.mean((y_hat - y_tensor) ** 2, dim=1)
    if regression_group_metric == "mse":
        loss = _reduce_by_group(mse_per_sample, c, mode=reduction_mode, group_metric="mean")
    elif regression_group_metric == "rmse":
        loss = _reduce_by_group(mse_per_sample, c, mode=reduction_mode, group_metric="rmse")
    elif regression_group_metric == "rrmse":
        loss = _reduce_by_group(
            mse_per_sample,
            c,
            mode=reduction_mode,
            group_metric="rrmse",
            y=y_tensor,
        )
    else:
        raise ValueError(f"Unsupported regression_group_metric '{regression_group_metric}'")

    return float(loss.detach().cpu())


def _get_by_path(d: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = d
    for tok in dotted.split("."):
        if isinstance(cur, Mapping) and tok in cur:
            cur = cur[tok]
        else:
            return None
    return cur


def select_metric(metrics: Mapping[str, Any], metric_path: str) -> float:
    toks = (metric_path or "").split(".")
    if not toks:
        raise KeyError("Empty metric_path.")

    head = toks[0]
    if head in {"overall", "macro", "worst_class"}:
        if len(toks) != 2:
            raise KeyError(f"Use '{head}.<metric>', got '{metric_path}'.")
        value = metrics.get(head, {}).get(toks[1], None)
    elif head == "per_class":
        pc = metrics.get("per_class", {})
        if len(toks) == 2:
            metric = toks[1]
            if not isinstance(pc, Mapping) or not pc:
                raise KeyError("metrics['per_class'] missing or empty.")
            total = 0.0
            for cls_id, cls_metrics in pc.items():
                if metric not in cls_metrics:
                    raise KeyError(f"Metric '{metric}' missing for class '{cls_id}'.")
                total += float(cls_metrics[metric])
            value = total
        elif len(toks) == 3:
            cls_str, metric = toks[1], toks[2]
            cls_id = int(cls_str)
            value = pc.get(cls_id, {}).get(metric, None)
        else:
            raise KeyError(f"Use 'per_class.<metric>' or 'per_class.<cls>.<metric>', got '{metric_path}'.")
    else:
        value = _get_by_path(metrics, metric_path)

    if value is None:
        raise KeyError(f"Metric path '{metric_path}' not found.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Metric at '{metric_path}' is not finite: {value}")
    return value


def latent_isotropy_stats_from_z(z_np: np.ndarray) -> dict[str, float]:
    _, d = z_np.shape
    sk = skew(z_np, axis=0, bias=False)
    ku = kurtosis(z_np, axis=0, fisher=False, bias=False)

    skewness_mean = float(np.mean(np.abs(sk)))
    kurtosis_mean = float(np.mean(ku))

    cov = np.cov(z_np, rowvar=False)
    eps = 1e-6
    cov_reg = cov + eps * np.eye(cov.shape[0], dtype=cov.dtype)
    evals = np.linalg.eigvalsh(cov_reg)
    eigval_std = float(np.std(evals))

    mu = z_np.mean(axis=0, keepdims=True)
    w, v = np.linalg.eigh(cov_reg)
    w_inv_sqrt = np.diag(1.0 / np.sqrt(w))
    z_whitened = (z_np - mu) @ v @ w_inv_sqrt
    d2 = np.sum(z_whitened ** 2, axis=1)
    dists = np.sqrt(d2)

    mahalanobis_mean = float(np.mean(dists))
    mahalanobis_median = float(np.median(dists))
    _, ks_p = kstest(d2, chi2(df=d).cdf)
    mahalanobis_ks_p = float(ks_p)

    return {
        "skewness_mean": skewness_mean,
        "kurtosis_mean": kurtosis_mean,
        "mahalanobis_mean": mahalanobis_mean,
        "mahalanobis_median": mahalanobis_median,
        "mahalanobis_ks_p": mahalanobis_ks_p,
        "eigval_std": eigval_std,
    }


def compute_reconstruction_metrics(
    *,
    truth: torch.Tensor,
    recon: torch.Tensor,
) -> dict[str, float | int]:
    rrmse = (
        torch.sqrt(torch.mean((recon - truth) ** 2)) /
        (torch.sqrt(torch.mean(truth ** 2)) + 1e-12)
    ).item()
    ss_res = torch.sum((truth - recon) ** 2)
    ss_tot = torch.sum((truth - truth.mean(dim=0)) ** 2)
    r2 = (1 - ss_res / (ss_tot + 1e-12)).item()
    return {
        "rrmse": float(rrmse),
        "r2": float(r2),
        "n": int(truth.shape[0]),
    }


def compute_flowpre_latent_isotropy_stats(model, x: torch.Tensor, c: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        z, _ = model.forward(x, c)
    return latent_isotropy_stats_from_z(z.detach().cpu().numpy())


def compute_flowpre_latent_isotropy_stats_per_class(model, x: torch.Tensor, c: torch.Tensor) -> dict[int, dict[str, Any]]:
    model.eval()
    out: dict[int, dict[str, Any]] = {}
    with torch.no_grad():
        for cls in torch.unique(c):
            mask = (c == cls)
            if not mask.any():
                continue
            z, _ = model.forward(x[mask], c[mask])
            stats = latent_isotropy_stats_from_z(z.detach().cpu().numpy())
            stats["n"] = int(mask.sum().item())
            out[int(cls.item())] = stats
    return out


def compute_flowpre_iso_rrmse_per_class(
    model,
    x: torch.Tensor,
    c: torch.Tensor,
    loss_kwargs: Mapping[str, Any],
) -> dict[int, dict[str, float | int]]:
    model.eval()
    per_class: dict[int, dict[str, float | int]] = {}
    with torch.no_grad():
        for cls in torch.unique(c):
            mask = (c == cls)
            if not mask.any():
                continue
            _, _, (rrmse_mean, rrmse_std) = flexible_flow_loss_from_model(
                model,
                x[mask],
                c[mask],
                **dict(loss_kwargs or {}),
            )
            per_class[int(cls.item())] = {
                "rrmse_mean": float(rrmse_mean),
                "rrmse_std": float(rrmse_std),
                "n": int(mask.sum().item()),
            }
    return per_class


def compute_flowpre_split_metrics(
    *,
    model,
    x: torch.Tensor,
    c: torch.Tensor,
    loss_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        z, _ = model.forward(x, c)
        x_recon = model.inverse(z, c)[0]
        _, _, (rrmse_mean_whole, rrmse_std_whole) = flexible_flow_loss_from_model(
            model,
            x,
            c,
            **dict(loss_kwargs or {}),
        )

    recon_metrics = compute_reconstruction_metrics(truth=x, recon=x_recon)
    return {
        "rrmse_recon": recon_metrics["rrmse"],
        "r2_recon": recon_metrics["r2"],
        "n": recon_metrics["n"],
        "rrmse_mean_whole": float(rrmse_mean_whole),
        "rrmse_std_whole": float(rrmse_std_whole),
        "per_class_iso_rrmse": compute_flowpre_iso_rrmse_per_class(model, x, c, loss_kwargs),
        "isotropy_stats": compute_flowpre_latent_isotropy_stats(model, x, c),
        "isotropy_stats_per_class": compute_flowpre_latent_isotropy_stats_per_class(model, x, c),
    }


def compute_flowgen_latent_isotropy_stats(model, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        z, _ = model.forward_xy(x, y, c)
    return latent_isotropy_stats_from_z(z.detach().cpu().numpy())


def compute_flowgen_latent_isotropy_stats_per_class(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
) -> dict[int, dict[str, Any]]:
    model.eval()
    out: dict[int, dict[str, Any]] = {}
    with torch.no_grad():
        for cls in torch.unique(c):
            mask = (c == cls)
            if not mask.any():
                continue
            z, _ = model.forward_xy(x[mask], y[mask], c[mask])
            stats = latent_isotropy_stats_from_z(z.detach().cpu().numpy())
            stats["n"] = int(mask.sum().item())
            out[int(cls.item())] = stats
    return out


def compute_flowgen_iso_rrmse_per_class(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    loss_kwargs: Mapping[str, Any],
) -> dict[int, dict[str, float | int]]:
    model.eval()
    per_class: dict[int, dict[str, float | int]] = {}
    with torch.no_grad():
        for cls in torch.unique(c):
            mask = (c == cls)
            if not mask.any():
                continue
            _, _, rmx, rxs, rmy, rys = flowgen_loss(
                model,
                x[mask],
                y[mask],
                c[mask],
                epoch=0,
                batch_index=0,
                **dict(loss_kwargs or {}),
            )
            per_class[int(cls.item())] = {
                "rrmse_x_mean": float(rmx),
                "rrmse_x_std": float(rxs),
                "rrmse_y_mean": float(rmy),
                "rrmse_y_std": float(rys),
                "n": int(mask.sum().item()),
            }
    return per_class


def compute_flowgen_split_metrics(
    *,
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    loss_kwargs: Mapping[str, Any],
    include_realism: bool = False,
    device: Optional[torch.device] = None,
    seed: int | None = None,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        z, _ = model.forward_xy(x, y, c)
        (x_recon, y_recon), _ = model.inverse_xy(z, c)

    recon_x = compute_reconstruction_metrics(truth=x, recon=x_recon)
    recon_y = compute_reconstruction_metrics(truth=y, recon=y_recon)

    loss_kwargs_forced = {**dict(loss_kwargs or {}), "enforce_realism": True}
    with torch.no_grad():
        _, _, rrmse_x_mean_whole, rrmse_x_std_whole, rrmse_y_mean_whole, rrmse_y_std_whole = flowgen_loss(
            model,
            x,
            y,
            c,
            epoch=0,
            batch_index=0,
            **loss_kwargs_forced,
        )

    out: dict[str, Any] = {
        "rrmse_x_recon": recon_x["rrmse"],
        "rrmse_y_recon": recon_y["rrmse"],
        "r2_x_recon": recon_x["r2"],
        "r2_y_recon": recon_y["r2"],
        "n": recon_x["n"],
        "loss_rrmse_x_mean_whole": float(rrmse_x_mean_whole),
        "loss_rrmse_x_std_whole": float(rrmse_x_std_whole),
        "loss_rrmse_y_mean_whole": float(rrmse_y_mean_whole),
        "loss_rrmse_y_std_whole": float(rrmse_y_std_whole),
        "per_class_iso_rrmse": compute_flowgen_iso_rrmse_per_class(model, x, y, c, loss_kwargs),
        "isotropy_stats": compute_flowgen_latent_isotropy_stats(model, x, y, c),
        "isotropy_stats_per_class": compute_flowgen_latent_isotropy_stats_per_class(model, x, y, c),
    }

    if include_realism:
        from evaluation.realism import compute_realism_metrics_for_set

        eval_device = device or _ensure_device(model)
        out["realism"] = compute_realism_metrics_for_set(
            model,
            x_ref=x,
            y_ref=y,
            c_ref=c,
            loss_like_kwargs=dict(loss_kwargs or {}),
            device=eval_device,
            seed=seed,
        )

    return out


def format_iso_dict(iso_dict: Mapping[int, Mapping[str, Any]]) -> str:
    compact: dict[int, Any] = {}
    for cls, vals in iso_dict.items():
        if all(k in vals for k in ("rrmse_x_mean", "rrmse_x_std", "rrmse_y_mean", "rrmse_y_std")):
            compact[int(cls)] = (
                float(vals["rrmse_x_mean"]),
                float(vals["rrmse_x_std"]),
                float(vals["rrmse_y_mean"]),
                float(vals["rrmse_y_std"]),
                int(vals.get("n", 0)),
            )
        elif all(k in vals for k in ("rrmse_mean", "rrmse_std")):
            compact[int(cls)] = (
                float(vals["rrmse_mean"]),
                float(vals["rrmse_std"]),
                int(vals.get("n", 0)),
            )
        else:
            compact[int(cls)] = {"n": int(vals.get("n", 0))}
    return str(compact)
