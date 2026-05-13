from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from evaluation.realism import compute_realism_metrics_for_set
from losses.flowgen_loss import (
    EPS,
    _fro_rel,
    _iqr,
    _ks_w1_matrix,
    _mmd_rbf_biased,
    _pearson_corr,
    _perdim_w1_normed,
    _softclip_asinh,
    _spearman_corr,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _date_str(value: Any) -> str | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0 and ts.nanosecond == 0:
        return ts.strftime("%Y-%m-%d")
    return ts.isoformat()


def resolve_temporal_realism_config(
    train_cfg: Mapping[str, Any] | None,
    loss_like_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    train_cfg = dict(train_cfg or {})
    loss_like_kwargs = dict(loss_like_kwargs or {})

    base_bootstrap = max(1, _safe_int(loss_like_kwargs.get("realism_bootstrap"), 10))
    base_rvr_bootstrap = max(1, _safe_int(loss_like_kwargs.get("realism_rvr_bootstrap"), base_bootstrap))
    temporal_bootstrap_default = min(base_bootstrap, 25)
    temporal_rvr_default = min(base_rvr_bootstrap, temporal_bootstrap_default)

    return {
        "enabled": bool(train_cfg.get("temporal_realism_enabled", True)),
        "write_sidecars": bool(train_cfg.get("temporal_realism_write_sidecars", True)),
        "bootstrap": max(1, _safe_int(train_cfg.get("temporal_realism_bootstrap"), temporal_bootstrap_default)),
        "rvr_bootstrap": max(
            1,
            _safe_int(train_cfg.get("temporal_realism_rvr_bootstrap"), temporal_rvr_default),
        ),
        "baseline_policy": "class_count_matched_to_slice",
        "families": ("quartiles", "prefix_suffix"),
    }


def _soft_ranks(x: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    x = x.to(torch.float32)
    n, d = x.shape
    ranks = []
    for j in range(d):
        col = x[:, j].unsqueeze(1)
        probs = torch.sigmoid((col - col.t()) / float(tau))
        ranks.append(1.0 + probs.sum(dim=1))
    return torch.stack(ranks, dim=1) if ranks else x.new_zeros((n, 0))


def _xyblock_fro_gap(
    xr: torch.Tensor,
    yr: torch.Tensor,
    xc: torch.Tensor,
    yc: torch.Tensor,
    *,
    corr: str,
    tau: float = 0.05,
    relative: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=xr.device.type, enabled=False):
        zr = torch.cat([xr, yr], dim=1).to(torch.float32)
        zc = torch.cat([xc, yc], dim=1).to(torch.float32)
        dx = xr.shape[1]

        if corr == "pearson":
            cr = _pearson_corr(zr)
            cc = _pearson_corr(zc)
        elif corr == "spearman":
            cr = _pearson_corr(_soft_ranks(zr, tau=float(tau)))
            cc = _pearson_corr(_soft_ranks(zc, tau=float(tau)))
        else:
            raise ValueError(f"Unsupported corr='{corr}'.")

        cr_xy = cr[:dx, dx:]
        cc_xy = cc[:dx, dx:]
        diff = cr_xy - cc_xy
        fro_abs = torch.linalg.norm(diff, ord="fro")
        fro_rel = fro_abs / (torch.linalg.norm(cr_xy, ord="fro") + EPS) if relative else fro_abs
        return fro_abs, fro_rel


def _empty_suite(*, include_corr: bool) -> dict[str, float | None]:
    base: dict[str, float | None] = {
        "ks_mean": 0.0,
        "ks_median": 0.0,
        "w1_mean": 0.0,
        "w1_median": 0.0,
        "w1_mean_trainaligned": 0.0,
        "w1_median_trainaligned": 0.0,
        "mmd2_rvs": 0.0,
        "mmd2_rvr_med": 0.0,
        "mmd2_ratio": 0.0,
        "xy_pearson_fro": 0.0,
        "xy_pearson_fro_rel": 0.0,
        "xy_spearman_fro": 0.0,
        "xy_spearman_fro_rel": 0.0,
        "_xycorr_count": 0.0,
    }
    if include_corr:
        base.update(
            {
                "pearson_fro": 0.0,
                "pearson_fro_rel": 0.0,
                "spearman_fro": 0.0,
                "spearman_fro_rel": 0.0,
                "_corr_count": 0.0,
            }
        )
    return base


def _finalize_suite(acc: Mapping[str, float | None], *, boots: int) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "ks_mean": float(acc["ks_mean"]) / boots,
        "ks_median": float(acc["ks_median"]) / boots,
        "w1_mean": float(acc["w1_mean"]) / boots,
        "w1_median": float(acc["w1_median"]) / boots,
        "w1_mean_trainaligned": float(acc["w1_mean_trainaligned"]) / boots,
        "w1_median_trainaligned": float(acc["w1_median_trainaligned"]) / boots,
        "pearson_fro": None,
        "pearson_fro_rel": None,
        "spearman_fro": None,
        "spearman_fro_rel": None,
        "xy_pearson_fro": None,
        "xy_pearson_fro_rel": None,
        "xy_spearman_fro": None,
        "xy_spearman_fro_rel": None,
        "mmd2_rvs": float(acc["mmd2_rvs"]) / boots,
        "mmd2_rvr_med": float(acc["mmd2_rvr_med"]) / boots,
        "mmd2_ratio": float(acc["mmd2_ratio"]) / boots,
    }
    if "_corr_count" in acc and float(acc["_corr_count"]) > 0:
        k = float(acc["_corr_count"])
        out["pearson_fro"] = float(acc["pearson_fro"]) / k
        out["pearson_fro_rel"] = float(acc["pearson_fro_rel"]) / k
        out["spearman_fro"] = float(acc["spearman_fro"]) / k
        out["spearman_fro_rel"] = float(acc["spearman_fro_rel"]) / k
    if "_xycorr_count" in acc and float(acc["_xycorr_count"]) > 0:
        kxy = float(acc["_xycorr_count"])
        out["xy_pearson_fro"] = float(acc["xy_pearson_fro"]) / kxy
        out["xy_pearson_fro_rel"] = float(acc["xy_pearson_fro_rel"]) / kxy
        out["xy_spearman_fro"] = float(acc["xy_spearman_fro"]) / kxy
        out["xy_spearman_fro_rel"] = float(acc["xy_spearman_fro_rel"]) / kxy
    return out


def _apply_trainaligned_w1(
    values: torch.Tensor,
    *,
    softclip_s: float,
    clip_perdim: float,
) -> torch.Tensor:
    aligned = values.to(torch.float32)
    if softclip_s > 0:
        aligned = _softclip_asinh(aligned, float(softclip_s))
    if clip_perdim > 0:
        aligned = torch.clamp(aligned, max=float(clip_perdim))
    return aligned


def _aggregate_trainaligned_w1(
    values: torch.Tensor,
    *,
    softclip_s: float,
    clip_perdim: float,
    agg_softcap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    aligned = _apply_trainaligned_w1(
        values,
        softclip_s=float(softclip_s),
        clip_perdim=float(clip_perdim),
    )
    mean = aligned.mean()
    median = aligned.median()
    if agg_softcap > 0:
        mean = _softclip_asinh(mean, float(agg_softcap))
        median = _softclip_asinh(median, float(agg_softcap))
    return mean, median


def _aggregate_trainaligned_w1_xy(
    x_values: torch.Tensor,
    y_values: torch.Tensor,
    *,
    loss_like_kwargs: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    x_aligned = _apply_trainaligned_w1(
        x_values,
        softclip_s=float(loss_like_kwargs.get("w1_x_softclip_s", 0.0)),
        clip_perdim=float(loss_like_kwargs.get("w1_x_clip_perdim", 0.0)),
    )
    y_aligned = _apply_trainaligned_w1(
        y_values,
        softclip_s=float(loss_like_kwargs.get("w1_y_softclip_s", 0.0)),
        clip_perdim=float(loss_like_kwargs.get("w1_y_clip_perdim", 0.0)),
    )
    aligned = torch.cat([x_aligned, y_aligned], dim=0)
    mean = aligned.mean()
    median = aligned.median()
    agg_x = float(loss_like_kwargs.get("w1_x_agg_softcap", 0.0))
    agg_y = float(loss_like_kwargs.get("w1_y_agg_softcap", 0.0))
    if agg_x > 0 or agg_y > 0:
        agg = max(agg_x, agg_y)
        mean = _softclip_asinh(mean, agg)
        median = _softclip_asinh(median, agg)
    return mean, median


def build_temporal_order_table(
    r_val: pd.DataFrame,
    cxy_val: pd.DataFrame,
    *,
    condition_col: str = "type",
) -> pd.DataFrame:
    if "post_cleaning_index" not in r_val.columns:
        raise ValueError("r_val must include 'post_cleaning_index' to build temporal order.")
    if "post_cleaning_index" not in cxy_val.columns:
        raise ValueError("cxy_val must include 'post_cleaning_index' to build temporal order.")
    if condition_col not in cxy_val.columns:
        raise ValueError(f"cxy_val must include condition column '{condition_col}'.")

    cxy_index = cxy_val[["post_cleaning_index", condition_col]].copy()
    cxy_index["tensor_row_index"] = np.arange(len(cxy_index), dtype=int)
    cxy_index.rename(columns={condition_col: "class_id"}, inplace=True)

    keep_cols = ["post_cleaning_index"]
    for col in ("date", "source_row_number", "split_row_id", "split"):
        if col in r_val.columns:
            keep_cols.append(col)
    merged = pd.merge(
        cxy_index,
        r_val[keep_cols].copy(),
        on="post_cleaning_index",
        how="left",
        validate="one_to_one",
    )

    merged["date_sort"] = pd.to_datetime(merged.get("date"), errors="coerce")
    merged["source_row_number_sort"] = pd.to_numeric(merged.get("source_row_number"), errors="coerce")
    merged["source_row_number_sort"] = merged["source_row_number_sort"].fillna(np.inf)
    merged["date_missing"] = merged["date_sort"].isna().astype(int)

    merged = merged.sort_values(
        ["date_missing", "date_sort", "source_row_number_sort", "post_cleaning_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    merged["temporal_rank"] = np.arange(len(merged), dtype=int)
    return merged


def build_temporal_slice_assignments(order_table: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    n = len(order_table)
    if n <= 0:
        raise ValueError("Temporal slice assignments require a non-empty validation order table.")

    families: dict[str, list[dict[str, Any]]] = {"quartiles": [], "prefix_suffix": []}

    quartile_parts = np.array_split(np.arange(n, dtype=int), 4)
    quartile_fracs = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00)]
    for idx, row_idx in enumerate(quartile_parts, start=1):
        families["quartiles"].append(
            {
                "family": "quartiles",
                "slice_id": f"q{idx}",
                "label": f"{int(quartile_fracs[idx - 1][0] * 100)}-{int(quartile_fracs[idx - 1][1] * 100)}%",
                "order": idx,
                "start_frac": quartile_fracs[idx - 1][0],
                "end_frac": quartile_fracs[idx - 1][1],
                "row_indices": row_idx,
            }
        )

    def _prefix_suffix_rows(frac: float) -> tuple[np.ndarray, np.ndarray]:
        count = max(1, int(np.floor(n * frac)))
        return np.arange(count, dtype=int), np.arange(n - count, n, dtype=int)

    prefix25, suffix25 = _prefix_suffix_rows(0.25)
    prefix50, suffix50 = _prefix_suffix_rows(0.50)
    families["prefix_suffix"].extend(
        [
            {
                "family": "prefix_suffix",
                "slice_id": "prefix_25",
                "label": "prefix-25%",
                "order": 1,
                "start_frac": 0.00,
                "end_frac": 0.25,
                "row_indices": prefix25,
            },
            {
                "family": "prefix_suffix",
                "slice_id": "suffix_25",
                "label": "suffix-25%",
                "order": 2,
                "start_frac": 0.75,
                "end_frac": 1.00,
                "row_indices": suffix25,
            },
            {
                "family": "prefix_suffix",
                "slice_id": "prefix_50",
                "label": "prefix-50%",
                "order": 3,
                "start_frac": 0.00,
                "end_frac": 0.50,
                "row_indices": prefix50,
            },
            {
                "family": "prefix_suffix",
                "slice_id": "suffix_50",
                "label": "suffix-50%",
                "order": 4,
                "start_frac": 0.50,
                "end_frac": 1.00,
                "row_indices": suffix50,
            },
        ]
    )
    return families


def _slice_metadata(order_table: pd.DataFrame, spec: Mapping[str, Any]) -> dict[str, Any]:
    rows = order_table.iloc[list(spec["row_indices"])]
    class_counts = {
        str(cls): int(count)
        for cls, count in rows["class_id"].value_counts(dropna=False).sort_index().items()
    }
    return {
        "label": str(spec["label"]),
        "order": int(spec["order"]),
        "start_frac": float(spec["start_frac"]),
        "end_frac": float(spec["end_frac"]),
        "n_obs": int(len(rows)),
        "start_date": _date_str(rows["date_sort"].iloc[0] if len(rows) else None),
        "end_date": _date_str(rows["date_sort"].iloc[-1] if len(rows) else None),
        "class_counts": class_counts,
    }


def _metrics_at(dct: Mapping[str, Any], *path: str) -> float | None:
    cur: Any = dct
    for token in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(token)
    return _safe_float(cur)


def _suite_summary_delta(
    slices: Mapping[str, Mapping[str, Any]],
    *,
    left_id: str,
    right_id: str,
    suite_name: str,
    metric_name: str,
) -> float | None:
    left = _metrics_at(slices.get(left_id, {}), suite_name, "overall", metric_name)
    right = _metrics_at(slices.get(right_id, {}), suite_name, "overall", metric_name)
    if left is None or right is None:
        return None
    return float(right - left)


def _build_family_summary(family_name: str, family_slices: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if family_name == "quartiles":
        return {
            "generated_vs_slice": {
                "overall_w1_q4_minus_q1": _suite_summary_delta(
                    family_slices, left_id="q1", right_id="q4", suite_name="generated_vs_slice", metric_name="w1_mean"
                ),
                "overall_ks_q4_minus_q1": _suite_summary_delta(
                    family_slices, left_id="q1", right_id="q4", suite_name="generated_vs_slice", metric_name="ks_mean"
                ),
            },
            "train_ref_vs_slice_real": {
                "overall_w1_q4_minus_q1": _suite_summary_delta(
                    family_slices,
                    left_id="q1",
                    right_id="q4",
                    suite_name="train_ref_vs_slice_real",
                    metric_name="w1_mean",
                ),
                "overall_ks_q4_minus_q1": _suite_summary_delta(
                    family_slices,
                    left_id="q1",
                    right_id="q4",
                    suite_name="train_ref_vs_slice_real",
                    metric_name="ks_mean",
                ),
            },
        }
    if family_name == "prefix_suffix":
        return {
            "generated_vs_slice": {
                "overall_w1_suffix25_minus_prefix25": _suite_summary_delta(
                    family_slices,
                    left_id="prefix_25",
                    right_id="suffix_25",
                    suite_name="generated_vs_slice",
                    metric_name="w1_mean",
                ),
                "overall_ks_suffix25_minus_prefix25": _suite_summary_delta(
                    family_slices,
                    left_id="prefix_25",
                    right_id="suffix_25",
                    suite_name="generated_vs_slice",
                    metric_name="ks_mean",
                ),
                "overall_w1_suffix50_minus_prefix50": _suite_summary_delta(
                    family_slices,
                    left_id="prefix_50",
                    right_id="suffix_50",
                    suite_name="generated_vs_slice",
                    metric_name="w1_mean",
                ),
                "overall_ks_suffix50_minus_prefix50": _suite_summary_delta(
                    family_slices,
                    left_id="prefix_50",
                    right_id="suffix_50",
                    suite_name="generated_vs_slice",
                    metric_name="ks_mean",
                ),
            },
            "train_ref_vs_slice_real": {
                "overall_w1_suffix25_minus_prefix25": _suite_summary_delta(
                    family_slices,
                    left_id="prefix_25",
                    right_id="suffix_25",
                    suite_name="train_ref_vs_slice_real",
                    metric_name="w1_mean",
                ),
                "overall_ks_suffix25_minus_prefix25": _suite_summary_delta(
                    family_slices,
                    left_id="prefix_25",
                    right_id="suffix_25",
                    suite_name="train_ref_vs_slice_real",
                    metric_name="ks_mean",
                ),
                "overall_w1_suffix50_minus_prefix50": _suite_summary_delta(
                    family_slices,
                    left_id="prefix_50",
                    right_id="suffix_50",
                    suite_name="train_ref_vs_slice_real",
                    metric_name="w1_mean",
                ),
                "overall_ks_suffix50_minus_prefix50": _suite_summary_delta(
                    family_slices,
                    left_id="prefix_50",
                    right_id="suffix_50",
                    suite_name="train_ref_vs_slice_real",
                    metric_name="ks_mean",
                ),
            },
        }
    return {}


def _accumulate_real_vs_real_once(
    *,
    ref_x_by_cls: Mapping[int, torch.Tensor],
    ref_y_by_cls: Mapping[int, torch.Tensor],
    cmp_x_by_cls: Mapping[int, torch.Tensor],
    cmp_y_by_cls: Mapping[int, torch.Tensor],
    loss_like_kwargs: Mapping[str, Any],
    overall_acc: dict[str, float | None],
    x_acc: dict[str, float | None],
    y_acc: dict[str, float | None],
    per_class_acc: dict[int, dict[str, dict[str, float | None]]],
    rvr_boots: int,
    generator: torch.Generator,
    device: torch.device,
) -> None:
    classes = [int(cls) for cls in ref_x_by_cls.keys()]
    xr_all = torch.cat([ref_x_by_cls[cls] for cls in classes], dim=0)
    yr_all = torch.cat([ref_y_by_cls[cls] for cls in classes], dim=0)
    xc_all = torch.cat([cmp_x_by_cls[cls] for cls in classes], dim=0)
    yc_all = torch.cat([cmp_y_by_cls[cls] for cls in classes], dim=0)

    real_all_cat = torch.cat([xr_all, yr_all], dim=1)
    cmp_all_cat = torch.cat([xc_all, yc_all], dim=1)

    denom_all_global = _iqr(real_all_cat).to(torch.float32).clamp_min(1e-4)
    denom_x_global = _iqr(xr_all).to(torch.float32).clamp_min(1e-4)
    denom_y_global = _iqr(yr_all).to(torch.float32).clamp_min(1e-4)

    ks_grid_x = int(loss_like_kwargs.get("ks_grid_points_x", 64))
    ks_tau_x = float(loss_like_kwargs.get("ks_tau_x", 0.05))
    w1_norm_x = str(loss_like_kwargs.get("w1_x_norm", "iqr"))
    ks_grid_y = int(loss_like_kwargs.get("ks_grid_points_y", 64))
    ks_tau_y = float(loss_like_kwargs.get("ks_tau_y", 0.05))
    w1_norm_y = str(loss_like_kwargs.get("w1_y_norm", "iqr"))
    ks_grid_all = int(loss_like_kwargs.get("ks_grid_points_all", max(ks_grid_x, ks_grid_y)))
    ks_tau_all = float(loss_like_kwargs.get("ks_tau_all", 0.5 * (ks_tau_x + ks_tau_y)))
    w1_norm_all = str(loss_like_kwargs.get("w1_norm_all", "iqr"))
    corr_xy_tau = float(loss_like_kwargs.get("corr_xy_tau", 0.05))

    def _mmd_rvr_median(real_mat: torch.Tensor, sigma: float, n_draws: int) -> float:
        n = real_mat.size(0)
        vals = []
        for _ in range(n_draws):
            idx1 = torch.randint(0, n, (n,), generator=generator, device="cpu").to(device)
            idx2 = torch.randint(0, n, (n,), generator=generator, device="cpu").to(device)
            r1 = real_mat.index_select(0, idx1)
            r2 = real_mat.index_select(0, idx2)
            vals.append(float(_mmd_rbf_biased(r1, r2, sigma=sigma)[0].detach()))
        return float(np.median(vals)) if vals else 0.0

    with torch.no_grad():
        ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
            real_all_cat,
            cmp_all_cat,
            grid_points=ks_grid_all,
            tau=ks_tau_all,
            norm=w1_norm_all,
            denom_override=denom_all_global,
        )
        w1x_all_perdim, _ = _perdim_w1_normed(xr_all, xc_all, norm=w1_norm_x, denom_override=denom_x_global)
        w1y_all_perdim, _ = _perdim_w1_normed(yr_all, yc_all, norm=w1_norm_y, denom_override=denom_y_global)
        w1_mean_trainaligned_all, w1_med_trainaligned_all = _aggregate_trainaligned_w1_xy(
            w1x_all_perdim,
            w1y_all_perdim,
            loss_like_kwargs=loss_like_kwargs,
        )
        mmd2_rvs_all, sigma_all = _mmd_rbf_biased(real_all_cat, cmp_all_cat)
        mmd2_rvr_med_all = _mmd_rvr_median(real_all_cat, sigma_all, rvr_boots)
        overall_acc["ks_mean"] += float(ks_mean.detach())
        overall_acc["ks_median"] += float(ks_med.detach())
        overall_acc["w1_mean"] += float(w1_mean.detach())
        overall_acc["w1_median"] += float(w1_med.detach())
        overall_acc["w1_mean_trainaligned"] += float(w1_mean_trainaligned_all.detach())
        overall_acc["w1_median_trainaligned"] += float(w1_med_trainaligned_all.detach())
        overall_acc["mmd2_rvs"] += float(mmd2_rvs_all.detach())
        overall_acc["mmd2_rvr_med"] += float(mmd2_rvr_med_all)
        overall_acc["mmd2_ratio"] += float((mmd2_rvs_all / (mmd2_rvr_med_all + EPS)).detach())

        pe_abs_xy, pe_rel_xy = _xyblock_fro_gap(xr_all, yr_all, xc_all, yc_all, corr="pearson", relative=True)
        sp_abs_xy, sp_rel_xy = _xyblock_fro_gap(
            xr_all,
            yr_all,
            xc_all,
            yc_all,
            corr="spearman",
            tau=corr_xy_tau,
            relative=True,
        )
        overall_acc["xy_pearson_fro"] += float(pe_abs_xy.detach())
        overall_acc["xy_pearson_fro_rel"] += float(pe_rel_xy.detach())
        overall_acc["xy_spearman_fro"] += float(sp_abs_xy.detach())
        overall_acc["xy_spearman_fro_rel"] += float(sp_rel_xy.detach())
        overall_acc["_xycorr_count"] += 1.0

        ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
            xr_all,
            xc_all,
            grid_points=ks_grid_x,
            tau=ks_tau_x,
            norm=w1_norm_x,
            denom_override=denom_x_global,
        )
        w1x_perdim, _ = _perdim_w1_normed(xr_all, xc_all, norm=w1_norm_x, denom_override=denom_x_global)
        w1_mean_trainaligned_x, w1_med_trainaligned_x = _aggregate_trainaligned_w1(
            w1x_perdim,
            softclip_s=float(loss_like_kwargs.get("w1_x_softclip_s", 0.0)),
            clip_perdim=float(loss_like_kwargs.get("w1_x_clip_perdim", 0.0)),
            agg_softcap=float(loss_like_kwargs.get("w1_x_agg_softcap", 0.0)),
        )
        mmd2_rvs_x, sigma_x = _mmd_rbf_biased(xr_all, xc_all)
        mmd2_rvr_med_x = _mmd_rvr_median(xr_all, sigma_x, rvr_boots)
        x_acc["ks_mean"] += float(ks_mean.detach())
        x_acc["ks_median"] += float(ks_med.detach())
        x_acc["w1_mean"] += float(w1_mean.detach())
        x_acc["w1_median"] += float(w1_med.detach())
        x_acc["w1_mean_trainaligned"] += float(w1_mean_trainaligned_x.detach())
        x_acc["w1_median_trainaligned"] += float(w1_med_trainaligned_x.detach())
        x_acc["mmd2_rvs"] += float(mmd2_rvs_x.detach())
        x_acc["mmd2_rvr_med"] += float(mmd2_rvr_med_x)
        x_acc["mmd2_ratio"] += float((mmd2_rvs_x / (mmd2_rvr_med_x + EPS)).detach())
        if xr_all.shape[1] >= 2:
            cr = _pearson_corr(xr_all)
            cc = _pearson_corr(xc_all)
            pe_abs, pe_rel = _fro_rel(cr, cc)
            crs = _spearman_corr(xr_all)
            ccs = _spearman_corr(xc_all)
            sp_abs, sp_rel = _fro_rel(crs, ccs)
            x_acc["pearson_fro"] += float(pe_abs.detach())
            x_acc["pearson_fro_rel"] += float(pe_rel.detach())
            x_acc["spearman_fro"] += float(sp_abs.detach())
            x_acc["spearman_fro_rel"] += float(sp_rel.detach())
            x_acc["_corr_count"] += 1.0

        ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
            yr_all,
            yc_all,
            grid_points=ks_grid_y,
            tau=ks_tau_y,
            norm=w1_norm_y,
            denom_override=denom_y_global,
        )
        w1y_perdim, _ = _perdim_w1_normed(yr_all, yc_all, norm=w1_norm_y, denom_override=denom_y_global)
        w1_mean_trainaligned_y, w1_med_trainaligned_y = _aggregate_trainaligned_w1(
            w1y_perdim,
            softclip_s=float(loss_like_kwargs.get("w1_y_softclip_s", 0.0)),
            clip_perdim=float(loss_like_kwargs.get("w1_y_clip_perdim", 0.0)),
            agg_softcap=float(loss_like_kwargs.get("w1_y_agg_softcap", 0.0)),
        )
        mmd2_rvs_y, sigma_y = _mmd_rbf_biased(yr_all, yc_all)
        mmd2_rvr_med_y = _mmd_rvr_median(yr_all, sigma_y, rvr_boots)
        y_acc["ks_mean"] += float(ks_mean.detach())
        y_acc["ks_median"] += float(ks_med.detach())
        y_acc["w1_mean"] += float(w1_mean.detach())
        y_acc["w1_median"] += float(w1_med.detach())
        y_acc["w1_mean_trainaligned"] += float(w1_mean_trainaligned_y.detach())
        y_acc["w1_median_trainaligned"] += float(w1_med_trainaligned_y.detach())
        y_acc["mmd2_rvs"] += float(mmd2_rvs_y.detach())
        y_acc["mmd2_rvr_med"] += float(mmd2_rvr_med_y)
        y_acc["mmd2_ratio"] += float((mmd2_rvs_y / (mmd2_rvr_med_y + EPS)).detach())
        if yr_all.shape[1] >= 2:
            cr = _pearson_corr(yr_all)
            cc = _pearson_corr(yc_all)
            pe_abs, pe_rel = _fro_rel(cr, cc)
            crs = _spearman_corr(yr_all)
            ccs = _spearman_corr(yc_all)
            sp_abs, sp_rel = _fro_rel(crs, ccs)
            y_acc["pearson_fro"] += float(pe_abs.detach())
            y_acc["pearson_fro_rel"] += float(pe_rel.detach())
            y_acc["spearman_fro"] += float(sp_abs.detach())
            y_acc["spearman_fro_rel"] += float(sp_rel.detach())
            y_acc["_corr_count"] += 1.0

        for cls in classes:
            xr_c = ref_x_by_cls[cls]
            yr_c = ref_y_by_cls[cls]
            xc_c = cmp_x_by_cls[cls]
            yc_c = cmp_y_by_cls[cls]
            pc_over = per_class_acc[cls]["overall"]
            pc_x = per_class_acc[cls]["x"]
            pc_y = per_class_acc[cls]["y"]

            real_cat = torch.cat([xr_c, yr_c], dim=1)
            cmp_cat = torch.cat([xc_c, yc_c], dim=1)
            denom_cls_all = _iqr(real_cat).to(torch.float32).clamp_min(1e-4)
            denom_cls_x = _iqr(xr_c).to(torch.float32).clamp_min(1e-4)
            denom_cls_y = _iqr(yr_c).to(torch.float32).clamp_min(1e-4)
            ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                real_cat,
                cmp_cat,
                grid_points=ks_grid_all,
                tau=ks_tau_all,
                norm=w1_norm_all,
                denom_override=denom_cls_all,
            )
            w1x_cls_perdim, _ = _perdim_w1_normed(xr_c, xc_c, norm=w1_norm_x, denom_override=denom_cls_x)
            w1y_cls_perdim, _ = _perdim_w1_normed(yr_c, yc_c, norm=w1_norm_y, denom_override=denom_cls_y)
            w1_mean_trainaligned_cls_all, w1_med_trainaligned_cls_all = _aggregate_trainaligned_w1_xy(
                w1x_cls_perdim,
                w1y_cls_perdim,
                loss_like_kwargs=loss_like_kwargs,
            )
            mmd2_rvs_cls, sigma_cls = _mmd_rbf_biased(real_cat, cmp_cat)
            mmd2_rvr_med_cls = _mmd_rvr_median(real_cat, sigma_cls, rvr_boots)
            pc_over["ks_mean"] += float(ks_mean.detach())
            pc_over["ks_median"] += float(ks_med.detach())
            pc_over["w1_mean"] += float(w1_mean.detach())
            pc_over["w1_median"] += float(w1_med.detach())
            pc_over["w1_mean_trainaligned"] += float(w1_mean_trainaligned_cls_all.detach())
            pc_over["w1_median_trainaligned"] += float(w1_med_trainaligned_cls_all.detach())
            pc_over["mmd2_rvs"] += float(mmd2_rvs_cls.detach())
            pc_over["mmd2_rvr_med"] += float(mmd2_rvr_med_cls)
            pc_over["mmd2_ratio"] += float((mmd2_rvs_cls / (mmd2_rvr_med_cls + EPS)).detach())

            pe_abs_xy, pe_rel_xy = _xyblock_fro_gap(xr_c, yr_c, xc_c, yc_c, corr="pearson", relative=True)
            sp_abs_xy, sp_rel_xy = _xyblock_fro_gap(
                xr_c,
                yr_c,
                xc_c,
                yc_c,
                corr="spearman",
                tau=corr_xy_tau,
                relative=True,
            )
            pc_over["xy_pearson_fro"] += float(pe_abs_xy.detach())
            pc_over["xy_pearson_fro_rel"] += float(pe_rel_xy.detach())
            pc_over["xy_spearman_fro"] += float(sp_abs_xy.detach())
            pc_over["xy_spearman_fro_rel"] += float(sp_rel_xy.detach())
            pc_over["_xycorr_count"] += 1.0

            ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                xr_c,
                xc_c,
                grid_points=ks_grid_x,
                tau=ks_tau_x,
                norm=w1_norm_x,
                denom_override=denom_cls_x,
            )
            w1x_cls_perdim, _ = _perdim_w1_normed(xr_c, xc_c, norm=w1_norm_x, denom_override=denom_cls_x)
            w1_mean_trainaligned_cls_x, w1_med_trainaligned_cls_x = _aggregate_trainaligned_w1(
                w1x_cls_perdim,
                softclip_s=float(loss_like_kwargs.get("w1_x_softclip_s", 0.0)),
                clip_perdim=float(loss_like_kwargs.get("w1_x_clip_perdim", 0.0)),
                agg_softcap=float(loss_like_kwargs.get("w1_x_agg_softcap", 0.0)),
            )
            mmd2_rvs_cls, sigma_cls = _mmd_rbf_biased(xr_c, xc_c)
            mmd2_rvr_med_cls = _mmd_rvr_median(xr_c, sigma_cls, rvr_boots)
            pc_x["ks_mean"] += float(ks_mean.detach())
            pc_x["ks_median"] += float(ks_med.detach())
            pc_x["w1_mean"] += float(w1_mean.detach())
            pc_x["w1_median"] += float(w1_med.detach())
            pc_x["w1_mean_trainaligned"] += float(w1_mean_trainaligned_cls_x.detach())
            pc_x["w1_median_trainaligned"] += float(w1_med_trainaligned_cls_x.detach())
            pc_x["mmd2_rvs"] += float(mmd2_rvs_cls.detach())
            pc_x["mmd2_rvr_med"] += float(mmd2_rvr_med_cls)
            pc_x["mmd2_ratio"] += float((mmd2_rvs_cls / (mmd2_rvr_med_cls + EPS)).detach())
            if xr_c.shape[1] >= 2:
                cr = _pearson_corr(xr_c)
                cc = _pearson_corr(xc_c)
                pe_abs, pe_rel = _fro_rel(cr, cc)
                crs = _spearman_corr(xr_c)
                ccs = _spearman_corr(xc_c)
                sp_abs, sp_rel = _fro_rel(crs, ccs)
                pc_x["pearson_fro"] += float(pe_abs.detach())
                pc_x["pearson_fro_rel"] += float(pe_rel.detach())
                pc_x["spearman_fro"] += float(sp_abs.detach())
                pc_x["spearman_fro_rel"] += float(sp_rel.detach())
                pc_x["_corr_count"] += 1.0

            ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                yr_c,
                yc_c,
                grid_points=ks_grid_y,
                tau=ks_tau_y,
                norm=w1_norm_y,
                denom_override=denom_cls_y,
            )
            w1y_cls_perdim, _ = _perdim_w1_normed(yr_c, yc_c, norm=w1_norm_y, denom_override=denom_cls_y)
            w1_mean_trainaligned_cls_y, w1_med_trainaligned_cls_y = _aggregate_trainaligned_w1(
                w1y_cls_perdim,
                softclip_s=float(loss_like_kwargs.get("w1_y_softclip_s", 0.0)),
                clip_perdim=float(loss_like_kwargs.get("w1_y_clip_perdim", 0.0)),
                agg_softcap=float(loss_like_kwargs.get("w1_y_agg_softcap", 0.0)),
            )
            mmd2_rvs_cls, sigma_cls = _mmd_rbf_biased(yr_c, yc_c)
            mmd2_rvr_med_cls = _mmd_rvr_median(yr_c, sigma_cls, rvr_boots)
            pc_y["ks_mean"] += float(ks_mean.detach())
            pc_y["ks_median"] += float(ks_med.detach())
            pc_y["w1_mean"] += float(w1_mean.detach())
            pc_y["w1_median"] += float(w1_med.detach())
            pc_y["w1_mean_trainaligned"] += float(w1_mean_trainaligned_cls_y.detach())
            pc_y["w1_median_trainaligned"] += float(w1_med_trainaligned_cls_y.detach())
            pc_y["mmd2_rvs"] += float(mmd2_rvs_cls.detach())
            pc_y["mmd2_rvr_med"] += float(mmd2_rvr_med_cls)
            pc_y["mmd2_ratio"] += float((mmd2_rvs_cls / (mmd2_rvr_med_cls + EPS)).detach())
            if yr_c.shape[1] >= 2:
                cr = _pearson_corr(yr_c)
                cc = _pearson_corr(yc_c)
                pe_abs, pe_rel = _fro_rel(cr, cc)
                crs = _spearman_corr(yr_c)
                ccs = _spearman_corr(yc_c)
                sp_abs, sp_rel = _fro_rel(crs, ccs)
                pc_y["pearson_fro"] += float(pe_abs.detach())
                pc_y["pearson_fro_rel"] += float(pe_rel.detach())
                pc_y["spearman_fro"] += float(sp_abs.detach())
                pc_y["spearman_fro_rel"] += float(sp_rel.detach())
                pc_y["_corr_count"] += 1.0


def compute_train_ref_vs_slice_real_gap(
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    c_train: torch.Tensor,
    x_slice: torch.Tensor,
    y_slice: torch.Tensor,
    c_slice: torch.Tensor,
    loss_like_kwargs: Mapping[str, Any],
    device: torch.device,
    seed: int | None = None,
) -> dict[str, Any]:
    classes = [int(ci) for ci in torch.unique(c_slice).tolist()]
    counts = {cls: int((c_slice == cls).sum().item()) for cls in classes}

    ref_x_by_cls = {cls: x_slice[c_slice == cls] for cls in classes}
    ref_y_by_cls = {cls: y_slice[c_slice == cls] for cls in classes}
    train_x_by_cls = {cls: x_train[c_train == cls] for cls in classes}
    train_y_by_cls = {cls: y_train[c_train == cls] for cls in classes}

    temporal_boots = max(1, _safe_int(loss_like_kwargs.get("realism_bootstrap"), 10))
    rvr_boots = max(1, _safe_int(loss_like_kwargs.get("realism_rvr_bootstrap"), temporal_boots))

    gen_cpu = torch.Generator(device="cpu")
    if seed is not None:
        gen_cpu.manual_seed(int(seed) + int(loss_like_kwargs.get("realism_seed_offset", 0)))

    overall_acc = _empty_suite(include_corr=False)
    x_acc = _empty_suite(include_corr=True)
    y_acc = _empty_suite(include_corr=True)
    per_class_acc = {
        cls: {
            "overall": _empty_suite(include_corr=False),
            "x": _empty_suite(include_corr=True),
            "y": _empty_suite(include_corr=True),
        }
        for cls in classes
    }

    for _rep in range(temporal_boots):
        cmp_x_by_cls: dict[int, torch.Tensor] = {}
        cmp_y_by_cls: dict[int, torch.Tensor] = {}
        for cls, n_slice in counts.items():
            x_cls = train_x_by_cls[cls]
            y_cls = train_y_by_cls[cls]
            n_train = int(x_cls.size(0))
            if n_train <= 0:
                raise ValueError(f"Train split has zero rows for class {cls}; cannot build temporal baseline.")
            if n_train >= n_slice:
                idx = torch.randperm(n_train, generator=gen_cpu, device="cpu")[:n_slice].to(device)
            else:
                idx = torch.randint(0, n_train, (n_slice,), generator=gen_cpu, device="cpu").to(device)
            cmp_x_by_cls[cls] = x_cls.index_select(0, idx)
            cmp_y_by_cls[cls] = y_cls.index_select(0, idx)

        _accumulate_real_vs_real_once(
            ref_x_by_cls=ref_x_by_cls,
            ref_y_by_cls=ref_y_by_cls,
            cmp_x_by_cls=cmp_x_by_cls,
            cmp_y_by_cls=cmp_y_by_cls,
            loss_like_kwargs=loss_like_kwargs,
            overall_acc=overall_acc,
            x_acc=x_acc,
            y_acc=y_acc,
            per_class_acc=per_class_acc,
            rvr_boots=rvr_boots,
            generator=gen_cpu,
            device=device,
        )

    realism = {
        "overall": _finalize_suite(overall_acc, boots=temporal_boots),
        "x": _finalize_suite(x_acc, boots=temporal_boots),
        "y": _finalize_suite(y_acc, boots=temporal_boots),
        "per_class": {},
    }
    for cls, suites in per_class_acc.items():
        realism["per_class"][int(cls)] = {
            "overall": _finalize_suite(suites["overall"], boots=temporal_boots),
            "x": _finalize_suite(suites["x"], boots=temporal_boots),
            "y": _finalize_suite(suites["y"], boots=temporal_boots),
        }
    return realism


def _metrics_rows_for_suite(
    *,
    run_id: str,
    family: str,
    slice_id: str,
    slice_label: str,
    slice_order: int,
    suite_name: str,
    slice_meta: Mapping[str, Any],
    suite_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component in ("overall", "x", "y"):
        component_payload = suite_payload.get(component) or {}
        for metric_name, metric_value in component_payload.items():
            rows.append(
                {
                    "run_id": run_id,
                    "split": "val",
                    "metric_group": "temporal_realism",
                    "family": family,
                    "slice_id": slice_id,
                    "slice_label": slice_label,
                    "slice_order": int(slice_order),
                    "suite": suite_name,
                    "metric_scope": "overall",
                    "component": component,
                    "class_id": None,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "n_obs": int(slice_meta["n_obs"]),
                    "start_date": slice_meta["start_date"],
                    "end_date": slice_meta["end_date"],
                    "start_frac": float(slice_meta["start_frac"]),
                    "end_frac": float(slice_meta["end_frac"]),
                }
            )
    per_class = suite_payload.get("per_class") or {}
    for cls_id, suites in per_class.items():
        for component in ("overall", "x", "y"):
            component_payload = (suites or {}).get(component) or {}
            for metric_name, metric_value in component_payload.items():
                rows.append(
                    {
                        "run_id": run_id,
                        "split": "val",
                        "metric_group": "temporal_realism",
                        "family": family,
                        "slice_id": slice_id,
                        "slice_label": slice_label,
                        "slice_order": int(slice_order),
                        "suite": suite_name,
                        "metric_scope": "per_class",
                        "component": component,
                        "class_id": _safe_int(cls_id),
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "n_obs": int(slice_meta["n_obs"]),
                        "start_date": slice_meta["start_date"],
                        "end_date": slice_meta["end_date"],
                        "start_frac": float(slice_meta["start_frac"]),
                        "end_frac": float(slice_meta["end_frac"]),
                    }
                )
    return rows


def build_temporal_realism_block(
    *,
    model,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    c_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    c_val: torch.Tensor,
    cxy_val: pd.DataFrame,
    r_val: pd.DataFrame,
    loss_like_kwargs: Mapping[str, Any],
    temporal_cfg: Mapping[str, Any],
    device: torch.device,
    seed: int | None,
    run_id: str,
    split_id: str,
    condition_col: str = "type",
) -> dict[str, Any]:
    order_table = build_temporal_order_table(r_val=r_val, cxy_val=cxy_val, condition_col=condition_col)
    family_specs = build_temporal_slice_assignments(order_table)

    temporal_loss_kwargs = dict(loss_like_kwargs or {})
    temporal_loss_kwargs["realism_bootstrap"] = int(temporal_cfg["bootstrap"])
    temporal_loss_kwargs["realism_rvr_bootstrap"] = int(temporal_cfg["rvr_bootstrap"])

    metrics_rows: list[dict[str, Any]] = []
    assignments_rows: list[dict[str, Any]] = []
    block: dict[str, Any] = {
        "schema_version": 1,
        "meta": {
            "target_split": "val",
            "split_id": str(split_id),
            "generated_reference": "full_val_realism_kept_unchanged",
            "ordering": {
                "primary_key": "date",
                "tiebreak_key": "source_row_number",
                "join_key": "post_cleaning_index",
                "fallback_key": "post_cleaning_index",
            },
            "baseline_policy": {
                "train_reference": str(temporal_cfg.get("baseline_policy", "class_count_matched_to_slice")),
                "bootstrap": int(temporal_cfg["bootstrap"]),
                "rvr_bootstrap": int(temporal_cfg["rvr_bootstrap"]),
            },
        },
        "quartiles": {"materialization": "full", "summary": {}, "slices": {}},
        "prefix_suffix": {"materialization": "full", "summary": {}, "slices": {}},
        "rolling_windows": {"materialization": "summary_only", "status": "not_enabled_in_v1", "summary": None},
        "artifacts": {},
    }

    for family_name in ("quartiles", "prefix_suffix"):
        family_slices: dict[str, Any] = {}
        for spec in family_specs.get(family_name, []):
            slice_meta = _slice_metadata(order_table, spec)
            rows = order_table.iloc[list(spec["row_indices"])].copy()
            tensor_idx = torch.as_tensor(rows["tensor_row_index"].to_numpy(dtype=np.int64), device=device)
            x_slice = x_val.index_select(0, tensor_idx)
            y_slice = y_val.index_select(0, tensor_idx)
            c_slice = c_val.index_select(0, tensor_idx)

            generated_vs_slice = compute_realism_metrics_for_set(
                model,
                x_ref=x_slice,
                y_ref=y_slice,
                c_ref=c_slice,
                loss_like_kwargs=dict(temporal_loss_kwargs),
                device=device,
                seed=seed,
            )
            train_ref_vs_slice_real = compute_train_ref_vs_slice_real_gap(
                x_train=x_train,
                y_train=y_train,
                c_train=c_train,
                x_slice=x_slice,
                y_slice=y_slice,
                c_slice=c_slice,
                loss_like_kwargs=dict(temporal_loss_kwargs),
                device=device,
                seed=seed,
            )

            slice_payload = {
                **slice_meta,
                "generated_vs_slice": generated_vs_slice,
                "train_ref_vs_slice_real": train_ref_vs_slice_real,
            }
            family_slices[str(spec["slice_id"])] = slice_payload

            for _, row in rows.iterrows():
                assignments_rows.append(
                    {
                        "run_id": run_id,
                        "split": "val",
                        "family": family_name,
                        "slice_id": str(spec["slice_id"]),
                        "slice_label": str(spec["label"]),
                        "slice_order": int(spec["order"]),
                        "post_cleaning_index": _safe_int(row["post_cleaning_index"]),
                        "tensor_row_index": _safe_int(row["tensor_row_index"]),
                        "class_id": _safe_int(row["class_id"]),
                        "date": _date_str(row.get("date_sort")),
                        "source_row_number": _safe_int(row.get("source_row_number"), default=-1),
                        "start_frac": float(spec["start_frac"]),
                        "end_frac": float(spec["end_frac"]),
                    }
                )

            metrics_rows.extend(
                _metrics_rows_for_suite(
                    run_id=run_id,
                    family=family_name,
                    slice_id=str(spec["slice_id"]),
                    slice_label=str(spec["label"]),
                    slice_order=int(spec["order"]),
                    suite_name="generated_vs_slice",
                    slice_meta=slice_meta,
                    suite_payload=generated_vs_slice,
                )
            )
            metrics_rows.extend(
                _metrics_rows_for_suite(
                    run_id=run_id,
                    family=family_name,
                    slice_id=str(spec["slice_id"]),
                    slice_label=str(spec["label"]),
                    slice_order=int(spec["order"]),
                    suite_name="train_ref_vs_slice_real",
                    slice_meta=slice_meta,
                    suite_payload=train_ref_vs_slice_real,
                )
            )

        block[family_name]["slices"] = family_slices
        block[family_name]["summary"] = _build_family_summary(family_name, family_slices)

    assignments_df = pd.DataFrame(assignments_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    manifest = {
        "schema_version": 1,
        "created_at": _utc_now_iso(),
        "run_id": run_id,
        "split_id": str(split_id),
        "target_split": "val",
        "ordering": block["meta"]["ordering"],
        "baseline_policy": block["meta"]["baseline_policy"],
        "families": list(temporal_cfg.get("families", ("quartiles", "prefix_suffix"))),
        "artifacts": {},
        "rows": {
            "slice_assignments": int(len(assignments_df)),
            "metrics_long": int(len(metrics_df)),
        },
    }
    return {
        "block": block,
        "slice_assignments": assignments_df,
        "metrics_long": metrics_df,
        "manifest": manifest,
    }


def write_temporal_realism_sidecars(
    *,
    out_dir: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    base_dir = Path(out_dir)
    root = base_dir / "temporal"
    root.mkdir(parents=True, exist_ok=True)

    assignments_path = root / "temporal_slice_assignments.csv"
    metrics_path = root / "temporal_metrics_long.csv"
    manifest_path = root / "temporal_manifest.json"

    assignments_df = payload.get("slice_assignments")
    metrics_df = payload.get("metrics_long")
    manifest = dict(payload.get("manifest") or {})

    if isinstance(assignments_df, pd.DataFrame):
        assignments_df.to_csv(assignments_path, index=False)
    if isinstance(metrics_df, pd.DataFrame):
        metrics_df.to_csv(metrics_path, index=False)

    manifest["artifacts"] = {
        "slice_assignments_relpath": str(assignments_path.relative_to(base_dir)),
        "metrics_long_relpath": str(metrics_path.relative_to(base_dir)),
        "manifest_relpath": str(manifest_path.relative_to(base_dir)),
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=_json_default)

    return manifest["artifacts"]
