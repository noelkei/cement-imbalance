from __future__ import annotations

from typing import Dict, Optional

import torch

from losses.flowgen_loss import (
    EPS,
    _fro_rel,
    _iqr,
    _ks_w1_matrix,
    _mmd_rbf_biased,
    _perdim_w1_normed,
    _pearson_corr,
    _pearson_xyblock_fro_gap,
    _softclip_asinh,
    _softspearman_xyblock_fro_gap,
    _spearman_corr,
)


def compute_realism_metrics_for_set(
    model,
    x_ref: torch.Tensor,
    y_ref: torch.Tensor,
    c_ref: torch.Tensor,
    *,
    loss_like_kwargs: dict,
    device: torch.device,
    seed: int | None = None,
) -> dict:
    """
    Full-set realism metrics using all reference data, averaged over bootstrap replicates.
    """

    def _finite_and_clamp_to_real(xr_c, yr_c, xs_c, ys_c):
        finite = torch.isfinite(xs_c).all(dim=1) & torch.isfinite(ys_c).all(dim=1)
        xs_c = xs_c[finite]
        ys_c = ys_c[finite]

        q25x = xr_c.quantile(0.25, dim=0)
        q75x = xr_c.quantile(0.75, dim=0)
        iqr_x = (q75x - q25x).clamp_min(1e-6)
        lo_x = q25x - 5.0 * iqr_x
        hi_x = q75x + 5.0 * iqr_x

        q25y = yr_c.quantile(0.25, dim=0)
        q75y = yr_c.quantile(0.75, dim=0)
        iqr_y = (q75y - q25y).clamp_min(1e-6)
        lo_y = q25y - 5.0 * iqr_y
        hi_y = q75y + 5.0 * iqr_y

        xs_c = xs_c.clamp(min=lo_x, max=hi_x)
        ys_c = ys_c.clamp(min=lo_y, max=hi_y)
        return xs_c, ys_c

    model.eval()
    gen_cpu = torch.Generator(device="cpu")
    if seed is not None:
        gen_cpu.manual_seed(int(seed) + int(loss_like_kwargs.get("realism_seed_offset", 0)))

    with torch.no_grad():
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

        dx = int(x_ref.shape[1])
        dy = int(y_ref.shape[1])
        dxy = dx + dy

        boots = int(loss_like_kwargs.get("realism_bootstrap", 10))
        rvr_boots = int(loss_like_kwargs.get("realism_rvr_bootstrap", boots))

        ks_grid_x = int(loss_like_kwargs.get("ks_grid_points_x", 64))
        ks_tau_x = float(loss_like_kwargs.get("ks_tau_x", 0.05))
        w1_norm_x = str(loss_like_kwargs.get("w1_x_norm", "iqr"))

        ks_grid_y = int(loss_like_kwargs.get("ks_grid_points_y", 64))
        ks_tau_y = float(loss_like_kwargs.get("ks_tau_y", 0.05))
        w1_norm_y = str(loss_like_kwargs.get("w1_y_norm", "iqr"))

        ks_grid_all = int(loss_like_kwargs.get("ks_grid_points_all", max(ks_grid_x, ks_grid_y)))
        ks_tau_all = float(loss_like_kwargs.get("ks_tau_all", 0.5 * (ks_tau_x + ks_tau_y)))
        w1_norm_all = str(loss_like_kwargs.get("w1_norm_all", "iqr"))

        classes = [int(ci) for ci in torch.unique(c_ref).tolist()]
        counts = {cls: int((c_ref == cls).sum().item()) for cls in classes}

        real_x_by_cls = {cls: x_ref[c_ref == cls] for cls in classes}
        real_y_by_cls = {cls: y_ref[c_ref == cls] for cls in classes}

        xr_all = torch.cat([real_x_by_cls[cls] for cls in classes], dim=0)
        yr_all = torch.cat([real_y_by_cls[cls] for cls in classes], dim=0)
        real_all_cat = torch.cat([xr_all, yr_all], dim=1)

        denom_all_global = _iqr(real_all_cat).to(torch.float32).clamp_min(1e-4)
        denom_x_global = _iqr(xr_all).to(torch.float32).clamp_min(1e-4)
        denom_y_global = _iqr(yr_all).to(torch.float32).clamp_min(1e-4)

        def _empty_suite(include_corr: bool) -> Dict[str, Optional[float]]:
            base = {
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
                "_xycorr_count": 0,
            }
            if include_corr:
                base.update(
                    {
                        "pearson_fro": 0.0,
                        "pearson_fro_rel": 0.0,
                        "spearman_fro": 0.0,
                        "spearman_fro_rel": 0.0,
                        "_corr_count": 0,
                    }
                )
            return base

        overall_acc = _empty_suite(include_corr=False)
        x_acc = _empty_suite(include_corr=True)
        y_acc = _empty_suite(include_corr=True)
        per_class_acc: Dict[int, Dict[str, Dict[str, Optional[float]]]] = {
            cls: {
                "overall": _empty_suite(include_corr=False),
                "x": _empty_suite(include_corr=True),
                "y": _empty_suite(include_corr=True),
            }
            for cls in counts.keys()
        }

        for _rep in range(boots):
            synth_x_by_cls: Dict[int, torch.Tensor] = {}
            synth_y_by_cls: Dict[int, torch.Tensor] = {}
            for cls, n_real in counts.items():
                ns = int(n_real)
                if ns <= 0:
                    synth_x_by_cls[cls] = real_x_by_cls[cls].new_zeros((0, dx))
                    synth_y_by_cls[cls] = real_y_by_cls[cls].new_zeros((0, dy))
                    continue
                z_s = torch.randn(ns, dxy, generator=gen_cpu, device="cpu").to(device)
                c_s = torch.full((ns,), cls, dtype=torch.long, device=device)
                (xs_c, ys_c), _ = model.inverse_xy(z_s, c_s)
                xs_c, ys_c = _finite_and_clamp_to_real(real_x_by_cls[cls], real_y_by_cls[cls], xs_c, ys_c)
                synth_x_by_cls[cls] = xs_c
                synth_y_by_cls[cls] = ys_c

            xs_all = torch.cat([synth_x_by_cls[cls] for cls in counts.keys()], dim=0)
            ys_all = torch.cat([synth_y_by_cls[cls] for cls in counts.keys()], dim=0)
            synth_all_cat = torch.cat([xs_all, ys_all], dim=1)

            def _mmd_rvr_median(real_mat: torch.Tensor, sigma: float, n_draws: int) -> float:
                n = real_mat.size(0)
                vals = []
                for _ in range(n_draws):
                    idx1 = torch.randint(0, n, (n,), generator=gen_cpu, device="cpu").to(device)
                    idx2 = torch.randint(0, n, (n,), generator=gen_cpu, device="cpu").to(device)
                    r1 = real_mat.index_select(0, idx1)
                    r2 = real_mat.index_select(0, idx2)
                    m2, _ = _mmd_rbf_biased(r1, r2, sigma=sigma)
                    vals.append(float(m2.detach()))
                return float(real_mat.new_tensor(vals).median().detach())

            ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                real_all_cat,
                synth_all_cat,
                grid_points=ks_grid_all,
                tau=ks_tau_all,
                norm=w1_norm_all,
                denom_override=denom_all_global,
            )
            w1x_all_perdim, _ = _perdim_w1_normed(xr_all, xs_all, norm=w1_norm_x, denom_override=denom_x_global)
            w1y_all_perdim, _ = _perdim_w1_normed(yr_all, ys_all, norm=w1_norm_y, denom_override=denom_y_global)
            w1_mean_trainaligned_all, w1_med_trainaligned_all = _aggregate_trainaligned_w1_xy(
                w1x_all_perdim,
                w1y_all_perdim,
            )
            mmd2_rvs_all, sigma_all = _mmd_rbf_biased(real_all_cat, synth_all_cat)
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

            ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                xr_all,
                xs_all,
                grid_points=ks_grid_x,
                tau=ks_tau_x,
                norm=w1_norm_x,
                denom_override=denom_x_global,
            )
            w1x_perdim, _ = _perdim_w1_normed(xr_all, xs_all, norm=w1_norm_x, denom_override=denom_x_global)
            w1_mean_trainaligned_x, w1_med_trainaligned_x = _aggregate_trainaligned_w1(
                w1x_perdim,
                softclip_s=float(loss_like_kwargs.get("w1_x_softclip_s", 0.0)),
                clip_perdim=float(loss_like_kwargs.get("w1_x_clip_perdim", 0.0)),
                agg_softcap=float(loss_like_kwargs.get("w1_x_agg_softcap", 0.0)),
            )
            mmd2_rvs_x, sigma_x = _mmd_rbf_biased(xr_all, xs_all)
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

            if dx >= 2:
                cr = _pearson_corr(xr_all)
                cs = _pearson_corr(xs_all)
                pe_abs, pe_rel = _fro_rel(cr, cs)
                crs = _spearman_corr(xr_all)
                css = _spearman_corr(xs_all)
                sp_abs, sp_rel = _fro_rel(crs, css)
                x_acc["pearson_fro"] += float(pe_abs.detach())
                x_acc["pearson_fro_rel"] += float(pe_rel.detach())
                x_acc["spearman_fro"] += float(sp_abs.detach())
                x_acc["spearman_fro_rel"] += float(sp_rel.detach())
                x_acc["_corr_count"] += 1

            ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                yr_all,
                ys_all,
                grid_points=ks_grid_y,
                tau=ks_tau_y,
                norm=w1_norm_y,
                denom_override=denom_y_global,
            )
            w1y_perdim, _ = _perdim_w1_normed(yr_all, ys_all, norm=w1_norm_y, denom_override=denom_y_global)
            w1_mean_trainaligned_y, w1_med_trainaligned_y = _aggregate_trainaligned_w1(
                w1y_perdim,
                softclip_s=float(loss_like_kwargs.get("w1_y_softclip_s", 0.0)),
                clip_perdim=float(loss_like_kwargs.get("w1_y_clip_perdim", 0.0)),
                agg_softcap=float(loss_like_kwargs.get("w1_y_agg_softcap", 0.0)),
            )
            mmd2_rvs_y, sigma_y = _mmd_rbf_biased(yr_all, ys_all)
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

            if dy >= 2:
                cr = _pearson_corr(yr_all)
                cs = _pearson_corr(ys_all)
                pe_abs, pe_rel = _fro_rel(cr, cs)
                crs = _spearman_corr(yr_all)
                css = _spearman_corr(ys_all)
                sp_abs, sp_rel = _fro_rel(crs, css)
                y_acc["pearson_fro"] += float(pe_abs.detach())
                y_acc["pearson_fro_rel"] += float(pe_rel.detach())
                y_acc["spearman_fro"] += float(sp_abs.detach())
                y_acc["spearman_fro_rel"] += float(sp_rel.detach())
                y_acc["_corr_count"] += 1

            pe_abs_xy, pe_rel_xy = _pearson_xyblock_fro_gap(
                model,
                xr_all,
                yr_all,
                xs_all,
                ys_all,
                relative=True,
            )
            sp_abs_xy, sp_rel_xy = _softspearman_xyblock_fro_gap(
                model,
                xr_all,
                yr_all,
                xs_all,
                ys_all,
                tau=float(loss_like_kwargs.get("corr_xy_tau", 0.05)),
                relative=True,
            )
            overall_acc["xy_pearson_fro"] += float(pe_abs_xy.detach())
            overall_acc["xy_pearson_fro_rel"] += float(pe_rel_xy.detach())
            overall_acc["xy_spearman_fro"] += float(sp_abs_xy.detach())
            overall_acc["xy_spearman_fro_rel"] += float(sp_rel_xy.detach())
            overall_acc["_xycorr_count"] += 1

            for cls in counts.keys():
                xr_c = real_x_by_cls[cls]
                yr_c = real_y_by_cls[cls]
                xs_c = synth_x_by_cls[cls]
                ys_c = synth_y_by_cls[cls]

                rc_all = torch.cat([xr_c, yr_c], dim=1)
                sc_all = torch.cat([xs_c, ys_c], dim=1)
                ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                    rc_all,
                    sc_all,
                    grid_points=ks_grid_all,
                    tau=ks_tau_all,
                    norm=w1_norm_all,
                    denom_override=denom_all_global,
                )
                w1x_perdim_cls, _ = _perdim_w1_normed(xr_c, xs_c, norm=w1_norm_x, denom_override=denom_x_global)
                w1y_perdim_cls, _ = _perdim_w1_normed(yr_c, ys_c, norm=w1_norm_y, denom_override=denom_y_global)
                w1_mean_trainaligned_cls_all, w1_med_trainaligned_cls_all = _aggregate_trainaligned_w1_xy(
                    w1x_perdim_cls,
                    w1y_perdim_cls,
                )
                mmd2_rvs, sigma = _mmd_rbf_biased(rc_all, sc_all)
                mmd2_rvr_med = _mmd_rvr_median(rc_all, sigma, rvr_boots)

                pc_over = per_class_acc[cls]["overall"]
                pc_over["ks_mean"] += float(ks_mean.detach())
                pc_over["ks_median"] += float(ks_med.detach())
                pc_over["w1_mean"] += float(w1_mean.detach())
                pc_over["w1_median"] += float(w1_med.detach())
                pc_over["w1_mean_trainaligned"] += float(w1_mean_trainaligned_cls_all.detach())
                pc_over["w1_median_trainaligned"] += float(w1_med_trainaligned_cls_all.detach())
                pc_over["mmd2_rvs"] += float(mmd2_rvs.detach())
                pc_over["mmd2_rvr_med"] += float(mmd2_rvr_med)
                pc_over["mmd2_ratio"] += float((mmd2_rvs / (mmd2_rvr_med + EPS)).detach())

                ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                    xr_c,
                    xs_c,
                    grid_points=ks_grid_x,
                    tau=ks_tau_x,
                    norm=w1_norm_x,
                    denom_override=denom_x_global,
                )
                w1x_perdim_cls_only, _ = _perdim_w1_normed(xr_c, xs_c, norm=w1_norm_x, denom_override=denom_x_global)
                w1_mean_trainaligned_cls_x, w1_med_trainaligned_cls_x = _aggregate_trainaligned_w1(
                    w1x_perdim_cls_only,
                    softclip_s=float(loss_like_kwargs.get("w1_x_softclip_s", 0.0)),
                    clip_perdim=float(loss_like_kwargs.get("w1_x_clip_perdim", 0.0)),
                    agg_softcap=float(loss_like_kwargs.get("w1_x_agg_softcap", 0.0)),
                )
                mmd2_rvs_x, sigma_x = _mmd_rbf_biased(xr_c, xs_c)
                mmd2_rvr_med_x = _mmd_rvr_median(xr_c, sigma_x, rvr_boots)
                pc_x = per_class_acc[cls]["x"]
                pc_x["ks_mean"] += float(ks_mean.detach())
                pc_x["ks_median"] += float(ks_med.detach())
                pc_x["w1_mean"] += float(w1_mean.detach())
                pc_x["w1_median"] += float(w1_med.detach())
                pc_x["w1_mean_trainaligned"] += float(w1_mean_trainaligned_cls_x.detach())
                pc_x["w1_median_trainaligned"] += float(w1_med_trainaligned_cls_x.detach())
                pc_x["mmd2_rvs"] += float(mmd2_rvs_x.detach())
                pc_x["mmd2_rvr_med"] += float(mmd2_rvr_med_x)
                pc_x["mmd2_ratio"] += float((mmd2_rvs_x / (mmd2_rvr_med_x + EPS)).detach())

                if xr_c.shape[1] >= 2:
                    cr = _pearson_corr(xr_c)
                    cs = _pearson_corr(xs_c)
                    pe_abs, pe_rel = _fro_rel(cr, cs)
                    crs = _spearman_corr(xr_c)
                    css = _spearman_corr(xs_c)
                    sp_abs, sp_rel = _fro_rel(crs, css)
                    pc_x["pearson_fro"] += float(pe_abs.detach())
                    pc_x["pearson_fro_rel"] += float(pe_rel.detach())
                    pc_x["spearman_fro"] += float(sp_abs.detach())
                    pc_x["spearman_fro_rel"] += float(sp_rel.detach())
                    pc_x["_corr_count"] += 1

                ks_mean, ks_med, w1_mean, w1_med = _ks_w1_matrix(
                    yr_c,
                    ys_c,
                    grid_points=ks_grid_y,
                    tau=ks_tau_y,
                    norm=w1_norm_y,
                    denom_override=denom_y_global,
                )
                w1y_perdim_cls_only, _ = _perdim_w1_normed(yr_c, ys_c, norm=w1_norm_y, denom_override=denom_y_global)
                w1_mean_trainaligned_cls_y, w1_med_trainaligned_cls_y = _aggregate_trainaligned_w1(
                    w1y_perdim_cls_only,
                    softclip_s=float(loss_like_kwargs.get("w1_y_softclip_s", 0.0)),
                    clip_perdim=float(loss_like_kwargs.get("w1_y_clip_perdim", 0.0)),
                    agg_softcap=float(loss_like_kwargs.get("w1_y_agg_softcap", 0.0)),
                )
                mmd2_rvs_y, sigma_y = _mmd_rbf_biased(yr_c, ys_c)
                mmd2_rvr_med_y = _mmd_rvr_median(yr_c, sigma_y, rvr_boots)
                pc_y = per_class_acc[cls]["y"]
                pc_y["ks_mean"] += float(ks_mean.detach())
                pc_y["ks_median"] += float(ks_med.detach())
                pc_y["w1_mean"] += float(w1_mean.detach())
                pc_y["w1_median"] += float(w1_med.detach())
                pc_y["w1_mean_trainaligned"] += float(w1_mean_trainaligned_cls_y.detach())
                pc_y["w1_median_trainaligned"] += float(w1_med_trainaligned_cls_y.detach())
                pc_y["mmd2_rvs"] += float(mmd2_rvs_y.detach())
                pc_y["mmd2_rvr_med"] += float(mmd2_rvr_med_y)
                pc_y["mmd2_ratio"] += float((mmd2_rvs_y / (mmd2_rvr_med_y + EPS)).detach())

                if yr_c.shape[1] >= 2:
                    cr = _pearson_corr(yr_c)
                    cs = _pearson_corr(ys_c)
                    pe_abs, pe_rel = _fro_rel(cr, cs)
                    crs = _spearman_corr(yr_c)
                    css = _spearman_corr(ys_c)
                    sp_abs, sp_rel = _fro_rel(crs, css)
                    pc_y["pearson_fro"] += float(pe_abs.detach())
                    pc_y["pearson_fro_rel"] += float(pe_rel.detach())
                    pc_y["spearman_fro"] += float(sp_abs.detach())
                    pc_y["spearman_fro_rel"] += float(sp_rel.detach())
                    pc_y["_corr_count"] += 1

                pe_abs_xy, pe_rel_xy = _pearson_xyblock_fro_gap(
                    model,
                    xr_c,
                    yr_c,
                    xs_c,
                    ys_c,
                    relative=True,
                )
                sp_abs_xy, sp_rel_xy = _softspearman_xyblock_fro_gap(
                    model,
                    xr_c,
                    yr_c,
                    xs_c,
                    ys_c,
                    tau=float(loss_like_kwargs.get("corr_xy_tau", 0.05)),
                    relative=True,
                )
                pc_over["xy_pearson_fro"] += float(pe_abs_xy.detach())
                pc_over["xy_pearson_fro_rel"] += float(pe_rel_xy.detach())
                pc_over["xy_spearman_fro"] += float(sp_abs_xy.detach())
                pc_over["xy_spearman_fro_rel"] += float(sp_rel_xy.detach())
                pc_over["_xycorr_count"] += 1

        def _finalize_suite(acc: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
            out = {
                "ks_mean": acc["ks_mean"] / boots,
                "ks_median": acc["ks_median"] / boots,
                "w1_mean": acc["w1_mean"] / boots,
                "w1_median": acc["w1_median"] / boots,
                "w1_mean_trainaligned": acc["w1_mean_trainaligned"] / boots,
                "w1_median_trainaligned": acc["w1_median_trainaligned"] / boots,
                "pearson_fro": None,
                "pearson_fro_rel": None,
                "spearman_fro": None,
                "spearman_fro_rel": None,
                "xy_pearson_fro": None,
                "xy_pearson_fro_rel": None,
                "xy_spearman_fro": None,
                "xy_spearman_fro_rel": None,
                "mmd2_rvs": acc["mmd2_rvs"] / boots,
                "mmd2_rvr_med": acc["mmd2_rvr_med"] / boots,
                "mmd2_ratio": acc["mmd2_ratio"] / boots,
            }
            if "_corr_count" in acc and acc["_corr_count"] > 0:
                k = float(acc["_corr_count"])
                out["pearson_fro"] = acc["pearson_fro"] / k
                out["pearson_fro_rel"] = acc["pearson_fro_rel"] / k
                out["spearman_fro"] = acc["spearman_fro"] / k
                out["spearman_fro_rel"] = acc["spearman_fro_rel"] / k
            if "_xycorr_count" in acc and acc["_xycorr_count"] > 0:
                kxy = float(acc["_xycorr_count"])
                out["xy_pearson_fro"] = acc["xy_pearson_fro"] / kxy
                out["xy_pearson_fro_rel"] = acc["xy_pearson_fro_rel"] / kxy
                out["xy_spearman_fro"] = acc["xy_spearman_fro"] / kxy
                out["xy_spearman_fro_rel"] = acc["xy_spearman_fro_rel"] / kxy
            return {key: (None if value is None else float(value)) for key, value in out.items()}

        realism = {
            "overall": _finalize_suite(overall_acc),
            "x": _finalize_suite(x_acc),
            "y": _finalize_suite(y_acc),
            "per_class": {},
        }
        for cls, suites in per_class_acc.items():
            realism["per_class"][int(cls)] = {
                "overall": _finalize_suite(suites["overall"]),
                "x": _finalize_suite(suites["x"]),
                "y": _finalize_suite(suites["y"]),
            }
        return realism
