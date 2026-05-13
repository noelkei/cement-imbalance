from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


FLOWPRE_BRANCHES = ("rrmse", "mvn", "fair")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def robust_z(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if pd.isna(mad) or mad <= 1e-12:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return 0.6745 * (series - median) / mad


def summarize_flowpre_results(
    results_path: str | Path,
    *,
    branch_id: str,
    run_id: str,
    cfg_id: str,
    phase: str,
    seed: int,
) -> dict[str, Any]:
    payload = _load_yaml(results_path)
    train = dict(payload.get("train") or {})
    val = dict(payload.get("val") or {})
    val_stats = dict(val.get("isotropy_stats") or {})
    train_stats = dict(train.get("isotropy_stats") or {})

    val_pc = val.get("per_class_iso_rrmse") or {}
    per_class_rows = []
    for cls_id, metrics in dict(val_pc).items():
        row = dict(metrics or {})
        row["class_id"] = int(cls_id)
        per_class_rows.append(row)
    per_class_df = pd.DataFrame(per_class_rows)

    val_rrmse_mean = float(val.get("rrmse_mean_whole", math.inf))
    val_rrmse_std = float(val.get("rrmse_std_whole", math.inf))
    train_rrmse_mean = float(train.get("rrmse_mean_whole", math.inf))
    train_rrmse_std = float(train.get("rrmse_std_whole", math.inf))

    row = {
        "branch_id": branch_id,
        "run_id": run_id,
        "cfg_id": cfg_id,
        "phase": phase,
        "seed": int(seed),
        "best_epoch": payload.get("best_epoch"),
        "total_epochs": payload.get("total_epochs"),
        "train_rrmse_mean": train_rrmse_mean,
        "train_rrmse_std": train_rrmse_std,
        "val_rrmse_mean": val_rrmse_mean,
        "val_rrmse_std": val_rrmse_std,
        "gap_val_train_mean": abs(val_rrmse_mean - train_rrmse_mean),
        "gap_val_train_std": abs(val_rrmse_std - train_rrmse_std),
        "gap_val_train_sum": abs(val_rrmse_mean - train_rrmse_mean) + abs(val_rrmse_std - train_rrmse_std),
        "train_rrmse_recon": float(train.get("rrmse_recon", math.inf)),
        "val_rrmse_recon": float(val.get("rrmse_recon", math.inf)),
        "train_r2_recon": float(train.get("r2_recon", -math.inf)),
        "val_r2_recon": float(val.get("r2_recon", -math.inf)),
        "train_skew_abs": abs(float(train_stats.get("skewness_mean", math.inf))),
        "val_skew_abs": abs(float(val_stats.get("skewness_mean", math.inf))),
        "train_kurt_excess_abs": abs(float(train_stats.get("kurtosis_mean", math.inf)) - 3.0),
        "val_kurt_excess_abs": abs(float(val_stats.get("kurtosis_mean", math.inf)) - 3.0),
        "train_mahal_mu": float(train_stats.get("mahalanobis_mean", math.inf)),
        "val_mahal_mu": float(val_stats.get("mahalanobis_mean", math.inf)),
        "train_mahal_md": float(train_stats.get("mahalanobis_median", math.inf)),
        "val_mahal_md": float(val_stats.get("mahalanobis_median", math.inf)),
        "train_eigstd": float(train_stats.get("eigval_std", math.inf)),
        "val_eigstd": float(val_stats.get("eigval_std", math.inf)),
        "val_pc_worst_mean": math.inf,
        "val_pc_worst_std": math.inf,
        "val_pc_wavg_mean": math.inf,
    }

    if not per_class_df.empty:
        row["val_pc_worst_mean"] = float(per_class_df["rrmse_mean"].max())
        row["val_pc_worst_std"] = float(per_class_df["rrmse_std"].max())
        weights = per_class_df["n"].clip(lower=1).astype(float)
        row["val_pc_wavg_mean"] = float(np.average(per_class_df["rrmse_mean"], weights=weights))
    return row


def rank_flowpre_branch(df: pd.DataFrame, branch_id: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ranked = df.copy()
    ranked["historical_support_only"] = False
    ranked["selection_score"] = np.nan

    if branch_id == "rrmse":
        ranked = ranked.sort_values(
            ["val_rrmse_mean", "val_rrmse_std", "gap_val_train_sum", "val_rrmse_recon", "val_eigstd"],
            ascending=[True, True, True, True, True],
            na_position="last",
        )
    elif branch_id == "mvn":
        target_mu = math.sqrt(43 - 0.5)
        iso_dev = pd.DataFrame(
            {
                "skew": ranked["val_skew_abs"],
                "kurt": ranked["val_kurt_excess_abs"],
                "eig": ranked["val_eigstd"],
                "mu": (ranked["val_mahal_mu"] - target_mu).abs(),
                "md": (ranked["val_mahal_md"] - target_mu).abs(),
            }
        )
        iso_weights = pd.Series({"skew": 1.0, "kurt": 1.0, "eig": 1.2, "mu": 0.3, "md": 0.3})
        ranked["selection_score"] = (iso_dev * iso_weights).sum(axis=1) / float(iso_weights.sum())
        ranked["historical_support_only"] = True
        ranked = ranked.sort_values(
            ["selection_score", "val_eigstd", "val_kurt_excess_abs", "val_skew_abs", "gap_val_train_sum"],
            ascending=[True, True, True, True, True],
            na_position="last",
        )
    elif branch_id == "fair":
        fair_terms = pd.DataFrame(
            {
                "pc_worst_mean": ranked["val_pc_worst_mean"],
                "pc_worst_std": ranked["val_pc_worst_std"],
                "pc_wavg_mean": ranked["val_pc_wavg_mean"],
            }
        )
        fair_z = fair_terms.apply(robust_z, axis=0)
        ranked["selection_score"] = fair_z.mean(axis=1)
        ranked["historical_support_only"] = True
        ranked = ranked.sort_values(
            ["selection_score", "val_pc_worst_mean", "val_pc_worst_std", "val_rrmse_mean", "val_eigstd"],
            ascending=[True, True, True, True, True],
            na_position="last",
        )
    else:
        raise ValueError(f"Unsupported FlowPre branch_id: {branch_id}")

    ranked["branch_rank"] = range(1, len(ranked) + 1)
    return ranked.reset_index(drop=True)


def summarize_flowgen_results(
    results_path: str | Path,
    *,
    run_id: str,
    cfg_id: str,
    phase: str,
    seed: int,
) -> dict[str, Any]:
    payload = _load_yaml(results_path)
    val = dict(payload.get("val") or {})
    realism = dict(val.get("realism") or {})
    overall = dict(realism.get("overall") or {})
    per_class = dict(realism.get("per_class") or {})

    worst_w1 = -math.inf
    worst_ks = -math.inf
    for suites in per_class.values():
        overall_cls = dict((suites or {}).get("overall") or {})
        worst_w1 = max(worst_w1, float(overall_cls.get("w1_mean", -math.inf)))
        worst_ks = max(worst_ks, float(overall_cls.get("ks_mean", -math.inf)))

    val_stats = dict(val.get("isotropy_stats") or {})
    return {
        "run_id": run_id,
        "cfg_id": cfg_id,
        "phase": phase,
        "seed": int(seed),
        "phase1_best_epoch": (payload.get("phase1") or {}).get("best_epoch"),
        "finetune_best_epoch": (payload.get("finetune") or {}).get("best_epoch"),
        "val_realism_w1_mean": float(overall.get("w1_mean", math.inf)),
        "val_realism_ks_mean": float(overall.get("ks_mean", math.inf)),
        "val_realism_mmd2_rvs": float(overall.get("mmd2_rvs", math.inf)),
        "val_realism_worst_class_w1_mean": float(worst_w1),
        "val_realism_worst_class_ks_mean": float(worst_ks),
        "val_loss_rrmse_x_mean_whole": float(val.get("loss_rrmse_x_mean_whole", math.inf)),
        "val_loss_rrmse_y_mean_whole": float(val.get("loss_rrmse_y_mean_whole", math.inf)),
        "val_rrmse_x_recon": float(val.get("rrmse_x_recon", math.inf)),
        "val_rrmse_y_recon": float(val.get("rrmse_y_recon", math.inf)),
        "val_eigstd": float(val_stats.get("eigval_std", math.inf)),
    }


def rank_flowgen(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ranked = df.sort_values(
        [
            "val_realism_w1_mean",
            "val_realism_ks_mean",
            "val_realism_mmd2_rvs",
            "val_realism_worst_class_w1_mean",
            "val_realism_worst_class_ks_mean",
            "val_loss_rrmse_x_mean_whole",
            "val_loss_rrmse_y_mean_whole",
            "val_eigstd",
            "val_rrmse_x_recon",
            "val_rrmse_y_recon",
        ],
        ascending=[True, True, True, True, True, True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["flowgen_rank"] = range(1, len(ranked) + 1)
    ranked["historical_support_only"] = False
    return ranked
