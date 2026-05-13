from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.f6_selection import FLOWPRE_BRANCHES, rank_flowpre_branch, robust_z
from scripts.f6_flowpre_explore_v2 import _discover_existing_runs


REPORT_ROOT = ROOT / "outputs" / "reports" / "f6_explore_v2_results"
EXPLORE_CONTRACT_ID = "f6_flowpre_explore_v2"
REVALIDATE_CONTRACT_ID = "f6_flowpre_revalidation_v1"
MVN_TARGET_MU = math.sqrt(43 - 0.5)
MVN_WEIGHT_SKEW = 1.0
MVN_WEIGHT_KURT = 1.0
MVN_WEIGHT_EIG = 1.2
MVN_WEIGHT_MU = 0.3
MVN_WEIGHT_MD = 0.3
MVN_WEIGHT_TOTAL = MVN_WEIGHT_SKEW + MVN_WEIGHT_KURT + MVN_WEIGHT_EIG + MVN_WEIGHT_MU + MVN_WEIGHT_MD
BALANCED_RRMSE_PENALTY_WEIGHT = 0.25
TOP_N = 10
LEGACY_TOP_N = 3
BALANCE_RATIO_EPS = 1e-12
FLOWGENCANDIDATE_PRIORFIT_WEIGHTS = {"train": 0.50, "val": 0.30, "gap": 0.20}
FLOWGENCANDIDATE_ROBUST_WEIGHTS = {"train": 0.25, "val": 0.35, "gap": 0.40}
FLOWGENCANDIDATE_HYBRID_WEIGHTS = {"train": 0.35, "val": 0.35, "gap": 0.30}
FLOWGENCANDIDATE_VAL_SURFACE_WEIGHTS = {"mvn": 0.40, "sum": 0.30, "recon": 0.20, "balance": 0.10}
FLOWGENCANDIDATE_VAL_BALANCE_RZ_CLIP = 2.5
FLOWGENCANDIDATE_ELIGIBILITY_ORDER = {"eligible": 0, "caution": 1, "reject": 2}
GLOBAL_RECONSTRUCTION_ORDER = {"pass": 0, "caution": 1, "fail": 2}
FLOWGENCANDIDATE_RELATIVE_POOL_MIN = 5
GLOBAL_RECONSTRUCTION_FAIL_THRESHOLDS = {
    "train_rrmse_recon": 0.25,
    "val_rrmse_recon": 5e-5,
    "recon_gap_rrmse": 0.25,
}
GLOBAL_RECONSTRUCTION_CAUTION_THRESHOLDS = {
    "train_rrmse_recon": 0.05,
    "val_rrmse_recon": 1e-5,
    "recon_gap_rrmse": 0.05,
}
FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS = {
    "train_rrmse_recon": 0.15,
    "val_rrmse_recon": GLOBAL_RECONSTRUCTION_FAIL_THRESHOLDS["val_rrmse_recon"],
    "train_sum_rrmse": 0.25,
    "val_sum_rrmse": 0.60,
    "train_mvn_score": 15.0,
    "val_mvn_score": 3.5,
    "gap_val_train_sum": 0.75,
    "mvn_gap_abs": 10.0,
    "recon_gap_rrmse": 0.15,
}
FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS = {
    "train_rrmse_recon": GLOBAL_RECONSTRUCTION_CAUTION_THRESHOLDS["train_rrmse_recon"],
    "val_rrmse_recon": GLOBAL_RECONSTRUCTION_CAUTION_THRESHOLDS["val_rrmse_recon"],
    "train_sum_rrmse": 0.15,
    "val_sum_rrmse": 0.55,
    "train_mvn_score": 7.5,
    "val_mvn_score": 2.5,
    "gap_val_train_sum": 0.50,
    "mvn_gap_abs": 4.0,
    "recon_gap_rrmse": 0.05,
}
RRMSE_PRIMARY_WEIGHTS = {"train": 0.35, "val": 0.35, "gap": 0.15, "train_balance": 0.05, "val_balance": 0.10}

LENS_DISPLAY_NAMES = {
    "rrmse_primary": "rrmse_primary",
    "mvn": "mvn",
    "fair": "fair",
    "flowgencandidate_priorfit": "flowgencandidate_priorfit",
    "flowgencandidate_robust": "flowgencandidate_robust",
    "flowgencandidate_hybrid": "flowgencandidate_hybrid",
}


def _campaign_origin(row: pd.Series) -> str:
    contract_id = str(row.get("contract_id") or "")
    run_id = str(row.get("run_id") or "")
    if contract_id == EXPLORE_CONTRACT_ID or run_id.startswith("flowprex2_"):
        return "explore_v2"
    if contract_id == REVALIDATE_CONTRACT_ID or run_id.startswith("flowpre_"):
        return "revalidate_v1"
    return "other"


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_no rows_"
    cols = list(frame.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = [
        "| " + " | ".join(str(row[col]) for col in cols) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, sep, *body])


def _robust_z_against_reference(series: pd.Series, reference: pd.Series) -> pd.Series:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    values = pd.to_numeric(series, errors="coerce")
    if ref.empty:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    median = ref.median()
    mad = (ref - median).abs().median()
    if pd.isna(mad) or mad <= 1e-12:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return 0.6745 * (values - median) / mad


def _apply_status_layer(
    out: pd.DataFrame,
    *,
    fail_reason_defs: list[tuple[str, pd.Series, str]],
    caution_reason_defs: list[tuple[str, pd.Series, str]],
    status_col: str,
    rank_col: str,
    reasons_col: str,
    fail_reason_count_col: str,
    caution_reason_count_col: str,
    order_map: dict[str, int],
    default_status: str,
    caution_status: str,
    fail_status: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    for col_name, mask, _ in [*fail_reason_defs, *caution_reason_defs]:
        out[col_name] = pd.Series(mask, index=out.index).fillna(False).astype(bool)

    fail_reason_cols = [col_name for col_name, _, _ in fail_reason_defs]
    caution_reason_cols = [col_name for col_name, _, _ in caution_reason_defs]

    fail_mask = out[fail_reason_cols].any(axis=1) if fail_reason_cols else pd.Series(False, index=out.index)
    caution_mask = (
        (~fail_mask) & out[caution_reason_cols].any(axis=1)
        if caution_reason_cols
        else pd.Series(False, index=out.index)
    )

    out[fail_reason_count_col] = out[fail_reason_cols].astype(int).sum(axis=1) if fail_reason_cols else 0
    out[caution_reason_count_col] = (
        out[caution_reason_cols].astype(int).sum(axis=1) if caution_reason_cols else 0
    )

    reasons: list[str] = []
    for idx, row in out.iterrows():
        labels: list[str] = []
        if bool(fail_mask.loc[idx]):
            for col_name, _, label in fail_reason_defs:
                if bool(row[col_name]):
                    labels.append(label)
        elif bool(caution_mask.loc[idx]):
            for col_name, _, label in caution_reason_defs:
                if bool(row[col_name]):
                    labels.append(label)
        reasons.append("; ".join(labels) if labels else "none")

    out[reasons_col] = reasons
    out[status_col] = default_status
    out.loc[caution_mask, status_col] = caution_status
    out.loc[fail_mask, status_col] = fail_status
    out[rank_col] = out[status_col].map(order_map).fillna(99).astype(int)
    return out, fail_mask, caution_mask


def _build_inventory() -> pd.DataFrame:
    inventory = _discover_existing_runs()
    if inventory.empty:
        return inventory

    inventory = inventory.copy()
    inventory["campaign_origin"] = inventory.apply(_campaign_origin, axis=1)
    inventory["is_new_campaign"] = inventory["campaign_origin"].eq("explore_v2")
    inventory["is_complete_artifacts"] = (
        inventory["has_results"].fillna(False)
        & inventory["has_metrics_long"].fillna(False)
        & inventory["has_config"].fillna(False)
        & inventory["is_complete_core"].fillna(False)
    )
    inventory["ordering_scope"] = "all_official_flowpre_runs"
    return inventory


def _augment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["train_sum_rrmse"] = out["train_rrmse_mean"] + out["train_rrmse_std"]
    out["val_sum_rrmse"] = out["val_rrmse_mean"] + out["val_rrmse_std"]
    out["both_sum_rrmse"] = out["train_sum_rrmse"] + out["val_sum_rrmse"]
    out["train_balance_gap_rrmse"] = (out["train_rrmse_mean"] - out["train_rrmse_std"]).abs()
    out["val_balance_gap_rrmse"] = (out["val_rrmse_mean"] - out["val_rrmse_std"]).abs()
    out["balance_gap_rrmse"] = out["train_balance_gap_rrmse"] + out["val_balance_gap_rrmse"]
    out["train_rrmse_balance_ratio"] = out["train_balance_gap_rrmse"] / (out["train_sum_rrmse"] + BALANCE_RATIO_EPS)
    out["val_rrmse_balance_ratio"] = out["val_balance_gap_rrmse"] / (out["val_sum_rrmse"] + BALANCE_RATIO_EPS)
    out["balanced_rrmse_penalty"] = BALANCED_RRMSE_PENALTY_WEIGHT * out["balance_gap_rrmse"]
    out["balanced_rrmse_score"] = out["both_sum_rrmse"] + out["balanced_rrmse_penalty"]
    out["rrmse_primary_score"] = (
        RRMSE_PRIMARY_WEIGHTS["train"] * out["train_sum_rrmse"]
        + RRMSE_PRIMARY_WEIGHTS["val"] * out["val_sum_rrmse"]
        + RRMSE_PRIMARY_WEIGHTS["gap"] * out["gap_val_train_sum"]
        + RRMSE_PRIMARY_WEIGHTS["train_balance"] * out["train_rrmse_balance_ratio"]
        + RRMSE_PRIMARY_WEIGHTS["val_balance"] * out["val_rrmse_balance_ratio"]
    )

    out["mvn_target_mu"] = MVN_TARGET_MU
    out["mvn_mu_abs_dev"] = (out["val_mahal_mu"] - MVN_TARGET_MU).abs()
    out["mvn_md_abs_dev"] = (out["val_mahal_md"] - MVN_TARGET_MU).abs()
    out["mvn_weighted_skew"] = MVN_WEIGHT_SKEW * out["val_skew_abs"]
    out["mvn_weighted_kurt"] = MVN_WEIGHT_KURT * out["val_kurt_excess_abs"]
    out["mvn_weighted_eig"] = MVN_WEIGHT_EIG * out["val_eigstd"]
    out["mvn_weighted_mu"] = MVN_WEIGHT_MU * out["mvn_mu_abs_dev"]
    out["mvn_weighted_md"] = MVN_WEIGHT_MD * out["mvn_md_abs_dev"]
    out["mvn_weight_total"] = MVN_WEIGHT_TOTAL
    out["mvn_selection_score_manual"] = (
        out["mvn_weighted_skew"]
        + out["mvn_weighted_kurt"]
        + out["mvn_weighted_eig"]
        + out["mvn_weighted_mu"]
        + out["mvn_weighted_md"]
    ) / MVN_WEIGHT_TOTAL

    out["train_mvn_mu_abs_dev"] = (out["train_mahal_mu"] - MVN_TARGET_MU).abs()
    out["train_mvn_md_abs_dev"] = (out["train_mahal_md"] - MVN_TARGET_MU).abs()
    out["train_mvn_weighted_skew"] = MVN_WEIGHT_SKEW * out["train_skew_abs"]
    out["train_mvn_weighted_kurt"] = MVN_WEIGHT_KURT * out["train_kurt_excess_abs"]
    out["train_mvn_weighted_eig"] = MVN_WEIGHT_EIG * out["train_eigstd"]
    out["train_mvn_weighted_mu"] = MVN_WEIGHT_MU * out["train_mvn_mu_abs_dev"]
    out["train_mvn_weighted_md"] = MVN_WEIGHT_MD * out["train_mvn_md_abs_dev"]
    out["train_mvn_score"] = (
        out["train_mvn_weighted_skew"]
        + out["train_mvn_weighted_kurt"]
        + out["train_mvn_weighted_eig"]
        + out["train_mvn_weighted_mu"]
        + out["train_mvn_weighted_md"]
    ) / MVN_WEIGHT_TOTAL
    out["val_mvn_score"] = out["mvn_selection_score_manual"]

    out["fair_z_pc_worst_mean"] = robust_z(out["val_pc_worst_mean"])
    out["fair_z_pc_worst_std"] = robust_z(out["val_pc_worst_std"])
    out["fair_z_pc_wavg_mean"] = robust_z(out["val_pc_wavg_mean"])
    out["fair_selection_score_manual"] = (
        out["fair_z_pc_worst_mean"] + out["fair_z_pc_worst_std"] + out["fair_z_pc_wavg_mean"]
    ) / 3.0

    out["recon_gap_rrmse"] = (out["train_rrmse_recon"] - out["val_rrmse_recon"]).abs()
    out["mvn_gap_abs"] = (out["train_mvn_score"] - out["val_mvn_score"]).abs()

    global_recon_frame = out[["train_rrmse_recon", "val_rrmse_recon", "recon_gap_rrmse"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite_global_recon = pd.Series(
        np.isfinite(global_recon_frame.to_numpy(dtype=float)).all(axis=1),
        index=global_recon_frame.index,
    )
    global_fail_reason_defs = [
        ("global_fail_nonfinite_reconstruction", ~finite_global_recon, "nonfinite_reconstruction_metric"),
        (
            "global_fail_train_rrmse_recon",
            out["train_rrmse_recon"] > GLOBAL_RECONSTRUCTION_FAIL_THRESHOLDS["train_rrmse_recon"],
            f"train_rrmse_recon>{GLOBAL_RECONSTRUCTION_FAIL_THRESHOLDS['train_rrmse_recon']}",
        ),
        (
            "global_fail_val_rrmse_recon",
            out["val_rrmse_recon"] > GLOBAL_RECONSTRUCTION_FAIL_THRESHOLDS["val_rrmse_recon"],
            f"val_rrmse_recon>{GLOBAL_RECONSTRUCTION_FAIL_THRESHOLDS['val_rrmse_recon']}",
        ),
        (
            "global_fail_recon_gap_rrmse",
            out["recon_gap_rrmse"] > GLOBAL_RECONSTRUCTION_FAIL_THRESHOLDS["recon_gap_rrmse"],
            f"recon_gap_rrmse>{GLOBAL_RECONSTRUCTION_FAIL_THRESHOLDS['recon_gap_rrmse']}",
        ),
    ]
    global_caution_reason_defs = [
        (
            "global_caution_train_rrmse_recon",
            out["train_rrmse_recon"] > GLOBAL_RECONSTRUCTION_CAUTION_THRESHOLDS["train_rrmse_recon"],
            f"train_rrmse_recon>{GLOBAL_RECONSTRUCTION_CAUTION_THRESHOLDS['train_rrmse_recon']}",
        ),
        (
            "global_caution_val_rrmse_recon",
            out["val_rrmse_recon"] > GLOBAL_RECONSTRUCTION_CAUTION_THRESHOLDS["val_rrmse_recon"],
            f"val_rrmse_recon>{GLOBAL_RECONSTRUCTION_CAUTION_THRESHOLDS['val_rrmse_recon']}",
        ),
        (
            "global_caution_recon_gap_rrmse",
            out["recon_gap_rrmse"] > GLOBAL_RECONSTRUCTION_CAUTION_THRESHOLDS["recon_gap_rrmse"],
            f"recon_gap_rrmse>{GLOBAL_RECONSTRUCTION_CAUTION_THRESHOLDS['recon_gap_rrmse']}",
        ),
    ]
    out, global_fail_mask, _ = _apply_status_layer(
        out,
        fail_reason_defs=global_fail_reason_defs,
        caution_reason_defs=global_caution_reason_defs,
        status_col="global_reconstruction_status",
        rank_col="global_reconstruction_rank",
        reasons_col="global_reconstruction_reasons",
        fail_reason_count_col="global_reconstruction_fail_reason_count",
        caution_reason_count_col="global_reconstruction_caution_reason_count",
        order_map=GLOBAL_RECONSTRUCTION_ORDER,
        default_status="pass",
        caution_status="caution",
        fail_status="fail",
    )

    critical_cols = [
        "train_mvn_score",
        "val_mvn_score",
        "train_sum_rrmse",
        "val_sum_rrmse",
        "train_rrmse_recon",
        "val_rrmse_recon",
        "gap_val_train_sum",
        "mvn_gap_abs",
        "recon_gap_rrmse",
    ]
    critical_frame = out[critical_cols].apply(pd.to_numeric, errors="coerce")
    finite_critical = pd.Series(
        np.isfinite(critical_frame.to_numpy(dtype=float)).all(axis=1),
        index=critical_frame.index,
    )

    reject_reason_defs = [
        ("raw_reject_nonfinite_critical", ~finite_critical, "nonfinite_critical_metric"),
        (
            "raw_reject_global_reconstruction_fail",
            global_fail_mask,
            "global_reconstruction_fail",
        ),
        (
            "raw_reject_train_rrmse_recon",
            out["train_rrmse_recon"] > FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS["train_rrmse_recon"],
            f"train_rrmse_recon>{FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS['train_rrmse_recon']}",
        ),
        (
            "raw_reject_val_rrmse_recon",
            out["val_rrmse_recon"] > FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS["val_rrmse_recon"],
            f"val_rrmse_recon>{FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS['val_rrmse_recon']}",
        ),
        (
            "raw_reject_train_sum_rrmse",
            out["train_sum_rrmse"] > FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS["train_sum_rrmse"],
            f"train_sum_rrmse>{FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS['train_sum_rrmse']}",
        ),
        (
            "raw_reject_val_sum_rrmse",
            out["val_sum_rrmse"] > FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS["val_sum_rrmse"],
            f"val_sum_rrmse>{FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS['val_sum_rrmse']}",
        ),
        (
            "raw_reject_train_mvn_score",
            out["train_mvn_score"] > FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS["train_mvn_score"],
            f"train_mvn_score>{FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS['train_mvn_score']}",
        ),
        (
            "raw_reject_val_mvn_score",
            out["val_mvn_score"] > FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS["val_mvn_score"],
            f"val_mvn_score>{FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS['val_mvn_score']}",
        ),
        (
            "raw_reject_gap_val_train_sum",
            out["gap_val_train_sum"] > FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS["gap_val_train_sum"],
            f"gap_val_train_sum>{FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS['gap_val_train_sum']}",
        ),
        (
            "raw_reject_mvn_gap_abs",
            out["mvn_gap_abs"] > FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS["mvn_gap_abs"],
            f"mvn_gap_abs>{FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS['mvn_gap_abs']}",
        ),
        (
            "raw_reject_recon_gap_rrmse",
            out["recon_gap_rrmse"] > FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS["recon_gap_rrmse"],
            f"recon_gap_rrmse>{FLOWGENCANDIDATE_RAW_REJECT_THRESHOLDS['recon_gap_rrmse']}",
        ),
    ]
    caution_reason_defs = [
        (
            "raw_caution_train_rrmse_recon",
            out["train_rrmse_recon"] > FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS["train_rrmse_recon"],
            f"train_rrmse_recon>{FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS['train_rrmse_recon']}",
        ),
        (
            "raw_caution_val_rrmse_recon",
            out["val_rrmse_recon"] > FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS["val_rrmse_recon"],
            f"val_rrmse_recon>{FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS['val_rrmse_recon']}",
        ),
        (
            "raw_caution_train_sum_rrmse",
            out["train_sum_rrmse"] > FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS["train_sum_rrmse"],
            f"train_sum_rrmse>{FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS['train_sum_rrmse']}",
        ),
        (
            "raw_caution_val_sum_rrmse",
            out["val_sum_rrmse"] > FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS["val_sum_rrmse"],
            f"val_sum_rrmse>{FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS['val_sum_rrmse']}",
        ),
        (
            "raw_caution_train_mvn_score",
            out["train_mvn_score"] > FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS["train_mvn_score"],
            f"train_mvn_score>{FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS['train_mvn_score']}",
        ),
        (
            "raw_caution_val_mvn_score",
            out["val_mvn_score"] > FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS["val_mvn_score"],
            f"val_mvn_score>{FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS['val_mvn_score']}",
        ),
        (
            "raw_caution_gap_val_train_sum",
            out["gap_val_train_sum"] > FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS["gap_val_train_sum"],
            f"gap_val_train_sum>{FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS['gap_val_train_sum']}",
        ),
        (
            "raw_caution_mvn_gap_abs",
            out["mvn_gap_abs"] > FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS["mvn_gap_abs"],
            f"mvn_gap_abs>{FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS['mvn_gap_abs']}",
        ),
        (
            "raw_caution_recon_gap_rrmse",
            out["recon_gap_rrmse"] > FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS["recon_gap_rrmse"],
            f"recon_gap_rrmse>{FLOWGENCANDIDATE_RAW_CAUTION_THRESHOLDS['recon_gap_rrmse']}",
        ),
    ]

    out, _, _ = _apply_status_layer(
        out,
        fail_reason_defs=reject_reason_defs,
        caution_reason_defs=caution_reason_defs,
        status_col="raw_acceptability_status",
        rank_col="raw_acceptability_rank",
        reasons_col="raw_acceptability_reasons",
        fail_reason_count_col="raw_reject_reason_count",
        caution_reason_count_col="raw_caution_reason_count",
        order_map=FLOWGENCANDIDATE_ELIGIBILITY_ORDER,
        default_status="eligible",
        caution_status="caution",
        fail_status="reject",
    )

    out["eligibility_status"] = out["raw_acceptability_status"]
    out["eligibility_rank"] = out["raw_acceptability_rank"]
    out["fgc_redflag_count"] = out["raw_caution_reason_count"]

    acceptable_mask = out["raw_acceptability_status"].isin(["eligible", "caution"])
    if int(acceptable_mask.sum()) >= FLOWGENCANDIDATE_RELATIVE_POOL_MIN:
        reference_pool = out.loc[acceptable_mask].copy()
        out["flowgencandidate_relative_pool_scope"] = "acceptable_pool"
    else:
        reference_pool = out.copy()
        out["flowgencandidate_relative_pool_scope"] = "all_complete_fallback"
    out["flowgencandidate_relative_pool_size"] = int(len(reference_pool))

    out["rz_train_mvn_score"] = _robust_z_against_reference(out["train_mvn_score"], reference_pool["train_mvn_score"])
    out["rz_val_mvn_score"] = _robust_z_against_reference(out["val_mvn_score"], reference_pool["val_mvn_score"])
    out["rz_train_sum_rrmse"] = _robust_z_against_reference(out["train_sum_rrmse"], reference_pool["train_sum_rrmse"])
    out["rz_val_sum_rrmse"] = _robust_z_against_reference(out["val_sum_rrmse"], reference_pool["val_sum_rrmse"])
    out["rz_train_rrmse_recon"] = _robust_z_against_reference(
        out["train_rrmse_recon"], reference_pool["train_rrmse_recon"]
    )
    out["rz_val_rrmse_recon"] = _robust_z_against_reference(
        out["val_rrmse_recon"], reference_pool["val_rrmse_recon"]
    )
    out["rz_val_rrmse_balance_ratio"] = _robust_z_against_reference(
        out["val_rrmse_balance_ratio"], reference_pool["val_rrmse_balance_ratio"]
    ).clip(lower=-FLOWGENCANDIDATE_VAL_BALANCE_RZ_CLIP, upper=FLOWGENCANDIDATE_VAL_BALANCE_RZ_CLIP)
    out["rz_gap_val_train_sum"] = _robust_z_against_reference(
        out["gap_val_train_sum"], reference_pool["gap_val_train_sum"]
    )
    out["rz_mvn_gap_abs"] = _robust_z_against_reference(out["mvn_gap_abs"], reference_pool["mvn_gap_abs"])
    out["rz_recon_gap_rrmse"] = _robust_z_against_reference(
        out["recon_gap_rrmse"], reference_pool["recon_gap_rrmse"]
    )

    out["train_surface"] = (
        0.45 * out["rz_train_mvn_score"]
        + 0.35 * out["rz_train_sum_rrmse"]
        + 0.20 * out["rz_train_rrmse_recon"]
    )
    out["val_surface"] = (
        FLOWGENCANDIDATE_VAL_SURFACE_WEIGHTS["mvn"] * out["rz_val_mvn_score"]
        + FLOWGENCANDIDATE_VAL_SURFACE_WEIGHTS["sum"] * out["rz_val_sum_rrmse"]
        + FLOWGENCANDIDATE_VAL_SURFACE_WEIGHTS["recon"] * out["rz_val_rrmse_recon"]
        + FLOWGENCANDIDATE_VAL_SURFACE_WEIGHTS["balance"] * out["rz_val_rrmse_balance_ratio"]
    )
    out["gap_surface"] = (
        0.40 * out["rz_gap_val_train_sum"]
        + 0.35 * out["rz_mvn_gap_abs"]
        + 0.25 * out["rz_recon_gap_rrmse"]
    )
    out["priorfit_score"] = (
        FLOWGENCANDIDATE_PRIORFIT_WEIGHTS["train"] * out["train_surface"]
        + FLOWGENCANDIDATE_PRIORFIT_WEIGHTS["val"] * out["val_surface"]
        + FLOWGENCANDIDATE_PRIORFIT_WEIGHTS["gap"] * out["gap_surface"]
    )
    out["robust_score"] = (
        FLOWGENCANDIDATE_ROBUST_WEIGHTS["train"] * out["train_surface"]
        + FLOWGENCANDIDATE_ROBUST_WEIGHTS["val"] * out["val_surface"]
        + FLOWGENCANDIDATE_ROBUST_WEIGHTS["gap"] * out["gap_surface"]
    )
    out["hybrid_score"] = (
        FLOWGENCANDIDATE_HYBRID_WEIGHTS["train"] * out["train_surface"]
        + FLOWGENCANDIDATE_HYBRID_WEIGHTS["val"] * out["val_surface"]
        + FLOWGENCANDIDATE_HYBRID_WEIGHTS["gap"] * out["gap_surface"]
    )
    return out


def _build_current_all_runs(inventory: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "run_id",
        "branch_id",
        "cfg_id",
        "phase",
        "seed",
        "contract_id",
        "campaign_origin",
        "is_new_campaign",
        "is_complete_artifacts",
        "run_manifest_path",
        "results_path",
        "metrics_long_path",
        "config_path",
        "cfg_signature",
        "train_rrmse_mean",
        "train_rrmse_std",
        "val_rrmse_mean",
        "val_rrmse_std",
        "gap_val_train_mean",
        "gap_val_train_std",
        "gap_val_train_sum",
        "train_rrmse_recon",
        "val_rrmse_recon",
        "train_r2_recon",
        "val_r2_recon",
        "train_skew_abs",
        "val_skew_abs",
        "train_kurt_excess_abs",
        "val_kurt_excess_abs",
        "train_mahal_mu",
        "val_mahal_mu",
        "train_mahal_md",
        "val_mahal_md",
        "train_eigstd",
        "val_eigstd",
        "val_pc_worst_mean",
        "val_pc_worst_std",
        "val_pc_wavg_mean",
    ]
    all_runs = inventory[keep_cols].copy()
    all_runs = _augment_metrics(all_runs)
    return all_runs.sort_values(["campaign_origin", "branch_id", "phase", "seed", "run_id"]).reset_index(drop=True)


def _complete_runs(all_runs: pd.DataFrame) -> pd.DataFrame:
    return all_runs[all_runs["is_complete_artifacts"]].copy()


def _rerank_view(
    frame: pd.DataFrame,
    *,
    objective_view: str,
    ranking_stage: str,
    table_score_col: str,
    ordering_rule: str,
) -> pd.DataFrame:
    ranked = frame.reset_index(drop=True).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["objective_view"] = objective_view
    ranked["ranking_stage"] = ranking_stage
    ranked["table_score"] = ranked[table_score_col]
    ranked["ordering_rule"] = ordering_rule
    return ranked


def _sort_ranked_view(
    frame: pd.DataFrame,
    *,
    objective_view: str,
    ranking_stage: str,
    table_score_col: str,
    sort_cols: list[str],
    ordering_rule: str,
) -> pd.DataFrame:
    sorted_frame = frame.sort_values(sort_cols, ascending=[True] * len(sort_cols), na_position="last")
    return _rerank_view(
        sorted_frame,
        objective_view=objective_view,
        ranking_stage=ranking_stage,
        table_score_col=table_score_col,
        ordering_rule=ordering_rule,
    )


def _descriptive_final_runs(all_runs: pd.DataFrame) -> pd.DataFrame:
    complete = _complete_runs(all_runs)
    return complete[complete["global_reconstruction_status"].isin(["pass", "caution"])].copy()


def _rank_rrmse_canonical(all_runs: pd.DataFrame, *, ranking_stage: str) -> pd.DataFrame:
    base = _complete_runs(all_runs) if ranking_stage == "raw" else _descriptive_final_runs(all_runs)
    ranked = rank_flowpre_branch(base, "rrmse")
    return _rerank_view(
        ranked,
        objective_view="rrmse_canonical",
        ranking_stage=ranking_stage,
        table_score_col="val_rrmse_mean",
        ordering_rule=(
            "val_rrmse_mean -> val_rrmse_std -> gap_val_train_sum -> val_rrmse_recon -> val_eigstd"
        ),
    )


def _rank_rrmse_primary(all_runs: pd.DataFrame, *, ranking_stage: str) -> pd.DataFrame:
    base = _complete_runs(all_runs) if ranking_stage == "raw" else _descriptive_final_runs(all_runs)
    return _sort_ranked_view(
        base,
        objective_view="rrmse_primary",
        ranking_stage=ranking_stage,
        table_score_col="rrmse_primary_score",
        sort_cols=["rrmse_primary_score", "val_sum_rrmse", "train_sum_rrmse", "gap_val_train_sum", "run_id"],
        ordering_rule=(
            "rrmse_primary_score = 0.35*train_sum_rrmse + 0.35*val_sum_rrmse + 0.15*gap_val_train_sum "
            "+ 0.05*train_rrmse_balance_ratio + 0.10*val_rrmse_balance_ratio "
            "-> val_sum_rrmse -> train_sum_rrmse -> gap_val_train_sum -> run_id"
        ),
    )


def _rank_mvn(all_runs: pd.DataFrame, *, ranking_stage: str) -> pd.DataFrame:
    base = _complete_runs(all_runs) if ranking_stage == "raw" else _descriptive_final_runs(all_runs)
    ranked = rank_flowpre_branch(base, "mvn")
    return _rerank_view(
        ranked,
        objective_view="mvn",
        ranking_stage=ranking_stage,
        table_score_col="selection_score",
        ordering_rule=(
            "selection_score = (1.0*val_skew_abs + 1.0*val_kurt_excess_abs + 1.2*val_eigstd + "
            "0.3*abs(val_mahal_mu-target_mu) + 0.3*abs(val_mahal_md-target_mu)) / 3.8; "
            "tie -> val_eigstd -> val_kurt_excess_abs -> val_skew_abs -> gap_val_train_sum"
        ),
    )


def _rank_fair(all_runs: pd.DataFrame, *, ranking_stage: str) -> pd.DataFrame:
    base = _complete_runs(all_runs) if ranking_stage == "raw" else _descriptive_final_runs(all_runs)
    ranked = rank_flowpre_branch(base, "fair")
    return _rerank_view(
        ranked,
        objective_view="fair",
        ranking_stage=ranking_stage,
        table_score_col="selection_score",
        ordering_rule=(
            "selection_score = mean(robust_z(val_pc_worst_mean), robust_z(val_pc_worst_std), robust_z(val_pc_wavg_mean)); "
            "tie -> val_pc_worst_mean -> val_pc_worst_std -> val_rrmse_mean -> val_eigstd"
        ),
    )


def _rank_flowgencandidate_priorfit(all_runs: pd.DataFrame, *, ranking_stage: str) -> pd.DataFrame:
    base = _complete_runs(all_runs)
    if ranking_stage == "final":
        base = base[base["raw_acceptability_status"].isin(["eligible", "caution"])].copy()
        sort_cols = ["eligibility_rank", "priorfit_score", "train_surface", "val_rrmse_recon", "run_id"]
        ordering_rule = (
            "eligible+caution only; eligibility_rank -> priorfit_score = 0.50*train_surface + 0.30*val_surface + "
            "0.20*gap_surface -> train_surface -> val_rrmse_recon -> run_id"
        )
    else:
        sort_cols = ["priorfit_score", "train_surface", "val_rrmse_recon", "run_id"]
        ordering_rule = (
            "priorfit_score = 0.50*train_surface + 0.30*val_surface + 0.20*gap_surface "
            "-> train_surface -> val_rrmse_recon -> run_id"
        )
    return _sort_ranked_view(
        base,
        objective_view="flowgencandidate_priorfit",
        ranking_stage=ranking_stage,
        table_score_col="priorfit_score",
        sort_cols=sort_cols,
        ordering_rule=ordering_rule,
    )


def _rank_flowgencandidate_robust(all_runs: pd.DataFrame, *, ranking_stage: str) -> pd.DataFrame:
    base = _complete_runs(all_runs)
    if ranking_stage == "final":
        base = base[base["raw_acceptability_status"].isin(["eligible", "caution"])].copy()
        sort_cols = ["eligibility_rank", "robust_score", "gap_surface", "val_rrmse_recon", "run_id"]
        ordering_rule = (
            "eligible+caution only; eligibility_rank -> robust_score = 0.25*train_surface + 0.35*val_surface + "
            "0.40*gap_surface -> gap_surface -> val_rrmse_recon -> run_id"
        )
    else:
        sort_cols = ["robust_score", "gap_surface", "val_rrmse_recon", "run_id"]
        ordering_rule = (
            "robust_score = 0.25*train_surface + 0.35*val_surface + 0.40*gap_surface "
            "-> gap_surface -> val_rrmse_recon -> run_id"
        )
    return _sort_ranked_view(
        base,
        objective_view="flowgencandidate_robust",
        ranking_stage=ranking_stage,
        table_score_col="robust_score",
        sort_cols=sort_cols,
        ordering_rule=ordering_rule,
    )


def _rank_flowgencandidate_hybrid(all_runs: pd.DataFrame, *, ranking_stage: str) -> pd.DataFrame:
    base = _complete_runs(all_runs)
    if ranking_stage == "final":
        base = base[base["raw_acceptability_status"].isin(["eligible", "caution"])].copy()
        sort_cols = ["eligibility_rank", "hybrid_score", "fair_selection_score_manual", "val_rrmse_recon", "run_id"]
        ordering_rule = (
            "eligible+caution only; eligibility_rank -> hybrid_score = 0.35*train_surface + 0.35*val_surface + "
            "0.30*gap_surface -> fair_selection_score_manual -> val_rrmse_recon -> run_id"
        )
    else:
        sort_cols = ["hybrid_score", "fair_selection_score_manual", "val_rrmse_recon", "run_id"]
        ordering_rule = (
            "hybrid_score = 0.35*train_surface + 0.35*val_surface + 0.30*gap_surface "
            "-> fair_selection_score_manual -> val_rrmse_recon -> run_id"
        )
    return _sort_ranked_view(
        base,
        objective_view="flowgencandidate_hybrid",
        ranking_stage=ranking_stage,
        table_score_col="hybrid_score",
        sort_cols=sort_cols,
        ordering_rule=ordering_rule,
    )


def _rrmse_columns() -> list[str]:
    return [
        "rank",
        "run_id",
        "branch_id",
        "cfg_id",
        "seed",
        "phase",
        "campaign_origin",
        "global_reconstruction_status",
        "train_rrmse_recon",
        "val_rrmse_recon",
        "train_rrmse_mean",
        "train_rrmse_std",
        "train_sum_rrmse",
        "val_rrmse_mean",
        "val_rrmse_std",
        "val_sum_rrmse",
        "both_sum_rrmse",
        "train_balance_gap_rrmse",
        "val_balance_gap_rrmse",
        "balance_gap_rrmse",
        "train_rrmse_balance_ratio",
        "val_rrmse_balance_ratio",
        "balanced_rrmse_penalty",
        "table_score",
        "ordering_rule",
    ]


def _mvn_columns() -> list[str]:
    return [
        "rank",
        "run_id",
        "branch_id",
        "cfg_id",
        "seed",
        "phase",
        "campaign_origin",
        "global_reconstruction_status",
        "train_rrmse_recon",
        "val_rrmse_recon",
        "val_skew_abs",
        "val_kurt_excess_abs",
        "val_eigstd",
        "val_mahal_mu",
        "val_mahal_md",
        "mvn_target_mu",
        "mvn_mu_abs_dev",
        "mvn_md_abs_dev",
        "mvn_weighted_skew",
        "mvn_weighted_kurt",
        "mvn_weighted_eig",
        "mvn_weighted_mu",
        "mvn_weighted_md",
        "selection_score",
        "gap_val_train_sum",
        "ordering_rule",
    ]


def _fair_columns() -> list[str]:
    return [
        "rank",
        "run_id",
        "branch_id",
        "cfg_id",
        "seed",
        "phase",
        "campaign_origin",
        "global_reconstruction_status",
        "train_rrmse_recon",
        "val_rrmse_recon",
        "val_pc_worst_mean",
        "val_pc_worst_std",
        "val_pc_wavg_mean",
        "fair_z_pc_worst_mean",
        "fair_z_pc_worst_std",
        "fair_z_pc_wavg_mean",
        "selection_score",
        "val_rrmse_mean",
        "val_eigstd",
        "ordering_rule",
    ]


def _top3_table(ranked: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return ranked.head(3)[columns].reset_index(drop=True)


def _top3_flowgencandidate_table(ranked: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    visible = ranked[ranked["raw_acceptability_status"] != "reject"].copy()
    return visible.head(3)[columns].reset_index(drop=True)


def _flowgencandidate_columns(score_col: str) -> list[str]:
    cols = [
        "rank",
        "run_id",
        "branch_id",
        "cfg_id",
        "seed",
        "phase",
        "campaign_origin",
        "global_reconstruction_status",
        "global_reconstruction_reasons",
        "raw_acceptability_status",
        "raw_acceptability_rank",
        "raw_acceptability_reasons",
        "raw_reject_reason_count",
        "raw_caution_reason_count",
        "train_rrmse_mean",
        "train_rrmse_std",
        "val_rrmse_mean",
        "val_rrmse_std",
        "train_mvn_score",
        "val_mvn_score",
        "train_sum_rrmse",
        "val_sum_rrmse",
        "val_rrmse_balance_ratio",
        "train_rrmse_recon",
        "val_rrmse_recon",
        "gap_val_train_sum",
        "mvn_gap_abs",
        "recon_gap_rrmse",
        "flowgencandidate_relative_pool_scope",
        "flowgencandidate_relative_pool_size",
        "rz_train_mvn_score",
        "rz_val_mvn_score",
        "rz_train_sum_rrmse",
        "rz_val_sum_rrmse",
        "rz_train_rrmse_recon",
        "rz_val_rrmse_recon",
        "rz_val_rrmse_balance_ratio",
        "rz_gap_val_train_sum",
        "rz_mvn_gap_abs",
        "rz_recon_gap_rrmse",
        "train_surface",
        "val_surface",
        "gap_surface",
        score_col,
        "ordering_rule",
    ]
    if score_col == "hybrid_score":
        insert_at = cols.index("ordering_rule")
        cols.insert(insert_at, "fair_selection_score_manual")
    return cols


def _rrmse_primary_columns() -> list[str]:
    return [
        "rank",
        "run_id",
        "branch_id",
        "cfg_id",
        "seed",
        "campaign_origin",
        "global_reconstruction_status",
        "train_rrmse_recon",
        "val_rrmse_recon",
        "train_rrmse_mean",
        "train_rrmse_std",
        "train_sum_rrmse",
        "val_rrmse_mean",
        "val_rrmse_std",
        "val_sum_rrmse",
        "gap_val_train_sum",
        "train_rrmse_balance_ratio",
        "val_rrmse_balance_ratio",
        "rrmse_primary_score",
        "ordering_rule",
    ]


def _topn_table(ranked: pd.DataFrame, columns: list[str], n: int = TOP_N) -> pd.DataFrame:
    return ranked.head(n)[columns].reset_index(drop=True)


def _write_summary(
    *,
    all_runs: pd.DataFrame,
    inventory: pd.DataFrame,
    top10_raw_tables: dict[str, pd.DataFrame],
    top10_final_tables: dict[str, pd.DataFrame],
    secondary_top3_tables: dict[str, pd.DataFrame],
) -> Path:
    total_runs = int(len(inventory))
    incomplete = inventory[~inventory["is_complete_artifacts"]].copy()
    campaign_counts = inventory["campaign_origin"].value_counts().to_dict()
    raw_status_counts = (
        all_runs["raw_acceptability_status"].value_counts().reindex(["eligible", "caution", "reject"], fill_value=0)
    )
    raw_status_frame = pd.DataFrame({"status": raw_status_counts.index, "count": raw_status_counts.astype(int).tolist()})
    global_recon_counts = (
        all_runs["global_reconstruction_status"].value_counts().reindex(["pass", "caution", "fail"], fill_value=0)
    )
    global_recon_frame = pd.DataFrame(
        {"status": global_recon_counts.index, "count": global_recon_counts.astype(int).tolist()}
    )
    final_descriptive_pool = int(
        (
            all_runs["is_complete_artifacts"].fillna(False)
            & all_runs["global_reconstruction_status"].isin(["pass", "caution"])
        ).sum()
    )
    final_flowgencandidate_pool = int(
        (
            all_runs["is_complete_artifacts"].fillna(False)
            & all_runs["raw_acceptability_status"].isin(["eligible", "caution"])
        ).sum()
    )

    lines = [
        "# FlowPre Current Evaluation Summary",
        "",
        "## Estado general",
        f"- Total official FlowPre runs observed: `{total_runs}`",
        f"- Reparto por `campaign_origin`: `{campaign_counts}`",
        f"- Runs with complete core artifacts: `{int(inventory['is_complete_artifacts'].sum())}`",
        f"- Runs with incomplete core artifacts: `{len(incomplete)}`",
        f"- Pool final de lentes descriptivas (`pass + caution`): `{final_descriptive_pool}`",
        f"- Pool final de `flowgencandidate` (`eligible + caution`): `{final_flowgencandidate_pool}`",
        "- Esta superficie de evaluacion es operativa pero provisional: `FlowPre` sigue en fase abierta y continuan previstas rondas adicionales controladas antes de cualquier cierre formal.",
        "",
        "## Como se computa cada score",
        "",
        "### Variables directas y mappings reales",
        "- `train_rrmse_mean` <- `results.yaml` -> `train.rrmse_mean_whole`",
        "- `train_rrmse_std` <- `results.yaml` -> `train.rrmse_std_whole`",
        "- `val_rrmse_mean` <- `results.yaml` -> `val.rrmse_mean_whole`",
        "- `val_rrmse_std` <- `results.yaml` -> `val.rrmse_std_whole`",
        "- `val_skew_abs` <- `abs(results.yaml -> val.isotropy_stats.skewness_mean)`",
        "- `val_kurt_excess_abs` <- `abs(results.yaml -> val.isotropy_stats.kurtosis_mean - 3.0)`",
        "- `val_mahal_mu` <- `results.yaml` -> `val.isotropy_stats.mahalanobis_mean`",
        "- `val_mahal_md` <- `results.yaml` -> `val.isotropy_stats.mahalanobis_median`",
        "- `val_eigstd` <- `results.yaml` -> `val.isotropy_stats.eigval_std`",
        "- `val_pc_worst_mean` <- `max(val.per_class_iso_rrmse[*].rrmse_mean)`",
        "- `val_pc_worst_std` <- `max(val.per_class_iso_rrmse[*].rrmse_std)`",
        "- `val_pc_wavg_mean` <- media ponderada de `val.per_class_iso_rrmse[*].rrmse_mean` con pesos `n`",
        "- `cfg_id` se toma de `run_manifest.json -> run_level_axes.cfg_id` cuando existe; si no, queda el inferido por el inventario.",
        "- `campaign_origin` es derivado del `contract_id` y del prefijo del `run_id` (`flowprex2_` => `explore_v2`).",
        "",
        "### Criterio real del repo para rrmse",
        "- `evaluation/f6_selection.py::rank_flowpre_branch(..., 'rrmse')` usa orden lexicografico:",
        "  `val_rrmse_mean -> val_rrmse_std -> gap_val_train_sum -> val_rrmse_recon -> val_eigstd`.",
        "",
        "### Scores derivados extra para rrmse en este reporte",
        "- `val_sum_rrmse = val_rrmse_mean + val_rrmse_std`",
        "- `train_sum_rrmse = train_rrmse_mean + train_rrmse_std`",
        "- `both_sum_rrmse = train_rrmse_mean + train_rrmse_std + val_rrmse_mean + val_rrmse_std`",
        "- `train_rrmse_balance_ratio = abs(train_rrmse_mean - train_rrmse_std) / (train_sum_rrmse + eps)`",
        "- `val_rrmse_balance_ratio = abs(val_rrmse_mean - val_rrmse_std) / (val_sum_rrmse + eps)`",
        f"- `balanced_rrmse_penalty = {BALANCED_RRMSE_PENALTY_WEIGHT} * balance_gap_rrmse`",
        "- `balanced_rrmse_score = both_sum_rrmse + balanced_rrmse_penalty`",
        "- Interpretacion: `rrmse_primary` mantiene suma baja en train+val y anade una penalizacion suave y acotada por desequilibrio `mean/std`.",
        "- `rrmse_primary_score = 0.35*train_sum_rrmse + 0.35*val_sum_rrmse + 0.15*gap_val_train_sum + 0.05*train_rrmse_balance_ratio + 0.10*val_rrmse_balance_ratio`",
        "",
        "### Criterio real del repo para mvn",
        f"- `target_mu = sqrt(43 - 0.5) = {MVN_TARGET_MU:.6f}`",
        "- `mvn_mu_abs_dev = abs(val_mahal_mu - target_mu)`",
        "- `mvn_md_abs_dev = abs(val_mahal_md - target_mu)`",
        f"- `selection_score = ({MVN_WEIGHT_SKEW}*val_skew_abs + {MVN_WEIGHT_KURT}*val_kurt_excess_abs + {MVN_WEIGHT_EIG}*val_eigstd + "
        f"{MVN_WEIGHT_MU}*mvn_mu_abs_dev + {MVN_WEIGHT_MD}*mvn_md_abs_dev) / {MVN_WEIGHT_TOTAL}`",
        "- Desempate real: `val_eigstd -> val_kurt_excess_abs -> val_skew_abs -> gap_val_train_sum`.",
        "",
        "### Criterio real del repo para fair",
        "- `fair_z_pc_worst_mean = robust_z(val_pc_worst_mean)`",
        "- `fair_z_pc_worst_std = robust_z(val_pc_worst_std)`",
        "- `fair_z_pc_wavg_mean = robust_z(val_pc_wavg_mean)`",
        "- `selection_score = mean(fair_z_pc_worst_mean, fair_z_pc_worst_std, fair_z_pc_wavg_mean)`",
        "- Desempate real: `val_pc_worst_mean -> val_pc_worst_std -> val_rrmse_mean -> val_eigstd`.",
        f"- Importante: estos robust-z se calculan respecto al conjunto actual de comparacion (`{total_runs}` runs completas), asi que si el universo de runs cambia, el score puede moverse.",
        "",
        "### Lente adicional `flowgencandidate`",
        "- Filosofia: ofrecer una lente adicional provisional para orientar la eleccion futura del `FlowPre` base/upstream de `FlowGen`, no cerrar todavia un winner final de `FlowPre`.",
        "- Capa global de sanidad: `global_reconstruction_status` marca cada run como `pass`, `caution` o `fail` usando solo `train_rrmse_recon`, `val_rrmse_recon`, `recon_gap_rrmse` y no-finitud.",
        "- Thresholds globales de reconstruccion: `train_rrmse_recon` pass `<=0.05`, caution `(0.05, 0.25]`, fail `>0.25`; `val_rrmse_recon` pass `<=1e-5`, caution `(1e-5, 5e-5]`, fail `>5e-5`; `recon_gap_rrmse` pass `<=0.05`, caution `(0.05, 0.25]`, fail `>0.25`.",
        "- Las lentes descriptivas (`rrmse_primary`, `mvn`, `fair`) tienen ahora doble vista: `raw` (todas las runs completas) y `final` (solo `pass + caution`).",
        "- `flowgencandidate_*` tambien tiene doble vista: `raw` ordena por score bruto, mientras `final` usa la lectura operativa actual sobre `eligible + caution`.",
        "- `train_mvn_score` usa la misma formula de `mvn`, pero aplicada sobre metricas de `train`.",
        "- `val_mvn_score = mvn_selection_score_manual`.",
        "- En `flowgencandidate`, `global_reconstruction_status=fail` fuerza `raw_acceptability_status=reject`.",
        "- Reject thresholds raw: `train_rrmse_recon>0.15`, `val_rrmse_recon>5e-5`, `train_sum_rrmse>0.25`, `val_sum_rrmse>0.60`, `train_mvn_score>15`, `val_mvn_score>3.5`, `gap_val_train_sum>0.75`, `mvn_gap_abs>10`, `recon_gap_rrmse>0.15`, o metricas criticas no finitas.",
        "- Caution thresholds raw: `train_rrmse_recon>0.05`, `val_rrmse_recon>1e-5`, `train_sum_rrmse>0.15`, `val_sum_rrmse>0.55`, `train_mvn_score>7.5`, `val_mvn_score>2.5`, `gap_val_train_sum>0.50`, `mvn_gap_abs>4`, `recon_gap_rrmse>0.05`.",
        "- `val_rrmse_balance_ratio = abs(val_rrmse_mean - val_rrmse_std) / (val_sum_rrmse + eps)`",
        "- `train_surface = 0.45*rz(train_mvn_score) + 0.35*rz(train_sum_rrmse) + 0.20*rz(train_rrmse_recon)`",
        "- `val_surface = 0.40*rz(val_mvn_score) + 0.30*rz(val_sum_rrmse) + 0.20*rz(val_rrmse_recon) + 0.10*clip(rz(val_rrmse_balance_ratio), -2.5, 2.5)`",
        "- `gap_surface = 0.40*rz(gap_val_train_sum) + 0.35*rz(abs(train_mvn_score - val_mvn_score)) + 0.25*rz(abs(train_rrmse_recon - val_rrmse_recon))`",
        "- `priorfit_score = 0.50*train_surface + 0.30*val_surface + 0.20*gap_surface`",
        "- `robust_score = 0.25*train_surface + 0.35*val_surface + 0.40*gap_surface`",
        "- `hybrid_score = 0.35*train_surface + 0.35*val_surface + 0.30*gap_surface`",
        "",
        "## Resumen global de reconstruccion",
        _frame_to_markdown(global_recon_frame),
        "",
        "## Resumen raw de flowgencandidate",
        _frame_to_markdown(raw_status_frame),
        "",
        "## TOP 10 por lente",
    ]

    for lens_id, display_name in LENS_DISPLAY_NAMES.items():
        lines.extend(
            [
                "",
                f"### {display_name}",
                "- `raw` = score bruto de la lente sin aplicar la capa operativa final.",
                "- `final` = lectura operativa actual tras aplicar la capa final correspondiente.",
                "",
                "#### TOP 10 raw",
                _frame_to_markdown(top10_raw_tables[lens_id]),
                "",
                "#### TOP 10 final",
                _frame_to_markdown(top10_final_tables[lens_id]),
            ]
        )

    if secondary_top3_tables:
        lines.extend(
            [
                "",
                "## Vistas diagnosticas legacy de rrmse",
                "- Estas tablas se mantienen por compatibilidad y apoyo diagnostico; la lectura principal sigue siendo `rrmse_primary` con vistas `raw/final`.",
                "",
                "### rrmse por suma en val",
                _frame_to_markdown(secondary_top3_tables["rrmse_valsum"]),
                "",
                "### rrmse por suma en train",
                _frame_to_markdown(secondary_top3_tables["rrmse_trainsum"]),
                "",
                "### rrmse por suma train+val",
                _frame_to_markdown(secondary_top3_tables["rrmse_bothsum"]),
                "",
                "### rrmse por criterio balanceado",
                _frame_to_markdown(secondary_top3_tables["rrmse_balanced"]),
                "",
                "### rrmse canonico legacy",
                _frame_to_markdown(secondary_top3_tables["rrmse_canonical"]),
            ]
        )

    lines.extend(
        [
            "",
            "## Lectura corta",
            "- `rrmse_primary` deja trazabilidad en dos capas: `raw` para diagnostico y `final` para lectura operativa con `pass + caution`.",
            "- `flowgencandidate_*` deja visibles las runs que puntuan bien en bruto pero caen en aceptabilidad, lo que ayuda a distinguir score de operatividad.",
            "- La penalizacion suave por desequilibrio `val mean/std` no domina la nota: actua como ajuste fino dentro de `rrmse_primary` y de `val_surface`.",
        ]
    )

    if incomplete.empty:
        lines.extend(["", "## Advertencias", f"- No se detectaron runs incompletas entre las `{total_runs}` observadas."])
    else:
        lines.extend(["", "## Advertencias", "- Hay runs incompletas; revisar `flowpre_current_inventory.csv`."])

    out_path = REPORT_ROOT / "flowpre_current_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    inventory = _build_inventory()
    if inventory.empty:
        raise RuntimeError("No FlowPre official runs found under outputs/models/official/flow_pre.")

    all_runs = _build_current_all_runs(inventory)

    ranked_rrmse_primary_raw = _rank_rrmse_primary(all_runs, ranking_stage="raw")
    ranked_rrmse_primary_final = _rank_rrmse_primary(all_runs, ranking_stage="final")
    ranked_rrmse_valsum = _sort_ranked_view(
        _descriptive_final_runs(all_runs),
        objective_view="rrmse_valsum",
        ranking_stage="final",
        table_score_col="val_sum_rrmse",
        sort_cols=["val_sum_rrmse", "val_rrmse_mean", "val_rrmse_std", "gap_val_train_sum", "run_id"],
        ordering_rule="val_sum_rrmse = val_rrmse_mean + val_rrmse_std; tie -> val_rrmse_mean -> val_rrmse_std -> gap_val_train_sum",
    )
    ranked_rrmse_trainsum = _sort_ranked_view(
        _descriptive_final_runs(all_runs),
        objective_view="rrmse_trainsum",
        ranking_stage="final",
        table_score_col="train_sum_rrmse",
        sort_cols=["train_sum_rrmse", "train_rrmse_mean", "train_rrmse_std", "run_id"],
        ordering_rule="train_sum_rrmse = train_rrmse_mean + train_rrmse_std; tie -> train_rrmse_mean -> train_rrmse_std",
    )
    ranked_rrmse_bothsum = _sort_ranked_view(
        _descriptive_final_runs(all_runs),
        objective_view="rrmse_bothsum",
        ranking_stage="final",
        table_score_col="both_sum_rrmse",
        sort_cols=["both_sum_rrmse", "val_sum_rrmse", "train_sum_rrmse", "run_id"],
        ordering_rule="both_sum_rrmse = train_sum_rrmse + val_sum_rrmse; tie -> val_sum_rrmse -> train_sum_rrmse",
    )
    ranked_rrmse_balanced = _sort_ranked_view(
        _descriptive_final_runs(all_runs),
        objective_view="rrmse_balanced",
        ranking_stage="final",
        table_score_col="balanced_rrmse_score",
        sort_cols=["balanced_rrmse_score", "both_sum_rrmse", "balance_gap_rrmse", "run_id"],
        ordering_rule=(
            f"balanced_rrmse_score = both_sum_rrmse + {BALANCED_RRMSE_PENALTY_WEIGHT}*balance_gap_rrmse; "
            "tie -> both_sum_rrmse -> balance_gap_rrmse"
        ),
    )
    ranked_rrmse_canonical = _rank_rrmse_canonical(all_runs, ranking_stage="final")
    ranked_mvn_raw = _rank_mvn(all_runs, ranking_stage="raw")
    ranked_mvn_final = _rank_mvn(all_runs, ranking_stage="final")
    ranked_fair_raw = _rank_fair(all_runs, ranking_stage="raw")
    ranked_fair_final = _rank_fair(all_runs, ranking_stage="final")
    ranked_flowgencandidate_priorfit_raw = _rank_flowgencandidate_priorfit(all_runs, ranking_stage="raw")
    ranked_flowgencandidate_priorfit_final = _rank_flowgencandidate_priorfit(all_runs, ranking_stage="final")
    ranked_flowgencandidate_robust_raw = _rank_flowgencandidate_robust(all_runs, ranking_stage="raw")
    ranked_flowgencandidate_robust_final = _rank_flowgencandidate_robust(all_runs, ranking_stage="final")
    ranked_flowgencandidate_hybrid_raw = _rank_flowgencandidate_hybrid(all_runs, ranking_stage="raw")
    ranked_flowgencandidate_hybrid_final = _rank_flowgencandidate_hybrid(all_runs, ranking_stage="final")

    top10_raw_tables = {
        "rrmse_primary": _topn_table(ranked_rrmse_primary_raw, _rrmse_primary_columns()),
        "mvn": _topn_table(ranked_mvn_raw, _mvn_columns()),
        "fair": _topn_table(ranked_fair_raw, _fair_columns()),
        "flowgencandidate_priorfit": _topn_table(
            ranked_flowgencandidate_priorfit_raw, _flowgencandidate_columns("priorfit_score")
        ),
        "flowgencandidate_robust": _topn_table(
            ranked_flowgencandidate_robust_raw, _flowgencandidate_columns("robust_score")
        ),
        "flowgencandidate_hybrid": _topn_table(
            ranked_flowgencandidate_hybrid_raw, _flowgencandidate_columns("hybrid_score")
        ),
    }
    top10_final_tables = {
        "rrmse_primary": _topn_table(ranked_rrmse_primary_final, _rrmse_primary_columns()),
        "mvn": _topn_table(ranked_mvn_final, _mvn_columns()),
        "fair": _topn_table(ranked_fair_final, _fair_columns()),
        "flowgencandidate_priorfit": _topn_table(
            ranked_flowgencandidate_priorfit_final, _flowgencandidate_columns("priorfit_score")
        ),
        "flowgencandidate_robust": _topn_table(
            ranked_flowgencandidate_robust_final, _flowgencandidate_columns("robust_score")
        ),
        "flowgencandidate_hybrid": _topn_table(
            ranked_flowgencandidate_hybrid_final, _flowgencandidate_columns("hybrid_score")
        ),
    }

    top3_rrmse_primary = _topn_table(top10_final_tables["rrmse_primary"], _rrmse_primary_columns(), n=LEGACY_TOP_N)
    top3_mvn = _topn_table(top10_final_tables["mvn"], _mvn_columns(), n=LEGACY_TOP_N)
    top3_fair = _topn_table(top10_final_tables["fair"], _fair_columns(), n=LEGACY_TOP_N)
    top3_flowgencandidate_priorfit = _topn_table(
        top10_final_tables["flowgencandidate_priorfit"], _flowgencandidate_columns("priorfit_score"), n=LEGACY_TOP_N
    )
    top3_flowgencandidate_robust = _topn_table(
        top10_final_tables["flowgencandidate_robust"], _flowgencandidate_columns("robust_score"), n=LEGACY_TOP_N
    )
    top3_flowgencandidate_hybrid = _topn_table(
        top10_final_tables["flowgencandidate_hybrid"], _flowgencandidate_columns("hybrid_score"), n=LEGACY_TOP_N
    )
    top3_rrmse_valsum = _topn_table(ranked_rrmse_valsum, _rrmse_columns(), n=LEGACY_TOP_N)
    top3_rrmse_trainsum = _topn_table(ranked_rrmse_trainsum, _rrmse_columns(), n=LEGACY_TOP_N)
    top3_rrmse_bothsum = _topn_table(ranked_rrmse_bothsum, _rrmse_columns(), n=LEGACY_TOP_N)
    top3_rrmse_balanced = _topn_table(ranked_rrmse_balanced, _rrmse_columns(), n=LEGACY_TOP_N)
    top3_rrmse_canonical = _topn_table(ranked_rrmse_canonical, _rrmse_columns(), n=LEGACY_TOP_N)

    inventory.to_csv(REPORT_ROOT / "flowpre_current_inventory.csv", index=False)
    all_runs.to_csv(REPORT_ROOT / "flowpre_current_all_runs.csv", index=False)
    top3_rrmse_primary.to_csv(REPORT_ROOT / "flowpre_current_top3_rrmse_primary.csv", index=False)
    top3_rrmse_valsum.to_csv(REPORT_ROOT / "flowpre_current_top3_rrmse_valsum.csv", index=False)
    top3_rrmse_trainsum.to_csv(REPORT_ROOT / "flowpre_current_top3_rrmse_trainsum.csv", index=False)
    top3_rrmse_bothsum.to_csv(REPORT_ROOT / "flowpre_current_top3_rrmse_bothsum.csv", index=False)
    top3_rrmse_balanced.to_csv(REPORT_ROOT / "flowpre_current_top3_rrmse_balanced.csv", index=False)
    top3_rrmse_canonical.to_csv(REPORT_ROOT / "flowpre_current_top3_rrmse_canonical.csv", index=False)
    top3_mvn.to_csv(REPORT_ROOT / "flowpre_current_top3_mvn.csv", index=False)
    top3_fair.to_csv(REPORT_ROOT / "flowpre_current_top3_fair.csv", index=False)
    top3_flowgencandidate_priorfit.to_csv(
        REPORT_ROOT / "flowpre_current_top3_flowgencandidate_priorfit.csv", index=False
    )
    top3_flowgencandidate_robust.to_csv(
        REPORT_ROOT / "flowpre_current_top3_flowgencandidate_robust.csv", index=False
    )
    top3_flowgencandidate_hybrid.to_csv(
        REPORT_ROOT / "flowpre_current_top3_flowgencandidate_hybrid.csv", index=False
    )

    for lens_id, frame in top10_raw_tables.items():
        frame.to_csv(REPORT_ROOT / f"flowpre_current_top10_{lens_id}_raw.csv", index=False)
    for lens_id, frame in top10_final_tables.items():
        frame.to_csv(REPORT_ROOT / f"flowpre_current_top10_{lens_id}_final.csv", index=False)

    pd.concat(
        [frame.assign(view_id=lens_id) for lens_id, frame in top10_raw_tables.items()],
        ignore_index=True,
        sort=False,
    ).to_csv(REPORT_ROOT / "flowpre_current_top10_raw_combined.csv", index=False)
    pd.concat(
        [frame.assign(view_id=lens_id) for lens_id, frame in top10_final_tables.items()],
        ignore_index=True,
        sort=False,
    ).to_csv(REPORT_ROOT / "flowpre_current_top10_final_combined.csv", index=False)

    pd.concat(
        [
            ranked_flowgencandidate_priorfit_raw.assign(view_id="flowgencandidate_priorfit"),
            ranked_flowgencandidate_robust_raw.assign(view_id="flowgencandidate_robust"),
            ranked_flowgencandidate_hybrid_raw.assign(view_id="flowgencandidate_hybrid"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(REPORT_ROOT / "flowpre_current_flowgencandidate_ranked_raw.csv", index=False)
    pd.concat(
        [
            ranked_flowgencandidate_priorfit_final.assign(view_id="flowgencandidate_priorfit"),
            ranked_flowgencandidate_robust_final.assign(view_id="flowgencandidate_robust"),
            ranked_flowgencandidate_hybrid_final.assign(view_id="flowgencandidate_hybrid"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(REPORT_ROOT / "flowpre_current_flowgencandidate_ranked.csv", index=False)
    pd.concat(
        [
            top3_rrmse_primary.assign(view_id="rrmse_primary"),
            top3_rrmse_valsum.assign(view_id="rrmse_valsum"),
            top3_rrmse_trainsum.assign(view_id="rrmse_trainsum"),
            top3_rrmse_bothsum.assign(view_id="rrmse_bothsum"),
            top3_rrmse_balanced.assign(view_id="rrmse_balanced"),
            top3_rrmse_canonical.assign(view_id="rrmse_canonical"),
            top3_mvn.assign(view_id="mvn"),
            top3_fair.assign(view_id="fair"),
            top3_flowgencandidate_priorfit.assign(view_id="flowgencandidate_priorfit"),
            top3_flowgencandidate_robust.assign(view_id="flowgencandidate_robust"),
            top3_flowgencandidate_hybrid.assign(view_id="flowgencandidate_hybrid"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(REPORT_ROOT / "flowpre_current_top3_combined.csv", index=False)

    _write_summary(
        all_runs=all_runs,
        inventory=inventory,
        top10_raw_tables=top10_raw_tables,
        top10_final_tables=top10_final_tables,
        secondary_top3_tables={
            "rrmse_valsum": top3_rrmse_valsum,
            "rrmse_trainsum": top3_rrmse_trainsum,
            "rrmse_bothsum": top3_rrmse_bothsum,
            "rrmse_balanced": top3_rrmse_balanced,
            "rrmse_canonical": top3_rrmse_canonical,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
