from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sets import DEFAULT_OFFICIAL_DATASET_NAME
from scripts.f6_common import write_yaml
from scripts.f6_flowpre_evaluate_current import (
    _build_current_all_runs,
    _build_inventory,
    _rank_fair,
    _rank_flowgencandidate_hybrid,
    _rank_flowgencandidate_priorfit,
    _rank_flowgencandidate_robust,
    _rank_mvn,
    _rank_rrmse_primary,
)
from scripts.f6_flowpre_explore_v2 import (
    ANCHORS,
    HISTORICAL_GRID,
    _bool_mode,
    _config_from_spec,
    _config_signature,
    _format_lr,
    _format_ratio,
    _generate_param_combinations,
    _spec_distance,
)


OFFICIAL_FLOWPRE_ROOT = ROOT / "outputs" / "models" / "official" / "flow_pre"
REPORT_ROOT = ROOT / "outputs" / "reports" / "f6_explore_v3"
CONFIG_ROOT = REPORT_ROOT / "configs" / "flowpre"
EXPLORE_CONTRACT_ID = "f6_flowpre_explore_v3"
OFFICIAL_SPLIT_ID = "init_temporal_processed_v1"
COMMON_EXPLORE_SEED = 5678
TOP_UNIQUE_OBSERVED = 8
PROPOSALS_PER_RANKING = 5
TOTAL_UNIQUE_PROPOSALS = 30
RELAXED_DIVERSITY_AFTER = 3

RANKING_ORDER = [
    "rrmse_primary",
    "mvn",
    "fair",
    "flowgencandidate_priorfit",
    "flowgencandidate_robust",
    "flowgencandidate_hybrid",
]

RANKING_SHORT = {
    "rrmse_primary": "rp",
    "mvn": "mvn",
    "fair": "fair",
    "flowgencandidate_priorfit": "fgp",
    "flowgencandidate_robust": "fgr",
    "flowgencandidate_hybrid": "fgh",
}

RANKING_TITLES = {
    "rrmse_primary": "rrmse_primary",
    "mvn": "mvn",
    "fair": "fair",
    "flowgencandidate_priorfit": "flowgencandidate_priorfit",
    "flowgencandidate_robust": "flowgencandidate_robust",
    "flowgencandidate_hybrid": "flowgencandidate_hybrid",
}

RANKING_DESCRIPTIONS = {
    "rrmse_primary": "Lectura principal actual de escalado/isotropia rrmse; prioriza train+val con penalizacion moderada por gap y desbalance mean/std.",
    "mvn": "Lente de gaussianidad multivariante; observa isotropia, skew, kurtosis y proximidad Mahalanobis al target.",
    "fair": "Lente de equilibrio per-class; prioriza peor clase, dispersion per-class y media ponderada por clase.",
    "flowgencandidate_priorfit": "Lente provisional para upstream; prioriza prior-fit y sampleability en train con control de degradacion hacia val.",
    "flowgencandidate_robust": "Lente provisional para upstream; prioriza menor fragilidad train-val y gaps mas contenidos.",
    "flowgencandidate_hybrid": "Lente provisional balanceada; mezcla prior-fit, robustez y un tie-break ligero por fairness.",
}

RANKING_SPECS: dict[str, dict[str, Any]] = {
    "rrmse_primary": {
        "sort_cols": ["rrmse_primary_score", "val_sum_rrmse", "train_sum_rrmse", "gap_val_train_sum", "run_id"],
        "exclude_reject": False,
        "score_col": "rrmse_primary_score",
        "top_table_cols": [
            "unique_rank",
            "cfg_signature",
            "branch_id",
            "cfg_id",
            "campaign_origin",
            "raw_acceptability_status",
            "train_sum_rrmse",
            "val_sum_rrmse",
            "gap_val_train_sum",
            "balance_gap_rrmse",
            "rrmse_primary_score",
        ],
        "axis_weights": {"hf": 0.8, "layers": 1.2, "rq": 1.2, "lr": 1.5, "ms": 2.0, "sk": 1.0},
        "prototype_rank_penalty": 0.10,
        "distance_weight": 0.62,
        "preference_weight": 0.38,
        "diversity_threshold": 0.55,
    },
    "mvn": {
        "sort_cols": ["mvn_selection_score_manual", "val_eigstd", "val_kurt_excess_abs", "val_skew_abs", "gap_val_train_sum", "run_id"],
        "exclude_reject": False,
        "score_col": "mvn_selection_score_manual",
        "top_table_cols": [
            "unique_rank",
            "cfg_signature",
            "branch_id",
            "cfg_id",
            "campaign_origin",
            "raw_acceptability_status",
            "val_eigstd",
            "val_kurt_excess_abs",
            "val_skew_abs",
            "mvn_selection_score_manual",
        ],
        "axis_weights": {"hf": 0.8, "layers": 1.0, "rq": 1.4, "lr": 1.6, "ms": 1.6, "sk": 0.7},
        "prototype_rank_penalty": 0.08,
        "distance_weight": 0.68,
        "preference_weight": 0.32,
        "diversity_threshold": 0.45,
    },
    "fair": {
        "sort_cols": ["fair_selection_score_manual", "val_pc_worst_mean", "val_pc_worst_std", "val_rrmse_mean", "val_eigstd", "run_id"],
        "exclude_reject": False,
        "score_col": "fair_selection_score_manual",
        "top_table_cols": [
            "unique_rank",
            "cfg_signature",
            "branch_id",
            "cfg_id",
            "campaign_origin",
            "raw_acceptability_status",
            "val_pc_worst_mean",
            "val_pc_worst_std",
            "val_pc_wavg_mean",
            "fair_selection_score_manual",
        ],
        "axis_weights": {"hf": 0.8, "layers": 1.4, "rq": 1.4, "lr": 1.5, "ms": 1.6, "sk": 0.7},
        "prototype_rank_penalty": 0.08,
        "distance_weight": 0.68,
        "preference_weight": 0.32,
        "diversity_threshold": 0.45,
    },
    "flowgencandidate_priorfit": {
        "sort_cols": ["eligibility_rank", "priorfit_score", "train_surface", "val_rrmse_recon", "run_id"],
        "exclude_reject": True,
        "score_col": "priorfit_score",
        "top_table_cols": [
            "unique_rank",
            "cfg_signature",
            "branch_id",
            "cfg_id",
            "campaign_origin",
            "raw_acceptability_status",
            "train_surface",
            "val_surface",
            "gap_surface",
            "priorfit_score",
        ],
        "axis_weights": {"hf": 1.0, "layers": 1.0, "rq": 1.1, "lr": 1.3, "ms": 2.1, "sk": 2.0},
        "prototype_rank_penalty": 0.12,
        "distance_weight": 0.64,
        "preference_weight": 0.36,
        "diversity_threshold": 0.50,
    },
    "flowgencandidate_robust": {
        "sort_cols": ["eligibility_rank", "robust_score", "gap_surface", "val_rrmse_recon", "run_id"],
        "exclude_reject": True,
        "score_col": "robust_score",
        "top_table_cols": [
            "unique_rank",
            "cfg_signature",
            "branch_id",
            "cfg_id",
            "campaign_origin",
            "raw_acceptability_status",
            "train_surface",
            "val_surface",
            "gap_surface",
            "robust_score",
        ],
        "axis_weights": {"hf": 1.2, "layers": 1.2, "rq": 1.0, "lr": 1.2, "ms": 2.1, "sk": 2.0},
        "prototype_rank_penalty": 0.05,
        "distance_weight": 0.72,
        "preference_weight": 0.28,
        "diversity_threshold": 0.50,
    },
    "flowgencandidate_hybrid": {
        "sort_cols": ["eligibility_rank", "hybrid_score", "fair_selection_score_manual", "val_rrmse_recon", "run_id"],
        "exclude_reject": True,
        "score_col": "hybrid_score",
        "top_table_cols": [
            "unique_rank",
            "cfg_signature",
            "branch_id",
            "cfg_id",
            "campaign_origin",
            "raw_acceptability_status",
            "train_surface",
            "val_surface",
            "gap_surface",
            "hybrid_score",
        ],
        "axis_weights": {"hf": 1.0, "layers": 1.1, "rq": 1.05, "lr": 1.25, "ms": 2.1, "sk": 2.0},
        "prototype_rank_penalty": 0.09,
        "distance_weight": 0.68,
        "preference_weight": 0.32,
        "diversity_threshold": 0.50,
    },
}

RANKING_FUNCTIONS = {
    "rrmse_primary": _rank_rrmse_primary,
    "mvn": _rank_mvn,
    "fair": _rank_fair,
    "flowgencandidate_priorfit": _rank_flowgencandidate_priorfit,
    "flowgencandidate_robust": _rank_flowgencandidate_robust,
    "flowgencandidate_hybrid": _rank_flowgencandidate_hybrid,
}

SIGNATURE_RE = re.compile(
    r"hf(?P<hf>\d+)\|l(?P<layers>\d+)\|rq1x(?P<rq>\d+)\|frq(?P<frq>\d+)\|lr(?P<lr>[^|]+)\|ms(?P<ms>[^|]+)\|sk(?P<sk>[^|]+)"
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plan or train the next exploratory FlowPre campaign (v3).")
    ap.add_argument("--dry-run", action="store_true", help="Analyze current official runs and write the v3 plan only.")
    ap.add_argument("--train", action="store_true", help="Train the planned v3 configs after writing the dry-run plan.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true", help="Re-run planned v3 runs even if they already exist.")
    ap.add_argument("--quiet", action="store_true", help="Reduce trainer logs in --train mode.")
    return ap.parse_args()


def _mode_from_args(args: argparse.Namespace) -> str:
    if args.dry_run and args.train:
        raise RuntimeError("Use either --dry-run or --train, not both.")
    if args.train:
        return "train"
    return "dry-run"


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


def _sorted_unique(values: pd.Series) -> str:
    cleaned = sorted({str(value) for value in values if pd.notna(value) and str(value) != ""})
    return "|".join(cleaned)


def _signature_fields_from_match(match: re.Match[str]) -> dict[str, Any]:
    groups = match.groupdict()
    lr_token = str(groups["lr"])
    lr_value = {"1e-3": 1e-3, "1e-4": 1e-4, "1e-5": 1e-5}[lr_token]
    return {
        "hidden_features": int(groups["hf"]),
        "num_layers": int(groups["layers"]),
        "affine_rq_ratio": [1, int(groups["rq"])],
        "final_rq_layers": int(groups["frq"]),
        "learning_rate": float(lr_value),
        "use_mean_penalty": groups["ms"] == "on",
        "use_std_penalty": groups["ms"] == "on",
        "use_skew_penalty": groups["sk"] == "on",
        "use_kurtosis_penalty": groups["sk"] == "on",
        "hf": groups["hf"],
        "layers": groups["layers"],
        "rq": groups["rq"],
        "lr": lr_token,
        "ms": groups["ms"],
        "sk": groups["sk"],
    }


def _parse_signature(signature: str) -> dict[str, Any]:
    match = SIGNATURE_RE.fullmatch(str(signature))
    if not match:
        raise RuntimeError(f"Unparseable cfg_signature: {signature}")
    return _signature_fields_from_match(match)


def _with_signature_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed_rows: list[dict[str, Any]] = []
    for signature in out["cfg_signature"].astype(str):
        parsed_rows.append(_parse_signature(signature))
    parsed_df = pd.DataFrame(parsed_rows, index=out.index)
    for col in parsed_df.columns:
        out[col] = parsed_df[col]
    out["affine_rq_ratio_str"] = out["affine_rq_ratio"].apply(_format_ratio)
    out["meanstd_mode"] = out["ms"]
    out["skewkurt_mode"] = out["sk"]
    out["complexity_score"] = (
        out["hidden_features"].astype(int) * out["num_layers"].astype(int) * out["final_rq_layers"].astype(int)
    )
    return out


def _build_candidate_pool(existing_signatures: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for param_set in _generate_param_combinations(HISTORICAL_GRID):
        spec = {
            "hidden_features": int(param_set["model.hidden_features"]),
            "num_layers": int(param_set["model.num_layers"]),
            "affine_rq_ratio": [int(x) for x in param_set["model.affine_rq_ratio"]],
            "final_rq_layers": int(param_set["model.final_rq_layers"]),
            "learning_rate": float(param_set["training.learning_rate"]),
            "use_mean_penalty": bool(param_set["training.use_mean_penalty"]),
            "use_std_penalty": bool(param_set["training.use_std_penalty"]),
            "use_skew_penalty": bool(param_set["training.use_skew_penalty"]),
            "use_kurtosis_penalty": bool(param_set["training.use_kurtosis_penalty"]),
        }
        signature = _config_signature(spec)
        rows.append(
            {
                **spec,
                "cfg_signature": signature,
                "hf": str(spec["hidden_features"]),
                "layers": str(spec["num_layers"]),
                "rq": str(spec["final_rq_layers"]),
                "lr": _format_lr(float(spec["learning_rate"])),
                "ms": _bool_mode(spec["use_mean_penalty"], spec["use_std_penalty"]),
                "sk": _bool_mode(spec["use_skew_penalty"], spec["use_kurtosis_penalty"]),
                "affine_rq_ratio_str": _format_ratio(spec["affine_rq_ratio"]),
                "meanstd_mode": _bool_mode(spec["use_mean_penalty"], spec["use_std_penalty"]),
                "skewkurt_mode": _bool_mode(spec["use_skew_penalty"], spec["use_kurtosis_penalty"]),
                "complexity_score": int(spec["hidden_features"]) * int(spec["num_layers"]) * int(spec["final_rq_layers"]),
                "is_already_trained": signature in existing_signatures,
            }
        )
    pool = pd.DataFrame(rows).drop_duplicates(subset=["cfg_signature"]).reset_index(drop=True)
    pool = pool[~pool["is_already_trained"]].copy().reset_index(drop=True)
    return pool.sort_values(
        ["hidden_features", "num_layers", "final_rq_layers", "learning_rate", "cfg_signature"]
    ).reset_index(drop=True)


def _ranked_unique_observed(all_runs: pd.DataFrame, ranking_name: str) -> pd.DataFrame:
    ranked = RANKING_FUNCTIONS[ranking_name](all_runs.copy())
    if RANKING_SPECS[ranking_name]["exclude_reject"]:
        ranked = ranked[ranked["raw_acceptability_status"] != "reject"].copy()
    ranked = ranked.drop_duplicates(subset=["cfg_signature"], keep="first").reset_index(drop=True)
    ranked["unique_rank"] = range(1, len(ranked) + 1)
    return _with_signature_parts(ranked)


def _enrichment_rows(top_df: pd.DataFrame, universe_df: pd.DataFrame, axis: str) -> list[dict[str, Any]]:
    top_counts = top_df[axis].value_counts(dropna=False)
    universe_counts = universe_df[axis].value_counts(dropna=False)
    rows: list[dict[str, Any]] = []
    top_n = max(1, len(top_df))
    universe_n = max(1, len(universe_df))
    values = sorted({str(v) for v in top_counts.index.astype(str)} | {str(v) for v in universe_counts.index.astype(str)})
    for value in values:
        top_share = float(top_counts.get(value, 0)) / float(top_n)
        global_share = float(universe_counts.get(value, 0)) / float(universe_n)
        rows.append(
            {
                "axis": axis,
                "value": value,
                "top_share": top_share,
                "global_share": global_share,
                "delta": top_share - global_share,
            }
        )
    return rows


def _format_enrichment_bullets(enrichment_df: pd.DataFrame, *, positive: bool) -> list[str]:
    if enrichment_df.empty:
        return ["- sin senal clara"]
    if positive:
        view = enrichment_df[enrichment_df["delta"] >= 0.15].sort_values(["delta", "top_share"], ascending=[False, False])
    else:
        view = enrichment_df[(enrichment_df["delta"] <= -0.15) & (enrichment_df["global_share"] >= 0.10)].sort_values(
            ["delta", "global_share"], ascending=[True, False]
        )
    if view.empty:
        return ["- sin senal clara"]
    bullets = []
    for row in view.head(5).itertuples(index=False):
        bullets.append(
            f"- `{row.axis}={row.value}` (top={row.top_share:.2f}, global={row.global_share:.2f}, delta={row.delta:+.2f})"
        )
    return bullets


def _pair_synergies(top_df: pd.DataFrame) -> list[str]:
    pairs = [
        ("hf", "layers"),
        ("rq", "lr"),
        ("ms", "sk"),
    ]
    bullets: list[str] = []
    for lhs, rhs in pairs:
        counts = top_df.groupby([lhs, rhs]).size().sort_values(ascending=False)
        if counts.empty:
            continue
        combo = counts.index[0]
        count = int(counts.iloc[0])
        if count < 2:
            continue
        bullets.append(f"- `{lhs}={combo[0]} + {rhs}={combo[1]}` aparece {count} veces en el top unico observado.")
    return bullets or ["- sin repeticion dominante clara"]


def _dominant_axis_summary(top_df: pd.DataFrame) -> str:
    parts: list[str] = []
    for axis in ("hf", "layers", "rq", "lr", "ms", "sk"):
        counts = top_df[axis].value_counts()
        if counts.empty:
            continue
        top_values = list(counts.head(2).index.astype(str))
        parts.append(f"{axis}=" + "/".join(top_values))
    return ", ".join(parts)


def _nearest_prototype_details(candidate: pd.Series, top_df: pd.DataFrame, ranking_name: str) -> dict[str, Any]:
    best_distance = math.inf
    best_row: pd.Series | None = None
    rank_penalty = float(RANKING_SPECS[ranking_name]["prototype_rank_penalty"])
    candidate_dict = candidate.to_dict()
    for _, row in top_df.iterrows():
        proto_spec = {
            "hidden_features": int(row["hidden_features"]),
            "num_layers": int(row["num_layers"]),
            "affine_rq_ratio": [1, int(row["rq"])],
            "final_rq_layers": int(row["final_rq_layers"]),
            "learning_rate": float(row["learning_rate"]),
            "use_mean_penalty": bool(row["use_mean_penalty"]),
            "use_skew_penalty": bool(row["use_skew_penalty"]),
        }
        distance = _spec_distance(candidate_dict, proto_spec) + rank_penalty * (int(row["unique_rank"]) - 1)
        if distance < best_distance:
            best_distance = distance
            best_row = row
    if best_row is None:
        raise RuntimeError(f"No prototype found for ranking={ranking_name}")
    return {
        "prototype_signature": str(best_row["cfg_signature"]),
        "prototype_branch_id": str(best_row["branch_id"]),
        "prototype_cfg_id": str(best_row["cfg_id"]),
        "prototype_campaign_origin": str(best_row["campaign_origin"]),
        "prototype_unique_rank": int(best_row["unique_rank"]),
        "prototype_distance": float(best_distance),
    }


def _candidate_preference_penalty(candidate: pd.Series, top_df: pd.DataFrame, ranking_name: str) -> float:
    axis_weights = dict(RANKING_SPECS[ranking_name]["axis_weights"])
    top_n = max(1, len(top_df))
    penalty = 0.0
    for axis, weight in axis_weights.items():
        counts = top_df[axis].value_counts(dropna=False).to_dict()
        penalty += float(weight) * (1.0 - float(counts.get(str(candidate[axis]), 0)) / float(top_n))
    return float(penalty)


def _candidate_score(candidate: pd.Series, top_df: pd.DataFrame, ranking_name: str) -> tuple[float, dict[str, Any]]:
    prototype = _nearest_prototype_details(candidate, top_df, ranking_name)
    preference_penalty = _candidate_preference_penalty(candidate, top_df, ranking_name)
    score = (
        float(RANKING_SPECS[ranking_name]["distance_weight"]) * float(prototype["prototype_distance"])
        + float(RANKING_SPECS[ranking_name]["preference_weight"]) * float(preference_penalty)
    )
    return float(score), {**prototype, "preference_penalty": float(preference_penalty)}


def _changes_vs_prototype(candidate: pd.Series, prototype_signature: str) -> str:
    proto = _parse_signature(prototype_signature)
    changes: list[str] = []
    if int(candidate["hidden_features"]) != int(proto["hidden_features"]):
        changes.append(f"hf {proto['hidden_features']}->{int(candidate['hidden_features'])}")
    if int(candidate["num_layers"]) != int(proto["num_layers"]):
        changes.append(f"layers {proto['num_layers']}->{int(candidate['num_layers'])}")
    if int(candidate["final_rq_layers"]) != int(proto["final_rq_layers"]):
        changes.append(f"rq {proto['final_rq_layers']}->{int(candidate['final_rq_layers'])}")
    if not math.isclose(float(candidate["learning_rate"]), float(proto["learning_rate"])):
        changes.append(f"lr {_format_lr(float(proto['learning_rate']))}->{_format_lr(float(candidate['learning_rate']))}")
    if bool(candidate["use_mean_penalty"]) != bool(proto["use_mean_penalty"]):
        changes.append(f"mean/std {proto['ms']}->{candidate['ms']}")
    if bool(candidate["use_skew_penalty"]) != bool(proto["use_skew_penalty"]):
        changes.append(f"skew/kurt {proto['sk']}->{candidate['sk']}")
    return ", ".join(changes) if changes else "sin cambio estructural grande"


def _proposal_id(ranking_name: str, candidate: pd.Series) -> str:
    return (
        f"{RANKING_SHORT[ranking_name]}_"
        f"hf{int(candidate['hidden_features'])}_"
        f"l{int(candidate['num_layers'])}_"
        f"rq{int(candidate['final_rq_layers'])}_"
        f"lr{_format_lr(float(candidate['learning_rate']))}_"
        f"ms{candidate['ms']}_"
        f"sk{candidate['sk']}"
    )


def _proposal_reason(ranking_name: str, candidate: pd.Series, dominant_summary: str) -> str:
    changes = _changes_vs_prototype(candidate, str(candidate["prototype_signature"]))
    return (
        f"Sigue la zona dominante `{dominant_summary}` del ranking `{ranking_name}`; "
        f"toma como prototipo `{candidate['prototype_cfg_id']}` ({candidate['prototype_branch_id']}) y solo mueve "
        f"{changes} dentro del grid historico no entrenado."
    )


def _build_candidate_lists(
    pool_df: pd.DataFrame,
    ranking_top_df: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    lists: dict[str, pd.DataFrame] = {}
    for ranking_name in RANKING_ORDER:
        top_df = ranking_top_df[ranking_name].head(TOP_UNIQUE_OBSERVED).copy()
        dominant_summary = _dominant_axis_summary(top_df)
        rows: list[dict[str, Any]] = []
        for _, candidate in pool_df.iterrows():
            score, meta = _candidate_score(candidate, top_df, ranking_name)
            row = candidate.to_dict()
            row.update(meta)
            row["source_ranking"] = ranking_name
            row["candidate_score"] = float(score)
            row["dominant_summary"] = dominant_summary
            row["anchor_branch"] = str(meta["prototype_branch_id"])
            row["proposal_id"] = _proposal_id(ranking_name, candidate)
            row["proposal_reason"] = _proposal_reason(ranking_name, pd.Series(row), dominant_summary)
            rows.append(row)
        ranked = pd.DataFrame(rows).sort_values(["candidate_score", "cfg_signature"]).reset_index(drop=True)
        ranked["candidate_rank"] = range(1, len(ranked) + 1)
        lists[ranking_name] = ranked
    return lists


def _select_for_ranking(
    ranked_candidates: pd.DataFrame,
    *,
    already_used: set[str],
    desired_n: int,
    ranking_name: str,
) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    diversity_threshold = float(RANKING_SPECS[ranking_name]["diversity_threshold"])
    for _, row in ranked_candidates.iterrows():
        signature = str(row["cfg_signature"])
        if signature in already_used:
            continue
        if len(selected_rows) >= desired_n:
            break
        if len(selected_rows) < RELAXED_DIVERSITY_AFTER:
            too_close = False
            for prev in selected_rows:
                distance = _spec_distance(row.to_dict(), prev.to_dict())
                if distance < diversity_threshold:
                    too_close = True
                    break
            if too_close:
                continue
        selected_rows.append(row)
    if len(selected_rows) < desired_n:
        for _, row in ranked_candidates.iterrows():
            signature = str(row["cfg_signature"])
            if signature in already_used or any(str(prev["cfg_signature"]) == signature for prev in selected_rows):
                continue
            selected_rows.append(row)
            if len(selected_rows) >= desired_n:
                break
    if len(selected_rows) < desired_n:
        raise RuntimeError(f"Ranking {ranking_name} could only propose {len(selected_rows)} configs.")
    out = pd.DataFrame([row.to_dict() for row in selected_rows]).reset_index(drop=True)
    out["proposal_slot"] = range(1, len(out) + 1)
    return out


def _build_initial_proposals(candidate_lists: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ranking_name in RANKING_ORDER:
        selected = _select_for_ranking(
            candidate_lists[ranking_name],
            already_used=set(),
            desired_n=PROPOSALS_PER_RANKING,
            ranking_name=ranking_name,
        )
        rows.append(selected)
    initial = pd.concat(rows, ignore_index=True)
    dup_counts = initial["cfg_signature"].value_counts()
    initial["initial_duplicate_group_size"] = initial["cfg_signature"].map(dup_counts).fillna(1).astype(int)
    initial["initial_cross_ranking_duplicate"] = initial["initial_duplicate_group_size"] > 1
    initial["initial_duplicate_rankings"] = initial["cfg_signature"].map(
        initial.groupby("cfg_signature")["source_ranking"].apply(lambda s: "|".join(sorted(set(s))))
    )
    return initial.sort_values(["source_ranking", "proposal_slot"]).reset_index(drop=True)


def _build_deduped_proposals(candidate_lists: dict[str, pd.DataFrame], initial_df: pd.DataFrame) -> pd.DataFrame:
    used_signatures: set[str] = set()
    pointers = defaultdict(int)
    counts = defaultdict(int)
    selected_rows: list[dict[str, Any]] = []
    initial_lookup = {
        (str(row["source_ranking"]), str(row["cfg_signature"])): int(row["proposal_slot"])
        for _, row in initial_df.iterrows()
    }

    while any(counts[name] < PROPOSALS_PER_RANKING for name in RANKING_ORDER):
        progressed = False
        for ranking_name in RANKING_ORDER:
            if counts[ranking_name] >= PROPOSALS_PER_RANKING:
                continue
            ranked = candidate_lists[ranking_name]
            while pointers[ranking_name] < len(ranked):
                row = ranked.iloc[pointers[ranking_name]]
                pointers[ranking_name] += 1
                signature = str(row["cfg_signature"])
                if signature in used_signatures:
                    continue
                if counts[ranking_name] < RELAXED_DIVERSITY_AFTER:
                    prev_same = [r for r in selected_rows if r["source_ranking"] == ranking_name]
                    too_close = False
                    for prev in prev_same:
                        distance = _spec_distance(row.to_dict(), prev)
                        if distance < float(RANKING_SPECS[ranking_name]["diversity_threshold"]):
                            too_close = True
                            break
                    if too_close:
                        continue
                used_signatures.add(signature)
                counts[ranking_name] += 1
                out = row.to_dict()
                out["final_slot"] = counts[ranking_name]
                out["selected_candidate_rank"] = pointers[ranking_name]
                out["was_in_initial_top5"] = (ranking_name, signature) in initial_lookup
                out["initial_slot_if_any"] = initial_lookup.get((ranking_name, signature))
                out["resolved_via_alternative"] = not out["was_in_initial_top5"]
                selected_rows.append(out)
                progressed = True
                break
        if not progressed:
            raise RuntimeError("Could not finish global dedupe; no further unique candidates available.")

    deduped = pd.DataFrame(selected_rows).sort_values(["source_ranking", "final_slot"]).reset_index(drop=True)
    if deduped["cfg_signature"].nunique() != TOTAL_UNIQUE_PROPOSALS:
        raise RuntimeError(
            f"Expected {TOTAL_UNIQUE_PROPOSALS} unique configs, got {deduped['cfg_signature'].nunique()}."
        )
    return deduped


def _build_existing_configs_report(all_runs: pd.DataFrame, ranked_unique: dict[str, pd.DataFrame]) -> pd.DataFrame:
    runs = _with_signature_parts(all_runs[all_runs["cfg_signature"].notna()].copy())
    spec_cols = [
        "cfg_signature",
        "hidden_features",
        "num_layers",
        "affine_rq_ratio_str",
        "final_rq_layers",
        "learning_rate",
        "use_mean_penalty",
        "use_std_penalty",
        "use_skew_penalty",
        "use_kurtosis_penalty",
        "meanstd_mode",
        "skewkurt_mode",
        "complexity_score",
    ]
    grouped = runs.groupby(spec_cols, as_index=False).agg(
        observed_run_count=("run_id", "size"),
        observed_branches=("branch_id", _sorted_unique),
        observed_cfg_ids=("cfg_id", _sorted_unique),
        observed_campaigns=("campaign_origin", _sorted_unique),
        observed_phases=("phase", _sorted_unique),
        observed_seeds=("seed", lambda s: "|".join(str(int(x)) for x in sorted(set(s)))),
        observed_run_ids=("run_id", _sorted_unique),
        raw_acceptability_statuses=("raw_acceptability_status", _sorted_unique),
    )
    for ranking_name, ranked in ranked_unique.items():
        lens = (
            ranked[["cfg_signature", "unique_rank", RANKING_SPECS[ranking_name]["score_col"]]]
            .rename(
                columns={
                    "unique_rank": f"{ranking_name}_unique_rank",
                    RANKING_SPECS[ranking_name]["score_col"]: f"{ranking_name}_score",
                }
            )
            .groupby("cfg_signature", as_index=False)
            .first()
        )
        grouped = grouped.merge(lens, on="cfg_signature", how="left")
    return grouped.sort_values(["rrmse_primary_unique_rank", "mvn_unique_rank", "fair_unique_rank"]).reset_index(drop=True)


def _build_training_plan(deduped_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in deduped_df.iterrows():
        run_id = (
            f"flowprex3_{row['anchor_branch']}_tpv1_{row['proposal_id']}_seed{COMMON_EXPLORE_SEED}_v1"
        )
        config_path = CONFIG_ROOT / str(row["anchor_branch"]) / f"{row['proposal_id']}.yaml"
        rows.append(
            {
                "source_ranking": row["source_ranking"],
                "final_slot": int(row["final_slot"]),
                "proposal_id": row["proposal_id"],
                "cfg_signature": row["cfg_signature"],
                "anchor_branch": row["anchor_branch"],
                "prototype_branch_id": row["prototype_branch_id"],
                "prototype_cfg_id": row["prototype_cfg_id"],
                "seed": int(COMMON_EXPLORE_SEED),
                "planned_run_id": run_id,
                "planned_config_path": str(config_path),
                "planned_output_dir": str(OFFICIAL_FLOWPRE_ROOT / run_id),
                "contract_id": EXPLORE_CONTRACT_ID,
                "split_id": OFFICIAL_SPLIT_ID,
                "uses_test": False,
                "creates_promotion_manifest": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["source_ranking", "final_slot"]).reset_index(drop=True)


def _write_ranking_signal_analysis(
    *,
    all_runs: pd.DataFrame,
    ranked_unique: dict[str, pd.DataFrame],
    proposals_by_ranking: pd.DataFrame,
) -> Path:
    universe_unique = _with_signature_parts(all_runs.drop_duplicates(subset=["cfg_signature"]).copy())
    lines = [
        "# FlowPre V3 Ranking Signal Analysis",
        "",
        f"- Universo actual de runs oficiales completas: `{len(all_runs)}`.",
        f"- Firmas/configs ya observadas: `{all_runs['cfg_signature'].nunique()}`.",
        f"- Seed comun propuesta para `v3`: `{COMMON_EXPLORE_SEED}`.",
        "- Este analisis es operativo y provisional; prepara la siguiente ronda de FlowPre sin cerrar la fase.",
        "",
    ]

    for ranking_name in RANKING_ORDER:
        top_df = ranked_unique[ranking_name].head(TOP_UNIQUE_OBSERVED).copy()
        enrichment_rows = []
        for axis in ("hf", "layers", "rq", "lr", "ms", "sk"):
            enrichment_rows.extend(_enrichment_rows(top_df, universe_unique, axis))
        enrichment_df = pd.DataFrame(enrichment_rows)
        ranking_props = proposals_by_ranking[proposals_by_ranking["source_ranking"] == ranking_name].copy()

        lines.extend(
            [
                f"## {RANKING_TITLES[ranking_name]}",
                RANKING_DESCRIPTIONS[ranking_name],
                "",
                "### Top unico observado",
                _frame_to_markdown(top_df[RANKING_SPECS[ranking_name]["top_table_cols"]].reset_index(drop=True)),
                "",
                "### Hiperparametros que parecen ayudar",
                *_format_enrichment_bullets(enrichment_df, positive=True),
                "",
                "### Hiperparametros que parecen restar o verse fragiles",
                *_format_enrichment_bullets(enrichment_df, positive=False),
                "",
                "### Sinergias repetidas",
                *_pair_synergies(top_df),
                "",
                "### Zona nueva que merece la pena probar",
                f"- {top_df['hf'].value_counts().head(2).index.astype(str).tolist()} en `hidden_features`, "
                f"{top_df['layers'].value_counts().head(2).index.astype(str).tolist()} en `num_layers`, "
                f"{top_df['rq'].value_counts().head(2).index.astype(str).tolist()} en `final_rq_layers`, "
                f"con `lr` dominantes {top_df['lr'].value_counts().head(2).index.astype(str).tolist()}.",
                "",
                "### 5 propuestas iniciales de esta lente",
                _frame_to_markdown(
                    ranking_props[
                        [
                            "proposal_slot",
                            "proposal_id",
                            "cfg_signature",
                            "anchor_branch",
                            "prototype_branch_id",
                            "prototype_cfg_id",
                            "candidate_score",
                            "proposal_reason",
                        ]
                    ].reset_index(drop=True)
                ),
                "",
            ]
        )

    out_path = REPORT_ROOT / "flowpre_v3_ranking_signal_analysis.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_summary(
    *,
    inventory_df: pd.DataFrame,
    all_runs: pd.DataFrame,
    existing_configs_df: pd.DataFrame,
    candidate_pool_df: pd.DataFrame,
    proposals_by_ranking: pd.DataFrame,
    deduped_df: pd.DataFrame,
    training_plan_df: pd.DataFrame,
) -> Path:
    campaign_counts = inventory_df["campaign_origin"].value_counts().to_dict()
    initial_duplicates = proposals_by_ranking[proposals_by_ranking["initial_cross_ranking_duplicate"]].copy()
    duplicate_groups = (
        initial_duplicates.groupby("cfg_signature")["source_ranking"].apply(lambda s: "|".join(sorted(set(s)))).reset_index()
        if not initial_duplicates.empty
        else pd.DataFrame(columns=["cfg_signature", "source_ranking"])
    )
    final_counts = deduped_df["source_ranking"].value_counts().reindex(RANKING_ORDER, fill_value=0)

    lines = [
        "# FlowPre Explore V3 Summary",
        "",
        "## Estado de partida",
        f"- Runs oficiales observadas ahora: `{len(inventory_df)}`.",
        f"- Split por campana: `{campaign_counts}`.",
        f"- Firmas/configs ya entrenadas: `{existing_configs_df['cfg_signature'].nunique()}`.",
        f"- Pool historico no entrenado disponible para explorar: `{len(candidate_pool_df)}` configs.",
        f"- Seed comun fijada para toda la ronda `v3`: `{COMMON_EXPLORE_SEED}`.",
        "- Esta campana sigue siendo parte de una fase FlowPre abierta; no cierra la fase ni fija un ganador definitivo.",
        f"- Este summary debe leerse como snapshot de preparacion sobre el universo entonces vigente de `{len(inventory_df)}` runs oficiales.",
        "",
        "## Analisis y propuestas",
        "- Se han generado 5 propuestas iniciales por cada una de las 6 lentes operativas actuales.",
        f"- Despues se han deduplicado globalmente contra el universo oficial observado en ese momento (`{len(inventory_df)}` runs) y entre propuestas nuevas.",
        f"- Propuestas finales unicas listas para training: `{deduped_df['cfg_signature'].nunique()}`.",
        "",
        "### Reparto final por lente",
        _frame_to_markdown(
            pd.DataFrame(
                {
                    "source_ranking": list(final_counts.index),
                    "final_count": [int(x) for x in final_counts.tolist()],
                }
            )
        ),
        "",
    ]

    if duplicate_groups.empty:
        lines.extend(["### Duplicados iniciales entre lentes", "- No hubo duplicados iniciales entre las 30 propuestas conceptuales.", ""])
    else:
        lines.extend(
            [
                "### Duplicados iniciales entre lentes",
                _frame_to_markdown(duplicate_groups.rename(columns={"source_ranking": "duplicate_rankings"})),
                "",
            ]
        )

    lines.extend(
        [
            "## Training plan original de la fase de preparacion",
            "- En el momento de este dry-run todavia no se habia ejecutado `--train`.",
            "- Todas las configs finales siguen dentro del grid historico definido y no repiten ninguna firma ya entrenada.",
            "",
            _frame_to_markdown(
                training_plan_df[
                    ["source_ranking", "final_slot", "proposal_id", "cfg_signature", "anchor_branch", "planned_run_id"]
                ].reset_index(drop=True)
            ),
            "",
            "## Comando a ejecutar si se confirma la ronda",
            "```bash",
            "MPLCONFIGDIR=/tmp/codex-mpl .venv/bin/python scripts/f6_flowpre_explore_v3.py --train --device auto",
            "```",
            "",
        ]
    )

    out_path = REPORT_ROOT / "flowpre_v3_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_reports(
    *,
    existing_configs_df: pd.DataFrame,
    proposals_by_ranking: pd.DataFrame,
    deduped_df: pd.DataFrame,
    training_plan_df: pd.DataFrame,
    analysis_path: Path,
    summary_path: Path,
) -> dict[str, Path]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "existing_configs": REPORT_ROOT / "flowpre_v3_existing_configs.csv",
        "proposals_by_ranking": REPORT_ROOT / "flowpre_v3_proposals_by_ranking.csv",
        "proposals_deduped": REPORT_ROOT / "flowpre_v3_proposals_deduped.csv",
        "training_plan": REPORT_ROOT / "flowpre_v3_training_plan.csv",
        "analysis": analysis_path,
        "summary": summary_path,
    }
    existing_configs_df.to_csv(paths["existing_configs"], index=False)
    proposals_by_ranking.to_csv(paths["proposals_by_ranking"], index=False)
    deduped_df.to_csv(paths["proposals_deduped"], index=False)
    training_plan_df.to_csv(paths["training_plan"], index=False)
    return paths


def _write_train_configs(deduped_df: pd.DataFrame) -> None:
    for _, row in deduped_df.iterrows():
        spec = {
            "hidden_features": int(row["hidden_features"]),
            "num_layers": int(row["num_layers"]),
            "affine_rq_ratio": [int(x) for x in row["affine_rq_ratio"]],
            "final_rq_layers": int(row["final_rq_layers"]),
            "learning_rate": float(row["learning_rate"]),
            "use_mean_penalty": bool(row["use_mean_penalty"]),
            "use_std_penalty": bool(row["use_std_penalty"]),
            "use_skew_penalty": bool(row["use_skew_penalty"]),
            "use_kurtosis_penalty": bool(row["use_kurtosis_penalty"]),
        }
        config = _config_from_spec(str(row["anchor_branch"]), spec)
        config_path = CONFIG_ROOT / str(row["anchor_branch"]) / f"{row['proposal_id']}.yaml"
        write_yaml(config_path, config)


def _run_train_plan(deduped_df: pd.DataFrame, *, device: str, verbose: bool, force: bool) -> None:
    from training.train_flow_pre import train_flowpre_pipeline

    _write_train_configs(deduped_df)
    inventory_df = _build_inventory()
    existing_signatures = set(inventory_df["cfg_signature"].dropna().astype(str))
    for _, row in deduped_df.iterrows():
        cfg_signature = str(row["cfg_signature"])
        if cfg_signature in existing_signatures:
            continue

        seed = int(COMMON_EXPLORE_SEED)
        run_id = f"flowprex3_{row['anchor_branch']}_tpv1_{row['proposal_id']}_seed{seed}_v1"
        run_dir = OFFICIAL_FLOWPRE_ROOT / run_id
        results_path = run_dir / f"{run_id}_results.yaml"
        if results_path.exists() and not force:
            continue

        config_path = CONFIG_ROOT / str(row["anchor_branch"]) / f"{row['proposal_id']}.yaml"
        eval_ctx = {
            "dataset_name": DEFAULT_OFFICIAL_DATASET_NAME,
            "split_id": OFFICIAL_SPLIT_ID,
            "contract_id": EXPLORE_CONTRACT_ID,
            "seed_set_id": "f6_flowpre_explore_v3_screen",
            "base_config_id": f"flowpre_explore_v3_{row['anchor_branch']}",
            "objective_metric_id": f"flowpre_{row['anchor_branch']}_selection",
            "run_level_axes": {
                "campaign_id": "f6_explore_v3",
                "branch_id": str(row["anchor_branch"]),
                "cfg_id": str(row["proposal_id"]),
                "phase": "screen",
                "seed": seed,
                "source_ranking": str(row["source_ranking"]),
                "ranking_slot": int(row["final_slot"]),
            },
        }
        train_flowpre_pipeline(
            config_filename=str(config_path),
            base_name=run_id.removesuffix("_v1"),
            device=device,
            seed=seed,
            verbose=verbose,
            allow_test_holdout=False,
            evaluation_context=eval_ctx,
            output_namespace="official",
        )
        existing_signatures.add(cfg_signature)


def main() -> int:
    args = _parse_args()
    mode = _mode_from_args(args)

    inventory_df = _build_inventory()
    if inventory_df.empty:
        raise RuntimeError("No FlowPre official runs found under outputs/models/official/flow_pre.")

    all_runs = _build_current_all_runs(inventory_df)
    if len(all_runs) != 42:
        # v3 nacio sobre una foto concreta del inventario; si cambia, avisamos pero no bloqueamos.
        print(
            f"[flowpre_v3] aviso: el universo actual ya no coincide con la foto original de 42 runs; "
            f"ahora hay {len(all_runs)} runs oficiales."
        )

    existing_signatures = set(all_runs["cfg_signature"].dropna().astype(str))
    candidate_pool_df = _build_candidate_pool(existing_signatures)
    ranked_unique = {ranking: _ranked_unique_observed(all_runs, ranking) for ranking in RANKING_ORDER}
    candidate_lists = _build_candidate_lists(candidate_pool_df, ranked_unique)
    proposals_by_ranking = _build_initial_proposals(candidate_lists)
    deduped_df = _build_deduped_proposals(candidate_lists, proposals_by_ranking)
    training_plan_df = _build_training_plan(deduped_df)
    existing_configs_df = _build_existing_configs_report(all_runs, ranked_unique)

    analysis_path = _write_ranking_signal_analysis(
        all_runs=all_runs,
        ranked_unique=ranked_unique,
        proposals_by_ranking=proposals_by_ranking,
    )
    summary_path = _write_summary(
        inventory_df=inventory_df,
        all_runs=all_runs,
        existing_configs_df=existing_configs_df,
        candidate_pool_df=candidate_pool_df,
        proposals_by_ranking=proposals_by_ranking,
        deduped_df=deduped_df,
        training_plan_df=training_plan_df,
    )
    _write_reports(
        existing_configs_df=existing_configs_df,
        proposals_by_ranking=proposals_by_ranking,
        deduped_df=deduped_df,
        training_plan_df=training_plan_df,
        analysis_path=analysis_path,
        summary_path=summary_path,
    )

    if mode == "train":
        _run_train_plan(deduped_df, device=args.device, verbose=not args.quiet, force=args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
