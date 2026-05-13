from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.f6_reporting import FLOWPRE_ANCHORS, FLOWPRE_METRICS
from evaluation.f6_selection import FLOWPRE_BRANCHES, rank_flowpre_branch, summarize_flowpre_results
from scripts.f6_common import BUDGET_LEDGER_PATH, REPORTS_ROOT, sync_campaign_budget_ledger, write_json
from scripts.f6_flowpre_partial_audit import _build_inventory_rows, _branch_summary_rows, _discover_runs
from scripts.f6_flowpre_revalidate import COMMON_SEEDS, SCREENING_SEEDS


ALL_RUNS_PATH = REPORTS_ROOT / "flowpre_all_runs.csv"
AGGREGATE_PATH = REPORTS_ROOT / "flowpre_aggregate.csv"
SUMMARY_PATH = REPORTS_ROOT / "flowpre_revalidation_summary.csv"
COMPARISON_MD_PATH = REPORTS_ROOT / "comparison_summary.md"
SELECTION_STATUS_MD_PATH = REPORTS_ROOT / "flowpre_selection_status.md"

SUMMARY_COLUMNS = [
    "model_family",
    "branch_id",
    "anchor_run_id",
    "revalidated_run_id",
    "revalidated_cfg_id",
    "revalidated_phase",
    "revalidated_seed",
    "metric_name",
    "metric_scope",
    "historical_support_only",
    "anchor_val",
    "revalidated_val",
    "absolute_delta",
    "relative_delta_pct",
    "comparison_label",
    "selection_status",
    "selection_basis",
    "formal_selection_reconstructible",
    "promotion_manifest_available",
    "screening_complete",
    "screening_n_found",
    "reseed_n_found",
    "top2_screen_cfg_ids",
    "full_reseed_cfg_ids",
    "branch_runs_observed",
    "branch_runs_expected",
    "has_alert",
    "recommended_action",
    "status_for_next_phase",
    "notes",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _metric_value(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key, math.nan)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(numeric):
        return math.nan
    return numeric


def _flowpre_mvn_selection_score(summary: dict[str, Any]) -> float:
    target_mu = math.sqrt(43 - 0.5)
    weights = {
        "val_skew_abs": 1.0,
        "val_kurt_excess_abs": 1.0,
        "val_eigstd": 1.2,
        "val_mahal_mu": 0.3,
        "val_mahal_md": 0.3,
    }
    values = {
        "val_skew_abs": _metric_value(summary, "val_skew_abs"),
        "val_kurt_excess_abs": _metric_value(summary, "val_kurt_excess_abs"),
        "val_eigstd": _metric_value(summary, "val_eigstd"),
        "val_mahal_mu": abs(_metric_value(summary, "val_mahal_mu") - target_mu),
        "val_mahal_md": abs(_metric_value(summary, "val_mahal_md") - target_mu),
    }
    if any(not math.isfinite(v) for v in values.values()):
        return math.nan
    total_weight = float(sum(weights.values()))
    return float(sum(values[k] * weights[k] for k in weights) / total_weight)


def _augment_flowpre_summary(summary: dict[str, Any], branch_id: str) -> dict[str, Any]:
    out = dict(summary)
    if branch_id == "mvn":
        out["selection_score"] = _flowpre_mvn_selection_score(summary)
    return out


def _compare_values(anchor_val: float, revalidated_val: float) -> tuple[float, float, str]:
    if not math.isfinite(anchor_val) or not math.isfinite(revalidated_val):
        return math.nan, math.nan, "not_comparable"
    absolute_delta = revalidated_val - anchor_val
    denom = max(abs(anchor_val), 1e-8)
    relative_delta_pct = 100.0 * absolute_delta / denom
    if relative_delta_pct < -5.0:
        label = "mejora_clara"
    elif abs(relative_delta_pct) <= 5.0:
        label = "parecido_dentro_del_ruido"
    elif relative_delta_pct <= 15.0:
        label = "empeora_moderadamente"
    else:
        label = "empeora_fuerte_y_conviene_revisar"
    return absolute_delta, relative_delta_pct, label


def _read_config_fields(config_path: str | Path | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    cfg = _load_yaml(path)
    model = dict(cfg.get("model") or {})
    training = dict(cfg.get("training") or {})
    affine = model.get("affine_rq_ratio")
    return {
        "hidden_features": model.get("hidden_features"),
        "num_layers": model.get("num_layers"),
        "affine_rq_ratio": json.dumps(list(affine)) if isinstance(affine, (list, tuple)) else affine,
        "final_rq_layers": model.get("final_rq_layers"),
        "learning_rate": training.get("learning_rate"),
        "use_mean_penalty": training.get("use_mean_penalty"),
        "use_std_penalty": training.get("use_std_penalty"),
        "use_skew_penalty": training.get("use_skew_penalty"),
        "use_kurtosis_penalty": training.get("use_kurtosis_penalty"),
    }


def _build_all_runs_df(inventory_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if inventory_df.empty:
        return pd.DataFrame()

    for rec in inventory_df.to_dict("records"):
        results_path = rec.get("results_path")
        if not results_path:
            continue
        branch_id = str(rec.get("branch_id"))
        cfg_id = str(rec.get("cfg_id"))
        phase = str(rec.get("phase"))
        seed = int(rec.get("seed"))
        summary = _augment_flowpre_summary(
            summarize_flowpre_results(
                results_path,
                branch_id=branch_id,
                run_id=str(rec.get("run_id")),
                cfg_id=cfg_id,
                phase=phase,
                seed=seed,
            ),
            branch_id,
        )
        row = {
            **summary,
            "run_dir": rec.get("run_dir"),
            "results_path": results_path,
            "metrics_long_path": rec.get("metrics_long_path"),
            "run_manifest_path": rec.get("run_manifest_path"),
            "promotion_manifest_path": rec.get("promotion_manifest_path"),
            "config_path": rec.get("config_path"),
            "base_config_id": rec.get("base_config_id"),
            "objective_metric_id": rec.get("objective_metric_id"),
            "seed_set_id": rec.get("seed_set_id"),
            "comparison_group_id": rec.get("comparison_group_id"),
            "test_enabled": rec.get("test_enabled"),
            "has_promotion_manifest": bool(rec.get("has_promotion_manifest")),
            "is_complete_core": bool(rec.get("is_complete_core")),
            "is_complete_rrmse": bool(rec.get("is_complete_rrmse")),
        }
        row.update(_read_config_fields(rec.get("config_path")))
        rows.append(row)

    all_runs_df = pd.DataFrame(rows)
    if all_runs_df.empty:
        return all_runs_df
    sort_cols = ["branch_id", "phase", "cfg_id", "seed", "run_id"]
    return all_runs_df.sort_values(sort_cols).reset_index(drop=True)


def _screening_rankings(all_runs_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rankings: dict[str, pd.DataFrame] = {}
    for branch in FLOWPRE_BRANCHES:
        screening_seed = SCREENING_SEEDS[branch]
        screen_df = all_runs_df[
            (all_runs_df["branch_id"] == branch)
            & (all_runs_df["phase"] == "screen")
            & (all_runs_df["seed"] == screening_seed)
        ].copy()
        rankings[branch] = rank_flowpre_branch(screen_df, branch) if not screen_df.empty else pd.DataFrame()
    return rankings


def _full_seed_set_for_cfg(all_runs_df: pd.DataFrame, branch: str, cfg_id: str) -> set[int]:
    subset = all_runs_df[(all_runs_df["branch_id"] == branch) & (all_runs_df["cfg_id"] == cfg_id)]
    return {int(seed) for seed in subset["seed"].dropna().astype(int)}


def _aggregate_branch(
    branch: str,
    all_runs_df: pd.DataFrame,
    branch_status: dict[str, Any],
    screening_ranked: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    branch_df = all_runs_df[all_runs_df["branch_id"] == branch].copy()
    if branch_df.empty:
        return pd.DataFrame(), {}

    numeric_cols = [
        col for col in branch_df.columns if pd.api.types.is_numeric_dtype(branch_df[col]) and col not in {"seed"}
    ]
    agg_df = branch_df.groupby(["branch_id", "cfg_id"], as_index=False)[numeric_cols].mean()
    run_counts = branch_df.groupby(["branch_id", "cfg_id"]).size().rename("n_runs").reset_index()
    agg_df = agg_df.merge(run_counts, on=["branch_id", "cfg_id"], how="left")
    agg_df = rank_flowpre_branch(agg_df, branch)

    screening_top2 = screening_ranked.head(2)["cfg_id"].tolist() if not screening_ranked.empty else []
    full_reseed_cfgs: list[str] = []
    coverage_notes: list[str] = []
    for cfg_id in agg_df["cfg_id"].tolist():
        seeds_present = _full_seed_set_for_cfg(all_runs_df, branch, str(cfg_id))
        has_all_common_seeds = set(COMMON_SEEDS).issubset(seeds_present)
        if has_all_common_seeds:
            full_reseed_cfgs.append(str(cfg_id))
        if SCREENING_SEEDS[branch] not in seeds_present:
            coverage_status = "unexpected_missing_screen"
        elif has_all_common_seeds:
            coverage_status = "screen_plus_full_reseed"
        elif len(seeds_present) > 1:
            coverage_status = "screen_plus_partial_reseed"
        else:
            coverage_status = "screen_only"
        coverage_notes.append(coverage_status)
        agg_df.loc[agg_df["cfg_id"] == cfg_id, "observed_seed_ids"] = "|".join(str(seed) for seed in sorted(seeds_present))
        agg_df.loc[agg_df["cfg_id"] == cfg_id, "has_all_common_seeds"] = has_all_common_seeds
        agg_df.loc[agg_df["cfg_id"] == cfg_id, "coverage_status"] = coverage_status
        agg_df.loc[agg_df["cfg_id"] == cfg_id, "screen_is_top2"] = str(cfg_id) in screening_top2

    screen_rank_lookup = {
        str(row["cfg_id"]): int(row["branch_rank"])
        for row in screening_ranked.to_dict("records")
    }
    agg_df["screen_rank"] = agg_df["cfg_id"].map(lambda cfg_id: screen_rank_lookup.get(str(cfg_id)))
    agg_df["screening_complete_branch"] = bool(branch_status.get("screening_complete"))
    agg_df["top2_screen_cfg_ids"] = "|".join(str(cfg) for cfg in screening_top2)
    agg_df["full_reseed_cfg_ids"] = "|".join(full_reseed_cfgs)
    agg_df["formal_selection_reconstructible_branch"] = bool(
        branch_status.get("screening_complete") and len(screening_top2) == 2 and set(screening_top2).issubset(set(full_reseed_cfgs))
    )
    agg_df["aggregate_basis"] = "observed_runs_only"
    agg_df["is_branch_leader_observed"] = agg_df["branch_rank"] == 1

    observation: dict[str, Any] = {
        "screening_top2": screening_top2,
        "full_reseed_cfgs": full_reseed_cfgs,
        "formal_selection_reconstructible": bool(agg_df["formal_selection_reconstructible_branch"].iloc[0]),
        "coverage_notes": sorted(set(str(note) for note in coverage_notes)),
    }

    if agg_df.empty:
        return agg_df, observation

    winner_cfg = str(agg_df.iloc[0]["cfg_id"])
    winner_runs = rank_flowpre_branch(
        branch_df[branch_df["cfg_id"] == winner_cfg].copy(),
        branch,
    )
    best_run = winner_runs.iloc[0].to_dict()
    observation.update(
        {
            "winner_cfg": winner_cfg,
            "winner_run_id": str(best_run["run_id"]),
            "winner_seed": int(best_run["seed"]),
            "winner_phase": str(best_run["phase"]),
            "winner_results_path": str(
                branch_df.loc[branch_df["run_id"] == best_run["run_id"], "results_path"].iloc[0]
            ),
        }
    )
    return agg_df, observation


def _status_triplet(observation: dict[str, Any]) -> tuple[str, str, str, bool]:
    if not observation:
        return "sin_runs_observadas", "no_observation", "completar_artifacts", True
    if observation.get("formal_selection_reconstructible"):
        return (
            "branch_reconstructible_no_promotion",
            "screening_top2_plus_full_reseed",
            "promocion_formal_pendiente",
            True,
        )
    return (
        "partial_observation_only",
        "observed_runs_only",
        "completar_f6_flowpre_antes_de_promover",
        True,
    )


def _build_revalidation_summary_df(
    all_runs_df: pd.DataFrame,
    branch_summary_df: pd.DataFrame,
    branch_observations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summary_lookup = {
        str(row["branch_id"]): row for row in branch_summary_df.to_dict("records")
    }
    for branch in FLOWPRE_BRANCHES:
        observation = branch_observations.get(branch, {})
        branch_status = summary_lookup.get(branch, {})
        selection_status, selection_basis, recommended_action, has_alert = _status_triplet(observation)
        if not observation:
            rows.append(
                {
                    "model_family": "flowpre",
                    "branch_id": branch,
                    "anchor_run_id": FLOWPRE_ANCHORS[branch]["run_id"],
                    "revalidated_run_id": None,
                    "revalidated_cfg_id": None,
                    "revalidated_phase": None,
                    "revalidated_seed": None,
                    "metric_name": None,
                    "metric_scope": None,
                    "historical_support_only": None,
                    "anchor_val": None,
                    "revalidated_val": None,
                    "absolute_delta": None,
                    "relative_delta_pct": None,
                    "comparison_label": "not_available",
                    "selection_status": selection_status,
                    "selection_basis": selection_basis,
                    "formal_selection_reconstructible": False,
                    "promotion_manifest_available": False,
                    "screening_complete": bool(branch_status.get("screening_complete", False)),
                    "screening_n_found": int(branch_status.get("screening_n_found", 0)),
                    "reseed_n_found": int(branch_status.get("reseed_n_found", 0)),
                    "top2_screen_cfg_ids": "",
                    "full_reseed_cfg_ids": "",
                    "branch_runs_observed": int(branch_status.get("n_runs_discovered", 0)),
                    "branch_runs_expected": 0,
                    "has_alert": has_alert,
                    "recommended_action": recommended_action,
                    "status_for_next_phase": "block",
                    "notes": "No hay runs observadas para esta rama.",
                }
            )
            continue

        anchor_summary = _augment_flowpre_summary(
            summarize_flowpre_results(
                FLOWPRE_ANCHORS[branch]["results_path"],
                branch_id=branch,
                run_id=FLOWPRE_ANCHORS[branch]["run_id"],
                cfg_id="anchor",
                phase="historical_anchor",
                seed=0,
            ),
            branch,
        )
        observed_summary = _augment_flowpre_summary(
            summarize_flowpre_results(
                observation["winner_results_path"],
                branch_id=branch,
                run_id=observation["winner_run_id"],
                cfg_id=observation["winner_cfg"],
                phase=observation["winner_phase"],
                seed=int(observation["winner_seed"]),
            ),
            branch,
        )

        formal_selection_reconstructible = bool(observation.get("formal_selection_reconstructible", False))
        promotion_manifest_available = bool(
            all_runs_df.loc[all_runs_df["branch_id"] == branch, "has_promotion_manifest"].any()
        )
        status_for_next_phase = "caution" if formal_selection_reconstructible else "block"
        branch_runs_expected = int(branch_status.get("screening_n_found", 0) + branch_status.get("reseed_nominal_max_runs", 0))
        notes = (
            "La selección de rama puede reconstruirse desde screening completo y reseed completo de las top-2, "
            "pero no existe promotion manifest formal."
            if formal_selection_reconstructible
            else "La rama solo permite lectura observacional: faltan runs de reseed para sostener selección formal."
        )

        for metric_name, metric_scope, historical_support_only in FLOWPRE_METRICS[branch]:
            anchor_val = _metric_value(anchor_summary, metric_name)
            observed_val = _metric_value(observed_summary, metric_name)
            absolute_delta, relative_delta_pct, comparison_label = _compare_values(anchor_val, observed_val)
            rows.append(
                {
                    "model_family": "flowpre",
                    "branch_id": branch,
                    "anchor_run_id": FLOWPRE_ANCHORS[branch]["run_id"],
                    "revalidated_run_id": observation["winner_run_id"],
                    "revalidated_cfg_id": observation["winner_cfg"],
                    "revalidated_phase": observation["winner_phase"],
                    "revalidated_seed": int(observation["winner_seed"]),
                    "metric_name": metric_name,
                    "metric_scope": metric_scope,
                    "historical_support_only": historical_support_only,
                    "anchor_val": anchor_val,
                    "revalidated_val": observed_val,
                    "absolute_delta": absolute_delta,
                    "relative_delta_pct": relative_delta_pct,
                    "comparison_label": comparison_label,
                    "selection_status": selection_status,
                    "selection_basis": selection_basis,
                    "formal_selection_reconstructible": formal_selection_reconstructible,
                    "promotion_manifest_available": promotion_manifest_available,
                    "screening_complete": bool(branch_status.get("screening_complete", False)),
                    "screening_n_found": int(branch_status.get("screening_n_found", 0)),
                    "reseed_n_found": int(branch_status.get("reseed_n_found", 0)),
                    "top2_screen_cfg_ids": "|".join(str(cfg) for cfg in observation.get("screening_top2", [])),
                    "full_reseed_cfg_ids": "|".join(str(cfg) for cfg in observation.get("full_reseed_cfgs", [])),
                    "branch_runs_observed": int(branch_status.get("n_runs_discovered", 0)),
                    "branch_runs_expected": branch_runs_expected,
                    "has_alert": has_alert,
                    "recommended_action": recommended_action,
                    "status_for_next_phase": status_for_next_phase,
                    "notes": notes,
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _render_selection_status_md(
    branch_summary_df: pd.DataFrame,
    branch_observations: dict[str, dict[str, Any]],
) -> str:
    summary_lookup = {
        str(row["branch_id"]): row for row in branch_summary_df.to_dict("records")
    }
    lines = [
        "# FlowPre Selection Status",
        "",
        "- Este archivo documenta por qué no se genera `flowpre_selected.csv` como selección formal global en el estado actual.",
        "- Fuente de verdad: artefactos reales bajo `outputs/models/official/flow_pre`.",
        "",
    ]
    for branch in FLOWPRE_BRANCHES:
        branch_status = summary_lookup.get(branch, {})
        observation = branch_observations.get(branch, {})
        lines.append(f"## {branch}")
        if not observation:
            lines.append("")
            lines.append("- Sin runs observadas para esta rama.")
            lines.append("")
            continue
        lines.append("")
        lines.append(f"- Screening completo: `{bool(branch_status.get('screening_complete', False))}`")
        lines.append(f"- Top-2 de screening observadas: `{', '.join(observation.get('screening_top2', [])) or 'ninguna'}`")
        lines.append(f"- CFGs con seeds completas `{COMMON_SEEDS}`: `{', '.join(observation.get('full_reseed_cfgs', [])) or 'ninguna'}`")
        lines.append(f"- Winner observada por datos existentes: `{observation.get('winner_cfg')}` -> `{observation.get('winner_run_id')}`")
        if observation.get("formal_selection_reconstructible"):
            lines.append("- La rama es reconstruible de forma defendible a nivel local, pero no existe `promotion_manifest` formal.")
        else:
            lines.append("- La rama no es reconstruible formalmente porque faltan runs de reseed respecto a la lógica real de F6.")
        lines.append("")
    lines.append("## Decisión de reporting")
    lines.append("")
    lines.append("- No se escribe `flowpre_selected.csv` en esta fase para no aparentar un cierre formal inexistente.")
    lines.append("- `rrmse` sí tiene una winner de rama reconstruible localmente, pero sin promotion formal no se promueve a upstream canónico.")
    lines.append("- `mvn` y `fair` siguen sin base suficiente para selección formal de rama.")
    lines.append("")
    return "\n".join(lines)


def _branch_summary_line(summary_df: pd.DataFrame, branch: str) -> str:
    branch_df = summary_df[summary_df["branch_id"] == branch]
    if branch_df.empty:
        return f"- `{branch}`: sin runs observadas."
    row = branch_df.iloc[0]
    top2 = row["top2_screen_cfg_ids"] or "ninguna"
    full_reseed = row["full_reseed_cfg_ids"] or "ninguna"
    if bool(row["formal_selection_reconstructible"]):
        return (
            f"- `{branch}`: screening completo y top-2 reseedeadas de forma suficiente; "
            f"winner observada=`{row['revalidated_cfg_id']}` -> `{row['revalidated_run_id']}`, "
            f"pero sin `promotion_manifest`, así que no hay upstream formal. "
            f"Top-2 screening=`{top2}`; full reseed=`{full_reseed}`."
        )
    return (
        f"- `{branch}`: estado parcial; winner observada=`{row['revalidated_cfg_id']}` -> `{row['revalidated_run_id']}`, "
        f"pero la rama no soporta selección formal. "
        f"Top-2 screening=`{top2}`; full reseed=`{full_reseed}`."
    )


def _render_comparison_summary_md(
    summary_df: pd.DataFrame,
    budget_snapshot: dict[str, Any],
) -> str:
    counts = budget_snapshot.get("effective_counts") or {}
    limits = budget_snapshot.get("limits") or {}
    lines = [
        "# F6 Comparison Summary",
        "",
        "- Reporting state: `FlowPre partial / rebuilt from observed artifacts`",
        f"- Budget FlowPre: `{counts.get('flowpre', 0)}` / `{limits.get('flowpre', 28)}`",
        f"- Budget FlowGen: `{counts.get('flowgen', 0)}` / `{limits.get('flowgen', 12)}`",
        f"- Budget Total: `{counts.get('total', 0)}` / `{limits.get('total', 40)}`",
        "",
        "## FlowPre",
    ]
    for branch in FLOWPRE_BRANCHES:
        lines.append(_branch_summary_line(summary_df, branch))
    lines.extend(
        [
            "",
            "## FlowGen",
            "- `flowgen`: bloqueado. No existe todavía upstream `FlowPre rrmse` promovido formalmente.",
            "",
            "## Alerts",
            "- No hay `promotion_manifest` de `FlowPre` en `outputs/models/official/flow_pre`.",
            "- `rrmse` tiene winner observada reconstruible localmente, pero no promoción formal.",
            "- `mvn` y `fair` siguen en estado parcial para selección de rama.",
            "- `flowpre_selected.csv` no se escribe en esta fase; ver `flowpre_selection_status.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def _regularize_budget_snapshot(budget_snapshot: dict[str, Any]) -> dict[str, Any]:
    out = dict(budget_snapshot)
    effective_counts = dict(out.get("effective_counts") or {})
    out["ledger_counts"] = dict(effective_counts)
    out["consumed_from_manifests"] = dict(out.get("consumed_from_manifests") or effective_counts)
    out["recomputed_from_manifests"] = True
    write_json(BUDGET_LEDGER_PATH, out)
    return out


def main() -> int:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    runs, discovery_warnings = _discover_runs()
    inventory_df, inventory_warnings, _mapping_notes = _build_inventory_rows(runs)
    all_runs_df = _build_all_runs_df(inventory_df)
    branch_summary_df = _branch_summary_rows(inventory_df)
    screening_rankings = _screening_rankings(all_runs_df)

    branch_aggregates: list[pd.DataFrame] = []
    branch_observations: dict[str, dict[str, Any]] = {}
    for branch in FLOWPRE_BRANCHES:
        branch_status = next(
            (row for row in branch_summary_df.to_dict("records") if str(row["branch_id"]) == branch),
            {},
        )
        agg_df, observation = _aggregate_branch(branch, all_runs_df, branch_status, screening_rankings.get(branch, pd.DataFrame()))
        if not agg_df.empty:
            branch_aggregates.append(agg_df)
        branch_observations[branch] = observation

    aggregate_df = (
        pd.concat(branch_aggregates, axis=0, ignore_index=True).sort_values(["branch_id", "branch_rank"])
        if branch_aggregates
        else pd.DataFrame()
    )
    summary_df = _build_revalidation_summary_df(all_runs_df, branch_summary_df, branch_observations)
    budget_snapshot = _regularize_budget_snapshot(sync_campaign_budget_ledger())

    all_runs_df.to_csv(ALL_RUNS_PATH, index=False)
    aggregate_df.to_csv(AGGREGATE_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    COMPARISON_MD_PATH.write_text(
        _render_comparison_summary_md(summary_df, budget_snapshot),
        encoding="utf-8",
    )
    SELECTION_STATUS_MD_PATH.write_text(
        _render_selection_status_md(branch_summary_df, branch_observations),
        encoding="utf-8",
    )

    if discovery_warnings or inventory_warnings:
        print("Warnings detected during rebuild:")
        for warning in discovery_warnings + inventory_warnings:
            print(f"- {warning}")

    print(f"Wrote all runs: {ALL_RUNS_PATH}")
    print(f"Wrote aggregate: {AGGREGATE_PATH}")
    print(f"Wrote summary: {SUMMARY_PATH}")
    print(f"Wrote comparison markdown: {COMPARISON_MD_PATH}")
    print(f"Wrote selection status markdown: {SELECTION_STATUS_MD_PATH}")
    print("flowpre_selected.csv not written: no formal global selection can be defended from current artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
