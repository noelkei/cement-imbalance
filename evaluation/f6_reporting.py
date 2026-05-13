from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evaluation.f6_selection import summarize_flowgen_results, summarize_flowpre_results


ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = ROOT / "outputs" / "reports" / "f6"
OFFICIAL_MODELS_ROOT = ROOT / "outputs" / "models" / "official"
FLOWPRE_FINALISTS_ROOT = OFFICIAL_MODELS_ROOT / "flowpre_finalists"

FLOWPRE_ANCHORS = {
    "rrmse": {
        "run_id": "flow_pre_rrmse_r131_s5678_v1",
        "results_path": ROOT / "outputs" / "models" / "flow_pre" / "flow_pre_rrmse_r131_s5678_v1" / "flow_pre_rrmse_r131_s5678_v1_results.yaml",
    },
    "mvn": {
        "run_id": "flow_pre_mvn_r349_s9101_v1",
        "results_path": ROOT / "outputs" / "models" / "flow_pre" / "flow_pre_mvn_r349_s9101_v1" / "flow_pre_mvn_r349_s9101_v1_results.yaml",
    },
    "fair": {
        "run_id": "flow_pre_fair_r25_s1234_v1",
        "results_path": ROOT / "outputs" / "models" / "flow_pre" / "flow_pre_fair_r25_s1234_v1" / "flow_pre_fair_r25_s1234_v1_results.yaml",
    },
}

FLOWGEN_ANCHOR = {
    "run_id": "W_flowgen_seed2898_v1",
    "results_path": ROOT / "outputs" / "models" / "flowgen" / "W_flowgen_seed2898_v1" / "W_flowgen_seed2898_v1_results.yaml",
}

SUMMARY_COLUMNS = [
    "model_family",
    "branch_id",
    "anchor_run_id",
    "revalidated_run_id",
    "metric_name",
    "metric_scope",
    "historical_support_only",
    "anchor_val",
    "revalidated_val",
    "absolute_delta",
    "relative_delta_pct",
    "comparison_label",
    "has_alert",
    "recommended_action",
    "status_for_next_phase",
]

FLOWPRE_METRICS = {
    "rrmse": [
        ("val_rrmse_mean", "primary", False),
        ("val_rrmse_std", "primary", False),
        ("gap_val_train_sum", "secondary", False),
        ("val_rrmse_recon", "secondary", False),
        ("val_eigstd", "secondary", False),
    ],
    "mvn": [
        ("selection_score", "primary", True),
        ("val_eigstd", "primary", False),
        ("val_skew_abs", "primary", False),
        ("val_kurt_excess_abs", "primary", False),
        ("val_mahal_mu", "secondary", False),
        ("val_mahal_md", "secondary", False),
        ("gap_val_train_sum", "secondary", False),
    ],
    "fair": [
        ("val_pc_worst_mean", "primary", False),
        ("val_pc_worst_std", "primary", False),
        ("val_pc_wavg_mean", "secondary", False),
        ("val_rrmse_mean", "secondary", False),
        ("val_eigstd", "secondary", False),
    ],
}

FLOWGEN_METRICS = [
    ("val_realism_w1_mean", "primary", False),
    ("val_realism_ks_mean", "primary", False),
    ("val_realism_mmd2_rvs", "primary", False),
    ("val_realism_worst_class_w1_mean", "primary", False),
    ("val_realism_worst_class_ks_mean", "primary", False),
    ("val_loss_rrmse_x_mean_whole", "secondary", False),
    ("val_loss_rrmse_y_mean_whole", "secondary", False),
    ("val_eigstd", "secondary", False),
    ("val_rrmse_x_recon", "secondary", False),
    ("val_rrmse_y_recon", "secondary", False),
]


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _write_markdown(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _empty_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SUMMARY_COLUMNS)


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


def _run_manifest_to_results(run_manifest_path: Path) -> tuple[str, Path]:
    manifest = _load_json(run_manifest_path)
    run_id = str(manifest.get("run_id") or run_manifest_path.parent.name)
    return run_id, run_manifest_path.parent / f"{run_id}_results.yaml"


def _load_selected_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _validate_flowpre_selected_row(row: pd.Series, branch_id: str) -> dict[str, Any] | None:
    promotion_manifest_path = Path(str(row.get("promotion_manifest_path", "")))
    if not promotion_manifest_path.exists():
        return None
    promotion = _load_json(promotion_manifest_path)
    if str(promotion.get("model_family")) != "flowpre":
        return None
    if str(promotion.get("branch_id")) != branch_id:
        return None
    run_manifest_path = Path(str(promotion.get("source_run_manifest", "")))
    if not run_manifest_path.exists():
        return None
    run_id, results_path = _run_manifest_to_results(run_manifest_path)
    if str(row.get("run_id")) != run_id:
        return None
    if not results_path.exists():
        return None
    return {
        "run_id": run_id,
        "results_path": results_path,
        "promotion_manifest_path": promotion_manifest_path,
    }


def _validate_flowgen_selected_row(row: pd.Series) -> dict[str, Any] | None:
    promotion_manifest_path = Path(str(row.get("promotion_manifest_path", "")))
    if not promotion_manifest_path.exists():
        return None
    promotion = _load_json(promotion_manifest_path)
    if str(promotion.get("model_family")) != "flowgen":
        return None
    run_manifest_path = Path(str(promotion.get("source_run_manifest", "")))
    if not run_manifest_path.exists():
        return None
    run_id, results_path = _run_manifest_to_results(run_manifest_path)
    if str(row.get("run_id")) != run_id:
        return None
    if not results_path.exists():
        return None
    branch_id = str(
        promotion.get("branch_id")
        or promotion.get("flowgen_work_base_id")
        or row.get("flowgen_work_base_id")
        or row.get("branch_id")
        or "unknown"
    )
    return {
        "run_id": run_id,
        "results_path": results_path,
        "promotion_manifest_path": promotion_manifest_path,
        "branch_id": branch_id,
    }


def _resolve_unique_flowpre_promotion(branch_id: str) -> dict[str, Any] | None:
    finalists_root = FLOWPRE_FINALISTS_ROOT / branch_id
    finalist_candidates = sorted(finalists_root.glob("*/*_promotion_manifest.json")) if finalists_root.exists() else []
    if len(finalist_candidates) == 1:
        chosen_path = finalist_candidates[0]
    else:
        candidates: list[Path] = []
        family_root = OFFICIAL_MODELS_ROOT / "flow_pre"
        if not family_root.exists():
            return None
        for path in sorted(family_root.glob("*/*_promotion_manifest.json")):
            payload = _load_json(path)
            if str(payload.get("model_family")) != "flowpre":
                continue
            if str(payload.get("branch_id")) != branch_id:
                continue
            candidates.append(path)
        if len(candidates) != 1:
            return None
        chosen_path = candidates[0]

    promotion = _load_json(chosen_path)
    run_manifest_path = Path(str(promotion.get("source_run_manifest", "")))
    if not run_manifest_path.exists():
        return None
    run_id, results_path = _run_manifest_to_results(run_manifest_path)
    if not results_path.exists():
        return None
    return {
        "run_id": run_id,
        "results_path": results_path,
        "promotion_manifest_path": chosen_path,
    }


def _resolve_unique_flowgen_promotion() -> dict[str, Any] | None:
    family_root = OFFICIAL_MODELS_ROOT / "flowgen"
    if not family_root.exists():
        return None
    candidates = sorted(family_root.glob("*/*_promotion_manifest.json"))
    valid: list[Path] = []
    for path in candidates:
        payload = _load_json(path)
        if str(payload.get("model_family")) != "flowgen":
            continue
        valid.append(path)
    if len(valid) != 1:
        return None
    promotion = _load_json(valid[0])
    run_manifest_path = Path(str(promotion.get("source_run_manifest", "")))
    if not run_manifest_path.exists():
        return None
    run_id, results_path = _run_manifest_to_results(run_manifest_path)
    if not results_path.exists():
        return None
    return {
        "run_id": run_id,
        "results_path": results_path,
        "promotion_manifest_path": valid[0],
        "branch_id": str(promotion.get("branch_id") or promotion.get("flowgen_work_base_id") or "unknown"),
    }


def _resolve_flowpre_formal_selections() -> dict[str, dict[str, Any] | None]:
    selections: dict[str, dict[str, Any] | None] = {}
    selected_df = _load_selected_csv(REPORTS_ROOT / "flowpre_selected.csv")
    for branch_id in ("rrmse", "mvn", "fair"):
        branch_df = selected_df[selected_df["branch_id"] == branch_id].copy() if not selected_df.empty else pd.DataFrame()
        resolved = None
        if len(branch_df) == 1:
            resolved = _validate_flowpre_selected_row(branch_df.iloc[0], branch_id)
        if resolved is None:
            resolved = _resolve_unique_flowpre_promotion(branch_id)
        selections[branch_id] = resolved
    return selections


def _resolve_flowgen_formal_selection() -> dict[str, Any] | None:
    selected_df = _load_selected_csv(REPORTS_ROOT / "flowgen_selected.csv")
    if len(selected_df) == 1:
        resolved = _validate_flowgen_selected_row(selected_df.iloc[0])
        if resolved is not None:
            return resolved
    return _resolve_unique_flowgen_promotion()


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


def _status_from_metric_labels(labels: list[str]) -> tuple[str, str, bool]:
    n_moderate = sum(label == "empeora_moderadamente" for label in labels)
    n_strong = sum(label == "empeora_fuerte_y_conviene_revisar" for label in labels)
    if n_strong > 0 or n_moderate >= 2:
        return "block", "parar_antes_de_usar_upstream", True
    if n_moderate == 1:
        return "caution", "revisar_vecindario", True
    return "ok", "seguir_igual", False


def _build_flowpre_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    formal_selections = _resolve_flowpre_formal_selections()
    for branch_id, selection in formal_selections.items():
        if selection is None:
            continue
        anchor = FLOWPRE_ANCHORS[branch_id]
        anchor_summary = _augment_flowpre_summary(
            summarize_flowpre_results(
                anchor["results_path"],
                branch_id=branch_id,
                run_id=anchor["run_id"],
                cfg_id="anchor",
                phase="historical_anchor",
                seed=0,
            ),
            branch_id,
        )
        new_summary = _augment_flowpre_summary(
            summarize_flowpre_results(
                selection["results_path"],
                branch_id=branch_id,
                run_id=selection["run_id"],
                cfg_id="selected",
                phase="revalidated",
                seed=0,
            ),
            branch_id,
        )

        primary_labels: list[str] = []
        metric_rows: list[dict[str, Any]] = []
        for metric_name, metric_scope, historical_support_only in FLOWPRE_METRICS[branch_id]:
            anchor_val = _metric_value(anchor_summary, metric_name)
            new_val = _metric_value(new_summary, metric_name)
            absolute_delta, relative_delta_pct, comparison_label = _compare_values(anchor_val, new_val)
            if metric_scope == "primary" and comparison_label != "not_comparable":
                primary_labels.append(comparison_label)
            metric_rows.append(
                {
                    "model_family": "flowpre",
                    "branch_id": branch_id,
                    "anchor_run_id": anchor["run_id"],
                    "revalidated_run_id": selection["run_id"],
                    "metric_name": metric_name,
                    "metric_scope": metric_scope,
                    "historical_support_only": historical_support_only,
                    "anchor_val": anchor_val,
                    "revalidated_val": new_val,
                    "absolute_delta": absolute_delta,
                    "relative_delta_pct": relative_delta_pct,
                    "comparison_label": comparison_label,
                }
            )

        status_for_next_phase, recommended_action, has_alert = _status_from_metric_labels(primary_labels)
        for row in metric_rows:
            row["has_alert"] = has_alert
            row["recommended_action"] = recommended_action
            row["status_for_next_phase"] = status_for_next_phase
        rows.extend(metric_rows)
    if not rows:
        return _empty_summary_df()
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _build_flowgen_summary() -> pd.DataFrame:
    selection = _resolve_flowgen_formal_selection()
    if selection is None:
        return _empty_summary_df()

    anchor_summary = summarize_flowgen_results(
        FLOWGEN_ANCHOR["results_path"],
        run_id=FLOWGEN_ANCHOR["run_id"],
        cfg_id="anchor",
        phase="historical_anchor",
        seed=0,
    )
    new_summary = summarize_flowgen_results(
        selection["results_path"],
        run_id=selection["run_id"],
        cfg_id="selected",
        phase="revalidated",
        seed=0,
    )

    primary_labels: list[str] = []
    rows: list[dict[str, Any]] = []
    branch_id = str(selection.get("branch_id") or "unknown")
    for metric_name, metric_scope, historical_support_only in FLOWGEN_METRICS:
        anchor_val = _metric_value(anchor_summary, metric_name)
        new_val = _metric_value(new_summary, metric_name)
        absolute_delta, relative_delta_pct, comparison_label = _compare_values(anchor_val, new_val)
        if metric_scope == "primary" and comparison_label != "not_comparable":
            primary_labels.append(comparison_label)
        rows.append(
            {
                "model_family": "flowgen",
                "branch_id": branch_id,
                "anchor_run_id": FLOWGEN_ANCHOR["run_id"],
                "revalidated_run_id": selection["run_id"],
                "metric_name": metric_name,
                "metric_scope": metric_scope,
                "historical_support_only": historical_support_only,
                "anchor_val": anchor_val,
                "revalidated_val": new_val,
                "absolute_delta": absolute_delta,
                "relative_delta_pct": relative_delta_pct,
                "comparison_label": comparison_label,
            }
        )

    status_for_next_phase, recommended_action, has_alert = _status_from_metric_labels(primary_labels)
    for row in rows:
        row["has_alert"] = has_alert
        row["recommended_action"] = recommended_action
        row["status_for_next_phase"] = status_for_next_phase
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _branch_status_line(df: pd.DataFrame, branch_id: str) -> str:
    branch_df = df[df["branch_id"] == branch_id]
    if branch_df.empty:
        return f"- `{branch_id}`: pending"
    first = branch_df.iloc[0]
    return (
        f"- `{branch_id}`: `{first['status_for_next_phase']}` "
        f"({first['recommended_action']}) against `{first['anchor_run_id']}` -> `{first['revalidated_run_id']}`"
    )


def _flowgen_status_line(df: pd.DataFrame) -> str:
    if df.empty:
        return "- `flowgen`: pending"
    first = df.iloc[0]
    return (
        f"- `flowgen`: `{first['status_for_next_phase']}` "
        f"({first['recommended_action']}) against `{first['anchor_run_id']}` -> `{first['revalidated_run_id']}`"
    )


def _reporting_state(flowpre_df: pd.DataFrame, flowgen_df: pd.DataFrame) -> str:
    flowpre_branches_present = {str(branch) for branch in flowpre_df["branch_id"].unique()} if not flowpre_df.empty else set()
    flowpre_complete = flowpre_branches_present == {"rrmse", "mvn", "fair"}
    if not flowpre_branches_present and flowgen_df.empty:
        return "no revalidated runs yet"
    if not flowpre_complete:
        return "FlowPre partial"
    if flowgen_df.empty:
        return "FlowPre complete / FlowGen pending"
    return "FlowPre complete / FlowGen available"


def _render_markdown(flowpre_df: pd.DataFrame, flowgen_df: pd.DataFrame, *, budget_snapshot: dict[str, Any] | None) -> str:
    state = _reporting_state(flowpre_df, flowgen_df)
    lines = [
        "# F6 Comparison Summary",
        "",
        f"- Reporting state: `{state}`",
    ]
    if budget_snapshot:
        counts = budget_snapshot.get("effective_counts") or {}
        limits = budget_snapshot.get("limits") or {}
        lines.extend(
            [
                f"- Budget FlowPre: `{counts.get('flowpre', 0)}` / `{limits.get('flowpre', 28)}`",
                f"- Budget FlowGen: `{counts.get('flowgen', 0)}` / `{limits.get('flowgen', 12)}`",
                f"- Budget Total: `{counts.get('total', 0)}` / `{limits.get('total', 40)}`",
            ]
        )

    lines.extend(["", "## FlowPre"])
    for branch_id in ("rrmse", "mvn", "fair"):
        lines.append(_branch_status_line(flowpre_df, branch_id))

    lines.extend(["", "## FlowGen", _flowgen_status_line(flowgen_df), "", "## Alerts"])
    alert_lines: list[str] = []
    for df in (flowpre_df, flowgen_df):
        if df.empty:
            continue
        alerted = df[df["has_alert"] == True]  # noqa: E712
        for _, row in alerted.drop_duplicates(subset=["model_family", "branch_id"]).iterrows():
            alert_lines.append(
                f"- `{row['model_family']}:{row['branch_id']}` => `{row['status_for_next_phase']}` "
                f"({row['recommended_action']})"
            )
    if not alert_lines:
        alert_lines.append("- No active alerts from available comparisons.")
    lines.extend(alert_lines)
    lines.append("")
    return "\n".join(lines)


def write_f6_reports(*, budget_snapshot: dict[str, Any] | None = None) -> dict[str, str]:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    flowpre_df = _build_flowpre_summary()
    flowgen_df = _build_flowgen_summary()

    flowpre_path = REPORTS_ROOT / "flowpre_revalidation_summary.csv"
    flowgen_path = REPORTS_ROOT / "flowgen_revalidation_summary.csv"
    markdown_path = REPORTS_ROOT / "comparison_summary.md"

    flowpre_df.to_csv(flowpre_path, index=False)
    flowgen_df.to_csv(flowgen_path, index=False)
    _write_markdown(markdown_path, _render_markdown(flowpre_df, flowgen_df, budget_snapshot=budget_snapshot))

    return {
        "flowpre_summary_path": str(flowpre_path),
        "flowgen_summary_path": str(flowgen_path),
        "comparison_markdown_path": str(markdown_path),
    }
