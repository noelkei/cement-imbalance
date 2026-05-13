from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.f6_selection import FLOWPRE_BRANCHES, rank_flowpre_branch, summarize_flowpre_results
from scripts.f6_flowpre_revalidate import COMMON_SEEDS, SCREENING_SEEDS, _build_branch_candidates


FLOWPRE_ROOT = ROOT / "outputs" / "models" / "official" / "flow_pre"
REPORT_ROOT = ROOT / "outputs" / "reports" / "f6"

RUN_MANIFEST_SUFFIX = "_run_manifest.json"
RESULTS_SUFFIX = "_results.yaml"
METRICS_LONG_SUFFIX = "_metrics_long.csv"
PROMOTION_SUFFIX = "_promotion_manifest.json"

RUN_ID_RE = re.compile(
    r"^flowpre_(?P<branch_id>rrmse|mvn|fair)_tpv1_(?P<cfg_id>.+?)_seed(?P<seed>\d+)_v\d+$"
)

RRMSE_METRIC_MAPPING = {
    "train_mean_whole_rrmse": ("train", "rrmse_mean_whole"),
    "train_std_whole_rrmse": ("train", "rrmse_std_whole"),
    "val_mean_whole_rrmse": ("val", "rrmse_mean_whole"),
    "val_std_whole_rrmse": ("val", "rrmse_std_whole"),
}

METRICS_LONG_FILTERS = {
    "rrmse_mean_whole": {
        "metric_group": "isotropy",
        "metric_name": "rrmse_mean_whole",
        "metric_scope": "overall",
        "component": "z",
        "value_space": "latent",
    },
    "rrmse_std_whole": {
        "metric_group": "isotropy",
        "metric_name": "rrmse_std_whole",
        "metric_scope": "overall",
        "component": "z",
        "value_space": "latent",
    },
}


@dataclass
class RunArtifacts:
    run_id: str
    run_dir: Path
    run_manifest_path: Path | None = None
    results_path: Path | None = None
    metrics_long_path: Path | None = None
    promotion_manifest_path: Path | None = None


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _parse_run_id(run_id: str) -> dict[str, Any]:
    match = RUN_ID_RE.match(run_id)
    if not match:
        return {}
    return {
        "branch_id": match.group("branch_id"),
        "cfg_id": match.group("cfg_id"),
        "seed": int(match.group("seed")),
    }


def _discover_runs() -> tuple[list[RunArtifacts], list[str]]:
    runs: dict[str, RunArtifacts] = {}
    warnings: list[str] = []

    if not FLOWPRE_ROOT.exists():
        return [], [f"No existe la raíz esperada: {FLOWPRE_ROOT}"]

    for entry in sorted(FLOWPRE_ROOT.iterdir()):
        if entry.is_dir():
            runs.setdefault(entry.name, RunArtifacts(run_id=entry.name, run_dir=entry))

    for path in sorted(FLOWPRE_ROOT.rglob(f"*{RUN_MANIFEST_SUFFIX}")):
        run_id = path.name[: -len(RUN_MANIFEST_SUFFIX)]
        run = runs.setdefault(run_id, RunArtifacts(run_id=run_id, run_dir=path.parent))
        run.run_manifest_path = path
    for path in sorted(FLOWPRE_ROOT.rglob(f"*{RESULTS_SUFFIX}")):
        run_id = path.name[: -len(RESULTS_SUFFIX)]
        run = runs.setdefault(run_id, RunArtifacts(run_id=run_id, run_dir=path.parent))
        run.results_path = path
    for path in sorted(FLOWPRE_ROOT.rglob(f"*{METRICS_LONG_SUFFIX}")):
        run_id = path.name[: -len(METRICS_LONG_SUFFIX)]
        run = runs.setdefault(run_id, RunArtifacts(run_id=run_id, run_dir=path.parent))
        run.metrics_long_path = path
    for path in sorted(FLOWPRE_ROOT.rglob(f"*{PROMOTION_SUFFIX}")):
        run_id = path.name[: -len(PROMOTION_SUFFIX)]
        run = runs.setdefault(run_id, RunArtifacts(run_id=run_id, run_dir=path.parent))
        run.promotion_manifest_path = path

    for run_id, run in sorted(runs.items()):
        if run.run_dir.name != run_id:
            warnings.append(
                f"Run {run_id}: el directorio detectado {run.run_dir.name} no coincide con el run_id inferido."
            )

    return sorted(runs.values(), key=lambda item: item.run_id), warnings


def _extract_from_results(results_payload: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {key: None for key in RRMSE_METRIC_MAPPING}
    for out_key, (split, field) in RRMSE_METRIC_MAPPING.items():
        split_payload = results_payload.get(split) or {}
        value = split_payload.get(field)
        out[out_key] = float(value) if isinstance(value, (int, float)) else None
    return out


def _extract_from_metrics_long(metrics_path: Path | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {key: None for key in RRMSE_METRIC_MAPPING}
    if metrics_path is None or not metrics_path.exists():
        return out

    df = pd.read_csv(metrics_path)
    for out_key, (split, field) in RRMSE_METRIC_MAPPING.items():
        filters = METRICS_LONG_FILTERS[field]
        mask = (
            (df["split"] == split)
            & (df["metric_group"] == filters["metric_group"])
            & (df["metric_name"] == filters["metric_name"])
            & (df["metric_scope"] == filters["metric_scope"])
            & (df["component"] == filters["component"])
            & (df["value_space"] == filters["value_space"])
        )
        subset = df.loc[mask, "metric_value"]
        if len(subset) == 1 and pd.notna(subset.iloc[0]):
            out[out_key] = float(subset.iloc[0])
    return out


def _stringify_path(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _infer_axes(
    run_id: str,
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    parsed = _parse_run_id(run_id)
    axes = dict((run_manifest.get("run_level_axes") or {}))
    merged = {}
    for key in ("branch_id", "cfg_id", "seed", "phase"):
        value = axes.get(key)
        if value is None:
            value = parsed.get(key)
        merged[key] = value
    return merged


def _build_inventory_rows(runs: list[RunArtifacts]) -> tuple[pd.DataFrame, list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    mapping_notes: list[str] = []

    for run in runs:
        run_manifest = _load_json(run.run_manifest_path)
        results_payload = _load_yaml(run.results_path)
        axes = _infer_axes(run.run_id, run_manifest)

        from_results = _extract_from_results(results_payload)
        from_metrics_long = _extract_from_metrics_long(run.metrics_long_path)

        for metric_key in RRMSE_METRIC_MAPPING:
            a = from_results.get(metric_key)
            b = from_metrics_long.get(metric_key)
            if a is not None and b is not None and not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
                warnings.append(
                    f"Run {run.run_id}: discrepancia entre results.yaml y metrics_long.csv para {metric_key} ({a} vs {b})."
                )

        metrics = {
            metric_key: from_results.get(metric_key)
            if from_results.get(metric_key) is not None
            else from_metrics_long.get(metric_key)
            for metric_key in RRMSE_METRIC_MAPPING
        }

        parsed = _parse_run_id(run.run_id)
        manifest_run_id = run_manifest.get("run_id")
        if manifest_run_id is not None and str(manifest_run_id) != run.run_id:
            warnings.append(
                f"Run {run.run_id}: run_manifest.json declara run_id={manifest_run_id}, distinto del nombre del directorio."
            )

        missing_core_artifacts = [
            name
            for name, path in (
                ("run_manifest", run.run_manifest_path),
                ("results", run.results_path),
                ("metrics_long", run.metrics_long_path),
            )
            if path is None or not path.exists()
        ]

        missing_rrmse_fields = [key for key, value in metrics.items() if value is None]
        is_complete_core = not missing_core_artifacts
        is_complete_rrmse = is_complete_core and not missing_rrmse_fields
        total_rrmse_whole = (
            sum(float(metrics[key]) for key in RRMSE_METRIC_MAPPING)
            if is_complete_rrmse
            else None
        )

        rows.append(
            {
                "run_id": run.run_id,
                "branch_id": axes.get("branch_id"),
                "cfg_id": axes.get("cfg_id"),
                "seed": axes.get("seed"),
                "phase": axes.get("phase"),
                "phase_inferred_from_run_id": None if axes.get("phase") is not None else parsed.get("phase"),
                "run_dir": str(run.run_dir),
                "run_manifest_path": _stringify_path(run.run_manifest_path),
                "results_path": _stringify_path(run.results_path),
                "metrics_long_path": _stringify_path(run.metrics_long_path),
                "promotion_manifest_path": _stringify_path(run.promotion_manifest_path),
                "config_path": run_manifest.get("config_path"),
                "base_config_id": run_manifest.get("base_config_id"),
                "objective_metric_id": run_manifest.get("objective_metric_id"),
                "seed_set_id": run_manifest.get("seed_set_id"),
                "comparison_group_id": run_manifest.get("comparison_group_id"),
                "test_enabled": run_manifest.get("test_enabled"),
                "train_mean_whole_rrmse": metrics["train_mean_whole_rrmse"],
                "train_std_whole_rrmse": metrics["train_std_whole_rrmse"],
                "val_mean_whole_rrmse": metrics["val_mean_whole_rrmse"],
                "val_std_whole_rrmse": metrics["val_std_whole_rrmse"],
                "total_rrmse_whole": total_rrmse_whole,
                "has_run_manifest": run.run_manifest_path is not None,
                "has_results": run.results_path is not None,
                "has_metrics_long": run.metrics_long_path is not None,
                "has_promotion_manifest": run.promotion_manifest_path is not None,
                "is_complete_core": is_complete_core,
                "is_complete_rrmse": is_complete_rrmse,
                "missing_core_artifacts": "|".join(missing_core_artifacts) if missing_core_artifacts else "",
                "missing_rrmse_fields": "|".join(missing_rrmse_fields) if missing_rrmse_fields else "",
            }
        )

        if run.results_path is not None and run.metrics_long_path is not None and not mapping_notes:
            mapping_notes.extend(
                [
                    "Mapping real detectado en results.yaml: "
                    "train.rrmse_mean_whole, train.rrmse_std_whole, val.rrmse_mean_whole, val.rrmse_std_whole.",
                    "Mapping real detectado en metrics_long.csv para esas mismas métricas: "
                    "split=train|val, metric_group=isotropy, metric_name=rrmse_mean_whole|rrmse_std_whole, "
                    "metric_scope=overall, component=z, value_space=latent.",
                ]
            )

    df = pd.DataFrame(rows).sort_values(["branch_id", "phase", "cfg_id", "seed", "run_id"], na_position="last")
    return df, warnings, mapping_notes


def _branch_summary_rows(inventory_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_cfgs = {
        branch: [cfg_id for cfg_id, _ in _build_branch_candidates(branch)]
        for branch in FLOWPRE_BRANCHES
    }
    reseed_expected_seeds = {
        branch: sorted(seed for seed in COMMON_SEEDS if seed != SCREENING_SEEDS[branch])
        for branch in FLOWPRE_BRANCHES
    }

    for branch in FLOWPRE_BRANCHES:
        branch_df = inventory_df[inventory_df["branch_id"] == branch].copy()
        expected_screen_cfgs = expected_cfgs[branch]
        screen_seed = SCREENING_SEEDS[branch]
        screen_df = branch_df[(branch_df["phase"] == "screen") & (branch_df["seed"] == screen_seed)].copy()
        reseed_df = branch_df[branch_df["phase"] == "reseed"].copy()

        screen_cfgs_found = sorted(str(v) for v in screen_df["cfg_id"].dropna().unique())
        missing_cfgs = sorted(cfg for cfg in expected_screen_cfgs if cfg not in screen_cfgs_found)

        rows.append(
            {
                "branch_id": branch,
                "n_runs_discovered": int(len(branch_df)),
                "n_complete_core": int(branch_df["is_complete_core"].sum()) if not branch_df.empty else 0,
                "n_complete_rrmse": int(branch_df["is_complete_rrmse"].sum()) if not branch_df.empty else 0,
                "n_with_promotion_manifest": int(branch_df["has_promotion_manifest"].sum()) if not branch_df.empty else 0,
                "screening_seed_expected": screen_seed,
                "screening_expected_cfg_ids": "|".join(expected_screen_cfgs),
                "screening_found_cfg_ids": "|".join(screen_cfgs_found),
                "screening_missing_cfg_ids": "|".join(missing_cfgs),
                "screening_n_found": int(len(screen_df)),
                "screening_complete": len(missing_cfgs) == 0 and len(screen_df) == len(expected_screen_cfgs),
                "reseed_n_found": int(len(reseed_df)),
                "reseed_cfg_ids_found": "|".join(sorted(str(v) for v in reseed_df["cfg_id"].dropna().unique())),
                "reseed_seeds_expected": "|".join(str(v) for v in reseed_expected_seeds[branch]),
                "reseed_seeds_found": "|".join(str(v) for v in sorted(reseed_df["seed"].dropna().unique())),
                "reseed_started": bool(len(reseed_df) > 0),
                "reseed_seems_partial": bool(0 < len(reseed_df) < 4),
                "reseed_nominal_max_runs": 4,
            }
        )
    return pd.DataFrame(rows).sort_values("branch_id")


def _build_branch_snapshot(inventory_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    complete_results = inventory_df[inventory_df["is_complete_rrmse"] & inventory_df["results_path"].notna()].copy()

    for branch in FLOWPRE_BRANCHES:
        branch_df = complete_results[complete_results["branch_id"] == branch].copy()
        if branch_df.empty:
            rows.append(
                {
                    "branch_id": branch,
                    "best_run_id": None,
                    "best_cfg_id": None,
                    "best_seed": None,
                    "best_phase": None,
                    "best_val_mean_whole_rrmse": None,
                    "best_val_std_whole_rrmse": None,
                    "best_total_rrmse_whole": None,
                    "selection_basis": "no_complete_runs",
                }
            )
            continue

        summary_rows = []
        for rec in branch_df.to_dict("records"):
            summary = summarize_flowpre_results(
                rec["results_path"],
                branch_id=str(rec["branch_id"]),
                run_id=str(rec["run_id"]),
                cfg_id=str(rec["cfg_id"]),
                phase=str(rec["phase"]),
                seed=int(rec["seed"]),
            )
            summary["total_rrmse_whole"] = rec["total_rrmse_whole"]
            summary_rows.append(summary)

        ranked = rank_flowpre_branch(pd.DataFrame(summary_rows), branch)
        best = ranked.iloc[0].to_dict()
        rows.append(
            {
                "branch_id": branch,
                "best_run_id": best.get("run_id"),
                "best_cfg_id": best.get("cfg_id"),
                "best_seed": best.get("seed"),
                "best_phase": best.get("phase"),
                "best_val_mean_whole_rrmse": best.get("val_rrmse_mean"),
                "best_val_std_whole_rrmse": best.get("val_rrmse_std"),
                "best_total_rrmse_whole": best.get("total_rrmse_whole"),
                "selection_basis": "rank_flowpre_branch",
            }
        )
    return pd.DataFrame(rows).sort_values("branch_id")


def _build_rrmse_top3(inventory_df: pd.DataFrame) -> pd.DataFrame:
    rrmse_df = inventory_df[inventory_df["branch_id"] == "rrmse"].copy()
    complete_rrmse_df = rrmse_df[rrmse_df["is_complete_rrmse"]].copy()
    if complete_rrmse_df.empty:
        return complete_rrmse_df

    top3 = complete_rrmse_df.sort_values(
        ["total_rrmse_whole", "val_mean_whole_rrmse", "val_std_whole_rrmse", "train_mean_whole_rrmse"],
        na_position="last",
    ).head(3)
    top3 = top3.copy()
    top3["sum_breakdown"] = top3.apply(
        lambda row: (
            f"{row['total_rrmse_whole']:.12f} = "
            f"{row['train_mean_whole_rrmse']:.12f} + "
            f"{row['train_std_whole_rrmse']:.12f} + "
            f"{row['val_mean_whole_rrmse']:.12f} + "
            f"{row['val_std_whole_rrmse']:.12f}"
        ),
        axis=1,
    )
    return top3


def _fmt_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.6f}"


def _render_markdown(
    inventory_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    branch_snapshot_df: pd.DataFrame,
    top3_df: pd.DataFrame,
    warnings: list[str],
    discovery_warnings: list[str],
    mapping_notes: list[str],
) -> str:
    total_runs = len(inventory_df)
    complete_runs = int(inventory_df["is_complete_core"].sum()) if not inventory_df.empty else 0
    incomplete_runs = total_runs - complete_runs

    lines: list[str] = []
    lines.append("# FlowPre Partial Audit")
    lines.append("")
    lines.append("## Alcance")
    lines.append("")
    lines.append("- Auditoría parcial y estrictamente read-only sobre artefactos ya existentes en `outputs/models/official/flow_pre`.")
    lines.append("- No se usa `test`, no se usa `mtime` y no se infiere nada desde procesos vivos.")
    lines.append("- La raíz inspeccionada es `outputs/models/official/flow_pre`.")
    lines.append("")
    lines.append("## Mapping real de métricas")
    lines.append("")
    for note in mapping_notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Inventario resumido")
    lines.append("")
    lines.append(f"- Runs descubiertas: `{total_runs}`")
    lines.append(f"- Runs con core completo (`run_manifest + results + metrics_long`): `{complete_runs}`")
    lines.append(f"- Runs core incompletas: `{incomplete_runs}`")
    lines.append(f"- Promotions presentes: `{int(inventory_df['has_promotion_manifest'].sum()) if not inventory_df.empty else 0}`")
    lines.append("")
    lines.append("## Screening y reseed por rama")
    lines.append("")
    for row in summary_df.to_dict("records"):
        lines.append(
            f"- `{row['branch_id']}`: runs=`{row['n_runs_discovered']}`, screening_found=`{row['screening_n_found']}`, "
            f"screening_complete=`{row['screening_complete']}`, missing_cfgs=`{row['screening_missing_cfg_ids'] or 'ninguno'}`, "
            f"reseed_found=`{row['reseed_n_found']}`, reseed_started=`{row['reseed_started']}`, "
            f"reseed_seems_partial=`{row['reseed_seems_partial']}`"
        )
    lines.append("")
    lines.append("## Snapshot por variante")
    lines.append("")
    for row in branch_snapshot_df.to_dict("records"):
        if row["best_run_id"] is None:
            lines.append(f"- `{row['branch_id']}`: sin runs completas para snapshot.")
            continue
        lines.append(
            f"- `{row['branch_id']}`: mejor run observada=`{row['best_run_id']}`, cfg=`{row['best_cfg_id']}`, "
            f"seed=`{row['best_seed']}`, phase=`{row['best_phase']}`, "
            f"val_mean_whole_rrmse=`{_fmt_metric(row['best_val_mean_whole_rrmse'])}`, "
            f"val_std_whole_rrmse=`{_fmt_metric(row['best_val_std_whole_rrmse'])}`, "
            f"total_rrmse_whole=`{_fmt_metric(row['best_total_rrmse_whole'])}`"
        )
    lines.append("")
    lines.append("## Resumen rrmse")
    lines.append("")
    rrmse_df = inventory_df[inventory_df["branch_id"] == "rrmse"].copy()
    complete_rrmse_df = rrmse_df[rrmse_df["is_complete_rrmse"]].copy()
    incomplete_rrmse_df = rrmse_df[~rrmse_df["is_complete_rrmse"]].copy()
    lines.append(
        f"- Runs `rrmse` completas para ranking ad hoc: `{len(complete_rrmse_df)}`; incompletas excluidas: `{len(incomplete_rrmse_df)}`."
    )
    if not incomplete_rrmse_df.empty:
        for row in incomplete_rrmse_df.to_dict("records"):
            lines.append(
                f"- Incompleta `{row['run_id']}`: missing_core=`{row['missing_core_artifacts'] or 'none'}`, "
                f"missing_rrmse=`{row['missing_rrmse_fields'] or 'none'}`"
            )
    lines.append("")
    lines.append("## TOP 3 rrmse por total_rrmse_whole")
    lines.append("")
    if top3_df.empty:
        lines.append("- No hay suficientes runs completas de `rrmse` para construir TOP 3.")
    else:
        for _, row in top3_df.iterrows():
            lines.append(
                f"- `{row['run_id']}`: total_rrmse_whole=`{_fmt_metric(row['total_rrmse_whole'])}`, "
                f"train_mean_whole_rrmse=`{_fmt_metric(row['train_mean_whole_rrmse'])}`, "
                f"train_std_whole_rrmse=`{_fmt_metric(row['train_std_whole_rrmse'])}`, "
                f"val_mean_whole_rrmse=`{_fmt_metric(row['val_mean_whole_rrmse'])}`, "
                f"val_std_whole_rrmse=`{_fmt_metric(row['val_std_whole_rrmse'])}`"
            )
            lines.append(f"- Desglose `{row['run_id']}`: `{row['sum_breakdown']}`")
    lines.append("")
    lines.append("## Advertencias")
    lines.append("")
    if not discovery_warnings and not warnings:
        lines.append("- No se detectaron advertencias estructurales adicionales en los artefactos leídos.")
    else:
        for warning in discovery_warnings + warnings:
            lines.append(f"- {warning}")
    lines.append("")
    lines.append("## Notas")
    lines.append("")
    lines.append(
        "- El ranking principal pedido aquí es una auditoría parcial ad hoc sobre lo ya materializado; no sustituye al ranking canónico final de F6."
    )
    lines.append(
        "- La lectura actual sugiere screening completo para `fair` y `mvn`, y screening completo más reseed parcial en `rrmse`."
    )
    lines.append(
        "- No hay `promotion_manifest` todavía, así que la campaña parece no haber cerrado la promoción F6b en ninguna rama."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    runs, discovery_warnings = _discover_runs()
    inventory_df, inventory_warnings, mapping_notes = _build_inventory_rows(runs)
    summary_df = _branch_summary_rows(inventory_df)
    branch_snapshot_df = _build_branch_snapshot(inventory_df)
    top3_df = _build_rrmse_top3(inventory_df)

    inventory_path = REPORT_ROOT / "flowpre_partial_inventory.csv"
    summary_path = REPORT_ROOT / "flowpre_partial_summary_by_branch.csv"
    top3_path = REPORT_ROOT / "flowpre_rrmse_top3_partial.csv"
    markdown_path = REPORT_ROOT / "flowpre_partial_audit.md"

    inventory_df.to_csv(inventory_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    top3_df.to_csv(top3_path, index=False)
    markdown_path.write_text(
        _render_markdown(
            inventory_df=inventory_df,
            summary_df=summary_df,
            branch_snapshot_df=branch_snapshot_df,
            top3_df=top3_df,
            warnings=inventory_warnings,
            discovery_warnings=discovery_warnings,
            mapping_notes=mapping_notes,
        ),
        encoding="utf-8",
    )

    print(f"Wrote inventory: {inventory_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote top3: {top3_path}")
    print(f"Wrote markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
