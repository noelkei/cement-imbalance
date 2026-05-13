from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sets import DEFAULT_OFFICIAL_DATASET_NAME
from scripts.f6_common import write_yaml
from scripts.f6_flowpre_evaluate_current import _build_current_all_runs, _build_inventory
from scripts.f6_flowpre_explore_v2 import _config_from_spec, _config_signature, _format_lr


OFFICIAL_FLOWPRE_ROOT = ROOT / "outputs" / "models" / "official" / "flow_pre"
REPORT_ROOT = ROOT / "outputs" / "reports" / "f6_explore_v4"
CONFIG_ROOT = REPORT_ROOT / "configs" / "flowpre"
EXPLORE_CONTRACT_ID = "f6_flowpre_explore_v4"
OFFICIAL_SPLIT_ID = "init_temporal_processed_v1"
COMMON_EXPLORE_SEED = 5678
V2_COMMON_SEED = 5678
V3_COMMON_SEED = 5678

SIGNATURE_RE = re.compile(
    r"hf(?P<hf>\d+)\|l(?P<layers>\d+)\|rq1x(?P<rq>\d+)\|frq(?P<frq>\d+)\|lr(?P<lr>[^|]+)\|ms(?P<ms>on|off)\|sk(?P<sk>on|off)"
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plan or train the explicit FlowPre exploration round v4.")
    ap.add_argument("--dry-run", action="store_true", help="Write the v4 planning reports only.")
    ap.add_argument("--train", action="store_true", help="Train the planned v4 configs after writing the reports.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true", help="Re-run planned v4 runs even if they already exist.")
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


def _parse_signature(signature: str) -> dict[str, Any]:
    match = SIGNATURE_RE.fullmatch(str(signature))
    if not match:
        raise RuntimeError(f"Unparseable cfg_signature: {signature}")
    groups = match.groupdict()
    lr_token = str(groups["lr"])
    lr_value = {"1e-3": 1e-3, "1e-4": 1e-4, "1e-5": 1e-5}[lr_token]
    meanstd_on = groups["ms"] == "on"
    skewkurt_on = groups["sk"] == "on"
    spec = {
        "hidden_features": int(groups["hf"]),
        "num_layers": int(groups["layers"]),
        "affine_rq_ratio": [1, int(groups["rq"])],
        "final_rq_layers": int(groups["frq"]),
        "learning_rate": float(lr_value),
        "use_mean_penalty": bool(meanstd_on),
        "use_std_penalty": bool(meanstd_on),
        "use_skew_penalty": bool(skewkurt_on),
        "use_kurtosis_penalty": bool(skewkurt_on),
    }
    if _config_signature(spec) != signature:
        raise RuntimeError(f"Signature roundtrip mismatch for {signature}")
    spec.update(
        {
            "hf": str(groups["hf"]),
            "layers": str(groups["layers"]),
            "rq": str(groups["rq"]),
            "frq": str(groups["frq"]),
            "lr": lr_token,
            "ms": groups["ms"],
            "sk": groups["sk"],
        }
    )
    return spec


def _proposal_id_from_signature(signature: str) -> str:
    parsed = _parse_signature(signature)
    return (
        f"hf{parsed['hf']}_"
        f"l{parsed['layers']}_"
        f"rq{parsed['rq']}_"
        f"lr{parsed['lr']}_"
        f"ms{parsed['ms']}_"
        f"sk{parsed['sk']}"
    )


def _ensure_proposal_id(df: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    ensured = df.copy()
    if "proposal_id" not in ensured.columns:
        if "cfg_signature" in ensured.columns:
            ensured["proposal_id"] = ensured["cfg_signature"].astype(str).map(_proposal_id_from_signature)
        else:
            ensured["proposal_id"] = [f"p{idx:03d}" for idx in range(1, len(ensured) + 1)]
    else:
        missing = ensured["proposal_id"].isna() | (ensured["proposal_id"].astype(str).str.strip() == "")
        if missing.any():
            if "cfg_signature" in ensured.columns:
                ensured.loc[missing, "proposal_id"] = (
                    ensured.loc[missing, "cfg_signature"].astype(str).map(_proposal_id_from_signature)
                )
            else:
                fill_values = [f"p{idx:03d}" for idx in range(1, int(missing.sum()) + 1)]
                ensured.loc[missing, "proposal_id"] = fill_values
    if "proposal_id" not in ensured.columns:
        raise RuntimeError(f"{source_name} no contiene proposal_id y no se pudo reconstruir.")
    return ensured


def _proposal_entry(
    *,
    raw_order: int,
    source_section: str,
    proposal_group: str,
    anchor_branch: str,
    signature: str,
    reason: str,
) -> dict[str, Any]:
    spec = _parse_signature(signature)
    return {
        "raw_order": int(raw_order),
        "source_section": str(source_section),
        "proposal_group": str(proposal_group),
        "anchor_branch": str(anchor_branch),
        "cfg_signature": str(signature),
        "proposal_id": _proposal_id_from_signature(signature),
        "proposal_reason": str(reason),
        **spec,
    }


RAW_PROPOSALS: list[dict[str, Any]] = [
    _proposal_entry(
        raw_order=1,
        source_section="analysis_core",
        proposal_group="core",
        anchor_branch="rrmse",
        signature="hf256|l2|rq1x5|frq5|lr1e-3|mson|skoff",
        reason="Crossover principal entre el polo shallow rq3 y el polo deep rq5 de las lentes FlowGen-oriented.",
    ),
    _proposal_entry(
        raw_order=2,
        source_section="analysis_core",
        proposal_group="core",
        anchor_branch="rrmse",
        signature="hf192|l2|rq1x3|frq3|lr1e-3|mson|skoff",
        reason="Vecino compacto de las mejores familias rrmse/upstream con menos anchura y misma regularizacion mean/std.",
    ),
    _proposal_entry(
        raw_order=3,
        source_section="analysis_core",
        proposal_group="core",
        anchor_branch="rrmse",
        signature="hf192|l2|rq1x6|frq6|lr1e-4|mson|skoff",
        reason="Vecino directo del top actual de rrmse_primary para tensionar profundidad corta con rq6 y lr1e-4.",
    ),
    _proposal_entry(
        raw_order=4,
        source_section="analysis_core",
        proposal_group="core",
        anchor_branch="rrmse",
        signature="hf256|l2|rq1x6|frq6|lr1e-3|mson|skoff",
        reason="Extiende la esquina shallow fuerte de upstream hacia rq6 sin abandonar la zona lr1e-3 dominante.",
    ),
    _proposal_entry(
        raw_order=5,
        source_section="analysis_core",
        proposal_group="core",
        anchor_branch="fair",
        signature="hf128|l2|rq1x6|frq6|lr1e-4|mson|skoff".replace("|mson|", "|msoff|"),
        reason="Vecino mvn-oriented alrededor del frente msoff/rq6, anclado en la familia fair que hoy domina mvn cross-branch.",
    ),
    _proposal_entry(
        raw_order=6,
        source_section="analysis_core",
        proposal_group="core",
        anchor_branch="fair",
        signature="hf128|l2|rq1x3|frq3|lr1e-3|mson|skoff".replace("|mson|", "|msoff|"),
        reason="Prueba el salto local a rq3 dentro de la familia msoff shallow que sigue fuerte en mvn.",
    ),
    _proposal_entry(
        raw_order=7,
        source_section="rrmse_primary",
        proposal_group="lens_neighbor",
        anchor_branch="rrmse",
        signature="hf192|l2|rq1x6|frq6|lr1e-4|mson|skoff",
        reason="Vecino no probado del top actual de rrmse_primary con una sola bajada de profundidad.",
    ),
    _proposal_entry(
        raw_order=8,
        source_section="rrmse_primary",
        proposal_group="lens_neighbor",
        anchor_branch="rrmse",
        signature="hf128|l3|rq1x6|frq6|lr1e-4|mson|skoff",
        reason="Mantiene la morfologia fuerte de rrmse_primary pero comprime hidden_features para comprobar si 128 aun puede competir.",
    ),
    _proposal_entry(
        raw_order=9,
        source_section="rrmse_primary",
        proposal_group="lens_neighbor",
        anchor_branch="rrmse",
        signature="hf192|l3|rq1x6|frq6|lr1e-3|mson|skoff",
        reason="Mismo nucleo rq6 del top rrmse_primary pero con lr1e-3 para testear si el frente aun no ha cerrado del todo.",
    ),
    _proposal_entry(
        raw_order=10,
        source_section="mvn",
        proposal_group="lens_neighbor",
        anchor_branch="fair",
        signature="hf128|l2|rq1x6|frq6|lr1e-4|mson|skoff".replace("|mson|", "|msoff|"),
        reason="Vecino mvn-oriented del top fair-branch actual con un unico cambio de lr hacia 1e-4.",
    ),
    _proposal_entry(
        raw_order=11,
        source_section="mvn",
        proposal_group="lens_neighbor",
        anchor_branch="fair",
        signature="hf128|l2|rq1x3|frq3|lr1e-3|mson|skoff".replace("|mson|", "|msoff|"),
        reason="Variante mvn-oriented que reduce rq manteniendo el resto de la familia msoff shallow.",
    ),
    _proposal_entry(
        raw_order=12,
        source_section="mvn",
        proposal_group="lens_neighbor",
        anchor_branch="fair",
        signature="hf256|l2|rq1x6|frq6|lr1e-3|msoff|skoff",
        reason="Escala la mejor familia mvn msoff/rq6 a 256 ocultas para comprobar si gana isotropia sin romper reconstruccion.",
    ),
    _proposal_entry(
        raw_order=13,
        source_section="fair",
        proposal_group="lens_neighbor",
        anchor_branch="rrmse",
        signature="hf256|l2|rq1x5|frq5|lr1e-3|mson|skoff",
        reason="Cruce explicito entre el frente actual de fair y el polo upstream shallow de la familia rrmse.",
    ),
    _proposal_entry(
        raw_order=14,
        source_section="fair",
        proposal_group="lens_neighbor",
        anchor_branch="rrmse",
        signature="hf128|l4|rq1x5|frq5|lr1e-3|mson|skoff",
        reason="Explora si fair todavia admite una version mas compacta del frente deep rq5 que hoy domina varias lentes.",
    ),
    _proposal_entry(
        raw_order=15,
        source_section="fair",
        proposal_group="lens_neighbor",
        anchor_branch="rrmse",
        signature="hf256|l4|rq1x8|frq8|lr1e-3|mson|skoff",
        reason="Chequea si fair puede mejorar con una expansion de rq hacia 8 sin salir del regimen lr1e-3 dominante.",
    ),
    _proposal_entry(
        raw_order=16,
        source_section="flowgencandidate_priorfit",
        proposal_group="lens_neighbor",
        anchor_branch="rrmse",
        signature="hf256|l2|rq1x5|frq5|lr1e-3|mson|skoff",
        reason="Candidata principal no probada para priorfit: shallow por profundidad pero con rq5 como el polo fuerte del top actual.",
    ),
    _proposal_entry(
        raw_order=17,
        source_section="flowgencandidate_priorfit",
        proposal_group="lens_neighbor",
        anchor_branch="rrmse",
        signature="hf192|l2|rq1x3|frq3|lr1e-3|mson|skoff",
        reason="Vecino compacto de priorfit con menor anchura y misma familia de estabilidad en train.",
    ),
    _proposal_entry(
        raw_order=18,
        source_section="flowgencandidate_priorfit",
        proposal_group="lens_neighbor",
        anchor_branch="rrmse",
        signature="hf256|l2|rq1x6|frq6|lr1e-3|mson|skoff",
        reason="Amplia la esquina shallow priorfit hacia rq6 para medir si aun hay margen en expressividad.",
    ),
    _proposal_entry(
        raw_order=19,
        source_section="flowgencandidate_robust",
        proposal_group="lens_neighbor",
        anchor_branch="rrmse",
        signature="hf256|l2|rq1x5|frq5|lr1e-3|mson|skoff",
        reason="Vecino directo del top robust/hybrid que podria unir gap sano con el polo rq5 del frente profundo.",
    ),
    _proposal_entry(
        raw_order=20,
        source_section="flowgencandidate_robust",
        proposal_group="lens_neighbor",
        anchor_branch="rrmse",
        signature="hf192|l2|rq1x3|frq3|lr1e-3|mson|skoff",
        reason="Prueba una version mas compacta del frente robust basado en shallow rq3.",
    ),
    _proposal_entry(
        raw_order=21,
        source_section="flowgencandidate_robust",
        proposal_group="lens_neighbor",
        anchor_branch="rrmse",
        signature="hf256|l2|rq1x6|frq6|lr1e-3|mson|skoff",
        reason="Ultimo vecino explicito del frente robust para comprobar si rq6 roba puesto dentro de la familia shallow.",
    ),
]


def _build_raw_proposals_df() -> pd.DataFrame:
    raw = pd.DataFrame(RAW_PROPOSALS).copy()
    raw["raw_duplicate_group_size"] = raw.groupby("cfg_signature")["cfg_signature"].transform("size")
    raw["raw_duplicate_source_sections"] = raw.groupby("cfg_signature")["source_section"].transform(
        lambda s: "|".join(sorted(set(str(x) for x in s)))
    )
    raw["raw_duplicate_count"] = raw["raw_duplicate_group_size"] - 1
    raw = raw.sort_values(["raw_order"]).reset_index(drop=True)
    return _ensure_proposal_id(raw, source_name="raw_df")


def _build_existing_configs_report(all_runs: pd.DataFrame) -> pd.DataFrame:
    runs = all_runs[all_runs["cfg_signature"].notna()].copy()
    grouped = runs.groupby("cfg_signature", as_index=False).agg(
        observed_run_count=("run_id", "size"),
        observed_branches=("branch_id", _sorted_unique),
        observed_cfg_ids=("cfg_id", _sorted_unique),
        observed_campaigns=("campaign_origin", _sorted_unique),
        observed_phases=("phase", _sorted_unique),
        observed_seeds=("seed", lambda s: "|".join(str(int(x)) for x in sorted(set(s)))),
        observed_statuses=("global_reconstruction_status", _sorted_unique),
        raw_acceptability_statuses=("raw_acceptability_status", _sorted_unique),
        observed_run_ids=("run_id", _sorted_unique),
    )
    return grouped.sort_values(["cfg_signature"]).reset_index(drop=True)


def _dedupe_proposals(raw_df: pd.DataFrame, existing_signatures: set[str]) -> pd.DataFrame:
    raw_df = _ensure_proposal_id(raw_df, source_name="raw_df")
    rows: list[dict[str, Any]] = []
    grouped = raw_df.groupby("cfg_signature", sort=False)
    for final_slot, (signature, group) in enumerate(grouped, start=1):
        first = group.iloc[0].to_dict()
        first["final_slot"] = int(final_slot)
        first["source_sections"] = "|".join(sorted(set(group["source_section"].astype(str))))
        first["proposal_groups"] = "|".join(sorted(set(group["proposal_group"].astype(str))))
        first["merged_raw_rows"] = int(len(group))
        first["was_duplicate_within_v4"] = bool(len(group) > 1)
        first["already_trained"] = signature in existing_signatures
        first["keep_for_training"] = signature not in existing_signatures
        rows.append(first)
    deduped = pd.DataFrame(rows).sort_values(["final_slot"]).reset_index(drop=True)
    deduped["proposal_id"] = deduped["cfg_signature"].map(_proposal_id_from_signature)
    return _ensure_proposal_id(deduped, source_name="deduped_df")


def _build_training_plan(deduped_df: pd.DataFrame) -> pd.DataFrame:
    deduped_df = _ensure_proposal_id(deduped_df, source_name="deduped_df")
    plan_rows: list[dict[str, Any]] = []
    trainable = deduped_df[deduped_df["keep_for_training"]].copy().reset_index(drop=True)
    for _, row in trainable.iterrows():
        run_id = f"flowprex4_{row['anchor_branch']}_tpv1_{row['proposal_id']}_seed{COMMON_EXPLORE_SEED}_v1"
        config_path = CONFIG_ROOT / str(row["anchor_branch"]) / f"{row['proposal_id']}.yaml"
        plan_rows.append(
            {
                "final_slot": int(row["final_slot"]),
                "proposal_id": str(row["proposal_id"]),
                "cfg_signature": str(row["cfg_signature"]),
                "anchor_branch": str(row["anchor_branch"]),
                "source_sections": str(row["source_sections"]),
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
    plan_df = pd.DataFrame(plan_rows).sort_values(["final_slot"]).reset_index(drop=True)
    return _ensure_proposal_id(plan_df, source_name="training_plan_df")


def _write_summary(
    *,
    inventory_df: pd.DataFrame,
    existing_configs_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    deduped_df: pd.DataFrame,
    training_plan_df: pd.DataFrame,
) -> Path:
    total_runs = int(len(inventory_df))
    campaign_counts = inventory_df["campaign_origin"].value_counts().to_dict()
    duplicate_view = deduped_df[deduped_df["was_duplicate_within_v4"]][
        ["cfg_signature", "source_sections", "merged_raw_rows"]
    ].reset_index(drop=True)
    if duplicate_view.empty:
        duplicate_block = "- No hubo duplicados internos entre las propuestas raw de `v4`."
    else:
        duplicate_block = _frame_to_markdown(duplicate_view)

    lines = [
        "# FlowPre Explore V4 Summary",
        "",
        "## Estado de partida",
        f"- Runs oficiales observadas actualmente: `{total_runs}`.",
        f"- Reparto por `campaign_origin`: `{campaign_counts}`.",
        f"- Firmas/configs ya entrenadas en oficial: `{existing_configs_df['cfg_signature'].nunique()}`.",
        f"- Seed comun real en `v2`: `{V2_COMMON_SEED}`.",
        f"- Seed comun real en `v3`: `{V3_COMMON_SEED}`.",
        f"- Seed fijada para `v4`: `{COMMON_EXPLORE_SEED}`.",
        "- `v4` materializa solo propuestas explicitas del analisis reciente; no introduce una heuristica nueva de exploracion.",
        "",
        "## Propuestas de v4",
        f"- Filas raw totales: `{len(raw_df)}`.",
        f"- Firmas unicas tras dedupe interno: `{deduped_df['cfg_signature'].nunique()}`.",
        f"- Firmas ya entrenadas detectadas: `{int(deduped_df['already_trained'].sum())}`.",
        f"- Firmas finales listas para training: `{len(training_plan_df)}`.",
        "",
        "### Duplicados internos resueltos",
        duplicate_block,
        "",
        "### Training plan preparado",
        _frame_to_markdown(
            training_plan_df[
                ["final_slot", "proposal_id", "cfg_signature", "anchor_branch", "source_sections", "planned_run_id"]
            ]
        ),
        "",
        "## Comandos a ejecutar manualmente",
        "```bash",
        "MPLCONFIGDIR=/tmp/codex-mpl .venv/bin/python scripts/f6_flowpre_explore_v4.py --dry-run",
        "MPLCONFIGDIR=/tmp/codex-mpl .venv/bin/python scripts/f6_flowpre_explore_v4.py --train --device auto",
        "```",
        "",
        "- Esta campana sigue siendo parte de una fase FlowPre abierta; no fija cierre formal ni promotion.",
    ]
    out_path = REPORT_ROOT / "flowpre_v4_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_reports(
    *,
    existing_configs_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    deduped_df: pd.DataFrame,
    training_plan_df: pd.DataFrame,
    summary_path: Path,
) -> dict[str, Path]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "existing_configs": REPORT_ROOT / "flowpre_v4_existing_configs.csv",
        "proposals_raw": REPORT_ROOT / "flowpre_v4_proposals_raw.csv",
        "proposals_deduped": REPORT_ROOT / "flowpre_v4_proposals_deduped.csv",
        "training_plan": REPORT_ROOT / "flowpre_v4_training_plan.csv",
        "summary": summary_path,
    }
    existing_configs_df.to_csv(paths["existing_configs"], index=False)
    raw_df.to_csv(paths["proposals_raw"], index=False)
    deduped_df.to_csv(paths["proposals_deduped"], index=False)
    training_plan_df.to_csv(paths["training_plan"], index=False)
    return paths


def _write_train_configs(training_plan_df: pd.DataFrame, deduped_df: pd.DataFrame) -> None:
    training_plan_df = _ensure_proposal_id(training_plan_df, source_name="training_plan_df")
    deduped_df = _ensure_proposal_id(deduped_df, source_name="deduped_df")
    if "proposal_id" not in training_plan_df.columns:
        raise RuntimeError("training_plan_df no contiene proposal_id antes de _write_train_configs.")
    dedup_lookup = deduped_df.set_index("proposal_id", drop=False)
    for _, row in training_plan_df.iterrows():
        proposal = dedup_lookup.loc[str(row["proposal_id"])]
        spec = {
            "hidden_features": int(proposal["hidden_features"]),
            "num_layers": int(proposal["num_layers"]),
            "affine_rq_ratio": [int(x) for x in proposal["affine_rq_ratio"]],
            "final_rq_layers": int(proposal["final_rq_layers"]),
            "learning_rate": float(proposal["learning_rate"]),
            "use_mean_penalty": bool(proposal["use_mean_penalty"]),
            "use_std_penalty": bool(proposal["use_std_penalty"]),
            "use_skew_penalty": bool(proposal["use_skew_penalty"]),
            "use_kurtosis_penalty": bool(proposal["use_kurtosis_penalty"]),
        }
        config = _config_from_spec(str(proposal["anchor_branch"]), spec)
        config_path = CONFIG_ROOT / str(proposal["anchor_branch"]) / f"{row['proposal_id']}.yaml"
        write_yaml(config_path, config)


def _run_train_plan(training_plan_df: pd.DataFrame, deduped_df: pd.DataFrame, *, device: str, verbose: bool, force: bool) -> None:
    from training.train_flow_pre import train_flowpre_pipeline

    training_plan_df = _ensure_proposal_id(training_plan_df, source_name="training_plan_df")
    deduped_df = _ensure_proposal_id(deduped_df, source_name="deduped_df")
    if "proposal_id" not in training_plan_df.columns:
        raise RuntimeError("training_plan_df no contiene proposal_id antes de _run_train_plan.")
    _write_train_configs(training_plan_df, deduped_df)
    inventory_df = _build_inventory()
    all_runs = _build_current_all_runs(inventory_df)
    existing_signatures = set(all_runs["cfg_signature"].dropna().astype(str))

    dedup_lookup = deduped_df.set_index("proposal_id", drop=False)
    for _, row in training_plan_df.iterrows():
        cfg_signature = str(row["cfg_signature"])
        if cfg_signature in existing_signatures and not force:
            continue

        proposal = dedup_lookup.loc[str(row["proposal_id"])]
        run_id = str(row["planned_run_id"])
        run_dir = OFFICIAL_FLOWPRE_ROOT / run_id
        results_path = run_dir / f"{run_id}_results.yaml"
        if results_path.exists() and not force:
            continue

        config_path = CONFIG_ROOT / str(proposal["anchor_branch"]) / f"{row['proposal_id']}.yaml"
        eval_ctx = {
            "dataset_name": DEFAULT_OFFICIAL_DATASET_NAME,
            "split_id": OFFICIAL_SPLIT_ID,
            "contract_id": EXPLORE_CONTRACT_ID,
            "seed_set_id": "f6_flowpre_explore_v4_screen",
            "base_config_id": f"flowpre_explore_v4_{proposal['anchor_branch']}",
            "objective_metric_id": f"flowpre_{proposal['anchor_branch']}_selection",
            "run_level_axes": {
                "campaign_id": "f6_explore_v4",
                "branch_id": str(proposal["anchor_branch"]),
                "cfg_id": str(proposal["proposal_id"]),
                "phase": "screen",
                "seed": int(COMMON_EXPLORE_SEED),
                "source_sections": str(proposal["source_sections"]),
                "final_slot": int(proposal["final_slot"]),
            },
        }
        train_flowpre_pipeline(
            config_filename=str(config_path),
            base_name=run_id.removesuffix("_v1"),
            device=device,
            seed=int(COMMON_EXPLORE_SEED),
            verbose=verbose,
            allow_test_holdout=False,
            evaluation_context=eval_ctx,
            output_namespace="official",
        )
        existing_signatures.add(cfg_signature)


def build_v4_plan_data() -> dict[str, Any]:
    inventory_df = _build_inventory()
    if inventory_df.empty:
        raise RuntimeError("No FlowPre official runs found under outputs/models/official/flow_pre.")
    all_runs = _build_current_all_runs(inventory_df)
    existing_configs_df = _build_existing_configs_report(all_runs)
    existing_signatures = set(existing_configs_df["cfg_signature"].dropna().astype(str))
    raw_df = _build_raw_proposals_df()
    deduped_df = _dedupe_proposals(raw_df, existing_signatures)
    training_plan_df = _build_training_plan(deduped_df)
    return {
        "inventory_df": inventory_df,
        "all_runs": all_runs,
        "existing_configs_df": existing_configs_df,
        "raw_df": raw_df,
        "deduped_df": deduped_df,
        "training_plan_df": training_plan_df,
    }


def write_v4_plan_reports(plan: dict[str, Any]) -> dict[str, Path]:
    summary_path = _write_summary(
        inventory_df=plan["inventory_df"],
        existing_configs_df=plan["existing_configs_df"],
        raw_df=plan["raw_df"],
        deduped_df=plan["deduped_df"],
        training_plan_df=plan["training_plan_df"],
    )
    return _write_reports(
        existing_configs_df=plan["existing_configs_df"],
        raw_df=plan["raw_df"],
        deduped_df=plan["deduped_df"],
        training_plan_df=plan["training_plan_df"],
        summary_path=summary_path,
    )


def main() -> int:
    args = _parse_args()
    mode = _mode_from_args(args)
    plan = build_v4_plan_data()
    write_v4_plan_reports(plan)
    if mode == "train":
        _run_train_plan(
            plan["training_plan_df"],
            plan["deduped_df"],
            device=args.device,
            verbose=not args.quiet,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
