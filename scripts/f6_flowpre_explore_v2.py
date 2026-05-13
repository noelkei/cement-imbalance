from __future__ import annotations

import argparse
import copy
import itertools
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.f6_selection import FLOWPRE_BRANCHES, rank_flowpre_branch, summarize_flowpre_results
from scripts.f6_common import load_json, load_yaml, write_yaml
from scripts.f6_flowpre_revalidate import (
    ANCHORS,
    COMMON_SEEDS,
    SCREENING_SEEDS,
    _build_branch_candidates,
)


OFFICIAL_FLOWPRE_ROOT = ROOT / "outputs" / "models" / "official" / "flow_pre"
REPORT_ROOT = ROOT / "outputs" / "reports" / "f6_explore_v2"
CONFIG_ROOT = REPORT_ROOT / "configs" / "flowpre"
EXPLORE_CONTRACT_ID = "f6_flowpre_explore_v2"
OFFICIAL_SPLIT_ID = "init_temporal_processed_v1"
COMMON_EXPLORE_SEED = 5678
TARGET_BUDGETS = {"rrmse": 10, "mvn": 5, "fair": 5}
TARGET_ORDER = ("rrmse", "mvn", "fair")


HISTORICAL_GRID = {
    "training.learning_rate": [1e-3, 1e-4, 1e-5],
    "model.hidden_features": [128, 192, 256],
    "model.num_layers": [2, 3, 4],
    "model.affine_rq_ratio,model.final_rq_layers": [
        ([1, 3], 3),
        ([1, 5], 5),
        ([1, 6], 6),
        ([1, 8], 8),
    ],
    "training.use_mean_penalty,training.use_std_penalty": [
        (False, False),
        (True, True),
    ],
    "training.use_skew_penalty,training.use_kurtosis_penalty": [
        (False, False),
        (True, True),
    ],
}


def _spec(
    hidden_features: int,
    num_layers: int,
    rq_layers: int,
    learning_rate: float,
    *,
    meanstd_on: bool,
    skewkurt_on: bool,
) -> dict[str, Any]:
    return {
        "hidden_features": int(hidden_features),
        "num_layers": int(num_layers),
        "affine_rq_ratio": [1, int(rq_layers)],
        "final_rq_layers": int(rq_layers),
        "learning_rate": float(learning_rate),
        "use_mean_penalty": bool(meanstd_on),
        "use_std_penalty": bool(meanstd_on),
        "use_skew_penalty": bool(skewkurt_on),
        "use_kurtosis_penalty": bool(skewkurt_on),
    }


PROPOSAL_BLUEPRINTS: dict[str, list[dict[str, Any]]] = {
    "rrmse": [
        {
            "spec": _spec(256, 4, 5, 1e-4, meanstd_on=True, skewkurt_on=False),
            "reason": "Extiende la mejor zona observada de rrmse (`lr1e-4`) con el vecino RQ5 ya definido en el grid historico.",
            "source_kind": "historical_grid_neighbor",
        },
        {
            "spec": _spec(256, 4, 6, 1e-4, meanstd_on=True, skewkurt_on=False),
            "reason": "Explora el salto local desde `lr1e-4` hacia la familia `rq6`, que sale fuerte cross-branch en isotropia.",
            "source_kind": "historical_grid_neighbor",
        },
        {
            "spec": _spec(256, 3, 3, 1e-4, meanstd_on=True, skewkurt_on=False),
            "reason": "Combina la reduccion a 3 capas ya probada con el learning rate que mejor esta funcionando en rrmse.",
            "source_kind": "historical_grid_neighbor",
        },
        {
            "spec": _spec(192, 4, 3, 1e-4, meanstd_on=True, skewkurt_on=False),
            "reason": "Recorre la vecindad directa del candidato `hidden192` manteniendo el learning rate fuerte de rrmse.",
            "source_kind": "historical_grid_neighbor",
        },
        {
            "spec": _spec(192, 4, 5, 1e-4, meanstd_on=True, skewkurt_on=False),
            "reason": "Mezcla la anchura 192 ya validada con la variante RQ5 para abrir una zona intermedia de complejidad similar.",
            "source_kind": "historical_grid_neighbor",
        },
        {
            "spec": _spec(256, 4, 3, 1e-5, meanstd_on=True, skewkurt_on=False),
            "reason": "Testa el mismo prototipo fuerte de rrmse con un unico salto local de LR hacia abajo.",
            "source_kind": "historical_grid_neighbor",
        },
        {
            "spec": _spec(256, 4, 5, 1e-5, meanstd_on=True, skewkurt_on=False),
            "reason": "Abre la esquina `rq5 + lr1e-5` sin cambiar la complejidad estructural dominante.",
            "source_kind": "historical_grid_neighbor",
        },
        {
            "spec": _spec(256, 3, 5, 1e-4, meanstd_on=True, skewkurt_on=False),
            "reason": "Profundidad 3 ya conocida, pero con RQ5 para refinar la mejor zona sin un salto arbitrario.",
            "source_kind": "historical_grid_neighbor",
        },
        {
            "spec": _spec(192, 4, 6, 1e-4, meanstd_on=True, skewkurt_on=False),
            "reason": "Sube expresividad en la familia `hidden192` con un salto local a `rq6`, manteniendo el mismo regimen de LR.",
            "source_kind": "historical_grid_neighbor",
        },
        {
            "spec": _spec(256, 4, 3, 1e-4, meanstd_on=True, skewkurt_on=True),
            "reason": "Mantiene el prototipo fuerte de rrmse y solo activa skew/kurt para testear regularizacion adicional controlada.",
            "source_kind": "historical_grid_neighbor",
        },
    ],
    "mvn": [
        {
            "spec": _spec(256, 2, 8, 1e-5, meanstd_on=False, skewkurt_on=False),
            "reason": "Escala en anchura la zona ancla isotropica `rq8 lr1e-5` sin cambiar su forma basica.",
            "source_kind": "historical_grid_neighbor",
        },
        {
            "spec": _spec(256, 2, 6, 1e-5, meanstd_on=False, skewkurt_on=False),
            "reason": "Traslada la buena familia `rq6` a 256 ocultas con el LR mas estable de isotropia.",
            "source_kind": "historical_grid_neighbor",
        },
        {
            "spec": _spec(192, 2, 6, 1e-5, meanstd_on=False, skewkurt_on=True),
            "reason": "Parte del mejor vecino `rq6` y activa skew/kurt como regularizacion local, sin tocar el resto del regimen.",
            "source_kind": "historical_grid_neighbor",
        },
        {
            "spec": _spec(192, 4, 6, 1e-5, meanstd_on=False, skewkurt_on=False),
            "reason": "Explora mas profundidad sobre la familia isotropica `rq6` con el mismo LR historicamente fuerte.",
            "source_kind": "historical_grid_neighbor",
        },
        {
            "spec": _spec(192, 3, 5, 1e-5, meanstd_on=False, skewkurt_on=False),
            "reason": "Hibrida la zona `layers3` con RQ5 para cubrir la frontera entre isotropia y fairness observada cross-branch.",
            "source_kind": "historical_grid_neighbor",
        },
    ],
    "fair": [
        {
            "spec": _spec(128, 3, 5, 1e-5, meanstd_on=False, skewkurt_on=False),
            "reason": "Recupera literalmente la config historica RT_165, ya definida y no entrenada en oficial temporal.",
            "source_kind": "historical_grid_exact",
            "historical_ref": "RT_165",
        },
        {
            "spec": _spec(128, 2, 3, 1e-5, meanstd_on=False, skewkurt_on=False),
            "reason": "Recupera literalmente RT_145; es la unica propuesta deliberadamente mas simple para fairness.",
            "source_kind": "historical_grid_exact",
            "historical_ref": "RT_145",
        },
        {
            "spec": _spec(192, 2, 5, 1e-5, meanstd_on=False, skewkurt_on=False),
            "reason": "Recupera literalmente RT_197, muy alineada con la zona cross-branch que sale fuerte en fairness.",
            "source_kind": "historical_grid_exact",
            "historical_ref": "RT_197",
        },
        {
            "spec": _spec(192, 2, 5, 1e-3, meanstd_on=False, skewkurt_on=False),
            "reason": "Recupera literalmente RT_053 para cubrir el mismo patron con un LR mas agresivo ya visto en historico.",
            "source_kind": "historical_grid_exact",
            "historical_ref": "RT_053",
        },
        {
            "spec": _spec(192, 2, 6, 1e-4, meanstd_on=False, skewkurt_on=True),
            "reason": "Se apoya en la buena zona cross-branch de `mvn` y solo activa skew/kurt con un LR intermedio local.",
            "source_kind": "historical_grid_neighbor",
        },
    ],
}


HISTORICAL_SIGNATURE_NOTES = {
    "hf256|l4|rq1x3|frq3|lr1e-4|mson|skoff": "RT_131 anchor historico de reconstruccion/fit",
    "hf192|l2|rq1x8|frq8|lr1e-5|msoff|skoff": "RT_349 anchor historico de isotropia/gaussianidad",
    "hf128|l3|rq1x6|frq6|lr1e-3|msoff|skoff": "RT_025 anchor historico de fairness/balance",
    "hf128|l3|rq1x5|frq5|lr1e-5|msoff|skoff": "RT_165 fairness historico",
    "hf128|l2|rq1x3|frq3|lr1e-5|msoff|skoff": "RT_145 fairness historico",
    "hf192|l2|rq1x5|frq5|lr1e-5|msoff|skoff": "RT_197 fairness historico",
    "hf192|l2|rq1x5|frq5|lr1e-3|msoff|skoff": "RT_053 fairness historico",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plan or train a separate exploratory FlowPre campaign (v2).")
    ap.add_argument("--dry-run", action="store_true", help="Inspect existing runs, build the v2 plan, and write reports only.")
    ap.add_argument("--train", action="store_true", help="Train the proposed v2 configs after writing the dry-run plan.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true", help="Re-run planned v2 runs even if they already exist.")
    ap.add_argument("--quiet", action="store_true", help="Reduce trainer logs in --train mode.")
    return ap.parse_args()


def _format_lr(value: float) -> str:
    if math.isclose(value, 1e-3):
        return "1e-3"
    if math.isclose(value, 1e-4):
        return "1e-4"
    if math.isclose(value, 1e-5):
        return "1e-5"
    return f"{value:.0e}"


def _format_ratio(ratio: list[int] | tuple[int, ...]) -> str:
    return "x".join(str(int(x)) for x in ratio)


def _bool_mode(first: bool, second: bool) -> str:
    if first and second:
        return "on"
    if not first and not second:
        return "off"
    return "mixed"


def _config_signature(spec: dict[str, Any]) -> str:
    meanstd_mode = _bool_mode(bool(spec["use_mean_penalty"]), bool(spec["use_std_penalty"]))
    skewkurt_mode = _bool_mode(bool(spec["use_skew_penalty"]), bool(spec["use_kurtosis_penalty"]))
    return (
        f"hf{int(spec['hidden_features'])}"
        f"|l{int(spec['num_layers'])}"
        f"|rq{_format_ratio(spec['affine_rq_ratio'])}"
        f"|frq{int(spec['final_rq_layers'])}"
        f"|lr{_format_lr(float(spec['learning_rate']))}"
        f"|ms{meanstd_mode}"
        f"|sk{skewkurt_mode}"
    )


def _proposal_id(spec: dict[str, Any]) -> str:
    meanstd_mode = _bool_mode(bool(spec["use_mean_penalty"]), bool(spec["use_std_penalty"]))
    skewkurt_mode = _bool_mode(bool(spec["use_skew_penalty"]), bool(spec["use_kurtosis_penalty"]))
    return (
        f"hf{int(spec['hidden_features'])}_"
        f"l{int(spec['num_layers'])}_"
        f"rq{int(spec['final_rq_layers'])}_"
        f"lr{_format_lr(float(spec['learning_rate']))}_"
        f"ms{meanstd_mode}_"
        f"sk{skewkurt_mode}"
    )


def _spec_from_config(config: dict[str, Any]) -> dict[str, Any] | None:
    model = dict(config.get("model") or {})
    training = dict(config.get("training") or {})
    ratio = model.get("affine_rq_ratio")
    if ratio is None:
        return None
    try:
        spec = {
            "hidden_features": int(model["hidden_features"]),
            "num_layers": int(model["num_layers"]),
            "affine_rq_ratio": [int(x) for x in ratio],
            "final_rq_layers": int(model["final_rq_layers"]),
            "learning_rate": float(training["learning_rate"]),
            "use_mean_penalty": bool(training["use_mean_penalty"]),
            "use_std_penalty": bool(training["use_std_penalty"]),
            "use_skew_penalty": bool(training["use_skew_penalty"]),
            "use_kurtosis_penalty": bool(training["use_kurtosis_penalty"]),
        }
    except KeyError:
        return None
    spec["cfg_signature"] = _config_signature(spec)
    spec["affine_rq_ratio_str"] = _format_ratio(spec["affine_rq_ratio"])
    spec["meanstd_mode"] = _bool_mode(spec["use_mean_penalty"], spec["use_std_penalty"])
    spec["skewkurt_mode"] = _bool_mode(spec["use_skew_penalty"], spec["use_kurtosis_penalty"])
    spec["complexity_score"] = (
        int(spec["hidden_features"]) * int(spec["num_layers"]) * int(spec["final_rq_layers"])
    )
    return spec


def _generate_param_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    expanded = []
    for raw_key, values in grid.items():
        keys = raw_key.split(",")
        per_key = []
        for value in values:
            if isinstance(value, tuple):
                packed = value
            else:
                packed = (value,)
            per_key.append((keys, packed))
        expanded.append(per_key)

    combos: list[dict[str, Any]] = []
    for combo in itertools.product(*expanded):
        flat: dict[str, Any] = {}
        for keys, values in combo:
            for key, value in zip(keys, values):
                flat[key] = value
        combos.append(flat)
    return combos


def _deep_set(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = config
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def _config_from_spec(target: str, spec: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(load_yaml(ANCHORS[target]))
    updates = {
        "model.hidden_features": int(spec["hidden_features"]),
        "model.num_layers": int(spec["num_layers"]),
        "model.affine_rq_ratio": [int(x) for x in spec["affine_rq_ratio"]],
        "model.final_rq_layers": int(spec["final_rq_layers"]),
        "training.learning_rate": float(spec["learning_rate"]),
        "training.use_mean_penalty": bool(spec["use_mean_penalty"]),
        "training.use_std_penalty": bool(spec["use_std_penalty"]),
        "training.use_skew_penalty": bool(spec["use_skew_penalty"]),
        "training.use_kurtosis_penalty": bool(spec["use_kurtosis_penalty"]),
    }
    for dotted_key, value in updates.items():
        _deep_set(config, dotted_key, value)
    return config


def _resolve_phase(manifest: dict[str, Any]) -> str:
    axes = dict(manifest.get("run_level_axes") or {})
    phase = axes.get("phase")
    if phase:
        return str(phase)
    seed_set_id = str(manifest.get("seed_set_id") or "")
    if "reseed" in seed_set_id:
        return "reseed"
    if "screen" in seed_set_id:
        return "screen"
    return "unknown"


def _discover_existing_runs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(OFFICIAL_FLOWPRE_ROOT.glob("*/*_run_manifest.json")):
        try:
            manifest = load_json(manifest_path)
        except Exception:
            continue

        run_id = str(manifest.get("run_id") or manifest_path.stem.replace("_run_manifest", ""))
        run_dir = manifest_path.parent
        axes = dict(manifest.get("run_level_axes") or {})
        branch_id = str(axes.get("branch_id") or "")
        cfg_id = str(axes.get("cfg_id") or "")
        phase = _resolve_phase(manifest)
        seed = int(axes.get("seed") or manifest.get("seed") or -1)
        results_path = run_dir / f"{run_id}_results.yaml"
        metrics_long_path = Path(str(manifest.get("metrics_long_path") or run_dir / f"{run_id}_metrics_long.csv"))
        versioned_config_path = run_dir / f"{run_id}.yaml"
        source_config_path = Path(str(manifest.get("config_path") or "")) if manifest.get("config_path") else None
        config_path = versioned_config_path if versioned_config_path.exists() else source_config_path

        row: dict[str, Any] = {
            "run_id": run_id,
            "branch_id": branch_id,
            "cfg_id": cfg_id,
            "phase": phase,
            "seed": seed,
            "run_dir": str(run_dir),
            "run_manifest_path": str(manifest_path),
            "results_path": str(results_path),
            "metrics_long_path": str(metrics_long_path),
            "config_path": None if config_path is None else str(config_path),
            "contract_id": manifest.get("contract_id"),
            "seed_set_id": manifest.get("seed_set_id"),
            "base_config_id": manifest.get("base_config_id"),
            "objective_metric_id": manifest.get("objective_metric_id"),
            "comparison_group_id": manifest.get("comparison_group_id"),
            "test_enabled": bool(manifest.get("test_enabled", False)),
            "has_results": results_path.exists(),
            "has_metrics_long": metrics_long_path.exists(),
            "has_config": bool(config_path and Path(config_path).exists()),
        }
        row["is_complete_core"] = bool(row["has_results"] and row["has_metrics_long"] and row["has_config"])

        if results_path.exists():
            row.update(
                summarize_flowpre_results(
                    results_path,
                    branch_id=branch_id,
                    run_id=run_id,
                    cfg_id=cfg_id,
                    phase=phase,
                    seed=seed,
                )
            )

        if config_path and Path(config_path).exists():
            try:
                spec = _spec_from_config(load_yaml(config_path))
            except Exception:
                spec = None
            if spec is not None:
                row.update(spec)
                row["historical_signature_note"] = HISTORICAL_SIGNATURE_NOTES.get(spec["cfg_signature"])
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for column in ("branch_id", "cfg_id", "phase"):
        df[column] = df[column].fillna("").astype(str)
    df["seed"] = df["seed"].fillna(-1).astype(int)
    return df.sort_values(["branch_id", "cfg_id", "phase", "seed", "run_id"]).reset_index(drop=True)


def _sorted_unique(values: pd.Series) -> str:
    cleaned = sorted({str(value) for value in values if pd.notna(value) and str(value) != ""})
    return "|".join(cleaned)


def _aggregate_existing_configs(inventory_df: pd.DataFrame) -> pd.DataFrame:
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
    metrics_cols = [
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
    usable = inventory_df[inventory_df["cfg_signature"].notna()].copy()
    if usable.empty:
        return usable

    agg_numeric = usable.groupby(spec_cols, as_index=False)[metrics_cols].mean(numeric_only=True)
    counts = usable.groupby(spec_cols).size().rename("observed_run_count").reset_index()
    branches = usable.groupby(spec_cols)["branch_id"].apply(_sorted_unique).rename("observed_branches").reset_index()
    cfg_ids = usable.groupby(spec_cols)["cfg_id"].apply(_sorted_unique).rename("observed_cfg_ids").reset_index()
    phases = usable.groupby(spec_cols)["phase"].apply(_sorted_unique).rename("observed_phases").reset_index()
    seeds = usable.groupby(spec_cols)["seed"].apply(lambda s: "|".join(str(int(x)) for x in sorted(set(s)))).rename("observed_seeds").reset_index()
    run_ids = usable.groupby(spec_cols)["run_id"].apply(_sorted_unique).rename("observed_run_ids").reset_index()
    notes = usable.groupby(spec_cols)["historical_signature_note"].apply(_sorted_unique).rename("historical_signature_note").reset_index()

    agg = agg_numeric.merge(counts, on=spec_cols, how="left")
    agg = agg.merge(branches, on=spec_cols, how="left")
    agg = agg.merge(cfg_ids, on=spec_cols, how="left")
    agg = agg.merge(phases, on=spec_cols, how="left")
    agg = agg.merge(seeds, on=spec_cols, how="left")
    agg = agg.merge(run_ids, on=spec_cols, how="left")
    agg = agg.merge(notes, on=spec_cols, how="left")
    agg["is_defined_in_historical_grid"] = True
    agg["is_already_trained"] = True

    for target in TARGET_ORDER:
        ranked = rank_flowpre_branch(agg.copy(), target)
        ranked = ranked[["cfg_signature", "branch_rank", "selection_score"]].rename(
            columns={
                "branch_rank": f"{target}_rank_cross_branch",
                "selection_score": f"{target}_selection_score_cross_branch",
            }
        )
        agg = agg.merge(ranked, on="cfg_signature", how="left")

    return agg.sort_values(["rrmse_rank_cross_branch", "mvn_rank_cross_branch", "fair_rank_cross_branch"]).reset_index(drop=True)


def _build_candidate_pool(existing_configs_df: pd.DataFrame) -> pd.DataFrame:
    trained_by_signature = (
        existing_configs_df.set_index("cfg_signature").to_dict(orient="index")
        if not existing_configs_df.empty
        else {}
    )
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
        spec["cfg_signature"] = _config_signature(spec)
        spec["affine_rq_ratio_str"] = _format_ratio(spec["affine_rq_ratio"])
        spec["meanstd_mode"] = _bool_mode(spec["use_mean_penalty"], spec["use_std_penalty"])
        spec["skewkurt_mode"] = _bool_mode(spec["use_skew_penalty"], spec["use_kurtosis_penalty"])
        spec["complexity_score"] = (
            int(spec["hidden_features"]) * int(spec["num_layers"]) * int(spec["final_rq_layers"])
        )
        trained = trained_by_signature.get(spec["cfg_signature"], {})
        row = {
            **spec,
            "defined_source": "training_scripts/flowpre_scripts/vm_training.py::GRID",
            "historical_signature_note": HISTORICAL_SIGNATURE_NOTES.get(spec["cfg_signature"]),
            "already_trained": bool(trained),
            "observed_run_count": trained.get("observed_run_count", 0),
            "observed_branches": trained.get("observed_branches"),
            "observed_cfg_ids": trained.get("observed_cfg_ids"),
            "rrmse_rank_cross_branch": trained.get("rrmse_rank_cross_branch"),
            "mvn_rank_cross_branch": trained.get("mvn_rank_cross_branch"),
            "fair_rank_cross_branch": trained.get("fair_rank_cross_branch"),
        }
        rows.append(row)

    pool = pd.DataFrame(rows).drop_duplicates(subset=["cfg_signature"]).reset_index(drop=True)
    return pool.sort_values(
        ["already_trained", "hidden_features", "num_layers", "final_rq_layers", "learning_rate", "cfg_signature"]
    ).reset_index(drop=True)


def _log_lr_distance(lhs: float, rhs: float) -> float:
    if lhs <= 0 or rhs <= 0:
        return math.inf
    return abs(math.log10(lhs) - math.log10(rhs))


def _spec_distance(candidate: dict[str, Any], observed: dict[str, Any]) -> float:
    return (
        abs(int(candidate["hidden_features"]) - int(observed["hidden_features"])) / 64.0
        + abs(int(candidate["num_layers"]) - int(observed["num_layers"]))
        + abs(int(candidate["final_rq_layers"]) - int(observed["final_rq_layers"]))
        + abs(int(candidate["affine_rq_ratio"][1]) - int(observed["affine_rq_ratio"][1])) * 0.5
        + _log_lr_distance(float(candidate["learning_rate"]), float(observed["learning_rate"]))
        + (0.0 if bool(candidate["use_mean_penalty"]) == bool(observed["use_mean_penalty"]) else 1.5)
        + (0.0 if bool(candidate["use_skew_penalty"]) == bool(observed["use_skew_penalty"]) else 1.0)
    )


def _complexity_relation(candidate: dict[str, Any], prototype: dict[str, Any]) -> str:
    delta = int(candidate["complexity_score"]) - int(prototype["complexity_score"])
    if delta > 0:
        return "higher"
    if delta < 0:
        return "lower"
    return "similar"


def _nearest_prototype(existing_configs_df: pd.DataFrame, *, target: str, candidate: dict[str, Any]) -> dict[str, Any]:
    ranked = existing_configs_df.sort_values([f"{target}_rank_cross_branch", "observed_run_count"]).copy()
    best_row: dict[str, Any] | None = None
    best_distance = math.inf
    for _, row in ranked.iterrows():
        observed = {
            "hidden_features": int(row["hidden_features"]),
            "num_layers": int(row["num_layers"]),
            "affine_rq_ratio": [int(x) for x in str(row["affine_rq_ratio_str"]).split("x")],
            "final_rq_layers": int(row["final_rq_layers"]),
            "learning_rate": float(row["learning_rate"]),
            "use_mean_penalty": bool(row["use_mean_penalty"]),
            "use_skew_penalty": bool(row["use_skew_penalty"]),
            "complexity_score": int(row["complexity_score"]),
        }
        distance = _spec_distance(candidate, observed)
        rank_value = float(row[f"{target}_rank_cross_branch"])
        if distance < best_distance or (math.isclose(distance, best_distance) and rank_value < float(best_row[f"{target}_rank_cross_branch"])):  # type: ignore[index]
            best_distance = distance
            best_row = row.to_dict()
            best_row["prototype_distance"] = distance
    if best_row is None:
        raise RuntimeError(f"No observed prototype available for target={target}.")
    return best_row


def _validate_blueprints(candidate_pool_df: pd.DataFrame) -> None:
    pool_lookup = candidate_pool_df.set_index("cfg_signature").to_dict(orient="index")
    seen: set[tuple[str, str]] = set()
    for target, blueprints in PROPOSAL_BLUEPRINTS.items():
        if len(blueprints) != TARGET_BUDGETS[target]:
            raise RuntimeError(f"Target {target} has {len(blueprints)} proposals, expected {TARGET_BUDGETS[target]}.")
        for blueprint in blueprints:
            spec = dict(blueprint["spec"])
            spec["cfg_signature"] = _config_signature(spec)
            key = (target, spec["cfg_signature"])
            if key in seen:
                raise RuntimeError(f"Duplicate proposal inside target {target}: {spec['cfg_signature']}")
            seen.add(key)
            pool_row = pool_lookup.get(spec["cfg_signature"])
            if pool_row is None:
                raise RuntimeError(f"Proposal {spec['cfg_signature']} is not present in the historical defined pool.")
            if bool(pool_row["already_trained"]):
                raise RuntimeError(f"Proposal {spec['cfg_signature']} was already trained and should not be proposed again.")


def _build_proposals(existing_configs_df: pd.DataFrame, candidate_pool_df: pd.DataFrame) -> pd.DataFrame:
    _validate_blueprints(candidate_pool_df)
    pool_lookup = candidate_pool_df.set_index("cfg_signature").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for target in TARGET_ORDER:
        for order_idx, blueprint in enumerate(PROPOSAL_BLUEPRINTS[target], start=1):
            spec = dict(blueprint["spec"])
            spec["cfg_signature"] = _config_signature(spec)
            spec["affine_rq_ratio_str"] = _format_ratio(spec["affine_rq_ratio"])
            spec["meanstd_mode"] = _bool_mode(spec["use_mean_penalty"], spec["use_std_penalty"])
            spec["skewkurt_mode"] = _bool_mode(spec["use_skew_penalty"], spec["use_kurtosis_penalty"])
            spec["complexity_score"] = (
                int(spec["hidden_features"]) * int(spec["num_layers"]) * int(spec["final_rq_layers"])
            )
            pool_row = dict(pool_lookup[spec["cfg_signature"]])
            prototype = _nearest_prototype(existing_configs_df, target=target, candidate=spec)
            rows.append(
                {
                    "target_branch": target,
                    "proposal_order": order_idx,
                    "proposal_id": _proposal_id(spec),
                    "screening_seed": COMMON_EXPLORE_SEED,
                    "reuse_source": pool_row["defined_source"],
                    "source_kind": blueprint["source_kind"],
                    "historical_ref": blueprint.get("historical_ref"),
                    "reason": blueprint["reason"],
                    "prototype_signature": prototype["cfg_signature"],
                    "prototype_observed_branches": prototype["observed_branches"],
                    "prototype_observed_cfg_ids": prototype["observed_cfg_ids"],
                    "prototype_rrmse_rank_cross_branch": prototype["rrmse_rank_cross_branch"],
                    "prototype_mvn_rank_cross_branch": prototype["mvn_rank_cross_branch"],
                    "prototype_fair_rank_cross_branch": prototype["fair_rank_cross_branch"],
                    "prototype_distance": prototype["prototype_distance"],
                    "complexity_relation_vs_prototype": _complexity_relation(spec, prototype),
                    **spec,
                }
            )
    proposed = pd.DataFrame(rows)
    duplicate_mask = proposed.duplicated(subset=["cfg_signature"], keep=False)
    if duplicate_mask.any():
        dupes = proposed.loc[duplicate_mask, ["target_branch", "proposal_id", "cfg_signature"]]
        raise RuntimeError(f"Cross-target duplicate proposals detected:\n{dupes.to_string(index=False)}")
    return proposed.sort_values(["target_branch", "proposal_order"]).reset_index(drop=True)


def _build_training_plan(proposed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in proposed_df.iterrows():
        run_id = f"flowprex2_{row['target_branch']}_tpv1_{row['proposal_id']}_seed{int(row['screening_seed'])}_v1"
        config_path = CONFIG_ROOT / str(row["target_branch"]) / f"{row['proposal_id']}.yaml"
        rows.append(
            {
                "target_branch": row["target_branch"],
                "proposal_order": int(row["proposal_order"]),
                "proposal_id": row["proposal_id"],
                "cfg_signature": row["cfg_signature"],
                "seed": int(row["screening_seed"]),
                "planned_run_id": run_id,
                "planned_config_path": str(config_path),
                "planned_output_dir": str(OFFICIAL_FLOWPRE_ROOT / run_id),
                "contract_id": EXPLORE_CONTRACT_ID,
                "split_id": OFFICIAL_SPLIT_ID,
                "creates_promotion_manifest": False,
                "uses_test": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["target_branch", "proposal_order"]).reset_index(drop=True)


def _branch_phase_counts(inventory_df: pd.DataFrame) -> str:
    if inventory_df.empty:
        return "- no runs found"
    counts = (
        inventory_df.groupby(["branch_id", "phase"])
        .size()
        .rename("n_runs")
        .reset_index()
        .sort_values(["branch_id", "phase"])
    )
    return "\n".join(
        f"- {row.branch_id} / {row.phase}: {int(row.n_runs)}"
        for row in counts.itertuples(index=False)
    )


def _audit_22_vs_28(inventory_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    screening_candidates = {branch: _build_branch_candidates(branch) for branch in FLOWPRE_BRANCHES}
    planned_screen = sum(len(specs) for specs in screening_candidates.values())
    planned_reseed = len(FLOWPRE_BRANCHES) * 2 * 2
    planned_total = planned_screen + planned_reseed
    observed_total = int(len(inventory_df))

    screen_df = inventory_df[inventory_df["phase"] == "screen"].copy()
    top2_cfgs: dict[str, list[str]] = {}
    for branch in FLOWPRE_BRANCHES:
        ranked = rank_flowpre_branch(screen_df[screen_df["branch_id"] == branch].copy(), branch)
        top2_cfgs[branch] = ranked.head(2)["cfg_id"].astype(str).tolist()

    expected_rows: list[dict[str, Any]] = []
    for branch in FLOWPRE_BRANCHES:
        screening_seed = SCREENING_SEEDS[branch]
        reseed_seeds = [seed for seed in COMMON_SEEDS if seed != screening_seed]
        for cfg_id in top2_cfgs[branch]:
            for seed in reseed_seeds:
                run_id = f"flowpre_{branch}_tpv1_{cfg_id}_seed{seed}_v1"
                expected_rows.append(
                    {
                        "branch_id": branch,
                        "cfg_id": cfg_id,
                        "seed": int(seed),
                        "expected_run_id": run_id,
                    }
                )
    expected_df = pd.DataFrame(expected_rows)
    observed_run_ids = set(inventory_df["run_id"].astype(str))
    expected_df["is_observed"] = expected_df["expected_run_id"].isin(observed_run_ids)
    missing_df = expected_df[~expected_df["is_observed"]].copy().reset_index(drop=True)

    if missing_df.empty:
        classification = "no, es una lectura incorrecta"
    elif len(missing_df) == (planned_total - observed_total):
        classification = "parcialmente cierto pero con matices"
    else:
        classification = "parcialmente cierto pero con matices"

    audit = {
        "planned_screen": planned_screen,
        "planned_reseed": planned_reseed,
        "planned_total": planned_total,
        "observed_total": observed_total,
        "screening_top2_by_branch": top2_cfgs,
        "missing_count_vs_plan": int(len(missing_df)),
        "classification": classification,
        "explanation_28": "presupuesto teorico del script original `f6_flowpre_revalidate.py`: 16 screening + 12 reseed",
        "explanation_22": "runs oficiales FlowPre realmente observadas/materializadas ahora mismo bajo outputs/models/official/flow_pre",
        "missing_runs_identifiable": bool(not missing_df.empty),
    }
    return audit, missing_df


def _top_cross_branch_table(existing_configs_df: pd.DataFrame, target: str, limit: int = 5) -> pd.DataFrame:
    cols = [
        "cfg_signature",
        "observed_branches",
        "observed_cfg_ids",
        "observed_run_count",
        f"{target}_rank_cross_branch",
        "val_rrmse_mean",
        "val_rrmse_std",
        "val_skew_abs",
        "val_kurt_excess_abs",
        "val_eigstd",
        "val_pc_worst_mean",
        "val_pc_worst_std",
        "val_pc_wavg_mean",
    ]
    available = [col for col in cols if col in existing_configs_df.columns]
    return (
        existing_configs_df.sort_values([f"{target}_rank_cross_branch", "observed_run_count"])
        .head(limit)[available]
        .reset_index(drop=True)
    )


def _write_summary(
    *,
    audit: dict[str, Any],
    missing_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    existing_configs_df: pd.DataFrame,
    proposed_df: pd.DataFrame,
) -> Path:
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

    lines = [
        "# FlowPre Explore V2 Summary",
        "",
        "## Audit 22 vs 28",
        f"- `28` = {audit['explanation_28']}.",
        f"- `22` = {audit['explanation_22']}.",
        f"- Clasificacion defendible: **{audit['classification']}**.",
        f"- Planned screen = {audit['planned_screen']}, planned reseed = {audit['planned_reseed']}, planned total = {audit['planned_total']}.",
        f"- Observed official runs now = {audit['observed_total']}.",
        f"- Top-2 de screening reconstruidos con la logica real del script original: {audit['screening_top2_by_branch']}.",
    ]
    if missing_df.empty:
        lines.extend(["- No hay runs faltantes respecto al plan teorico original.", ""])
    else:
        lines.append(
            "- Las runs no observadas respecto al plan teorico original son identificables, pero eso no implica por si solo que haya una deuda canonicamente obligatoria."
        )
        lines.append("")
        lines.append("### Planned-but-unobserved runs from the original script")
        for row in missing_df.itertuples(index=False):
            lines.append(f"- {row.expected_run_id}")
        lines.append("")

    lines.extend(
        [
            "## Observed inventory",
            _branch_phase_counts(inventory_df),
            "",
            "## Top observed zones (cross-branch, by target-specific ranking)",
            "",
            "### rrmse",
            _frame_to_markdown(_top_cross_branch_table(existing_configs_df, "rrmse")),
            "",
            "### mvn",
            _frame_to_markdown(_top_cross_branch_table(existing_configs_df, "mvn")),
            "",
            "### fair",
            _frame_to_markdown(_top_cross_branch_table(existing_configs_df, "fair")),
            "",
            "## Proposed exploratory configs",
            "- Todas las propuestas salen de configs ya definidas en el grid historico; no se han inventado combinaciones fuera de ese espacio.",
            f"- La campana exploratoria v2 fija una seed comun unica para todos los targets: `{COMMON_EXPLORE_SEED}`.",
            "- No se han lanzado entrenamientos en este dry-run.",
            "- No se crean promotion manifests y no se toca el reporting canonico de F6 actual.",
            "",
        ]
    )

    for target in TARGET_ORDER:
        lines.append(f"### {target}")
        target_df = proposed_df[proposed_df["target_branch"] == target].copy()
        for row in target_df.itertuples(index=False):
            hist = f" [{row.historical_ref}]" if getattr(row, "historical_ref") else ""
            lines.append(
                f"- `{row.proposal_id}`{hist}: {row.reason} "
                f"(prototipo observado={row.prototype_signature}, complejidad_vs_prototipo={row.complexity_relation_vs_prototype})."
            )
        lines.append("")

    out_path = REPORT_ROOT / "flowpre_v2_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_reports(
    *,
    inventory_df: pd.DataFrame,
    existing_configs_df: pd.DataFrame,
    candidate_pool_df: pd.DataFrame,
    proposed_df: pd.DataFrame,
    training_plan_df: pd.DataFrame,
    audit: dict[str, Any],
    missing_df: pd.DataFrame,
) -> dict[str, Path]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "inventory": REPORT_ROOT / "flowpre_v2_inventory.csv",
        "existing_configs": REPORT_ROOT / "flowpre_v2_existing_configs.csv",
        "candidate_pool": REPORT_ROOT / "flowpre_v2_candidate_pool.csv",
        "proposed_configs": REPORT_ROOT / "flowpre_v2_proposed_configs.csv",
        "training_plan": REPORT_ROOT / "flowpre_v2_training_plan.csv",
    }
    inventory_df.to_csv(paths["inventory"], index=False)
    existing_configs_df.to_csv(paths["existing_configs"], index=False)
    candidate_pool_df.to_csv(paths["candidate_pool"], index=False)
    proposed_df.to_csv(paths["proposed_configs"], index=False)
    training_plan_df.to_csv(paths["training_plan"], index=False)
    paths["summary"] = _write_summary(
        audit=audit,
        missing_df=missing_df,
        inventory_df=inventory_df,
        existing_configs_df=existing_configs_df,
        proposed_df=proposed_df,
    )
    return paths


def _write_train_configs(proposed_df: pd.DataFrame) -> None:
    for _, row in proposed_df.iterrows():
        spec = {
            "hidden_features": int(row["hidden_features"]),
            "num_layers": int(row["num_layers"]),
            "affine_rq_ratio": [int(x) for x in str(row["affine_rq_ratio_str"]).split("x")],
            "final_rq_layers": int(row["final_rq_layers"]),
            "learning_rate": float(row["learning_rate"]),
            "use_mean_penalty": bool(row["use_mean_penalty"]),
            "use_std_penalty": bool(row["use_std_penalty"]),
            "use_skew_penalty": bool(row["use_skew_penalty"]),
            "use_kurtosis_penalty": bool(row["use_kurtosis_penalty"]),
        }
        config = _config_from_spec(str(row["target_branch"]), spec)
        config_path = CONFIG_ROOT / str(row["target_branch"]) / f"{row['proposal_id']}.yaml"
        write_yaml(config_path, config)


def _run_train_plan(proposed_df: pd.DataFrame, *, device: str, verbose: bool, force: bool) -> None:
    from training.train_flow_pre import train_flowpre_pipeline

    _write_train_configs(proposed_df)
    existing_inventory = _discover_existing_runs()
    existing_signatures = set(existing_inventory["cfg_signature"].dropna().astype(str))
    for _, row in proposed_df.iterrows():
        seed = int(row["screening_seed"])
        cfg_signature = str(row["cfg_signature"])
        if cfg_signature in existing_signatures:
            continue
        run_id = f"flowprex2_{row['target_branch']}_tpv1_{row['proposal_id']}_seed{seed}_v1"
        run_dir = OFFICIAL_FLOWPRE_ROOT / run_id
        results_path = run_dir / f"{run_id}_results.yaml"
        if results_path.exists() and not force:
            continue

        config_path = CONFIG_ROOT / str(row["target_branch"]) / f"{row['proposal_id']}.yaml"
        eval_ctx = {
            "dataset_name": "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1",
            "split_id": OFFICIAL_SPLIT_ID,
            "contract_id": EXPLORE_CONTRACT_ID,
            "seed_set_id": "f6_flowpre_explore_v2_screen",
            "base_config_id": f"flowpre_explore_v2_{row['target_branch']}",
            "objective_metric_id": f"flowpre_{row['target_branch']}_selection",
            "run_level_axes": {
                "campaign_id": "f6_explore_v2",
                "branch_id": str(row["target_branch"]),
                "cfg_id": str(row["proposal_id"]),
                "phase": "screen",
                "seed": seed,
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


def _mode_from_args(args: argparse.Namespace) -> str:
    if args.dry_run and args.train:
        raise RuntimeError("Use either --dry-run or --train, not both.")
    if args.train:
        return "train"
    return "dry-run"


def main() -> int:
    args = _parse_args()
    mode = _mode_from_args(args)

    inventory_df = _discover_existing_runs()
    if inventory_df.empty:
        raise RuntimeError("No FlowPre official runs found under outputs/models/official/flow_pre.")

    audit, missing_df = _audit_22_vs_28(inventory_df)
    existing_configs_df = _aggregate_existing_configs(inventory_df)
    candidate_pool_df = _build_candidate_pool(existing_configs_df)
    proposed_df = _build_proposals(existing_configs_df, candidate_pool_df)
    training_plan_df = _build_training_plan(proposed_df)
    _write_reports(
        inventory_df=inventory_df,
        existing_configs_df=existing_configs_df,
        candidate_pool_df=candidate_pool_df,
        proposed_df=proposed_df,
        training_plan_df=training_plan_df,
        audit=audit,
        missing_df=missing_df,
    )

    if mode == "train":
        _run_train_plan(proposed_df, device=args.device, verbose=not args.quiet, force=args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
