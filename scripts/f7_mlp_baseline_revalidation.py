from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

from evaluation.metrics import compute_mlp_metrics, select_metric
from training.train_mlp import train_mlp_pipeline
from training.utils import ROOT_PATH, load_scaled_sets, select_training_device


REPORT_COLUMNS = [
    "phase",
    "run_label",
    "cfg_id",
    "seed",
    "status",
    "objective_metric_id",
    "objective_value",
    "val_macro_rrmse",
    "val_overall_rrmse",
    "val_overall_rmse",
    "train_macro_rrmse",
    "test_macro_rrmse",
    "runtime_s",
    "device",
    "run_dir",
    "error",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Spec at {path} is not a YAML mapping.")
    return data


def _dump_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _deep_set(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = data
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def _load_scaled_bundle(spec: dict[str, Any]) -> tuple[pd.DataFrame, ...]:
    dataset_cfg = spec["dataset"]
    return load_scaled_sets(
        dataset_cfg["scaled_set_name"],
        require_condition_col=dataset_cfg.get("condition_col", "type"),
        verbose=False,
    )


def _load_y_scaler(spec: dict[str, Any]):
    manifest_path = ROOT_PATH / spec["dataset"]["dataset_manifest_path"]
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    y_scaler_rel = manifest["scaler_artifacts"]["y"]
    return joblib.load(ROOT_PATH / y_scaler_rel)


def _outputs_dir(study_id: str) -> Path:
    return ROOT_PATH / "outputs" / "reports" / study_id


def _generated_config_dir(study_id: str) -> Path:
    return ROOT_PATH / "config" / "generated" / study_id


def _phase_key(phase: str, run_label: str | None) -> str:
    return f"{phase}__{run_label}" if run_label else phase


def _existing_run_dirs() -> set[Path]:
    root = ROOT_PATH / "outputs" / "models" / "mlp"
    if not root.exists():
        return set()
    return {p for p in root.iterdir() if p.is_dir()}


def _infer_new_run_dir(before: set[Path], after: set[Path], base_name: str) -> Path | None:
    candidates = sorted(
        (p for p in after - before if p.name.startswith(base_name + "_v")),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _find_latest_completed_run(base_name: str) -> Path | None:
    root = ROOT_PATH / "outputs" / "models" / "mlp"
    if not root.exists():
        return None
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith(base_name + "_v")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        has_results = (run_dir / "results.yaml").exists() or any(run_dir.glob("*_results.yaml"))
        has_manifest = (run_dir / "run_manifest.json").exists() or any(run_dir.glob("*_run_manifest.json"))
        if has_results and has_manifest:
            return run_dir
    return None


def _resolve_results_path(run_dir: Path) -> Path:
    stable = run_dir / "results.yaml"
    if stable.exists():
        return stable
    candidates = sorted(run_dir.glob("*_results.yaml"))
    if not candidates:
        raise FileNotFoundError(f"Could not resolve results artifact under {run_dir}")
    return candidates[0]


def _recover_row_from_run_dir(
    *,
    run_dir: Path,
    phase: str,
    run_label: str | None,
    cfg_id: str,
    seed: int,
    objective_metric_id: str,
    objective_metric_path: str,
    device: str,
) -> dict[str, Any]:
    results_path = _resolve_results_path(run_dir)
    with results_path.open("r", encoding="utf-8") as handle:
        results = yaml.safe_load(handle) or {}
    raw_real = results.get("raw_real") or {}
    train_metrics = raw_real.get("train") or {}
    val_metrics = raw_real.get("val") or {}
    test_metrics = raw_real.get("test") or {}
    return {
        "phase": phase,
        "run_label": run_label,
        "cfg_id": cfg_id,
        "seed": seed,
        "status": "ok",
        "objective_metric_id": objective_metric_id,
        "objective_value": _macro_metric(val_metrics, objective_metric_path),
        "val_macro_rrmse": _macro_metric(val_metrics, objective_metric_path),
        "val_overall_rrmse": _metric_overall(val_metrics, "rrmse"),
        "val_overall_rmse": _metric_overall(val_metrics, "rmse"),
        "train_macro_rrmse": _macro_metric(train_metrics, objective_metric_path),
        "test_macro_rrmse": (
            _macro_metric(test_metrics, objective_metric_path) if test_metrics else None
        ),
        "runtime_s": None,
        "device": device,
        "run_dir": str(run_dir),
        "error": None,
    }


def _build_trial_config(
    *,
    base_cfg: dict[str, Any],
    spec: dict[str, Any],
    variant: dict[str, Any],
    seed: int,
    allow_test_holdout: bool,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    protocol = spec["protocol"]

    for dotted, value in protocol.get("fixed_overrides", {}).items():
        _deep_set(cfg, dotted, value)
    for dotted, value in variant.get("params", {}).items():
        _deep_set(cfg, dotted, value)

    cfg.setdefault("contract", {})
    cfg["contract"]["mlp_base_config_id"] = variant["cfg_id"]
    cfg["contract"]["objective_metric_id"] = protocol["objective_metric_id"]
    cfg["contract"]["allow_test_holdout_default"] = bool(allow_test_holdout)

    cfg.setdefault("training", {})
    cfg["training"]["seed"] = int(seed)
    cfg["seed"] = int(seed)

    return cfg


def _run_eval_context(
    *,
    spec: dict[str, Any],
    y_scaler,
    phase: str,
    seed: int,
    cfg_id: str,
) -> dict[str, Any]:
    dataset_cfg = spec["dataset"]
    protocol = spec["protocol"]
    return {
        "contract_id": spec["study"]["study_id"],
        "comparison_group_id": f"{spec['study']['study_id']}__{cfg_id}",
        "seed_set_id": protocol["seeds"]["seed_set_id"],
        "mlp_base_config_id": cfg_id,
        "objective_metric_id": protocol["objective_metric_id"],
        "dataset_name": dataset_cfg["dataset_name"],
        "dataset_manifest_path": str(ROOT_PATH / dataset_cfg["dataset_manifest_path"]),
        "split_id": dataset_cfg["split_id"],
        "dataset_level_axes": dataset_cfg["dataset_level_axes"],
        "y_scaler": y_scaler,
        "upstream_variant_fingerprint": f"{spec['study']['study_id']}::{phase}::{cfg_id}::seed{seed}",
    }


def _metric_overall(metrics: dict[str, Any], name: str) -> float | None:
    overall = metrics.get("overall")
    if not isinstance(overall, dict):
        return None
    value = overall.get(name)
    return float(value) if value is not None else None


def _macro_metric(metrics: dict[str, Any], path: str) -> float:
    value = select_metric(metrics, path)
    return float(value)


def _compute_real_scale_metrics(
    *,
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    condition_col: str,
    y_scaler,
) -> dict[str, Any]:
    return compute_mlp_metrics(
        model=model,
        X=X,
        y=y,
        condition_col=condition_col,
        y_scaler=y_scaler,
    )


def _variant_risk_row(variant: dict[str, Any]) -> dict[str, Any]:
    params = variant.get("params", {})
    batch_size = params.get("training.batch_size")
    row = {
        "cfg_id": variant["cfg_id"],
        "rationale": variant.get("rationale"),
        "batch_size": batch_size,
        "num_epochs": params.get("training.num_epochs"),
        "learning_rate": params.get("training.learning_rate"),
        "hidden_dim": params.get("model.hidden_dim"),
        "num_layers": params.get("model.num_layers"),
        "embedding_dim": params.get("model.embedding_dim"),
        "f7_balanced_cycling_compatible": variant.get("checks", {}).get("f7_balanced_cycling_compatible"),
        "batch_divisible_by_3": (int(batch_size) % 3 == 0) if batch_size is not None else None,
        "note": variant.get("checks", {}).get("note"),
    }
    return row


def _write_markdown_report(
    *,
    report_dir: Path,
    spec: dict[str, Any],
    runs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    param_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    device_df: pd.DataFrame,
    winner_test_df: pd.DataFrame | None,
) -> None:
    study_id = spec["study"]["study_id"]
    dataset_name = spec["dataset"]["dataset_name"]
    objective_metric_id = spec["protocol"]["objective_metric_id"]
    lines: list[str] = []
    lines.append(f"# {study_id}")
    lines.append("")
    lines.append("## Contexto")
    lines.append("")
    lines.append(f"- Dataset fijo: `{dataset_name}`")
    lines.append("- Regimen fijo: `plain = baseline + no cycling`")
    lines.append("- Loss fija: `overall + rmse`")
    lines.append(f"- Metrica guia: `{objective_metric_id}` en `val`")
    lines.append("")
    lines.append("## Ranking agregado por configuracion")
    lines.append("")
    if summary_df.empty:
        lines.append("No hay resultados agregados todavia.")
    else:
        top = summary_df.sort_values(["successful_runs", "mean_val_macro_rrmse"], ascending=[False, True]).head(10)
        lines.append("```text")
        lines.append(top.to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## Chequeos de compatibilidad")
    lines.append("")
    if risk_df.empty:
        lines.append("No hay chequeos registrados.")
    else:
        lines.append("```text")
        lines.append(risk_df.to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## Sensibilidad por hiperparametro")
    lines.append("")
    if param_df.empty:
        lines.append("No hay agregaciones por hiperparametro todavia.")
    else:
        lines.append("```text")
        lines.append(param_df.to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## Benchmark por dispositivo")
    lines.append("")
    if device_df.empty:
        lines.append("No hay benchmark de dispositivo todavia.")
    else:
        lines.append("```text")
        lines.append(device_df.to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## Winner test check")
    lines.append("")
    if winner_test_df is None or winner_test_df.empty:
        lines.append("Aun no se ha ejecutado el chequeo final en `test` del ganador.")
    else:
        lines.append("```text")
        lines.append(winner_test_df.to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## Artefactos")
    lines.append("")
    lines.append(f"- Runs por seed: `{(report_dir / 'seed_runs.csv').relative_to(ROOT_PATH)}`")
    lines.append(f"- Resumen agregado: `{(report_dir / 'config_summary.csv').relative_to(ROOT_PATH)}`")
    lines.append(f"- Resumen por dispositivo: `{(report_dir / 'device_summary.csv').relative_to(ROOT_PATH)}`")
    lines.append(f"- Chequeos por parametro: `{(report_dir / 'parameter_slices.csv').relative_to(ROOT_PATH)}`")
    lines.append(f"- Riesgos/compatibilidad: `{(report_dir / 'variant_risk_checks.csv').relative_to(ROOT_PATH)}`")
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parameter_slices(summary_df: pd.DataFrame, variants: list[dict[str, Any]]) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    summary_by_cfg = summary_df.set_index(["run_label", "cfg_id"])
    rows: list[dict[str, Any]] = []
    keys = [
        "training.batch_size",
        "training.num_epochs",
        "training.learning_rate",
        "model.hidden_dim",
        "model.num_layers",
        "model.embedding_dim",
    ]
    for key in keys:
        for variant in variants:
            if key not in variant.get("params", {}):
                continue
            cfg_id = variant["cfg_id"]
            for run_label in summary_df["run_label"].drop_duplicates().tolist():
                idx = (run_label, cfg_id)
                if idx not in summary_by_cfg.index:
                    continue
                row = summary_by_cfg.loc[idx].to_dict()
                rows.append(
                    {
                        "run_label": run_label,
                        "parameter": key,
                        "value": variant["params"][key],
                        "cfg_id": cfg_id,
                        "mean_val_macro_rrmse": row.get("mean_val_macro_rrmse"),
                        "std_val_macro_rrmse": row.get("std_val_macro_rrmse"),
                        "mean_runtime_s": row.get("mean_runtime_s"),
                        "successful_runs": row.get("successful_runs"),
                        "all_runs_ok": row.get("all_runs_ok"),
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()
    success_df = runs_df[runs_df["status"] == "ok"].copy()
    if success_df.empty:
        return pd.DataFrame()
    grouped = success_df.groupby(["run_label", "cfg_id"], dropna=False)
    summary = grouped.agg(
        successful_runs=("cfg_id", "size"),
        mean_val_macro_rrmse=("val_macro_rrmse", "mean"),
        std_val_macro_rrmse=("val_macro_rrmse", "std"),
        mean_runtime_s=("runtime_s", "mean"),
        median_runtime_s=("runtime_s", "median"),
        max_runtime_s=("runtime_s", "max"),
    ).reset_index()

    total_runs = runs_df.groupby(["run_label", "cfg_id"]).size().rename("total_runs").reset_index()
    summary = summary.merge(total_runs, on=["run_label", "cfg_id"], how="left")
    summary["all_runs_ok"] = summary["successful_runs"] == summary["total_runs"]
    return summary.sort_values(["run_label", "all_runs_ok", "mean_val_macro_rrmse"], ascending=[True, False, True]).reset_index(drop=True)


def _build_device_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()
    success_df = runs_df[runs_df["status"] == "ok"].copy()
    if success_df.empty:
        return pd.DataFrame()
    return (
        success_df.groupby(["run_label", "device"], dropna=False)
        .agg(
            successful_runs=("cfg_id", "size"),
            mean_runtime_s=("runtime_s", "mean"),
            median_runtime_s=("runtime_s", "median"),
            max_runtime_s=("runtime_s", "max"),
            mean_val_macro_rrmse=("val_macro_rrmse", "mean"),
        )
        .reset_index()
        .sort_values(["run_label", "device"])
        .reset_index(drop=True)
    )


def _load_existing_rows(csv_path: Path, phase: str) -> dict[tuple[str, str | None, str, int], dict[str, Any]]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    df = df[df["phase"] == phase].copy()
    rows: dict[tuple[str, str | None, str, int], dict[str, Any]] = {}
    for _, row in df.iterrows():
        run_label = row["run_label"] if "run_label" in row and pd.notna(row["run_label"]) else None
        key = (str(row["phase"]), run_label, str(row["cfg_id"]), int(row["seed"]))
        rows[key] = row.to_dict()
    return rows


def run_phase(
    *,
    spec_path: Path,
    device: str,
    run_label: str | None = None,
    include_test_for_all: bool = False,
    winner_only: bool = False,
) -> None:
    spec = _load_yaml(spec_path)
    study_id = spec["study"]["study_id"]
    report_dir = _outputs_dir(study_id)
    generated_cfg_dir = _generated_config_dir(study_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = ROOT_PATH / spec["protocol"]["base_config_path"]
    base_cfg = _load_yaml(base_cfg_path)
    X_train, X_val, X_test, y_train, y_val, y_test = _load_scaled_bundle(spec)
    y_scaler = _load_y_scaler(spec)
    device_obj = select_training_device(device)
    objective_metric_path = spec["protocol"]["objective_metric_path"]
    seeds = [int(x) for x in spec["protocol"]["seeds"]["values"]]
    variants = list(spec["variants"])
    phase_key = _phase_key("winner_test_check" if winner_only else "val_grid", run_label)

    if winner_only:
        existing_summary = pd.read_csv(report_dir / "config_summary.csv")
        if existing_summary.empty:
            raise RuntimeError("No previous summary found to choose the winner.")
        if "run_label" in existing_summary.columns:
            target = existing_summary[
                existing_summary["run_label"].fillna("") == (run_label or "")
            ]
        else:
            target = existing_summary
        if target.empty:
            raise RuntimeError(f"No previous summary found for run_label={run_label!r}.")
        winner_cfg_id = str(
            target.sort_values(
                ["all_runs_ok", "mean_val_macro_rrmse"],
                ascending=[False, True],
            ).iloc[0]["cfg_id"]
        )
        variants = [v for v in variants if v["cfg_id"] == winner_cfg_id]
        phase = phase_key
        allow_test_holdout = True
    else:
        phase = phase_key
        allow_test_holdout = bool(include_test_for_all)

    runs_path = report_dir / ("winner_test_runs.csv" if winner_only else "seed_runs.csv")
    run_rows_map = _load_existing_rows(runs_path, phase)

    for variant in variants:
        cfg_id = variant["cfg_id"]
        for seed in seeds:
            key = (phase, run_label, cfg_id, seed)
            existing = run_rows_map.get(key)
            if existing is not None and str(existing.get("status")) == "ok":
                continue

            base_name = f"{study_id}__{phase}__{cfg_id}__seed{seed}"
            recovered_run_dir = _find_latest_completed_run(base_name)
            if recovered_run_dir is not None:
                run_rows_map[key] = _recover_row_from_run_dir(
                    run_dir=recovered_run_dir,
                    phase=phase,
                    run_label=run_label,
                    cfg_id=cfg_id,
                    seed=seed,
                    objective_metric_id=spec["protocol"]["objective_metric_id"],
                    objective_metric_path=objective_metric_path,
                    device=str(device_obj),
                )
                pd.DataFrame(run_rows_map.values(), columns=REPORT_COLUMNS).to_csv(runs_path, index=False)
                continue

            before_dirs = _existing_run_dirs()
            cfg = _build_trial_config(
                base_cfg=base_cfg,
                spec=spec,
                variant=variant,
                seed=seed,
                allow_test_holdout=allow_test_holdout,
            )
            cfg_rel = Path("generated") / study_id / f"{cfg_id}__seed{seed}.yaml"
            cfg_path = ROOT_PATH / "config" / cfg_rel
            _dump_yaml(cfg, cfg_path)

            eval_ctx = _run_eval_context(
                spec=spec,
                y_scaler=y_scaler,
                phase=phase,
                seed=seed,
                cfg_id=cfg_id,
            )

            started = time.time()
            row = {
                "phase": phase,
                "run_label": run_label,
                "cfg_id": cfg_id,
                "seed": seed,
                "status": "failed",
                "objective_metric_id": spec["protocol"]["objective_metric_id"],
                "objective_value": None,
                "val_macro_rrmse": None,
                "val_overall_rrmse": None,
                "val_overall_rmse": None,
                "train_macro_rrmse": None,
                "test_macro_rrmse": None,
                "runtime_s": None,
                "device": str(device_obj),
                "run_dir": None,
                "error": None,
            }
            try:
                model, _ = train_mlp_pipeline(
                    condition_col=spec["dataset"].get("condition_col", "type"),
                    config_filename=str(cfg_path),
                    base_name=base_name,
                    device=str(device_obj),
                    seed=seed,
                    verbose=True,
                    allow_test_holdout=allow_test_holdout,
                    X_train=X_train,
                    X_val=X_val,
                    X_test=X_test,
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                    evaluation_context=eval_ctx,
                )
                val_metrics = _compute_real_scale_metrics(
                    model=model,
                    X=X_val,
                    y=y_val,
                    condition_col=spec["dataset"].get("condition_col", "type"),
                    y_scaler=y_scaler,
                )
                train_metrics = _compute_real_scale_metrics(
                    model=model,
                    X=X_train,
                    y=y_train,
                    condition_col=spec["dataset"].get("condition_col", "type"),
                    y_scaler=y_scaler,
                )
                test_metrics = None
                if allow_test_holdout:
                    test_metrics = _compute_real_scale_metrics(
                        model=model,
                        X=X_test,
                        y=y_test,
                        condition_col=spec["dataset"].get("condition_col", "type"),
                        y_scaler=y_scaler,
                    )
                after_dirs = _existing_run_dirs()
                new_run_dir = _infer_new_run_dir(before_dirs, after_dirs, base_name)
                row.update(
                    {
                        "status": "ok",
                        "objective_value": _macro_metric(val_metrics, objective_metric_path),
                        "val_macro_rrmse": _macro_metric(val_metrics, objective_metric_path),
                        "val_overall_rrmse": _metric_overall(val_metrics, "rrmse"),
                        "val_overall_rmse": _metric_overall(val_metrics, "rmse"),
                        "train_macro_rrmse": _macro_metric(train_metrics, objective_metric_path),
                        "test_macro_rrmse": (
                            _macro_metric(test_metrics, objective_metric_path) if test_metrics is not None else None
                        ),
                        "run_dir": str(new_run_dir) if new_run_dir is not None else None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            finally:
                row["runtime_s"] = round(time.time() - started, 3)
                run_rows_map[key] = row
                pd.DataFrame(run_rows_map.values(), columns=REPORT_COLUMNS).to_csv(runs_path, index=False)

    runs_df = pd.DataFrame(run_rows_map.values(), columns=REPORT_COLUMNS)
    runs_df.to_csv(runs_path, index=False)

    if winner_only:
        winner_test_df = runs_df
        if (report_dir / "seed_runs.csv").exists():
            base_runs_df = pd.read_csv(report_dir / "seed_runs.csv")
        else:
            base_runs_df = pd.DataFrame(columns=REPORT_COLUMNS)
        summary_df = _aggregate_runs(base_runs_df)
        param_df = _build_parameter_slices(summary_df, list(spec["variants"]))
        risk_df = pd.DataFrame([_variant_risk_row(v) for v in spec["variants"]])
        device_df = _build_device_summary(base_runs_df)
        _write_markdown_report(
            report_dir=report_dir,
            spec=spec,
            runs_df=base_runs_df,
            summary_df=summary_df,
            param_df=param_df,
            risk_df=risk_df,
            device_df=device_df,
            winner_test_df=winner_test_df,
        )
        return

    summary_df = _aggregate_runs(runs_df)
    param_df = _build_parameter_slices(summary_df, variants)
    risk_df = pd.DataFrame([_variant_risk_row(v) for v in variants])
    device_df = _build_device_summary(runs_df)

    summary_df.to_csv(report_dir / "config_summary.csv", index=False)
    device_df.to_csv(report_dir / "device_summary.csv", index=False)
    param_df.to_csv(report_dir / "parameter_slices.csv", index=False)
    risk_df.to_csv(report_dir / "variant_risk_checks.csv", index=False)

    _write_markdown_report(
        report_dir=report_dir,
        spec=spec,
        runs_df=runs_df,
        summary_df=summary_df,
        param_df=param_df,
        risk_df=risk_df,
        device_df=device_df,
        winner_test_df=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-revalidacion 3x40 del baseline MLP de F7.")
    parser.add_argument(
        "--spec",
        default="config/f7_mlp_baseline_revalidation_v1.yaml",
        help="Ruta al spec YAML de la mini-revalidacion.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device preferido para training: auto | mps | cuda | cpu.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Etiqueta para separar benchmark runs, por ejemplo 'cpu' o 'mps'.",
    )
    parser.add_argument(
        "--phase",
        choices=["val_grid", "winner_test_check"],
        default="val_grid",
        help="Fase a ejecutar: grid principal en val o chequeo final del ganador con test.",
    )
    parser.add_argument(
        "--include-test-for-all",
        action="store_true",
        help="Solo para debugging. Habilita test en todas las runs del grid; no recomendado.",
    )
    args = parser.parse_args()

    spec_path = ROOT_PATH / args.spec
    if args.phase == "winner_test_check":
        run_phase(spec_path=spec_path, device=args.device, run_label=args.run_label, winner_only=True)
    else:
        run_phase(
            spec_path=spec_path,
            device=args.device,
            run_label=args.run_label,
            include_test_for_all=bool(args.include_test_for_all),
            winner_only=False,
        )


if __name__ == "__main__":
    main()
