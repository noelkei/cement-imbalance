from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sets import DEFAULT_OFFICIAL_DATASET_NAME, load_or_create_raw_splits
from evaluation.metrics import select_metric
from training.train_xgboost import build_xgboost_design_matrix, train_xgboost_model


ROOT_PATH = ROOT
TYPE_CATEGORIES = [0, 1, 2]
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
    "train_val_balanced_score",
    "runtime_s",
    "best_iteration",
    "best_val_rmse_native",
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


def _outputs_dir(study_id: str) -> Path:
    return ROOT_PATH / "outputs" / "reports" / study_id


def _generated_config_dir(study_id: str) -> Path:
    return ROOT_PATH / "config" / "generated" / study_id


def _models_dir() -> Path:
    return ROOT_PATH / "outputs" / "models" / "xgboost"


def _phase_key(phase: str, run_label: str | None) -> str:
    return f"{phase}__{run_label}" if run_label else phase


def _load_bundle(spec: dict[str, Any]) -> dict[str, pd.DataFrame]:
    dataset_cfg = spec["dataset"]
    df_name = dataset_cfg.get("df_name") or DEFAULT_OFFICIAL_DATASET_NAME
    if df_name == "df_input_v6":
        df_name = DEFAULT_OFFICIAL_DATASET_NAME
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = load_or_create_raw_splits(
        df_name=df_name,
        split_mode=dataset_cfg.get("split_mode", "official"),
        verbose=False,
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def _canonical_xgb_dataset_manifest_path(dataset_name: str) -> Path:
    return (
        ROOT_PATH
        / "data"
        / "sets"
        / "official"
        / "init_temporal_processed_v1"
        / "xgboost"
        / dataset_name
        / "meta"
        / "manifest.json"
    )


def build_design_matrix(
    X: pd.DataFrame,
    *,
    type_categories: list[int],
    numeric_feature_order: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    return build_xgboost_design_matrix(
        X,
        type_categories=type_categories,
        numeric_feature_order=numeric_feature_order,
    )


def _balanced_score(train_macro_rrmse: float | None, val_macro_rrmse: float | None) -> float | None:
    if train_macro_rrmse is None or val_macro_rrmse is None:
        return None
    return float((train_macro_rrmse + val_macro_rrmse) / 2.0)


def _metric_overall(metrics: dict[str, Any], name: str) -> float | None:
    overall = metrics.get("overall")
    if not isinstance(overall, dict):
        return None
    value = overall.get(name)
    return float(value) if value is not None else None


def _macro_metric(metrics: dict[str, Any], path: str) -> float:
    return float(select_metric(metrics, path))


def _versioned_run_dir(base_name: str) -> tuple[str, Path]:
    root = _models_dir()
    root.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        run_id = f"{base_name}_v{version}"
        run_dir = root / run_id
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_id, run_dir
        version += 1


def _resolve_run_manifest_path(run_dir: Path) -> Path | None:
    stable = run_dir / "run_manifest.json"
    if stable.exists():
        return stable
    candidates = sorted(run_dir.glob("*_run_manifest.json"))
    return candidates[0] if candidates else None


def _find_latest_completed_run(base_name: str) -> Path | None:
    root = _models_dir()
    if not root.exists():
        return None
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith(base_name + "_v")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        if (run_dir / "results.yaml").exists() and _resolve_run_manifest_path(run_dir) is not None:
            return run_dir
    return None


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


def _variant_risk_row(spec: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    merged = dict(spec["protocol"]["base_config"])
    merged.update(variant.get("params", {}))
    return {
        "cfg_id": variant["cfg_id"],
        "rationale": variant.get("rationale"),
        "max_depth": merged.get("max_depth"),
        "min_child_weight": merged.get("min_child_weight"),
        "subsample": merged.get("subsample"),
        "colsample_bytree": merged.get("colsample_bytree"),
        "learning_rate": merged.get("learning_rate"),
        "reg_alpha": merged.get("reg_alpha"),
        "reg_lambda": merged.get("reg_lambda"),
        "gamma": merged.get("gamma"),
        "n_estimators": merged.get("n_estimators"),
        "early_stopping_rounds": merged.get("early_stopping_rounds"),
    }


def _build_trial_config(spec: dict[str, Any], variant: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg = {
        "study_id": spec["study"]["study_id"],
        "cfg_id": variant["cfg_id"],
        "seed": int(seed),
        "dataset": deepcopy(spec["dataset"]),
        "contract": {
            "closure_contract_id": "f7_contract_v1",
            "xgb_base_config_id": variant["cfg_id"],
            "seed_set_id": spec["protocol"]["seeds"]["seed_set_id"],
            "objective_metric_id": spec["protocol"]["objective_metric_id"],
            "allow_test_holdout_default": True,
        },
        "training": deepcopy(spec["protocol"]["base_config"]),
    }
    cfg["training"].update(variant.get("params", {}))
    cfg["training"]["random_state"] = int(seed)
    return cfg


def _train_one_run(
    *,
    bundle: dict[str, pd.DataFrame],
    trial_config: dict[str, Any],
    cfg_path: Path,
    cfg_id: str,
    seed: int,
    dataset_manifest_path: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    _, results = train_xgboost_model(
        bundle["X_train"],
        bundle["y_train"],
        bundle["X_val"],
        bundle["y_val"],
        X_test=bundle["X_test"],
        y_test=bundle["y_test"],
        allow_test_holdout=True,
        seed=int(seed),
        config_filename=cfg_path,
        config=trial_config,
        base_name=trial_config["study_id"],
        run_id=run_dir.name,
        run_dir=run_dir,
        verbose=False,
        evaluation_context={
            "dataset_name": trial_config["dataset"]["dataset_name"],
            "dataset_manifest_path": str(dataset_manifest_path),
            "split_id": trial_config["dataset"]["split_id"],
            "contract_id": trial_config["contract"]["closure_contract_id"],
            "seed_set_id": trial_config["contract"]["seed_set_id"],
            "xgb_base_config_id": cfg_id,
            "objective_metric_id": trial_config["contract"]["objective_metric_id"],
            "dataset_level_axes": deepcopy(trial_config["dataset"]["representation"]),
        },
    )
    aux_manifest = json.loads((run_dir / "aux_manifest.json").read_text(encoding="utf-8"))
    feature_names = json.loads(Path(aux_manifest["feature_names_path"]).read_text(encoding="utf-8"))
    X_val_matrix, _, _, _ = build_design_matrix(
        bundle["X_val"],
        type_categories=TYPE_CATEGORIES,
        numeric_feature_order=aux_manifest["numeric_feature_order"],
    )
    return (
        results,
        aux_manifest,
        {"X_val": X_val_matrix, "feature_names": feature_names},
        dict(results["raw_real"]),
    )


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
        mean_train_macro_rrmse=("train_macro_rrmse", "mean"),
        mean_balanced_score=("train_val_balanced_score", "mean"),
        mean_runtime_s=("runtime_s", "mean"),
        median_runtime_s=("runtime_s", "median"),
        max_runtime_s=("runtime_s", "max"),
        mean_best_iteration=("best_iteration", "mean"),
        mean_best_val_rmse_native=("best_val_rmse_native", "mean"),
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
            mean_balanced_score=("train_val_balanced_score", "mean"),
            mean_best_iteration=("best_iteration", "mean"),
        )
        .reset_index()
    )


def _write_markdown_report(
    *,
    report_dir: Path,
    spec: dict[str, Any],
    summary_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    device_df: pd.DataFrame,
    shap_df: pd.DataFrame | None,
) -> None:
    lines: list[str] = []
    lines.append(f"# {spec['study']['study_id']}")
    lines.append("")
    lines.append("## Contexto")
    lines.append("")
    lines.append(f"- Dataset fijo: `{spec['dataset']['dataset_name']}`")
    lines.append("- Representacion fija: `raw_numeric_plus_type_onehot`")
    lines.append("- Metrica guia: `raw_real.macro.rrmse` en `val`")
    lines.append("")
    lines.append("## Ranking agregado por configuracion")
    lines.append("")
    if summary_df.empty:
        lines.append("No hay resultados agregados todavia.")
    else:
        lines.append("```text")
        lines.append(summary_df.head(12).to_string(index=False))
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
    lines.append("## Benchmark de runtime")
    lines.append("")
    if device_df.empty:
        lines.append("No hay benchmark todavia.")
    else:
        lines.append("```text")
        lines.append(device_df.to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## SHAP smoke")
    lines.append("")
    if shap_df is None or shap_df.empty:
        lines.append("Aun no se ha ejecutado el smoke test de SHAP.")
    else:
        lines.append("```text")
        lines.append(shap_df.to_string(index=False))
        lines.append("```")
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_val_grid(*, spec_path: Path, run_label: str | None) -> None:
    spec = _load_yaml(spec_path)
    study_id = spec["study"]["study_id"]
    report_dir = _outputs_dir(study_id)
    generated_cfg_dir = _generated_config_dir(study_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)
    bundle = _load_bundle(spec)
    dataset_manifest_path = _canonical_xgb_dataset_manifest_path(spec["dataset"]["dataset_name"])
    objective_metric_path = spec["protocol"]["objective_metric_path"]
    phase = _phase_key("val_grid", run_label)
    runs_path = report_dir / "seed_runs.csv"
    run_rows_map = _load_existing_rows(runs_path, phase)

    for variant in spec["variants"]:
        cfg_id = variant["cfg_id"]
        for seed in spec["protocol"]["seeds"]["values"]:
            seed = int(seed)
            key = (phase, run_label, cfg_id, seed)
            existing = run_rows_map.get(key)
            if existing is not None and str(existing.get("status")) == "ok":
                continue

            base_name = f"{study_id}__{phase}__{cfg_id}__seed{seed}"
            recovered = _find_latest_completed_run(base_name)
            if recovered is not None:
                with (recovered / "results.yaml").open("r", encoding="utf-8") as handle:
                    results = yaml.safe_load(handle) or {}
                val_metrics = results["raw_real"]["val"]
                train_metrics = results["raw_real"]["train"]
                manifest_path = _resolve_run_manifest_path(recovered)
                training_summary = {}
                if manifest_path is not None:
                    training_summary = json.loads(manifest_path.read_text(encoding="utf-8")).get("training_summary", {})
                run_rows_map[key] = {
                    "phase": phase,
                    "run_label": run_label,
                    "cfg_id": cfg_id,
                    "seed": seed,
                    "status": "ok",
                    "objective_metric_id": spec["protocol"]["objective_metric_id"],
                    "objective_value": _macro_metric(val_metrics, objective_metric_path),
                    "val_macro_rrmse": _macro_metric(val_metrics, objective_metric_path),
                    "val_overall_rrmse": _metric_overall(val_metrics, "rrmse"),
                    "val_overall_rmse": _metric_overall(val_metrics, "rmse"),
                    "train_macro_rrmse": _macro_metric(train_metrics, objective_metric_path),
                    "train_val_balanced_score": _balanced_score(_macro_metric(train_metrics, objective_metric_path), _macro_metric(val_metrics, objective_metric_path)),
                    "runtime_s": training_summary.get("runtime_s"),
                    "best_iteration": training_summary.get("best_iteration"),
                    "best_val_rmse_native": training_summary.get("best_val_rmse_native"),
                    "device": "cpu",
                    "run_dir": str(recovered),
                    "error": None,
                }
                pd.DataFrame(run_rows_map.values(), columns=REPORT_COLUMNS).to_csv(runs_path, index=False)
                continue

            cfg = _build_trial_config(spec, variant, seed)
            cfg_path = generated_cfg_dir / f"{cfg_id}__seed{seed}.yaml"
            _dump_yaml(cfg, cfg_path)

            run_id, run_dir = _versioned_run_dir(base_name)
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
                "train_val_balanced_score": None,
                "runtime_s": None,
                "best_iteration": None,
                "best_val_rmse_native": None,
                "device": "cpu",
                "run_dir": str(run_dir),
                "error": None,
            }
            try:
                results, aux_manifest, _, metrics = _train_one_run(
                    bundle=bundle,
                    trial_config=cfg,
                    cfg_path=cfg_path,
                    cfg_id=cfg_id,
                    seed=seed,
                    dataset_manifest_path=dataset_manifest_path,
                    run_dir=run_dir,
                )
                train_macro = _macro_metric(metrics["train"], objective_metric_path)
                val_macro = _macro_metric(metrics["val"], objective_metric_path)
                training_summary = dict(results.get("training_summary") or {})
                if "runtime_s" not in training_summary:
                    training_summary["runtime_s"] = round(time.time() - started, 3)
                row.update(
                    {
                        "status": "ok",
                        "objective_value": val_macro,
                        "val_macro_rrmse": val_macro,
                        "val_overall_rrmse": _metric_overall(metrics["val"], "rrmse"),
                        "val_overall_rmse": _metric_overall(metrics["val"], "rmse"),
                        "train_macro_rrmse": train_macro,
                        "train_val_balanced_score": _balanced_score(train_macro, val_macro),
                        "runtime_s": training_summary["runtime_s"],
                        "best_iteration": training_summary["best_iteration"],
                        "best_val_rmse_native": training_summary["best_val_rmse_native"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            finally:
                if row["runtime_s"] is None:
                    row["runtime_s"] = round(time.time() - started, 3)
                run_rows_map[key] = row
                pd.DataFrame(run_rows_map.values(), columns=REPORT_COLUMNS).to_csv(runs_path, index=False)

    runs_df = pd.DataFrame(run_rows_map.values(), columns=REPORT_COLUMNS)
    summary_df = _aggregate_runs(runs_df)
    risk_df = pd.DataFrame([_variant_risk_row(spec, v) for v in spec["variants"]])
    device_df = _build_device_summary(runs_df)
    summary_df.to_csv(report_dir / "config_summary.csv", index=False)
    risk_df.to_csv(report_dir / "variant_risk_checks.csv", index=False)
    device_df.to_csv(report_dir / "device_summary.csv", index=False)
    _write_markdown_report(report_dir=report_dir, spec=spec, summary_df=summary_df, risk_df=risk_df, device_df=device_df, shap_df=None)


def run_shap_smoke(*, spec_path: Path, run_label: str | None) -> None:
    try:
        import shap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The `shap` package is required only for `--phase shap_smoke`. "
            "Install it in the active environment before running the SHAP smoke test."
        ) from exc

    spec = _load_yaml(spec_path)
    study_id = spec["study"]["study_id"]
    report_dir = _outputs_dir(study_id)
    summary_df = pd.read_csv(report_dir / "config_summary.csv")
    if "run_label" in summary_df.columns:
        summary_df = summary_df[summary_df["run_label"].fillna("") == (run_label or "")]
    top_k = int(spec["protocol"].get("top_k_shap", 3))
    top_cfgs = summary_df.sort_values(["all_runs_ok", "mean_val_macro_rrmse"], ascending=[False, True]).head(top_k)["cfg_id"].tolist()
    runs_df = pd.read_csv(report_dir / "seed_runs.csv")
    if "run_label" in runs_df.columns:
        runs_df = runs_df[runs_df["run_label"].fillna("") == (run_label or "")]
    bundle = _load_bundle(spec)
    X_val, _, feature_names, _ = build_design_matrix(
        bundle["X_val"],
        type_categories=TYPE_CATEGORIES,
        numeric_feature_order=[col for col in bundle["X_train"].columns if col not in {"post_cleaning_index", "type"}],
    )
    sample_size = int(spec["protocol"].get("shap_sample_size", 128))
    X_val_sample = X_val[:sample_size]
    shap_rows: list[dict[str, Any]] = []

    for cfg_id in top_cfgs:
        sub = runs_df[(runs_df["cfg_id"] == cfg_id) & (runs_df["status"] == "ok")].sort_values("objective_value")
        if sub.empty:
            continue
        run_dir = Path(str(sub.iloc[0]["run_dir"]))
        manifest_path = _resolve_run_manifest_path(run_dir)
        if manifest_path is None:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_path = Path(str(manifest["model_path"]))
        started = time.time()
        row = {
            "cfg_id": cfg_id,
            "run_dir": str(run_dir),
            "status": "failed",
            "shap_runtime_s": None,
            "sample_size": sample_size,
            "n_features": X_val_sample.shape[1],
            "mean_abs_shap": None,
            "error": None,
        }
        try:
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val_sample)
            mean_abs = float(np.abs(shap_values).mean())
            row.update(
                {
                    "status": "ok",
                    "mean_abs_shap": mean_abs,
                }
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        finally:
            row["shap_runtime_s"] = round(time.time() - started, 3)
            shap_rows.append(row)

    shap_df = pd.DataFrame(shap_rows)
    shap_df.to_csv(report_dir / "shap_smoke.csv", index=False)
    risk_df = pd.read_csv(report_dir / "variant_risk_checks.csv") if (report_dir / "variant_risk_checks.csv").exists() else pd.DataFrame()
    device_df = pd.read_csv(report_dir / "device_summary.csv") if (report_dir / "device_summary.csv").exists() else pd.DataFrame()
    _write_markdown_report(
        report_dir=report_dir,
        spec=spec,
        summary_df=pd.read_csv(report_dir / "config_summary.csv"),
        risk_df=risk_df,
        device_df=device_df,
        shap_df=shap_df,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-revalidacion 3x20 del baseline XGBoost para F7.")
    parser.add_argument("--spec", default="config/f7_xgb_baseline_revalidation_v1.yaml")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--phase", choices=["val_grid", "shap_smoke"], default="val_grid")
    args = parser.parse_args()

    spec_path = ROOT_PATH / args.spec
    if args.phase == "shap_smoke":
        run_shap_smoke(spec_path=spec_path, run_label=args.run_label)
    else:
        run_val_grid(spec_path=spec_path, run_label=args.run_label)


if __name__ == "__main__":
    main()
