from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml

from data.sets import DEFAULT_OFFICIAL_DATASET_NAME, official_raw_bundle_manifest_path
from evaluation.predictive_metrics import (
    build_predictive_metric_spaces,
    build_predictive_results_payload,
)
from evaluation.xgb_interpretability import compute_and_persist_xgb_interpretability
from evaluation.artifacts import (
    build_artifact_index_payload,
    build_prediction_sidecar_payload_from_raw,
    uses_f7_stable_single_copy_policy,
    write_prediction_sidecar,
)
from evaluation.raw_metric_contract import (
    build_raw_inversion_status,
    validate_raw_metric_contract,
)
from evaluation.results import build_run_context, save_canonical_run_artifacts
from training.utils import ROOT_PATH, flowpre_log, load_yaml_config, setup_training_logs_and_dirs


DEFAULT_XGB_TYPE_CATEGORIES = [0, 1, 2]


def build_xgboost_design_matrix(
    X: pd.DataFrame,
    *,
    type_categories: list[int],
    numeric_feature_order: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    type_values = X["type"].astype(int).to_numpy()
    unknown_types = sorted(set(type_values.tolist()) - set(type_categories))
    if unknown_types:
        raise ValueError(f"Unexpected type codes found: {unknown_types}")

    if numeric_feature_order is None:
        numeric_feature_order = [col for col in X.columns if col not in {"post_cleaning_index", "type", "is_synth"}]

    numeric = X[numeric_feature_order].to_numpy(dtype=np.float32)
    type_onehot = np.zeros((len(X), len(type_categories)), dtype=np.float32)
    category_to_col = {category: idx for idx, category in enumerate(type_categories)}
    for row_idx, category in enumerate(type_values):
        type_onehot[row_idx, category_to_col[int(category)]] = 1.0

    matrix = np.concatenate([type_onehot, numeric], axis=1)
    feature_names = [f"type_{category}" for category in type_categories] + numeric_feature_order
    post_idx = X["post_cleaning_index"].to_numpy(dtype=np.int64)
    return matrix, type_values, feature_names, post_idx


def _resolve_run_dir(
    *,
    base_name: str,
    config_filename: str | Path,
    config: Mapping[str, Any],
    run_id: str | None,
    run_dir: str | Path | None,
    verbose: bool,
) -> tuple[Path, str, Path | None]:
    if run_dir is not None:
        resolved_run_dir = Path(run_dir)
        resolved_run_dir.mkdir(parents=True, exist_ok=True)
        resolved_run_id = str(run_id or resolved_run_dir.name)
        return resolved_run_dir, resolved_run_id, None

    versioned_dir, versioned_name, log_file_path, _ = setup_training_logs_and_dirs(
        base_name,
        str(config_filename),
        dict(config),
        verbose,
        should_save_states=False,
        log_training=bool(dict(config).get("training", {}).get("log_training", False)),
        subdir="xgboost",
    )
    return versioned_dir, versioned_name, Path(log_file_path) if log_file_path else None


def _resolve_config_path(config_filename: str | Path) -> Path:
    path = Path(config_filename)
    if not path.parent or path.parent == Path("."):
        path = ROOT_PATH / "config" / path
    if not path.suffix:
        path = path.with_suffix(".yaml")
    return path


def _load_dataset_manifest_payload(dataset_manifest_path: str | Path | None) -> dict[str, Any]:
    if dataset_manifest_path is None:
        return {}
    candidate = Path(dataset_manifest_path)
    if not candidate.is_absolute():
        candidate = ROOT_PATH / candidate
    if not candidate.exists():
        return {}
    return json.loads(candidate.read_text(encoding="utf-8"))


def _build_xgb_run_level_axes(
    *,
    objective_metric_id: str,
    xgb_config: Mapping[str, Any],
) -> dict[str, Any]:
    tracked_keys = (
        "objective",
        "eval_metric",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "gamma",
        "tree_method",
        "max_bin",
        "early_stopping_rounds",
    )
    return {
        "objective_metric": {
            "id": objective_metric_id,
            "implemented_now_scope": "post_run_selection",
            "trainer_eval_metric": str(xgb_config.get("eval_metric", "rmse")),
        },
        "booster_config": {
            key: xgb_config.get(key)
            for key in tracked_keys
            if key in xgb_config
        },
    }


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    *,
    X_test: pd.DataFrame | None = None,
    y_test: pd.DataFrame | None = None,
    allow_test_holdout: bool = False,
    seed: int | None = None,
    config_filename: str | Path = "f7_xgb_base_v1.yaml",
    config: Optional[dict[str, Any]] = None,
    base_name: str = "xgboost",
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    verbose: bool = True,
    evaluation_context: Optional[Dict[str, Any]] = None,
) -> tuple[xgb.XGBRegressor, dict[str, Any]]:
    config_path = _resolve_config_path(config_filename)
    loaded_config = config or load_yaml_config(config_path)
    contract_cfg = loaded_config.get("contract", {})
    train_cfg = dict(loaded_config.get("training", {}))

    if seed is None:
        seed = int(train_cfg.get("random_state", 42))
    train_cfg["random_state"] = int(seed)

    eval_ctx = dict(evaluation_context or {})
    dataset_manifest_path = eval_ctx.get("dataset_manifest_path")
    if dataset_manifest_path is None:
        dataset_manifest_path = official_raw_bundle_manifest_path()
    dataset_manifest_payload = _load_dataset_manifest_payload(dataset_manifest_path)
    dataset_level_axes = dict(eval_ctx.get("dataset_level_axes") or dataset_manifest_payload.get("dataset_level_axes") or {})

    resolved_run_dir, resolved_run_id, log_file_path = _resolve_run_dir(
        base_name=base_name,
        config_filename=config_path,
        config=loaded_config,
        run_id=run_id,
        run_dir=run_dir,
        verbose=verbose,
    )
    if log_file_path is not None and log_file_path.exists():
        log_file_path.unlink()

    type_categories = eval_ctx.get("type_categories") or DEFAULT_XGB_TYPE_CATEGORIES
    numeric_feature_order = eval_ctx.get("numeric_feature_order")
    X_train_matrix, c_train, feature_names, _ = build_xgboost_design_matrix(
        X_train,
        type_categories=list(type_categories),
        numeric_feature_order=numeric_feature_order,
    )
    X_val_matrix, c_val, _, _ = build_xgboost_design_matrix(
        X_val,
        type_categories=list(type_categories),
        numeric_feature_order=numeric_feature_order,
    )
    X_test_matrix = None
    c_test = None
    if X_test is not None and y_test is not None and allow_test_holdout:
        X_test_matrix, c_test, _, _ = build_xgboost_design_matrix(
            X_test,
            type_categories=list(type_categories),
            numeric_feature_order=numeric_feature_order,
        )

    target_col = str(eval_ctx.get("target_col", "init"))
    y_train_np = y_train[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_val_np = y_val[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_test_np = (
        y_test[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        if y_test is not None and allow_test_holdout
        else None
    )

    estimator_cfg = {
        key: value
        for key, value in train_cfg.items()
        if key not in {"save_model", "save_results"}
    }
    model = xgb.XGBRegressor(**estimator_cfg)
    train_started_at = time.perf_counter()
    model.fit(
        X_train_matrix,
        y_train_np.reshape(-1),
        eval_set=[(X_train_matrix, y_train_np.reshape(-1)), (X_val_matrix, y_val_np.reshape(-1))],
        verbose=False,
    )
    training_runtime_s = round(time.perf_counter() - train_started_at, 3)

    best_iteration = int(model.best_iteration) + 1 if model.best_iteration is not None else int(model.n_estimators)
    best_val_rmse_native = float(model.best_score) if model.best_score is not None else float("nan")
    predictions_by_split: dict[str, Any] = {
        "train": model.predict(X_train_matrix, iteration_range=(0, best_iteration)).reshape(-1, 1),
        "val": model.predict(X_val_matrix, iteration_range=(0, best_iteration)).reshape(-1, 1),
    }
    targets_by_split: dict[str, Any] = {
        "train": y_train_np,
        "val": y_val_np,
    }
    class_codes_by_split: dict[str, Any] = {
        "train": c_train,
        "val": c_val,
    }
    if X_test_matrix is not None and y_test_np is not None and c_test is not None:
        predictions_by_split["test"] = model.predict(X_test_matrix, iteration_range=(0, best_iteration)).reshape(-1, 1)
        targets_by_split["test"] = y_test_np
        class_codes_by_split["test"] = c_test

    metric_spaces = build_predictive_metric_spaces(
        predictions_by_split=predictions_by_split,
        targets_by_split=targets_by_split,
        class_codes_by_split=class_codes_by_split,
        native_enabled=False,
        raw_real_required=True,
        raw_real_from_native_when_no_scaler=False,
        y_scaler=None,
        device=torch.device("cpu"),
    )
    raw_inversion_status = build_raw_inversion_status(
        y_transform="raw",
        y_scaler_present=False,
        target_scaler_artifact=None,
        raw_real_available="raw_real" in metric_spaces,
    )
    raw_metric_contract_validation = validate_raw_metric_contract(
        metric_spaces=metric_spaces,
        test_enabled=bool("test" in predictions_by_split and allow_test_holdout),
        raw_inversion_status=raw_inversion_status,
        value_space_default="raw_real",
    )
    if not raw_metric_contract_validation["campaign_valid"]:
        raise ValueError(
            "F7 raw metric contract validation failed: "
            + ", ".join(str(item) for item in raw_metric_contract_validation["missing_items"])
        )

    objective_metric_id = str(
        eval_ctx.get("objective_metric_id", contract_cfg.get("objective_metric_id", "raw_real.macro.rrmse"))
    )
    base_config_id = (
        eval_ctx.get("xgb_base_config_id")
        or eval_ctx.get("base_config_id")
        or contract_cfg.get("xgb_base_config_id")
        or config_path.stem
    )
    run_level_axes = _build_xgb_run_level_axes(
        objective_metric_id=objective_metric_id,
        xgb_config=train_cfg,
    )
    training_summary = {
        "status": "completed",
        "device": "cpu",
        "runtime_s": training_runtime_s,
        "best_iteration": best_iteration,
        "best_val_rmse_native": best_val_rmse_native,
    }

    results = {
        "seed": int(seed),
        "best_iteration": best_iteration,
        "best_val_rmse_native": best_val_rmse_native,
        "feature_names_x": feature_names,
        "target_names": [target_col],
        "training_summary": training_summary,
        "comparison_contract": {
            "contract_id": eval_ctx.get("contract_id", contract_cfg.get("closure_contract_id")),
            "comparison_group_id": eval_ctx.get("comparison_group_id"),
            "seed_set_id": eval_ctx.get("seed_set_id", contract_cfg.get("seed_set_id")),
            "base_config_id": str(base_config_id),
            "objective_metric_id": objective_metric_id,
            "dataset_level_axes": dataset_level_axes,
            "run_level_axes": run_level_axes,
        },
    }
    results.update(
        build_predictive_results_payload(
            metric_spaces=metric_spaces,
            raw_metric_contract_validation=raw_metric_contract_validation,
            raw_inversion_status=raw_inversion_status,
        )
    )

    f7_interpretability_required = str(results["comparison_contract"]["contract_id"]) == "f7_contract_v1"
    xgb_interpretability_bundle = None
    xgb_interpretability_summary = None
    xgb_interpretability_validation = None
    campaign_valid_interpretability = not f7_interpretability_required
    campaign_valid_f7 = bool(raw_metric_contract_validation["campaign_valid"]) and bool(campaign_valid_interpretability)
    if f7_interpretability_required:
        split_arrays = {
            "val": (X_val_matrix, c_val),
            "test": (X_test_matrix, c_test) if X_test_matrix is not None and c_test is not None and allow_test_holdout else None,
        }
        xgb_interpretability_bundle = compute_and_persist_xgb_interpretability(
            model=model,
            out_dir=resolved_run_dir,
            feature_names=feature_names,
            x_train_matrix=X_train_matrix,
            c_train=c_train,
            split_arrays=split_arrays,
            predictions_by_split={k: v for k, v in predictions_by_split.items() if k in {"val", "test"}},
            run_mode=raw_metric_contract_validation["run_mode"],
            best_iteration=best_iteration,
            target_names=[target_col],
        )
        xgb_interpretability_summary = dict(xgb_interpretability_bundle["summary"])
        xgb_interpretability_validation = dict(xgb_interpretability_bundle["validation"])
        campaign_valid_interpretability = bool(
            xgb_interpretability_validation["campaign_valid_interpretability"]
        )
        campaign_valid_f7 = bool(raw_metric_contract_validation["campaign_valid"]) and bool(campaign_valid_interpretability)
        if not campaign_valid_interpretability:
            raise ValueError(
                "F7 XGBoost interpretability contract validation failed: "
                + ", ".join(str(item) for item in xgb_interpretability_validation["missing_items"])
            )

    results.update(
        {
            "xgb_interpretability": xgb_interpretability_summary,
            "xgb_interpretability_validation": xgb_interpretability_validation,
            "campaign_valid_interpretability": campaign_valid_interpretability,
            "campaign_valid_f7": campaign_valid_f7,
        }
    )

    results_path = resolved_run_dir / "results.yaml"
    with open(results_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(results, handle, sort_keys=False)

    analysis_contracts = dict(eval_ctx.get("analysis_contracts") or {})
    use_stable_single_copy = uses_f7_stable_single_copy_policy(analysis_contracts)
    feature_names_path = resolved_run_dir / f"{resolved_run_id}_feature_names.json"
    feature_names_path.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    model_path = resolved_run_dir / f"{resolved_run_id}_xgb_model.json"
    model.save_model(model_path)
    stable_config_snapshot_path = resolved_run_dir / "config.yaml"
    if config_path.exists():
        if use_stable_single_copy:
            shutil.copy2(config_path, stable_config_snapshot_path)
        else:
            versioned_config_snapshot_path = resolved_run_dir / f"{resolved_run_id}.yaml"
            shutil.copy2(config_path, versioned_config_snapshot_path)
            shutil.copy2(versioned_config_snapshot_path, stable_config_snapshot_path)

    run_context = build_run_context(
        model_family="xgboost",
        run_id=resolved_run_id,
        seed=seed,
        config=loaded_config,
        config_path=config_path,
        dataset_name=str(eval_ctx.get("dataset_name", dataset_manifest_payload.get("dataset_name", DEFAULT_OFFICIAL_DATASET_NAME))),
        dataset_manifest_path=dataset_manifest_path,
        split_id=str(eval_ctx.get("split_id", dataset_manifest_payload.get("split_id", "init_temporal_processed_v1"))),
        split_manifest_path=eval_ctx.get("split_manifest_path"),
        upstream_variant_fingerprint=eval_ctx.get("upstream_variant_fingerprint"),
        contract_id=results["comparison_contract"]["contract_id"],
        comparison_group_id=results["comparison_contract"]["comparison_group_id"],
        seed_set_id=results["comparison_contract"]["seed_set_id"],
        seed_panel_path=eval_ctx.get("seed_panel_path"),
        base_config_id=results["comparison_contract"]["base_config_id"],
        objective_metric_id=results["comparison_contract"]["objective_metric_id"],
        dataset_level_axes=results["comparison_contract"]["dataset_level_axes"],
        run_level_axes=results["comparison_contract"]["run_level_axes"],
        training_summary=training_summary,
        test_enabled=bool("test" in predictions_by_split and allow_test_holdout),
        campaign_id=eval_ctx.get("campaign_id"),
        dataset_candidate_id=eval_ctx.get("dataset_candidate_id"),
        run_spec_id=eval_ctx.get("run_spec_id"),
        trial_id=eval_ctx.get("trial_id"),
        raw_metric_contract_id=raw_metric_contract_validation["raw_metric_contract_id"],
        raw_metric_contract_validation=raw_metric_contract_validation,
        run_mode=raw_metric_contract_validation["run_mode"],
        campaign_valid=raw_metric_contract_validation["campaign_valid"],
        raw_inversion_status=raw_inversion_status,
    )
    split_meta = {
        "val": X_val[[col for col in ("post_cleaning_index", "type", "is_synth") if col in X_val.columns]].copy(),
        "test": X_test[[col for col in ("post_cleaning_index", "type", "is_synth") if col in X_test.columns]].copy()
        if X_test is not None and allow_test_holdout
        else None,
    }
    prediction_sidecar_payloads, _ = build_prediction_sidecar_payload_from_raw(
        split_meta=split_meta,
        predictions_by_split={k: v for k, v in predictions_by_split.items() if k in {"val", "test"}},
        targets_by_split={k: v for k, v in targets_by_split.items() if k in {"val", "test"}},
        include_splits=["val", "test"] if allow_test_holdout and "test" in predictions_by_split else ["val"],
    )
    prediction_sidecar_path = write_prediction_sidecar(
        out_dir=resolved_run_dir,
        split_payloads=prediction_sidecar_payloads,
        target_names=[target_col],
    )
    run_manifest_path, metrics_long_path = save_canonical_run_artifacts(
        results=results,
        context=run_context,
        out_dir=resolved_run_dir,
        stem=resolved_run_id,
        filename_policy="stable_single_copy" if use_stable_single_copy else "versioned_aliases",
        manifest_extra_fields=build_artifact_index_payload(
            model_family="xgboost",
            results_path=results_path,
            run_manifest_path=resolved_run_dir / "run_manifest.json",
            metrics_long_path=resolved_run_dir / "metrics_long.csv",
            config_snapshot_path=stable_config_snapshot_path,
            prediction_sidecar_path=prediction_sidecar_path,
            model_artifact_path=model_path,
            extra_artifact_paths={
                "feature_names_json": feature_names_path,
                "aux_manifest_json": resolved_run_dir / "aux_manifest.json",
            },
            interpretability_artifact_paths=(
                {}
                if xgb_interpretability_bundle is None
                else xgb_interpretability_bundle["artifact_paths"]
            ),
            interpretability_status_override=(
                {}
                if xgb_interpretability_bundle is None
                else {
                    "interpretability_policy_status": "implemented_xgb_block_11_v1",
                    "interpretability_artifacts": {
                        key: (None if value is None else str(value))
                        for key, value in xgb_interpretability_bundle["artifact_paths"].items()
                    },
                    "interpretability_required_now": True,
                    "interpretability_required_for_shortlist": True,
                    "interpretability_required_for_finalist": True,
                    "family_specific_implementation_pending": False,
                }
            ),
        ),
    )

    stable_run_manifest_path = resolved_run_dir / "run_manifest.json"
    stable_metrics_long_path = resolved_run_dir / "metrics_long.csv"

    aux_manifest = {
        "feature_names_path": str(feature_names_path),
        "model_path": str(model_path),
        "numeric_feature_order": numeric_feature_order
        or [col for col in X_train.columns if col not in {"post_cleaning_index", "type", "is_synth"}],
        "type_categories": list(type_categories),
        "results_path": str(results_path),
        "run_manifest_path": str(run_manifest_path),
        "metrics_long_path": str(metrics_long_path),
        "prediction_sidecar_path": None if prediction_sidecar_path is None else str(prediction_sidecar_path),
        "config_snapshot_path": str(stable_config_snapshot_path),
    }
    aux_manifest_path = resolved_run_dir / "aux_manifest.json"
    with open(aux_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(aux_manifest, handle, indent=2, sort_keys=True)

    canonical_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    canonical_manifest.update(
        {
            "results_path": str(results_path),
            "feature_names_path": str(feature_names_path),
            "model_path": str(model_path),
            "aux_manifest_path": str(aux_manifest_path),
            "numeric_feature_order": aux_manifest["numeric_feature_order"],
            "type_categories": aux_manifest["type_categories"],
            "prediction_sidecar_path": None if prediction_sidecar_path is None else str(prediction_sidecar_path),
            "config_snapshot_path": str(stable_config_snapshot_path),
            "analysis_contracts": analysis_contracts,
            "parsed_factor_fields": dict(eval_ctx.get("parsed_factor_fields") or {}),
            "xgb_interpretability_contract_id": (
                None
                if xgb_interpretability_validation is None
                else xgb_interpretability_validation["xgb_interpretability_contract_id"]
            ),
            "xgb_interpretability_validation": xgb_interpretability_validation,
            "campaign_valid_interpretability": campaign_valid_interpretability,
            "campaign_valid_f7": campaign_valid_f7,
        }
    )
    run_manifest_path.write_text(json.dumps(canonical_manifest, indent=2, sort_keys=True), encoding="utf-8")
    if run_manifest_path != stable_run_manifest_path:
        stable_run_manifest_path.write_text(json.dumps(canonical_manifest, indent=2, sort_keys=True), encoding="utf-8")
    if metrics_long_path != stable_metrics_long_path:
        shutil.copy2(metrics_long_path, stable_metrics_long_path)

    flowpre_log(
        f"🧾 Saved canonical metrics table to: {metrics_long_path}",
        filename_or_path=str(log_file_path) if log_file_path is not None else None,
        verbose=verbose,
    )
    return model, results
