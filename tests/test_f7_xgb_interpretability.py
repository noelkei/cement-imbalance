from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from evaluation.meta_context import get_f7_analysis_contract_bundle_with_paths
from evaluation.xgb_interpretability import (
    compute_and_persist_xgb_interpretability,
    compute_xgb_feature_delta_matrix,
    compute_xgb_shap_values,
)
from training.train_xgboost import train_xgboost_model


def _make_xgb_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _split(start_idx: int, rows_per_class: int, init_offset: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_rows: list[dict[str, float | int]] = []
        y_rows: list[dict[str, float | int]] = []
        idx = start_idx
        for cls in (0, 1, 2):
            for step in range(rows_per_class):
                x_rows.append(
                    {
                        "post_cleaning_index": idx,
                        "type": cls,
                        "f1": float(cls) + (0.1 * step),
                        "f2": (float(cls) * 2.0) + (0.05 * step) + init_offset,
                    }
                )
                y_rows.append(
                    {
                        "post_cleaning_index": idx,
                        "init": 10.0 + (2.0 * cls) + (0.2 * step) + init_offset,
                    }
                )
                idx += 1
        return pd.DataFrame(x_rows), pd.DataFrame(y_rows)

    X_train, y_train = _split(0, 4, 0.0)
    X_val, y_val = _split(100, 3, 0.3)
    X_test, y_test = _split(200, 3, 0.6)
    return X_train, y_train, X_val, y_val, X_test, y_test


def _xgb_config() -> dict[str, object]:
    return {
        "contract": {
            "closure_contract_id": "f7_contract_v1",
            "xgb_base_config_id": "unit_xgb_interpretability",
            "seed_set_id": "f7_seed_panel_v1",
            "objective_metric_id": "raw_real.macro.rrmse",
            "allow_test_holdout_default": True,
        },
        "dataset": {
            "dataset_name": "official_raw_xgb_base_v1",
            "split_id": "init_temporal_processed_v1",
            "condition_col": "type",
            "target_col": "init",
            "feature_policy": "raw_numeric_plus_type_onehot",
            "x_transform": "raw",
            "y_transform": "raw",
            "synthetic_policy": "none",
        },
        "training": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 8,
            "learning_rate": 0.1,
            "max_depth": 2,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "tree_method": "hist",
            "max_bin": 64,
            "early_stopping_rounds": 3,
            "verbosity": 0,
        },
    }


class TestF7XGBInterpretability(unittest.TestCase):
    def test_compute_xgb_feature_delta_matrix_returns_expected_shape(self) -> None:
        X = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float32)
        y = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=4,
            max_depth=2,
            learning_rate=0.3,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(X, y, verbose=False)
        preds = model.predict(X).reshape(-1, 1)
        class_codes = np.asarray([0, 1, 1], dtype=np.int64)
        class_feature_means = {
            0: np.asarray([1.0, 0.0], dtype=np.float32),
            1: np.asarray([2.5, 1.0], dtype=np.float32),
        }
        delta = compute_xgb_feature_delta_matrix(
            model=model,
            x_eval=X,
            class_codes=class_codes,
            y_pred_raw=preds,
            class_feature_means=class_feature_means,
            best_iteration=4,
            chunk_size=2,
        )
        self.assertEqual(delta.shape, X.shape)
        self.assertFalse(np.isnan(delta).any())

    def test_compute_xgb_shap_values_returns_matrix_and_expected_value(self) -> None:
        X = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float32)
        y = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=4,
            max_depth=2,
            learning_rate=0.3,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(X, y, verbose=False)
        shap_values, expected_value = compute_xgb_shap_values(model=model, x_eval=X)
        self.assertEqual(shap_values.shape, X.shape)
        self.assertFalse(np.isnan(shap_values).any())
        self.assertIsInstance(expected_value, float)

    def test_compute_and_persist_xgb_interpretability_selection_run(self) -> None:
        X_train, y_train, X_val, _y_val, _X_test, _y_test = _make_xgb_frames()
        numeric_cols = ["f1", "f2"]
        feature_names = ["type_0", "type_1", "type_2"] + numeric_cols
        x_train_matrix = np.concatenate(
            [pd.get_dummies(X_train["type"]).reindex(columns=[0, 1, 2], fill_value=0).to_numpy(dtype=np.float32), X_train[numeric_cols].to_numpy(dtype=np.float32)],
            axis=1,
        )
        x_val_matrix = np.concatenate(
            [pd.get_dummies(X_val["type"]).reindex(columns=[0, 1, 2], fill_value=0).to_numpy(dtype=np.float32), X_val[numeric_cols].to_numpy(dtype=np.float32)],
            axis=1,
        )
        c_train = X_train["type"].to_numpy(dtype=np.int64)
        c_val = X_val["type"].to_numpy(dtype=np.int64)
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=8,
            learning_rate=0.1,
            max_depth=2,
            tree_method="hist",
            verbosity=0,
        )
        y_train_np = y_train["init"].to_numpy(dtype=np.float32)
        model.fit(x_train_matrix, y_train_np, eval_set=[(x_train_matrix, y_train_np)], verbose=False)
        preds = model.predict(x_val_matrix, iteration_range=(0, 8)).reshape(-1, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = compute_and_persist_xgb_interpretability(
                model=model,
                out_dir=tmpdir,
                feature_names=feature_names,
                x_train_matrix=x_train_matrix,
                c_train=c_train,
                split_arrays={"val": (x_val_matrix, c_val)},
                predictions_by_split={"val": preds},
                run_mode="selection_run",
                best_iteration=8,
                target_names=["init"],
            )
            summary = bundle["summary"]
            validation = bundle["validation"]
            shap_global = pd.read_csv(Path(tmpdir) / "xgb_shap_feature_influence_global.csv")
            perturb_global = pd.read_csv(Path(tmpdir) / "xgb_perturbation_feature_influence_global.csv")
            self.assertEqual(summary["available_splits"], ["val"])
            self.assertTrue(validation["campaign_valid_interpretability"])
            self.assertIn("type_0", shap_global["feature_name"].tolist())
            self.assertIn("share_abs_importance", shap_global.columns)
            self.assertIn("mean_abs_shap", shap_global.columns)
            self.assertIn("mean_abs_delta_pred_raw", perturb_global.columns)
            self.assertAlmostEqual(float(shap_global["share_abs_importance"].sum()), 1.0, places=6)

    def test_train_xgboost_model_holdout_persists_both_layers(self) -> None:
        X_train, y_train, X_val, y_val, X_test, y_test = _make_xgb_frames()
        dataset_manifest_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "sets"
            / "official"
            / "init_temporal_processed_v1"
            / "xgboost"
            / "official_raw_xgb_base_v1"
            / "meta"
            / "manifest.json"
        )
        config = _xgb_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cfg_path = tmpdir_path / "unit_xgb_interpretability.yaml"
            cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            run_dir = tmpdir_path / "xgb_run"
            _, results = train_xgboost_model(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test=X_test,
                y_test=y_test,
                allow_test_holdout=True,
                seed=1234,
                config_filename=cfg_path,
                config=config,
                base_name="unit_xgb",
                run_id="unit_xgb_run_v1",
                run_dir=run_dir,
                verbose=False,
                evaluation_context={
                    "dataset_name": "official_raw_xgb_base_v1",
                    "dataset_manifest_path": str(dataset_manifest_path),
                    "split_id": "init_temporal_processed_v1",
                    "contract_id": "f7_contract_v1",
                    "analysis_contracts": get_f7_analysis_contract_bundle_with_paths(),
                    "seed_set_id": "f7_seed_panel_v1",
                    "xgb_base_config_id": "unit_xgb_interpretability",
                    "objective_metric_id": "raw_real.macro.rrmse",
                    "dataset_level_axes": {
                        "x_transform": "raw",
                        "y_transform": "raw",
                        "synthetic_policy": "none",
                        "feature_policy": "raw_numeric_plus_type_onehot",
                    },
                },
            )
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "xgb_interpretability_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["campaign_valid_interpretability"])
            self.assertTrue(manifest["campaign_valid_f7"])
            self.assertEqual(summary["available_splits"], ["val", "test"])
            self.assertIn("shap", summary["required_layers"])
            self.assertIn("perturbation", summary["required_layers"])
            self.assertTrue((run_dir / "xgb_shap_feature_influence_global.csv").exists())
            self.assertTrue((run_dir / "xgb_perturbation_feature_influence_global.csv").exists())
            self.assertIn("xgb_interpretability", results)
            self.assertIn("xgb_interpretability_validation", results)
            self.assertFalse((run_dir / "unit_xgb_run_v1_run_manifest.json").exists())
            self.assertFalse((run_dir / "unit_xgb_run_v1_metrics_long.csv").exists())
            self.assertFalse((run_dir / "unit_xgb_run_v1_results.yaml").exists())
            self.assertFalse((run_dir / "unit_xgb_run_v1.yaml").exists())


if __name__ == "__main__":
    unittest.main()
