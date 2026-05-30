from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import compute_regression_metrics_from_preds
from evaluation.meta_context import get_f7_analysis_contract_bundle_with_paths
from evaluation.predictive_metrics import (
    build_predictive_metric_spaces,
    compute_predictive_metrics_for_split,
)
from evaluation.raw_metric_contract import (
    build_raw_inversion_status,
    validate_raw_metric_contract,
)
from training.train_xgboost import train_xgboost_model


def _metric_block() -> dict[str, object]:
    y_true = torch.tensor([[1.0], [1.5], [2.0], [2.5], [3.0], [3.5]], dtype=torch.float32)
    y_pred = torch.tensor([[1.1], [1.4], [2.1], [2.3], [3.05], [3.45]], dtype=torch.float32)
    classes = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    return compute_regression_metrics_from_preds(y_hat=y_pred, y=y_true, c=classes)


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


class TestF7RawMetricContract(unittest.TestCase):
    def test_compute_regression_metrics_keeps_overall_macro_and_per_class(self) -> None:
        metrics = _metric_block()
        self.assertIn("overall", metrics)
        self.assertIn("macro", metrics)
        self.assertIn("per_class", metrics)
        self.assertTrue(metrics["per_class"])
        for scope_name in ("overall", "macro"):
            scope = metrics[scope_name]
            self.assertIn("r2", scope)
            self.assertIn("mse", scope)
            self.assertIn("rmse", scope)
            self.assertIn("rrmse", scope)
            self.assertIn("mape", scope)

    def test_selection_run_requires_train_and_val_only(self) -> None:
        split_metrics = _metric_block()
        metric_spaces = {"raw_real": {"train": split_metrics, "val": split_metrics}}
        validation = validate_raw_metric_contract(
            metric_spaces=metric_spaces,
            test_enabled=False,
            raw_inversion_status=build_raw_inversion_status(
                y_transform="raw",
                y_scaler_present=False,
                target_scaler_artifact=None,
                raw_real_available=True,
            ),
            value_space_default="raw_real",
        )
        self.assertEqual(validation["run_mode"], "selection_run")
        self.assertTrue(validation["campaign_valid"])

    def test_holdout_run_requires_test(self) -> None:
        split_metrics = _metric_block()
        metric_spaces = {"raw_real": {"train": split_metrics, "val": split_metrics}}
        validation = validate_raw_metric_contract(
            metric_spaces=metric_spaces,
            test_enabled=True,
            raw_inversion_status=build_raw_inversion_status(
                y_transform="raw",
                y_scaler_present=False,
                target_scaler_artifact=None,
                raw_real_available=True,
            ),
            value_space_default="raw_real",
        )
        self.assertEqual(validation["run_mode"], "holdout_run")
        self.assertFalse(validation["campaign_valid"])
        self.assertIn("raw_real.test", validation["missing_items"])

    def test_validation_fails_if_required_metric_is_missing(self) -> None:
        split_metrics = _metric_block()
        broken = json.loads(json.dumps(split_metrics))
        del broken["macro"]["mape"]
        metric_spaces = {"raw_real": {"train": broken, "val": broken}}
        validation = validate_raw_metric_contract(
            metric_spaces=metric_spaces,
            test_enabled=False,
            raw_inversion_status=build_raw_inversion_status(
                y_transform="raw",
                y_scaler_present=False,
                target_scaler_artifact=None,
                raw_real_available=True,
            ),
            value_space_default="raw_real",
        )
        self.assertFalse(validation["campaign_valid"])
        self.assertIn("raw_real.train.macro.mape", validation["missing_items"])

    def test_predictive_metric_spaces_with_scaler_emit_native_and_raw_real(self) -> None:
        raw_targets = np.array([[10.0], [12.0], [14.0], [16.0]], dtype=np.float32)
        scaler = StandardScaler().fit(raw_targets)
        native_targets = scaler.transform(raw_targets)
        native_preds = scaler.transform(raw_targets + np.array([[0.1], [-0.1], [0.2], [-0.2]], dtype=np.float32))
        class_codes = np.array([0, 1, 1, 2], dtype=np.int64)
        metric_spaces = build_predictive_metric_spaces(
            predictions_by_split={"train": native_preds, "val": native_preds},
            targets_by_split={"train": native_targets, "val": native_targets},
            class_codes_by_split={"train": class_codes, "val": class_codes},
            native_enabled=True,
            raw_real_required=True,
            raw_real_from_native_when_no_scaler=False,
            y_scaler=scaler,
            device=torch.device("cpu"),
        )
        self.assertIn("native", metric_spaces)
        self.assertIn("raw_real", metric_spaces)
        self.assertIn("rrmse", metric_spaces["native"]["train"]["macro"])
        self.assertIn("rrmse", metric_spaces["raw_real"]["train"]["macro"])

    def test_predictive_metric_spaces_raw_without_scaler_still_emit_raw_real(self) -> None:
        class_codes = np.array([0, 1, 1, 2], dtype=np.int64)
        split_metrics = build_predictive_metric_spaces(
            predictions_by_split={"train": np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)},
            targets_by_split={"train": np.array([[1.1], [1.9], [3.1], [3.9]], dtype=np.float32)},
            class_codes_by_split={"train": class_codes},
            native_enabled=True,
            raw_real_required=True,
            raw_real_from_native_when_no_scaler=True,
            y_scaler=None,
            device=torch.device("cpu"),
        )
        self.assertIn("raw_real", split_metrics)
        self.assertIn("train", split_metrics["raw_real"])

    def test_train_xgboost_model_writes_canonical_raw_results_and_meta_ids(self) -> None:
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
        config = {
            "contract": {
                "closure_contract_id": "f7_contract_v1",
                "xgb_base_config_id": "unit_xgb_contract",
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

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cfg_path = tmpdir_path / "unit_xgb_contract.yaml"
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
                    "xgb_base_config_id": "unit_xgb_contract",
                    "objective_metric_id": "raw_real.macro.rrmse",
                    "dataset_level_axes": {
                        "x_transform": "raw",
                        "y_transform": "raw",
                        "synthetic_policy": "none",
                        "feature_policy": "raw_numeric_plus_type_onehot",
                    },
                },
            )

            self.assertIn("raw_real", results)
            self.assertIn("test", results["raw_real"])
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            metrics_long = pd.read_csv(run_dir / "metrics_long.csv")

            self.assertTrue(manifest["campaign_valid"])
            self.assertEqual(manifest["raw_metric_contract_id"], "f7_raw_metric_contract_v1")
            self.assertEqual(manifest["dataset_candidate_id"], "xgb__x-raw-base-v1__y-raw__syn-none")
            self.assertEqual(manifest["run_spec_id"], "runspec__xgb__unit_xgb_contract")
            self.assertEqual(
                manifest["trial_id"],
                "trial__f7_campaign_v1__xgb__x-raw-base-v1__y-raw__syn-none__runspec__xgb__unit_xgb_contract__seed-1234",
            )
            self.assertTrue(Path(manifest["model_path"]).exists())
            self.assertEqual(manifest["artifact_policy_id"], "f7_artifact_persistence_contract_v1")
            self.assertEqual(manifest["artifact_tier"], "per_run")
            self.assertEqual(manifest["model_artifact_policy"], "persist_every_run")
            self.assertTrue(manifest["artifact_availability"]["predictions_eval_raw_csv_gz"])
            self.assertTrue(manifest["artifact_availability"]["model_artifact"])
            self.assertEqual(manifest["interpretability_policy_status"], "implemented_xgb_block_11_v1")
            self.assertTrue(manifest["interpretability_required_now"])
            self.assertTrue(manifest["interpretability_required_for_shortlist"])
            self.assertTrue(manifest["interpretability_required_for_finalist"])
            self.assertTrue(manifest["campaign_valid_interpretability"])
            self.assertTrue(manifest["campaign_valid_f7"])
            self.assertEqual(manifest["xgb_interpretability_contract_id"], "f7_xgb_interpretability_contract_v1")
            self.assertTrue(manifest["artifact_availability"]["xgb_shap_feature_influence_global_csv"])
            self.assertTrue(manifest["artifact_availability"]["xgb_perturbation_feature_influence_global_csv"])
            sidecar_path = Path(manifest["artifact_paths"]["predictions_eval_raw_csv_gz"])
            self.assertTrue(sidecar_path.exists())
            sidecar_df = pd.read_csv(sidecar_path, compression="gzip")
            self.assertEqual(sorted(sidecar_df["split"].unique().tolist()), ["test", "val"])
            self.assertTrue(
                {
                    "post_cleaning_index",
                    "type",
                    "is_synth",
                    "target_name",
                    "y_true_raw",
                    "y_pred_raw",
                    "abs_error_raw",
                    "squared_error_raw",
                }.issubset(sidecar_df.columns)
            )
            self.assertIn("raw_real", set(metrics_long["value_space"]))
            self.assertTrue({"train", "val", "test"}.issubset(set(metrics_long["split"])))
            raw_df = metrics_long[metrics_long["value_space"] == "raw_real"]
            self.assertTrue({"overall", "macro", "per_class"}.issubset(set(raw_df["metric_scope"])))
            self.assertTrue({"r2", "mse", "rmse", "rrmse", "mape"}.issubset(set(raw_df["metric_name"])))
            self.assertFalse((run_dir / "unit_xgb_run_v1_run_manifest.json").exists())
            self.assertFalse((run_dir / "unit_xgb_run_v1_metrics_long.csv").exists())
            self.assertFalse((run_dir / "unit_xgb_run_v1_results.yaml").exists())
            self.assertFalse((run_dir / "unit_xgb_run_v1.yaml").exists())


if __name__ == "__main__":
    unittest.main()
