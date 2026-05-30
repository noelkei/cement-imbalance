from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation.artifacts import (
    build_artifact_index_payload,
    build_prediction_sidecar_df,
    load_f7_artifact_persistence_contract,
    resolve_model_artifact_policy,
    resolve_f7_run_filename_policy,
    uses_f7_stable_single_copy_policy,
    write_prediction_sidecar,
)
from evaluation.results import build_run_context, save_canonical_run_artifacts


class TestF7ArtifactPersistenceContract(unittest.TestCase):
    def test_contract_loads_required_fields(self) -> None:
        contract = load_f7_artifact_persistence_contract()
        self.assertEqual(contract["artifact_policy_id"], "f7_artifact_persistence_contract_v1")
        self.assertIn("must_persist_per_run", contract)
        self.assertIn("prediction_sidecar_policy", contract)
        self.assertIn("filename_policy", contract)
        self.assertEqual(resolve_model_artifact_policy("mlp"), "shortlist_or_finalist_only")
        self.assertEqual(resolve_model_artifact_policy("xgboost"), "persist_every_run")
        self.assertEqual(resolve_f7_run_filename_policy(), "versioned_aliases")
        self.assertTrue(
            uses_f7_stable_single_copy_policy(
                {"artifact_policy_id": "f7_artifact_persistence_contract_v1"}
            )
        )

    def test_prediction_sidecar_builder_schema_and_values(self) -> None:
        df = build_prediction_sidecar_df(
            split_payloads={
                "val": {
                    "meta": pd.DataFrame(
                        [
                            {"post_cleaning_index": 10, "type": 1, "is_synth": 0},
                            {"post_cleaning_index": 11, "type": 2},
                        ]
                    ),
                    "y_true_raw": [[10.0], [12.0]],
                    "y_pred_raw": [[10.5], [11.0]],
                    "y_true_native": [[0.1], [0.2]],
                    "y_pred_native": [[0.15], [0.18]],
                }
            },
            target_names=["init"],
        )
        self.assertEqual(
            list(df.columns),
            [
                "split",
                "post_cleaning_index",
                "type",
                "is_synth",
                "target_name",
                "y_true_raw",
                "y_pred_raw",
                "abs_error_raw",
                "squared_error_raw",
                "y_true_native",
                "y_pred_native",
            ],
        )
        self.assertEqual(df["split"].unique().tolist(), ["val"])
        self.assertEqual(df["post_cleaning_index"].tolist(), [10, 11])
        self.assertEqual(df["is_synth"].tolist(), [0, 0])
        self.assertAlmostEqual(float(df.iloc[0]["abs_error_raw"]), 0.5)
        self.assertAlmostEqual(float(df.iloc[1]["squared_error_raw"]), 1.0)

    def test_write_prediction_sidecar_writes_gzip_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = write_prediction_sidecar(
                out_dir=tmpdir,
                split_payloads={
                    "val": {
                        "meta": pd.DataFrame([{"post_cleaning_index": 1, "type": 0, "is_synth": 0}]),
                        "y_true_raw": [[1.0]],
                        "y_pred_raw": [[1.1]],
                    }
                },
                target_names=["init"],
            )
            self.assertIsNotNone(out_path)
            sidecar_path = Path(out_path)
            self.assertTrue(sidecar_path.exists())
            self.assertEqual(sidecar_path.name, "predictions_eval_raw.csv.gz")
            loaded = pd.read_csv(sidecar_path, compression="gzip")
            self.assertEqual(len(loaded), 1)

    def test_artifact_index_payload_tracks_expected_flags(self) -> None:
        payload = build_artifact_index_payload(
            model_family="xgboost",
            results_path="/tmp/results.yaml",
            run_manifest_path="/tmp/run_manifest.json",
            metrics_long_path="/tmp/metrics_long.csv",
            config_snapshot_path="/tmp/config.yaml",
            prediction_sidecar_path="/tmp/predictions_eval_raw.csv.gz",
            model_artifact_path="/tmp/model.json",
            extra_artifact_paths={"feature_names_json": "/tmp/features.json"},
        )
        self.assertEqual(payload["artifact_policy_id"], "f7_artifact_persistence_contract_v1")
        self.assertEqual(payload["artifact_tier"], "per_run")
        self.assertEqual(payload["model_artifact_policy"], "persist_every_run")
        self.assertTrue(payload["artifact_availability"]["results_yaml"])
        self.assertTrue(payload["artifact_availability"]["predictions_eval_raw_csv_gz"])
        self.assertTrue(payload["artifact_availability"]["model_artifact"])
        self.assertEqual(payload["interpretability_policy_status"], "implemented_per_run_family_specific")
        self.assertTrue(payload["interpretability_required_now"])

    def test_artifact_index_payload_accepts_interpretability_override(self) -> None:
        payload = build_artifact_index_payload(
            model_family="mlp",
            results_path="/tmp/results.yaml",
            run_manifest_path="/tmp/run_manifest.json",
            metrics_long_path="/tmp/metrics_long.csv",
            config_snapshot_path="/tmp/config.yaml",
            prediction_sidecar_path="/tmp/predictions_eval_raw.csv.gz",
            interpretability_artifact_paths={
                "interpretability_summary_json": "/tmp/interpretability_summary.json",
                "input_feature_influence_global_csv": "/tmp/input_feature_influence_global.csv",
                "input_feature_influence_per_class_csv": "/tmp/input_feature_influence_per_class.csv",
                "feature_influence_global_csv": "/tmp/feature_influence_global.csv",
            },
            interpretability_status_override={
                "interpretability_policy_status": "implemented_mlp_block_10b_v1",
                "interpretability_required_now": True,
                "family_specific_implementation_pending": False,
            },
        )
        self.assertEqual(payload["interpretability_policy_status"], "implemented_mlp_block_10b_v1")
        self.assertTrue(payload["interpretability_required_now"])
        self.assertFalse(payload["interpretability_status"]["family_specific_implementation_pending"])
        self.assertTrue(payload["artifact_availability"]["interpretability_summary_json"])
        self.assertTrue(payload["artifact_availability"]["input_feature_influence_global_csv"])
        self.assertTrue(payload["artifact_availability"]["input_feature_influence_per_class_csv"])
        self.assertTrue(payload["artifact_availability"]["feature_influence_global_csv"])

    def test_save_canonical_run_artifacts_supports_stable_single_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context = build_run_context(
                model_family="mlp",
                run_id="f7_test_run_v1",
                seed=1234,
                config={"training": {"save_results": True}},
                contract_id="f7_test_contract",
                comparison_group_id="f7_group",
                dataset_level_axes={"x_transform": "standard"},
                run_level_axes={"synthetic_policy": "none"},
                raw_metric_contract_id="f7_raw_metric_contract_v1",
                raw_metric_contract_validation={"status": "ok"},
                run_mode="selection_run",
                campaign_valid=True,
            )
            results = {
                "train": {"overall": {"rrmse": 0.1, "rmse": 0.2, "mse": 0.04, "r2": 0.9, "mape": 0.11, "n": 10}},
                "val": {"overall": {"rrmse": 0.2, "rmse": 0.3, "mse": 0.09, "r2": 0.8, "mape": 0.22, "n": 5}},
                "test": {"overall": {"rrmse": 0.3, "rmse": 0.4, "mse": 0.16, "r2": 0.7, "mape": 0.33, "n": 5}},
                "raw_real": {
                    "train": {"overall": {"rrmse": 0.1, "rmse": 0.2, "mse": 0.04, "r2": 0.9, "mape": 0.11, "n": 10}},
                    "val": {"overall": {"rrmse": 0.2, "rmse": 0.3, "mse": 0.09, "r2": 0.8, "mape": 0.22, "n": 5}},
                    "test": {"overall": {"rrmse": 0.3, "rmse": 0.4, "mse": 0.16, "r2": 0.7, "mape": 0.33, "n": 5}},
                },
            }
            manifest_path, metrics_path = save_canonical_run_artifacts(
                results=results,
                context=context,
                out_dir=tmpdir,
                stem="f7_test_run_v1",
                filename_policy="stable_single_copy",
            )
            self.assertEqual(manifest_path.name, "run_manifest.json")
            self.assertEqual(metrics_path.name, "metrics_long.csv")
            self.assertTrue(manifest_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertFalse((Path(tmpdir) / "f7_test_run_v1_run_manifest.json").exists())
            self.assertFalse((Path(tmpdir) / "f7_test_run_v1_metrics_long.csv").exists())


if __name__ == "__main__":
    unittest.main()
