from __future__ import annotations

import csv
import json
import shutil
import unittest
from pathlib import Path

from evaluation.meta_context import get_f7_analysis_contract_bundle_with_paths
from evaluation.f7_campaign_lineage import (
    LINEAGE_SURFACE_MLP_FLOWPRE_NATIVE_LATENT,
    LINEAGE_SURFACE_SEMANTIC_BRIDGE,
    LINEAGE_SURFACE_XGB_NATIVE_SHAP,
    validate_lineage_pool_readiness,
    write_lineage_aggregate,
)
from evaluation.f7_campaign_state import build_campaign_paths


class TestF7CampaignLineage(unittest.TestCase):
    def setUp(self) -> None:
        self.campaign_ids = [
            "f7_campaign_lineage_primary_test_v1",
            "f7_campaign_lineage_extension_test_v1",
            "f7_campaign_lineage_scope_mismatch_v1",
        ]

    def tearDown(self) -> None:
        for campaign_id in self.campaign_ids:
            paths = build_campaign_paths(campaign_id)
            if paths.root.exists():
                shutil.rmtree(paths.root)
            for family in ("mlp", "xgboost"):
                campaign_root = Path("outputs/models") / family / "campaigns" / campaign_id
                if campaign_root.exists():
                    shutil.rmtree(campaign_root)
        aggregate_root = build_campaign_paths("f7_campaign_lineage_primary_test_v1").root / "lineage_aggregate"
        if aggregate_root.exists():
            shutil.rmtree(aggregate_root)

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_metrics_long(self, path: Path, *, seed: int, trial_id: str, campaign_id: str, model_family: str, dataset_candidate_id: str, run_spec_id: str, comparison_group_id: str) -> None:
        rows: list[dict[str, object]] = []
        for split, base in (("train", 0.05), ("val", 0.10), ("test", 0.12)):
            for metric_name, metric_value in {
                "mse": base,
                "rmse": base + 0.01,
                "rrmse": base + 0.02,
                "r2": 1.0 - base,
                "mape": base + 0.03,
            }.items():
                base_row = {
                    "run_id": f"{trial_id}__attempt-0001",
                    "variant_fingerprint": f"vf::{trial_id}",
                    "model_family": model_family,
                    "contract_id": "f7_contract_v1",
                    "comparison_group_id": comparison_group_id,
                    "campaign_id": campaign_id,
                    "dataset_candidate_id": dataset_candidate_id,
                    "run_spec_id": run_spec_id,
                    "trial_id": trial_id,
                    "replication_index": 1,
                    "seed_panel_version": 1,
                    "seed_set_id": "f7_seed_panel_test_v1",
                    "seed_panel_path": "config/f7_seed_panel_test_v1.yaml",
                    "base_config_id": "base",
                    "objective_metric_id": "raw_real.macro.rrmse",
                    "upstream_variant_fingerprint": f"{campaign_id}::{trial_id}",
                    "dataset_name": "dataset",
                    "dataset_manifest_path": "data/sets/test/manifest.json",
                    "dataset_level_axes": "{}",
                    "run_level_axes": "{}",
                    "split_id": "init_temporal_processed_v1",
                    "split_manifest_path": "data/splits/official/init_temporal_processed_v1/manifest.json",
                    "seed": seed,
                    "config_path": "outputs/config.yaml",
                    "config_sha256": "sha256",
                    "test_enabled": True,
                    "raw_metric_contract_id": "f7_raw_metric_contract_v1",
                    "run_mode": "holdout_run",
                    "campaign_valid": True,
                    "split": split,
                    "split_role": "holdout",
                    "metric_group": "predictive",
                    "metric_name": metric_name,
                    "component": "",
                    "target_name": "init",
                    "value_space": "raw_real",
                }
                rows.extend(
                    [
                        {
                            **base_row,
                            "metric_scope": "macro",
                            "class_id": "",
                            "metric_value": metric_value,
                            "n_obs": 10,
                        },
                        {
                            **base_row,
                            "metric_scope": "overall",
                            "class_id": "",
                            "metric_value": metric_value + 0.001,
                            "n_obs": 10,
                        },
                        {
                            **base_row,
                            "metric_scope": "overall_quantile",
                            "class_id": "",
                            "metric_value": metric_value + 0.002,
                            "n_obs": 10,
                        },
                        {
                            **base_row,
                            "metric_scope": "worst_class",
                            "class_id": "",
                            "metric_value": metric_value + 0.003,
                            "n_obs": 10,
                        },
                    ]
                )
                for class_id in ("0", "1", "2"):
                    rows.append(
                        {
                            **base_row,
                            "metric_scope": "per_class",
                            "class_id": class_id,
                            "metric_value": metric_value + (0.01 * int(class_id)),
                            "n_obs": 10,
                        }
                    )
                    rows.append(
                        {
                            **base_row,
                            "metric_scope": "per_class_quantile",
                            "class_id": class_id,
                            "metric_value": metric_value + 0.002 + (0.01 * int(class_id)),
                            "n_obs": 10,
                        }
                    )
        self._write_csv(path, rows)

    def _write_interpretability_bundle(self, run_dir: Path, *, xgb: bool, flowpre: bool) -> dict[str, str]:
        outputs: dict[str, str] = {}
        if xgb:
            global_rows = [
                {
                    "split": "val",
                    "feature_name": "f1",
                    "mean_abs_delta_pred_raw": 0.3,
                    "rank_abs": 1,
                    "feature_space_kind": "xgb_model_input_feature",
                    "projection_status": "direct_model_input",
                },
                {
                    "split": "test",
                    "feature_name": "f1",
                    "mean_abs_delta_pred_raw": 0.2,
                    "rank_abs": 1,
                    "feature_space_kind": "xgb_model_input_feature",
                    "projection_status": "direct_model_input",
                },
            ]
            per_class_rows = [
                {
                    "split": "val",
                    "type": 0,
                    "feature_name": "f1",
                    "mean_abs_delta_pred_raw": 0.3,
                    "rank_abs": 1,
                    "feature_space_kind": "xgb_model_input_feature",
                    "projection_status": "direct_model_input",
                }
            ]
            shap_rows = [
                {
                    "split": "val",
                    "feature_name": "f1",
                    "mean_abs_shap": 0.4,
                    "rank_abs": 1,
                    "feature_space_kind": "xgb_model_input_feature",
                    "projection_status": "direct_model_input",
                }
            ]
            self._write_csv(run_dir / "xgb_perturbation_feature_influence_global.csv", global_rows)
            self._write_csv(run_dir / "xgb_perturbation_feature_influence_per_class.csv", per_class_rows)
            self._write_csv(run_dir / "xgb_shap_feature_influence_global.csv", shap_rows)
            self._write_csv(run_dir / "xgb_shap_feature_influence_per_class.csv", per_class_rows)
            outputs.update(
                {
                    "xgb_perturbation_feature_influence_global_path": str((run_dir / "xgb_perturbation_feature_influence_global.csv").resolve()),
                    "xgb_perturbation_feature_influence_per_class_path": str((run_dir / "xgb_perturbation_feature_influence_per_class.csv").resolve()),
                    "xgb_shap_feature_influence_global_path": str((run_dir / "xgb_shap_feature_influence_global.csv").resolve()),
                    "xgb_shap_feature_influence_per_class_path": str((run_dir / "xgb_shap_feature_influence_per_class.csv").resolve()),
                    "interpretability_summary_path": str((run_dir / "xgb_interpretability_summary.json").resolve()),
                }
            )
            (run_dir / "xgb_interpretability_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            return outputs

        semantic_rows = [
            {
                "split": "val",
                "feature_name": "f1",
                "mean_abs_delta_pred_raw": 0.5,
                "rank_abs": 1,
                "feature_space_kind": "semantic_final_surface",
                "projection_status": "projected_ilr_groups",
            }
        ]
        per_class_rows = [
            {
                "split": "val",
                "type": 0,
                "feature_name": "f1",
                "mean_abs_delta_pred_raw": 0.5,
                "rank_abs": 1,
                "feature_space_kind": "semantic_final_surface",
                "projection_status": "projected_ilr_groups",
            }
        ]
        self._write_csv(run_dir / "feature_influence_global.csv", semantic_rows)
        self._write_csv(run_dir / "feature_influence_per_class.csv", per_class_rows)
        outputs.update(
            {
                "feature_influence_global_path": str((run_dir / "feature_influence_global.csv").resolve()),
                "feature_influence_per_class_path": str((run_dir / "feature_influence_per_class.csv").resolve()),
                "interpretability_summary_path": str((run_dir / "interpretability_summary.json").resolve()),
            }
        )
        if flowpre:
            input_rows = [
                {
                    "split": "val",
                    "feature_name": "f1",
                    "mean_abs_delta_pred_raw": 0.4,
                    "rank_abs": 1,
                    "feature_space_kind": "model_input_space",
                    "projection_status": "projected_from_flowpre_cache",
                }
            ]
            latent_rows = [
                {
                    "split": "val",
                    "latent_name": "z_1",
                    "mean_abs_delta_pred_raw": 0.2,
                    "rank_abs": 1,
                }
            ]
            self._write_csv(run_dir / "input_feature_influence_global.csv", input_rows)
            self._write_csv(run_dir / "input_feature_influence_per_class.csv", per_class_rows)
            self._write_csv(run_dir / "latent_feature_influence_global.csv", latent_rows)
            self._write_csv(run_dir / "latent_feature_influence_per_class.csv", [{"split": "val", "type": 0, "latent_name": "z_1", "mean_abs_delta_pred_raw": 0.2, "rank_abs": 1}])
            outputs.update(
                {
                    "input_feature_influence_global_path": str((run_dir / "input_feature_influence_global.csv").resolve()),
                    "input_feature_influence_per_class_path": str((run_dir / "input_feature_influence_per_class.csv").resolve()),
                    "latent_feature_influence_global_path": str((run_dir / "latent_feature_influence_global.csv").resolve()),
                    "latent_feature_influence_per_class_path": str((run_dir / "latent_feature_influence_per_class.csv").resolve()),
                }
            )
        (run_dir / "interpretability_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        return outputs

    def _write_campaign_fixture(self, *, campaign_id: str, root_campaign_id: str, parent_campaign_id: str | None, campaign_scope: str, seed: int) -> None:
        contract_bundle = {
            **get_f7_analysis_contract_bundle_with_paths(),
            "panel_build_timestamp": "2026-05-21T10:00:00+00:00",
        }
        paths = build_campaign_paths(campaign_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        trial_rows: list[dict[str, object]] = []

        entries = [
            ("mlp", "mlp__x-candidate1__y-minmax__syn-none", "runspec__mlp__baseline", "cmp__mlp__baseline", True),
            ("xgboost", "xgb__x-raw-base-v1__y-raw__syn-none", "runspec__xgb__baseline", "cmp__xgb__baseline", False),
        ]
        for family, dataset_candidate_id, run_spec_id, comparison_group_id, flowpre in entries:
            trial_id = f"trial__{campaign_id}__{family}__{dataset_candidate_id}__{run_spec_id}__seed-{seed}"
            run_id = f"{trial_id}__attempt-0001"
            run_dir = Path("outputs/models") / family / "campaigns" / campaign_id / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics_long.csv"
            self._write_metrics_long(
                metrics_path,
                seed=seed,
                trial_id=trial_id,
                campaign_id=campaign_id,
                model_family=family,
                dataset_candidate_id=dataset_candidate_id,
                run_spec_id=run_spec_id,
                comparison_group_id=comparison_group_id,
            )
            results_path = run_dir / "results.yaml"
            results_path.write_text("ok: true\n", encoding="utf-8")
            prediction_path = run_dir / "predictions_eval_raw.csv.gz"
            prediction_path.write_text("placeholder", encoding="utf-8")
            interpretability_paths = self._write_interpretability_bundle(run_dir, xgb=(family == "xgboost"), flowpre=flowpre)
            trial_rows.append(
                {
                    "trial_id": trial_id,
                    "campaign_id": campaign_id,
                    "campaign_lineage_id": f"{root_campaign_id}_lineage",
                    "root_campaign_id": root_campaign_id,
                    "parent_campaign_id": parent_campaign_id,
                    "model_family": family,
                    "dataset_candidate_id": dataset_candidate_id,
                    "run_spec_id": run_spec_id,
                    "comparison_group_id": comparison_group_id,
                    "lineage_trial_group_id": f"lineage_group__{root_campaign_id}_lineage__{family}__{dataset_candidate_id}__{run_spec_id}",
                    "seed": seed,
                    "objective_metric_id": "raw_real.macro.rrmse",
                    "value_space_default": "raw_real",
                    "metrics_long_path": str(metrics_path.resolve()),
                    "run_manifest_path": str((run_dir / "run_manifest.json").resolve()),
                    "results_path": str(results_path.resolve()),
                    "prediction_sidecar_path": str(prediction_path.resolve()),
                    "raw_metric_contract_id": "f7_raw_metric_contract_v1",
                    "raw_metric_contract_validation_status": "ok",
                    "execution_status": "completed",
                    "campaign_valid_f7": True,
                    "analysis_ready_comparable": True,
                    "run_mode": "holdout_run",
                    "allow_test_holdout": True,
                    "test_enabled": True,
                    "variant_fingerprint": f"vf::{trial_id}",
                    "class_ontology_contract_id": contract_bundle["class_ontology_contract_id"],
                    "target_contract_id": contract_bundle["target_contract_id"],
                    "metric_grammar_contract_id": contract_bundle["metric_grammar_contract_id"],
                    "metric_availability_contract_id": contract_bundle["metric_availability_contract_id"],
                    "metric_aggregation_contract_id": contract_bundle["metric_aggregation_contract_id"],
                    "evaluation_population_contract_id": contract_bundle["evaluation_population_contract_id"],
                    "prediction_row_join_contract_id": contract_bundle["prediction_row_join_contract_id"],
                    "prediction_row_join_key_kind": contract_bundle["prediction_row_join_key_kind"],
                    "feature_schema_contract_id": contract_bundle["feature_schema_contract_id"],
                    "factor_parser_contract_id": contract_bundle["factor_parser_contract_id"],
                    "feature_namespace": (
                        "flowpre_projected_semantic_input"
                        if family == "mlp"
                        else "xgb_perturbation"
                    ),
                    "primary_interpretability_surface_id": "semantic_bridge_perturbation",
                    "expected_seed_count": 1,
                    "panel_build_version": contract_bundle["panel_build_version"],
                    "panel_build_timestamp": contract_bundle["panel_build_timestamp"],
                    "lineage_aggregate_build_version": contract_bundle["lineage_aggregate_build_version"],
                    **interpretability_paths,
                }
            )
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "trial_id": trial_id,
                        "run_id": run_id,
                        "analysis_contracts": contract_bundle,
                        "parsed_factor_fields": {
                            "x_transform": "flowpre_candidate_1" if family == "mlp" else "raw_base_v1",
                            "y_transform": "minmax" if family == "mlp" else "raw",
                            "synthetic_policy": "none",
                            "run_policy": "baseline",
                            "flowpre_usage": bool(flowpre),
                            "flowgen_usage": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

        self._write_csv(paths.ledger_path, trial_rows)
        paths.summary_path.write_text(
            json.dumps({"lineage_pool_ready": True, "completed_valid_f7_count": len(trial_rows)}),
            encoding="utf-8",
        )
        paths.campaign_manifest_path.write_text(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "campaign_lineage_id": f"{root_campaign_id}_lineage",
                    "root_campaign_id": root_campaign_id,
                    "parent_campaign_id": parent_campaign_id,
                    "campaign_scope": campaign_scope,
                    "campaign_status": "closed_success",
                    "analysis_contracts": contract_bundle,
                    "expected_replication": {
                        "campaign_id": campaign_id,
                        "campaign_lineage_id": f"{root_campaign_id}_lineage",
                        "root_campaign_id": root_campaign_id,
                        "parent_campaign_id": parent_campaign_id,
                        "expected_seed_values": [seed],
                        "expected_seed_count": 1,
                        "expected_structural_group_ids": [row["lineage_trial_group_id"] for row in trial_rows],
                        "expected_seed_count_by_structural_group": {
                            str(row["lineage_trial_group_id"]): 1 for row in trial_rows
                        },
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def test_write_lineage_aggregate_builds_results_and_interpretability_outputs(self) -> None:
        root_campaign_id = "f7_campaign_lineage_primary_test_v1"
        self._write_campaign_fixture(
            campaign_id=root_campaign_id,
            root_campaign_id=root_campaign_id,
            parent_campaign_id=None,
            campaign_scope="lineage_scope",
            seed=1234,
        )
        self._write_campaign_fixture(
            campaign_id="f7_campaign_lineage_extension_test_v1",
            root_campaign_id=root_campaign_id,
            parent_campaign_id=root_campaign_id,
            campaign_scope="lineage_scope",
            seed=2345,
        )

        summary = write_lineage_aggregate(root_campaign_id)
        self.assertTrue(summary["lineage_pool_ready"])
        self.assertEqual(len(summary["included_campaign_ids"]), 2)
        self.assertEqual(summary["analysis_ready_comparable_count"], 4)
        self.assertEqual(summary["expected_seed_counts_by_group"], summary["observed_seed_counts_by_group"])
        self.assertEqual(summary["lineage_metric_panel_detailed_count"], 600)
        self.assertEqual(summary["lineage_metric_aggregate_detailed_count"], 300)

        output_dir = build_campaign_paths(root_campaign_id).root / "lineage_aggregate"
        with (output_dir / "lineage_metric_aggregate.csv").open("r", encoding="utf-8", newline="") as handle:
            metric_aggregate = list(csv.DictReader(handle))
        self.assertTrue(metric_aggregate)
        self.assertTrue(all(int(float(row["observed_seed_count"])) == 2 for row in metric_aggregate))
        with (output_dir / "lineage_metric_aggregate_detailed.csv").open("r", encoding="utf-8", newline="") as handle:
            metric_aggregate_detailed = list(csv.DictReader(handle))
        self.assertTrue(metric_aggregate_detailed)
        scopes = {row["metric_scope"] for row in metric_aggregate_detailed}
        self.assertIn("per_class", scopes)
        self.assertIn("per_class_quantile", scopes)
        self.assertIn("overall_quantile", scopes)
        self.assertIn("worst_class", scopes)

        with (output_dir / f"lineage_interpretability_aggregate__{LINEAGE_SURFACE_SEMANTIC_BRIDGE}__global.csv").open("r", encoding="utf-8", newline="") as handle:
            semantic_rows = list(csv.DictReader(handle))
        self.assertTrue(semantic_rows)
        with (output_dir / f"lineage_interpretability_aggregate__{LINEAGE_SURFACE_XGB_NATIVE_SHAP}__global.csv").open("r", encoding="utf-8", newline="") as handle:
            shap_rows = list(csv.DictReader(handle))
        self.assertTrue(shap_rows)
        with (output_dir / f"lineage_interpretability_aggregate__{LINEAGE_SURFACE_MLP_FLOWPRE_NATIVE_LATENT}__global.csv").open("r", encoding="utf-8", newline="") as handle:
            latent_rows = list(csv.DictReader(handle))
        self.assertTrue(latent_rows)

    def test_validate_lineage_pool_readiness_detects_contract_mismatch(self) -> None:
        root_campaign_id = "f7_campaign_lineage_primary_test_v1"
        self._write_campaign_fixture(
            campaign_id=root_campaign_id,
            root_campaign_id=root_campaign_id,
            parent_campaign_id=None,
            campaign_scope="lineage_scope",
            seed=1234,
        )
        self._write_campaign_fixture(
            campaign_id="f7_campaign_lineage_extension_test_v1",
            root_campaign_id=root_campaign_id,
            parent_campaign_id=root_campaign_id,
            campaign_scope="lineage_scope",
            seed=2345,
        )
        extension_manifest = build_campaign_paths("f7_campaign_lineage_extension_test_v1").campaign_manifest_path
        payload = json.loads(extension_manifest.read_text(encoding="utf-8"))
        payload["analysis_contracts"]["target_contract_id"] = "f7_target_contract_v999"
        extension_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        summary = validate_lineage_pool_readiness(root_campaign_id)
        self.assertFalse(summary["lineage_pool_ready"])
        self.assertIn("target_contract_id_mismatch", summary["lineage_pool_blockers"])

    def test_validate_lineage_pool_readiness_detects_expected_seed_values_mismatch(self) -> None:
        root_campaign_id = "f7_campaign_lineage_primary_test_v1"
        self._write_campaign_fixture(
            campaign_id=root_campaign_id,
            root_campaign_id=root_campaign_id,
            parent_campaign_id=None,
            campaign_scope="lineage_scope",
            seed=1234,
        )
        manifest_path = build_campaign_paths(root_campaign_id).campaign_manifest_path
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["expected_replication"]["expected_seed_values"] = [9999]
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        summary = validate_lineage_pool_readiness(root_campaign_id)
        self.assertFalse(summary["lineage_pool_ready"])
        self.assertIn(
            f"expected_seed_values_mismatch:{root_campaign_id}",
            summary["lineage_pool_blockers"],
        )

    def test_validate_lineage_pool_readiness_detects_scope_mismatch(self) -> None:
        root_campaign_id = "f7_campaign_lineage_primary_test_v1"
        self._write_campaign_fixture(
            campaign_id=root_campaign_id,
            root_campaign_id=root_campaign_id,
            parent_campaign_id=None,
            campaign_scope="lineage_scope",
            seed=1234,
        )
        self._write_campaign_fixture(
            campaign_id="f7_campaign_lineage_scope_mismatch_v1",
            root_campaign_id=root_campaign_id,
            parent_campaign_id=root_campaign_id,
            campaign_scope="other_scope",
            seed=2345,
        )
        summary = validate_lineage_pool_readiness(root_campaign_id)
        self.assertFalse(summary["lineage_pool_ready"])
        self.assertIn("campaign_scope_mismatch", summary["lineage_pool_blockers"])


if __name__ == "__main__":
    unittest.main()
