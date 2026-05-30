from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from evaluation.meta_context import get_f7_analysis_contract_bundle_with_paths
from evaluation.f7_campaign_state import (
    EXECUTION_STATUS_COMPLETED,
    EXECUTION_STATUS_PENDING,
    VALIDITY_STATUS_VALID_F7,
    acquire_trial_lock,
    append_attempt_event,
    build_campaign_paths,
    build_initial_trial_state,
    derive_lineage_trial_group_id,
    initialize_campaign_manifest,
    initialize_trial_state_files,
    refresh_campaign_reporting,
    release_trial_lock,
)


class TestF7CampaignState(unittest.TestCase):
    def setUp(self) -> None:
        self.campaign_id = "f7_campaign_state_test_v1"
        self.paths = build_campaign_paths(self.campaign_id)
        self.registry_backup = None
        if self.paths.registry_path.exists():
            self.registry_backup = self.paths.registry_path.read_text(encoding="utf-8")

    def tearDown(self) -> None:
        if self.paths.root.exists():
            shutil.rmtree(self.paths.root)
        if self.registry_backup is None:
            if self.paths.registry_path.exists():
                self.paths.registry_path.unlink()
        else:
            self.paths.registry_path.parent.mkdir(parents=True, exist_ok=True)
            self.paths.registry_path.write_text(self.registry_backup, encoding="utf-8")

    def test_state_refresh_and_lock_cycle(self) -> None:
        contract_bundle = get_f7_analysis_contract_bundle_with_paths()
        trial_row = {
            "trial_id": "trial__state_test__mlp",
            "campaign_id": self.campaign_id,
            "campaign_spec_id": "f7_campaign_state_test_spec_v1",
            "campaign_kind": "primary",
            "extension_type": None,
            "campaign_lineage_id": "f7_campaign_state_test_lineage_v1",
            "root_campaign_id": self.campaign_id,
            "parent_campaign_id": None,
            "pooling_group_id": "f7_campaign_state_test_lineage_v1",
            "is_primary_analysis_campaign": "true",
            "eligible_for_pooled_seed_analysis": "true",
            "model_family": "mlp",
            "dataset_candidate_id": "mlp__dummy",
            "run_spec_id": "runspec__mlp__dummy",
            "comparison_group_id": "cmp__mlp__dummy",
            "seed_set_id": "f7_seed_panel_v1",
            "seed_panel_path": "config/f7_seed_panel_v1.yaml",
            "seed": "1234",
            "replication_index": "1",
            "expected_seed_count": "30",
            "dataset_manifest_path": "data/sets/official/init_temporal_processed_v1/scaled/df_scaled_xflowpre_candidate_1_yminmax/meta/manifest.json",
            "base_config_id": "f7_mlp_base_v1",
            "objective_metric_id": "raw_real.macro.rrmse",
            "run_mode": "holdout_run",
            "allow_test_holdout": "true",
            "test_enabled": "true",
            "x_transform": "flowpre_candidate_1",
            "y_transform": "minmax",
            "synthetic_policy": "none",
            "run_policy": "plain__no_cycling__overall_rmse__allow_synth_true",
            "flowpre_usage": "true",
            "flowgen_usage": "false",
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
            "factor_parser_version": contract_bundle["factor_parser_version"],
            "metric_grammar_version": contract_bundle["metric_grammar_version"],
            "lineage_aggregate_build_version": contract_bundle["lineage_aggregate_build_version"],
            "panel_build_version": contract_bundle["panel_build_version"],
            "panel_build_timestamp": "2026-05-21T10:00:00+00:00",
            "target_name": contract_bundle["target_name"],
            "target_space": contract_bundle["target_space"],
            "target_unit_public": contract_bundle["target_unit_public"],
            "native_interpretability_layer": "perturbation",
            "bridge_interpretability_layer": "perturbation",
        }
        initialize_trial_state_files(self.paths, [trial_row])
        initialize_campaign_manifest(
            paths=self.paths,
            spec={
                "campaign_id": self.campaign_id,
                "campaign_spec_id": "f7_campaign_state_test_spec_v1",
                "campaign_kind": "primary",
                "extension_type": None,
                "campaign_lineage_id": "f7_campaign_state_test_lineage_v1",
                "root_campaign_id": self.campaign_id,
                "parent_campaign_id": None,
                "seed_set_id": "f7_seed_panel_v1",
                "seed_panel_path": "config/f7_seed_panel_v1.yaml",
                "pooling_group_id": "f7_campaign_state_test_lineage_v1",
                "is_primary_analysis_campaign": True,
                "eligible_for_pooled_seed_analysis": True,
                "campaign_scope": "state_test",
            },
            frozen_inputs={"campaign_spec_path": "dummy.yaml"},
            expansion_manifest={"counts": {"trials": {"total": 1}}},
        )

        acquired, _ = acquire_trial_lock(self.paths, trial_row["trial_id"], "runner-test")
        self.assertTrue(acquired)
        release_trial_lock(self.paths, trial_row["trial_id"])

        state = build_initial_trial_state(trial_row)
        state["execution_status"] = EXECUTION_STATUS_COMPLETED
        state["validity_status"] = VALIDITY_STATUS_VALID_F7
        state["campaign_valid_f7"] = True
        state["run_manifest_path"] = "outputs/models/mlp/campaigns/f7_campaign_state_test_v1/run_manifest.json"
        state["results_path"] = "outputs/models/mlp/campaigns/f7_campaign_state_test_v1/results.yaml"
        state["metrics_long_path"] = "outputs/models/mlp/campaigns/f7_campaign_state_test_v1/metrics_long.csv"
        state["prediction_sidecar_path"] = "outputs/models/mlp/campaigns/f7_campaign_state_test_v1/predictions_eval_raw.csv.gz"
        state["interpretability_summary_path"] = "outputs/models/mlp/campaigns/f7_campaign_state_test_v1/interpretability_summary.json"
        state["raw_metric_contract_id"] = "f7_raw_metric_contract_v1"
        state["raw_metric_contract_validation_status"] = "ok"
        state["raw_real_available"] = True
        state["requires_raw_inversion"] = False
        state["raw_inversion_status"] = "not_required"
        state["value_space_default"] = "raw_real"
        state["class_ontology_contract_id"] = contract_bundle["class_ontology_contract_id"]
        state["target_contract_id"] = contract_bundle["target_contract_id"]
        state["metric_grammar_contract_id"] = contract_bundle["metric_grammar_contract_id"]
        state["metric_availability_contract_id"] = contract_bundle["metric_availability_contract_id"]
        state["metric_aggregation_contract_id"] = contract_bundle["metric_aggregation_contract_id"]
        state["evaluation_population_contract_id"] = contract_bundle["evaluation_population_contract_id"]
        state["prediction_row_join_contract_id"] = contract_bundle["prediction_row_join_contract_id"]
        state["feature_schema_contract_id"] = contract_bundle["feature_schema_contract_id"]
        state["factor_parser_contract_id"] = contract_bundle["factor_parser_contract_id"]
        state["target_name"] = contract_bundle["target_name"]
        state["target_space"] = contract_bundle["target_space"]
        state["split_class_support_source_path"] = state["metrics_long_path"]
        state["analysis_ready_comparable"] = True
        state["warning_count_total"] = 2
        state["warning_count_silenced_known_noise"] = 1
        state["warning_count_surfaced"] = 1
        state["warning_policy_counts"] = {"silenced_known_noise": 1, "surfaced": 1}
        state["warning_signature_counts"] = {"FutureWarning::torch.nn.utils.weight_norm::x": 1}
        state["training_runtime_s"] = 1.0
        state["interpretability_runtime_s"] = 0.5
        state["total_runtime_s"] = 1.5
        state_path = self.paths.trial_state_dir / f"{trial_row['trial_id']}.json"
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

        append_attempt_event(
            self.paths,
            {"trial_id": trial_row["trial_id"], "event_type": "test_event"},
        )
        states, summary, manifest = refresh_campaign_reporting(self.paths)
        self.assertEqual(len(states), 1)
        self.assertEqual(summary["trial_count_total"], 1)
        self.assertEqual(summary["counts_by_status"][EXECUTION_STATUS_COMPLETED], 1)
        self.assertTrue(summary["lineage_pool_ready"])
        self.assertEqual(summary["analysis_ready_comparable_count"], 1)
        self.assertEqual(summary["raw_metric_contract_validation_status_counts"], {"ok": 1})
        self.assertEqual(summary["warning_policy_counts"], {"silenced_known_noise": 1, "surfaced": 1})
        self.assertEqual(
            summary["contract_coverage"]["target_contract_id_counts"][contract_bundle["target_contract_id"]],
            1,
        )
        self.assertEqual(manifest["campaign_status"], "closed_success")
        self.assertTrue(self.paths.ledger_path.exists())
        self.assertTrue(self.paths.summary_path.exists())
        self.assertEqual(
            state["lineage_trial_group_id"],
            derive_lineage_trial_group_id(
                campaign_lineage_id="f7_campaign_state_test_lineage_v1",
                model_family="mlp",
                dataset_candidate_id="mlp__dummy",
                run_spec_id="runspec__mlp__dummy",
            ),
        )


if __name__ == "__main__":
    unittest.main()
