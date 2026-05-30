from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.f7_campaign_spec import DEFAULT_SPEC_PATH, load_f7_campaign_spec, materialize_f7_campaign_spec
from evaluation.meta_context import derive_run_spec_id


class TestF7CampaignSpec(unittest.TestCase):
    def test_load_f7_campaign_spec(self) -> None:
        spec = load_f7_campaign_spec(DEFAULT_SPEC_PATH)
        self.assertEqual(spec["campaign_id"], "f7_campaign_v1")
        self.assertEqual(spec["campaign_spec_id"], "f7_campaign_spec_v1")
        self.assertTrue(spec["allow_test_holdout"])
        self.assertEqual(spec["expected_counts"]["trials"]["total"], 17400)

    def test_materialize_f7_campaign_spec_outputs_expected_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = materialize_f7_campaign_spec(output_root=tmpdir)
            self.assertEqual(len(bundle.dataset_candidates), 100)
            self.assertEqual(len(bundle.run_specs), 7)
            self.assertEqual(len(bundle.trials), 17400)
            self.assertTrue(bundle.output_paths["dataset_candidate_inventory_path"].exists())
            self.assertTrue(bundle.output_paths["run_spec_inventory_path"].exists())
            self.assertTrue(bundle.output_paths["trial_inventory_path"].exists())
            self.assertTrue(bundle.output_paths["expansion_manifest_path"].exists())

            by_family = {}
            for row in bundle.trials:
                by_family[row["model_family"]] = by_family.get(row["model_family"], 0) + 1
            self.assertEqual(by_family["mlp"], 17280)
            self.assertEqual(by_family["xgboost"], 120)

            trial_ids = {row["trial_id"] for row in bundle.trials}
            self.assertEqual(len(trial_ids), len(bundle.trials))

            run_specs_by_family = {}
            for row in bundle.run_specs:
                run_specs_by_family.setdefault(row["model_family"], []).append(row["run_spec_id"])
            self.assertEqual(len(run_specs_by_family["mlp"]), 6)
            self.assertEqual(run_specs_by_family["xgboost"], ["runspec__xgb__f7_xgb_base_v1"])
            self.assertTrue(bundle.output_paths["expected_replication_manifest_path"].exists())

    def test_materialized_trial_inventory_has_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = materialize_f7_campaign_spec(output_root=tmpdir)
            trial_path = bundle.output_paths["trial_inventory_path"]
            with trial_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                first = next(reader)
            for key in (
                "campaign_spec_id",
                "dataset_candidate_id",
                "run_spec_id",
                "trial_id",
                "comparison_group_id",
                "lineage_trial_group_id",
                "seed_set_id",
                "replication_index",
                "expected_seed_count",
                "x_transform",
                "y_transform",
                "synthetic_policy",
                "run_policy",
                "flowpre_usage",
                "flowgen_usage",
                "panel_build_timestamp",
                "target_contract_id",
                "metric_grammar_contract_id",
                "native_interpretability_layer",
                "bridge_interpretability_layer",
            ):
                self.assertIn(key, first)

    def test_materialize_outputs_expected_replication_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = materialize_f7_campaign_spec(output_root=tmpdir)
            payload = json.loads(
                bundle.output_paths["expected_replication_manifest_path"].read_text(encoding="utf-8")
            )
            self.assertEqual(payload["campaign_id"], bundle.spec["campaign_id"])
            self.assertEqual(payload["expected_seed_count"], 30)
            self.assertEqual(
                bundle.expansion_manifest["analysis_contracts"]["panel_build_timestamp"],
                bundle.trials[0]["panel_build_timestamp"],
            )
            self.assertEqual(
                payload["expected_structural_group_count"],
                len(
                    {
                        (
                            row["model_family"],
                            row["dataset_candidate_id"],
                            row["run_spec_id"],
                        )
                        for row in bundle.trials
                    }
                ),
            )

    def test_materialize_f7_campaign_pilot_v1_outputs_expected_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = materialize_f7_campaign_spec(
                spec_path=Path("config") / "f7_campaign_pilot_v1.yaml",
                output_root=tmpdir,
            )
            self.assertEqual(bundle.spec["campaign_id"], "f7_campaign_pilot_v1")
            self.assertEqual(bundle.spec["seed_set_id"], "f7_seed_panel_pilot_v1")
            self.assertEqual(len(bundle.dataset_candidates), 100)
            self.assertEqual(len(bundle.run_specs), 7)
            self.assertEqual(len(bundle.trials), 580)
            by_family = {}
            for row in bundle.trials:
                by_family[row["model_family"]] = by_family.get(row["model_family"], 0) + 1
            self.assertEqual(by_family["mlp"], 576)
            self.assertEqual(by_family["xgboost"], 4)

    def test_materialize_block13_validation_primary_outputs_expected_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = materialize_f7_campaign_spec(
                spec_path=Path("config") / "f7_campaign_block13_validation_primary_v1.yaml",
                output_root=tmpdir,
            )
            self.assertEqual(bundle.spec["campaign_id"], "f7_campaign_block13_validation_primary_v1")
            self.assertEqual(len(bundle.dataset_candidates), 12)
            self.assertEqual(len(bundle.run_specs), 7)
            self.assertEqual(len(bundle.trials), 104)
            by_family = {}
            for row in bundle.trials:
                by_family[row["model_family"]] = by_family.get(row["model_family"], 0) + 1
            self.assertEqual(by_family["mlp"], 96)
            self.assertEqual(by_family["xgboost"], 8)

    def test_materialize_block13_validation_extension_outputs_expected_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = materialize_f7_campaign_spec(
                spec_path=Path("config") / "f7_campaign_block13_validation_extension_v1.yaml",
                output_root=tmpdir,
            )
            self.assertEqual(bundle.spec["campaign_id"], "f7_campaign_block13_validation_extension_v1")
            self.assertEqual(bundle.spec["campaign_kind"], "extension")
            self.assertEqual(len(bundle.dataset_candidates), 12)
            self.assertEqual(len(bundle.run_specs), 7)
            self.assertEqual(len(bundle.trials), 52)
            by_family = {}
            for row in bundle.trials:
                by_family[row["model_family"]] = by_family.get(row["model_family"], 0) + 1
            self.assertEqual(by_family["mlp"], 48)
            self.assertEqual(by_family["xgboost"], 4)

    def test_materialize_block13_validation_extension2_outputs_expected_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = materialize_f7_campaign_spec(
                spec_path=Path("config") / "f7_campaign_block13_validation_extension2_v1.yaml",
                output_root=tmpdir,
            )
            self.assertEqual(bundle.spec["campaign_id"], "f7_campaign_block13_validation_extension2_v1")
            self.assertEqual(bundle.spec["campaign_kind"], "extension")
            self.assertEqual(bundle.spec["parent_campaign_id"], "f7_campaign_block13_validation_extension_v1")
            self.assertEqual(len(bundle.dataset_candidates), 12)
            self.assertEqual(len(bundle.run_specs), 7)
            self.assertEqual(len(bundle.trials), 52)
            by_family = {}
            for row in bundle.trials:
                by_family[row["model_family"]] = by_family.get(row["model_family"], 0) + 1
            self.assertEqual(by_family["mlp"], 48)
            self.assertEqual(by_family["xgboost"], 4)

    def test_derive_run_spec_id_maps_runtime_axes_to_campaign_tokens(self) -> None:
        plain = derive_run_spec_id(
            model_family="mlp",
            base_config_id="f7_mlp_base_v1",
            run_level_axes={
                "batch_policy": {"id": "baseline"},
                "cycling_policy": {"cycle_reals": False},
                "allow_synth": {"enabled": True},
                "loss_policy": {
                    "loss_reduction": "overall",
                    "regression_group_metric": "rmse",
                },
            },
        )
        imbalance = derive_run_spec_id(
            model_family="mlp",
            base_config_id="f7_mlp_base_v1",
            run_level_axes={
                "batch_policy": {"id": "balanced"},
                "cycling_policy": {"cycle_reals": True},
                "allow_synth": {"enabled": True},
                "loss_policy": {
                    "loss_reduction": "per_class_equal",
                    "regression_group_metric": "rrmse",
                },
            },
        )
        self.assertEqual(
            plain,
            "runspec__mlp__f7_mlp_base_v1__plain__no_cycling__overall_rmse__allow_synth-true",
        )
        self.assertEqual(
            imbalance,
            "runspec__mlp__f7_mlp_base_v1__imbalance_aware__cycling__per_class_equal_rrmse__allow_synth-true",
        )


if __name__ == "__main__":
    unittest.main()
