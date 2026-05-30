from __future__ import annotations

import unittest

from evaluation.meta_context import (
    get_f7_analysis_contract_bundle_with_paths,
    parse_f7_factor_fields,
    resolve_feature_namespace,
)


class TestF7MetaContext(unittest.TestCase):
    def test_analysis_contract_bundle_has_required_ids_and_paths(self) -> None:
        bundle = get_f7_analysis_contract_bundle_with_paths()
        for key in (
            "class_ontology_contract_id",
            "target_contract_id",
            "artifact_policy_id",
            "metric_grammar_contract_id",
            "metric_availability_contract_id",
            "metric_aggregation_contract_id",
            "evaluation_population_contract_id",
            "prediction_row_join_contract_id",
            "feature_schema_contract_id",
            "factor_parser_contract_id",
            "target_name",
            "target_space",
            "target_unit_public",
            "prediction_row_join_key_kind",
            "panel_build_version",
            "lineage_aggregate_build_version",
            "class_ontology_contract_path",
            "target_contract_path",
            "artifact_policy_contract_path",
            "feature_schema_contract_path",
        ):
            self.assertIn(key, bundle)
            self.assertTrue(bundle[key])

    def test_parse_f7_factor_fields_for_direct_mlp(self) -> None:
        payload = parse_f7_factor_fields(
            model_family="mlp",
            dataset_level_axes={
                "x_transform": "standard",
                "y_transform": "minmax",
                "synthetic_policy": "flowgen_official",
            },
            run_level_axes={
                "batch_policy_id": "plain",
                "cycling_policy_id": "cycling",
                "loss_policy_id": "overall_rmse",
                "allow_synth": True,
            },
            fallback_dataset_candidate_id="mlp__x-standard__y-minmax__syn-flowgen-official",
            fallback_run_spec_id="runspec__mlp__baseline",
        )
        self.assertEqual(payload["x_transform"], "standard")
        self.assertEqual(payload["y_transform"], "minmax")
        self.assertEqual(payload["synthetic_policy"], "flowgen_official")
        self.assertEqual(payload["run_policy"], "plain__cycling__overall_rmse__allow_synth-true")
        self.assertFalse(payload["flowpre_usage"])
        self.assertTrue(payload["flowgen_usage"])

    def test_resolve_feature_namespace_prefers_semantic_bridge_for_flowpre(self) -> None:
        namespace, surface_id = resolve_feature_namespace(
            model_family="mlp",
            has_input_projection=True,
            has_latent_surface=True,
        )
        self.assertEqual(namespace, "flowpre_projected_semantic_input")
        self.assertEqual(surface_id, "semantic_bridge_perturbation")


if __name__ == "__main__":
    unittest.main()
