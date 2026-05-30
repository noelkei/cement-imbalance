from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
import warnings
import gzip
from pathlib import Path

import yaml

from evaluation.meta_context import get_f7_analysis_contract_bundle_with_paths
from evaluation.f7_campaign_runner import (
    _classify_warning,
    _extract_state_fields_from_run_manifest,
    initialize_campaign_from_spec,
    rebuild_campaign_state,
    rerun_failed_campaign,
    resume_campaign,
    run_campaign,
    run_preflight,
    validate_extension_lineage,
)
from evaluation.f7_campaign_spec import DEFAULT_SPEC_PATH, load_f7_campaign_spec, materialize_f7_campaign_spec
from evaluation.f7_campaign_state import (
    build_campaign_paths,
    derive_attempt_id,
    derive_run_id,
    load_campaign_manifest,
    load_trial_state,
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class TestF7CampaignRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.base_spec = load_f7_campaign_spec(DEFAULT_SPEC_PATH)
        self.registry_path = Path("outputs/campaigns/f7_campaign_registry.csv")
        self.registry_backup = self.registry_path.read_text(encoding="utf-8") if self.registry_path.exists() else None
        self.test_campaign_ids: list[str] = []

    def tearDown(self) -> None:
        for campaign_id in self.test_campaign_ids:
            campaign_paths = build_campaign_paths(campaign_id)
            if campaign_paths.root.exists():
                shutil.rmtree(campaign_paths.root)
            mlp_root = Path("outputs/models/mlp") / campaign_id
            xgb_root = Path("outputs/models/xgboost") / campaign_id
            mlp_campaign_root = Path("outputs/models/mlp/campaigns") / campaign_id
            xgb_campaign_root = Path("outputs/models/xgboost/campaigns") / campaign_id
            if mlp_root.exists():
                shutil.rmtree(mlp_root)
            if xgb_root.exists():
                shutil.rmtree(xgb_root)
            if mlp_campaign_root.exists():
                shutil.rmtree(mlp_campaign_root)
            if xgb_campaign_root.exists():
                shutil.rmtree(xgb_campaign_root)
        if self.registry_backup is None:
            if self.registry_path.exists():
                self.registry_path.unlink()
        else:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            self.registry_path.write_text(self.registry_backup, encoding="utf-8")
        self.tmpdir.cleanup()

    def _write_seed_panel(self, panel_name: str, seeds: list[int]) -> Path:
        path = self.tmp_path / f"{panel_name}.yaml"
        payload = {
            "seed_panel": {
                "seed_set_id": panel_name,
                "panel_version": 1,
                "seeds": [{"index": idx + 1, "value": int(seed)} for idx, seed in enumerate(seeds)],
            }
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def _write_small_inventory(self) -> Path:
        source_path = Path(self.base_spec["canonical_sources"]["dataset_inventory_path"])
        rows = _read_csv_rows(source_path)
        mlp_row = next(
            row
            for row in rows
            if row["model_family_scope"] == "mlp"
            and row["dataset_role"] == "derived_modeling_bundle"
            and "df_scaled_xstandard_ystandard" in row["manifest_path"]
        )
        xgb_row = next(
            row
            for row in rows
            if row["model_family_scope"] == "xgboost" and row["dataset_role"] == "derived_modeling_bundle"
        )
        out_path = self.tmp_path / "small_inventory.csv"
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mlp_row.keys()))
            writer.writeheader()
            writer.writerows([mlp_row, xgb_row])
        return out_path

    def _write_spec(
        self,
        *,
        campaign_id: str,
        campaign_spec_id: str,
        seed_panel_path: Path,
        seed_set_id: str,
        inventory_path: Path,
        parent_campaign_id: str | None = None,
        root_campaign_id: str | None = None,
        campaign_kind: str = "primary",
        extension_type: str | None = None,
        is_primary_analysis_campaign: bool = True,
        output_name_prefix: str = "test",
    ) -> Path:
        spec = json.loads(json.dumps(self.base_spec))
        spec["campaign_id"] = campaign_id
        spec["campaign_spec_id"] = campaign_spec_id
        spec["campaign_scope"] = "f7_runner_test_scope"
        spec["campaign_kind"] = campaign_kind
        spec["extension_type"] = extension_type
        spec["campaign_lineage_id"] = f"{campaign_id}_lineage"
        spec["root_campaign_id"] = root_campaign_id or campaign_id
        spec["parent_campaign_id"] = parent_campaign_id
        spec["pooling_group_id"] = root_campaign_id or campaign_id
        spec["eligible_for_pooled_seed_analysis"] = True
        spec["is_primary_analysis_campaign"] = is_primary_analysis_campaign
        spec["seed_set_id"] = seed_set_id
        spec["seed_panel_path"] = str(seed_panel_path)
        spec["canonical_sources"]["dataset_inventory_path"] = str(inventory_path)
        spec["mlp"]["dataset_inventory_source"] = str(inventory_path)
        spec["xgboost"]["dataset_inventory_source"] = str(inventory_path)
        spec["outputs"] = {
            "dataset_candidate_inventory_path": f"outputs/reports/f7_campaign_spec/f7_campaign_dataset_candidates_{output_name_prefix}.csv",
            "run_spec_inventory_path": f"outputs/reports/f7_campaign_spec/f7_campaign_run_specs_{output_name_prefix}.csv",
            "trial_inventory_path": f"outputs/reports/f7_campaign_spec/f7_campaign_trials_{output_name_prefix}.csv",
            "expansion_manifest_path": f"outputs/reports/f7_campaign_spec/f7_campaign_expansion_manifest_{output_name_prefix}.json",
        }
        seed_count = len(yaml.safe_load(seed_panel_path.read_text(encoding="utf-8"))["seed_panel"]["seeds"])
        spec["expected_counts"] = {
            "dataset_candidates": {"mlp": 1, "xgboost": 1, "total": 2},
            "run_specs": {"mlp": 6, "xgboost": 1, "total": 7},
            "trials": {"mlp": 6 * seed_count, "xgboost": 1 * seed_count, "total": 7 * seed_count},
        }
        payload = {"campaign_spec": spec}
        out_path = self.tmp_path / f"{campaign_spec_id}.yaml"
        out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        campaign_paths = build_campaign_paths(campaign_id)
        for path in (
            campaign_paths.root,
            Path("outputs/models/mlp") / campaign_id,
            Path("outputs/models/xgboost") / campaign_id,
            Path("outputs/models/mlp/campaigns") / campaign_id,
            Path("outputs/models/xgboost/campaigns") / campaign_id,
        ):
            if path.exists():
                shutil.rmtree(path)
        self.test_campaign_ids.append(campaign_id)
        return out_path

    def test_preflight_and_mixed_run_resume(self) -> None:
        inventory_path = self._write_small_inventory()
        seed_panel_path = self._write_seed_panel("f7_seed_panel_test_v1", [1234])
        spec_path = self._write_spec(
            campaign_id="f7_campaign_runner_test_v1",
            campaign_spec_id="f7_campaign_runner_test_spec_v1",
            seed_panel_path=seed_panel_path,
            seed_set_id="f7_seed_panel_test_v1",
            inventory_path=inventory_path,
        )

        preflight = run_preflight(spec_path=spec_path)
        self.assertTrue(preflight["ok"])
        self.assertEqual(preflight["checked_trial_count"], 7)

        _, materialized = initialize_campaign_from_spec(spec_path)
        trial_rows = materialized.trials
        mlp_trial_id = next(row["trial_id"] for row in trial_rows if row["model_family"] == "mlp")
        xgb_trial_id = next(row["trial_id"] for row in trial_rows if row["model_family"] == "xgboost")
        trial_file = self.tmp_path / "selected_trials.txt"
        trial_file.write_text(f"{mlp_trial_id}\n{xgb_trial_id}\n", encoding="utf-8")

        result = run_campaign(spec_path=spec_path, device="cpu", trial_id_file=trial_file)
        self.assertEqual(result["campaign_id"], "f7_campaign_runner_test_v1")

        paths = build_campaign_paths("f7_campaign_runner_test_v1")
        mlp_state = load_trial_state(paths, mlp_trial_id)
        xgb_state = load_trial_state(paths, xgb_trial_id)
        self.assertEqual(mlp_state["execution_status"], "completed")
        self.assertTrue(mlp_state["campaign_valid_f7"])
        self.assertEqual(xgb_state["execution_status"], "completed")
        self.assertTrue(xgb_state["campaign_valid_f7"])

        resume_result = resume_campaign(
            campaign_id="f7_campaign_runner_test_v1",
            device="cpu",
            trial_id_file=trial_file,
        )
        self.assertEqual(resume_result["campaign_id"], "f7_campaign_runner_test_v1")
        attempts_lines = paths.attempts_path.read_text(encoding="utf-8").splitlines()
        self.assertTrue(any('"event_type": "skip_completed_valid"' in line for line in attempts_lines))

    def test_seed_extension_lineage_validation(self) -> None:
        inventory_path = self._write_small_inventory()
        base_seed_panel = self._write_seed_panel("f7_seed_panel_test_parent_v1", [1234])
        parent_spec_path = self._write_spec(
            campaign_id="f7_campaign_runner_parent_v1",
            campaign_spec_id="f7_campaign_runner_parent_spec_v1",
            seed_panel_path=base_seed_panel,
            seed_set_id="f7_seed_panel_test_parent_v1",
            inventory_path=inventory_path,
            output_name_prefix="parent",
        )
        initialize_campaign_from_spec(parent_spec_path)
        parent_paths = build_campaign_paths("f7_campaign_runner_parent_v1")
        parent_manifest = load_campaign_manifest(parent_paths)
        parent_manifest["campaign_status"] = "closed_success"
        parent_paths.campaign_manifest_path.write_text(json.dumps(parent_manifest, indent=2, sort_keys=True), encoding="utf-8")

        child_seed_panel = self._write_seed_panel("f7_seed_panel_test_child_v1", [2345])
        child_spec_path = self._write_spec(
            campaign_id="f7_campaign_runner_child_v1",
            campaign_spec_id="f7_campaign_runner_child_spec_v1",
            seed_panel_path=child_seed_panel,
            seed_set_id="f7_seed_panel_test_child_v1",
            inventory_path=inventory_path,
            parent_campaign_id="f7_campaign_runner_parent_v1",
            root_campaign_id="f7_campaign_runner_parent_v1",
            campaign_kind="extension",
            extension_type="seed_extension",
            is_primary_analysis_campaign=False,
            output_name_prefix="child",
        )
        child_materialized = materialize_f7_campaign_spec(spec_path=child_spec_path, write_outputs=False)
        lineage = validate_extension_lineage(spec=child_materialized.spec, materialized=child_materialized)
        self.assertTrue(lineage["ok"])

        overlapping_seed_panel = self._write_seed_panel("f7_seed_panel_test_child_overlap_v1", [1234])
        overlapping_spec_path = self._write_spec(
            campaign_id="f7_campaign_runner_child_overlap_v1",
            campaign_spec_id="f7_campaign_runner_child_overlap_spec_v1",
            seed_panel_path=overlapping_seed_panel,
            seed_set_id="f7_seed_panel_test_child_overlap_v1",
            inventory_path=inventory_path,
            parent_campaign_id="f7_campaign_runner_parent_v1",
            root_campaign_id="f7_campaign_runner_parent_v1",
            campaign_kind="extension",
            extension_type="seed_extension",
            is_primary_analysis_campaign=False,
            output_name_prefix="child_overlap",
        )
        overlapping_materialized = materialize_f7_campaign_spec(spec_path=overlapping_spec_path, write_outputs=False)
        overlap_lineage = validate_extension_lineage(
            spec=overlapping_materialized.spec,
            materialized=overlapping_materialized,
        )
        self.assertFalse(overlap_lineage["ok"])
        self.assertIn("seed_overlap_with_parent_lineage", overlap_lineage["issues"])

    def test_multi_generation_seed_extension_lineage_validation(self) -> None:
        inventory_path = self._write_small_inventory()

        primary_seed_panel = self._write_seed_panel("f7_seed_panel_test_lineage_primary_v1", [1234])
        primary_spec_path = self._write_spec(
            campaign_id="f7_campaign_runner_lineage_primary_v1",
            campaign_spec_id="f7_campaign_runner_lineage_primary_spec_v1",
            seed_panel_path=primary_seed_panel,
            seed_set_id="f7_seed_panel_test_lineage_primary_v1",
            inventory_path=inventory_path,
            output_name_prefix="lineage_primary",
        )
        initialize_campaign_from_spec(primary_spec_path)
        primary_paths = build_campaign_paths("f7_campaign_runner_lineage_primary_v1")
        primary_manifest = load_campaign_manifest(primary_paths)
        primary_manifest["campaign_status"] = "closed_success"
        primary_paths.campaign_manifest_path.write_text(json.dumps(primary_manifest, indent=2, sort_keys=True), encoding="utf-8")

        ext1_seed_panel = self._write_seed_panel("f7_seed_panel_test_lineage_ext1_v1", [2345])
        ext1_spec_path = self._write_spec(
            campaign_id="f7_campaign_runner_lineage_ext1_v1",
            campaign_spec_id="f7_campaign_runner_lineage_ext1_spec_v1",
            seed_panel_path=ext1_seed_panel,
            seed_set_id="f7_seed_panel_test_lineage_ext1_v1",
            inventory_path=inventory_path,
            parent_campaign_id="f7_campaign_runner_lineage_primary_v1",
            root_campaign_id="f7_campaign_runner_lineage_primary_v1",
            campaign_kind="extension",
            extension_type="seed_extension",
            is_primary_analysis_campaign=False,
            output_name_prefix="lineage_ext1",
        )
        initialize_campaign_from_spec(ext1_spec_path)
        ext1_paths = build_campaign_paths("f7_campaign_runner_lineage_ext1_v1")
        ext1_manifest = load_campaign_manifest(ext1_paths)
        ext1_manifest["campaign_status"] = "closed_success"
        ext1_paths.campaign_manifest_path.write_text(json.dumps(ext1_manifest, indent=2, sort_keys=True), encoding="utf-8")

        ext2_seed_panel = self._write_seed_panel("f7_seed_panel_test_lineage_ext2_v1", [3456])
        ext2_spec_path = self._write_spec(
            campaign_id="f7_campaign_runner_lineage_ext2_v1",
            campaign_spec_id="f7_campaign_runner_lineage_ext2_spec_v1",
            seed_panel_path=ext2_seed_panel,
            seed_set_id="f7_seed_panel_test_lineage_ext2_v1",
            inventory_path=inventory_path,
            parent_campaign_id="f7_campaign_runner_lineage_ext1_v1",
            root_campaign_id="f7_campaign_runner_lineage_primary_v1",
            campaign_kind="extension",
            extension_type="seed_extension",
            is_primary_analysis_campaign=False,
            output_name_prefix="lineage_ext2",
        )
        ext2_materialized = materialize_f7_campaign_spec(spec_path=ext2_spec_path, write_outputs=False)
        ext2_lineage = validate_extension_lineage(spec=ext2_materialized.spec, materialized=ext2_materialized)
        self.assertTrue(ext2_lineage["ok"])

        ext2_overlap_seed_panel = self._write_seed_panel("f7_seed_panel_test_lineage_ext2_overlap_v1", [1234])
        ext2_overlap_spec_path = self._write_spec(
            campaign_id="f7_campaign_runner_lineage_ext2_overlap_v1",
            campaign_spec_id="f7_campaign_runner_lineage_ext2_overlap_spec_v1",
            seed_panel_path=ext2_overlap_seed_panel,
            seed_set_id="f7_seed_panel_test_lineage_ext2_overlap_v1",
            inventory_path=inventory_path,
            parent_campaign_id="f7_campaign_runner_lineage_ext1_v1",
            root_campaign_id="f7_campaign_runner_lineage_primary_v1",
            campaign_kind="extension",
            extension_type="seed_extension",
            is_primary_analysis_campaign=False,
            output_name_prefix="lineage_ext2_overlap",
        )
        ext2_overlap_materialized = materialize_f7_campaign_spec(
            spec_path=ext2_overlap_spec_path,
            write_outputs=False,
        )
        ext2_overlap_lineage = validate_extension_lineage(
            spec=ext2_overlap_materialized.spec,
            materialized=ext2_overlap_materialized,
        )
        self.assertFalse(ext2_overlap_lineage["ok"])
        self.assertIn("seed_overlap_with_parent_lineage", ext2_overlap_lineage["issues"])

    def test_warning_classification_known_noise_vs_surfaced(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn_explicit(
                "`torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.",
                FutureWarning,
                "/tmp/site-packages/torch/nn/utils/weight_norm.py",
                143,
            )
            warnings.warn_explicit(
                "something new happened",
                UserWarning,
                "/tmp/site-packages/sklearn/utils/validation.py",
                2749,
            )
        known_noise = _classify_warning(caught[0])
        surfaced = _classify_warning(caught[1])
        self.assertEqual(known_noise["warning_policy"], "silenced_known_noise")
        self.assertTrue(known_noise["warning_is_known_noise"])
        self.assertEqual(surfaced["warning_policy"], "surfaced")
        self.assertFalse(surfaced["warning_is_known_noise"])

    def test_extract_state_fields_and_rebuild_state_rehydrates_flat_fields(self) -> None:
        contract_bundle = {
            **get_f7_analysis_contract_bundle_with_paths(),
            "panel_build_timestamp": "2026-05-21T10:00:00+00:00",
        }
        inventory_path = self._write_small_inventory()
        seed_panel_path = self._write_seed_panel("f7_seed_panel_rebuild_v1", [1234])
        spec_path = self._write_spec(
            campaign_id="f7_campaign_runner_rebuild_v1",
            campaign_spec_id="f7_campaign_runner_rebuild_spec_v1",
            seed_panel_path=seed_panel_path,
            seed_set_id="f7_seed_panel_rebuild_v1",
            inventory_path=inventory_path,
            output_name_prefix="rebuild",
        )
        _, materialized = initialize_campaign_from_spec(spec_path)
        trial_row = next(row for row in materialized.trials if row["model_family"] == "mlp")
        campaign_id = "f7_campaign_runner_rebuild_v1"
        paths = build_campaign_paths(campaign_id)
        attempt_index = 1
        attempt_id = derive_attempt_id(trial_row["trial_id"], attempt_index)
        run_id = derive_run_id(trial_row["trial_id"], attempt_index)
        run_dir = Path("outputs/models/mlp/campaigns") / campaign_id / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = run_dir / "metrics_long.csv"
        metrics_path.write_text(
            "split,metric_name,metric_scope,value_space,metric_value\n"
            "train,rrmse,macro,raw_real,0.05\n"
            "train,rrmse,overall,raw_real,0.05\n"
            "train,rrmse,per_class,raw_real,0.05\n"
            "val,rrmse,macro,raw_real,0.1\n"
            "val,rrmse,overall,raw_real,0.1\n"
            "val,rrmse,per_class,raw_real,0.1\n"
            "test,rrmse,macro,raw_real,0.2\n"
            "test,rrmse,overall,raw_real,0.2\n"
            "test,rrmse,per_class,raw_real,0.2\n",
            encoding="utf-8",
        )
        results_path = run_dir / "results.yaml"
        results_path.write_text("ok: true\n", encoding="utf-8")
        prediction_path = run_dir / "predictions_eval_raw.csv.gz"
        with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
            handle.write("post_cleaning_index,split,type,y_true,y_pred\n1,val,0,1,1.1\n")
        interp_summary = run_dir / "interpretability_summary.json"
        interp_summary.write_text(json.dumps({"interpretability_runtime_s": 0.25}), encoding="utf-8")
        feature_global = run_dir / "feature_influence_global.csv"
        feature_global.write_text(
            "split,feature_name,mean_abs_delta_pred_raw,rank_abs\nval,f1,0.2,1\n",
            encoding="utf-8",
        )
        feature_per_class = run_dir / "feature_influence_per_class.csv"
        feature_per_class.write_text(
            "split,type,feature_name,mean_abs_delta_pred_raw,rank_abs\nval,0,f1,0.2,1\n",
            encoding="utf-8",
        )
        input_feature_global = run_dir / "input_feature_influence_global.csv"
        input_feature_global.write_text(
            "split,feature_name,mean_abs_delta_pred_raw,rank_abs\nval,f1,0.1,1\n",
            encoding="utf-8",
        )
        input_feature_per_class = run_dir / "input_feature_influence_per_class.csv"
        input_feature_per_class.write_text(
            "split,type,feature_name,mean_abs_delta_pred_raw,rank_abs\nval,0,f1,0.1,1\n",
            encoding="utf-8",
        )
        latent_global = run_dir / "latent_feature_influence_global.csv"
        latent_global.write_text(
            "split,latent_name,mean_abs_delta_pred_raw,rank_abs\nval,z_1,0.05,1\n",
            encoding="utf-8",
        )
        latent_per_class = run_dir / "latent_feature_influence_per_class.csv"
        latent_per_class.write_text(
            "split,type,latent_name,mean_abs_delta_pred_raw,rank_abs\nval,0,z_1,0.05,1\n",
            encoding="utf-8",
        )
        config_path = paths.effective_configs_dir / f"{attempt_id}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("seed: 1234\n", encoding="utf-8")
        flowpre_cache = Path("outputs/reports/f7_flowpre_projection_cache/test_rebuild")
        flowpre_cache.mkdir(parents=True, exist_ok=True)
        flowpre_manifest = flowpre_cache / "projection_manifest.json"
        flowpre_manifest.write_text(json.dumps({"ok": True}), encoding="utf-8")

        run_manifest = {
            "trial_id": trial_row["trial_id"],
            "run_id": run_id,
            "contract_id": "f7_contract_v1",
            "variant_fingerprint": "vf_rebuild",
            "config_path": str(config_path.resolve()),
            "config_sha256": "sha256_rebuild",
            "campaign_valid": True,
            "campaign_valid_interpretability": True,
            "campaign_valid_f7": True,
            "results_path": str(results_path.resolve()),
            "metrics_long_path": str(metrics_path.resolve()),
            "prediction_sidecar_path": str(prediction_path.resolve()),
            "interpretability_summary_path": str(interp_summary.resolve()),
            "input_feature_influence_global_path": str(input_feature_global.resolve()),
            "input_feature_influence_per_class_path": str(input_feature_per_class.resolve()),
            "feature_influence_global_path": str(feature_global.resolve()),
            "feature_influence_per_class_path": str(feature_per_class.resolve()),
            "latent_feature_influence_global_path": str(latent_global.resolve()),
            "latent_feature_influence_per_class_path": str(latent_per_class.resolve()),
            "flowpre_projection_manifest_path": str(flowpre_manifest.resolve()),
            "flowpre_projection_cache_path": str(flowpre_cache.resolve()),
            "raw_metric_contract_id": "f7_raw_metric_contract_v1",
            "raw_metric_contract_validation": {
                "validation_status": "ok",
                "value_space_default": "raw_real",
                "raw_inversion_status": {
                    "raw_real_available": True,
                    "requires_raw_inversion": True,
                    "status": "ok",
                },
            },
            "analysis_contracts": contract_bundle,
            "parsed_factor_fields": {
                "x_transform": "flowpre_candidate_1",
                "y_transform": "minmax",
                "synthetic_policy": "none",
                "run_policy": "plain__no_cycling__overall_rmse__allow_synth_true",
                "flowpre_usage": True,
                "flowgen_usage": False,
            },
            "training_summary": {"runtime_s": 1.5},
            "artifact_paths": {
                "results_yaml": str(results_path.resolve()),
                "metrics_long_csv": str(metrics_path.resolve()),
                "predictions_eval_raw_csv_gz": str(prediction_path.resolve()),
                "interpretability_summary_json": str(interp_summary.resolve()),
                "feature_influence_global_csv": str(feature_global.resolve()),
                "feature_influence_per_class_csv": str(feature_per_class.resolve()),
                "input_feature_influence_global_csv": str(input_feature_global.resolve()),
                "input_feature_influence_per_class_csv": str(input_feature_per_class.resolve()),
                "latent_feature_influence_global_csv": str(latent_global.resolve()),
                "latent_feature_influence_per_class_csv": str(latent_per_class.resolve()),
                "flowpre_projection_manifest_json": str(flowpre_manifest.resolve()),
                "flowpre_projection_cache_path": str(flowpre_cache.resolve()),
            },
            "interpretability_status": {
                "interpretability_artifacts": {
                    "interpretability_summary_json": str(interp_summary.resolve()),
                    "feature_influence_global_csv": str(feature_global.resolve()),
                    "feature_influence_per_class_csv": str(feature_per_class.resolve()),
                    "input_feature_influence_global_csv": str(input_feature_global.resolve()),
                    "input_feature_influence_per_class_csv": str(input_feature_per_class.resolve()),
                    "latent_feature_influence_global_csv": str(latent_global.resolve()),
                    "latent_feature_influence_per_class_csv": str(latent_per_class.resolve()),
                    "flowpre_projection_manifest_json": str(flowpre_manifest.resolve()),
                    "flowpre_projection_cache_path": str(flowpre_cache.resolve()),
                }
            },
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")

        extracted = _extract_state_fields_from_run_manifest(
            run_manifest=run_manifest,
            family="mlp",
            run_dir=run_dir,
        )
        self.assertEqual(extracted["raw_metric_contract_validation_status"], "ok")
        self.assertEqual(extracted["target_contract_id"], contract_bundle["target_contract_id"])
        self.assertEqual(extracted["panel_build_timestamp"], contract_bundle["panel_build_timestamp"])
        self.assertEqual(extracted["feature_namespace"], "flowpre_projected_semantic_input")
        self.assertTrue(extracted["analysis_ready_comparable"])
        self.assertIsNotNone(extracted["feature_influence_global_path"])
        self.assertIsNotNone(extracted["input_feature_influence_global_path"])
        self.assertIsNotNone(extracted["latent_feature_influence_global_path"])

        warning_log = paths.warnings_dir / f"{attempt_id}.jsonl"
        warning_log.parent.mkdir(parents=True, exist_ok=True)
        warning_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "category": "FutureWarning",
                            "warning_policy": "silenced_known_noise",
                            "warning_signature": "FutureWarning::torch.nn.utils.weight_norm::x",
                        }
                    ),
                    json.dumps(
                        {
                            "category": "UserWarning",
                            "warning_policy": "surfaced",
                            "warning_signature": "UserWarning::sklearn.utils.validation::y",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = rebuild_campaign_state(campaign_id=campaign_id)
        self.assertEqual(result["campaign_id"], campaign_id)
        rebuilt_state = load_trial_state(paths, trial_row["trial_id"])
        self.assertEqual(rebuilt_state["execution_status"], "completed")
        self.assertEqual(rebuilt_state["raw_metric_contract_validation_status"], "ok")
        self.assertEqual(rebuilt_state["target_contract_id"], contract_bundle["target_contract_id"])
        self.assertEqual(rebuilt_state["panel_build_timestamp"], contract_bundle["panel_build_timestamp"])
        self.assertEqual(
            rebuilt_state["prediction_row_join_contract_id"],
            contract_bundle["prediction_row_join_contract_id"],
        )
        self.assertEqual(rebuilt_state["feature_namespace"], "flowpre_projected_semantic_input")
        self.assertTrue(rebuilt_state["analysis_ready_comparable"])
        self.assertEqual(rebuilt_state["analysis_ready_blockers"], [])
        self.assertEqual(rebuilt_state["observed_seed_count"], 1)
        self.assertEqual(rebuilt_state["warning_count_total"], 2)
        self.assertEqual(rebuilt_state["warning_count_silenced_known_noise"], 1)
        self.assertEqual(rebuilt_state["warning_count_surfaced"], 1)
        self.assertIsNotNone(rebuilt_state["feature_influence_global_path"])
        self.assertIsNotNone(rebuilt_state["run_manifest_path"])

    def test_extract_state_fields_marks_analysis_not_ready_when_prediction_join_contract_fails(self) -> None:
        contract_bundle = {
            **get_f7_analysis_contract_bundle_with_paths(),
            "panel_build_timestamp": "2026-05-21T10:00:00+00:00",
        }
        run_dir = self.tmp_path / "bad_sidecar"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics_long.csv"
        metrics_path.write_text(
            "split,metric_name,metric_scope,value_space,metric_value\n"
            "val,rrmse,macro,raw_real,0.1\n"
            "val,rrmse,overall,raw_real,0.1\n"
            "val,rrmse,per_class,raw_real,0.1\n",
            encoding="utf-8",
        )
        prediction_path = run_dir / "predictions_eval_raw.csv.gz"
        with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
            handle.write("y_true,y_pred\n1,1.1\n")
        interp_summary = run_dir / "interpretability_summary.json"
        interp_summary.write_text(json.dumps({"interpretability_runtime_s": 0.25}), encoding="utf-8")
        feature_global = run_dir / "feature_influence_global.csv"
        feature_global.write_text(
            "split,feature_name,mean_abs_delta_pred_raw,rank_abs\nval,f1,0.2,1\n",
            encoding="utf-8",
        )
        feature_per_class = run_dir / "feature_influence_per_class.csv"
        feature_per_class.write_text(
            "split,type,feature_name,mean_abs_delta_pred_raw,rank_abs\nval,0,f1,0.2,1\n",
            encoding="utf-8",
        )
        run_manifest = {
            "trial_id": "trial__bad_sidecar",
            "run_id": "trial__bad_sidecar__attempt-0001",
            "campaign_valid_f7": True,
            "results_path": str((run_dir / "results.yaml").resolve()),
            "metrics_long_path": str(metrics_path.resolve()),
            "prediction_sidecar_path": str(prediction_path.resolve()),
            "interpretability_summary_path": str(interp_summary.resolve()),
            "feature_influence_global_path": str(feature_global.resolve()),
            "feature_influence_per_class_path": str(feature_per_class.resolve()),
            "raw_metric_contract_id": "f7_raw_metric_contract_v1",
            "raw_metric_contract_validation": {
                "validation_status": "ok",
                "value_space_default": "raw_real",
                "raw_inversion_status": {
                    "raw_real_available": True,
                    "requires_raw_inversion": False,
                    "status": "not_required",
                },
            },
            "analysis_contracts": contract_bundle,
            "parsed_factor_fields": {
                "x_transform": "standard",
                "y_transform": "minmax",
                "synthetic_policy": "none",
                "run_policy": "plain__cycling__overall_rmse__allow_synth-true",
                "flowpre_usage": False,
                "flowgen_usage": False,
            },
        }
        (run_dir / "results.yaml").write_text("ok: true\n", encoding="utf-8")
        extracted = _extract_state_fields_from_run_manifest(
            run_manifest=run_manifest,
            family="mlp",
            run_dir=run_dir,
        )
        self.assertFalse(extracted["analysis_ready_comparable"])
        self.assertIn("prediction_sidecar_missing_column:post_cleaning_index", extracted["analysis_ready_blockers"])


if __name__ == "__main__":
    unittest.main()
