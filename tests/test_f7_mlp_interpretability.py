from __future__ import annotations

import json
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from evaluation.meta_context import get_f7_analysis_contract_bundle_with_paths
from evaluation.flowpre_projection import _build_projection_rows
from evaluation.compositional_projection import (
    build_final_semantic_surface_spec,
    distribute_event_deltas_to_surfaces,
    inverse_ilr_to_normalized_components,
)
from evaluation.mlp_interpretability import (
    aggregate_effect_matrices,
    compute_class_conditioned_feature_means,
    compute_mlp_feature_delta_matrix,
    project_latent_effects_to_semantic_space,
)
from training.train_mlp import train_mlp_model


class _ToyMLPModel:
    def __call__(self, x, c):
        c_term = c.to(torch.float32).reshape(-1, 1) * 0.5
        return (2.0 * x[:, :1]) - x[:, 1:2] + c_term


class _DummyFlowPreModel:
    def inverse(self, z, c):
        c_term = c.to(torch.float32).reshape(-1, 1)
        x_rec = torch.stack(
            (
                z[:, 0] + (0.1 * c_term[:, 0]),
                (2.0 * z[:, 1]) - (0.2 * c_term[:, 0]),
                z[:, 0] - z[:, 1],
            ),
            dim=1,
        )
        return x_rec, None


def _make_standard_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    def _split(start_idx: int, rows_per_class: int, offset: float) -> pd.DataFrame:
        rows: list[dict[str, float | int]] = []
        idx = start_idx
        for cls in (0, 1, 2):
            for step in range(rows_per_class):
                f1 = float(cls) + (0.2 * step) + offset
                f2 = (float(cls) * 0.5) + (0.1 * step) - offset
                f3 = (float(cls) * -0.3) + (0.05 * step) + offset
                init = 8.0 + (1.7 * cls) + (0.4 * f1) - (0.2 * f2) + (0.1 * f3)
                rows.append(
                    {
                        "post_cleaning_index": idx,
                        "type": cls,
                        "f1": f1,
                        "f2": f2,
                        "f3": f3,
                        "init": init,
                    }
                )
                idx += 1
        return pd.DataFrame(rows)

    train = _split(0, 6, 0.0)
    val = _split(100, 4, 0.15)
    test = _split(200, 4, 0.3)
    scaler = StandardScaler().fit(train[["init"]].to_numpy())
    for frame in (train, val, test):
        frame["init"] = scaler.transform(frame[["init"]].to_numpy()).reshape(-1)
    return train, val, test, scaler


def _make_flowpre_latent_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    def _split(start_idx: int, rows_per_class: int, offset: float) -> pd.DataFrame:
        rows: list[dict[str, float | int]] = []
        idx = start_idx
        for cls in (0, 1, 2):
            for step in range(rows_per_class):
                z0 = float(cls) + (0.25 * step) + offset
                z1 = (float(cls) * -0.4) + (0.15 * step) - offset
                z2 = (float(cls) * 0.6) - (0.05 * step) + offset
                init = 9.0 + (0.8 * z0) - (0.5 * z1) + (0.2 * z2)
                rows.append(
                    {
                        "post_cleaning_index": idx,
                        "type": cls,
                        "z_0": z0,
                        "z_1": z1,
                        "z_2": z2,
                        "init": init,
                    }
                )
                idx += 1
        return pd.DataFrame(rows)

    train = _split(0, 6, 0.0)
    val = _split(100, 4, 0.1)
    test = _split(200, 4, 0.2)
    scaler = StandardScaler().fit(train[["init"]].to_numpy())
    for frame in (train, val, test):
        frame["init"] = scaler.transform(frame[["init"]].to_numpy()).reshape(-1)
    return train, val, test, scaler


def _write_mlp_config(path: Path) -> None:
    payload = {
        "contract": {
            "closure_contract_id": "f7_contract_v1",
            "mlp_base_config_id": "f7_mlp_interpretability_test_v1",
            "seed_set_id": "f7_seed_panel_v1",
            "objective_metric_id": "raw_real.macro.rrmse",
            "allow_test_holdout_default": False,
        },
        "model": {
            "embedding_dim": 4,
            "hidden_dim": 16,
            "num_layers": 2,
            "activation": "relu",
            "dropout": 0.0,
            "batchnorm": False,
            "use_weight_norm": False,
            "residual": False,
            "final_activation": None,
            "task": "regression",
            "context_mode": "embed",
            "target_names": ["init"],
        },
        "training": {
            "optimizer": "adam",
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "batch_size": 6,
            "learning_rate": 1e-3,
            "num_epochs": 3,
            "early_stopping_patience": 3,
            "early_stop_min_delta": 0.0,
            "lr_scheduler": "plateau",
            "lr_decay_factor": 0.5,
            "lr_decay_patience": 2,
            "plateau_threshold": 1e-4,
            "plateau_threshold_mode": "rel",
            "lr_cooldown": 0,
            "min_lr": 0.0,
            "save_states": False,
            "log_training": False,
            "save_results": True,
            "save_model": False,
            "loss_reduction": "overall",
            "regression_group_metric": "rmse",
            "dataloader_mode": "baseline",
            "cycle_reals": False,
            "allow_synth": True,
        },
        "interpretability": {
            "compute_shap": False,
            "save_influence": False,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _dataset_manifest_payload(*, x_transform: str, y_transform: str, upstream_model_manifests: list[str] | None = None) -> dict:
    return {
        "dataset_name": f"dataset__x-{x_transform}__y-{y_transform}__syn-none__v1",
        "dataset_level_axes": {
            "x_transform": x_transform,
            "y_transform": y_transform,
            "synthetic_policy": "none",
        },
        "scaler_artifacts": {"y": "fake_y_scaler.pkl"},
        "upstream_model_manifests": list(upstream_model_manifests or []),
    }


class TestF7MLPInterpretability(unittest.TestCase):
    def test_compute_mlp_feature_delta_matrix_matches_expected_linear_effects(self) -> None:
        x_eval = np.asarray([[3.0, 4.0], [10.0, 20.0]], dtype=np.float32)
        class_codes = np.asarray([0, 1], dtype=np.int64)
        class_feature_means = {
            0: np.asarray([1.0, 1.0], dtype=np.float32),
            1: np.asarray([5.0, 5.0], dtype=np.float32),
        }
        y_pred_native = np.asarray([[2.0], [0.5]], dtype=np.float32)
        delta_matrix = compute_mlp_feature_delta_matrix(
            model=_ToyMLPModel(),
            x_eval=x_eval,
            class_codes=class_codes,
            y_pred_native=y_pred_native,
            class_feature_means=class_feature_means,
            y_scaler=None,
            device=torch.device("cpu"),
            chunk_size=2,
        )
        expected = np.asarray(
            [
                [4.0, -3.0],
                [10.0, -15.0],
            ],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(delta_matrix, expected))

    def test_class_conditioned_feature_means(self) -> None:
        means = compute_class_conditioned_feature_means(
            np.asarray([[1.0, 10.0], [3.0, 30.0], [2.0, 20.0], [4.0, 40.0]], dtype=np.float32),
            np.asarray([0, 1, 0, 1], dtype=np.int64),
        )
        self.assertEqual(sorted(means.keys()), [0, 1])
        self.assertTrue(np.allclose(means[0], np.asarray([1.5, 15.0], dtype=np.float32)))
        self.assertTrue(np.allclose(means[1], np.asarray([3.5, 35.0], dtype=np.float32)))

    def test_aggregate_effect_matrices_emits_global_and_per_class_views(self) -> None:
        global_df, per_class_df = aggregate_effect_matrices(
            split_name="val",
            item_names=["f1", "f2"],
            class_codes=np.asarray([0, 0, 1], dtype=np.int64),
            signed_matrix=np.asarray([[1.0, -2.0], [3.0, -1.0], [5.0, -4.0]], dtype=np.float32),
            abs_matrix=np.asarray([[1.0, 2.0], [3.0, 1.0], [5.0, 4.0]], dtype=np.float32),
            name_col="feature_name",
            feature_space_kind="model_input_space",
            projection_status="direct_semantic",
        )
        self.assertEqual(sorted(global_df["feature_name"].tolist()), ["f1", "f2"])
        self.assertEqual(sorted(per_class_df["type"].unique().tolist()), [0, 1])
        self.assertTrue((global_df["rank_abs"] >= 1).all())
        self.assertTrue((per_class_df["rank_abs"] >= 1).all())
        for column in [
            "sum_abs_delta_pred_raw",
            "std_abs_delta_pred_raw",
            "median_abs_delta_pred_raw",
            "p90_abs_delta_pred_raw",
            "p95_abs_delta_pred_raw",
            "stderr_abs_delta_pred_raw",
            "share_abs_importance",
        ]:
            self.assertIn(column, global_df.columns)
            self.assertIn(column, per_class_df.columns)
        f1_global = global_df.loc[global_df["feature_name"] == "f1"].iloc[0]
        self.assertAlmostEqual(float(f1_global["mean_abs_delta_pred_raw"]), 3.0, places=6)
        self.assertAlmostEqual(float(f1_global["mean_signed_delta_pred_raw"]), 3.0, places=6)
        self.assertAlmostEqual(float(global_df["share_abs_importance"].sum()), 1.0, places=6)
        share_sums = per_class_df.groupby(["split", "type"], sort=False)["share_abs_importance"].sum().tolist()
        self.assertTrue(all(abs(float(value) - 1.0) <= 1e-6 for value in share_sums))

    def test_project_latent_effects_to_semantic_space_uses_class_specific_weights(self) -> None:
        projection_table = pd.DataFrame(
            [
                {"type": 0, "latent_name": "z_0", "semantic_feature": "chem_a", "weight_raw_std": 1.0, "weight_norm": 0.8},
                {"type": 0, "latent_name": "z_0", "semantic_feature": "chem_b", "weight_raw_std": 1.0, "weight_norm": 0.2},
                {"type": 0, "latent_name": "z_1", "semantic_feature": "chem_a", "weight_raw_std": 1.0, "weight_norm": 0.1},
                {"type": 0, "latent_name": "z_1", "semantic_feature": "chem_b", "weight_raw_std": 1.0, "weight_norm": 0.9},
                {"type": 1, "latent_name": "z_0", "semantic_feature": "chem_a", "weight_raw_std": 1.0, "weight_norm": 0.3},
                {"type": 1, "latent_name": "z_0", "semantic_feature": "chem_b", "weight_raw_std": 1.0, "weight_norm": 0.7},
                {"type": 1, "latent_name": "z_1", "semantic_feature": "chem_a", "weight_raw_std": 1.0, "weight_norm": 0.6},
                {"type": 1, "latent_name": "z_1", "semantic_feature": "chem_b", "weight_raw_std": 1.0, "weight_norm": 0.4},
            ]
        )
        signed_semantic, abs_semantic, semantic_names = project_latent_effects_to_semantic_space(
            signed_latent_matrix=np.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32),
            class_codes=np.asarray([0, 1], dtype=np.int64),
            latent_names=["z_0", "z_1"],
            projection_table=projection_table,
        )
        self.assertEqual(semantic_names, ["chem_a", "chem_b"])
        self.assertTrue(np.allclose(signed_semantic[0], np.asarray([0.6, -1.6], dtype=np.float32)))
        self.assertTrue(np.allclose(abs_semantic[0], np.asarray([1.0, 2.0], dtype=np.float32)))
        self.assertTrue(np.allclose(signed_semantic[1], np.asarray([3.3, 3.7], dtype=np.float32)))

    def test_inverse_ilr_to_normalized_components_returns_valid_composition(self) -> None:
        restored = inverse_ilr_to_normalized_components(
            np.asarray([[0.2, -0.1]], dtype=np.float32),
            n_components=3,
        )
        self.assertEqual(restored.shape, (1, 3))
        self.assertAlmostEqual(float(restored.sum()), 1.0, places=6)
        self.assertTrue((restored >= 0.0).all())

    def test_inverse_ilr_to_normalized_components_is_stable_for_large_values(self) -> None:
        restored = inverse_ilr_to_normalized_components(
            np.asarray([[50.0, -40.0], [-60.0, 55.0]], dtype=np.float32),
            n_components=3,
        )
        self.assertFalse(np.isnan(restored).any())
        self.assertTrue(np.allclose(restored.sum(axis=1), np.ones(2), atol=1e-6))

    def test_distribute_event_deltas_projects_ilr_group_and_preserves_abs_mass(self) -> None:
        distributed = distribute_event_deltas_to_surfaces(
            signed_event_deltas=np.asarray([[2.0], [-3.0]], dtype=np.float32),
            input_feature_names=["ilr_chem_1", "ilr_chem_2", "water"],
            original_input=np.asarray(
                [
                    [0.1, -0.2, 1.0],
                    [0.3, 0.0, 2.0],
                ],
                dtype=np.float32,
            ),
            perturbed_inputs=np.asarray(
                [
                    [
                        [0.0, -0.2, 1.0],
                        [0.1, 0.0, 2.0],
                    ]
                ],
                dtype=np.float32,
            ),
            component_names_by_group={"chem": ["chem_a", "chem_b", "chem_c"]},
            fallback_input_indices=[0],
        )
        spec = distributed["surface_spec"]
        self.assertEqual(spec.final_feature_names, ["water", "chem_a", "chem_b", "chem_c"])
        self.assertAlmostEqual(float(distributed["final_abs"][0].sum()), 2.0, places=6)
        self.assertAlmostEqual(float(distributed["final_abs"][1].sum()), 3.0, places=6)
        self.assertFalse(any(name.startswith("ilr_") for name in spec.final_feature_names))

    def test_distribute_event_deltas_keeps_direct_features_as_passthrough(self) -> None:
        distributed = distribute_event_deltas_to_surfaces(
            signed_event_deltas=np.asarray([[1.5]], dtype=np.float32),
            input_feature_names=["water", "d90"],
            original_input=np.asarray([[1.0, 2.0]], dtype=np.float32),
            perturbed_inputs=np.asarray([[[0.25, 2.0]]], dtype=np.float32),
            fallback_input_indices=[0],
        )
        spec = distributed["surface_spec"]
        self.assertEqual(spec.final_feature_names, ["water", "d90"])
        self.assertTrue(np.allclose(distributed["input_abs"], distributed["final_abs"]))
        self.assertAlmostEqual(float(distributed["final_abs"].sum()), 1.5, places=6)

    def test_build_final_semantic_surface_spec_excludes_ilr_names_from_final_surface(self) -> None:
        spec = build_final_semantic_surface_spec(
            ["water", "ilr_chem_1", "ilr_chem_2", "ilr_phase_1", "ilr_phase_2", "d90"],
            component_names_by_group={
                "chem": ["chem_a", "chem_b", "chem_c"],
                "phase": ["phase_a", "phase_b", "phase_c"],
            },
        )
        self.assertEqual(spec.direct_feature_names, ["water", "d90"])
        self.assertFalse(any(name.startswith("ilr_") for name in spec.final_feature_names))
        self.assertEqual(
            spec.final_feature_names,
            ["water", "d90", "chem_a", "chem_b", "chem_c", "phase_a", "phase_b", "phase_c"],
        )

    def test_flowpre_projection_rows_are_class_conditioned_and_row_normalized(self) -> None:
        x_train_raw = pd.DataFrame(
            [
                {"post_cleaning_index": 0, "type": 0, "chem_a": 1.0, "chem_b": 2.0, "chem_c": 3.0},
                {"post_cleaning_index": 1, "type": 0, "chem_a": 1.5, "chem_b": 2.5, "chem_c": 3.5},
                {"post_cleaning_index": 2, "type": 1, "chem_a": 2.0, "chem_b": 1.0, "chem_c": 0.0},
                {"post_cleaning_index": 3, "type": 1, "chem_a": 2.5, "chem_b": 1.5, "chem_c": 0.5},
            ]
        )
        encoded_train = pd.DataFrame(
            [
                {"type": 0, "z_0": 0.2, "z_1": -0.1},
                {"type": 0, "z_0": 0.4, "z_1": 0.2},
                {"type": 1, "z_0": 1.1, "z_1": -0.3},
                {"type": 1, "z_0": 1.3, "z_1": 0.1},
            ]
        )
        fake_flowpre_module = types.ModuleType("training.train_flow_pre")
        fake_flowpre_module.encode_with_flowpre_model = lambda *args, **kwargs: encoded_train
        with patch.dict(sys.modules, {"training.train_flow_pre": fake_flowpre_module}):
            projection_df, stats = _build_projection_rows(
                model=_DummyFlowPreModel(),
                device=torch.device("cpu"),
                x_train_raw=x_train_raw,
                condition_col="type",
            )
        self.assertEqual(stats["n_classes"], 2)
        self.assertEqual(stats["n_latent_features"], 2)
        self.assertEqual(stats["n_semantic_features"], 3)
        self.assertEqual(sorted(projection_df["type"].unique().tolist()), [0, 1])
        self.assertEqual(sorted(projection_df["latent_name"].unique().tolist()), ["z_0", "z_1"])
        self.assertEqual(sorted(projection_df["semantic_feature"].unique().tolist()), ["chem_a", "chem_b", "chem_c"])
        row_sums = (
            projection_df.groupby(["type", "latent_name"], sort=True)["weight_norm"].sum().round(6).tolist()
        )
        self.assertEqual(row_sums, [1.0, 1.0, 1.0, 1.0])

    def test_train_mlp_standard_selection_persists_interpretability(self) -> None:
        train_df, val_df, _test_df, scaler = _make_standard_frames()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cfg_path = tmpdir_path / "mlp_standard.yaml"
            _write_mlp_config(cfg_path)
            dataset_manifest_path = tmpdir_path / "dataset_manifest.json"
            dataset_manifest_path.write_text(
                json.dumps(_dataset_manifest_payload(x_transform="standard", y_transform="standard"), indent=2),
                encoding="utf-8",
            )
            run_dir = tmpdir_path / "mlp_selection_run"
            snapshots_dir = run_dir / "snapshots"
            run_dir.mkdir(parents=True, exist_ok=True)
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            with patch(
                "training.train_mlp.setup_training_logs_and_dirs",
                return_value=(run_dir, "mlp_selection_run", None, snapshots_dir),
            ):
                train_mlp_model(
                    train_df.copy(),
                    val_df.copy(),
                    "type",
                    allow_test_holdout=False,
                    seed=1234,
                    config_filename=str(cfg_path),
                    base_name="mlp_selection_run",
                    device="cpu",
                    verbose=False,
                    evaluation_context={
                        "dataset_name": "mlp__x-standard__y-standard__syn-none",
                        "dataset_manifest_path": str(dataset_manifest_path),
                        "split_id": "init_temporal_processed_v1",
                        "contract_id": "f7_contract_v1",
                        "analysis_contracts": get_f7_analysis_contract_bundle_with_paths(),
                        "seed_set_id": "f7_seed_panel_v1",
                        "base_config_id": "f7_mlp_interpretability_test_v1",
                        "objective_metric_id": "raw_real.macro.rrmse",
                        "dataset_level_axes": {
                            "x_transform": "standard",
                            "y_transform": "standard",
                            "synthetic_policy": "none",
                        },
                        "y_transform": "standard",
                        "y_scaler": scaler,
                        "target_scaler_artifact": "fake_y_scaler.pkl",
                    },
                )
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "interpretability_summary.json").read_text(encoding="utf-8"))
            input_feature_global = pd.read_csv(run_dir / "input_feature_influence_global.csv")
            feature_global = pd.read_csv(run_dir / "feature_influence_global.csv")
            feature_per_class = pd.read_csv(run_dir / "feature_influence_per_class.csv")
            top_global = pd.read_csv(run_dir / "top_features_global.csv")
            self.assertTrue(manifest["campaign_valid_interpretability"])
            self.assertTrue(manifest["campaign_valid_f7"])
            self.assertEqual(manifest["interpretability_policy_status"], "implemented_mlp_block_10b_v1")
            self.assertTrue(manifest["interpretability_required_now"])
            self.assertTrue(manifest["artifact_availability"]["feature_influence_global_csv"])
            self.assertTrue(manifest["artifact_availability"]["input_feature_influence_global_csv"])
            self.assertFalse(manifest["artifact_availability"]["latent_feature_influence_global_csv"])
            self.assertEqual(summary["feature_space_kind_primary"], "semantic_final_surface")
            self.assertEqual(summary["compositional_projection_status"], "not_needed")
            self.assertEqual(summary["available_splits"], ["val"])
            self.assertEqual(sorted(feature_global["split"].unique().tolist()), ["val"])
            self.assertEqual(sorted(feature_per_class["split"].unique().tolist()), ["val"])
            self.assertEqual(sorted(top_global["split"].unique().tolist()), ["val"])
            self.assertEqual(sorted(input_feature_global["feature_name"].unique().tolist()), ["f1", "f2", "f3"])
            self.assertFalse(feature_global["feature_name"].astype(str).str.startswith("ilr_").any())
            self.assertIn("share_abs_importance", feature_global.columns)
            self.assertIn("p90_abs_delta_pred_raw", feature_global.columns)
            self.assertAlmostEqual(float(feature_global["share_abs_importance"].sum()), 1.0, places=6)
            self.assertFalse((run_dir / "mlp_selection_run_run_manifest.json").exists())
            self.assertFalse((run_dir / "mlp_selection_run_metrics_long.csv").exists())
            self.assertFalse((run_dir / "mlp_selection_run_results.yaml").exists())
            self.assertFalse((run_dir / "mlp_selection_run.yaml").exists())

    def test_train_mlp_standard_holdout_persists_val_and_test_interpretability(self) -> None:
        train_df, val_df, test_df, scaler = _make_standard_frames()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cfg_path = tmpdir_path / "mlp_standard_holdout.yaml"
            _write_mlp_config(cfg_path)
            dataset_manifest_path = tmpdir_path / "dataset_manifest.json"
            dataset_manifest_path.write_text(
                json.dumps(_dataset_manifest_payload(x_transform="standard", y_transform="standard"), indent=2),
                encoding="utf-8",
            )
            run_dir = tmpdir_path / "mlp_holdout_run"
            snapshots_dir = run_dir / "snapshots"
            run_dir.mkdir(parents=True, exist_ok=True)
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            with patch(
                "training.train_mlp.setup_training_logs_and_dirs",
                return_value=(run_dir, "mlp_holdout_run", None, snapshots_dir),
            ):
                train_mlp_model(
                    train_df.copy(),
                    val_df.copy(),
                    "type",
                    cXy_test=test_df.copy(),
                    allow_test_holdout=True,
                    seed=2345,
                    config_filename=str(cfg_path),
                    base_name="mlp_holdout_run",
                    device="cpu",
                    verbose=False,
                    evaluation_context={
                        "dataset_name": "mlp__x-standard__y-standard__syn-none",
                        "dataset_manifest_path": str(dataset_manifest_path),
                        "split_id": "init_temporal_processed_v1",
                        "contract_id": "f7_contract_v1",
                        "analysis_contracts": get_f7_analysis_contract_bundle_with_paths(),
                        "seed_set_id": "f7_seed_panel_v1",
                        "base_config_id": "f7_mlp_interpretability_test_v1",
                        "objective_metric_id": "raw_real.macro.rrmse",
                        "dataset_level_axes": {
                            "x_transform": "standard",
                            "y_transform": "standard",
                            "synthetic_policy": "none",
                        },
                        "y_transform": "standard",
                        "y_scaler": scaler,
                        "target_scaler_artifact": "fake_y_scaler.pkl",
                    },
                )
            summary = json.loads((run_dir / "interpretability_summary.json").read_text(encoding="utf-8"))
            feature_global = pd.read_csv(run_dir / "feature_influence_global.csv")
            self.assertEqual(sorted(summary["available_splits"]), ["test", "val"])
            self.assertEqual(sorted(feature_global["split"].unique().tolist()), ["test", "val"])
            self.assertFalse(feature_global["feature_name"].astype(str).str.startswith("ilr_").any())
            share_sums = feature_global.groupby("split", sort=False)["share_abs_importance"].sum().tolist()
            self.assertTrue(all(abs(float(value) - 1.0) <= 1e-6 for value in share_sums))

    def test_train_mlp_flowpre_selection_uses_cached_projection(self) -> None:
        train_df, val_df, _test_df, scaler = _make_flowpre_latent_frames()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cfg_path = tmpdir_path / "mlp_flowpre.yaml"
            _write_mlp_config(cfg_path)
            fake_promotion_manifest_path = tmpdir_path / "fake_flowpre_promotion_manifest.json"
            fake_promotion_manifest_path.write_text(
                json.dumps(
                    {
                        "source_id": "flowpre__candidate_1__init_temporal_processed_v1__v1",
                        "source_run_manifest": str(tmpdir_path / "unused_run_manifest.json"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            projection_cache_root = tmpdir_path / "projection_cache"
            projection_cache_dir = (
                projection_cache_root / "flowpre__candidate_1__init_temporal_processed_v1__v1__v1"
            )
            projection_cache_dir.mkdir(parents=True, exist_ok=True)
            projection_table = pd.DataFrame(
                [
                    {"type": cls, "latent_name": latent, "semantic_feature": semantic, "weight_raw_std": 1.0, "weight_norm": weight}
                    for cls in (0, 1, 2)
                    for latent, weights in {
                        "z_0": [0.7, 0.2, 0.1],
                        "z_1": [0.2, 0.6, 0.2],
                        "z_2": [0.1, 0.3, 0.6],
                    }.items()
                    for semantic, weight in zip(["chem_a", "chem_b", "chem_c"], weights)
                ]
            )
            projection_table.to_csv(projection_cache_dir / "latent_to_semantic_per_class.csv", index=False)
            (projection_cache_dir / "projection_manifest.json").write_text(
                json.dumps(
                    {
                        "projection_contract_id": "f7_flowpre_projection_cache_v1",
                        "source_id": "flowpre__candidate_1__init_temporal_processed_v1__v1",
                        "projection_table_path": str(projection_cache_dir / "latent_to_semantic_per_class.csv"),
                        "projection_cache_path": str(projection_cache_dir),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            dataset_manifest_path = tmpdir_path / "dataset_manifest.json"
            dataset_manifest_path.write_text(
                json.dumps(
                    _dataset_manifest_payload(
                        x_transform="flowpre_candidate_1",
                        y_transform="standard",
                        upstream_model_manifests=[str(fake_promotion_manifest_path)],
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
            run_dir = tmpdir_path / "mlp_flowpre_selection_run"
            snapshots_dir = run_dir / "snapshots"
            run_dir.mkdir(parents=True, exist_ok=True)
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            with (
                patch(
                    "training.train_mlp.setup_training_logs_and_dirs",
                    return_value=(run_dir, "mlp_flowpre_selection_run", None, snapshots_dir),
                ),
                patch(
                    "evaluation.mlp_interpretability.load_flowpre_decoder_runtime",
                    return_value={
                        "promotion_manifest_path": str(fake_promotion_manifest_path),
                        "promotion_manifest": {"source_id": "flowpre__candidate_1__init_temporal_processed_v1__v1"},
                        "run_manifest_path": str(tmpdir_path / "unused_run_manifest.json"),
                        "run_manifest": {},
                        "decoder_model": _DummyFlowPreModel(),
                        "device": torch.device("cpu"),
                        "semantic_feature_names": ["chem_a", "chem_b", "chem_c"],
                    },
                ),
            ):
                train_mlp_model(
                    train_df.copy(),
                    val_df.copy(),
                    "type",
                    allow_test_holdout=False,
                    seed=3456,
                    config_filename=str(cfg_path),
                    base_name="mlp_flowpre_selection_run",
                    device="cpu",
                    verbose=False,
                    evaluation_context={
                        "dataset_name": "mlp__x-candidate1__y-standard__syn-none",
                        "dataset_manifest_path": str(dataset_manifest_path),
                        "split_id": "init_temporal_processed_v1",
                        "contract_id": "f7_contract_v1",
                        "analysis_contracts": get_f7_analysis_contract_bundle_with_paths(),
                        "seed_set_id": "f7_seed_panel_v1",
                        "base_config_id": "f7_mlp_interpretability_test_v1",
                        "objective_metric_id": "raw_real.macro.rrmse",
                        "dataset_level_axes": {
                            "x_transform": "flowpre_candidate_1",
                            "y_transform": "standard",
                            "synthetic_policy": "none",
                        },
                        "y_transform": "standard",
                        "y_scaler": scaler,
                        "target_scaler_artifact": "fake_y_scaler.pkl",
                        "flowpre_projection_cache_root": str(projection_cache_root),
                    },
                )
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "interpretability_summary.json").read_text(encoding="utf-8"))
            input_feature_global = pd.read_csv(run_dir / "input_feature_influence_global.csv")
            feature_global = pd.read_csv(run_dir / "feature_influence_global.csv")
            latent_global = pd.read_csv(run_dir / "latent_feature_influence_global.csv")
            self.assertTrue(manifest["campaign_valid_interpretability"])
            self.assertTrue(manifest["campaign_valid_f7"])
            self.assertEqual(summary["feature_space_kind_primary"], "semantic_final_surface")
            self.assertTrue(summary["uses_flowpre_projection"])
            self.assertEqual(summary["compositional_projection_status"], "not_needed")
            self.assertIn("share_abs_importance", feature_global.columns)
            share_sums = feature_global.groupby("split", sort=False)["share_abs_importance"].sum().tolist()
            self.assertTrue(all(abs(float(value) - 1.0) <= 1e-6 for value in share_sums))
            self.assertEqual(sorted(input_feature_global["feature_name"].unique().tolist()), ["chem_a", "chem_b", "chem_c"])
            self.assertEqual(sorted(feature_global["feature_name"].unique().tolist()), ["chem_a", "chem_b", "chem_c"])
            self.assertFalse(feature_global["feature_name"].astype(str).str.startswith("ilr_").any())
            self.assertEqual(sorted(latent_global["latent_name"].unique().tolist()), ["z_0", "z_1", "z_2"])
            self.assertIsNotNone(manifest["flowpre_projection_manifest_path"])
            self.assertIsNotNone(manifest["flowpre_projection_cache_path"])

    def test_train_mlp_flowpre_holdout_persists_val_and_test_projected_artifacts(self) -> None:
        train_df, val_df, test_df, scaler = _make_flowpre_latent_frames()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cfg_path = tmpdir_path / "mlp_flowpre_holdout.yaml"
            _write_mlp_config(cfg_path)
            fake_promotion_manifest_path = tmpdir_path / "fake_flowpre_promotion_manifest.json"
            fake_promotion_manifest_path.write_text(
                json.dumps(
                    {
                        "source_id": "flowpre__candidate_2__init_temporal_processed_v1__v1",
                        "source_run_manifest": str(tmpdir_path / "unused_run_manifest.json"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            projection_cache_root = tmpdir_path / "projection_cache"
            projection_cache_dir = (
                projection_cache_root / "flowpre__candidate_2__init_temporal_processed_v1__v1__v1"
            )
            projection_cache_dir.mkdir(parents=True, exist_ok=True)
            projection_table = pd.DataFrame(
                [
                    {"type": cls, "latent_name": latent, "semantic_feature": semantic, "weight_raw_std": 1.0, "weight_norm": weight}
                    for cls in (0, 1, 2)
                    for latent, weights in {
                        "z_0": [0.5, 0.3, 0.2],
                        "z_1": [0.1, 0.7, 0.2],
                        "z_2": [0.2, 0.2, 0.6],
                    }.items()
                    for semantic, weight in zip(["chem_a", "chem_b", "chem_c"], weights)
                ]
            )
            projection_table.to_csv(projection_cache_dir / "latent_to_semantic_per_class.csv", index=False)
            (projection_cache_dir / "projection_manifest.json").write_text(
                json.dumps(
                    {
                        "projection_contract_id": "f7_flowpre_projection_cache_v1",
                        "source_id": "flowpre__candidate_2__init_temporal_processed_v1__v1",
                        "projection_table_path": str(projection_cache_dir / "latent_to_semantic_per_class.csv"),
                        "projection_cache_path": str(projection_cache_dir),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            dataset_manifest_path = tmpdir_path / "dataset_manifest.json"
            dataset_manifest_path.write_text(
                json.dumps(
                    _dataset_manifest_payload(
                        x_transform="flowpre_candidate_2",
                        y_transform="standard",
                        upstream_model_manifests=[str(fake_promotion_manifest_path)],
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
            run_dir = tmpdir_path / "mlp_flowpre_holdout_run"
            snapshots_dir = run_dir / "snapshots"
            run_dir.mkdir(parents=True, exist_ok=True)
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            with (
                patch(
                    "training.train_mlp.setup_training_logs_and_dirs",
                    return_value=(run_dir, "mlp_flowpre_holdout_run", None, snapshots_dir),
                ),
                patch(
                    "evaluation.mlp_interpretability.load_flowpre_decoder_runtime",
                    return_value={
                        "promotion_manifest_path": str(fake_promotion_manifest_path),
                        "promotion_manifest": {"source_id": "flowpre__candidate_2__init_temporal_processed_v1__v1"},
                        "run_manifest_path": str(tmpdir_path / "unused_run_manifest.json"),
                        "run_manifest": {},
                        "decoder_model": _DummyFlowPreModel(),
                        "device": torch.device("cpu"),
                        "semantic_feature_names": ["chem_a", "chem_b", "chem_c"],
                    },
                ),
            ):
                train_mlp_model(
                    train_df.copy(),
                    val_df.copy(),
                    "type",
                    cXy_test=test_df.copy(),
                    allow_test_holdout=True,
                    seed=4567,
                    config_filename=str(cfg_path),
                    base_name="mlp_flowpre_holdout_run",
                    device="cpu",
                    verbose=False,
                    evaluation_context={
                        "dataset_name": "mlp__x-candidate2__y-standard__syn-none",
                        "dataset_manifest_path": str(dataset_manifest_path),
                        "split_id": "init_temporal_processed_v1",
                        "contract_id": "f7_contract_v1",
                        "analysis_contracts": get_f7_analysis_contract_bundle_with_paths(),
                        "seed_set_id": "f7_seed_panel_v1",
                        "base_config_id": "f7_mlp_interpretability_test_v1",
                        "objective_metric_id": "raw_real.macro.rrmse",
                        "dataset_level_axes": {
                            "x_transform": "flowpre_candidate_2",
                            "y_transform": "standard",
                            "synthetic_policy": "none",
                        },
                        "y_transform": "standard",
                        "y_scaler": scaler,
                        "target_scaler_artifact": "fake_y_scaler.pkl",
                        "flowpre_projection_cache_root": str(projection_cache_root),
                    },
                )
            summary = json.loads((run_dir / "interpretability_summary.json").read_text(encoding="utf-8"))
            feature_global = pd.read_csv(run_dir / "feature_influence_global.csv")
            latent_per_class = pd.read_csv(run_dir / "latent_feature_influence_per_class.csv")
            self.assertEqual(sorted(summary["available_splits"]), ["test", "val"])
            self.assertEqual(sorted(feature_global["split"].unique().tolist()), ["test", "val"])
            self.assertEqual(sorted(latent_per_class["split"].unique().tolist()), ["test", "val"])
            self.assertFalse(feature_global["feature_name"].astype(str).str.startswith("ilr_").any())


if __name__ == "__main__":
    unittest.main()
