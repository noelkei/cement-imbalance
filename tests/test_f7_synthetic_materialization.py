from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data.dataset_contract import (
    build_canonical_dataset_name,
    classify_supported_dataset_space,
)
from data.f7_dataset_materialization import (
    load_f7_materialization_inventory,
    resolve_f7_materialization_batches,
)
from data.f7_synthetic_cap_policy import (
    F7SyntheticCapPolicy,
    resolve_f7_synthetic_targets_from_real_counts,
)
from data.f7_synthetic_guardrails import (
    F7CleaningAuditArtifacts,
    F7SyntheticAcceptanceEngine,
    F7SyntheticGuardrailPolicy,
)
from data.kmeans_smote_joint import (
    KMeansSmoteJointConfig,
    generate_kmeans_smote_joint_samples,
)
from data import sets as sets_module


def _make_real_train_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    targets = []
    idx = 0
    for cls, count in [(0, 10), (1, 6), (2, 4)]:
        for offset in range(count):
            rows.append(
                {
                    "post_cleaning_index": idx,
                    "type": cls,
                    "h2o": 1.0 + cls + (0.1 * offset),
                    "a": -1.0 - (0.1 * offset),
                    "ilr_chem_1": -0.2 + (0.01 * offset),
                }
            )
            targets.append({"post_cleaning_index": idx, "init": 10.0 + cls + (0.5 * offset)})
            idx += 1
    return pd.DataFrame(rows), pd.DataFrame(targets)


class TestF7SyntheticCapAndGuardrails(unittest.TestCase):
    def test_target_resolution_normal_case(self) -> None:
        summary = resolve_f7_synthetic_targets_from_real_counts(
            real_counts={0: 10, 1: 6, 2: 4},
            policy=F7SyntheticCapPolicy(),
        )
        self.assertEqual(summary["targets_by_class"], {"0": 0, "1": 3, "2": 2})

    def test_tied_majority_yields_zero_synth(self) -> None:
        summary = resolve_f7_synthetic_targets_from_real_counts(
            real_counts={0: 10, 1: 10, 2: 4},
            policy=F7SyntheticCapPolicy(),
        )
        self.assertEqual(summary["targets_by_class"], {"0": 0, "1": 0, "2": 2})

    def test_guardrails_reject_negative_real_column_but_exempt_a_and_ilr(self) -> None:
        X_real, y_real = _make_real_train_frames()
        engine = F7SyntheticAcceptanceEngine(
            real_X_train=X_real,
            real_y_train=y_real,
            cap_policy=F7SyntheticCapPolicy(),
            guardrail_policy=F7SyntheticGuardrailPolicy(),
        )

        X_batch = pd.DataFrame(
            [
                {
                    "post_cleaning_index": -1,
                    "type": 1,
                    "h2o": 1.5,
                    "a": -3.0,
                    "ilr_chem_1": -0.7,
                },
                {
                    "post_cleaning_index": -2,
                    "type": 1,
                    "h2o": -0.1,
                    "a": -3.0,
                    "ilr_chem_1": -0.7,
                },
            ]
        )
        y_batch = pd.DataFrame(
            [
                {"post_cleaning_index": -1, "init": 12.0},
                {"post_cleaning_index": -2, "init": 12.5},
            ]
        )

        accepted_x, _, _, _, summary = engine.accept_batch(
            X_batch_materialized=X_batch,
            y_batch_materialized=y_batch,
            X_batch_domain=X_batch,
            y_batch_domain=y_batch,
        )
        self.assertEqual(len(accepted_x), 1)
        self.assertEqual(summary["reject_counts_by_reason"], {"negative_modeled_raw:h2o": 1})

    def test_learned_cleaning_audit_is_soft_only(self) -> None:
        X_real = pd.DataFrame(
            [
                {"post_cleaning_index": 0, "type": "type_a", "h2o": 1.0},
                {"post_cleaning_index": 1, "type": "type_a", "h2o": 1.1},
                {"post_cleaning_index": 2, "type": "type_a", "h2o": 1.15},
                {"post_cleaning_index": 3, "type": "type_b", "h2o": 1.2},
                {"post_cleaning_index": 4, "type": "type_b", "h2o": 1.25},
            ]
        )
        y_real = pd.DataFrame(
            [
                {"post_cleaning_index": 0, "init": 10.0},
                {"post_cleaning_index": 1, "init": 10.5},
                {"post_cleaning_index": 2, "init": 10.8},
                {"post_cleaning_index": 3, "init": 11.0},
                {"post_cleaning_index": 4, "init": 11.2},
            ]
        )
        artifacts = F7CleaningAuditArtifacts(
            cleaning_policy_id="dummy",
            type_col="type",
            univariate_rules={
                "type_col": "type",
                "n_bins": 2,
                "only_threshold": 0.99,
                "columns_to_check": ["h2o"],
                "per_type": {
                    "type_b": {
                        "h2o": {
                            "bin_edges": [0.0, 1.0, 2.0],
                            "keep_bins": [0, 0],
                            "train_rows": 1,
                            "train_kept_rows": 0,
                        }
                    }
                },
            },
            iforest_models_by_type=None,
            iforest_columns_to_check=[],
        )
        engine = F7SyntheticAcceptanceEngine(
            real_X_train=X_real,
            real_y_train=y_real,
            cap_policy=F7SyntheticCapPolicy(),
            guardrail_policy=F7SyntheticGuardrailPolicy(),
            cleaning_audit_artifacts=artifacts,
            real_X_audit_view=X_real,
        )

        X_batch = pd.DataFrame(
            [{"post_cleaning_index": -1, "type": "type_b", "h2o": 1.4}]
        )
        y_batch = pd.DataFrame([{"post_cleaning_index": -1, "init": 10.7}])
        accepted_x, _, _, _, summary = engine.accept_batch(
            X_batch_materialized=X_batch,
            y_batch_materialized=y_batch,
            X_batch_domain=X_batch,
            y_batch_domain=y_batch,
        )
        self.assertEqual(len(accepted_x), 1)
        self.assertEqual(
            summary["soft_audit_summary"]["counts"].get("univariate_flagged_rows"),
            1,
        )

    def test_raw_raw_xgb_space_is_supported_for_kmeans(self) -> None:
        support = classify_supported_dataset_space(
            x_transform="raw",
            y_transform="raw",
            synthetic_policy="kmeans_smote",
        )
        self.assertEqual(support["support_status"], "materialized_now")
        self.assertEqual(support["dataset_storage_family"], "official_xgb_augmented_raw")


class TestKMeansSmoteF7Flow(unittest.TestCase):
    def test_kmeans_retries_until_target_with_validator(self) -> None:
        X_train = pd.DataFrame(
            [
                {"post_cleaning_index": i, "type": 0, "f1": 0.1 * i, "f2": 0.2 * i}
                for i in range(6)
            ]
            + [
                {"post_cleaning_index": 100 + i, "type": 1, "f1": 1.0 + (0.1 * i), "f2": 2.0 + (0.1 * i)}
                for i in range(4)
            ]
        )
        y_train = pd.DataFrame(
            [{"post_cleaning_index": int(idx), "init": float(idx) / 10.0} for idx in X_train["post_cleaning_index"]]
        )
        cfg = KMeansSmoteJointConfig(
            synthetic_policy_config_id="test_cfg",
            synthetic_seed=42,
            condition_col="type",
            target_mode="f7_policy",
            metric_space_mode="standard",
            cluster_k_mode="fixed",
            cluster_k_fixed=1,
            neighbor_k_mode="fixed",
            neighbor_k_value=1,
            min_cluster_size=2,
        )
        attempts = {"count": 0}

        def validator(X_batch, y_batch, *, class_value, class_label, attempt_idx):
            attempts["count"] += 1
            if int(class_value) != 1 or attempt_idx > 1:
                return X_batch, y_batch, {
                    "accepted_count": len(X_batch),
                    "rejected_count": 0,
                    "reject_counts_by_reason": {},
                    "soft_audit_summary": {"counts": {}},
                }
            accepted_x = X_batch.iloc[0:1].copy()
            accepted_y = y_batch.iloc[0:1].copy()
            return accepted_x, accepted_y, {
                "accepted_count": 1,
                "rejected_count": int(len(X_batch) - 1),
                "reject_counts_by_reason": {"forced_retry": int(len(X_batch) - 1)},
                "soft_audit_summary": {"counts": {}},
            }

        X_synth, y_synth, report = generate_kmeans_smote_joint_samples(
            X_train=X_train,
            y_train=y_train,
            config=cfg,
            explicit_target_count_by_class={0: 6, 1: 6},
            candidate_validator=validator,
            max_attempt_batches_per_class=3,
        )
        self.assertEqual(len(X_synth), 2)
        self.assertEqual(len(y_synth), 2)
        self.assertGreaterEqual(attempts["count"], 2)
        self.assertEqual(report["class_reports"]["type_1"]["generated_count"], 2)
        self.assertEqual(report["target_mode"], "f7_policy")


class TestFlowGenF7Materialization(unittest.TestCase):
    def test_flowgen_materialization_emits_f7_manifest(self) -> None:
        X_raw_train = pd.DataFrame(
            [
                {"post_cleaning_index": 0, "type": 0, "h2o": 1.0},
                {"post_cleaning_index": 1, "type": 0, "h2o": 1.2},
                {"post_cleaning_index": 2, "type": 0, "h2o": 1.4},
                {"post_cleaning_index": 3, "type": 0, "h2o": 1.6},
                {"post_cleaning_index": 4, "type": 1, "h2o": 2.0},
                {"post_cleaning_index": 5, "type": 1, "h2o": 2.2},
            ]
        )
        y_raw_train = pd.DataFrame(
            [
                {"post_cleaning_index": idx, "init": value}
                for idx, value in zip(X_raw_train["post_cleaning_index"], [10, 11, 12, 13, 20, 21])
            ]
        )
        X_raw_val = X_raw_train.iloc[0:2].copy()
        X_raw_test = X_raw_train.iloc[2:4].copy()
        y_raw_val = y_raw_train.iloc[0:2].copy()
        y_raw_test = y_raw_train.iloc[2:4].copy()
        r_train = pd.DataFrame({"post_cleaning_index": X_raw_train["post_cleaning_index"], "split": "train"})
        r_val = pd.DataFrame({"post_cleaning_index": X_raw_val["post_cleaning_index"], "split": "val"})
        r_test = pd.DataFrame({"post_cleaning_index": X_raw_test["post_cleaning_index"], "split": "test"})

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw_bundle"
            source_root = tmp_path / "source_bundle"
            out_root = tmp_path / "official_sets"
            for root in (raw_root, source_root):
                for sub in ("X", "y", "removed", "scalers", "meta"):
                    (root / sub).mkdir(parents=True, exist_ok=True)

            def _save_split(root: Path, prefix: str, X_df: pd.DataFrame, y_df: pd.DataFrame, r_df: pd.DataFrame) -> None:
                X_df.to_csv(root / "X" / f"{prefix}_X_train.csv", index=False)
                y_df.to_csv(root / "y" / f"{prefix}_y_train.csv", index=False)
                r_df.to_csv(root / "removed" / f"{prefix}_r_train.csv", index=False)
                X_raw_val.to_csv(root / "X" / f"{prefix}_X_val.csv", index=False)
                X_raw_test.to_csv(root / "X" / f"{prefix}_X_test.csv", index=False)
                y_raw_val.to_csv(root / "y" / f"{prefix}_y_val.csv", index=False)
                y_raw_test.to_csv(root / "y" / f"{prefix}_y_test.csv", index=False)
                r_val.to_csv(root / "removed" / f"{prefix}_r_val.csv", index=False)
                r_test.to_csv(root / "removed" / f"{prefix}_r_test.csv", index=False)

            _save_split(raw_root, "raw_bundle", X_raw_train, y_raw_train, r_train)

            x_scaler = StandardScaler().fit(X_raw_train[["h2o"]])
            y_scaler = MinMaxScaler().fit(y_raw_train[["init"]])
            joblib.dump(x_scaler, source_root / "scalers" / "x.pkl")
            joblib.dump(y_scaler, source_root / "scalers" / "y.pkl")

            X_source_train = X_raw_train.copy()
            X_source_train["h2o"] = x_scaler.transform(X_source_train[["h2o"]])
            y_source_train = y_raw_train.copy()
            y_source_train["init"] = y_scaler.transform(y_source_train[["init"]])
            X_source_val = X_raw_val.copy()
            X_source_val["h2o"] = x_scaler.transform(X_source_val[["h2o"]])
            X_source_test = X_raw_test.copy()
            X_source_test["h2o"] = x_scaler.transform(X_source_test[["h2o"]])
            y_source_val = y_raw_val.copy()
            y_source_val["init"] = y_scaler.transform(y_source_val[["init"]])
            y_source_test = y_raw_test.copy()
            y_source_test["init"] = y_scaler.transform(y_source_test[["init"]])

            X_source_train.to_csv(source_root / "X" / "source_bundle_X_train.csv", index=False)
            X_source_val.to_csv(source_root / "X" / "source_bundle_X_val.csv", index=False)
            X_source_test.to_csv(source_root / "X" / "source_bundle_X_test.csv", index=False)
            y_source_train.to_csv(source_root / "y" / "source_bundle_y_train.csv", index=False)
            y_source_val.to_csv(source_root / "y" / "source_bundle_y_val.csv", index=False)
            y_source_test.to_csv(source_root / "y" / "source_bundle_y_test.csv", index=False)
            r_train.to_csv(source_root / "removed" / "source_bundle_r_train.csv", index=False)
            r_val.to_csv(source_root / "removed" / "source_bundle_r_val.csv", index=False)
            r_test.to_csv(source_root / "removed" / "source_bundle_r_test.csv", index=False)

            raw_manifest_path = raw_root / "meta" / "manifest.json"
            raw_manifest = {
                "dataset_name": "raw_bundle",
                "cleaning_policy_id": "trainfit_overlap_cap1pct_holdoutflag_v1",
                "artifacts": {
                    "X": {
                        "train": str(raw_root / "X" / "raw_bundle_X_train.csv"),
                        "val": str(raw_root / "X" / "raw_bundle_X_val.csv"),
                        "test": str(raw_root / "X" / "raw_bundle_X_test.csv"),
                    },
                    "y": {
                        "train": str(raw_root / "y" / "raw_bundle_y_train.csv"),
                        "val": str(raw_root / "y" / "raw_bundle_y_val.csv"),
                        "test": str(raw_root / "y" / "raw_bundle_y_test.csv"),
                    },
                    "removed": {
                        "train": str(raw_root / "removed" / "raw_bundle_r_train.csv"),
                        "val": str(raw_root / "removed" / "raw_bundle_r_val.csv"),
                        "test": str(raw_root / "removed" / "raw_bundle_r_test.csv"),
                    },
                },
            }
            raw_manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")

            source_manifest_path = source_root / "meta" / "manifest.json"
            source_manifest = {
                "policy_status": "canonical",
                "dataset_role": "derived_modeling_bundle",
                "dataset_name": "dataset__x-standard__y-minmax__syn-none__v1",
                "split_id": "init_temporal_processed_v1",
                "cleaning_policy_id": "trainfit_overlap_cap1pct_holdoutflag_v1",
                "source_dataset_manifest": str(raw_manifest_path),
                "dataset_level_axes": {
                    "x_transform": "standard",
                    "y_transform": "minmax",
                    "synthetic_policy": "none",
                },
                "artifacts": {
                    "storage_name": "source_bundle",
                    "X": {
                        "train": str(source_root / "X" / "source_bundle_X_train.csv"),
                        "val": str(source_root / "X" / "source_bundle_X_val.csv"),
                        "test": str(source_root / "X" / "source_bundle_X_test.csv"),
                    },
                    "y": {
                        "train": str(source_root / "y" / "source_bundle_y_train.csv"),
                        "val": str(source_root / "y" / "source_bundle_y_val.csv"),
                        "test": str(source_root / "y" / "source_bundle_y_test.csv"),
                    },
                    "removed": {
                        "train": str(source_root / "removed" / "source_bundle_r_train.csv"),
                        "val": str(source_root / "removed" / "source_bundle_r_val.csv"),
                        "test": str(source_root / "removed" / "source_bundle_r_test.csv"),
                    },
                },
                "scaler_artifacts": {
                    "X": str(source_root / "scalers" / "x.pkl"),
                    "y": str(source_root / "scalers" / "y.pkl"),
                },
                "x_scaler": {"fit_cols": ["h2o"]},
                "y_scaler": {"fit_cols": ["init"]},
                "counts_by_split": {"train": 6, "val": 2, "test": 2},
                "counts_by_class": {
                    "train": {"0": 4, "1": 2},
                    "val": {"0": 2},
                    "test": {"0": 2},
                },
                "upstream_model_manifests": [],
            }
            source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")

            synth_X_raw = pd.DataFrame(
                [{"post_cleaning_index": 999, "type": 1, "h2o": 2.4}]
            )
            synth_y_raw = pd.DataFrame([{"post_cleaning_index": 999, "init": 22.0}])

            def _mock_sample_flowgen_f7_raw(*args, **kwargs):
                return synth_X_raw.copy(), synth_y_raw.copy(), {"1": {"target_synth_rows": 1}}

            with patch.object(sets_module, "_sample_flowgen_f7_raw", side_effect=_mock_sample_flowgen_f7_raw), patch.object(
                sets_module, "_OFFICIAL_SETS_ROOT", out_root
            ):
                base_dir, manifest = sets_module.materialize_f7_flowgen_augmented_set(
                    str(source_manifest_path),
                    flowgen_promotion_manifest_path=str(source_manifest_path),
                    synthetic_policy_variant="flowgen_official",
                    force=True,
                    device="cpu",
                    verbose=False,
                )

            self.assertTrue((base_dir / "meta" / "manifest.json").exists())
            self.assertEqual(
                manifest["dataset_level_axes"]["synthetic_policy"],
                "flowgen_official",
            )
            self.assertIn("guardrail_summary", manifest)
            self.assertTrue(
                manifest["f7_synthetic_cap_validation"]["is_valid_campaign_ready"]
            )
            self.assertEqual(
                manifest["dataset_name"],
                build_canonical_dataset_name(
                    x_transform="standard",
                    y_transform="minmax",
                    synthetic_policy="flowgen_official",
                ),
            )


class TestF7XgbBaseMaterialization(unittest.TestCase):
    def test_xgb_raw_base_materialization_emits_raw_identity_manifest(self) -> None:
        X_train, y_train = _make_real_train_frames()
        X_val = X_train.iloc[:3].copy()
        X_test = X_train.iloc[3:6].copy()
        y_val = y_train.iloc[:3].copy()
        y_test = y_train.iloc[3:6].copy()
        r_train = pd.DataFrame({"post_cleaning_index": X_train["post_cleaning_index"], "split": "train"})
        r_val = pd.DataFrame({"post_cleaning_index": X_val["post_cleaning_index"], "split": "val"})
        r_test = pd.DataFrame({"post_cleaning_index": X_test["post_cleaning_index"], "split": "test"})

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw_bundle"
            out_root = tmp_path / "official_sets"
            for sub in ("X", "y", "removed", "meta"):
                (raw_root / sub).mkdir(parents=True, exist_ok=True)

            for split_name, X_df, y_df, r_df in (
                ("train", X_train, y_train, r_train),
                ("val", X_val, y_val, r_val),
                ("test", X_test, y_test, r_test),
            ):
                X_df.to_csv(raw_root / "X" / f"raw_bundle_X_{split_name}.csv", index=False)
                y_df.to_csv(raw_root / "y" / f"raw_bundle_y_{split_name}.csv", index=False)
                r_df.to_csv(raw_root / "removed" / f"raw_bundle_r_{split_name}.csv", index=False)

            raw_manifest_path = raw_root / "meta" / "manifest.json"
            raw_manifest = {
                "dataset_name": "raw_bundle",
                "split_id": "init_temporal_processed_v1",
                "cleaning_policy_id": "trainfit_overlap_cap1pct_holdoutflag_v1",
                "artifacts": {
                    "X": {split: str(raw_root / "X" / f"raw_bundle_X_{split}.csv") for split in ("train", "val", "test")},
                    "y": {split: str(raw_root / "y" / f"raw_bundle_y_{split}.csv") for split in ("train", "val", "test")},
                    "removed": {split: str(raw_root / "removed" / f"raw_bundle_r_{split}.csv") for split in ("train", "val", "test")},
                },
                "counts_by_split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
                "counts_by_class": {
                    "train": {"0": 10, "1": 6, "2": 4},
                    "val": {"0": 3},
                    "test": {"0": 3},
                },
            }
            raw_manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")

            with patch.object(sets_module, "_OFFICIAL_SETS_ROOT", out_root):
                base_dir, manifest = sets_module.materialize_f7_xgb_raw_base_set(
                    source_raw_bundle_manifest_path=str(raw_manifest_path),
                    force=True,
                    verbose=False,
                )

            self.assertTrue((base_dir / "meta" / "manifest.json").exists())
            self.assertEqual(manifest["dataset_level_axes"]["x_transform"], "raw")
            self.assertEqual(manifest["dataset_level_axes"]["y_transform"], "raw")
            self.assertEqual(manifest["feature_policy"], "raw_numeric_plus_type_onehot")
            self.assertEqual(manifest["train_source_rows"], len(X_train))
            self.assertEqual(manifest["generated_rows_total"], 0)


class TestF7DatasetMaterializationInventory(unittest.TestCase):
    def test_inventory_batches_match_expected_counts(self) -> None:
        _, inventory_df = load_f7_materialization_inventory()
        batches = resolve_f7_materialization_batches(inventory_df)
        self.assertEqual(len(batches["mlp_base"]), 24)
        self.assertEqual(len(batches["mlp_kmeans"]), 24)
        self.assertEqual(len(batches["mlp_flowgen_official"]), 24)
        self.assertEqual(len(batches["mlp_flowgen_trainonly"]), 24)
        self.assertEqual(len(batches["xgb"]), 4)


if __name__ == "__main__":
    unittest.main()
