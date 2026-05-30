from __future__ import annotations

import unittest

import pandas as pd

from evaluation.f7_storage_footprint import project_storage_bytes, summarize_storage_rows


class TestF7StorageFootprint(unittest.TestCase):
    def test_summarize_storage_rows_groups_sizes(self) -> None:
        frame = pd.DataFrame(
            [
                {"campaign_id": "c1", "model_family": "mlp", "flowpre_usage": True, "run_size_bytes": 1000},
                {"campaign_id": "c1", "model_family": "mlp", "flowpre_usage": True, "run_size_bytes": 3000},
                {"campaign_id": "c2", "model_family": "xgboost", "flowpre_usage": False, "run_size_bytes": 2000},
            ]
        )
        rows = summarize_storage_rows(frame, ["model_family"])
        self.assertEqual(len(rows), 2)
        mlp_row = next(row for row in rows if row["model_family"] == "mlp")
        self.assertEqual(mlp_row["run_count"], 2)
        self.assertEqual(mlp_row["total_size"]["bytes"], 4000)
        self.assertEqual(mlp_row["mean_size"]["bytes"], 2000)

    def test_project_storage_bytes_uses_group_means(self) -> None:
        observed = pd.DataFrame(
            [
                {"model_family": "mlp", "flowpre_usage": True, "run_size_bytes": 100},
                {"model_family": "mlp", "flowpre_usage": True, "run_size_bytes": 300},
                {"model_family": "mlp", "flowpre_usage": False, "run_size_bytes": 200},
                {"model_family": "xgboost", "flowpre_usage": False, "run_size_bytes": 50},
            ]
        )
        expected = pd.DataFrame(
            [
                {"model_family": "mlp", "flowpre_usage": True},
                {"model_family": "mlp", "flowpre_usage": True},
                {"model_family": "mlp", "flowpre_usage": False},
                {"model_family": "xgboost", "flowpre_usage": False},
            ]
        )
        report = project_storage_bytes(observed, expected)
        self.assertEqual(report["expected_trial_count"], 4)
        self.assertEqual(report["projected_total_size"]["bytes"], 650)
        self.assertEqual(report["fallbacks"], [])

    def test_project_storage_bytes_falls_back_to_family_mean(self) -> None:
        observed = pd.DataFrame(
            [
                {"model_family": "mlp", "flowpre_usage": True, "run_size_bytes": 100},
                {"model_family": "mlp", "flowpre_usage": True, "run_size_bytes": 300},
            ]
        )
        expected = pd.DataFrame(
            [
                {"model_family": "mlp", "flowpre_usage": False},
                {"model_family": "mlp", "flowpre_usage": False},
            ]
        )
        report = project_storage_bytes(observed, expected)
        self.assertEqual(report["projected_total_size"]["bytes"], 400)
        self.assertEqual(len(report["fallbacks"]), 1)
        self.assertEqual(report["fallbacks"][0]["fallback_kind"], "family_mean")


if __name__ == "__main__":
    unittest.main()
