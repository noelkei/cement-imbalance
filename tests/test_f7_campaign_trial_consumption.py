from __future__ import annotations

import tempfile
import unittest

from evaluation.f7_campaign_spec import materialize_f7_campaign_spec
from evaluation.f7_campaign_trial_consumption import (
    select_structured_trial_sample,
    validate_trial_consumption_sample,
)


class TestF7CampaignTrialConsumption(unittest.TestCase):
    def test_select_structured_trial_sample_returns_10_with_family_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = materialize_f7_campaign_spec(output_root=tmpdir)
            sample = select_structured_trial_sample(bundle.trials, sample_size=10)
            self.assertEqual(len(sample), 10)
            mlp = sum(1 for row in sample if row["model_family"] == "mlp")
            xgb = sum(1 for row in sample if row["model_family"] == "xgboost")
            self.assertEqual(mlp, 6)
            self.assertEqual(xgb, 4)

    def test_validate_trial_consumption_sample_all_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = materialize_f7_campaign_spec(output_root=tmpdir)
            validation = validate_trial_consumption_sample(
                trial_inventory_path=bundle.output_paths["trial_inventory_path"],
                output_root=tmpdir,
                sample_size=10,
            )
            self.assertEqual(validation.summary["ok_count"], 10)
            self.assertEqual(validation.summary["failed_count"], 0)


if __name__ == "__main__":
    unittest.main()
