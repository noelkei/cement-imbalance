from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from data.utils import ROOT_PATH
from evaluation.f7_launch_readiness import (
    collect_environment_freeze,
    generate_f7_launch_readiness_report,
    validate_planned_campaign_chain,
)


class TestF7LaunchReadiness(unittest.TestCase):
    def test_validation_chain_ok_for_block13_validation_specs(self) -> None:
        root = Path(ROOT_PATH)
        payload = validate_planned_campaign_chain(
            primary_spec_path=root / "config" / "f7_campaign_block13_validation_primary_v1.yaml",
            extension_spec_paths=[
                root / "config" / "f7_campaign_block13_validation_extension_v1.yaml",
                root / "config" / "f7_campaign_block13_validation_extension2_v1.yaml",
            ],
        )
        self.assertTrue(payload["ok"], payload["issues"])
        self.assertEqual(payload["campaign_count"], 3)
        self.assertEqual(payload["observed_total_seed_count"], 4)

    def test_environment_freeze_contains_core_fields(self) -> None:
        payload = collect_environment_freeze()
        self.assertTrue(payload["python_version"])
        self.assertIn("platform", payload)
        self.assertIn("git_commit", payload)

    def test_report_generation_emits_files(self) -> None:
        root = Path(ROOT_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "readiness.json"
            md_path = Path(tmpdir) / "readiness.md"
            payload = generate_f7_launch_readiness_report(
                primary_spec_path=root / "config" / "f7_campaign_block13_validation_primary_v1.yaml",
                extension_spec_paths=[
                    root / "config" / "f7_campaign_block13_validation_extension_v1.yaml",
                    root / "config" / "f7_campaign_block13_validation_extension2_v1.yaml",
                ],
                json_output_path=json_path,
                markdown_output_path=md_path,
            )
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            self.assertIn(payload["go_no_go"], {"go", "no_go"})


if __name__ == "__main__":
    unittest.main()
