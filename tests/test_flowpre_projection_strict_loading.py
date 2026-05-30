from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.flowpre_projection import (
    _resolve_flowpre_run_config_path,
    _resolve_flowpre_run_weights_path,
)


class TestFlowPreProjectionStrictLoading(unittest.TestCase):
    def test_prefers_explicit_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "flowpre_run"
            run_dir.mkdir()
            external_cfg = root / "external_config.yaml"
            external_cfg.write_text("model: {}\n", encoding="utf-8")
            external_pt = root / "external_weights.pt"
            external_pt.write_bytes(b"pt")
            run_manifest = {
                "config_path": str(external_cfg),
                "model_path": str(external_pt),
            }
            self.assertEqual(
                _resolve_flowpre_run_config_path(run_dir, run_manifest, "flowpre_run_v1"),
                external_cfg,
            )
            self.assertEqual(
                _resolve_flowpre_run_weights_path(run_dir, run_manifest, "flowpre_run_v1"),
                external_pt,
            )

    def test_uses_exact_run_artifacts_without_directory_scan_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "flowpre_run"
            run_dir.mkdir()
            exact_cfg = run_dir / "flowpre_run_v1.yaml"
            exact_cfg.write_text("model: {}\n", encoding="utf-8")
            exact_pt = run_dir / "flowpre_run_v1.pt"
            exact_pt.write_bytes(b"pt")
            (run_dir / "other_config.yaml").write_text("model: {bad: true}\n", encoding="utf-8")
            (run_dir / "other_weights.pt").write_bytes(b"other")
            run_manifest: dict[str, str] = {}
            self.assertEqual(
                _resolve_flowpre_run_config_path(run_dir, run_manifest, "flowpre_run_v1"),
                exact_cfg,
            )
            self.assertEqual(
                _resolve_flowpre_run_weights_path(run_dir, run_manifest, "flowpre_run_v1"),
                exact_pt,
            )

    def test_raises_if_exact_artifacts_missing_instead_of_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "flowpre_run"
            run_dir.mkdir()
            (run_dir / "other_config.yaml").write_text("model: {bad: true}\n", encoding="utf-8")
            (run_dir / "other_weights.pt").write_bytes(b"other")
            run_manifest = json.loads("{}")
            with self.assertRaises(FileNotFoundError):
                _resolve_flowpre_run_config_path(run_dir, run_manifest, "flowpre_run_v1")
            with self.assertRaises(FileNotFoundError):
                _resolve_flowpre_run_weights_path(run_dir, run_manifest, "flowpre_run_v1")


if __name__ == "__main__":
    unittest.main()
