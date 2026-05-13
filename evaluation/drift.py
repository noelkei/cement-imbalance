from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT_PATH = Path(__file__).resolve().parents[1]


OFFICIAL_SPLIT_ID = "init_temporal_processed_v1"


def resolve_split_bundle_dir(split_id: str = OFFICIAL_SPLIT_ID, *, root: Path | None = None) -> Path:
    return (root or Path(ROOT_PATH)) / "data" / "splits" / "official" / split_id


def load_official_drift_bundle(split_id: str = OFFICIAL_SPLIT_ID, *, root: Path | None = None) -> dict:
    bundle_dir = resolve_split_bundle_dir(split_id, root=root)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    drift_numeric_path = bundle_dir / "drift_numeric.csv"
    drift_target_path = bundle_dir / "drift_target_init.csv"
    drift_type_path = bundle_dir / "drift_type.csv"

    return {
        "bundle_dir": bundle_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "drift_numeric": pd.read_csv(drift_numeric_path),
        "drift_target_init": pd.read_csv(drift_target_path),
        "drift_type": pd.read_csv(drift_type_path),
    }
