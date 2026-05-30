from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

from training.train_xgboost import train_xgboost_model


STUDY_ID = "f7_xgb_interpretability_validation_v1"
REPORT_COLUMNS = [
    "job_id",
    "dataset_key",
    "dataset_name",
    "synthetic_policy",
    "run_mode",
    "allow_test_holdout",
    "seed",
    "status",
    "campaign_valid",
    "campaign_valid_interpretability",
    "campaign_valid_f7",
    "feature_space_kind_primary",
    "type_feature_policy",
    "available_splits",
    "training_runtime_s",
    "interpretability_runtime_s_total",
    "interpretability_runtime_s_shap",
    "interpretability_runtime_s_perturbation",
    "total_runtime_s",
    "best_iteration",
    "device",
    "prediction_sidecar_present",
    "run_dir",
    "config_path",
    "error",
]

DATASET_SPECS = {
    "xgb_none": {
        "dataset_manifest_path": (
            "data/sets/official/init_temporal_processed_v1/xgboost/"
            "official_raw_xgb_base_v1/meta/manifest.json"
        ),
    },
    "xgb_kmeans_smote": {
        "dataset_manifest_path": (
            "data/sets/official/init_temporal_processed_v1/augmented_scaled/"
            "official_raw_xgb_base_v1__syn-kmeans_smote__kmeans_smote_joint_base_v1__seed42__v1/meta/manifest.json"
        ),
    },
    "xgb_flowgen_official": {
        "dataset_manifest_path": (
            "data/sets/official/init_temporal_processed_v1/augmented_scaled/"
            "official_raw_xgb_base_v1__syn-flowgen_official__flowgen__winner__init_temporal_processed_v1__v1/meta/manifest.json"
        ),
    },
    "xgb_flowgen_train_only": {
        "dataset_manifest_path": (
            "data/sets/official/init_temporal_processed_v1/augmented_scaled/"
            "official_raw_xgb_base_v1__syn-flowgen_train_only__flowgen_trainonly__winner__init_temporal_processed_v1__v1/meta/manifest.json"
        ),
    },
}


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    repo_candidate = ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate
    raise FileNotFoundError(f"Path not found: {path}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML mapping at {path}")
    return payload


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = _repo_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON mapping at {resolved}")
    return payload


def _read_bundle_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(_repo_path(path))
    if "post_cleaning_index" not in df.columns:
        raise ValueError(f"Missing post_cleaning_index in {path}")
    df["post_cleaning_index"] = df["post_cleaning_index"].astype(int)
    return df.sort_values("post_cleaning_index").reset_index(drop=True)


def _load_bundle_frames_from_manifest(
    manifest_payload: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifacts = dict(manifest_payload.get("artifacts") or {})
    x_artifacts = dict(artifacts.get("X") or {})
    y_artifacts = dict(artifacts.get("y") or {})
    return (
        _read_bundle_csv(x_artifacts["train"]),
        _read_bundle_csv(x_artifacts["val"]),
        _read_bundle_csv(x_artifacts["test"]),
        _read_bundle_csv(y_artifacts["train"]),
        _read_bundle_csv(y_artifacts["val"]),
        _read_bundle_csv(y_artifacts["test"]),
    )


def _load_seed_values() -> list[int]:
    payload = _load_yaml(ROOT / "config" / "f7_seed_panel_v1.yaml")
    seed_panel = payload.get("seed_panel") or payload.get("f7_seed_panel") or payload
    if not isinstance(seed_panel, dict):
        raise ValueError("Could not resolve seed panel mapping from config/f7_seed_panel_v1.yaml")
    if isinstance(seed_panel.get("values"), list) and seed_panel["values"]:
        return [int(item) for item in seed_panel["values"]]
    seeds = seed_panel.get("seeds")
    if isinstance(seeds, list) and seeds:
        resolved: list[int] = []
        for item in seeds:
            if isinstance(item, dict) and "value" in item:
                resolved.append(int(item["value"]))
            else:
                resolved.append(int(item))
        return resolved
    raise ValueError("Could not resolve seed values from config/f7_seed_panel_v1.yaml")


def _build_validation_jobs(seed_values: list[int]) -> list[dict[str, Any]]:
    if len(seed_values) < 3:
        raise ValueError("Need at least 3 seeds in the canonical panel to build the validation matrix.")
    s0, s1, s2 = seed_values[:3]
    jobs: list[dict[str, Any]] = []
    for dataset_key in (
        "xgb_none",
        "xgb_kmeans_smote",
        "xgb_flowgen_official",
        "xgb_flowgen_train_only",
    ):
        jobs.append(
            {
                "dataset_key": dataset_key,
                "run_mode": "holdout_run",
                "allow_test_holdout": True,
                "seed": s0,
            }
        )
        jobs.append(
            {
                "dataset_key": dataset_key,
                "run_mode": "holdout_run",
                "allow_test_holdout": True,
                "seed": s1,
            }
        )
    jobs.append(
        {
            "dataset_key": "xgb_none",
            "run_mode": "holdout_run",
            "allow_test_holdout": True,
            "seed": s2,
        }
    )
    jobs.append(
        {
            "dataset_key": "xgb_kmeans_smote",
            "run_mode": "holdout_run",
            "allow_test_holdout": True,
            "seed": s2,
        }
    )
    return jobs


def _build_trial_config(
    *,
    base_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("contract", {})
    cfg["contract"]["xgb_base_config_id"] = "f7_xgb_interpretability_validation_v1"
    cfg["contract"]["objective_metric_id"] = "raw_real.macro.rrmse"
    cfg["contract"]["allow_test_holdout_default"] = True

    cfg.setdefault("training", {})
    cfg["training"]["random_state"] = int(seed)
    cfg["training"]["save_results"] = True
    cfg["training"]["save_model"] = True
    cfg["seed"] = int(seed)
    return cfg


def _generated_config_dir() -> Path:
    return ROOT / "config" / "generated" / STUDY_ID


def _report_dir() -> Path:
    return ROOT / "outputs" / "reports" / STUDY_ID


def _existing_run_dirs() -> set[Path]:
    root = ROOT / "outputs" / "models" / "xgboost"
    if not root.exists():
        return set()
    return {p for p in root.iterdir() if p.is_dir()}


def _infer_new_run_dir(before: set[Path], after: set[Path], base_name: str) -> Path | None:
    candidates = sorted(
        (p for p in after - before if p.name.startswith(base_name + "_v")),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _existing_rows(report_path: Path) -> dict[str, dict[str, Any]]:
    if not report_path.exists():
        return {}
    df = pd.read_csv(report_path)
    if df.empty or "job_id" not in df.columns:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        rows[str(row["job_id"])] = row.to_dict()
    return rows


def _run_markdown_report(report_dir: Path, rows_df: pd.DataFrame) -> None:
    ok_df = rows_df[rows_df["status"] == "ok"].copy() if not rows_df.empty else pd.DataFrame()
    lines: list[str] = []
    lines.append(f"# {STUDY_ID}")
    lines.append("")
    lines.append("## Matrix")
    lines.append("")
    lines.append("- 10 runs estructuradas")
    lines.append("- todas con `holdout_run` y `test=true`")
    lines.append("- xgb_none: 3 seeds")
    lines.append("- xgb_kmeans_smote: 3 seeds")
    lines.append("- xgb_flowgen_official: 2 seeds")
    lines.append("- xgb_flowgen_train_only: 2 seeds")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if rows_df.empty:
        lines.append("No rows yet.")
    else:
        lines.append(f"- total runs: {len(rows_df)}")
        lines.append(f"- ok: {int((rows_df['status'] == 'ok').sum())}")
        lines.append(f"- failed: {int((rows_df['status'] != 'ok').sum())}")
        if not ok_df.empty:
            lines.append(f"- mean training_runtime_s: {round(float(ok_df['training_runtime_s'].mean()), 3)}")
            lines.append(
                f"- mean interpretability_runtime_s_total: {round(float(ok_df['interpretability_runtime_s_total'].mean()), 3)}"
            )
            lines.append(
                f"- mean interpretability_runtime_s_shap: {round(float(ok_df['interpretability_runtime_s_shap'].mean()), 3)}"
            )
            lines.append(
                f"- mean interpretability_runtime_s_perturbation: {round(float(ok_df['interpretability_runtime_s_perturbation'].mean()), 3)}"
            )
            lines.append(f"- mean total_runtime_s: {round(float(ok_df['total_runtime_s'].mean()), 3)}")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    if rows_df.empty:
        lines.append("No run table yet.")
    else:
        lines.append("```text")
        lines.append(rows_df.to_string(index=False))
        lines.append("```")
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_run_outputs(*, run_dir: Path) -> dict[str, Any]:
    manifest = _load_json(run_dir / "run_manifest.json")
    summary = _load_json(run_dir / "xgb_interpretability_summary.json")
    shap_global = pd.read_csv(run_dir / "xgb_shap_feature_influence_global.csv")
    perturb_global = pd.read_csv(run_dir / "xgb_perturbation_feature_influence_global.csv")
    prediction_sidecar_present = (run_dir / "predictions_eval_raw.csv.gz").exists()

    expected_splits = ["test", "val"]
    actual_splits = list(summary.get("available_splits", []))
    if sorted(actual_splits) != expected_splits:
        raise ValueError(f"Unexpected interpretability splits: expected {expected_splits}, got {actual_splits}")
    if sorted(shap_global["split"].unique().tolist()) != expected_splits:
        raise ValueError("xgb_shap_feature_influence_global.csv split surface does not match expected run_mode.")
    if sorted(perturb_global["split"].unique().tolist()) != expected_splits:
        raise ValueError("xgb_perturbation_feature_influence_global.csv split surface does not match expected run_mode.")

    shap_cols = [
        "mean_abs_shap",
        "mean_signed_shap",
        "sum_abs_shap",
        "std_abs_shap",
        "median_abs_shap",
        "p90_abs_shap",
        "p95_abs_shap",
        "stderr_abs_shap",
        "share_abs_importance",
    ]
    perturb_cols = [
        "mean_abs_delta_pred_raw",
        "mean_signed_delta_pred_raw",
        "sum_abs_delta_pred_raw",
        "std_abs_delta_pred_raw",
        "median_abs_delta_pred_raw",
        "p90_abs_delta_pred_raw",
        "p95_abs_delta_pred_raw",
        "stderr_abs_delta_pred_raw",
        "share_abs_importance",
    ]
    missing_shap_cols = [col for col in shap_cols if col not in shap_global.columns]
    missing_perturb_cols = [col for col in perturb_cols if col not in perturb_global.columns]
    if missing_shap_cols:
        raise ValueError(f"xgb_shap_feature_influence_global.csv missing columns: {missing_shap_cols}")
    if missing_perturb_cols:
        raise ValueError(f"xgb_perturbation_feature_influence_global.csv missing columns: {missing_perturb_cols}")
    if shap_global[shap_cols].isna().any().any():
        raise ValueError("xgb_shap_feature_influence_global.csv contains NaN values.")
    if perturb_global[perturb_cols].isna().any().any():
        raise ValueError("xgb_perturbation_feature_influence_global.csv contains NaN values.")
    for split_name, split_df in shap_global.groupby("split", sort=False):
        share_sum = float(split_df["share_abs_importance"].sum())
        if not abs(share_sum - 1.0) <= 1e-5:
            raise ValueError(f"SHAP share_abs_importance does not sum to 1 for split {split_name}: {share_sum}")
    for split_name, split_df in perturb_global.groupby("split", sort=False):
        share_sum = float(split_df["share_abs_importance"].sum())
        if not (abs(share_sum - 1.0) <= 1e-5 or abs(share_sum - 0.0) <= 1e-8):
            raise ValueError(
                f"Perturbation share_abs_importance has unexpected sum for split {split_name}: {share_sum}"
            )

    runtime_by_layer = dict(summary.get("interpretability_runtime_s_by_layer") or {})
    training_summary = dict(manifest.get("training_summary") or {})
    return {
        "campaign_valid": bool(manifest.get("campaign_valid")),
        "campaign_valid_interpretability": bool(manifest.get("campaign_valid_interpretability")),
        "campaign_valid_f7": bool(manifest.get("campaign_valid_f7")),
        "feature_space_kind_primary": summary.get("feature_space_kind_primary"),
        "type_feature_policy": summary.get("type_feature_policy"),
        "available_splits": ",".join(actual_splits),
        "training_runtime_s": training_summary.get("runtime_s"),
        "interpretability_runtime_s_total": summary.get("interpretability_runtime_s_total"),
        "interpretability_runtime_s_shap": runtime_by_layer.get("shap"),
        "interpretability_runtime_s_perturbation": runtime_by_layer.get("perturbation"),
        "best_iteration": training_summary.get("best_iteration"),
        "device": training_summary.get("device"),
        "prediction_sidecar_present": prediction_sidecar_present,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a 10-run empirical validation matrix for F7 XGBoost interpretability with test enabled."
    )
    parser.add_argument("--run-label", default=None, help="Optional suffix for report/config separation.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N jobs after filtering.")
    parser.add_argument(
        "--only-dataset",
        choices=sorted(DATASET_SPECS.keys()),
        default=None,
        help="Restrict to one dataset family.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run jobs even if they are already marked ok.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned matrix and exit.")
    args = parser.parse_args()

    seed_values = _load_seed_values()
    jobs = _build_validation_jobs(seed_values)
    if args.only_dataset is not None:
        jobs = [job for job in jobs if job["dataset_key"] == args.only_dataset]
    if args.limit is not None:
        jobs = jobs[: int(args.limit)]

    for job in jobs:
        job["job_id"] = (
            f"{job['dataset_key']}__{job['run_mode']}__seed{job['seed']}"
            + (f"__{args.run_label}" if args.run_label else "")
        )

    if args.dry_run:
        print(pd.DataFrame(jobs).to_string(index=False))
        return

    base_cfg = _load_yaml(ROOT / "config" / "f7_xgb_base_v1.yaml")
    report_dir = _report_dir() if not args.run_label else (_report_dir() / args.run_label)
    report_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = _generated_config_dir() if not args.run_label else (_generated_config_dir() / args.run_label)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "run_summary.csv"
    rows_map = _existing_rows(report_path)

    for job in jobs:
        existing = rows_map.get(job["job_id"])
        if existing is not None and str(existing.get("status")) == "ok" and not args.force:
            continue

        dataset_spec = DATASET_SPECS[job["dataset_key"]]
        dataset_manifest_path = _repo_path(dataset_spec["dataset_manifest_path"])
        dataset_manifest = _load_json(dataset_manifest_path)
        X_train, X_val, X_test, y_train, y_val, y_test = _load_bundle_frames_from_manifest(dataset_manifest)

        cfg_payload = _build_trial_config(
            base_cfg=base_cfg,
            seed=int(job["seed"]),
        )
        cfg_path = cfg_dir / f"{job['job_id']}.yaml"
        _dump_yaml(cfg_path, cfg_payload)

        dataset_axes = dict(dataset_manifest.get("dataset_level_axes") or {})
        run_mode = str(job["run_mode"])
        base_name = f"{STUDY_ID}__{job['job_id']}"
        eval_ctx = {
            "contract_id": "f7_contract_v1",
            "comparison_group_id": f"{STUDY_ID}__{job['dataset_key']}__{run_mode}",
            "seed_set_id": "f7_seed_panel_v1",
            "xgb_base_config_id": "f7_xgb_interpretability_validation_v1",
            "objective_metric_id": "raw_real.macro.rrmse",
            "dataset_name": dataset_manifest.get("dataset_name"),
            "dataset_manifest_path": str(dataset_manifest_path),
            "split_id": dataset_manifest.get("split_id", "init_temporal_processed_v1"),
            "dataset_level_axes": dataset_axes,
            "upstream_variant_fingerprint": f"{STUDY_ID}::{job['job_id']}",
        }

        row = {
            "job_id": job["job_id"],
            "dataset_key": job["dataset_key"],
            "dataset_name": dataset_manifest.get("dataset_name"),
            "synthetic_policy": dataset_axes.get("synthetic_policy"),
            "run_mode": run_mode,
            "allow_test_holdout": True,
            "seed": int(job["seed"]),
            "status": "failed",
            "campaign_valid": None,
            "campaign_valid_interpretability": None,
            "campaign_valid_f7": None,
            "feature_space_kind_primary": None,
            "type_feature_policy": None,
            "available_splits": None,
            "training_runtime_s": None,
            "interpretability_runtime_s_total": None,
            "interpretability_runtime_s_shap": None,
            "interpretability_runtime_s_perturbation": None,
            "total_runtime_s": None,
            "best_iteration": None,
            "device": None,
            "prediction_sidecar_present": None,
            "run_dir": None,
            "config_path": str(cfg_path),
            "error": None,
        }

        started = time.time()
        before_dirs = _existing_run_dirs()
        try:
            train_xgboost_model(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test=X_test,
                y_test=y_test,
                allow_test_holdout=True,
                seed=int(job["seed"]),
                config_filename=str(cfg_path),
                config=cfg_payload,
                base_name=base_name,
                verbose=False,
                evaluation_context=eval_ctx,
            )
            after_dirs = _existing_run_dirs()
            run_dir = _infer_new_run_dir(before_dirs, after_dirs, base_name)
            if run_dir is None:
                raise RuntimeError("Could not infer the created XGBoost run directory.")
            validation = _validate_run_outputs(run_dir=run_dir)
            row.update(validation)
            row["run_dir"] = str(run_dir)
            row["status"] = "ok" if bool(validation["campaign_valid_f7"]) else "failed_validation"
        except Exception as exc:  # noqa: BLE001
            row["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        finally:
            row["total_runtime_s"] = round(time.time() - started, 3)
            rows_map[job["job_id"]] = row
            rows_df = pd.DataFrame(rows_map.values(), columns=REPORT_COLUMNS).sort_values(
                by=["dataset_key", "seed", "job_id"],
                kind="mergesort",
            )
            rows_df.to_csv(report_path, index=False)
            _run_markdown_report(report_dir, rows_df)

    rows_df = pd.DataFrame(rows_map.values(), columns=REPORT_COLUMNS).sort_values(
        by=["dataset_key", "seed", "job_id"],
        kind="mergesort",
    )
    rows_df.to_csv(report_path, index=False)
    _run_markdown_report(report_dir, rows_df)
    print(f"Validation report written to: {report_path}")
    print(f"Markdown summary written to: {report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
