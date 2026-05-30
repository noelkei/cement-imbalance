from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import yaml

from data import sets as sets_module
from data.utils import ROOT_PATH, dump_json, path_relative_to_root


F7_DATASET_MATERIALIZATION_VERSION = "f7_dataset_materialization_v1"
F7_DATASET_MATERIALIZATION_REPORT_ROOT = Path(ROOT_PATH) / "outputs" / "reports" / "f7_dataset_materialization"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml_dict(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict YAML at {path}")
    return payload


def _read_inventory_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "model_family",
        "dataset_id",
        "x_base",
        "y_transform",
        "synthetic_policy",
        "x_transform",
    }
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Inventory CSV is missing required columns: {missing}")
    return df


def _git_revision_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_PATH,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--short"],
                cwd=ROOT_PATH,
                text=True,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _resolve_official_flowpre_promotion_manifest(tag: str) -> Path:
    finalist_path = Path(ROOT_PATH) / "config" / "finalists" / f"official_flowpre_{tag}.yaml"
    finalist = _load_yaml_dict(finalist_path)
    local_refs = dict(finalist.get("local_artifact_refs") or {})
    promotion_ref = dict(local_refs.get("promotion_manifest") or {})
    path = promotion_ref.get("path")
    if not path:
        raise FileNotFoundError(f"Missing promotion manifest reference in {finalist_path}")
    return sets_module._resolve_repo_path(path)


def _resolve_flowgen_winner_promotion_manifest(*, variant: str) -> Path:
    if variant == "flowgen_official":
        finalist_path = Path(ROOT_PATH) / "config" / "finalists" / "official_flowgen_winner.yaml"
    elif variant == "flowgen_train_only":
        finalist_path = Path(ROOT_PATH) / "config" / "finalists" / "trainonly_flowgen_winner.yaml"
    else:
        raise ValueError(f"Unsupported flowgen variant: {variant}")

    finalist = _load_yaml_dict(finalist_path)
    selected_run_id = str(finalist.get("selected_run_id") or "")
    local_refs = dict(finalist.get("local_artifact_refs") or {})
    promotion_manifest_ref = dict(local_refs.get("promotion_manifest") or {})
    promotion_manifest_path = promotion_manifest_ref.get("path")
    if promotion_manifest_path:
        resolved = sets_module._resolve_repo_path(promotion_manifest_path)
        if resolved.exists():
            return resolved

    run_manifest_ref = dict(local_refs.get("run_manifest") or {})
    run_manifest_path_value = run_manifest_ref.get("path")
    run_manifest_path = (
        sets_module._resolve_repo_path(run_manifest_path_value)
        if run_manifest_path_value
        else None
    )
    run_dir = run_manifest_path.parent if run_manifest_path is not None else None

    selection_manifest_ref = dict(local_refs.get("selection_manifest") or {})
    selection_manifest_path_value = selection_manifest_ref.get("path")
    selection_manifest_path = (
        sets_module._resolve_repo_path(selection_manifest_path_value)
        if selection_manifest_path_value
        else None
    )

    rationale_ref = dict(local_refs.get("rationale_md") or {})
    rationale_path_value = rationale_ref.get("path")
    rationale_path = (
        sets_module._resolve_repo_path(rationale_path_value)
        if rationale_path_value
        else None
    )

    candidates = []
    if selected_run_id:
        if run_dir is not None:
            candidates.append(run_dir / f"{selected_run_id}_promotion_manifest.json")
        if selection_manifest_path is not None:
            candidates.append(selection_manifest_path.parent / selected_run_id / f"{selected_run_id}_promotion_manifest.json")
        if rationale_path is not None:
            candidates.append(rationale_path.parent / selected_run_id / f"{selected_run_id}_promotion_manifest.json")
    if run_dir is not None:
        candidates.append(run_dir / "promotion_manifest.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if selected_run_id:
        outputs_models_root = Path(ROOT_PATH) / "outputs" / "models"
        fallback_matches = sorted(outputs_models_root.rglob(f"{selected_run_id}_promotion_manifest.json"))
        if fallback_matches:
            return fallback_matches[0]
    raise FileNotFoundError(f"Could not resolve FlowGen promotion manifest for {variant} from {finalist_path}")


def _manifest_artifact_hashes(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return sets_module._artifact_hashes_from_manifest(manifest)


def load_f7_materialization_inventory(
    *,
    inventory_config_path: str | Path = Path(ROOT_PATH) / "config" / "f7_dataset_inventory_v1.yaml",
) -> tuple[dict[str, Any], pd.DataFrame]:
    inventory_cfg = _load_yaml_dict(inventory_config_path)
    csv_path = Path(ROOT_PATH) / str(inventory_cfg["inventory"]["inventory_csv_path"])
    inventory_df = _read_inventory_csv(csv_path)
    return inventory_cfg, inventory_df


def validate_f7_materialization_prerequisites(
    *,
    inventory_config_path: str | Path = Path(ROOT_PATH) / "config" / "f7_dataset_inventory_v1.yaml",
) -> dict[str, Any]:
    inventory_cfg, inventory_df = load_f7_materialization_inventory(
        inventory_config_path=inventory_config_path
    )
    raw_manifest = sets_module.official_raw_bundle_manifest_path(
        split_id=str(inventory_cfg["inventory"]["split_id"])
    )
    preflight = {
        "inventory_config_path": path_relative_to_root(inventory_config_path),
        "inventory_csv_path": str(inventory_cfg["inventory"]["inventory_csv_path"]),
        "inventory_rows": int(len(inventory_df)),
        "python_executable": sys.executable,
        "checks": {
            "raw_bundle_manifest_exists": raw_manifest.exists(),
            "f7_cap_policy_exists": (Path(ROOT_PATH) / "config" / "f7_synthetic_cap_policy_v1.yaml").exists(),
            "f7_guardrail_policy_exists": (Path(ROOT_PATH) / "config" / "f7_synthetic_guardrails_v1.yaml").exists(),
            "flowpre_candidate_1_promotion_exists": _resolve_official_flowpre_promotion_manifest("candidate_1").exists(),
            "flowpre_candidate_2_promotion_exists": _resolve_official_flowpre_promotion_manifest("candidate_2").exists(),
            "flowgen_official_promotion_exists": _resolve_flowgen_winner_promotion_manifest(variant="flowgen_official").exists(),
            "flowgen_trainonly_promotion_exists": _resolve_flowgen_winner_promotion_manifest(variant="flowgen_train_only").exists(),
            "numpy_importable": importlib.util.find_spec("numpy") is not None,
            "pandas_importable": importlib.util.find_spec("pandas") is not None,
            "yaml_importable": importlib.util.find_spec("yaml") is not None,
            "sklearn_importable": importlib.util.find_spec("sklearn") is not None,
            "torch_importable": importlib.util.find_spec("torch") is not None,
            "nflows_importable": importlib.util.find_spec("nflows") is not None,
        },
    }
    preflight["all_ok"] = all(preflight["checks"].values())
    return preflight


def resolve_f7_materialization_batches(
    inventory_df: pd.DataFrame,
) -> dict[str, list[str]]:
    def _ids(model_family: str, synthetic_policy: str) -> list[str]:
        mask = (inventory_df["model_family"] == model_family) & (inventory_df["synthetic_policy"] == synthetic_policy)
        return sorted(inventory_df.loc[mask, "dataset_id"].astype(str).tolist())

    return {
        "mlp_base": _ids("mlp", "none"),
        "mlp_kmeans": _ids("mlp", "kmeans_smote"),
        "mlp_flowgen_official": _ids("mlp", "flowgen_official"),
        "mlp_flowgen_trainonly": _ids("mlp", "flowgen_train_only"),
        "xgb": sorted(inventory_df.loc[inventory_df["model_family"] == "xgboost", "dataset_id"].astype(str).tolist()),
    }


def _inventory_row_by_dataset_id(inventory_df: pd.DataFrame, dataset_id: str) -> pd.Series:
    matches = inventory_df.loc[inventory_df["dataset_id"] == dataset_id]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one inventory row for dataset_id={dataset_id}")
    return matches.iloc[0]


def _canonical_mlp_base_manifest_path(*, x_base: str, y_transform: str, split_id: str) -> Path:
    if x_base in {"candidate_1", "candidate_2"}:
        return sets_module._official_flowpre_dirs(x_base, y_transform, split_id=split_id)["meta"] / "manifest.json"
    storage_name = f"df_scaled_x{x_base}_y{y_transform}"
    return sets_module._official_scaled_bundle_dir(storage_name, split_id=split_id) / "meta" / "manifest.json"


def _materialize_mlp_base_bundle(
    row: pd.Series,
    *,
    split_id: str,
    force: bool,
    verbose: bool,
) -> Path:
    x_base = str(row["x_base"])
    y_transform = str(row["y_transform"])
    if x_base in {"candidate_1", "candidate_2"}:
        promotion_manifest = _resolve_official_flowpre_promotion_manifest(x_base)
        sets_module.materialize_official_flowpre_sets(
            promoted_upstreams={x_base: promotion_manifest},
            y_scalers=[y_transform],
            split_id=split_id,
            force=force,
            device="cpu",
            verbose=verbose,
        )
    else:
        sets_module.load_or_create_scaled_sets(
            split_id=split_id,
            x_scaler_type=x_base,
            y_scaler_type=y_transform,
            force=force,
            verbose=verbose,
        )
    return _canonical_mlp_base_manifest_path(x_base=x_base, y_transform=y_transform, split_id=split_id)


def _materialize_xgb_base_bundle(
    *,
    split_id: str,
    force: bool,
    verbose: bool,
) -> Path:
    base_dir, _ = sets_module.materialize_f7_xgb_raw_base_set(
        split_id=split_id,
        force=force,
        verbose=verbose,
    )
    return base_dir / "meta" / "manifest.json"


def _resolve_base_dataset_id(row: pd.Series) -> str:
    if str(row["model_family"]) == "mlp":
        return f"mlp__x-{str(row['dataset_id']).split('__x-')[1].split('__syn-')[0]}__syn-none"
    return "xgb__x-raw-base-v1__y-raw__syn-none"


def _resolve_mlp_base_dataset_id(row: pd.Series) -> str:
    x_slug = str(row["dataset_id"]).split("__x-")[1].split("__y-")[0]
    y_slug = str(row["dataset_id"]).split("__y-")[1].split("__syn-")[0]
    return f"mlp__x-{x_slug}__y-{y_slug}__syn-none"


def _artifact_hash_row(manifest: Mapping[str, Any]) -> dict[str, Any]:
    hashes = _manifest_artifact_hashes(manifest)
    artifacts = hashes.get("artifacts", {})
    return {
        "hash_train_X": dict(artifacts.get("X") or {}).get("train"),
        "hash_train_y": dict(artifacts.get("y") or {}).get("train"),
        "hash_train_removed": dict(artifacts.get("removed") or {}).get("train"),
        "artifact_hashes": hashes,
    }


def _finalize_materialized_manifest(
    *,
    manifest_path: str | Path,
    inventory_row: pd.Series,
    batch_id: str,
    shared_pool_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = sets_module._resolve_repo_path(manifest_path)
    manifest = sets_module._load_json_dict(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    artifact_hash_row = _artifact_hash_row(manifest)
    manifest.update(
        {
            "inventory_dataset_id": str(inventory_row["dataset_id"]),
            "materialization_batch_id": batch_id,
            "inventory_version": "f7_dataset_inventory_v1",
            "contract_version": manifest.get("contract_id"),
            "cap_policy_id": "f7_synthetic_cap_policy_v1",
            "guardrail_policy_id": manifest.get("guardrail_policy_id"),
            "artifact_hashes": artifact_hash_row["artifact_hashes"],
            "created_at_utc": manifest.get("created_at_utc") or _utc_now_iso(),
            "code_revision": _git_revision_info(),
            "materializer_version": F7_DATASET_MATERIALIZATION_VERSION,
        }
    )
    if shared_pool_manifest_path is not None:
        manifest["shared_pool_manifest_path"] = path_relative_to_root(shared_pool_manifest_path)

    dump_json(manifest, manifest_path)

    counts_by_split = dict(manifest.get("counts_by_split") or {})
    counts_by_class = dict(manifest.get("counts_by_class") or {})
    return {
        "inventory_dataset_id": str(inventory_row["dataset_id"]),
        "dataset_role": str(manifest.get("dataset_role")),
        "model_family_scope": str(inventory_row["model_family"]),
        "synthetic_policy": str(inventory_row["synthetic_policy"]),
        "status": "ok",
        "manifest_path": path_relative_to_root(manifest_path),
        "shared_pool_manifest_path": None if shared_pool_manifest_path is None else path_relative_to_root(shared_pool_manifest_path),
        "source_bundle_manifest_path": manifest.get("source_bundle_manifest_path"),
        "row_counts_train": counts_by_split.get("train"),
        "row_counts_val": counts_by_split.get("val"),
        "row_counts_test": counts_by_split.get("test"),
        "class_counts_train": json.dumps(counts_by_class.get("train", {}), ensure_ascii=True, sort_keys=True),
        "elapsed_seconds": None,
        "was_reused": False,
        "hash_train_X": artifact_hash_row["hash_train_X"],
        "hash_train_y": artifact_hash_row["hash_train_y"],
        "hash_train_removed": artifact_hash_row["hash_train_removed"],
    }


def materialize_f7_inventory_phase(
    *,
    phase: str,
    inventory_df: pd.DataFrame,
    batch_id: str,
    force: bool = False,
    strict: bool = False,
    verbose: bool = True,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    batches = resolve_f7_materialization_batches(inventory_df)
    split_id = "init_temporal_processed_v1"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    target_dataset_ids = list(batches.get(phase, []))
    if phase == "xgb":
        target_dataset_ids = sorted(target_dataset_ids)
    if dataset_id is not None:
        target_dataset_ids = [did for did in target_dataset_ids if did == dataset_id]

    flowgen_pool_manifest_by_variant: dict[str, Path] = {}
    if phase == "flowgen_official_pool":
        consumer_ids = batches["mlp_flowgen_official"] + [
            did for did in batches["xgb"] if "flowgen-official" in did
        ]
        base_dir, manifest = sets_module.materialize_f7_flowgen_synthetic_pool(
            flowgen_promotion_manifest_path=_resolve_flowgen_winner_promotion_manifest(variant="flowgen_official"),
            synthetic_policy_variant="flowgen_official",
            consumer_dataset_ids=consumer_ids,
            force=force,
            device="cpu",
            verbose=verbose,
        )
        flowgen_pool_manifest_by_variant["flowgen_official"] = base_dir / "meta" / "pool_manifest.json"
        rows.append(
            {
                "inventory_dataset_id": "__shared_pool__flowgen_official",
                "dataset_role": "flowgen_shared_synthetic_pool",
                "model_family_scope": "shared",
                "synthetic_policy": "flowgen_official",
                "status": "ok",
                "manifest_path": path_relative_to_root(base_dir / "meta" / "pool_manifest.json"),
                "shared_pool_manifest_path": path_relative_to_root(base_dir / "meta" / "pool_manifest.json"),
                "source_bundle_manifest_path": manifest.get("source_raw_bundle_manifest"),
                "row_counts_train": None,
                "row_counts_val": None,
                "row_counts_test": None,
                "class_counts_train": json.dumps(manifest.get("accepted_counts_by_class", {}), ensure_ascii=True, sort_keys=True),
                "elapsed_seconds": None,
                "was_reused": False,
                "hash_train_X": None,
                "hash_train_y": None,
                "hash_train_removed": None,
            }
        )
        return {"rows": rows, "failures": failures}

    if phase == "flowgen_trainonly_pool":
        consumer_ids = batches["mlp_flowgen_trainonly"] + [
            did for did in batches["xgb"] if "flowgen-train-only" in did
        ]
        base_dir, manifest = sets_module.materialize_f7_flowgen_synthetic_pool(
            flowgen_promotion_manifest_path=_resolve_flowgen_winner_promotion_manifest(variant="flowgen_train_only"),
            synthetic_policy_variant="flowgen_train_only",
            consumer_dataset_ids=consumer_ids,
            force=force,
            device="cpu",
            verbose=verbose,
        )
        flowgen_pool_manifest_by_variant["flowgen_train_only"] = base_dir / "meta" / "pool_manifest.json"
        rows.append(
            {
                "inventory_dataset_id": "__shared_pool__flowgen_train_only",
                "dataset_role": "flowgen_shared_synthetic_pool",
                "model_family_scope": "shared",
                "synthetic_policy": "flowgen_train_only",
                "status": "ok",
                "manifest_path": path_relative_to_root(base_dir / "meta" / "pool_manifest.json"),
                "shared_pool_manifest_path": path_relative_to_root(base_dir / "meta" / "pool_manifest.json"),
                "source_bundle_manifest_path": manifest.get("source_raw_bundle_manifest"),
                "row_counts_train": None,
                "row_counts_val": None,
                "row_counts_test": None,
                "class_counts_train": json.dumps(manifest.get("accepted_counts_by_class", {}), ensure_ascii=True, sort_keys=True),
                "elapsed_seconds": None,
                "was_reused": False,
                "hash_train_X": None,
                "hash_train_y": None,
                "hash_train_removed": None,
            }
        )
        return {"rows": rows, "failures": failures}

    for target_dataset_id in target_dataset_ids:
        started = time.perf_counter()
        try:
            row = _inventory_row_by_dataset_id(inventory_df, target_dataset_id)
            shared_pool_manifest_path = None
            if str(row["model_family"]) == "mlp" and str(row["synthetic_policy"]) == "none":
                manifest_path = _materialize_mlp_base_bundle(row, split_id=split_id, force=force, verbose=verbose)
            elif str(row["model_family"]) == "mlp" and str(row["synthetic_policy"]) == "kmeans_smote":
                base_manifest_path = _materialize_mlp_base_bundle(
                    _inventory_row_by_dataset_id(inventory_df, _resolve_mlp_base_dataset_id(row)),
                    split_id=split_id,
                    force=False,
                    verbose=verbose,
                )
                base_dir, _ = sets_module.materialize_kmeans_smote_joint_set(
                    base_manifest_path,
                    force=force,
                    verbose=verbose,
                )
                manifest_path = base_dir / "meta" / "manifest.json"
            elif str(row["model_family"]) == "mlp" and str(row["synthetic_policy"]) in {"flowgen_official", "flowgen_train_only"}:
                base_manifest_path = _materialize_mlp_base_bundle(
                    _inventory_row_by_dataset_id(inventory_df, _resolve_mlp_base_dataset_id(row)),
                    split_id=split_id,
                    force=False,
                    verbose=verbose,
                )
                variant = str(row["synthetic_policy"])
                pool_manifest = flowgen_pool_manifest_by_variant.get(variant)
                if pool_manifest is None or not Path(pool_manifest).exists():
                    pool_base_dir, _ = sets_module.materialize_f7_flowgen_synthetic_pool(
                        flowgen_promotion_manifest_path=_resolve_flowgen_winner_promotion_manifest(variant=variant),
                        synthetic_policy_variant=variant,
                        force=False,
                        device="cpu",
                        verbose=verbose,
                    )
                    pool_manifest = pool_base_dir / "meta" / "pool_manifest.json"
                    flowgen_pool_manifest_by_variant[variant] = pool_manifest
                base_dir, _ = sets_module.materialize_f7_flowgen_augmented_set_from_pool(
                    base_manifest_path,
                    flowgen_pool_manifest_path=pool_manifest,
                    synthetic_policy_variant=variant,
                    force=force,
                    device="cpu",
                    verbose=verbose,
                )
                manifest_path = base_dir / "meta" / "manifest.json"
                shared_pool_manifest_path = pool_manifest
            elif str(row["model_family"]) == "xgboost" and str(row["synthetic_policy"]) == "none":
                manifest_path = _materialize_xgb_base_bundle(split_id=split_id, force=force, verbose=verbose)
            elif str(row["model_family"]) == "xgboost" and str(row["synthetic_policy"]) == "kmeans_smote":
                base_manifest_path = _materialize_xgb_base_bundle(split_id=split_id, force=False, verbose=verbose)
                base_dir, _ = sets_module.materialize_kmeans_smote_joint_set(
                    base_manifest_path,
                    force=force,
                    verbose=verbose,
                )
                manifest_path = base_dir / "meta" / "manifest.json"
            elif str(row["model_family"]) == "xgboost" and str(row["synthetic_policy"]) in {"flowgen_official", "flowgen_train_only"}:
                base_manifest_path = _materialize_xgb_base_bundle(split_id=split_id, force=False, verbose=verbose)
                variant = str(row["synthetic_policy"])
                pool_manifest = sets_module.materialize_f7_flowgen_synthetic_pool(
                    flowgen_promotion_manifest_path=_resolve_flowgen_winner_promotion_manifest(variant=variant),
                    synthetic_policy_variant=variant,
                    force=False,
                    device="cpu",
                    verbose=verbose,
                )[0] / "meta" / "pool_manifest.json"
                base_dir, _ = sets_module.materialize_f7_flowgen_augmented_set_from_pool(
                    base_manifest_path,
                    flowgen_pool_manifest_path=pool_manifest,
                    synthetic_policy_variant=variant,
                    force=force,
                    device="cpu",
                    verbose=verbose,
                )
                manifest_path = base_dir / "meta" / "manifest.json"
                shared_pool_manifest_path = pool_manifest
            else:
                raise ValueError(f"Unsupported phase/materialization combination for {target_dataset_id}")

            summary_row = _finalize_materialized_manifest(
                manifest_path=manifest_path,
                inventory_row=row,
                batch_id=batch_id,
                shared_pool_manifest_path=shared_pool_manifest_path,
            )
            summary_row["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            rows.append(summary_row)
        except Exception as exc:
            failures.append(
                {
                    "phase": phase,
                    "inventory_dataset_id": target_dataset_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            if strict:
                raise
    return {"rows": rows, "failures": failures}


def write_f7_materialization_batch_report(
    *,
    batch_id: str,
    phase_results: Mapping[str, Mapping[str, Any]],
) -> Path:
    report_dir = F7_DATASET_MATERIALIZATION_REPORT_ROOT / batch_id
    report_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    phase_summary: list[dict[str, Any]] = []
    artifact_hash_rows: list[dict[str, Any]] = []

    for phase, payload in phase_results.items():
        phase_rows = list(payload.get("rows") or [])
        phase_failures = list(payload.get("failures") or [])
        inventory_rows.extend(phase_rows)
        failures.extend(phase_failures)
        phase_summary.append(
            {
                "phase": phase,
                "ok_count": int(sum(1 for row in phase_rows if row.get("status") == "ok")),
                "failure_count": int(len(phase_failures)),
            }
        )
        for row in phase_rows:
            artifact_hash_rows.append(
                {
                    "inventory_dataset_id": row.get("inventory_dataset_id"),
                    "manifest_path": row.get("manifest_path"),
                    "hash_train_X": row.get("hash_train_X"),
                    "hash_train_y": row.get("hash_train_y"),
                    "hash_train_removed": row.get("hash_train_removed"),
                }
            )

    pd.DataFrame(inventory_rows).to_csv(report_dir / "materialized_inventory.csv", index=False)
    pd.DataFrame(phase_summary).to_csv(report_dir / "phase_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(report_dir / "failures.csv", index=False)
    pd.DataFrame(artifact_hash_rows).to_csv(report_dir / "artifact_hash_index.csv", index=False)

    dump_json(
        {
            "batch_id": batch_id,
            "materializer_version": F7_DATASET_MATERIALIZATION_VERSION,
            "created_at_utc": _utc_now_iso(),
            "phase_names": list(phase_results.keys()),
            "ok_rows": int(sum(len(payload.get("rows") or []) for payload in phase_results.values())),
            "failure_rows": int(sum(len(payload.get("failures") or []) for payload in phase_results.values())),
        },
        report_dir / "materialization_batch_manifest.json",
    )
    return report_dir
