from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_VERSION = "finalists_registry_v1"
RAW_BUNDLE_DATASET_NAME = "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"
RAW_BUNDLE_MANIFEST = (
    "data/sets/official/init_temporal_processed_v1/raw/"
    "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1/manifest.json"
)


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    line: str
    model_family: str
    source_kind: str
    manifest_relpath: str
    semantic_role: str
    rationale_relpaths: tuple[str, ...]
    readme_relpaths: tuple[str, ...] = ()


ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        artifact_id="official_flowpre_rrmse",
        line="official",
        model_family="flowpre",
        source_kind="official_flowpre",
        manifest_relpath=(
            "outputs/models/official/flowpre_finalists/rrmse/"
            "flowprex2_rrmse_tpv1_hf256_l3_rq3_lr1e-4_mson_skoff_seed5678_v1/"
            "flowprex2_rrmse_tpv1_hf256_l3_rq3_lr1e-4_mson_skoff_seed5678_v1_promotion_manifest.json"
        ),
        semantic_role="specialized scaler/upstream for dataset derivation under the rrmse_primary lens",
        rationale_relpaths=("outputs/models/official/flowpre_finalists/rrmse/RATIONALE.md",),
        readme_relpaths=("outputs/models/official/flowpre_finalists/README.md",),
    ),
    ArtifactSpec(
        artifact_id="official_flowpre_mvn",
        line="official",
        model_family="flowpre",
        source_kind="official_flowpre",
        manifest_relpath=(
            "outputs/models/official/flowpre_finalists/mvn/"
            "flowpre_rrmse_tpv1_rq5_seed5678_v1/"
            "flowpre_rrmse_tpv1_rq5_seed5678_v1_promotion_manifest.json"
        ),
        semantic_role="specialized scaler/upstream for dataset derivation under the mvn lens",
        rationale_relpaths=("outputs/models/official/flowpre_finalists/mvn/RATIONALE.md",),
        readme_relpaths=("outputs/models/official/flowpre_finalists/README.md",),
    ),
    ArtifactSpec(
        artifact_id="official_flowpre_fair",
        line="official",
        model_family="flowpre",
        source_kind="official_flowpre",
        manifest_relpath=(
            "outputs/models/official/flowpre_finalists/fair/"
            "flowprex4_rrmse_tpv1_hf192_l3_rq6_lr1e-3_mson_skoff_seed5678_v1/"
            "flowprex4_rrmse_tpv1_hf192_l3_rq6_lr1e-3_mson_skoff_seed5678_v1_promotion_manifest.json"
        ),
        semantic_role="specialized scaler/upstream for dataset derivation under the fair lens",
        rationale_relpaths=("outputs/models/official/flowpre_finalists/fair/RATIONALE.md",),
        readme_relpaths=("outputs/models/official/flowpre_finalists/README.md",),
    ),
    ArtifactSpec(
        artifact_id="official_flowpre_candidate_1",
        line="official",
        model_family="flowpre",
        source_kind="official_flowpre",
        manifest_relpath=(
            "outputs/models/official/flowpre_finalists/candidate_1/"
            "flowprers1_rrmse_tpv1_hf192_l3_rq6_lr1e-3_mson_skoff_seed9101_v1/"
            "flowprers1_rrmse_tpv1_hf192_l3_rq6_lr1e-3_mson_skoff_seed9101_v1_promotion_manifest.json"
        ),
        semantic_role="primary official FlowGen work base with priorfit bias",
        rationale_relpaths=("outputs/models/official/flowpre_finalists/candidate_1/RATIONALE.md",),
        readme_relpaths=("outputs/models/official/flowpre_finalists/README.md",),
    ),
    ArtifactSpec(
        artifact_id="official_flowpre_candidate_2",
        line="official",
        model_family="flowpre",
        source_kind="official_flowpre",
        manifest_relpath=(
            "outputs/models/official/flowpre_finalists/candidate_2/"
            "flowprers1_rrmse_tpv1_hf256_l4_rq5_lr1e-3_mson_skoff_seed2468_v1/"
            "flowprers1_rrmse_tpv1_hf256_l4_rq5_lr1e-3_mson_skoff_seed2468_v1_promotion_manifest.json"
        ),
        semantic_role="secondary official FlowGen work base with robust/hybrid bias",
        rationale_relpaths=("outputs/models/official/flowpre_finalists/candidate_2/RATIONALE.md",),
        readme_relpaths=("outputs/models/official/flowpre_finalists/README.md",),
    ),
    ArtifactSpec(
        artifact_id="official_flowgen_winner",
        line="official",
        model_family="flowgen",
        source_kind="official_flowgen",
        manifest_relpath="outputs/models/official/flowgen_finalist/flowgen_final_selection_manifest.json",
        semantic_role="unique official FlowGen winner promoted for downstream comparison",
        rationale_relpaths=("outputs/models/official/flowgen_finalist/RATIONALE.md",),
        readme_relpaths=("outputs/models/official/flowgen_finalist/README.md",),
    ),
    ArtifactSpec(
        artifact_id="trainonly_flowpre_candidate_1",
        line="experimental_train_only",
        model_family="flowpre",
        source_kind="trainonly_flowpre",
        manifest_relpath="outputs/models/experimental/train_only/flowpre_finalists/candidate_trainonly_1/promotion_manifest.json",
        semantic_role="primary train-only FlowGen prior for local downstream comparison",
        rationale_relpaths=("outputs/models/experimental/train_only/flowpre_finalists/candidate_trainonly_1/RATIONALE.md",),
        readme_relpaths=("outputs/models/experimental/train_only/flowpre_finalists/README.md",),
    ),
    ArtifactSpec(
        artifact_id="trainonly_flowpre_candidate_2",
        line="experimental_train_only",
        model_family="flowpre",
        source_kind="trainonly_flowpre",
        manifest_relpath="outputs/models/experimental/train_only/flowpre_finalists/candidate_trainonly_2/promotion_manifest.json",
        semantic_role="backup train-only FlowGen prior for local downstream comparison",
        rationale_relpaths=("outputs/models/experimental/train_only/flowpre_finalists/candidate_trainonly_2/RATIONALE.md",),
        readme_relpaths=("outputs/models/experimental/train_only/flowpre_finalists/README.md",),
    ),
    ArtifactSpec(
        artifact_id="trainonly_flowgen_winner",
        line="experimental_train_only",
        model_family="flowgen",
        source_kind="trainonly_flowgen",
        manifest_relpath="outputs/models/experimental/train_only/flowgen_finalist/flowgen_trainonly_final_selection_manifest.json",
        semantic_role="unique local train-only FlowGen finalist for downstream comparison",
        rationale_relpaths=("outputs/models/experimental/train_only/flowgen_finalist/RATIONALE.md",),
        readme_relpaths=("outputs/models/experimental/train_only/flowgen_finalist/README.md",),
    ),
)


PATH_KEYS_TO_DROP_FROM_CONFIG = {
    "base_checkpoint",
    "base_run_manifest",
}


def _repo_path(rel_or_abs: str | Path) -> Path:
    candidate = Path(rel_or_abs)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _relpath(path: str | Path) -> str:
    p = _repo_path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(_repo_path(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON in {path}")
    return payload


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(_repo_path(path), "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict YAML in {path}")
    return payload


def _read_text(path: str | Path) -> str:
    return _repo_path(path).read_text(encoding="utf-8")


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = _repo_path(path)
    if not resolved.exists() or not resolved.is_file():
        return None
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_repo_absolute_path(value: str) -> bool:
    try:
        return Path(value).is_absolute() and str(REPO_ROOT.resolve()) in str(Path(value).resolve())
    except Exception:
        return False


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, nested in value.items():
            if key in PATH_KEYS_TO_DROP_FROM_CONFIG:
                continue
            sanitized[key] = _sanitize_value(nested)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, str):
        if _is_repo_absolute_path(value):
            return _relpath(value)
        if Path(value).is_absolute():
            return "__redacted_local_path__"
        return value
    return value


def _find_existing_file(run_dir: Path, patterns: list[str], *, strict: bool) -> str | None:
    for pattern in patterns:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return _relpath(matches[0])
    if strict:
        raise FileNotFoundError(f"Expected one of {patterns} under {run_dir}")
    return None


def _compact_markdown_summary(paths: tuple[str, ...], *, max_items: int = 8, max_chars: int = 900) -> list[str]:
    items: list[str] = []
    for relpath in paths:
        text = _read_text(relpath)
        in_code = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code or not line or line.startswith("#"):
                continue
            line = re.sub(r"`([^`]*)`", r"\1", line)
            if line.startswith("- "):
                cleaned = line[2:].strip()
            else:
                cleaned = line
            if cleaned and cleaned not in items:
                items.append(cleaned)
            if len(items) >= max_items or sum(len(x) for x in items) >= max_chars:
                return items[:max_items]
    return items[:max_items]


def _local_ref(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    rel = _relpath(path)
    return {
        "path": rel,
        "visibility": "local_only",
    }


def _clean_local_refs(refs: dict[str, dict[str, Any] | None]) -> dict[str, dict[str, Any]]:
    return {key: value for key, value in refs.items() if value is not None}


def _extract_flowpre_results_summary(results_payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("best_epoch", "total_epochs", "seed"):
        if key in results_payload:
            summary[key] = results_payload[key]
    for split in ("train", "val"):
        split_payload = results_payload.get(split)
        if not isinstance(split_payload, dict):
            continue
        split_summary: dict[str, Any] = {}
        for metric in (
            "rrmse_recon",
            "r2_recon",
            "rrmse_mean_whole",
            "rrmse_std_whole",
        ):
            if metric in split_payload:
                split_summary[metric] = split_payload[metric]
        if split_summary:
            summary[split] = split_summary
    return summary


def _extract_flowgen_training_summary(results_payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for phase in ("phase1", "finetune"):
        phase_payload = results_payload.get(phase)
        if isinstance(phase_payload, dict):
            phase_summary = {
                key: phase_payload[key]
                for key in ("enabled", "best_epoch", "total_epochs", "ramp_rebase_epoch")
                if key in phase_payload
            }
            if phase_summary:
                summary[phase] = phase_summary
    train_payload = results_payload.get("train")
    if isinstance(train_payload, dict):
        train_summary: dict[str, Any] = {}
        for metric in ("rrmse_x_recon", "rrmse_y_recon", "r2_x_recon", "r2_y_recon"):
            if metric in train_payload:
                train_summary[metric] = train_payload[metric]
        realism = train_payload.get("realism")
        if isinstance(realism, dict):
            overall = realism.get("overall")
            if isinstance(overall, dict):
                train_summary["realism_overall"] = {
                    metric: overall[metric]
                    for metric in ("ks_mean", "w1_mean", "xy_pearson_fro_rel", "xy_spearman_fro_rel", "mmd2_ratio")
                    if metric in overall
                }
        if train_summary:
            summary["train"] = train_summary
    return summary


def _build_integrity(*, manifest_path: str | None, config_path: str | None, results_path: str | None, rationale_paths: tuple[str, ...]) -> dict[str, Any]:
    integrity = {
        "manifest_sha256": _sha256_file(manifest_path),
        "config_sha256": _sha256_file(config_path),
        "results_sha256": _sha256_file(results_path),
    }
    rationale_hashes = {
        _relpath(path): _sha256_file(path)
        for path in rationale_paths
    }
    if rationale_hashes:
        integrity["rationale_sha256"] = rationale_hashes
    return integrity


def _extract_common_registry_fields(entry: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "artifact_id",
        "registry_version",
        "line",
        "model_family",
        "selection_role",
        "selection_status",
        "selection_phase",
        "semantic_role",
        "split_id",
        "cleaning_policy_id",
        "source_id",
        "selected_run_id",
        "selected_seed",
        "cfg_signature",
        "family_policy_id",
        "upstream_dependencies",
        "config_snapshot_ref",
        "results_summary_ref",
        "local_artifact_refs",
        "integrity",
    )
    return {key: entry.get(key) for key in keys}


def _load_official_flowpre(spec: ArtifactSpec, *, strict: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(spec.manifest_relpath)
    run_manifest_rel = manifest["source_run_manifest"]
    run_manifest = _read_json(run_manifest_rel)
    run_id = str(run_manifest.get("run_id"))
    run_dir = _repo_path(run_manifest_rel).parent
    config_rel = _find_existing_file(run_dir, [f"{run_id}.yaml"], strict=True)
    results_rel = _find_existing_file(run_dir, [f"{run_id}_results.yaml", "results.yaml"], strict=True)
    config_payload = _sanitize_value(_read_yaml(config_rel))
    results_payload = _read_yaml(results_rel)

    entry = {
        "artifact_id": spec.artifact_id,
        "registry_version": REGISTRY_VERSION,
        "line": spec.line,
        "model_family": spec.model_family,
        "selection_role": manifest.get("selection_role"),
        "selection_status": "closed",
        "selection_phase": manifest.get("selection_phase"),
        "semantic_role": spec.semantic_role,
        "split_id": manifest.get("split_id"),
        "cleaning_policy_id": manifest.get("cleaning_policy_id"),
        "source_id": manifest.get("source_id"),
        "selected_run_id": run_id,
        "selected_seed": manifest.get("selection_seed"),
        "cfg_signature": manifest.get("selection_cfg_id"),
        "family_policy_id": None,
        "upstream_dependencies": {
            "official_split_id": manifest.get("split_id"),
            "cleaning_policy_id": manifest.get("cleaning_policy_id"),
            "raw_bundle_dataset_name": RAW_BUNDLE_DATASET_NAME,
        },
        "config_snapshot_ref": f"config/finalists/config_snapshots/{spec.artifact_id}.yaml",
        "results_summary_ref": f"config/finalists/{spec.artifact_id}.yaml",
        "results_summary": {
            "selection_lens": manifest.get("lens_finalist"),
            "n_runs_aggregated": manifest.get("n_runs_aggregated"),
            "training_summary": _extract_flowpre_results_summary(results_payload),
        },
        "rationale_summary": _compact_markdown_summary(spec.rationale_relpaths + spec.readme_relpaths),
        "local_artifact_refs": _clean_local_refs({
            "promotion_manifest": _local_ref(spec.manifest_relpath),
            "run_manifest": _local_ref(run_manifest_rel),
            "config": _local_ref(config_rel),
            "results": _local_ref(results_rel),
            "rationale_md": _local_ref(spec.rationale_relpaths[0]),
            "readme_md": _local_ref(spec.readme_relpaths[0]) if spec.readme_relpaths else None,
        }),
        "integrity": _build_integrity(
            manifest_path=spec.manifest_relpath,
            config_path=config_rel,
            results_path=results_rel,
            rationale_paths=spec.rationale_relpaths,
        ),
        "notes": {
            "historical_support_only": bool(manifest.get("historical_support_only", False)),
            "branch_id": manifest.get("branch_id"),
            "is_flowgen_work_base": bool(manifest.get("is_flowgen_work_base", False)),
        },
    }
    return entry, config_payload


def _load_trainonly_flowpre(spec: ArtifactSpec, *, strict: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(spec.manifest_relpath)
    config_rel = manifest["artifacts"]["selected_config"]
    results_rel = manifest["artifacts"]["selected_results"]
    run_manifest_rel = manifest["artifacts"]["selected_run_manifest"]
    config_payload = _sanitize_value(_read_yaml(config_rel))

    entry = {
        "artifact_id": spec.artifact_id,
        "registry_version": REGISTRY_VERSION,
        "line": spec.line,
        "model_family": spec.model_family,
        "selection_role": manifest.get("selection_role"),
        "selection_status": "closed",
        "selection_phase": manifest.get("selection_phase"),
        "semantic_role": spec.semantic_role,
        "split_id": manifest.get("split_id"),
        "cleaning_policy_id": None,
        "source_id": manifest.get("source_id"),
        "selected_run_id": manifest.get("selected_run_id"),
        "selected_seed": manifest.get("selected_seed"),
        "cfg_signature": manifest.get("selected_cfg_signature"),
        "family_policy_id": None,
        "upstream_dependencies": {
            "official_split_id": manifest.get("split_id"),
            "monitoring_policy": manifest.get("monitoring_policy"),
            "expected_flowgen_role": manifest.get("expected_flowgen_role"),
        },
        "config_snapshot_ref": f"config/finalists/config_snapshots/{spec.artifact_id}.yaml",
        "results_summary_ref": f"config/finalists/{spec.artifact_id}.yaml",
        "results_summary": {
            "selection_basis": manifest.get("selection_basis"),
            "selected_family_rank": manifest.get("selected_family_rank"),
            "family_panel_seeds": manifest.get("family_panel_seeds"),
            "selected_run_key_metrics": manifest.get("selected_run_key_metrics"),
            "family_aggregate_metrics": manifest.get("family_aggregate_metrics"),
        },
        "rationale_summary": _compact_markdown_summary(spec.rationale_relpaths + spec.readme_relpaths),
        "local_artifact_refs": _clean_local_refs({
            "promotion_manifest": _local_ref(spec.manifest_relpath),
            "run_manifest": _local_ref(run_manifest_rel),
            "config": _local_ref(config_rel),
            "results": _local_ref(results_rel),
            "rationale_md": _local_ref(spec.rationale_relpaths[0]),
            "readme_md": _local_ref(spec.readme_relpaths[0]) if spec.readme_relpaths else None,
        }),
        "integrity": _build_integrity(
            manifest_path=spec.manifest_relpath,
            config_path=config_rel,
            results_path=results_rel,
            rationale_paths=spec.rationale_relpaths,
        ),
        "notes": {
            "selection_alias": manifest.get("selection_alias"),
            "monitoring_note": manifest.get("monitoring_note"),
            "is_flowgen_work_base": bool(manifest.get("is_flowgen_work_base", False)),
        },
    }
    return entry, config_payload


def _load_official_flowgen(spec: ArtifactSpec, *, strict: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(spec.manifest_relpath)
    run_dir = _repo_path(manifest["materialized_run_dir"])
    run_id = str(manifest["selection_run_id"])
    run_manifest_rel = _find_existing_file(run_dir, ["run_manifest.json", f"{run_id}_run_manifest.json"], strict=True)
    config_rel = _find_existing_file(run_dir, [f"{run_id}.yaml", "config.yaml"], strict=True)
    results_rel = _find_existing_file(run_dir, ["results.yaml", f"{run_id}_results.yaml"], strict=True)
    config_payload = _sanitize_value(_read_yaml(config_rel))
    results_payload = _read_yaml(results_rel)

    entry = {
        "artifact_id": spec.artifact_id,
        "registry_version": REGISTRY_VERSION,
        "line": spec.line,
        "model_family": spec.model_family,
        "selection_role": manifest.get("selection_role"),
        "selection_status": manifest.get("selection_status", "closed"),
        "selection_phase": manifest.get("selection_phase"),
        "semantic_role": spec.semantic_role,
        "split_id": manifest.get("split_id"),
        "cleaning_policy_id": manifest.get("cleaning_policy_id"),
        "source_id": manifest.get("source_id"),
        "selected_run_id": manifest.get("selection_run_id"),
        "selected_seed": manifest.get("selection_seed"),
        "cfg_signature": None,
        "family_policy_id": manifest.get("family_policy_id"),
        "upstream_dependencies": {
            "branch_id": manifest.get("branch_id"),
            "paired_flowpre_source_id": manifest.get("paired_flowpre_source_id"),
            "paired_flowpre_run_id": manifest.get("paired_flowpre_run_id"),
            "raw_bundle_dataset_name": RAW_BUNDLE_DATASET_NAME,
        },
        "config_snapshot_ref": f"config/finalists/config_snapshots/{spec.artifact_id}.yaml",
        "results_summary_ref": f"config/finalists/{spec.artifact_id}.yaml",
        "results_summary": {
            "family_policy_id": manifest.get("family_policy_id"),
            "family_seed_set": manifest.get("family_seed_set"),
            "selection_lenses": manifest.get("selection_lenses"),
            "selection_temporal_metrics": manifest.get("selection_temporal_metrics"),
            "selection_metrics": manifest.get("selection_metrics"),
            "selection_within_family_ranks": manifest.get("selection_within_family_ranks"),
            "selection_reason": manifest.get("selection_reason"),
            "training_summary": _extract_flowgen_training_summary(results_payload),
        },
        "rationale_summary": _compact_markdown_summary(spec.rationale_relpaths + spec.readme_relpaths),
        "local_artifact_refs": _clean_local_refs({
            "selection_manifest": _local_ref(spec.manifest_relpath),
            "run_manifest": _local_ref(run_manifest_rel),
            "config": _local_ref(config_rel),
            "results": _local_ref(results_rel),
            "rationale_md": _local_ref(spec.rationale_relpaths[0]),
            "readme_md": _local_ref(spec.readme_relpaths[0]) if spec.readme_relpaths else None,
        }),
        "integrity": _build_integrity(
            manifest_path=spec.manifest_relpath,
            config_path=config_rel,
            results_path=results_rel,
            rationale_paths=spec.rationale_relpaths,
        ),
        "notes": {
            "runner_up_run_id": manifest.get("runner_up_run_id"),
            "runner_up_seed": manifest.get("runner_up_seed"),
            "historical_support_only": bool(manifest.get("historical_support_only", False)),
        },
    }
    return entry, config_payload


def _load_trainonly_flowgen(spec: ArtifactSpec, *, strict: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(spec.manifest_relpath)
    run_dir = _repo_path(manifest["materialized_run_dir"])
    run_id = str(manifest["selection_run_id"])
    run_manifest_rel = _find_existing_file(run_dir, ["run_manifest.json", f"{run_id}_run_manifest.json"], strict=False)
    config_rel = _find_existing_file(run_dir, [f"{run_id}.yaml", "config.yaml"], strict=True)
    results_rel = _find_existing_file(run_dir, ["results.yaml", f"{run_id}_results.yaml"], strict=False)
    config_payload = _sanitize_value(_read_yaml(config_rel))
    results_payload = _read_yaml(results_rel) if results_rel else {}

    entry = {
        "artifact_id": spec.artifact_id,
        "registry_version": REGISTRY_VERSION,
        "line": spec.line,
        "model_family": spec.model_family,
        "selection_role": manifest.get("selection_role"),
        "selection_status": manifest.get("selection_status", "closed"),
        "selection_phase": manifest.get("selection_phase"),
        "semantic_role": spec.semantic_role,
        "split_id": manifest.get("split_id"),
        "cleaning_policy_id": manifest.get("cleaning_policy_id"),
        "source_id": manifest.get("source_id"),
        "selected_run_id": manifest.get("selection_run_id"),
        "selected_seed": manifest.get("selection_seed"),
        "cfg_signature": None,
        "family_policy_id": manifest.get("family_policy_id"),
        "upstream_dependencies": {
            "branch_id": manifest.get("branch_id"),
            "paired_flowpre_source_id": manifest.get("paired_flowpre_source_id"),
            "paired_flowpre_run_id": manifest.get("paired_flowpre_run_id"),
            "monitoring_policy": manifest.get("monitoring_policy"),
            "raw_bundle_dataset_name": RAW_BUNDLE_DATASET_NAME,
        },
        "config_snapshot_ref": f"config/finalists/config_snapshots/{spec.artifact_id}.yaml",
        "results_summary_ref": f"config/finalists/{spec.artifact_id}.yaml",
        "results_summary": {
            "family_policy_id": manifest.get("family_policy_id"),
            "family_seed_set": manifest.get("family_seed_set"),
            "selection_lenses": manifest.get("selection_lenses"),
            "selection_metrics": manifest.get("selection_metrics"),
            "family_metrics": manifest.get("family_metrics"),
            "selection_reason": manifest.get("selection_reason"),
            "monitoring_note": manifest.get("monitoring_note"),
            "training_summary": _extract_flowgen_training_summary(results_payload) if results_payload else {},
        },
        "rationale_summary": _compact_markdown_summary(spec.rationale_relpaths + spec.readme_relpaths),
        "local_artifact_refs": _clean_local_refs({
            "selection_manifest": _local_ref(spec.manifest_relpath),
            "run_manifest": _local_ref(run_manifest_rel),
            "config": _local_ref(config_rel),
            "results": _local_ref(results_rel),
            "rationale_md": _local_ref(spec.rationale_relpaths[0]),
            "readme_md": _local_ref(spec.readme_relpaths[0]) if spec.readme_relpaths else None,
        }),
        "integrity": _build_integrity(
            manifest_path=spec.manifest_relpath,
            config_path=config_rel,
            results_path=results_rel,
            rationale_paths=spec.rationale_relpaths,
        ),
        "notes": {
            "runner_up_run_id": manifest.get("runner_up_run_id"),
            "runner_up_seed": manifest.get("runner_up_seed"),
            "historical_support_only": bool(manifest.get("historical_support_only", False)),
        },
    }
    return entry, config_payload


def _build_artifact(spec: ArtifactSpec, *, strict: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    if spec.source_kind == "official_flowpre":
        return _load_official_flowpre(spec, strict=strict)
    if spec.source_kind == "trainonly_flowpre":
        return _load_trainonly_flowpre(spec, strict=strict)
    if spec.source_kind == "official_flowgen":
        return _load_official_flowgen(spec, strict=strict)
    if spec.source_kind == "trainonly_flowgen":
        return _load_trainonly_flowgen(spec, strict=strict)
    raise ValueError(f"Unsupported source_kind: {spec.source_kind}")


def _dump_yaml(payload: Any) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def _build_group_doc(title: str, artifact_ids: list[str], artifacts: dict[str, dict[str, Any]]) -> str:
    lines = [f"# {title}", ""]
    for artifact_id in artifact_ids:
        entry = artifacts[artifact_id]
        lines.extend(
            [
                f"## `{artifact_id}`",
                "",
                f"- linea: `{entry['line']}`",
                f"- rol semantico: {entry['semantic_role']}",
                f"- `selection_role`: `{entry['selection_role']}`",
                f"- `selected_run_id`: `{entry['selected_run_id']}`",
                f"- `selected_seed`: `{entry['selected_seed']}`",
            ]
        )
        if entry.get("cfg_signature"):
            lines.append(f"- `cfg_signature`: `{entry['cfg_signature']}`")
        if entry.get("family_policy_id"):
            lines.append(f"- `family_policy_id`: `{entry['family_policy_id']}`")
        upstream = entry.get("upstream_dependencies") or {}
        if upstream:
            lines.append(f"- dependencia upstream: `{json.dumps(upstream, ensure_ascii=False, sort_keys=True)}`")
        results = entry.get("results_summary") or {}
        selection_reason = results.get("selection_reason")
        if selection_reason:
            lines.append(f"- motivo de seleccion: {selection_reason}")
        rationale = entry.get("rationale_summary") or []
        if rationale:
            lines.append("- rationale resumido:")
            for item in rationale[:4]:
                lines.append(f"  - {item}")
        lines.extend(
            [
                "- capa local-only original:",
                f"  - config snapshot tracked: [`{entry['config_snapshot_ref']}`](../../{entry['config_snapshot_ref']})",
                f"  - manifiesto tracked: [`config/finalists/{artifact_id}.yaml`](../../config/finalists/{artifact_id}.yaml)",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _build_registry_md(artifacts: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Finalists Registry",
        "",
        "Capa tracked y public-safe de finalistas/winners. Resume la verdad ligera que hoy vive materialmente en `outputs/`, sin copiar pesos ni artefactos pesados.",
        "",
        "- indice machine-readable: [`config/finalists_registry.yaml`](../config/finalists_registry.yaml)",
        "- dossiers humanos:",
        "  - [`official_flowpre`](finalists/official_flowpre.md)",
        "  - [`official_flowgen`](finalists/official_flowgen.md)",
        "  - [`trainonly_flowpre`](finalists/trainonly_flowpre.md)",
        "  - [`trainonly_flowgen`](finalists/trainonly_flowgen.md)",
        "",
        "| artifact_id | line | model_family | selection_role | selected_run_id | seed | snapshot |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for artifact_id, entry in artifacts.items():
        lines.append(
            "| `{artifact_id}` | `{line}` | `{model_family}` | `{selection_role}` | `{selected_run_id}` | `{selected_seed}` | [`{snapshot}`](../{snapshot}) |".format(
                artifact_id=artifact_id,
                line=entry["line"],
                model_family=entry["model_family"],
                selection_role=entry["selection_role"],
                selected_run_id=entry["selected_run_id"],
                selected_seed=entry["selected_seed"],
                snapshot=entry["config_snapshot_ref"],
            )
        )
    lines.extend(
        [
            "",
            "## Regla operativa",
            "",
            "- esta capa tracked es la referencia ligera canónica de finalists/winners;",
            "- `outputs/` sigue siendo la fuente local pesada/original;",
            "- para restore completo siguen haciendo falta `data/raw/`, `config/local/` y los outputs promovidos.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _build_outputs(*, strict: bool) -> dict[str, str]:
    artifacts: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    for spec in ARTIFACT_SPECS:
        artifact, snapshot = _build_artifact(spec, strict=strict)
        artifacts[spec.artifact_id] = artifact
        snapshots[spec.artifact_id] = snapshot

    registry = {
        "registry_version": REGISTRY_VERSION,
        "artifacts": {
            artifact_id: _extract_common_registry_fields(entry)
            for artifact_id, entry in artifacts.items()
        },
    }

    outputs: dict[str, str] = {
        "config/finalists_registry.yaml": _dump_yaml(registry),
        "docs/finalists_registry.md": _build_registry_md(artifacts),
        "docs/finalists/official_flowpre.md": _build_group_doc(
            "Official FlowPre Finalists",
            [
                "official_flowpre_rrmse",
                "official_flowpre_mvn",
                "official_flowpre_fair",
                "official_flowpre_candidate_1",
                "official_flowpre_candidate_2",
            ],
            artifacts,
        ),
        "docs/finalists/official_flowgen.md": _build_group_doc(
            "Official FlowGen Winner",
            ["official_flowgen_winner"],
            artifacts,
        ),
        "docs/finalists/trainonly_flowpre.md": _build_group_doc(
            "Train-only FlowPre Finalists",
            [
                "trainonly_flowpre_candidate_1",
                "trainonly_flowpre_candidate_2",
            ],
            artifacts,
        ),
        "docs/finalists/trainonly_flowgen.md": _build_group_doc(
            "Train-only FlowGen Finalist",
            ["trainonly_flowgen_winner"],
            artifacts,
        ),
    }

    for artifact_id, entry in artifacts.items():
        outputs[f"config/finalists/{artifact_id}.yaml"] = _dump_yaml(entry)
        outputs[f"config/finalists/config_snapshots/{artifact_id}.yaml"] = _dump_yaml(snapshots[artifact_id])

    return outputs


def _write_outputs(outputs: dict[str, str]) -> None:
    for relpath, content in outputs.items():
        target = REPO_ROOT / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _check_outputs(outputs: dict[str, str]) -> list[str]:
    mismatches: list[str] = []
    for relpath, expected in outputs.items():
        target = REPO_ROOT / relpath
        if not target.exists():
            mismatches.append(f"Missing file: {relpath}")
            continue
        current = target.read_text(encoding="utf-8")
        if current != expected:
            mismatches.append(f"Outdated file: {relpath}")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description="Export tracked finalists registry from local outputs.")
    parser.add_argument("--write", action="store_true", help="Write generated tracked files.")
    parser.add_argument("--check", action="store_true", help="Check whether tracked files match generated output.")
    parser.add_argument("--strict", action="store_true", help="Fail if any expected local artifact is missing.")
    args = parser.parse_args()

    outputs = _build_outputs(strict=args.strict)

    if args.write:
        _write_outputs(outputs)

    if args.check:
        mismatches = _check_outputs(outputs)
        if mismatches:
            for mismatch in mismatches:
                print(mismatch)
            return 1
        print("Finalists registry is up to date.")
        return 0

    if not args.write:
        print("\n".join(sorted(outputs.keys())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
