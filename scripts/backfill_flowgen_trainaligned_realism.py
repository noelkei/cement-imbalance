#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sets import load_or_create_raw_splits
from evaluation.realism import compute_realism_metrics_for_set
from evaluation.results import flatten_run_results
from evaluation.temporal_realism import (
    build_temporal_realism_block,
    resolve_temporal_realism_config,
    write_temporal_realism_sidecars,
)
from scripts.f6_common import OFFICIAL_SPLIT_ID, load_json, load_yaml, write_yaml
from training.train_flowgen import (
    _loss_kwargs_from_train_cfg,
    build_flowgen_model,
    prepare_flowgen_dataloader,
    select_device,
)


MODEL_FAMILY = "flowgen"
DEFAULT_DATASET_NAME = "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"
OFFICIAL_FLOWGEN_ROOT = ROOT / "outputs" / "models" / "official" / MODEL_FAMILY
OFFICIAL_RESEED_ROOT = OFFICIAL_FLOWGEN_ROOT / "reseed_final"
TRAINONLY_ROUND1_ROOT = ROOT / "outputs" / "models" / "experimental" / "train_only" / MODEL_FAMILY / "round1"
REPORT_ROOT = ROOT / "outputs" / "reports" / "flowgen_trainaligned_realism_backfill"
OFFICIAL_SOURCE_RUN_IDS = [
    "flowgen_tpv1_c2_train_s01_e38_softclip_seed2468_v2",
    "flowgen_tpv1_c2_train_h01_bridge300_lowmmd_seed2468_v2",
    "flowgen_tpv1_c2_train_k01_e36_ksy_seed2468_v2",
    "flowgen_tpv1_c2_train_h02_bridge500_lowmmd_seed2468_v2",
    "flowgen_tpv1_c2_train_e03_seed2468_v1",
]
TRAIN_ONLY_POLICY = "train_only"


@dataclass
class RunTarget:
    cohort: str
    run_id: str
    run_dir: str
    manifest_path: str
    monitoring_policy: str
    need_realism: bool
    need_temporal: bool
    status: str
    note: str = ""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _release_memory() -> None:
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Backfill FlowGen realism W1 trainaligned metrics post-hoc for the official reseed-final panel "
            "+ its 5 source runs, and for experimental train_only round1. No retraining; load checkpoint, "
            "recompute metrics, patch results artifacts."
        )
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--condition-col", default="type")
    ap.add_argument("--realism-bootstrap", type=int, default=10)
    ap.add_argument("--realism-rvr-bootstrap", type=int, default=10)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--run-one", default=None, help="Execute exactly one planned run_id.")
    return ap.parse_args()


def _print(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(msg)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _resolve_required_artifact(run_dir: Path, run_id: str, kind: str) -> Path:
    if kind == "config":
        candidates = [run_dir / "config.yaml", run_dir / f"{run_id}.yaml"]
    elif kind == "results":
        candidates = [run_dir / "results.yaml", run_dir / f"{run_id}_results.yaml"]
    elif kind == "metrics_long":
        candidates = [run_dir / "metrics_long.csv", run_dir / f"{run_id}_metrics_long.csv"]
    elif kind == "run_manifest":
        candidates = [run_dir / "run_manifest.json", run_dir / f"{run_id}_run_manifest.json"]
    elif kind == "checkpoint":
        candidates = [run_dir / "checkpoint.pt", run_dir / f"{run_id}.pt"]
    else:
        raise ValueError(f"Unsupported artifact kind: {kind}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing required artifact '{kind}' under {run_dir}")


def _resolve_optional_alias(run_dir: Path, canonical_name: str, versioned_name: str) -> list[Path]:
    paths: list[Path] = []
    for candidate in (run_dir / canonical_name, run_dir / versioned_name):
        if candidate.exists():
            paths.append(candidate)
    unique: list[Path] = []
    seen: set[str] = set()
    for item in paths:
        key = str(item.resolve()) if item.exists() else str(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _load_results(run_dir: Path, run_id: str) -> tuple[dict[str, Any], list[Path]]:
    versioned = _resolve_required_artifact(run_dir, run_id, "results")
    payload = load_yaml(versioned)
    paths = _resolve_optional_alias(run_dir, "results.yaml", f"{run_id}_results.yaml")
    if versioned not in paths:
        paths.insert(0, versioned)
    return payload, paths


def _load_metrics_targets(run_dir: Path, run_id: str) -> list[Path]:
    versioned = _resolve_required_artifact(run_dir, run_id, "metrics_long")
    paths = _resolve_optional_alias(run_dir, "metrics_long.csv", f"{run_id}_metrics_long.csv")
    if versioned not in paths:
        paths.insert(0, versioned)
    return paths


def _build_cxy(X_df: pd.DataFrame, y_df: pd.DataFrame, *, condition_col: str) -> pd.DataFrame:
    X_df = X_df.sort_values("post_cleaning_index").reset_index(drop=True).copy()
    y_df = y_df.sort_values("post_cleaning_index").reset_index(drop=True).copy()
    df = X_df.merge(y_df, on="post_cleaning_index", how="inner", validate="one_to_one")
    x_cols = [col for col in X_df.columns if col not in ("post_cleaning_index", condition_col)]
    y_cols = [col for col in y_df.columns if col != "post_cleaning_index"]
    ordered = ["post_cleaning_index", condition_col] + x_cols + y_cols
    return df[ordered].copy()


def _safe_seed(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _monitoring_policy_from_manifest(manifest: dict[str, Any]) -> str:
    monitoring = manifest.get("monitoring") or {}
    if isinstance(monitoring, dict) and monitoring.get("policy"):
        return str(monitoring["policy"])
    axes = manifest.get("run_level_axes") or {}
    if isinstance(axes, dict) and axes.get("monitoring_policy"):
        return str(axes["monitoring_policy"])
    return "official"


def _has_trainaligned_realism_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    for component in ("overall", "x", "y"):
        suite = block.get(component)
        if not isinstance(suite, dict):
            return False
        if "w1_mean_trainaligned" not in suite or "w1_median_trainaligned" not in suite:
            return False
    per_class = block.get("per_class") or {}
    if isinstance(per_class, dict):
        for suites in per_class.values():
            if not isinstance(suites, dict):
                return False
            for component in ("overall", "x", "y"):
                suite = suites.get(component)
                if not isinstance(suite, dict):
                    return False
                if "w1_mean_trainaligned" not in suite or "w1_median_trainaligned" not in suite:
                    return False
    return True


def _has_trainaligned_temporal_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    for family in ("quartiles", "prefix_suffix"):
        family_block = block.get(family) or {}
        slices = family_block.get("slices") or {}
        if not isinstance(slices, dict) or not slices:
            return False
        for slice_payload in slices.values():
            if not isinstance(slice_payload, dict):
                return False
            for suite_name in ("generated_vs_slice", "train_ref_vs_slice_real"):
                if not _has_trainaligned_realism_block(slice_payload.get(suite_name)):
                    return False
    return True


def _collect_targets(*, force: bool) -> list[RunTarget]:
    targets: list[RunTarget] = []

    official_dirs = sorted({p.parent for p in OFFICIAL_RESEED_ROOT.rglob("run_manifest.json")})
    official_dirs.extend(OFFICIAL_FLOWGEN_ROOT / run_id for run_id in OFFICIAL_SOURCE_RUN_IDS)

    for run_dir in official_dirs:
        run_dir = run_dir.resolve()
        run_id = run_dir.name
        try:
            manifest_path = _resolve_required_artifact(run_dir, run_id, "run_manifest")
            manifest = load_json(manifest_path)
            results, _ = _load_results(run_dir, run_id)
            monitoring_policy = _monitoring_policy_from_manifest(manifest)
            need_realism = force or any(
                not _has_trainaligned_realism_block((results.get(split) or {}).get("realism"))
                for split in ("train", "val")
                if isinstance(results.get(split), dict)
            )
            temporal_block = ((results.get("val") or {}).get("temporal_realism"))
            need_temporal = force or not _has_trainaligned_temporal_block(temporal_block)
            status = "pending" if (need_realism or need_temporal) else "already_backfilled"
            note = "official_source_or_reseed"
            targets.append(
                RunTarget(
                    cohort="official",
                    run_id=run_id,
                    run_dir=str(run_dir),
                    manifest_path=str(manifest_path),
                    monitoring_policy=monitoring_policy,
                    need_realism=need_realism,
                    need_temporal=need_temporal,
                    status=status,
                    note=note,
                )
            )
        except Exception as exc:
            targets.append(
                RunTarget(
                    cohort="official",
                    run_id=run_id,
                    run_dir=str(run_dir),
                    manifest_path="",
                    monitoring_policy="official",
                    need_realism=False,
                    need_temporal=False,
                    status="error",
                    note=f"{type(exc).__name__}: {exc}",
                )
            )

    for run_dir in sorted({p.parent for p in TRAINONLY_ROUND1_ROOT.rglob("run_manifest.json")}):
        run_dir = run_dir.resolve()
        run_id = run_dir.name
        try:
            manifest_path = _resolve_required_artifact(run_dir, run_id, "run_manifest")
            manifest = load_json(manifest_path)
            results, _ = _load_results(run_dir, run_id)
            monitoring_policy = _monitoring_policy_from_manifest(manifest)
            need_realism = force or any(
                not _has_trainaligned_realism_block((results.get(split) or {}).get("realism"))
                for split in ("train", "val")
                if isinstance(results.get(split), dict)
            )
            targets.append(
                RunTarget(
                    cohort="train_only",
                    run_id=run_id,
                    run_dir=str(run_dir),
                    manifest_path=str(manifest_path),
                    monitoring_policy=monitoring_policy,
                    need_realism=need_realism,
                    need_temporal=False,
                    status="pending" if need_realism else "already_backfilled",
                    note="trainonly_round1",
                )
            )
        except Exception as exc:
            targets.append(
                RunTarget(
                    cohort="train_only",
                    run_id=run_id,
                    run_dir=str(run_dir),
                    manifest_path="",
                    monitoring_policy=TRAIN_ONLY_POLICY,
                    need_realism=False,
                    need_temporal=False,
                    status="error",
                    note=f"{type(exc).__name__}: {exc}",
                )
            )
    return sorted(targets, key=lambda item: (item.cohort, item.run_id))


def _write_plan_report(targets: list[RunTarget]) -> tuple[Path, Path]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    json_path = REPORT_ROOT / f"flowgen_trainaligned_backfill_{stamp}_plan.json"
    csv_path = REPORT_ROOT / f"flowgen_trainaligned_backfill_{stamp}_plan.csv"
    payload = {
        "analysis_id": stamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_targets": len(targets),
        "cohorts": {
            "official": sum(1 for t in targets if t.cohort == "official"),
            "train_only": sum(1 for t in targets if t.cohort == "train_only"),
        },
        "targets": [asdict(t) for t in targets],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cohort", "run_id", "run_dir", "monitoring_policy", "need_realism", "need_temporal", "status", "note"],
        )
        writer.writeheader()
        for target in targets:
            row = asdict(target)
            row.pop("manifest_path", None)
            writer.writerow(row)
    return json_path, csv_path


def _load_state_dict_into_model(model, checkpoint_path: Path, device: torch.device) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        model.load_state_dict(state_dict["state_dict"], strict=False)
    elif isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
    elif hasattr(state_dict, "state_dict"):
        model.load_state_dict(state_dict.state_dict(), strict=False)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def _rewrite_metrics_long(results: dict[str, Any], manifest: dict[str, Any], targets: list[Path]) -> None:
    metrics_df = flatten_run_results(results, manifest)
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(path, index=False)


def _patch_run(
    *,
    target: RunTarget,
    device_name: str,
    condition_col: str,
    realism_bootstrap: int,
    realism_rvr_bootstrap: int,
    quiet: bool,
) -> None:
    run_dir = Path(target.run_dir)
    run_id = target.run_id
    manifest_path = Path(target.manifest_path)

    manifest = load_json(manifest_path)
    if str(manifest.get("model_family")) != MODEL_FAMILY:
        raise ValueError(f"{run_id}: run_manifest model_family != flowgen")

    config_path = _resolve_required_artifact(run_dir, run_id, "config")
    checkpoint_path = _resolve_required_artifact(run_dir, run_id, "checkpoint")
    metrics_targets = _load_metrics_targets(run_dir, run_id)
    results, result_targets = _load_results(run_dir, run_id)
    config = load_yaml(config_path)

    model_cfg = dict(config.get("model") or {})
    train_cfg = dict(config.get("training") or {})
    if not model_cfg or not train_cfg:
        raise RuntimeError(f"{run_id}: config missing model/training sections")

    split_id = str(manifest.get("split_id") or OFFICIAL_SPLIT_ID)
    dataset_name = str(manifest.get("dataset_name") or DEFAULT_DATASET_NAME)
    monitoring_policy = _monitoring_policy_from_manifest(manifest)

    X_train, X_val, _X_test, y_train_df, y_val_df, _y_test, _r_train, r_val, _r_test = load_or_create_raw_splits(
        df_name=dataset_name,
        condition_col=condition_col,
        verbose=False,
        split_id=split_id,
        split_mode="official",
    )
    cxy_train = _build_cxy(X_train, y_train_df, condition_col=condition_col)
    cxy_val = cxy_train.copy() if monitoring_policy == TRAIN_ONLY_POLICY else _build_cxy(X_val, y_val_df, condition_col=condition_col)

    device = select_device(device_name)
    (
        x_train,
        y_train,
        x_val,
        y_val,
        _x_test,
        _y_test,
        c_train,
        c_val,
        _c_test,
        _feature_names_x,
        _target_names_y,
        _train_dataset,
        _train_dataloader,
    ) = prepare_flowgen_dataloader(
        df_train=cxy_train,
        df_val=cxy_val,
        condition_col=condition_col,
        batch_size=int(train_cfg["batch_size"]),
        device=device,
        df_test=None,
        seed=_safe_seed(manifest.get("seed")),
    )

    model = build_flowgen_model(
        model_cfg=model_cfg,
        x_dim=x_train.shape[1],
        y_dim=y_train.shape[1],
        num_classes=int(c_train.max().item()) + 1,
        device=device,
    )
    _load_state_dict_into_model(model, checkpoint_path, device)
    model.eval()

    loss_kwargs = _loss_kwargs_from_train_cfg(train_cfg)
    loss_kwargs["realism_bootstrap"] = int(realism_bootstrap)
    loss_kwargs["realism_rvr_bootstrap"] = int(realism_rvr_bootstrap)
    run_seed = _safe_seed(manifest.get("seed"))

    split_refs = {
        "train": (x_train, y_train, c_train),
        "val": (x_train, y_train, c_train) if monitoring_policy == TRAIN_ONLY_POLICY else (x_val, y_val, c_val),
    }
    for split_name, refs in split_refs.items():
        if split_name not in results or not isinstance(results.get(split_name), dict):
            continue
        if not target.need_realism and split_name in ("train", "val"):
            continue
        x_ref, y_ref, c_ref = refs
        results[split_name]["realism"] = compute_realism_metrics_for_set(
            model,
            x_ref=x_ref,
            y_ref=y_ref,
            c_ref=c_ref,
            loss_like_kwargs=dict(loss_kwargs),
            device=device,
            seed=run_seed,
        )

    if target.need_temporal and monitoring_policy != TRAIN_ONLY_POLICY:
        temporal_cfg = resolve_temporal_realism_config(train_cfg=train_cfg, loss_like_kwargs=loss_kwargs)
        temporal_cfg["bootstrap"] = int(realism_bootstrap)
        temporal_cfg["rvr_bootstrap"] = int(realism_rvr_bootstrap)
        temporal_payload = build_temporal_realism_block(
            model=model,
            x_train=x_train,
            y_train=y_train,
            c_train=c_train,
            x_val=x_val,
            y_val=y_val,
            c_val=c_val,
            cxy_val=_build_cxy(X_val, y_val_df, condition_col=condition_col),
            r_val=r_val,
            loss_like_kwargs=dict(loss_kwargs),
            temporal_cfg=temporal_cfg,
            device=device,
            seed=run_seed,
            run_id=run_id,
            split_id=split_id,
            condition_col=condition_col,
        )
        artifacts = write_temporal_realism_sidecars(out_dir=run_dir, payload=temporal_payload)
        block = dict(temporal_payload["block"])
        block["artifacts"] = dict(artifacts)
        results.setdefault("val", {})
        results["val"]["temporal_realism"] = block

    for path in result_targets:
        write_yaml(path, results)
    _rewrite_metrics_long(results, manifest, metrics_targets)
    _print(f"updated {run_id}", quiet=quiet)

    del model
    del x_train, y_train, x_val, y_val, c_train, c_val
    _release_memory()


def _run_subprocess(
    *,
    run_id: str,
    args: argparse.Namespace,
) -> int:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-one",
        run_id,
        "--device",
        str(args.device),
        "--condition-col",
        str(args.condition_col),
        "--realism-bootstrap",
        str(int(args.realism_bootstrap)),
        "--realism-rvr-bootstrap",
        str(int(args.realism_rvr_bootstrap)),
    ]
    if args.force:
        cmd.append("--force")
    if args.quiet:
        cmd.append("--quiet")
    completed = subprocess.run(cmd, cwd=ROOT)
    return int(completed.returncode)


def main() -> int:
    args = _parse_args()
    targets = _collect_targets(force=args.force)
    plan_json, plan_csv = _write_plan_report(targets)

    total = len(targets)
    official = sum(1 for t in targets if t.cohort == "official")
    train_only = sum(1 for t in targets if t.cohort == "train_only")
    pending = [t for t in targets if t.status == "pending"]
    errors = [t for t in targets if t.status == "error"]

    _print(f"Plan written to {plan_json}", quiet=args.quiet)
    _print(f"Plan CSV written to {plan_csv}", quiet=args.quiet)
    _print(
        f"targets total={total} official={official} train_only={train_only} pending={len(pending)} errors={len(errors)}",
        quiet=args.quiet,
    )

    if args.summary_only:
        return 0 if not errors else 1

    if args.run_one:
        match = next((t for t in targets if t.run_id == args.run_one), None)
        if match is None:
            raise KeyError(f"run_id not found in plan: {args.run_one}")
        if match.status == "error":
            raise RuntimeError(f"run_id {args.run_one} is invalid: {match.note}")
        if match.status != "pending":
            _print(f"skip {match.run_id}: status={match.status}", quiet=args.quiet)
            return 0
        _patch_run(
            target=match,
            device_name=str(args.device),
            condition_col=str(args.condition_col),
            realism_bootstrap=int(args.realism_bootstrap),
            realism_rvr_bootstrap=int(args.realism_rvr_bootstrap),
            quiet=args.quiet,
        )
        return 0

    failures = 0
    for idx, target in enumerate(pending, start=1):
        _print(
            f"[{idx}/{len(pending)}] {target.run_id} realism={target.need_realism} temporal={target.need_temporal}",
            quiet=args.quiet,
        )
        code = _run_subprocess(run_id=target.run_id, args=args)
        if code != 0:
            failures += 1
            print(f"[{idx}/{len(pending)}] failed: {target.run_id}")
        _release_memory()

    print("")
    print("Summary")
    print(f"- total targets: {total}")
    print(f"- official targets: {official}")
    print(f"- train_only targets: {train_only}")
    print(f"- pending executed: {len(pending)}")
    print(f"- pre-scan errors: {len(errors)}")
    print(f"- execution failures: {failures}")
    print(f"- plan json: {plan_json}")
    print(f"- plan csv: {plan_csv}")
    return 0 if not errors and failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
