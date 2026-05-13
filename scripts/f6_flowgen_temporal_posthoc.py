#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sets import load_or_create_raw_splits
from evaluation.temporal_realism import (
    build_temporal_realism_block,
    resolve_temporal_realism_config,
    write_temporal_realism_sidecars,
)
from scripts.f6_common import load_json, load_yaml, write_yaml
from training.train_flowgen import (
    _loss_kwargs_from_train_cfg,
    build_flowgen_model,
    prepare_flowgen_dataloader,
    select_device,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill temporal realism artifacts for an existing FlowGen run.")
    parser.add_argument("run_path", type=Path, help="Run directory or run_manifest.json path.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device used for temporal evaluation backfill.",
    )
    parser.add_argument(
        "--condition-col",
        type=str,
        default="type",
        help="Condition column used by FlowGen.",
    )
    return parser.parse_args()


def _resolve_run_dir_and_manifest(path: Path) -> tuple[Path, Path, dict[str, Any]]:
    candidate = path.resolve()
    if candidate.is_file():
        if candidate.name.endswith("_run_manifest.json") or candidate.name == "run_manifest.json":
            manifest_path = candidate
            run_dir = candidate.parent
        else:
            raise FileNotFoundError(f"Unsupported run path file: {candidate}")
    else:
        run_dir = candidate
        manifest_candidates = sorted(
            [run_dir / "run_manifest.json", *run_dir.glob("*_run_manifest.json")],
            key=lambda item: (item.name != "run_manifest.json", item.name),
        )
        manifest_path = next((item for item in manifest_candidates if item.exists()), None)
        if manifest_path is None:
            raise FileNotFoundError(f"Could not resolve run manifest under {run_dir}")
    manifest = load_json(manifest_path)
    return run_dir, manifest_path, manifest


def _resolve_existing(path_candidates: list[Path]) -> Path:
    for path in path_candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the expected artifacts exist: {[str(path) for path in path_candidates]}")


def _resolve_config_path(run_dir: Path, manifest: dict[str, Any], run_id: str) -> Path:
    configured = manifest.get("config_path")
    candidates: list[Path] = []
    if configured:
        cfg_path = Path(str(configured))
        if not cfg_path.is_absolute():
            cfg_path = run_dir / cfg_path
        candidates.append(cfg_path)
    candidates.extend([run_dir / "config.yaml", run_dir / f"{run_id}.yaml"])
    return _resolve_existing(candidates)


def _resolve_checkpoint_path(run_dir: Path, run_id: str) -> Path:
    return _resolve_existing([run_dir / "checkpoint.pt", run_dir / f"{run_id}.pt"])


def _resolve_results_paths(run_dir: Path, run_id: str) -> tuple[Path, Path | None]:
    versioned = _resolve_existing([run_dir / f"{run_id}_results.yaml", run_dir / "results.yaml"])
    canonical = run_dir / "results.yaml"
    return versioned, canonical if canonical.exists() else None


def _build_cxy(X_df, y_df, *, condition_col: str):
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


def main() -> int:
    args = _parse_args()
    run_dir, _manifest_path, manifest = _resolve_run_dir_and_manifest(args.run_path)
    run_id = str(manifest.get("run_id") or run_dir.name)

    results_path, canonical_results_path = _resolve_results_paths(run_dir, run_id)
    config_path = _resolve_config_path(run_dir, manifest, run_id)
    checkpoint_path = _resolve_checkpoint_path(run_dir, run_id)

    config = load_yaml(config_path)
    model_cfg = dict(config.get("model") or {})
    train_cfg = dict(config.get("training") or {})
    if not model_cfg or not train_cfg:
        raise RuntimeError(f"FlowGen config is missing model/training sections: {config_path}")

    split_id = str(manifest.get("split_id") or "init_temporal_processed_v1")
    dataset_name = str(manifest.get("dataset_name") or "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1")
    condition_col = str(args.condition_col)

    X_train, X_val, _X_test, y_train, y_val, _y_test, _r_train, r_val, _r_test = load_or_create_raw_splits(
        df_name=dataset_name,
        condition_col=condition_col,
        verbose=False,
        split_id=split_id,
        split_mode="official",
    )
    cxy_train = _build_cxy(X_train, y_train, condition_col=condition_col)
    cxy_val = _build_cxy(X_val, y_val, condition_col=condition_col)

    device = select_device(args.device)
    (
        x_train,
        y_train_t,
        x_val,
        y_val_t,
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
        y_dim=y_train_t.shape[1],
        num_classes=int(c_train.max().item()) + 1,
        device=device,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        model.load_state_dict(state_dict["state_dict"], strict=False)
    elif isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
    elif hasattr(state_dict, "state_dict"):
        model.load_state_dict(state_dict.state_dict(), strict=False)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.eval()

    loss_kwargs = _loss_kwargs_from_train_cfg(train_cfg)
    loss_kwargs.update({"x_ref_all": x_train, "y_ref_all": y_train_t, "c_ref_all": c_train})
    temporal_cfg = resolve_temporal_realism_config(train_cfg=train_cfg, loss_like_kwargs=loss_kwargs)

    payload = build_temporal_realism_block(
        model=model,
        x_train=x_train,
        y_train=y_train_t,
        c_train=c_train,
        x_val=x_val,
        y_val=y_val_t,
        c_val=c_val,
        cxy_val=cxy_val,
        r_val=r_val,
        loss_like_kwargs=loss_kwargs,
        temporal_cfg=temporal_cfg,
        device=device,
        seed=_safe_seed(manifest.get("seed")),
        run_id=run_id,
        split_id=split_id,
        condition_col=condition_col,
    )
    artifacts = write_temporal_realism_sidecars(out_dir=run_dir, payload=payload)

    results = load_yaml(results_path)
    results.setdefault("val", {})
    block = dict(payload["block"])
    block["artifacts"] = dict(artifacts)
    results["val"]["temporal_realism"] = block
    write_yaml(results_path, results)
    if canonical_results_path is not None and canonical_results_path != results_path:
        write_yaml(canonical_results_path, results)

    print(f"Temporal realism written under {run_dir / 'temporal'}")
    print(f"Updated results artifact: {results_path}")
    if canonical_results_path is not None and canonical_results_path != results_path:
        print(f"Updated canonical results artifact: {canonical_results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
