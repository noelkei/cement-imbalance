#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg_", dir="/private/tmp"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sets import DEFAULT_OFFICIAL_DATASET_NAME, load_or_create_raw_splits
from evaluation.metrics import compute_regression_metrics_from_preds


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "reports" / "xgboost_temporal_vs_legacy"
DEFAULT_SEED = 42
DEFAULT_TYPE_CATEGORIES = [0, 1, 2]
DEFAULT_XGB_CONFIG: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 1200,
    "learning_rate": 0.035,
    "max_depth": 4,
    "min_child_weight": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.02,
    "reg_lambda": 1.50,
    "gamma": 0.0,
    "tree_method": "hist",
    "max_bin": 256,
    "eval_metric": "rmse",
    "early_stopping_rounds": 60,
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    label: str
    df_name: str
    split_mode: str
    split_id: str
    notes: str


DATASET_SPECS = (
    DatasetSpec(
        dataset_id="legacy",
        label="Historical shuffled split",
        df_name="df_input",
        split_mode="legacy",
        split_id="legacy_random_split",
        notes="Historical random/shuffled split, kept only as legacy comparison.",
    ),
    DatasetSpec(
        dataset_id="official",
        label="Official temporal split",
        df_name=DEFAULT_OFFICIAL_DATASET_NAME,
        split_mode="official",
        split_id="init_temporal_processed_v1",
        notes="Canonical temporal split with val-before-test holdouts.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the same XGBoost regressor on the legacy shuffled bundle and the "
            "official temporal bundle, then export metrics and learning curves."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the report artifacts will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used by XGBoost.",
    )
    return parser.parse_args()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_bundle(spec: DatasetSpec) -> dict[str, pd.DataFrame]:
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = load_or_create_raw_splits(
        df_name=spec.df_name,
        split_mode=spec.split_mode,
        verbose=False,
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def build_design_matrix(
    X: pd.DataFrame,
    *,
    type_categories: list[int],
    numeric_feature_order: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    type_values = X["type"].astype(int).to_numpy()
    unknown_types = sorted(set(type_values.tolist()) - set(type_categories))
    if unknown_types:
        raise ValueError(f"Unexpected type codes found: {unknown_types}")

    if numeric_feature_order is None:
        numeric_feature_order = [
            col for col in X.columns
            if col not in {"post_cleaning_index", "type"}
        ]

    numeric = X[numeric_feature_order].to_numpy(dtype=np.float32)
    type_onehot = np.zeros((len(X), len(type_categories)), dtype=np.float32)
    category_to_col = {category: idx for idx, category in enumerate(type_categories)}
    for row_idx, category in enumerate(type_values):
        type_onehot[row_idx, category_to_col[int(category)]] = 1.0

    matrix = np.concatenate([type_onehot, numeric], axis=1)
    feature_names = [f"type_{category}" for category in type_categories] + numeric_feature_order
    return matrix, type_values, feature_names, X["post_cleaning_index"].to_numpy(dtype=np.int64)


def compute_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    type_codes: np.ndarray,
) -> dict[str, Any]:
    y_true_tensor = torch.tensor(y_true.reshape(-1, 1), dtype=torch.float32)
    y_pred_tensor = torch.tensor(y_pred.reshape(-1, 1), dtype=torch.float32)
    type_tensor = torch.tensor(type_codes, dtype=torch.long)
    return compute_regression_metrics_from_preds(
        y_hat=y_pred_tensor,
        y=y_true_tensor,
        c=type_tensor,
    )


def collect_curve_history(
    model: xgb.XGBRegressor,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_val: np.ndarray,
) -> pd.DataFrame:
    evals_result = model.evals_result()
    train_rmse = evals_result["validation_0"]["rmse"]
    val_rmse = evals_result["validation_1"]["rmse"]
    rows: list[dict[str, Any]] = []

    for round_idx in range(len(train_rmse)):
        upto = round_idx + 1
        train_pred = model.predict(X_train, iteration_range=(0, upto))
        val_pred = model.predict(X_val, iteration_range=(0, upto))

        train_metrics = compute_metrics(y_true=y_train, y_pred=train_pred, type_codes=c_train)
        val_metrics = compute_metrics(y_true=y_val, y_pred=val_pred, type_codes=c_val)

        rows.append(
            {
                "round": upto,
                "split": "train",
                "rmse": float(train_rmse[round_idx]),
                "r2": float(train_metrics["overall"]["r2"]),
                "rrmse": float(train_metrics["overall"]["rrmse"]),
            }
        )
        rows.append(
            {
                "round": upto,
                "split": "val",
                "rmse": float(val_rmse[round_idx]),
                "r2": float(val_metrics["overall"]["r2"]),
                "rrmse": float(val_metrics["overall"]["rrmse"]),
            }
        )

    return pd.DataFrame(rows)


def overall_rows(
    *,
    dataset_id: str,
    dataset_label: str,
    split: str,
    metrics: dict[str, Any],
    best_round: int,
    best_val_rmse: float,
) -> list[dict[str, Any]]:
    overall = metrics["overall"]
    macro = metrics.get("macro", {})
    worst = metrics.get("worst_class", {})
    return [{
        "dataset_id": dataset_id,
        "dataset_label": dataset_label,
        "split": split,
        "n_obs": int(overall["n"]),
        "best_round": int(best_round),
        "best_val_rmse": float(best_val_rmse),
        "r2": float(overall["r2"]),
        "rrmse": float(overall["rrmse"]),
        "rmse": float(overall["rmse"]),
        "mae": float(overall["mae"]),
        "medae": float(overall["medae"]),
        "mape": float(overall["mape"]),
        "macro_r2": float(macro.get("r2", np.nan)),
        "macro_rrmse": float(macro.get("rrmse", np.nan)),
        "worst_class_r2": float(worst.get("r2", np.nan)),
        "worst_class_rrmse": float(worst.get("rrmse", np.nan)),
    }]


def per_class_rows(
    *,
    dataset_id: str,
    dataset_label: str,
    split: str,
    metrics: dict[str, Any],
    best_round: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id, suite in sorted(metrics.get("per_class", {}).items(), key=lambda item: int(item[0])):
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_label": dataset_label,
                "split": split,
                "type_code": int(class_id),
                "n_obs": int(suite["n"]),
                "best_round": int(best_round),
                "r2": float(suite["r2"]),
                "rrmse": float(suite["rrmse"]),
                "rmse": float(suite["rmse"]),
                "mae": float(suite["mae"]),
                "medae": float(suite["medae"]),
                "mape": float(suite["mape"]),
            }
        )
    return rows


def dataset_overview_rows(
    spec: DatasetSpec,
    bundle: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        X = bundle[f"X_{split}"]
        y = bundle[f"y_{split}"]
        counts = X["type"].astype(int).value_counts().sort_index().to_dict()
        row = {
            "dataset_id": spec.dataset_id,
            "dataset_label": spec.label,
            "split": split,
            "n_rows": int(len(X)),
            "n_features_raw": int(X.shape[1] - 2),
            "target_mean": float(y["init"].mean()),
            "target_std": float(y["init"].std(ddof=0)),
        }
        for category in DEFAULT_TYPE_CATEGORIES:
            row[f"type_{category}_count"] = int(counts.get(category, 0))
        rows.append(row)
    return rows


def train_single_dataset(
    spec: DatasetSpec,
    *,
    seed: int,
    xgb_config: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    bundle = load_bundle(spec)
    type_categories = sorted(bundle["X_train"]["type"].astype(int).unique().tolist())
    if type_categories != DEFAULT_TYPE_CATEGORIES:
        raise ValueError(
            f"{spec.dataset_id}: expected train type categories {DEFAULT_TYPE_CATEGORIES}, got {type_categories}"
        )

    numeric_feature_order = [
        col for col in bundle["X_train"].columns
        if col not in {"post_cleaning_index", "type"}
    ]

    X_train, c_train, feature_names, train_idx = build_design_matrix(
        bundle["X_train"],
        type_categories=type_categories,
        numeric_feature_order=numeric_feature_order,
    )
    X_val, c_val, _, val_idx = build_design_matrix(
        bundle["X_val"],
        type_categories=type_categories,
        numeric_feature_order=numeric_feature_order,
    )
    X_test, c_test, _, test_idx = build_design_matrix(
        bundle["X_test"],
        type_categories=type_categories,
        numeric_feature_order=numeric_feature_order,
    )
    y_train = bundle["y_train"]["init"].to_numpy(dtype=np.float32)
    y_val = bundle["y_val"]["init"].to_numpy(dtype=np.float32)
    y_test = bundle["y_test"]["init"].to_numpy(dtype=np.float32)

    model = xgb.XGBRegressor(
        **xgb_config,
        random_state=seed,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    best_round = int(model.best_iteration) + 1 if model.best_iteration is not None else int(model.n_estimators)
    best_val_rmse = float(model.best_score) if model.best_score is not None else float("nan")

    predictions = {
        "train": model.predict(X_train, iteration_range=(0, best_round)),
        "val": model.predict(X_val, iteration_range=(0, best_round)),
        "test": model.predict(X_test, iteration_range=(0, best_round)),
    }

    final_metrics = {
        "train": compute_metrics(y_true=y_train, y_pred=predictions["train"], type_codes=c_train),
        "val": compute_metrics(y_true=y_val, y_pred=predictions["val"], type_codes=c_val),
        "test": compute_metrics(y_true=y_test, y_pred=predictions["test"], type_codes=c_test),
    }

    curve_history = collect_curve_history(
        model,
        X_train=X_train,
        y_train=y_train,
        c_train=c_train,
        X_val=X_val,
        y_val=y_val,
        c_val=c_val,
    )

    dataset_dir = ensure_dir(out_dir / spec.dataset_id)
    model.get_booster().save_model(dataset_dir / f"{spec.dataset_id}_xgb_model.json")
    curve_history.to_csv(dataset_dir / f"{spec.dataset_id}_curve_history.csv", index=False)

    prediction_rows = []
    for split, indices, target, preds, types in (
        ("train", train_idx, y_train, predictions["train"], c_train),
        ("val", val_idx, y_val, predictions["val"], c_val),
        ("test", test_idx, y_test, predictions["test"], c_test),
    ):
        split_frame = pd.DataFrame(
            {
                "dataset_id": spec.dataset_id,
                "split": split,
                "post_cleaning_index": indices,
                "type_code": types,
                "y_true": target,
                "y_pred": preds,
                "abs_error": np.abs(target - preds),
            }
        )
        prediction_rows.append(split_frame)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        dataset_dir / f"{spec.dataset_id}_predictions.csv",
        index=False,
    )

    return {
        "spec": spec,
        "feature_names": feature_names,
        "numeric_feature_order": numeric_feature_order,
        "type_categories": type_categories,
        "bundle": bundle,
        "overview_rows": dataset_overview_rows(spec, bundle),
        "curve_history": curve_history,
        "final_metrics": final_metrics,
        "best_round": best_round,
        "best_val_rmse": best_val_rmse,
    }


def plot_learning_curves(curve_history: pd.DataFrame, *, dataset_label: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    metric_specs = (
        ("rmse", "RMSE Loss", False),
        ("r2", "R2", True),
        ("rrmse", "RRMSE", False),
    )
    colors = {"train": "#1f77b4", "val": "#d62728"}

    for ax, (metric_name, title, maximize) in zip(axes, metric_specs):
        for split in ("train", "val"):
            subset = curve_history.loc[curve_history["split"] == split].sort_values("round")
            ax.plot(subset["round"], subset[metric_name], label=split, linewidth=2.0, color=colors[split])
        ax.set_title(f"{dataset_label}: {title}")
        ax.set_xlabel("Boosting round")
        ax.set_ylabel(metric_name.upper())
        ax.grid(alpha=0.25)
        if maximize:
            ax.set_ylim(bottom=min(-0.05, float(curve_history[metric_name].min()) - 0.02))
        axes[0].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_comparison(overall_df: pd.DataFrame, out_path: Path) -> None:
    compare_df = overall_df.loc[overall_df["split"].isin(["val", "test"])].copy()
    compare_df["dataset_split"] = compare_df["dataset_id"] + "_" + compare_df["split"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric_name, title in (
        (axes[0], "r2", "R2 on holdout splits"),
        (axes[1], "rrmse", "RRMSE on holdout splits"),
    ):
        bars = ax.bar(compare_df["dataset_split"], compare_df[metric_name], color=["#7f7f7f", "#7f7f7f", "#2ca02c", "#2ca02c"])
        ax.set_title(title)
        ax.set_ylabel(metric_name.upper())
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        for bar in bars:
            height = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    *,
    out_path: Path,
    run_dir: Path,
    seed: int,
    xgb_config: dict[str, Any],
    overview_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
) -> None:
    pivot_r2 = overall_df.pivot(index="split", columns="dataset_id", values="r2").reindex(["train", "val", "test"])
    pivot_rrmse = overall_df.pivot(index="split", columns="dataset_id", values="rrmse").reindex(["train", "val", "test"])
    difficulty = []
    for split in ("val", "test"):
        official_r2 = float(pivot_r2.loc[split, "official"])
        legacy_r2 = float(pivot_r2.loc[split, "legacy"])
        official_rrmse = float(pivot_rrmse.loc[split, "official"])
        legacy_rrmse = float(pivot_rrmse.loc[split, "legacy"])
        difficulty.append(
            {
                "split": split,
                "official_minus_legacy_r2": official_r2 - legacy_r2,
                "official_minus_legacy_rrmse": official_rrmse - legacy_rrmse,
            }
        )
    difficulty_df = pd.DataFrame(difficulty)

    train_test_gap_rows = []
    for dataset_id in ("legacy", "official"):
        train_r2 = float(pivot_r2.loc["train", dataset_id])
        test_r2 = float(pivot_r2.loc["test", dataset_id])
        train_rrmse = float(pivot_rrmse.loc["train", dataset_id])
        test_rrmse = float(pivot_rrmse.loc["test", dataset_id])
        train_test_gap_rows.append(
            {
                "dataset_id": dataset_id,
                "r2_gap_train_minus_test": train_r2 - test_r2,
                "rrmse_gap_test_minus_train": test_rrmse - train_rrmse,
            }
        )
    gap_df = pd.DataFrame(train_test_gap_rows)

    report_lines = [
        "# XGBoost comparison: legacy shuffled vs official temporal",
        "",
        "## Experiment setup",
        f"- Run directory: `{run_dir}`",
        f"- Seed: `{seed}`",
        "- Same hyperparameter configuration used for both datasets.",
        "- No feature scaling and no feature transforms beyond the existing cleaned raw bundles.",
        "- `type` is included as conditioning information through one-hot encoding (`type_0`, `type_1`, `type_2`).",
        "- Selection protocol: fit on `train`, early stopping monitored on `val`, final confirmation reported on `test`.",
        "",
        "```json",
        json.dumps(xgb_config, indent=2, sort_keys=True),
        "```",
        "",
        "## Dataset overview",
        dataframe_to_markdown(overview_df),
        "",
        "## Overall metrics",
        dataframe_to_markdown(overall_df),
        "",
        "## Per-class metrics",
        dataframe_to_markdown(per_class_df),
        "",
        "## Difficulty deltas",
        dataframe_to_markdown(difficulty_df),
        "",
        "## Generalization gaps",
        dataframe_to_markdown(gap_df),
        "",
        "## Reading guide",
        "- Lower `RRMSE` and higher `R2` indicate easier prediction.",
        "- `official_minus_legacy_r2 < 0` means the temporal split is harder than the shuffled split on that holdout.",
        "- `official_minus_legacy_rrmse > 0` means the temporal split is harder than the shuffled split on that holdout.",
        "- Large `train -> test` gaps flag stronger distribution shift or overfitting pressure.",
        "",
    ]
    out_path.write_text("\n".join(report_lines), encoding="utf-8")


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [format_markdown_value(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = ensure_dir(args.output_root / utc_stamp())

    config_payload = {
        "seed": int(args.seed),
        "xgb_config": DEFAULT_XGB_CONFIG,
        "dataset_specs": [spec.__dict__ for spec in DATASET_SPECS],
        "type_encoding": {
            "strategy": "one_hot",
            "categories": DEFAULT_TYPE_CATEGORIES,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")

    results = [
        train_single_dataset(spec, seed=int(args.seed), xgb_config=DEFAULT_XGB_CONFIG, out_dir=run_dir)
        for spec in DATASET_SPECS
    ]

    overview_rows: list[dict[str, Any]] = []
    overall_rows_acc: list[dict[str, Any]] = []
    per_class_rows_acc: list[dict[str, Any]] = []

    for result in results:
        spec = result["spec"]
        overview_rows.extend(result["overview_rows"])

        for split in ("train", "val", "test"):
            metrics = result["final_metrics"][split]
            overall_rows_acc.extend(
                overall_rows(
                    dataset_id=spec.dataset_id,
                    dataset_label=spec.label,
                    split=split,
                    metrics=metrics,
                    best_round=result["best_round"],
                    best_val_rmse=result["best_val_rmse"],
                )
            )
            per_class_rows_acc.extend(
                per_class_rows(
                    dataset_id=spec.dataset_id,
                    dataset_label=spec.label,
                    split=split,
                    metrics=metrics,
                    best_round=result["best_round"],
                )
            )

        plot_learning_curves(
            result["curve_history"],
            dataset_label=spec.label,
            out_path=run_dir / f"{spec.dataset_id}_learning_curves.png",
        )

    overview_df = pd.DataFrame(overview_rows)
    overall_df = pd.DataFrame(overall_rows_acc).sort_values(["dataset_id", "split"]).reset_index(drop=True)
    per_class_df = pd.DataFrame(per_class_rows_acc).sort_values(["dataset_id", "split", "type_code"]).reset_index(drop=True)

    overview_df.to_csv(run_dir / "dataset_overview.csv", index=False)
    overall_df.to_csv(run_dir / "overall_metrics.csv", index=False)
    per_class_df.to_csv(run_dir / "per_class_metrics.csv", index=False)

    plot_dataset_comparison(overall_df, run_dir / "holdout_metric_comparison.png")
    write_report(
        out_path=run_dir / "report.md",
        run_dir=run_dir,
        seed=int(args.seed),
        xgb_config=DEFAULT_XGB_CONFIG,
        overview_df=overview_df,
        overall_df=overall_df,
        per_class_df=per_class_df,
    )

    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
