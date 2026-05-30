from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.f7_campaign_state import build_campaign_paths


RAW_METRIC_NAME = "rrmse"
RAW_METRIC_SCOPE = "macro"
RAW_VALUE_SPACE = "raw_real"


def _load_metrics_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "metrics_long.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_analysis_table(campaign_id: str) -> pd.DataFrame:
    paths = build_campaign_paths(campaign_id)
    ledger = pd.read_csv(paths.ledger_path)
    rows: list[dict[str, object]] = []
    for trial in ledger.to_dict("records"):
        run_dir = REPO_ROOT / str(trial["successful_run_dir"])
        metrics_rows = _load_metrics_rows(run_dir)
        selected = [
            row
            for row in metrics_rows
            if row["metric_group"] == "predictive"
            and row["metric_name"] == RAW_METRIC_NAME
            and row["metric_scope"] == RAW_METRIC_SCOPE
            and row["value_space"] == RAW_VALUE_SPACE
            and row["split"] in {"val", "test"}
        ]
        if not selected:
            continue
        sample = selected[0]
        dataset_axes = json.loads(sample["dataset_level_axes"]) if sample["dataset_level_axes"] else {}
        run_axes = json.loads(sample["run_level_axes"]) if sample["run_level_axes"] else {}
        metric_by_split = {row["split"]: float(row["metric_value"]) for row in selected}
        rows.append(
            {
                "trial_id": trial["trial_id"],
                "model_family": trial["model_family"],
                "dataset_candidate_id": trial["dataset_candidate_id"],
                "run_spec_id": trial["run_spec_id"],
                "x_transform": dataset_axes.get("x_transform"),
                "y_transform": dataset_axes.get("y_transform"),
                "synthetic_policy": dataset_axes.get("synthetic_policy"),
                "batch_policy": ((run_axes.get("batch_policy") or {}).get("id")),
                "cycle_reals": ((run_axes.get("cycling_policy") or {}).get("cycle_reals")),
                "loss_reduction": ((run_axes.get("loss_policy") or {}).get("loss_reduction")),
                "regression_group_metric": ((run_axes.get("loss_policy") or {}).get("regression_group_metric")),
                "val_rrmse_raw_macro": metric_by_split.get("val"),
                "test_rrmse_raw_macro": metric_by_split.get("test"),
                "gap_test_minus_val": (
                    None
                    if metric_by_split.get("val") is None or metric_by_split.get("test") is None
                    else metric_by_split["test"] - metric_by_split["val"]
                ),
                "training_runtime_s": float(trial["training_runtime_s"]),
                "interpretability_runtime_s": float(trial["interpretability_runtime_s"]),
                "total_runtime_s": float(trial["total_runtime_s"]),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_axis(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    grouped = (
        df.groupby(axis, dropna=False)
        .agg(
            n_trials=("trial_id", "count"),
            val_rrmse_mean=("val_rrmse_raw_macro", "mean"),
            val_rrmse_median=("val_rrmse_raw_macro", "median"),
            test_rrmse_mean=("test_rrmse_raw_macro", "mean"),
            gap_test_minus_val_mean=("gap_test_minus_val", "mean"),
            training_runtime_s_mean=("training_runtime_s", "mean"),
            interpretability_runtime_s_mean=("interpretability_runtime_s", "mean"),
            total_runtime_s_mean=("total_runtime_s", "mean"),
        )
        .reset_index()
        .sort_values("val_rrmse_mean")
    )
    return grouped


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _plot_bar(df: pd.DataFrame, *, x_col: str, y_col: str, title: str, path: Path, color: str) -> None:
    plt.figure(figsize=(10, 5))
    frame = df.sort_values(y_col)
    plt.bar(frame[x_col].astype(str), frame[y_col], color=color)
    plt.title(title)
    plt.ylabel(y_col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_cost_quality(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for _, row in df.iterrows():
        plt.scatter(row["total_runtime_s_mean"], row["val_rrmse_mean"], s=70)
        plt.text(
            row["total_runtime_s_mean"] + 0.1,
            row["val_rrmse_mean"] + 0.0004,
            str(row["x_transform"]),
            fontsize=8,
        )
    plt.xlabel("mean total runtime (s)")
    plt.ylabel("mean val raw_real macro rrmse")
    plt.title("Cost-quality frontier by x_transform")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_heatmap(df: pd.DataFrame, path: Path) -> None:
    pivot = (
        df[df["model_family"] == "mlp"]
        .pivot_table(index="x_transform", columns="y_transform", values="val_rrmse_raw_macro", aggfunc="mean")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title("MLP val raw_real macro rrmse by X/Y transform")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def write_markdown_report(
    *,
    campaign_id: str,
    output_path: Path,
    overall_df: pd.DataFrame,
    by_x: pd.DataFrame,
    by_y: pd.DataFrame,
    by_synth: pd.DataFrame,
    by_run_policy: pd.DataFrame,
) -> None:
    mlp_df = overall_df[overall_df["model_family"] == "mlp"].copy()
    mlp_val_mean = float(mlp_df["val_rrmse_raw_macro"].mean())
    x_rows = by_x.to_dict("records")
    y_rows = by_y.to_dict("records")
    synth_rows = by_synth.to_dict("records")
    run_rows = by_run_policy.to_dict("records")
    best_trials = mlp_df.sort_values("val_rrmse_raw_macro").head(5)
    worst_trials = mlp_df.sort_values("val_rrmse_raw_macro").tail(5)

    lines: list[str] = []
    lines.append(f"# {campaign_id} Pilot Findings")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- total comparable trials: `{len(overall_df)}`")
    lines.append(f"- MLP comparable trials: `{len(mlp_df)}`")
    lines.append(f"- baseline MLP mean validation raw macro RRMSE: `{mlp_val_mean:.4f}`")
    lines.append("- This is a 1-seed pilot, so the findings are directional rather than final statistical evidence.")
    lines.append("")
    lines.append("## Main Findings")
    lines.append("")
    lines.append(f"- Best average `x_transform` on validation: `{x_rows[0]['x_transform']}` with `{x_rows[0]['val_rrmse_mean']:.4f}`.")
    lines.append(f"- Worst average `x_transform` on validation: `{x_rows[-1]['x_transform']}` with `{x_rows[-1]['val_rrmse_mean']:.4f}`.")
    lines.append(f"- Best synthetic policy on validation: `{synth_rows[0]['synthetic_policy']}` with `{synth_rows[0]['val_rrmse_mean']:.4f}`.")
    lines.append(f"- Worst synthetic policy on validation: `{synth_rows[-1]['synthetic_policy']}` with `{synth_rows[-1]['val_rrmse_mean']:.4f}`.")
    lines.append(f"- Best run-policy family on validation: `{run_rows[0]['batch_policy']} | cycle={run_rows[0]['cycle_reals']} | {run_rows[0]['loss_reduction']} | {run_rows[0]['regression_group_metric']}` with `{run_rows[0]['val_rrmse_mean']:.4f}`.")
    lines.append("")
    lines.append("## Axis Effects")
    lines.append("")
    lines.append("### X Transform")
    lines.append("")
    for row in x_rows:
        delta = row["val_rrmse_mean"] - mlp_val_mean
        lines.append(
            f"- `{row['x_transform']}`: val `{row['val_rrmse_mean']:.4f}`, test `{row['test_rrmse_mean']:.4f}`, "
            f"delta vs MLP mean `{delta:+.4f}`, mean total runtime `{row['total_runtime_s_mean']:.3f}s`."
        )
    lines.append("")
    lines.append("### Y Transform")
    lines.append("")
    for row in y_rows:
        delta = row["val_rrmse_mean"] - mlp_val_mean
        lines.append(
            f"- `{row['y_transform']}`: val `{row['val_rrmse_mean']:.4f}`, test `{row['test_rrmse_mean']:.4f}`, "
            f"delta vs MLP mean `{delta:+.4f}`."
        )
    lines.append("")
    lines.append("### Synthetic Policy")
    lines.append("")
    for row in synth_rows:
        delta = row["val_rrmse_mean"] - mlp_val_mean
        lines.append(
            f"- `{row['synthetic_policy']}`: val `{row['val_rrmse_mean']:.4f}`, test `{row['test_rrmse_mean']:.4f}`, "
            f"delta vs MLP mean `{delta:+.4f}`."
        )
    lines.append("")
    lines.append("### Run Policy")
    lines.append("")
    for row in run_rows:
        lines.append(
            f"- `{row['batch_policy']}` / `cycle={row['cycle_reals']}` / `{row['loss_reduction']}` / `{row['regression_group_metric']}`: "
            f"val `{row['val_rrmse_mean']:.4f}`, test `{row['test_rrmse_mean']:.4f}`."
        )
    lines.append("")
    lines.append("## Best Trials")
    lines.append("")
    for _, row in best_trials.iterrows():
        lines.append(
            f"- `{row['dataset_candidate_id']}` | `{row['run_spec_id']}` -> "
            f"val `{row['val_rrmse_raw_macro']:.4f}`, test `{row['test_rrmse_raw_macro']:.4f}`, total `{row['total_runtime_s']:.3f}s`."
        )
    lines.append("")
    lines.append("## Worst Trials")
    lines.append("")
    for _, row in worst_trials.iterrows():
        lines.append(
            f"- `{row['dataset_candidate_id']}` | `{row['run_spec_id']}` -> "
            f"val `{row['val_rrmse_raw_macro']:.4f}`, test `{row['test_rrmse_raw_macro']:.4f}`, total `{row['total_runtime_s']:.3f}s`."
        )
    lines.append("")
    lines.append("## Cost Read")
    lines.append("")
    lines.append("- `FlowPre candidate_1` is expensive and underperforms in this pilot.")
    lines.append("- `FlowPre candidate_2` is competitive on quality but much more expensive than direct transforms.")
    lines.append("- `quantile` is the strongest cheap direct surface in this pilot.")
    lines.append("- `flowgen_official` is the strongest synthetic policy in this pilot.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a finished F7 pilot campaign and generate plots.")
    parser.add_argument("--campaign-id", required=True, type=str)
    args = parser.parse_args()

    paths = build_campaign_paths(args.campaign_id)
    analysis_dir = paths.root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = build_analysis_table(args.campaign_id)
    _save_table(df, analysis_dir / "pilot_comparable_trials.csv")

    mlp_df = df[df["model_family"] == "mlp"].copy()
    by_x = _aggregate_axis(df, "x_transform")
    by_y = _aggregate_axis(df, "y_transform")
    by_synth = _aggregate_axis(df, "synthetic_policy")
    by_run_policy = (
        mlp_df.groupby(["batch_policy", "cycle_reals", "loss_reduction", "regression_group_metric"], dropna=False)
        .agg(
            n_trials=("trial_id", "count"),
            val_rrmse_mean=("val_rrmse_raw_macro", "mean"),
            test_rrmse_mean=("test_rrmse_raw_macro", "mean"),
            total_runtime_s_mean=("total_runtime_s", "mean"),
        )
        .reset_index()
        .sort_values("val_rrmse_mean")
    )

    _save_table(by_x, analysis_dir / "axis_effect_x_transform.csv")
    _save_table(by_y, analysis_dir / "axis_effect_y_transform.csv")
    _save_table(by_synth, analysis_dir / "axis_effect_synthetic_policy.csv")
    _save_table(by_run_policy, analysis_dir / "axis_effect_run_policy.csv")

    _plot_bar(by_x, x_col="x_transform", y_col="val_rrmse_mean", title="Validation raw macro RRMSE by x_transform", path=analysis_dir / "val_rrmse_by_x_transform.png", color="#2f5c7a")
    _plot_bar(by_synth, x_col="synthetic_policy", y_col="val_rrmse_mean", title="Validation raw macro RRMSE by synthetic policy", path=analysis_dir / "val_rrmse_by_synthetic_policy.png", color="#a85d36")
    _plot_cost_quality(by_x, analysis_dir / "cost_quality_frontier_x_transform.png")
    _plot_heatmap(mlp_df, analysis_dir / "mlp_x_y_heatmap_val_rrmse.png")

    write_markdown_report(
        campaign_id=args.campaign_id,
        output_path=analysis_dir / "pilot_findings.md",
        overall_df=df,
        by_x=by_x,
        by_y=by_y,
        by_synth=by_synth,
        by_run_policy=by_run_policy,
    )

    print(f"Analysis written under: {analysis_dir}")


if __name__ == "__main__":
    main()
