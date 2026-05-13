from __future__ import annotations

from typing import Optional

import pandas as pd


BLOCKED_SPLIT_ROLE = "test_holdout"


def _guard_split_role(split_role: str, allow_test_holdout: bool) -> None:
    if split_role == BLOCKED_SPLIT_ROLE and not allow_test_holdout:
        raise ValueError("test_holdout is blocked by default. Pass allow_test_holdout=True to opt in explicitly.")


def filter_metrics_table(
    metrics_long: pd.DataFrame,
    *,
    model_family: str | None = None,
    split_role: str = "val_selection",
    allow_test_holdout: bool = False,
    metric_group: str | None = None,
    metric_name: str | None = None,
    metric_scope: str | None = None,
    component: str | None = None,
    value_space: str | None = None,
) -> pd.DataFrame:
    _guard_split_role(split_role, allow_test_holdout)
    df = metrics_long.copy()
    df = df[df["split_role"] == split_role]
    if model_family is not None:
        df = df[df["model_family"] == model_family]
    if metric_group is not None:
        df = df[df["metric_group"] == metric_group]
    if metric_name is not None:
        df = df[df["metric_name"] == metric_name]
    if metric_scope is not None:
        df = df[df["metric_scope"] == metric_scope]
    if component is not None:
        df = df[df["component"] == component]
    if value_space is not None:
        df = df[df["value_space"] == value_space]
    return df.reset_index(drop=True)


def aggregate_metrics_by_seed(
    metrics_long: pd.DataFrame,
    *,
    split_role: str = "val_selection",
    allow_test_holdout: bool = False,
    model_family: str | None = None,
) -> pd.DataFrame:
    df = filter_metrics_table(
        metrics_long,
        model_family=model_family,
        split_role=split_role,
        allow_test_holdout=allow_test_holdout,
    )
    if df.empty:
        return df.copy()

    group_cols = [
        "variant_fingerprint",
        "model_family",
        "upstream_variant_fingerprint",
        "dataset_name",
        "dataset_manifest_path",
        "split_id",
        "split_manifest_path",
        "split",
        "split_role",
        "metric_group",
        "metric_name",
        "metric_scope",
        "component",
        "class_id",
        "target_name",
        "value_space",
    ]
    aggregated = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            metric_value_mean=("metric_value", "mean"),
            metric_value_std=("metric_value", "std"),
            metric_value_min=("metric_value", "min"),
            metric_value_max=("metric_value", "max"),
            n_obs_mean=("n_obs", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    aggregated["metric_value_std"] = aggregated["metric_value_std"].fillna(0.0)
    return aggregated


def compare_family_variants(
    metrics_long: pd.DataFrame,
    *,
    model_family: str,
    metric_group: str,
    metric_name: str,
    metric_scope: str = "overall",
    component: Optional[str] = None,
    value_space: Optional[str] = None,
    split_role: str = "val_selection",
    allow_test_holdout: bool = False,
    aggregate_seeds: bool = True,
    ascending: bool = True,
) -> pd.DataFrame:
    df = filter_metrics_table(
        metrics_long,
        model_family=model_family,
        split_role=split_role,
        allow_test_holdout=allow_test_holdout,
        metric_group=metric_group,
        metric_name=metric_name,
        metric_scope=metric_scope,
        component=component,
        value_space=value_space,
    )
    if aggregate_seeds:
        grouped = aggregate_metrics_by_seed(
            df,
            split_role=split_role,
            allow_test_holdout=allow_test_holdout,
            model_family=model_family,
        )
        sort_col = "metric_value_mean"
        return grouped.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    return df.sort_values("metric_value", ascending=ascending).reset_index(drop=True)
