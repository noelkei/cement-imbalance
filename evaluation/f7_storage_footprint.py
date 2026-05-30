from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from data.utils import ROOT_PATH
from evaluation.f7_campaign_lineage import load_lineage_campaign_records
from evaluation.f7_campaign_spec import DEFAULT_SPEC_PATH, materialize_f7_campaign_spec


DEFAULT_PRIMARY_SPEC_PATH = Path(ROOT_PATH) / "config" / "f7_campaign_spec_v1.yaml"
DEFAULT_EXTENSION_SPEC_PATH = Path(ROOT_PATH) / "config" / "f7_campaign_extension1_v1.yaml"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() == "true"


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def _safe_bytes_payload(value: float | int | None) -> dict[str, float | int | None]:
    if value is None:
        return {
            "bytes": None,
            "kb": None,
            "mb": None,
            "gb": None,
            "gib": None,
        }
    raw = float(value)
    return {
        "bytes": int(round(raw)),
        "kb": round(raw / 1_000.0, 6),
        "mb": round(raw / 1_000_000.0, 6),
        "gb": round(raw / 1_000_000_000.0, 6),
        "gib": round(raw / float(1024 ** 3), 6),
    }


def _series_stat(series: pd.Series, fn: str) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    if fn == "mean":
        return float(numeric.mean())
    if fn == "median":
        return float(numeric.median())
    if fn == "min":
        return float(numeric.min())
    if fn == "max":
        return float(numeric.max())
    if fn == "p90":
        return float(numeric.quantile(0.90))
    if fn == "p95":
        return float(numeric.quantile(0.95))
    raise ValueError(f"Unsupported stat '{fn}'.")


def summarize_storage_rows(df: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_cols, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        row = {column: _json_safe_scalar(keys[idx]) for idx, column in enumerate(group_cols)}
        row["run_count"] = int(len(group))
        total_bytes = _series_stat(group["run_size_bytes"], "mean")
        row.update(
            {
                "total_size": _safe_bytes_payload(pd.to_numeric(group["run_size_bytes"], errors="coerce").sum()),
                "mean_size": _safe_bytes_payload(total_bytes),
                "median_size": _safe_bytes_payload(_series_stat(group["run_size_bytes"], "median")),
                "p90_size": _safe_bytes_payload(_series_stat(group["run_size_bytes"], "p90")),
                "p95_size": _safe_bytes_payload(_series_stat(group["run_size_bytes"], "p95")),
                "min_size": _safe_bytes_payload(_series_stat(group["run_size_bytes"], "min")),
                "max_size": _safe_bytes_payload(_series_stat(group["run_size_bytes"], "max")),
            }
        )
        rows.append(row)
    rows.sort(key=lambda item: tuple(str(item.get(column)) for column in group_cols))
    return rows


def project_storage_bytes(
    observed_runs: pd.DataFrame,
    expected_trials: pd.DataFrame,
) -> dict[str, Any]:
    if observed_runs.empty:
        raise ValueError("Observed runs dataframe is empty.")
    if expected_trials.empty:
        raise ValueError("Expected trials dataframe is empty.")
    group_cols = ["model_family", "flowpre_usage"]
    observed = observed_runs.copy()
    expected = expected_trials.copy()
    observed["flowpre_usage"] = observed["flowpre_usage"].map(_coerce_bool)
    expected["flowpre_usage"] = expected["flowpre_usage"].map(_coerce_bool)

    group_means = (
        observed.groupby(group_cols, dropna=False)["run_size_bytes"].mean().reset_index(name="mean_bytes")
    )
    family_means = observed.groupby("model_family", dropna=False)["run_size_bytes"].mean().to_dict()
    overall_mean = float(pd.to_numeric(observed["run_size_bytes"], errors="coerce").dropna().mean())

    expected_counts = (
        expected.groupby(group_cols, dropna=False).size().reset_index(name="expected_trial_count")
    )
    merged = expected_counts.merge(group_means, on=group_cols, how="left")
    fallbacks: list[dict[str, Any]] = []
    projected_rows: list[dict[str, Any]] = []
    total_bytes = 0.0

    for row in merged.to_dict("records"):
        mean_bytes = row.get("mean_bytes")
        fallback_kind = "group_mean"
        if mean_bytes is None or (isinstance(mean_bytes, float) and math.isnan(mean_bytes)):
            family_key = str(row["model_family"])
            mean_bytes = family_means.get(family_key)
            fallback_kind = "family_mean"
        if mean_bytes is None or (isinstance(mean_bytes, float) and math.isnan(mean_bytes)):
            mean_bytes = overall_mean
            fallback_kind = "overall_mean"
        projected_bytes = float(mean_bytes) * int(row["expected_trial_count"])
        total_bytes += projected_bytes
        if fallback_kind != "group_mean":
            fallbacks.append(
                {
                    "model_family": row["model_family"],
                    "flowpre_usage": bool(row["flowpre_usage"]),
                    "fallback_kind": fallback_kind,
                }
            )
        projected_rows.append(
            {
                "model_family": _json_safe_scalar(row["model_family"]),
                "flowpre_usage": bool(row["flowpre_usage"]),
                "expected_trial_count": int(row["expected_trial_count"]),
                "mean_size": _safe_bytes_payload(mean_bytes),
                "projected_total_size": _safe_bytes_payload(projected_bytes),
            }
        )

    return {
        "expected_trial_count": int(expected_counts["expected_trial_count"].sum()),
        "group_rows": projected_rows,
        "fallbacks": fallbacks,
        "projected_total_size": _safe_bytes_payload(total_bytes),
    }


def _run_dir_size_bytes(run_dir: Path) -> int:
    return sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())


def _observed_storage_rows(root_campaign_id: str) -> pd.DataFrame:
    records = load_lineage_campaign_records(root_campaign_id)
    frames: list[pd.DataFrame] = []
    for record in records:
        ledger = record.ledger.copy()
        required = {"campaign_id", "trial_id", "model_family", "flowpre_usage", "run_manifest_path", "execution_status", "campaign_valid_f7"}
        if not required.issubset(set(ledger.columns)):
            continue
        ledger = ledger[
            ledger["execution_status"].astype(str).eq("completed")
            & ledger["campaign_valid_f7"].map(_coerce_bool)
        ].copy()
        if ledger.empty:
            continue
        frames.append(ledger)
    if not frames:
        return pd.DataFrame()
    observed = pd.concat(frames, ignore_index=True)
    observed["run_manifest_path"] = observed["run_manifest_path"].map(str)
    observed["run_dir"] = observed["run_manifest_path"].map(lambda value: str(Path(value).resolve().parent))
    observed["run_size_bytes"] = observed["run_dir"].map(lambda value: _run_dir_size_bytes(Path(value)))
    observed["flowpre_usage"] = observed["flowpre_usage"].map(_coerce_bool)
    observed = observed.drop_duplicates(subset=["campaign_id", "trial_id"]).reset_index(drop=True)
    return observed


def _expected_trials_from_spec(spec_path: str | Path) -> pd.DataFrame:
    materialized = materialize_f7_campaign_spec(spec_path=spec_path, write_outputs=False)
    frame = pd.DataFrame(materialized.trials)
    if frame.empty:
        return frame
    frame["flowpre_usage"] = frame["flowpre_usage"].map(_coerce_bool)
    return frame[["model_family", "flowpre_usage"]].copy()


def build_storage_footprint_report(
    *,
    root_campaign_id: str,
    primary_spec_path: str | Path = DEFAULT_PRIMARY_SPEC_PATH,
    extension_spec_path: str | Path = DEFAULT_EXTENSION_SPEC_PATH,
) -> dict[str, Any]:
    observed = _observed_storage_rows(root_campaign_id)
    if observed.empty:
        raise FileNotFoundError(f"No completed valid F7 run rows found for lineage root '{root_campaign_id}'.")

    primary_expected = _expected_trials_from_spec(primary_spec_path)
    extension_expected = _expected_trials_from_spec(extension_spec_path)
    full_chain_expected = pd.concat(
        [primary_expected, extension_expected, extension_expected, extension_expected],
        ignore_index=True,
    )

    by_family = summarize_storage_rows(observed, ["model_family"])
    by_flowpre = summarize_storage_rows(observed, ["flowpre_usage"])
    by_family_and_flowpre = summarize_storage_rows(observed, ["model_family", "flowpre_usage"])
    by_campaign = summarize_storage_rows(observed, ["campaign_id"])
    observed_total = _safe_bytes_payload(pd.to_numeric(observed["run_size_bytes"], errors="coerce").sum())

    return {
        "root_campaign_id": root_campaign_id,
        "observed": {
            "run_count": int(len(observed)),
            "total_size": observed_total,
            "mean_size": _safe_bytes_payload(_series_stat(observed["run_size_bytes"], "mean")),
            "median_size": _safe_bytes_payload(_series_stat(observed["run_size_bytes"], "median")),
            "p90_size": _safe_bytes_payload(_series_stat(observed["run_size_bytes"], "p90")),
            "p95_size": _safe_bytes_payload(_series_stat(observed["run_size_bytes"], "p95")),
            "by_campaign_id": by_campaign,
            "by_model_family": by_family,
            "by_flowpre_usage": by_flowpre,
            "by_model_family_and_flowpre_usage": by_family_and_flowpre,
        },
        "projections": {
            "primary_17400": project_storage_bytes(observed, primary_expected),
            "extension_17400": project_storage_bytes(observed, extension_expected),
            "full_4x17400": project_storage_bytes(observed, full_chain_expected),
        },
    }
