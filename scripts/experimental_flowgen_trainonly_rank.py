from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
FLOWGEN_ROOT = ROOT / "outputs" / "models" / "experimental" / "train_only" / "flowgen"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "reports" / "experimental" / "train_only" / "flowgen_rankings"

ROUND1_ROOT = FLOWGEN_ROOT / "round1"
ROUND2_SUMMARY_ROOT = FLOWGEN_ROOT / "campaign_summaries" / "round2"
ROUND3_SUMMARY_ROOT = FLOWGEN_ROOT / "campaign_summaries" / "round3"
ROUND3_CONFIRM_SUMMARY_ROOT = FLOWGEN_ROOT / "campaign_summaries" / "round3_confirm"

BASE_DISPLAY = {
    "candidate_trainonly_1": "ct1",
    "candidate_trainonly_2": "ct2",
}

LENS_WEIGHTS = {
    "balanced_trainonly": {
        "x_w1_ta": 0.40,
        "ks_mean": 0.20,
        "y_w1_ta": 0.15,
        "xy_pearson_rel": 0.125,
        "xy_spearman_rel": 0.125,
    },
    "x_priority": {
        "x_w1_ta": 0.60,
        "ks_mean": 0.15,
        "y_w1_ta": 0.10,
        "xy_pearson_rel": 0.075,
        "xy_spearman_rel": 0.075,
    },
}

SHORTLIST_TOP_K = 4
EPS = 1.0e-12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final train-only FlowGen ranking across rounds 1/2/3 plus the "
            "round3 ct2 confirmation, oriented toward deciding whether to "
            "reseed or keep exploring."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the ranking artifacts will be written.",
    )
    return parser.parse_args()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_yaml_first(candidates: list[Path]) -> dict[str, Any] | None:
    for path in candidates:
        if not path.exists():
            continue
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON mapping at {path}")
    return loaded


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["status"])
            writer.writeheader()
        return path

    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                key: (_stable_json(value) if isinstance(value, (dict, list)) else value)
                for key, value in row.items()
            }
        )
    fieldnames = sorted({key for row in normalized for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _round1_policy_label(run_id: str) -> str:
    if "_round1_" in run_id:
        return run_id.split("_round1_", 1)[1].rsplit("_seed", 1)[0]
    return run_id


def _record_from_results(
    *,
    source_round: str,
    base_token: str,
    run_id: str,
    policy_id: str,
    results: dict[str, Any],
    results_path: Path,
) -> dict[str, Any] | None:
    val = results.get("val")
    train = results.get("train")
    if not isinstance(val, dict) or not isinstance(train, dict):
        return None

    val_realism = val.get("realism") or {}
    train_realism = train.get("realism") or {}
    x_val_ta = _as_float(((val_realism.get("x") or {}).get("w1_mean_trainaligned")))
    y_val_ta = _as_float(((val_realism.get("y") or {}).get("w1_mean_trainaligned")))
    ks_val = _as_float(((val_realism.get("overall") or {}).get("ks_mean")))
    xyp_val = _as_float(((val_realism.get("overall") or {}).get("xy_pearson_fro_rel")))
    xys_val = _as_float(((val_realism.get("overall") or {}).get("xy_spearman_fro_rel")))

    if any(value is None for value in (x_val_ta, y_val_ta, ks_val, xyp_val, xys_val)):
        return None

    x_train_ta = _as_float(((train_realism.get("x") or {}).get("w1_mean_trainaligned")))
    y_train_ta = _as_float(((train_realism.get("y") or {}).get("w1_mean_trainaligned")))
    ks_train = _as_float(((train_realism.get("overall") or {}).get("ks_mean")))
    xyp_train = _as_float(((train_realism.get("overall") or {}).get("xy_pearson_fro_rel")))
    xys_train = _as_float(((train_realism.get("overall") or {}).get("xy_spearman_fro_rel")))

    xy_mean_val = statistics.mean([xyp_val, xys_val])
    xy_mean_train = None
    if all(value is not None for value in (xyp_train, xys_train)):
        xy_mean_train = statistics.mean([xyp_train, xys_train])  # type: ignore[arg-type]

    return {
        "source_round": source_round,
        "base_token": base_token,
        "base_short": BASE_DISPLAY.get(base_token, base_token),
        "run_id": run_id,
        "policy_id": policy_id,
        "results_path": str(results_path),
        "x_w1_ta": x_val_ta,
        "ks_mean": ks_val,
        "y_w1_ta": y_val_ta,
        "xy_pearson_rel": xyp_val,
        "xy_spearman_rel": xys_val,
        "xy_mean_rel": xy_mean_val,
        "train_x_w1_ta": x_train_ta,
        "train_ks_mean": ks_train,
        "train_y_w1_ta": y_train_ta,
        "train_xy_pearson_rel": xyp_train,
        "train_xy_spearman_rel": xys_train,
        "train_xy_mean_rel": xy_mean_train,
        "gap_x_w1_ta": (x_val_ta - x_train_ta) if x_train_ta is not None else None,
        "gap_ks_mean": (ks_val - ks_train) if ks_train is not None else None,
        "gap_y_w1_ta": (y_val_ta - y_train_ta) if y_train_ta is not None else None,
        "gap_xy_mean_rel": (xy_mean_val - xy_mean_train) if xy_mean_train is not None else None,
    }


def _discover_round1_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_token in BASE_DISPLAY:
        base_dir = ROUND1_ROOT / base_token
        if not base_dir.exists():
            continue
        for run_dir in sorted(base_dir.glob("*")):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            results = _load_yaml_first([run_dir / "results.yaml", run_dir / f"{run_id}_results.yaml"])
            if results is None:
                continue
            row = _record_from_results(
                source_round="round1",
                base_token=base_token,
                run_id=run_id,
                policy_id=_round1_policy_label(run_id),
                results=results,
                results_path=run_dir / "results.yaml",
            )
            if row is not None:
                rows.append(row)
    return rows


def _latest_results_json(summary_root: Path) -> Path | None:
    files = sorted(summary_root.glob("*_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _discover_campaign_runs(summary_root: Path, source_round: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path | None]:
    results_path = _latest_results_json(summary_root)
    completed_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    if results_path is None:
        return completed_rows, failed_rows, None

    payload = _load_json(results_path)
    for run in payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        status = str(run.get("status", ""))
        if status == "completed":
            result_paths = run.get("result_paths") or {}
            candidate = Path(str(result_paths.get("results", "")))
            results = _load_yaml_first([candidate])
            if results is None:
                continue
            row = _record_from_results(
                source_round=source_round,
                base_token=str(run.get("base_token")),
                run_id=str(run.get("run_id")),
                policy_id=str(run.get("policy_id")),
                results=results,
                results_path=candidate,
            )
            if row is not None:
                completed_rows.append(row)
        elif status == "failed":
            failed_rows.append(
                {
                    "source_round": source_round,
                    "base_token": str(run.get("base_token")),
                    "run_id": str(run.get("run_id")),
                    "policy_id": str(run.get("policy_id")),
                    "error": str(run.get("error", "")),
                    "output_dir": str(run.get("output_dir", "")),
                }
            )
    return completed_rows, failed_rows, results_path


def _robust_zscores(values: list[float | None]) -> list[float | None]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return [None for _ in values]
    med = statistics.median(finite)
    mad = statistics.median([abs(v - med) for v in finite])
    if mad <= EPS:
        return [0.0 if v is not None else None for v in values]

    out: list[float | None] = []
    for value in values:
        if value is None or not math.isfinite(float(value)):
            out.append(None)
        else:
            z = 0.6745 * (float(value) - med) / mad
            out.append(max(-5.0, min(5.0, z)))
    return out


def _apply_scores(rows: list[dict[str, Any]]) -> None:
    metric_names = ["x_w1_ta", "ks_mean", "y_w1_ta", "xy_pearson_rel", "xy_spearman_rel"]
    zmap: dict[str, list[float | None]] = {
        metric: _robust_zscores([_as_float(row.get(metric)) for row in rows]) for metric in metric_names
    }
    for idx, row in enumerate(rows):
        for metric in metric_names:
            row[f"{metric}_rz"] = zmap[metric][idx]

        for lens_name, weights in LENS_WEIGHTS.items():
            score = 0.0
            total = 0.0
            for metric, weight in weights.items():
                zvalue = _as_float(row.get(f"{metric}_rz"))
                if zvalue is None:
                    continue
                score += float(weight) * (-zvalue)
                total += float(weight)
            row[f"{lens_name}_score"] = score / total if total > 0 else None


def _policy_family_rows(
    completed_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_counts: dict[str, int] = defaultdict(int)
    bases_seen: dict[str, set[str]] = defaultdict(set)

    for row in completed_rows:
        grouped[row["policy_id"]].append(row)
        bases_seen[row["policy_id"]].add(str(row["base_token"]))
    for row in failed_rows:
        failure_counts[str(row["policy_id"])] += 1
        bases_seen[str(row["policy_id"])].add(str(row["base_token"]))

    output: list[dict[str, Any]] = []
    for policy_id, rows in grouped.items():
        balanced_scores = [_as_float(row.get("balanced_trainonly_score")) for row in rows]
        x_scores = [_as_float(row.get("x_priority_score")) for row in rows]
        balanced_finite = [x for x in balanced_scores if x is not None]
        x_finite = [x for x in x_scores if x is not None]
        metrics = {
            "policy_id": policy_id,
            "completed_count": len(rows),
            "failed_count": failure_counts.get(policy_id, 0),
            "base_coverage": len(bases_seen.get(policy_id, set())),
            "bases": sorted(bases_seen.get(policy_id, set())),
            "best_run_id": max(rows, key=lambda row: _as_float(row.get("balanced_trainonly_score")) or -1e9)["run_id"],
            "balanced_mean_score": statistics.mean(balanced_finite) if balanced_finite else None,
            "balanced_max_score": max(balanced_finite) if balanced_finite else None,
            "balanced_std_score": statistics.pstdev(balanced_finite) if len(balanced_finite) >= 2 else 0.0,
            "x_priority_mean_score": statistics.mean(x_finite) if x_finite else None,
            "x_priority_max_score": max(x_finite) if x_finite else None,
            "x_priority_std_score": statistics.pstdev(x_finite) if len(x_finite) >= 2 else 0.0,
            "mean_x_w1_ta": statistics.mean([row["x_w1_ta"] for row in rows]),
            "mean_ks_mean": statistics.mean([row["ks_mean"] for row in rows]),
            "mean_y_w1_ta": statistics.mean([row["y_w1_ta"] for row in rows]),
            "mean_xy_mean_rel": statistics.mean([row["xy_mean_rel"] for row in rows]),
        }
        output.append(metrics)

    output.sort(
        key=lambda row: (
            -(_as_float(row.get("balanced_mean_score")) or -1e9),
            -(_as_float(row.get("x_priority_mean_score")) or -1e9),
            row["failed_count"],
            row["policy_id"],
        )
    )
    return output


def _round_progression_rows(completed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for base_token in BASE_DISPLAY:
        base_rows = [row for row in completed_rows if row["base_token"] == base_token]
        for source_round in ("round1", "round2", "round3", "round3_confirm"):
            rows = [row for row in base_rows if row["source_round"] == source_round]
            if not rows:
                continue
            best_balanced = max(rows, key=lambda row: _as_float(row.get("balanced_trainonly_score")) or -1e9)
            best_x = max(rows, key=lambda row: _as_float(row.get("x_priority_score")) or -1e9)
            output.append(
                {
                    "base_token": base_token,
                    "source_round": source_round,
                    "run_count": len(rows),
                    "best_balanced_run_id": best_balanced["run_id"],
                    "best_balanced_policy_id": best_balanced["policy_id"],
                    "best_balanced_score": best_balanced.get("balanced_trainonly_score"),
                    "best_x_priority_run_id": best_x["run_id"],
                    "best_x_priority_policy_id": best_x["policy_id"],
                    "best_x_priority_score": best_x.get("x_priority_score"),
                    "best_x_w1_ta": min(row["x_w1_ta"] for row in rows),
                    "best_ks_mean": min(row["ks_mean"] for row in rows),
                    "best_y_w1_ta": min(row["y_w1_ta"] for row in rows),
                    "best_xy_mean_rel": min(row["xy_mean_rel"] for row in rows),
                }
            )
    return output


def _markdown_summary(
    *,
    completed_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    round_rows: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    output_paths: dict[str, Path],
) -> str:
    best_balanced = max(completed_rows, key=lambda row: _as_float(row.get("balanced_trainonly_score")) or -1e9)
    best_x = max(completed_rows, key=lambda row: _as_float(row.get("x_priority_score")) or -1e9)

    lines: list[str] = []
    lines.append("# FlowGen Train-Only Final Ranking")
    lines.append("")
    lines.append(f"- completed comparable runs: `{len(completed_rows)}`")
    lines.append(f"- failed runs inventoried: `{len(failed_rows)}`")
    lines.append("- ranking lenses:")
    lines.append("  - `balanced_trainonly`: X trainaligned first, then KS, Y, and XY structure")
    lines.append("  - `x_priority`: extra emphasis on X trainaligned to reflect the core train-only goal")
    lines.append("")
    lines.append("## Best Runs")
    lines.append("")
    lines.append(
        f"- best balanced: `{best_balanced['run_id']}` "
        f"(base={best_balanced['base_short']}, policy={best_balanced['policy_id']}, "
        f"x={best_balanced['x_w1_ta']:.4f}, ks={best_balanced['ks_mean']:.4f}, "
        f"y={best_balanced['y_w1_ta']:.4f}, xyP={best_balanced['xy_pearson_rel']:.4f}, "
        f"xyS={best_balanced['xy_spearman_rel']:.4f})"
    )
    lines.append(
        f"- best x-priority: `{best_x['run_id']}` "
        f"(base={best_x['base_short']}, policy={best_x['policy_id']}, "
        f"x={best_x['x_w1_ta']:.4f}, ks={best_x['ks_mean']:.4f}, "
        f"y={best_x['y_w1_ta']:.4f}, xyP={best_x['xy_pearson_rel']:.4f}, "
        f"xyS={best_x['xy_spearman_rel']:.4f})"
    )
    lines.append("")
    lines.append("## Round Progression")
    lines.append("")
    for base_token in BASE_DISPLAY:
        base_rows = [row for row in round_rows if row["base_token"] == base_token]
        if not base_rows:
            continue
        lines.append(f"- `{BASE_DISPLAY[base_token]}`")
        for row in base_rows:
            lines.append(
                f"  - {row['source_round']}: best balanced `{row['best_balanced_policy_id']}` "
                f"(score={row['best_balanced_score']:.3f}), best x-priority `{row['best_x_priority_policy_id']}` "
                f"(score={row['best_x_priority_score']:.3f}), "
                f"best x={row['best_x_w1_ta']:.4f}, best ks={row['best_ks_mean']:.4f}"
            )
    lines.append("")
    lines.append("## Family Read")
    lines.append("")
    for row in family_rows[:8]:
        lines.append(
            f"- `{row['policy_id']}`: bases={','.join(row['bases'])}, completed={row['completed_count']}, "
            f"failed={row['failed_count']}, balanced_mean={row['balanced_mean_score']:.3f}, "
            f"x_priority_mean={row['x_priority_mean_score']:.3f}, "
            f"mean_x={row['mean_x_w1_ta']:.4f}, mean_ks={row['mean_ks_mean']:.4f}"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "The evidence points to **reseed now**, not to another broad family exploration. "
        "Round3 confirmed that the useful direction is the `T06` corridor "
        "(RMS norm + tight clipping, optionally with light KS), while the orthogonal rescue families "
        "did not unlock a new regime."
    )
    lines.append("")
    lines.append("Recommended reseed shortlist:")
    for idx, row in enumerate(shortlist, start=1):
        lines.append(
            f"{idx}. `{row['run_id']}` | base={row['base_short']} | policy={row['policy_id']} | "
            f"balanced={row['balanced_trainonly_score']:.3f} | x-priority={row['x_priority_score']:.3f} | "
            f"x={row['x_w1_ta']:.4f} | ks={row['ks_mean']:.4f}"
        )
    lines.append("")
    lines.append("Suggested interpretation:")
    lines.append("- `R3B1_t06_ksx_light` is the cleanest balanced candidate.")
    lines.append("- `R3A2_t06_clip125` is the strongest pure X-fitting candidate and now transferred to both bases.")
    lines.append("- `R3A1_t06_w1x120` is the best ct2 balanced alternative if you want explicit base diversity.")
    lines.append("")
    lines.append("Artifacts:")
    for label, path in output_paths.items():
        lines.append(f"- `{label}`: `{path}`")
    return "\n".join(lines) + "\n"


def _select_shortlist(
    ranked_rows: list[dict[str, Any]],
    x_priority_rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    shortlist: list[dict[str, Any]] = []
    seen: set[str] = set()

    def push(row: dict[str, Any] | None) -> None:
        if row is None:
            return
        run_id = str(row["run_id"])
        if run_id in seen:
            return
        seen.add(run_id)
        shortlist.append(row)

    push(ranked_rows[0] if ranked_rows else None)
    push(x_priority_rows[0] if x_priority_rows else None)

    ct2_balanced = next((row for row in ranked_rows if row["base_token"] == "candidate_trainonly_2"), None)
    ct2_x = next((row for row in x_priority_rows if row["base_token"] == "candidate_trainonly_2"), None)
    push(ct2_balanced)
    push(ct2_x)

    for row in ranked_rows:
        if len(shortlist) >= top_k:
            break
        push(row)

    return shortlist[:top_k]


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.resolve()
    output_dir = output_root / f"flowgen_trainonly_final_rank_{_utc_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    round1_rows = _discover_round1_runs()
    round2_rows, round2_failed, round2_results = _discover_campaign_runs(ROUND2_SUMMARY_ROOT, "round2")
    round3_rows, round3_failed, round3_results = _discover_campaign_runs(ROUND3_SUMMARY_ROOT, "round3")
    confirm_rows, confirm_failed, confirm_results = _discover_campaign_runs(ROUND3_CONFIRM_SUMMARY_ROOT, "round3_confirm")

    completed_rows = round1_rows + round2_rows + round3_rows + confirm_rows
    failed_rows = round2_failed + round3_failed + confirm_failed
    if not completed_rows:
        raise RuntimeError("No comparable completed train-only FlowGen runs were found.")

    _apply_scores(completed_rows)

    ranked_rows = sorted(
        completed_rows,
        key=lambda row: (
            -(_as_float(row.get("balanced_trainonly_score")) or -1e9),
            -(_as_float(row.get("x_priority_score")) or -1e9),
            row["run_id"],
        ),
    )

    for idx, row in enumerate(ranked_rows, start=1):
        row["rank_balanced_trainonly"] = idx
    x_priority_rows = sorted(
        completed_rows,
        key=lambda row: (
            -(_as_float(row.get("x_priority_score")) or -1e9),
            -(_as_float(row.get("balanced_trainonly_score")) or -1e9),
            row["run_id"],
        ),
    )
    x_rank_map = {row["run_id"]: idx for idx, row in enumerate(x_priority_rows, start=1)}
    for row in ranked_rows:
        row["rank_x_priority"] = x_rank_map[row["run_id"]]

    family_rows = _policy_family_rows(completed_rows, failed_rows)
    round_rows = _round_progression_rows(completed_rows)

    shortlist = _select_shortlist(ranked_rows, x_priority_rows, top_k=SHORTLIST_TOP_K)

    runs_csv = _write_csv(output_dir / "flowgen_trainonly_runs_ranked.csv", ranked_rows)
    families_csv = _write_csv(output_dir / "flowgen_trainonly_policy_families.csv", family_rows)
    failures_csv = _write_csv(output_dir / "flowgen_trainonly_failures.csv", failed_rows)
    rounds_csv = _write_csv(output_dir / "flowgen_trainonly_round_progression.csv", round_rows)

    shortlist_payload = {
        "recommended_reseed_now": True,
        "recommended_run_count": len(shortlist),
        "runs": [
            {
                "run_id": row["run_id"],
                "base_token": row["base_token"],
                "policy_id": row["policy_id"],
                "source_round": row["source_round"],
                "balanced_trainonly_score": row["balanced_trainonly_score"],
                "x_priority_score": row["x_priority_score"],
                "x_w1_ta": row["x_w1_ta"],
                "ks_mean": row["ks_mean"],
                "y_w1_ta": row["y_w1_ta"],
                "xy_mean_rel": row["xy_mean_rel"],
                "results_path": row["results_path"],
            }
            for row in shortlist
        ],
    }
    shortlist_json = _write_json(output_dir / "flowgen_trainonly_shortlist.json", shortlist_payload)

    output_paths = {
        "runs_csv": runs_csv,
        "families_csv": families_csv,
        "failures_csv": failures_csv,
        "rounds_csv": rounds_csv,
        "shortlist_json": shortlist_json,
    }
    summary_md = output_dir / "summary.md"
    summary_md.write_text(
        _markdown_summary(
            completed_rows=completed_rows,
            failed_rows=failed_rows,
            family_rows=family_rows,
            round_rows=round_rows,
            shortlist=shortlist,
            output_paths=output_paths,
        ),
        encoding="utf-8",
    )

    manifest = {
        "script": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "round_inputs": {
            "round2_results": str(round2_results) if round2_results else None,
            "round3_results": str(round3_results) if round3_results else None,
            "round3_confirm_results": str(confirm_results) if confirm_results else None,
        },
        "lens_weights": LENS_WEIGHTS,
        "completed_run_count": len(completed_rows),
        "failed_run_count": len(failed_rows),
        "shortlist_run_ids": [row["run_id"] for row in shortlist],
    }
    manifest_json = _write_json(output_dir / "analysis_manifest.json", manifest)

    print(f"Completed comparable runs: {len(completed_rows)}")
    print(f"Failed runs inventoried: {len(failed_rows)}")
    print(f"Ranked runs CSV: {runs_csv}")
    print(f"Policy families CSV: {families_csv}")
    print(f"Shortlist JSON: {shortlist_json}")
    print(f"Summary MD: {summary_md}")
    print(f"Manifest JSON: {manifest_json}")
    print("Recommended reseed shortlist:")
    for row in shortlist:
        print(
            f"  - {row['run_id']} | base={row['base_short']} | policy={row['policy_id']} | "
            f"balanced={row['balanced_trainonly_score']:.3f} | x-priority={row['x_priority_score']:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
