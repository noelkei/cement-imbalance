from __future__ import annotations

import csv
import json
import math
import os
import shutil
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

FLOWGEN_ROOT = ROOT / "outputs" / "models" / "experimental" / "train_only" / "flowgen"
FINALIST_ROOT = ROOT / "outputs" / "models" / "experimental" / "train_only" / "flowgen_finalist"
REPORT_ROOT = ROOT / "outputs" / "reports" / "experimental" / "train_only" / "flowgen_finalist"

TRAIN_ONLY_POLICY = "train_only"
MODEL_FAMILY = "flowgen"
SPLIT_ID = "init_temporal_processed_v1"
LINE_ID = "experimental_train_only"
SELECTION_PHASE = "flowgen_trainonly_finalist_v1"
SOURCE_ID = "flowgen_trainonly__winner__init_temporal_processed_v1__v1"
DEFAULT_OFFICIAL_DATASET_NAME = "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"

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

EPS = 1.0e-12
RESEED_PANEL = (6769, 11863, 12979, 14143, 15427)

BASE_DISPLAY = {
    "candidate_trainonly_1": "ct1",
    "candidate_trainonly_2": "ct2",
}


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    base_token: str
    policy_id: str
    source_group: str
    source_run_id: str
    source_seed: int
    role: str
    note: str


FAMILY_SPECS: tuple[FamilySpec, ...] = (
    FamilySpec(
        family_id="ct1__R3B1_t06_ksx_light",
        base_token="candidate_trainonly_1",
        policy_id="R3B1_t06_ksx_light",
        source_group="round3",
        source_run_id="flowgen_trainonly_tpv1_ct1_round3_r3b1_t06_ksx_light_seed6769_v1",
        source_seed=6769,
        role="balanced_anchor",
        note=(
            "Balanced ct1 finalist candidate. It was the cleanest round3 compromise between X fit, KS control, "
            "Y realism, and XY structure."
        ),
    ),
    FamilySpec(
        family_id="ct1__R3A2_t06_clip125",
        base_token="candidate_trainonly_1",
        policy_id="R3A2_t06_clip125",
        source_group="round3",
        source_run_id="flowgen_trainonly_tpv1_ct1_round3_r3a2_t06_clip125_seed6769_v1",
        source_seed=6769,
        role="x_priority",
        note=(
            "Strongest ct1 X-priority family. It pushed the T06 corridor to the best X trainaligned regime without "
            "breaking the rest of the realism block."
        ),
    ),
    FamilySpec(
        family_id="ct2__R3A1_t06_w1x120",
        base_token="candidate_trainonly_2",
        policy_id="R3A1_t06_w1x120",
        source_group="round3",
        source_run_id="flowgen_trainonly_tpv1_ct2_round3_r3a1_t06_w1x120_seed6769_v1",
        source_seed=6769,
        role="ct2_balanced_anchor",
        note=(
            "Balanced ct2 family. It kept the second base alive with the best ct2-native T06-derived geometry."
        ),
    ),
    FamilySpec(
        family_id="ct2__R3A2_t06_clip125",
        base_token="candidate_trainonly_2",
        policy_id="R3A2_t06_clip125",
        source_group="round3_confirm",
        source_run_id="flowgen_trainonly_tpv1_ct2_round3confirm_r3a2_t06_clip125_seed6769_v1",
        source_seed=6769,
        role="ct2_transfer_confirm",
        note=(
            "Transfer confirmation that the clip125 corridor also works on ct2, making the family eligible for final reseed comparison."
        ),
    ),
)


@dataclass
class RunRecord:
    family_id: str
    base_token: str
    base_short: str
    policy_id: str
    role: str
    source_group: str
    run_id: str
    run_dir: str
    seed: int
    source_kind: str
    config_path: str
    results_path: str
    run_manifest_path: str
    paired_flowpre_source_id: str | None
    paired_flowpre_run_id: str | None
    paired_flowpre_seed: int | None
    x_w1_ta: float
    ks_mean: float
    y_w1_ta: float
    xy_pearson_rel: float
    xy_spearman_rel: float
    balanced_trainonly_score: float | None = None
    x_priority_score: float | None = None
    rank_balanced_trainonly: int | None = None
    rank_x_priority: int | None = None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


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


def _repo_rel_str(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    root_prefix = str(ROOT) + os.sep
    path_str = str(path)
    if path_str.startswith(root_prefix):
        return path_str[len(root_prefix) :]
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        try:
            return str(path.relative_to(ROOT))
        except Exception:
            return str(path)


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


def _find_run_dir(base_token: str, run_id: str, source_group: str) -> Path:
    run_dir = FLOWGEN_ROOT / source_group / base_token / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Expected run dir not found: {run_dir}")
    return run_dir


def _source_run_dir(spec: FamilySpec, seed: int) -> Path:
    if seed == spec.source_seed:
        return _find_run_dir(spec.base_token, spec.source_run_id, spec.source_group)
    base_short = BASE_DISPLAY[spec.base_token]
    run_id = f"flowgen_trainonly_tpv1_{base_short}_reseedfinal_{spec.policy_id.lower()}_seed{seed}_v1"
    return _find_run_dir(spec.base_token, run_id, "reseed_final")


def _required_artifact(run_dir: Path, run_id: str, kind: str) -> Path:
    if kind == "config":
        candidates = [run_dir / "config.yaml", run_dir / f"{run_id}.yaml"]
    elif kind == "results":
        candidates = [run_dir / "results.yaml", run_dir / f"{run_id}_results.yaml"]
    elif kind == "run_manifest":
        candidates = [run_dir / "run_manifest.json", run_dir / f"{run_id}_run_manifest.json"]
    else:
        raise ValueError(f"Unsupported artifact kind: {kind}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing required artifact '{kind}' under {run_dir}")


def _record_from_run(spec: FamilySpec, seed: int) -> RunRecord:
    run_dir = _source_run_dir(spec, seed)
    if seed == spec.source_seed:
        run_id = spec.source_run_id
        source_kind = "source"
    else:
        run_id = run_dir.name
        source_kind = "reseed"

    config_path = _required_artifact(run_dir, run_id, "config")
    results_path = _required_artifact(run_dir, run_id, "results")
    run_manifest_path = _required_artifact(run_dir, run_id, "run_manifest")
    results = _load_yaml(results_path)
    run_manifest = _load_json(run_manifest_path)

    val_realism = ((results.get("val") or {}).get("realism") or {})
    overall = val_realism.get("overall") or {}
    x = val_realism.get("x") or {}
    y = val_realism.get("y") or {}

    metrics = {
        "x_w1_ta": _as_float(x.get("w1_mean_trainaligned")),
        "ks_mean": _as_float(overall.get("ks_mean")),
        "y_w1_ta": _as_float(y.get("w1_mean_trainaligned")),
        "xy_pearson_rel": _as_float(overall.get("xy_pearson_fro_rel")),
        "xy_spearman_rel": _as_float(overall.get("xy_spearman_fro_rel")),
    }
    if any(value is None for value in metrics.values()):
        raise ValueError(f"Incomplete realism block in {results_path}")

    axes = run_manifest.get("run_level_axes") or {}
    return RunRecord(
        family_id=spec.family_id,
        base_token=spec.base_token,
        base_short=BASE_DISPLAY[spec.base_token],
        policy_id=spec.policy_id,
        role=spec.role,
        source_group=spec.source_group if source_kind == "source" else "reseed_final",
        run_id=run_id,
        run_dir=str(run_dir),
        seed=int(seed),
        source_kind=source_kind,
        config_path=str(config_path),
        results_path=str(results_path),
        run_manifest_path=str(run_manifest_path),
        paired_flowpre_source_id=None if axes.get("paired_flowpre_source_id") is None else str(axes.get("paired_flowpre_source_id")),
        paired_flowpre_run_id=None if axes.get("paired_flowpre_run_id") is None else str(axes.get("paired_flowpre_run_id")),
        paired_flowpre_seed=None if axes.get("paired_flowpre_seed") is None else int(axes.get("paired_flowpre_seed")),
        x_w1_ta=metrics["x_w1_ta"],  # type: ignore[arg-type]
        ks_mean=metrics["ks_mean"],  # type: ignore[arg-type]
        y_w1_ta=metrics["y_w1_ta"],  # type: ignore[arg-type]
        xy_pearson_rel=metrics["xy_pearson_rel"],  # type: ignore[arg-type]
        xy_spearman_rel=metrics["xy_spearman_rel"],  # type: ignore[arg-type]
    )


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


def _apply_run_scores(rows: list[RunRecord]) -> None:
    metric_names = ["x_w1_ta", "ks_mean", "y_w1_ta", "xy_pearson_rel", "xy_spearman_rel"]
    zmap: dict[str, list[float | None]] = {
        metric: _robust_zscores([getattr(row, metric) for row in rows]) for metric in metric_names
    }
    for idx, row in enumerate(rows):
        for lens_name, weights in LENS_WEIGHTS.items():
            score = 0.0
            total = 0.0
            for metric, weight in weights.items():
                zvalue = zmap[metric][idx]
                if zvalue is None:
                    continue
                score += float(weight) * (-zvalue)
                total += float(weight)
            setattr(row, f"{lens_name}_score", score / total if total > 0 else None)


def _rank_runs(rows: list[RunRecord]) -> None:
    balanced = sorted(
        rows,
        key=lambda row: (
            -(row.balanced_trainonly_score if row.balanced_trainonly_score is not None else -1e9),
            -(row.x_priority_score if row.x_priority_score is not None else -1e9),
            row.run_id,
        ),
    )
    for idx, row in enumerate(balanced, start=1):
        row.rank_balanced_trainonly = idx

    xprio = sorted(
        rows,
        key=lambda row: (
            -(row.x_priority_score if row.x_priority_score is not None else -1e9),
            -(row.balanced_trainonly_score if row.balanced_trainonly_score is not None else -1e9),
            row.run_id,
        ),
    )
    for idx, row in enumerate(xprio, start=1):
        row.rank_x_priority = idx


def _family_rows(rows: list[RunRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[RunRecord]] = {}
    for row in rows:
        grouped.setdefault(row.family_id, []).append(row)

    output: list[dict[str, Any]] = []
    for spec in FAMILY_SPECS:
        family_runs = grouped.get(spec.family_id, [])
        if len(family_runs) != 5:
            raise RuntimeError(f"Family {spec.family_id} does not have 5 runs available; found {len(family_runs)}")
        balanced_scores = [row.balanced_trainonly_score for row in family_runs if row.balanced_trainonly_score is not None]
        x_scores = [row.x_priority_score for row in family_runs if row.x_priority_score is not None]
        output.append(
            {
                "family_id": spec.family_id,
                "base_token": spec.base_token,
                "base_short": BASE_DISPLAY[spec.base_token],
                "policy_id": spec.policy_id,
                "selection_role": spec.role,
                "seed_panel": list(RESEED_PANEL),
                "run_ids": [row.run_id for row in family_runs],
                "balanced_trainonly_mean_score": statistics.mean(balanced_scores),
                "balanced_trainonly_std_score": statistics.pstdev(balanced_scores) if len(balanced_scores) >= 2 else 0.0,
                "balanced_trainonly_max_score": max(balanced_scores),
                "x_priority_mean_score": statistics.mean(x_scores),
                "x_priority_std_score": statistics.pstdev(x_scores) if len(x_scores) >= 2 else 0.0,
                "x_priority_max_score": max(x_scores),
                "mean_x_w1_ta": statistics.mean(row.x_w1_ta for row in family_runs),
                "std_x_w1_ta": statistics.pstdev(row.x_w1_ta for row in family_runs),
                "mean_ks_mean": statistics.mean(row.ks_mean for row in family_runs),
                "std_ks_mean": statistics.pstdev(row.ks_mean for row in family_runs),
                "mean_y_w1_ta": statistics.mean(row.y_w1_ta for row in family_runs),
                "std_y_w1_ta": statistics.pstdev(row.y_w1_ta for row in family_runs),
                "mean_xy_pearson_rel": statistics.mean(row.xy_pearson_rel for row in family_runs),
                "std_xy_pearson_rel": statistics.pstdev(row.xy_pearson_rel for row in family_runs),
                "mean_xy_spearman_rel": statistics.mean(row.xy_spearman_rel for row in family_runs),
                "std_xy_spearman_rel": statistics.pstdev(row.xy_spearman_rel for row in family_runs),
                "best_run_id_by_balanced": max(
                    family_runs,
                    key=lambda row: (
                        row.balanced_trainonly_score if row.balanced_trainonly_score is not None else -1e9,
                        row.x_priority_score if row.x_priority_score is not None else -1e9,
                    ),
                ).run_id,
            }
        )

    output.sort(
        key=lambda row: (
            -(row["balanced_trainonly_mean_score"] or -1e9),
            -(row["x_priority_mean_score"] or -1e9),
            row["balanced_trainonly_std_score"],
            row["family_id"],
        )
    )
    for idx, row in enumerate(output, start=1):
        row["family_rank_balanced_trainonly"] = idx
    xsorted = sorted(
        output,
        key=lambda row: (
            -(row["x_priority_mean_score"] or -1e9),
            -(row["balanced_trainonly_mean_score"] or -1e9),
            row["x_priority_std_score"],
            row["family_id"],
        )
    )
    xrank = {row["family_id"]: idx for idx, row in enumerate(xsorted, start=1)}
    for row in output:
        row["family_rank_x_priority"] = xrank[row["family_id"]]
    return output


def _winner_family_rows(rows: list[RunRecord], family_id: str) -> list[RunRecord]:
    out = [row for row in rows if row.family_id == family_id]
    out.sort(
        key=lambda row: (
            -(row.balanced_trainonly_score if row.balanced_trainonly_score is not None else -1e9),
            -(row.x_priority_score if row.x_priority_score is not None else -1e9),
            row.run_id,
        )
    )
    return out


def _family_seed_selection_rows(family_rows: list[RunRecord]) -> list[dict[str, Any]]:
    seeds_by_balanced = sorted(
        family_rows,
        key=lambda row: (
            -(row.balanced_trainonly_score if row.balanced_trainonly_score is not None else -1e9),
            -(row.x_priority_score if row.x_priority_score is not None else -1e9),
            row.run_id,
        ),
    )
    rank_balanced = {row.run_id: idx for idx, row in enumerate(seeds_by_balanced, start=1)}

    seeds_by_x = sorted(
        family_rows,
        key=lambda row: (
            -(row.x_priority_score if row.x_priority_score is not None else -1e9),
            -(row.balanced_trainonly_score if row.balanced_trainonly_score is not None else -1e9),
            row.run_id,
        ),
    )
    rank_x = {row.run_id: idx for idx, row in enumerate(seeds_by_x, start=1)}

    by_x_metric = sorted(family_rows, key=lambda row: (row.x_w1_ta, row.ks_mean, row.y_w1_ta, row.run_id))
    rank_x_metric = {row.run_id: idx for idx, row in enumerate(by_x_metric, start=1)}

    by_ks = sorted(family_rows, key=lambda row: (row.ks_mean, row.x_w1_ta, row.y_w1_ta, row.run_id))
    rank_ks = {row.run_id: idx for idx, row in enumerate(by_ks, start=1)}

    by_y = sorted(family_rows, key=lambda row: (row.y_w1_ta, row.x_w1_ta, row.ks_mean, row.run_id))
    rank_y = {row.run_id: idx for idx, row in enumerate(by_y, start=1)}

    by_xyp = sorted(family_rows, key=lambda row: (row.xy_pearson_rel, row.xy_spearman_rel, row.run_id))
    rank_xyp = {row.run_id: idx for idx, row in enumerate(by_xyp, start=1)}

    by_xys = sorted(family_rows, key=lambda row: (row.xy_spearman_rel, row.xy_pearson_rel, row.run_id))
    rank_xys = {row.run_id: idx for idx, row in enumerate(by_xys, start=1)}

    output: list[dict[str, Any]] = []
    for row in seeds_by_balanced:
        output.append(
            {
                "family_id": row.family_id,
                "run_id": row.run_id,
                "seed": row.seed,
                "source_kind": row.source_kind,
                "base_token": row.base_token,
                "policy_id": row.policy_id,
                "balanced_trainonly_score": row.balanced_trainonly_score,
                "x_priority_score": row.x_priority_score,
                "x_w1_ta": row.x_w1_ta,
                "ks_mean": row.ks_mean,
                "y_w1_ta": row.y_w1_ta,
                "xy_pearson_rel": row.xy_pearson_rel,
                "xy_spearman_rel": row.xy_spearman_rel,
                "rank_balanced_trainonly_within_family": rank_balanced[row.run_id],
                "rank_x_priority_within_family": rank_x[row.run_id],
                "rank_x_w1_ta_within_family": rank_x_metric[row.run_id],
                "rank_ks_mean_within_family": rank_ks[row.run_id],
                "rank_y_w1_ta_within_family": rank_y[row.run_id],
                "rank_xy_pearson_rel_within_family": rank_xyp[row.run_id],
                "rank_xy_spearman_rel_within_family": rank_xys[row.run_id],
                "results_path": row.results_path,
            }
        )
    return output


def _winner_and_runner_up(family_runs: list[RunRecord]) -> tuple[RunRecord, RunRecord]:
    ordered = sorted(
        family_runs,
        key=lambda row: (
            -(row.balanced_trainonly_score if row.balanced_trainonly_score is not None else -1e9),
            -(row.x_priority_score if row.x_priority_score is not None else -1e9),
            row.run_id,
        ),
    )
    return ordered[0], ordered[1]


def _relative_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    rel_target = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel_target)


def _maybe_alias(src_dir: Path, src_name: str, dst_dir: Path, dst_name: str | None = None) -> None:
    dst_name = src_name if dst_name is None else dst_name
    target = src_dir / src_name
    if target.exists():
        _relative_symlink(target, dst_dir / dst_name)


def _materialize_finalist_run_dir(final_root: Path, winner: RunRecord) -> Path:
    source_run_dir = Path(winner.run_dir)
    finalist_run_dir = final_root / winner.run_id
    finalist_run_dir.mkdir(parents=True, exist_ok=True)

    _relative_symlink(source_run_dir, finalist_run_dir / "selected_run")

    alias_pairs = [
        ("config.yaml", "config.yaml"),
        ("results.yaml", "results.yaml"),
        ("metrics_long.csv", "metrics_long.csv"),
        ("run_manifest.json", "run_manifest.json"),
        ("checkpoint.pt", "checkpoint.pt"),
        ("run.log", "run.log"),
        (f"{winner.run_id}.yaml", f"{winner.run_id}.yaml"),
        (f"{winner.run_id}_results.yaml", f"{winner.run_id}_results.yaml"),
        (f"{winner.run_id}_metrics_long.csv", f"{winner.run_id}_metrics_long.csv"),
        (f"{winner.run_id}_run_manifest.json", f"{winner.run_id}_run_manifest.json"),
        (f"{winner.run_id}.pt", f"{winner.run_id}.pt"),
        (f"{winner.run_id}.log", f"{winner.run_id}.log"),
        (f"{winner.run_id}_finetuned.pt", f"{winner.run_id}_finetuned.pt"),
        (f"{winner.run_id}_phase1.pt", f"{winner.run_id}_phase1.pt"),
    ]
    for src_name, dst_name in alias_pairs:
        _maybe_alias(source_run_dir, src_name, finalist_run_dir, dst_name)

    return finalist_run_dir


def _root_readme(
    *,
    winner: RunRecord,
    winner_family: dict[str, Any],
    family_selection_csv: Path,
    selection_manifest_path: Path,
    finalist_run_dir: Path,
) -> str:
    return (
        "# FlowGen Train-Only Finalist\n\n"
        "Esta carpeta materializa el cierre vigente de la rama experimental `train_only` de `FlowGen`.\n\n"
        "Importante:\n"
        "- vive bajo `outputs/models/experimental/train_only/`;\n"
        "- no sustituye al winner `official/` del proyecto;\n"
        "- representa el **winner local** de la línea `train_only`, pensado para uso downstream experimental.\n\n"
        "Estado vigente:\n\n"
        f"- familia ganadora post-reseed: `{winner_family['family_id']}`\n"
        f"- policy ganadora: `{winner.policy_id}`\n"
        f"- base de trabajo ganadora: `{winner.base_token}`\n"
        f"- seed final elegida dentro de esa familia: `{winner.seed}`\n"
        f"- run final promovida: `{winner.run_id}`\n\n"
        "Qué contiene:\n\n"
        f"- la run final materializada por symlink en `{finalist_run_dir.name}/`\n"
        "- `RATIONALE.md` con la justificación corta de la selección final\n"
        f"- `{family_selection_csv.name}` con la comparación programática de las `5` seeds de la familia ganadora\n"
        f"- `{selection_manifest_path.name}` como resumen machine-readable del cierre\n\n"
        "Interpretación operativa:\n\n"
        "- `outputs/models/experimental/train_only/flowgen/` queda como superficie histórica de exploración, confirmación y reseed\n"
        "- `outputs/reports/experimental/train_only/flowgen_finalist/` guarda la capa de ranking / agregación usada para este cierre\n"
        "- `outputs/models/experimental/train_only/flowgen_finalist/` pasa a ser el punto operativo de materialización del winner local de `FlowGen train_only`\n\n"
        "Nota metodológica importante:\n\n"
        "- en esta rama, la clave `val` de `results.yaml` es una pseudo-validación derivada de `train`\n"
        "- por tanto, este finalista es útil como cierre **experimental local** y como input downstream, pero no debe confundirse con un winner temporal oficial del proyecto\n"
    )


def _root_rationale(
    *,
    winner_family: dict[str, Any],
    winner: RunRecord,
    runner_up: RunRecord,
    family_selection_rows: list[dict[str, Any]],
) -> str:
    rank_map = {row["run_id"]: row for row in family_selection_rows}
    winner_row = rank_map[winner.run_id]
    runner_row = rank_map[runner_up.run_id]
    return (
        "# RATIONALE\n\n"
        "Esta carpeta no representa el winner canónico global del proyecto, sino el **finalista experimental único y vigente** de `FlowGen train-only`.\n\n"
        "## Secuencia de cierre\n\n"
        "1. se cerró la exploración `train_only` de `FlowGen` en varias rondas;\n"
        "2. se eligieron `4` cfg/familias finales para reseed por la misma lógica de ranking `train_only` usada en la shortlist previa;\n"
        "3. se ejecutó un panel de `5` seeds por familia (`6769 + 4` seeds nuevas);\n"
        "4. la comparación family-level se hizo promediando las lentes de ranking sobre las `5` seeds de cada familia;\n"
        "5. dentro de la familia ganadora se eligió una única seed representante para promoción local final.\n\n"
        "## Familia ganadora final\n\n"
        f"- familia/cfg ganadora: `{winner_family['family_id']}`\n"
        f"- base: `{winner_family['base_token']}`\n"
        f"- policy: `{winner_family['policy_id']}`\n"
        f"- panel comparado: `{', '.join(str(seed) for seed in winner_family['seed_panel'])}`\n"
        f"- score medio `balanced_trainonly`: `{winner_family['balanced_trainonly_mean_score']:.6f}`\n"
        f"- score medio `x_priority`: `{winner_family['x_priority_mean_score']:.6f}`\n\n"
        "## Por qué ganó esta familia\n\n"
        "Se eligió por la misma filosofía usada para llevar las cfgs al reseed final:\n"
        "- mantener la lente `balanced_trainonly` como criterio principal de cierre local;\n"
        "- usar `x_priority` como contraste secundario porque el objetivo real de la rama sigue siendo empujar `X` sin romper el resto;\n"
        "- pedir además estabilidad entre seeds y no solo un pico aislado.\n\n"
        "Lectura agregada de la familia ganadora:\n"
        f"- `balanced_trainonly_mean_score = {winner_family['balanced_trainonly_mean_score']:.6f}` (`rank #{winner_family['family_rank_balanced_trainonly']}`)\n"
        f"- `x_priority_mean_score = {winner_family['x_priority_mean_score']:.6f}` (`rank #{winner_family['family_rank_x_priority']}`)\n"
        f"- `mean_x_w1_ta = {winner_family['mean_x_w1_ta']:.6f}`\n"
        f"- `mean_ks_mean = {winner_family['mean_ks_mean']:.6f}`\n"
        f"- `mean_y_w1_ta = {winner_family['mean_y_w1_ta']:.6f}`\n"
        f"- `mean_xy_pearson_rel = {winner_family['mean_xy_pearson_rel']:.6f}`\n"
        f"- `mean_xy_spearman_rel = {winner_family['mean_xy_spearman_rel']:.6f}`\n\n"
        "## Seed final elegida\n\n"
        f"- seed elegida: `{winner.seed}`\n"
        f"- run final: `{winner.run_id}`\n\n"
        "## Por qué se eligió esta seed\n\n"
        "La selección no se hizo por ser reseed o por ser la original, sino por perfil de cierre dentro de la familia ganadora.\n\n"
        f"Dentro de las `5` seeds de `{winner_family['family_id']}`, la seed `{winner.seed}` es:\n"
        f"- `#{winner_row['rank_balanced_trainonly_within_family']}` en `balanced_trainonly`\n"
        f"- `#{winner_row['rank_x_priority_within_family']}` en `x_priority`\n"
        f"- `#{winner_row['rank_x_w1_ta_within_family']}` en `X w1_mean_trainaligned`\n"
        f"- `#{winner_row['rank_ks_mean_within_family']}` en `KS mean`\n"
        f"- `#{winner_row['rank_y_w1_ta_within_family']}` en `Y w1_mean_trainaligned`\n"
        f"- `#{winner_row['rank_xy_pearson_rel_within_family']}` en `XY pearson rel`\n"
        f"- `#{winner_row['rank_xy_spearman_rel_within_family']}` en `XY spearman rel`\n\n"
        "Interpretación:\n"
        f"- `{runner_up.seed}` (`{runner_up.run_id}`) es el challenger más fuerte; de hecho queda `#{runner_row['rank_balanced_trainonly_within_family']}` en `balanced_trainonly` y `#{runner_row['rank_x_priority_within_family']}` en `x_priority`\n"
        f"- aun así, `{winner.seed}` se promueve porque es la mejor seed global dentro de la familia según la lente principal de cierre local (`balanced_trainonly`) y además sigue saliendo fuerte en la lente `x_priority`\n"
        "- el cierre busca una seed representativa y fuerte, no una seed extrema en una sola cara del problema\n\n"
        "## Rol semántico vigente\n\n"
        f"`{winner.run_id}` es el winner local final de `FlowGen train-only`.\n\n"
        "La siguiente fase activa ya no es reseed ni exploración adicional de `FlowGen train-only`, sino su uso downstream experimental cuando corresponda.\n"
    )


def _winner_promotion_manifest(
    *,
    winner: RunRecord,
    winner_family: dict[str, Any],
    runner_up: RunRecord,
    family_selection_csv: Path,
    selection_manifest_path: Path,
    finalist_run_dir: Path,
) -> dict[str, Any]:
    run_manifest = _load_json(Path(winner.run_manifest_path))
    return {
        "line": LINE_ID,
        "model_family": MODEL_FAMILY,
        "selection_phase": SELECTION_PHASE,
        "selection_role": "flowgen_trainonly_unique_final_winner",
        "selection_policy": "promote the seed with the strongest balanced_trainonly profile inside the winning train-only family, using x_priority as secondary tie-break support",
        "selection_reason": (
            f"best final seed inside winning family {winner_family['family_id']}: strongest balanced_trainonly closeout inside the family, "
            f"while remaining high on x_priority and preserving the T06 clip125 regime that won the family-level comparison."
        ),
        "selection_family_cfg_id": winner_family["family_id"],
        "selection_family_policy_id": winner.policy_id,
        "selection_family_seed_set": list(RESEED_PANEL),
        "selection_seed": int(winner.seed),
        "selection_seed_comparison_path": str(family_selection_csv.relative_to(ROOT)),
        "selection_summary_path": str(selection_manifest_path.relative_to(ROOT)),
        "selection_winner_balanced_trainonly_score": winner.balanced_trainonly_score,
        "selection_winner_x_priority_score": winner.x_priority_score,
        "selection_winner_rank_balanced_trainonly_within_family": 1,
        "selection_winner_run_id": winner.run_id,
        "selection_runner_up_seed": int(runner_up.seed),
        "selection_runner_up_run_id": runner_up.run_id,
        "branch_id": winner.base_token,
        "monitoring_policy": TRAIN_ONLY_POLICY,
        "monitoring_note": "The results key 'val' is a train-derived pseudo-validation surface and not the official temporal validation split.",
        "paired_flowpre_source_id": winner.paired_flowpre_source_id,
        "paired_flowpre_run_id": winner.paired_flowpre_run_id,
        "raw_bundle_manifest_path": _repo_rel_str(run_manifest.get("dataset_manifest_path")),
        "split_id": SPLIT_ID,
        "source_id": SOURCE_ID,
        "historical_support_only": False,
        "is_unique_finalist": True,
        "source_metrics_long_path": str((finalist_run_dir / "metrics_long.csv").relative_to(ROOT)),
        "source_run_manifest": str((finalist_run_dir / "run_manifest.json").relative_to(ROOT)),
    }


def _selection_manifest(
    *,
    report_dir: Path,
    winner_family: dict[str, Any],
    winner: RunRecord,
    runner_up: RunRecord,
    family_selection_csv: Path,
    finalist_run_dir: Path,
) -> dict[str, Any]:
    run_manifest = _load_json(Path(winner.run_manifest_path))
    return {
        "model_family": MODEL_FAMILY,
        "selection_phase": SELECTION_PHASE,
        "selection_role": "unique_trainonly_finalist",
        "selection_status": "closed",
        "line": LINE_ID,
        "monitoring_policy": TRAIN_ONLY_POLICY,
        "monitoring_note": "The results key 'val' is a train-derived pseudo-validation surface and not the official temporal validation split.",
        "family_winner_cfg_id": winner_family["family_id"],
        "family_policy_id": winner.policy_id,
        "family_base_token": winner.base_token,
        "family_seed_set": list(RESEED_PANEL),
        "family_run_ids": winner_family["run_ids"],
        "family_selection_role": winner_family["selection_role"],
        "family_selection_note": next(spec.note for spec in FAMILY_SPECS if spec.family_id == winner_family["family_id"]),
        "family_metrics": {
            "balanced_trainonly_mean_score": winner_family["balanced_trainonly_mean_score"],
            "balanced_trainonly_std_score": winner_family["balanced_trainonly_std_score"],
            "x_priority_mean_score": winner_family["x_priority_mean_score"],
            "x_priority_std_score": winner_family["x_priority_std_score"],
            "mean_x_w1_ta": winner_family["mean_x_w1_ta"],
            "mean_ks_mean": winner_family["mean_ks_mean"],
            "mean_y_w1_ta": winner_family["mean_y_w1_ta"],
            "mean_xy_pearson_rel": winner_family["mean_xy_pearson_rel"],
            "mean_xy_spearman_rel": winner_family["mean_xy_spearman_rel"],
        },
        "family_ranks": {
            "balanced_trainonly": winner_family["family_rank_balanced_trainonly"],
            "x_priority": winner_family["family_rank_x_priority"],
        },
        "selection_run_id": winner.run_id,
        "selection_seed": int(winner.seed),
        "selection_reason": (
            "Selected as the best final seed inside the winning train-only family because it is the strongest "
            "balanced_trainonly closeout while staying high on x_priority and preserving the clip125 T06 regime "
            "that dominated the family-level reseed comparison."
        ),
        "selection_lenses": ["balanced_trainonly", "x_priority"],
        "selection_metrics": {
            "balanced_trainonly_score": winner.balanced_trainonly_score,
            "x_priority_score": winner.x_priority_score,
            "x_w1_ta": winner.x_w1_ta,
            "ks_mean": winner.ks_mean,
            "y_w1_ta": winner.y_w1_ta,
            "xy_pearson_rel": winner.xy_pearson_rel,
            "xy_spearman_rel": winner.xy_spearman_rel,
        },
        "runner_up_seed": int(runner_up.seed),
        "runner_up_run_id": runner_up.run_id,
        "runner_up_reason_not_selected": (
            "Strong challenger inside the family, but still behind the selected seed on the main balanced_trainonly lens."
        ),
        "branch_id": winner.base_token,
        "paired_flowpre_source_id": winner.paired_flowpre_source_id,
        "paired_flowpre_run_id": winner.paired_flowpre_run_id,
        "paired_flowpre_seed": winner.paired_flowpre_seed,
        "cleaning_policy_id": "trainfit_overlap_cap1pct_holdoutflag_v1",
        "split_id": SPLIT_ID,
        "source_id": SOURCE_ID,
        "historical_support_only": False,
        "raw_bundle_manifest_path": _repo_rel_str(run_manifest.get("dataset_manifest_path")),
        "selection_family_seed_comparison_path": str(family_selection_csv.relative_to(ROOT)),
        "selection_report_dir": str(report_dir.relative_to(ROOT)),
        "selection_report_summary_md": str((report_dir / "summary.md").relative_to(ROOT)),
        "selection_report_manifest_json": str((report_dir / "analysis_manifest.json").relative_to(ROOT)),
        "materialized_finalist_root": str(FINALIST_ROOT.relative_to(ROOT)),
        "materialized_run_dir": str(finalist_run_dir.relative_to(ROOT)),
    }


def _summary_md(
    *,
    family_rows: list[dict[str, Any]],
    winner_family: dict[str, Any],
    winner: RunRecord,
    runner_up: RunRecord,
    family_selection_csv: Path,
    selection_manifest_path: Path,
    finalist_run_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# FlowGen Train-Only Finalist Selection")
    lines.append("")
    lines.append(f"- compared families: `{len(family_rows)}`")
    lines.append(f"- compared seeds per family: `{len(RESEED_PANEL)}`")
    lines.append("- family selection rule: mean `balanced_trainonly` score across the full 5-seed panel")
    lines.append("- tie-breakers: mean `x_priority`, then lower family score dispersion")
    lines.append("")
    lines.append("## Family Ranking")
    lines.append("")
    for row in family_rows:
        lines.append(
            f"- `{row['family_id']}` | rank_balanced=`#{row['family_rank_balanced_trainonly']}` | "
            f"rank_x=`#{row['family_rank_x_priority']}` | balanced_mean={row['balanced_trainonly_mean_score']:.3f} | "
            f"x_mean={row['x_priority_mean_score']:.3f} | mean_x={row['mean_x_w1_ta']:.4f} | "
            f"mean_ks={row['mean_ks_mean']:.4f} | mean_y={row['mean_y_w1_ta']:.4f}"
        )
    lines.append("")
    lines.append("## Winner")
    lines.append("")
    lines.append(
        f"- winning family: `{winner_family['family_id']}` (policy `{winner.policy_id}`, base `{winner.base_token}`)"
    )
    lines.append(
        f"- promoted seed: `{winner.seed}` -> `{winner.run_id}`"
    )
    lines.append(
        f"- runner-up seed inside winning family: `{runner_up.seed}` -> `{runner_up.run_id}`"
    )
    lines.append("")
    lines.append("## Materialized Artifacts")
    lines.append("")
    lines.append(f"- finalist root: `{FINALIST_ROOT}`")
    lines.append(f"- winner dir: `{finalist_run_dir}`")
    lines.append(f"- family seed comparison CSV: `{family_selection_csv}`")
    lines.append(f"- selection manifest: `{selection_manifest_path}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    report_dir = REPORT_ROOT / f"flowgen_trainonly_finalist_{_utc_stamp()}"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[RunRecord] = []
    for spec in FAMILY_SPECS:
        for seed in RESEED_PANEL:
            rows.append(_record_from_run(spec, seed))

    _apply_run_scores(rows)
    _rank_runs(rows)

    family_rows = _family_rows(rows)
    winner_family = family_rows[0]
    winner_family_runs = _winner_family_rows(rows, winner_family["family_id"])
    winner, runner_up = _winner_and_runner_up(winner_family_runs)
    family_selection_rows = _family_seed_selection_rows(winner_family_runs)

    runs_ranked_csv = _write_csv(report_dir / "flowgen_trainonly_reseed_runs_ranked.csv", [asdict(row) for row in rows])
    families_ranked_csv = _write_csv(report_dir / "flowgen_trainonly_reseed_families_ranked.csv", family_rows)
    family_selection_csv = _write_csv(
        report_dir / f"{winner.base_short}_{winner.policy_id.lower()}_family_seed_selection.csv",
        family_selection_rows,
    )

    if FINALIST_ROOT.exists():
        shutil.rmtree(FINALIST_ROOT)
    FINALIST_ROOT.mkdir(parents=True, exist_ok=True)

    finalist_run_dir = _materialize_finalist_run_dir(FINALIST_ROOT, winner)

    selection_manifest_path = FINALIST_ROOT / "flowgen_trainonly_final_selection_manifest.json"
    selection_manifest = _selection_manifest(
        report_dir=report_dir,
        winner_family=winner_family,
        winner=winner,
        runner_up=runner_up,
        family_selection_csv=family_selection_csv,
        finalist_run_dir=finalist_run_dir,
    )
    _write_json(selection_manifest_path, selection_manifest)

    promotion_manifest_payload = _winner_promotion_manifest(
        winner=winner,
        winner_family=winner_family,
        runner_up=runner_up,
        family_selection_csv=family_selection_csv,
        selection_manifest_path=selection_manifest_path,
        finalist_run_dir=finalist_run_dir,
    )
    versioned_promotion_path = finalist_run_dir / f"{winner.run_id}_promotion_manifest.json"
    _write_json(versioned_promotion_path, promotion_manifest_payload)
    _relative_symlink(versioned_promotion_path, finalist_run_dir / "promotion_manifest.json")

    finalist_family_selection_csv = FINALIST_ROOT / family_selection_csv.name
    shutil.copy2(family_selection_csv, finalist_family_selection_csv)

    readme_path = FINALIST_ROOT / "README.md"
    rationale_path = FINALIST_ROOT / "RATIONALE.md"
    readme_path.write_text(
        _root_readme(
            winner=winner,
            winner_family=winner_family,
            family_selection_csv=finalist_family_selection_csv,
            selection_manifest_path=selection_manifest_path,
            finalist_run_dir=finalist_run_dir,
        ),
        encoding="utf-8",
    )
    rationale_path.write_text(
        _root_rationale(
            winner_family=winner_family,
            winner=winner,
            runner_up=runner_up,
            family_selection_rows=family_selection_rows,
        ),
        encoding="utf-8",
    )

    summary_md_path = report_dir / "summary.md"
    summary_md_path.write_text(
        _summary_md(
            family_rows=family_rows,
            winner_family=winner_family,
            winner=winner,
            runner_up=runner_up,
            family_selection_csv=family_selection_csv,
            selection_manifest_path=selection_manifest_path,
            finalist_run_dir=finalist_run_dir,
        ),
        encoding="utf-8",
    )

    analysis_manifest = {
        "script": str(Path(__file__).resolve()),
        "selection_phase": SELECTION_PHASE,
        "line": LINE_ID,
        "model_family": MODEL_FAMILY,
        "monitoring_policy": TRAIN_ONLY_POLICY,
        "lens_weights": LENS_WEIGHTS,
        "reseed_panel": list(RESEED_PANEL),
        "family_specs": [asdict(spec) for spec in FAMILY_SPECS],
        "winner_family_id": winner_family["family_id"],
        "winner_run_id": winner.run_id,
        "winner_seed": int(winner.seed),
        "runner_up_run_id": runner_up.run_id,
        "runner_up_seed": int(runner_up.seed),
        "artifacts": {
            "runs_ranked_csv": str(runs_ranked_csv),
            "families_ranked_csv": str(families_ranked_csv),
            "winner_family_seed_selection_csv": str(family_selection_csv),
            "finalist_root": str(FINALIST_ROOT),
            "selection_manifest": str(selection_manifest_path),
        },
    }
    analysis_manifest_path = _write_json(report_dir / "analysis_manifest.json", analysis_manifest)

    print(f"Winner family: {winner_family['family_id']}")
    print(f"Winner seed: {winner.seed}")
    print(f"Winner run: {winner.run_id}")
    print(f"Families ranked CSV: {families_ranked_csv}")
    print(f"Runs ranked CSV: {runs_ranked_csv}")
    print(f"Winner family seed CSV: {family_selection_csv}")
    print(f"Finalist root: {FINALIST_ROOT}")
    print(f"Selection manifest: {selection_manifest_path}")
    print(f"Analysis manifest: {analysis_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
