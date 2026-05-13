from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REPORTS_ROOT = ROOT / "outputs" / "reports" / "f6"
CONFIGS_ROOT = REPORTS_ROOT / "configs"
DEFAULT_Y_SCALERS = ["standard", "robust", "minmax", "quantile"]
OFFICIAL_MODELS_ROOT = ROOT / "outputs" / "models" / "official"
BUDGET_LEDGER_PATH = REPORTS_ROOT / "campaign_budget.json"

FLOWPRE_LIMIT = 28
FLOWGEN_LIMIT = 12
F6_TOTAL_LIMIT = 40

FLOWPRE_CONTRACT_ID = "f6_flowpre_revalidation_v1"
FLOWGEN_CONTRACT_ID = "f6_flowgen_revalidation_v1"
OFFICIAL_SPLIT_ID = "init_temporal_processed_v1"

_SUCCESS_STATUSES = {"completed", "successful", "success"}


def ensure_budget(*, planned: int, limit: int, label: str) -> None:
    if planned > limit:
        raise RuntimeError(f"{label} would require {planned} runs, exceeding the hard budget limit {limit}.")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def write_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return out_path


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return out_path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bool_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no", "off"}
    return not bool(value)


def _as_path(path_like: str | Path | None, *, base_dir: Path) -> Path | None:
    if path_like in (None, ""):
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _run_manifest_glob(model_family: str) -> Path:
    return OFFICIAL_MODELS_ROOT / model_family


def _promotion_manifest_glob(model_family: str) -> Path:
    return OFFICIAL_MODELS_ROOT / model_family


def _default_budget_payload() -> dict[str, Any]:
    return {
        "limits": {
            "flowpre": FLOWPRE_LIMIT,
            "flowgen": FLOWGEN_LIMIT,
            "total": F6_TOTAL_LIMIT,
        },
        "ledger_counts": {"flowpre": 0, "flowgen": 0, "total": 0},
        "consumed_from_manifests": {"flowpre": 0, "flowgen": 0, "total": 0},
        "effective_counts": {"flowpre": 0, "flowgen": 0, "total": 0},
        "recomputed_from_manifests": True,
        "counted_run_manifests": {"flowpre": [], "flowgen": []},
        "last_updated": _utc_now_iso(),
        "source_reports": {
            "flowpre_all_runs": str(REPORTS_ROOT / "flowpre_all_runs.csv"),
            "flowgen_all_runs": str(REPORTS_ROOT / "flowgen_all_runs.csv"),
            "flowpre_selected": str(REPORTS_ROOT / "flowpre_selected.csv"),
            "flowgen_selected": str(REPORTS_ROOT / "flowgen_selected.csv"),
        },
    }


def _expected_results_path(manifest_path: Path, payload: dict[str, Any]) -> Path:
    run_id = str(payload.get("run_id") or manifest_path.parent.name)
    return manifest_path.parent / f"{run_id}_results.yaml"


def _expected_metrics_long_path(manifest_path: Path, payload: dict[str, Any]) -> Path | None:
    metrics_path = _as_path(payload.get("metrics_long_path"), base_dir=manifest_path.parent)
    if metrics_path is not None:
        return metrics_path
    run_id = str(payload.get("run_id") or manifest_path.parent.name)
    return manifest_path.parent / f"{run_id}_metrics_long.csv"


def _run_status_completed(payload: dict[str, Any]) -> bool | None:
    for key in ("run_status", "status"):
        raw = payload.get(key)
        if raw is None:
            continue
        return str(raw).strip().lower() in _SUCCESS_STATUSES
    return None


def _is_countable_run(manifest_path: Path, payload: dict[str, Any], *, contract_id: str) -> bool:
    if str(payload.get("contract_id")) != contract_id:
        return False
    if str(payload.get("split_id")) != OFFICIAL_SPLIT_ID:
        return False
    if not _bool_false(payload.get("test_enabled")):
        return False

    run_status_completed = _run_status_completed(payload)
    if run_status_completed is not None:
        return run_status_completed

    metrics_path = _expected_metrics_long_path(manifest_path, payload)
    results_path = _expected_results_path(manifest_path, payload)
    return bool(metrics_path and metrics_path.exists() and results_path.exists())


def recompute_campaign_budget_from_manifests() -> dict[str, Any]:
    payload = _default_budget_payload()

    family_specs = {
        "flowpre": ("flow_pre", FLOWPRE_CONTRACT_ID),
        "flowgen": ("flowgen", FLOWGEN_CONTRACT_ID),
    }
    counted: dict[str, list[str]] = {"flowpre": [], "flowgen": []}

    for logical_family, (model_family, contract_id) in family_specs.items():
        family_root = _run_manifest_glob(model_family)
        if not family_root.exists():
            continue

        for manifest_path in sorted(family_root.glob("*/*_run_manifest.json")):
            try:
                manifest = load_json(manifest_path)
            except Exception:
                continue
            if _is_countable_run(manifest_path, manifest, contract_id=contract_id):
                counted[logical_family].append(str(manifest_path))

    counts = {
        "flowpre": len(counted["flowpre"]),
        "flowgen": len(counted["flowgen"]),
    }
    counts["total"] = counts["flowpre"] + counts["flowgen"]

    payload["consumed_from_manifests"] = counts
    payload["effective_counts"] = counts
    payload["counted_run_manifests"] = counted
    payload["recomputed_from_manifests"] = True
    payload["last_updated"] = _utc_now_iso()
    return payload


def sync_campaign_budget_ledger() -> dict[str, Any]:
    ledger_exists = BUDGET_LEDGER_PATH.exists()
    existing = load_json(BUDGET_LEDGER_PATH) if ledger_exists else {}
    recomputed = recompute_campaign_budget_from_manifests()

    ledger_counts = dict(existing.get("effective_counts") or existing.get("consumed_from_manifests") or {})
    if not ledger_counts:
        ledger_counts = dict(existing.get("ledger_counts") or {})
    for key in ("flowpre", "flowgen", "total"):
        ledger_counts.setdefault(key, 0)

    effective_counts = dict(recomputed["effective_counts"])
    out_payload = _default_budget_payload()
    out_payload["ledger_counts"] = ledger_counts
    out_payload["consumed_from_manifests"] = dict(recomputed["consumed_from_manifests"])
    out_payload["effective_counts"] = effective_counts
    out_payload["recomputed_from_manifests"] = (not ledger_exists) or (ledger_counts != effective_counts)
    out_payload["counted_run_manifests"] = dict(recomputed["counted_run_manifests"])
    out_payload["last_updated"] = _utc_now_iso()

    write_json(BUDGET_LEDGER_PATH, out_payload)
    return out_payload


def ensure_campaign_budget(*, planned_flowpre: int = 0, planned_flowgen: int = 0) -> dict[str, Any]:
    budget = sync_campaign_budget_ledger()
    effective = dict(budget["effective_counts"])

    next_flowpre = int(effective["flowpre"]) + int(planned_flowpre)
    next_flowgen = int(effective["flowgen"]) + int(planned_flowgen)
    next_total = int(effective["total"]) + int(planned_flowpre) + int(planned_flowgen)

    violations: list[str] = []
    if next_flowpre > FLOWPRE_LIMIT:
        violations.append(
            f"FlowPre consumed={effective['flowpre']} planned={planned_flowpre} limit={FLOWPRE_LIMIT}"
        )
    if next_flowgen > FLOWGEN_LIMIT:
        violations.append(
            f"FlowGen consumed={effective['flowgen']} planned={planned_flowgen} limit={FLOWGEN_LIMIT}"
        )
    if next_total > F6_TOTAL_LIMIT:
        violations.append(
            f"F6 total consumed={effective['total']} planned={planned_flowpre + planned_flowgen} limit={F6_TOTAL_LIMIT}"
        )

    if violations:
        raise RuntimeError(
            "F6 campaign budget would be exceeded. "
            + " | ".join(violations)
            + f" | effective_counts={effective}"
        )
    return budget


def find_promotion_manifests(model_family: str, *, branch_id: str | None = None) -> list[Path]:
    family_root = _promotion_manifest_glob(model_family)
    if not family_root.exists():
        return []

    matches: list[Path] = []
    for path in sorted(family_root.glob("*/*_promotion_manifest.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if branch_id is not None and str(payload.get("branch_id")) != branch_id:
            continue
        matches.append(path)
    return matches


def source_id_matches_flowpre_pattern(source_id: str, *, branch_id: str, split_id: str) -> bool:
    parts = str(source_id).split("__")
    return len(parts) >= 4 and parts[0] == "flowpre" and parts[1] == branch_id and parts[2] == split_id

