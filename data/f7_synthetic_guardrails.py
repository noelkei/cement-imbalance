from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from data.cleaning import apply_iforest_models, apply_univariate_density_rules
from data.f7_synthetic_cap_policy import (
    F7SyntheticCapPolicy,
    resolve_f7_synthetic_targets_from_real_counts,
)
from data.utils import ROOT_PATH, load_type_mapping


MODELED_RAW_RENAME_TO_CLEANING = {
    "90um_mesh": "90",
    "90um": "902",
    "75um": "753",
    "45um": "454",
    "30um": "305",
}


@dataclass(frozen=True)
class F7SyntheticGuardrailPolicy:
    policy_id: str = "f7_synthetic_guardrails_v1"
    split_id: str = "init_temporal_processed_v1"
    condition_col: str = "type"
    target_col: str = "init"
    post_cleaning_index_col: str = "post_cleaning_index"
    synth_flag_col: str = "is_synth"
    max_attempt_batches_per_class: int = 32
    batch_size_mode: str = "remaining_times_two_min_16"
    reject_non_finite: bool = True
    reject_invalid_class_label: bool = True
    reject_majority_class_synth: bool = True
    reject_quota_violations: bool = True
    reject_duplicate_real_rows: bool = True
    reject_duplicate_synth_rows: bool = True
    reject_negative_target: bool = True
    reject_negative_modeled_raw_numeric: bool = True
    nonnegative_exempt_columns: tuple[str, ...] = ("a", "b")
    nonnegative_exempt_prefixes: tuple[str, ...] = ("ilr_",)
    learned_cleaning_audit_mode: str = "audit_only"
    run_univariate_rules: bool = True
    run_iforest_rules: bool = True
    run_overlap_rules: bool = True
    run_distance_to_real_summary: bool = True

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "F7SyntheticGuardrailPolicy":
        payload = dict(payload or {})
        policy_cfg = dict(payload.get("policy") or {})
        retry_cfg = dict(payload.get("retry") or {})
        hard_cfg = dict(payload.get("hard_rejects") or {})
        audit_cfg = dict(payload.get("soft_audits") or {})

        cfg = cls(
            policy_id=str(policy_cfg.get("policy_id", cls.policy_id)),
            split_id=str(policy_cfg.get("split_id", cls.split_id)),
            condition_col=str(policy_cfg.get("condition_col", cls.condition_col)),
            target_col=str(policy_cfg.get("target_col", cls.target_col)),
            post_cleaning_index_col=str(
                policy_cfg.get("post_cleaning_index_col", cls.post_cleaning_index_col)
            ),
            synth_flag_col=str(policy_cfg.get("synth_flag_col", cls.synth_flag_col)),
            max_attempt_batches_per_class=int(
                retry_cfg.get("max_attempt_batches_per_class", cls.max_attempt_batches_per_class)
            ),
            batch_size_mode=str(retry_cfg.get("batch_size_mode", cls.batch_size_mode)),
            reject_non_finite=bool(hard_cfg.get("reject_non_finite", cls.reject_non_finite)),
            reject_invalid_class_label=bool(
                hard_cfg.get("reject_invalid_class_label", cls.reject_invalid_class_label)
            ),
            reject_majority_class_synth=bool(
                hard_cfg.get("reject_majority_class_synth", cls.reject_majority_class_synth)
            ),
            reject_quota_violations=bool(
                hard_cfg.get("reject_quota_violations", cls.reject_quota_violations)
            ),
            reject_duplicate_real_rows=bool(
                hard_cfg.get("reject_duplicate_real_rows", cls.reject_duplicate_real_rows)
            ),
            reject_duplicate_synth_rows=bool(
                hard_cfg.get("reject_duplicate_synth_rows", cls.reject_duplicate_synth_rows)
            ),
            reject_negative_target=bool(
                hard_cfg.get("reject_negative_target", cls.reject_negative_target)
            ),
            reject_negative_modeled_raw_numeric=bool(
                hard_cfg.get(
                    "reject_negative_modeled_raw_numeric",
                    cls.reject_negative_modeled_raw_numeric,
                )
            ),
            nonnegative_exempt_columns=tuple(
                str(col) for col in hard_cfg.get("nonnegative_exempt_columns", cls.nonnegative_exempt_columns)
            ),
            nonnegative_exempt_prefixes=tuple(
                str(prefix)
                for prefix in hard_cfg.get("nonnegative_exempt_prefixes", cls.nonnegative_exempt_prefixes)
            ),
            learned_cleaning_audit_mode=str(
                audit_cfg.get("learned_cleaning_audit_mode", cls.learned_cleaning_audit_mode)
            ),
            run_univariate_rules=bool(
                audit_cfg.get("run_univariate_rules", cls.run_univariate_rules)
            ),
            run_iforest_rules=bool(audit_cfg.get("run_iforest_rules", cls.run_iforest_rules)),
            run_overlap_rules=bool(audit_cfg.get("run_overlap_rules", cls.run_overlap_rules)),
            run_distance_to_real_summary=bool(
                audit_cfg.get("run_distance_to_real_summary", cls.run_distance_to_real_summary)
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.max_attempt_batches_per_class < 1:
            raise ValueError("max_attempt_batches_per_class must be >= 1.")
        if self.learned_cleaning_audit_mode != "audit_only":
            raise ValueError("Only learned_cleaning_audit_mode='audit_only' is supported in v1.")
        if self.batch_size_mode != "remaining_times_two_min_16":
            raise ValueError("Only batch_size_mode='remaining_times_two_min_16' is supported in v1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class F7CleaningAuditArtifacts:
    cleaning_policy_id: str
    type_col: str
    univariate_rules: dict[str, Any] | None = None
    iforest_models_by_type: dict[str, Any] | None = None
    iforest_columns_to_check: list[str] = field(default_factory=list)


def load_f7_synthetic_guardrail_policy(
    *,
    config_path: str | Path,
) -> tuple[F7SyntheticGuardrailPolicy, dict[str, Any], Path]:
    resolved_path = Path(config_path)
    with open(resolved_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML payload in {resolved_path}.")
    cfg = F7SyntheticGuardrailPolicy.from_payload(payload)
    return cfg, payload, resolved_path


def load_cleaning_audit_artifacts(
    *,
    cleaning_manifest_path: str | Path | None,
) -> F7CleaningAuditArtifacts | None:
    if cleaning_manifest_path is None:
        return None
    manifest_path = Path(cleaning_manifest_path)
    if not manifest_path.exists():
        return None

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = dict(payload.get("artifacts") or {})
    univariate_path = artifacts.get("univariate_rules")
    iforest_path = artifacts.get("iforest_models")

    univariate_rules = None
    if univariate_path:
        resolved = Path(univariate_path)
        if not resolved.exists():
            resolved = Path(ROOT_PATH) / str(univariate_path)
        if resolved.exists():
            univariate_rules = json.loads(resolved.read_text(encoding="utf-8"))

    iforest_models_by_type = None
    iforest_columns_to_check: list[str] = []
    if iforest_path:
        resolved = Path(iforest_path)
        if not resolved.exists():
            resolved = Path(ROOT_PATH) / str(iforest_path)
        if resolved.exists():
            bundle = joblib.load(resolved)
            iforest_models_by_type = dict(bundle.get("models_by_type") or {})
            iforest_columns_to_check = [str(col) for col in bundle.get("columns_to_check") or []]

    return F7CleaningAuditArtifacts(
        cleaning_policy_id=str(payload.get("cleaning_policy_id") or ""),
        type_col="type",
        univariate_rules=univariate_rules,
        iforest_models_by_type=iforest_models_by_type,
        iforest_columns_to_check=iforest_columns_to_check,
    )


def resolve_retry_batch_size(*, remaining: int, policy: F7SyntheticGuardrailPolicy) -> int:
    if policy.batch_size_mode == "remaining_times_two_min_16":
        return max(16, int(remaining) * 2)
    return int(remaining)


def build_modeled_raw_cleaning_audit_view(
    *,
    X_df: pd.DataFrame,
    y_df: pd.DataFrame | None = None,
    condition_col: str = "type",
) -> pd.DataFrame:
    if condition_col not in X_df.columns:
        raise ValueError(f"Missing condition_col '{condition_col}' in modeled raw frame.")

    view = X_df.copy()
    rename_map = {
        old: new for old, new in MODELED_RAW_RENAME_TO_CLEANING.items() if old in view.columns
    }
    if rename_map:
        view.rename(columns=rename_map, inplace=True)

    try:
        type_mapping = load_type_mapping(verbose=False)
        inverse_type_mapping = {int(idx): str(label) for label, idx in dict(type_mapping or {}).items()}
    except Exception:
        inverse_type_mapping = {}

    if pd.api.types.is_numeric_dtype(view[condition_col]):
        view[condition_col] = view[condition_col].map(
            lambda value: inverse_type_mapping.get(int(value), f"type_{int(value)}")
        )
    else:
        view[condition_col] = view[condition_col].astype(str)
    return view


def _stable_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_class_label(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _stable_row_fingerprint(
    *,
    x_row: pd.Series,
    y_row: pd.Series,
    condition_col: str,
    post_cleaning_index_col: str,
    synth_flag_col: str,
) -> str:
    x_payload = {
        str(col): _stable_value(val)
        for col, val in x_row.items()
        if col not in {post_cleaning_index_col, synth_flag_col}
    }
    y_payload = {
        str(col): _stable_value(val)
        for col, val in y_row.items()
        if col not in {post_cleaning_index_col, synth_flag_col}
    }
    payload = {"x": x_payload, "y": y_payload, "condition": x_payload.get(condition_col)}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _resolve_nonnegative_columns(
    *,
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    condition_col: str,
    policy: F7SyntheticGuardrailPolicy,
) -> list[str]:
    candidate_cols: list[str] = []
    exempt_cols = set(policy.nonnegative_exempt_columns)
    exempt_prefixes = tuple(policy.nonnegative_exempt_prefixes)
    for col in X_df.columns:
        if col in {policy.post_cleaning_index_col, condition_col, policy.synth_flag_col}:
            continue
        if col in exempt_cols or any(str(col).startswith(prefix) for prefix in exempt_prefixes):
            continue
        candidate_cols.append(str(col))
    for col in y_df.columns:
        if col in {policy.post_cleaning_index_col, policy.synth_flag_col}:
            continue
        if col == policy.target_col:
            candidate_cols.append(str(col))
    return candidate_cols


def _summarize_distance_to_real(
    *,
    accepted_audit_view: pd.DataFrame,
    real_audit_view: pd.DataFrame,
    policy: F7SyntheticGuardrailPolicy,
) -> dict[str, Any]:
    if accepted_audit_view.empty or real_audit_view.empty:
        return {"status": "skipped_empty"}

    summaries: dict[str, Any] = {}
    for cls, accepted_cls in accepted_audit_view.groupby(policy.condition_col, sort=True):
        real_cls = real_audit_view[real_audit_view[policy.condition_col] == cls]
        if real_cls.empty:
            summaries[str(cls)] = {"status": "skipped_no_real_reference"}
            continue

        feature_cols = [
            col
            for col in accepted_cls.columns
            if col not in {policy.post_cleaning_index_col, policy.condition_col, policy.synth_flag_col}
            and pd.api.types.is_numeric_dtype(accepted_cls[col])
            and col in real_cls.columns
        ]
        if not feature_cols:
            summaries[str(cls)] = {"status": "skipped_no_numeric_features"}
            continue

        scaler = StandardScaler()
        real_values = scaler.fit_transform(real_cls[feature_cols].astype(float))
        acc_values = scaler.transform(accepted_cls[feature_cols].astype(float))
        deltas = acc_values[:, None, :] - real_values[None, :, :]
        dists = np.sqrt(np.sum(np.square(deltas), axis=2))
        nearest = dists.min(axis=1) if dists.size else np.array([], dtype=float)
        summaries[str(cls)] = {
            "status": "ok",
            "n_rows": int(len(accepted_cls)),
            "mean_nearest_l2": float(nearest.mean()) if nearest.size else None,
            "min_nearest_l2": float(nearest.min()) if nearest.size else None,
            "max_nearest_l2": float(nearest.max()) if nearest.size else None,
        }
    return summaries


class F7SyntheticAcceptanceEngine:
    def __init__(
        self,
        *,
        real_X_train: pd.DataFrame,
        real_y_train: pd.DataFrame,
        cap_policy: F7SyntheticCapPolicy,
        guardrail_policy: F7SyntheticGuardrailPolicy,
        cleaning_audit_artifacts: F7CleaningAuditArtifacts | None = None,
        real_X_audit_view: pd.DataFrame | None = None,
    ) -> None:
        self.cap_policy = cap_policy
        self.guardrail_policy = guardrail_policy
        self.cleaning_audit_artifacts = cleaning_audit_artifacts
        self.real_X_train = real_X_train.copy().reset_index(drop=True)
        self.real_y_train = real_y_train.copy().reset_index(drop=True)
        self.real_X_audit_view = (
            real_X_audit_view.copy().reset_index(drop=True) if real_X_audit_view is not None else None
        )

        self.target_summary = resolve_f7_synthetic_targets_from_real_counts(
            real_counts={
                _normalize_class_label(label): int(count)
                for label, count in self.real_X_train[cap_policy.condition_col].value_counts().sort_index().to_dict().items()
            },
            policy=cap_policy,
        )
        self.target_counts_by_class = {
            _normalize_class_label(label): int(count)
            for label, count in dict(self.target_summary["targets_by_class"]).items()
        }
        self.majority_labels = set(self.target_summary["majority_reference_classes"])
        self.accepted_counts_by_class = {label: 0 for label in self.target_counts_by_class}
        self.attempt_batches_by_class = {label: 0 for label in self.target_counts_by_class}
        self.reject_counts_by_reason = Counter()
        self.reject_counts_by_class = defaultdict(Counter)
        self.soft_audit_counts_by_rule = Counter()
        self.distance_to_real_by_class: dict[str, Any] = {}

        self.next_index = int(self.real_X_train[guardrail_policy.post_cleaning_index_col].max()) + 1
        self.nonnegative_columns = _resolve_nonnegative_columns(
            X_df=self.real_X_train,
            y_df=self.real_y_train,
            condition_col=cap_policy.condition_col,
            policy=guardrail_policy,
        )
        self.real_row_fingerprints = {
            _stable_row_fingerprint(
                x_row=x_row,
                y_row=y_row,
                condition_col=cap_policy.condition_col,
                post_cleaning_index_col=guardrail_policy.post_cleaning_index_col,
                synth_flag_col=guardrail_policy.synth_flag_col,
            )
            for (_, x_row), (_, y_row) in zip(self.real_X_train.iterrows(), self.real_y_train.iterrows())
        }
        self.accepted_row_fingerprints: set[str] = set()

    def note_attempt_batch(self, *, class_label: str) -> None:
        self.attempt_batches_by_class[_normalize_class_label(class_label)] += 1

    def _record_reject(self, *, class_label: str, reasons: list[str]) -> None:
        label = _normalize_class_label(class_label)
        for reason in reasons:
            self.reject_counts_by_reason[str(reason)] += 1
            self.reject_counts_by_class[label][str(reason)] += 1

    def _validate_row(
        self,
        *,
        x_row_materialized: pd.Series,
        y_row_materialized: pd.Series,
        x_row_domain: pd.Series,
        y_row_domain: pd.Series,
    ) -> list[str]:
        reasons: list[str] = []
        condition_col = self.cap_policy.condition_col
        class_label = _normalize_class_label(x_row_materialized[condition_col])

        if self.guardrail_policy.reject_invalid_class_label and class_label not in self.target_counts_by_class:
            reasons.append("invalid_class_label")
            return reasons

        if self.guardrail_policy.reject_majority_class_synth and class_label in self.majority_labels:
            reasons.append("majority_class_synth_forbidden")

        if self.guardrail_policy.reject_quota_violations:
            allowed = int(self.target_counts_by_class.get(class_label, 0))
            current = int(self.accepted_counts_by_class.get(class_label, 0))
            if current >= allowed:
                reasons.append("quota_exhausted_for_class")

        materialized_numeric = pd.concat([x_row_materialized, y_row_materialized]).drop(
            labels=[
                self.guardrail_policy.synth_flag_col,
                self.guardrail_policy.post_cleaning_index_col,
                self.cap_policy.condition_col,
            ],
            errors="ignore",
        )
        domain_numeric = pd.concat([x_row_domain, y_row_domain]).drop(
            labels=[
                self.guardrail_policy.synth_flag_col,
                self.guardrail_policy.post_cleaning_index_col,
                self.cap_policy.condition_col,
            ],
            errors="ignore",
        )
        if self.guardrail_policy.reject_non_finite:
            values = pd.to_numeric(materialized_numeric, errors="coerce")
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                reasons.append("non_finite_values")
                return reasons

        if self.guardrail_policy.reject_negative_target:
            target_value = pd.to_numeric(
                pd.Series([y_row_domain.get(self.guardrail_policy.target_col)]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(target_value) and float(target_value) < 0:
                reasons.append("negative_target")

        if self.guardrail_policy.reject_negative_modeled_raw_numeric:
            for col in self.nonnegative_columns:
                if col not in x_row_domain.index and col not in y_row_domain.index:
                    continue
                value = x_row_domain.get(col, y_row_domain.get(col))
                numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.notna(numeric_value) and float(numeric_value) < 0:
                    reasons.append(f"negative_modeled_raw:{col}")
                    break

        fingerprint = _stable_row_fingerprint(
            x_row=x_row_materialized,
            y_row=y_row_materialized,
            condition_col=condition_col,
            post_cleaning_index_col=self.guardrail_policy.post_cleaning_index_col,
            synth_flag_col=self.guardrail_policy.synth_flag_col,
        )
        if self.guardrail_policy.reject_duplicate_real_rows and fingerprint in self.real_row_fingerprints:
            reasons.append("duplicate_real_row")
        if self.guardrail_policy.reject_duplicate_synth_rows and fingerprint in self.accepted_row_fingerprints:
            reasons.append("duplicate_synth_row")
        return reasons

    def _run_soft_audits(
        self,
        *,
        accepted_X_domain: pd.DataFrame,
    ) -> dict[str, Any]:
        artifacts = self.cleaning_audit_artifacts
        if artifacts is None or accepted_X_domain.empty:
            return {"status": "skipped_no_artifacts", "counts": {}}

        audit_view = build_modeled_raw_cleaning_audit_view(
            X_df=accepted_X_domain,
            condition_col=self.cap_policy.condition_col,
        )
        counts: dict[str, int] = {}

        if self.guardrail_policy.run_univariate_rules and artifacts.univariate_rules:
            keep_mask = apply_univariate_density_rules(
                audit_view,
                artifacts.univariate_rules,
                type_col=artifacts.type_col,
            )
            flagged = int((~keep_mask).sum())
            counts["univariate_flagged_rows"] = flagged
            self.soft_audit_counts_by_rule["univariate_flagged_rows"] += flagged
        if self.guardrail_policy.run_iforest_rules and artifacts.iforest_models_by_type:
            flags = apply_iforest_models(
                audit_view,
                artifacts.iforest_models_by_type,
                artifacts.iforest_columns_to_check,
                type_col=artifacts.type_col,
            )
            flagged = int(flags.sum())
            counts["iforest_flagged_rows"] = flagged
            self.soft_audit_counts_by_rule["iforest_flagged_rows"] += flagged
            if self.guardrail_policy.run_overlap_rules and artifacts.univariate_rules:
                keep_mask = apply_univariate_density_rules(
                    audit_view,
                    artifacts.univariate_rules,
                    type_col=artifacts.type_col,
                )
                overlap = (~keep_mask) & flags
                overlap_count = int(overlap.sum())
                counts["overlap_flagged_rows"] = overlap_count
                self.soft_audit_counts_by_rule["overlap_flagged_rows"] += overlap_count

        if (
            self.guardrail_policy.run_distance_to_real_summary
            and self.real_X_audit_view is not None
        ):
            self.distance_to_real_by_class = _summarize_distance_to_real(
                accepted_audit_view=audit_view,
                real_audit_view=self.real_X_audit_view,
                policy=self.guardrail_policy,
            )

        return {"status": "ok", "counts": counts}

    def accept_batch(
        self,
        *,
        X_batch_materialized: pd.DataFrame,
        y_batch_materialized: pd.DataFrame,
        X_batch_domain: pd.DataFrame | None = None,
        y_batch_domain: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
        if len(X_batch_materialized) != len(y_batch_materialized):
            raise ValueError("X_batch_materialized and y_batch_materialized must have the same length.")
        X_batch_domain = X_batch_materialized.copy() if X_batch_domain is None else X_batch_domain.copy()
        y_batch_domain = y_batch_materialized.copy() if y_batch_domain is None else y_batch_domain.copy()
        if len(X_batch_domain) != len(X_batch_materialized) or len(y_batch_domain) != len(y_batch_materialized):
            raise ValueError("Domain-view batches must match materialized batches row-for-row.")

        accepted_materialized_x: list[pd.Series] = []
        accepted_materialized_y: list[pd.Series] = []
        accepted_domain_x: list[pd.Series] = []
        accepted_domain_y: list[pd.Series] = []
        batch_reject_counts = Counter()

        for idx in range(len(X_batch_materialized)):
            x_mat = X_batch_materialized.iloc[idx].copy()
            y_mat = y_batch_materialized.iloc[idx].copy()
            x_dom = X_batch_domain.iloc[idx].copy()
            y_dom = y_batch_domain.iloc[idx].copy()

            class_label = _normalize_class_label(x_mat[self.cap_policy.condition_col])
            reasons = self._validate_row(
                x_row_materialized=x_mat,
                y_row_materialized=y_mat,
                x_row_domain=x_dom,
                y_row_domain=y_dom,
            )
            if reasons:
                for reason in reasons:
                    batch_reject_counts[str(reason)] += 1
                self._record_reject(class_label=class_label, reasons=reasons)
                continue

            assigned_index = int(self.next_index)
            self.next_index += 1
            for row in (x_mat, y_mat, x_dom, y_dom):
                if self.guardrail_policy.post_cleaning_index_col in row.index:
                    row[self.guardrail_policy.post_cleaning_index_col] = assigned_index

            fingerprint = _stable_row_fingerprint(
                x_row=x_mat,
                y_row=y_mat,
                condition_col=self.cap_policy.condition_col,
                post_cleaning_index_col=self.guardrail_policy.post_cleaning_index_col,
                synth_flag_col=self.guardrail_policy.synth_flag_col,
            )
            self.accepted_row_fingerprints.add(fingerprint)
            self.accepted_counts_by_class[class_label] = int(self.accepted_counts_by_class.get(class_label, 0)) + 1

            accepted_materialized_x.append(x_mat)
            accepted_materialized_y.append(y_mat)
            accepted_domain_x.append(x_dom)
            accepted_domain_y.append(y_dom)

        if accepted_materialized_x:
            acc_x_mat = pd.DataFrame(accepted_materialized_x)
            acc_y_mat = pd.DataFrame(accepted_materialized_y)
            acc_x_dom = pd.DataFrame(accepted_domain_x)
            acc_y_dom = pd.DataFrame(accepted_domain_y)
        else:
            acc_x_mat = X_batch_materialized.iloc[0:0].copy()
            acc_y_mat = y_batch_materialized.iloc[0:0].copy()
            acc_x_dom = X_batch_domain.iloc[0:0].copy()
            acc_y_dom = y_batch_domain.iloc[0:0].copy()

        audit_summary = self._run_soft_audits(accepted_X_domain=acc_x_dom)
        summary = {
            "accepted_count": int(len(acc_x_mat)),
            "rejected_count": int(len(X_batch_materialized) - len(acc_x_mat)),
            "reject_counts_by_reason": dict(batch_reject_counts),
            "soft_audit_summary": audit_summary,
        }
        return acc_x_mat, acc_y_mat, acc_x_dom, acc_y_dom, summary

    def summary(self) -> dict[str, Any]:
        return {
            "guardrail_policy_id": self.guardrail_policy.policy_id,
            "target_counts_by_class": {
                str(label): int(value) for label, value in self.target_counts_by_class.items()
            },
            "accepted_counts_by_class": {
                str(label): int(value) for label, value in self.accepted_counts_by_class.items()
            },
            "attempt_batches_by_class": {
                str(label): int(value) for label, value in self.attempt_batches_by_class.items()
            },
            "reject_counts_by_reason": dict(self.reject_counts_by_reason),
            "reject_counts_by_class": {
                str(label): dict(counter) for label, counter in self.reject_counts_by_class.items()
            },
            "soft_audit_counts_by_rule": dict(self.soft_audit_counts_by_rule),
            "distance_to_real_by_class": self.distance_to_real_by_class,
            "accepted_sample_fingerprint_sha256": sorted(self.accepted_row_fingerprints),
        }
