from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


@dataclass(frozen=True)
class F7SyntheticCapPolicy:
    policy_id: str = "f7_synthetic_cap_policy_v1"
    split_id: str = "init_temporal_processed_v1"
    condition_col: str = "type"
    synth_flag_col: str = "is_synth"
    max_fraction_of_real_per_minority_class: float = 0.5
    forbid_minorities_from_exceeding_majority_real_count: bool = True
    require_zero_synth_in_majority_classes: bool = True
    majority_tie_policy: str = "no_synth_for_tied_majority_classes"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "F7SyntheticCapPolicy":
        payload = dict(payload or {})
        policy_cfg = dict(payload.get("policy") or {})
        constraints_cfg = dict(payload.get("constraints") or {})
        cfg = cls(
            policy_id=str(policy_cfg.get("policy_id", cls.policy_id)),
            split_id=str(policy_cfg.get("split_id", cls.split_id)),
            condition_col=str(policy_cfg.get("condition_col", cls.condition_col)),
            synth_flag_col=str(policy_cfg.get("synth_flag_col", cls.synth_flag_col)),
            max_fraction_of_real_per_minority_class=float(
                constraints_cfg.get(
                    "max_fraction_of_real_per_minority_class",
                    cls.max_fraction_of_real_per_minority_class,
                )
            ),
            forbid_minorities_from_exceeding_majority_real_count=bool(
                constraints_cfg.get(
                    "forbid_minorities_from_exceeding_majority_real_count",
                    cls.forbid_minorities_from_exceeding_majority_real_count,
                )
            ),
            require_zero_synth_in_majority_classes=bool(
                constraints_cfg.get(
                    "require_zero_synth_in_majority_classes",
                    cls.require_zero_synth_in_majority_classes,
                )
            ),
            majority_tie_policy=str(policy_cfg.get("majority_tie_policy", cls.majority_tie_policy)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.max_fraction_of_real_per_minority_class < 0:
            raise ValueError("max_fraction_of_real_per_minority_class must be >= 0.")
        if self.majority_tie_policy != "no_synth_for_tied_majority_classes":
            raise ValueError(
                "Unsupported majority_tie_policy. Use 'no_synth_for_tied_majority_classes'."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_f7_synthetic_cap_policy(
    *,
    config_path: str | Path,
) -> tuple[F7SyntheticCapPolicy, dict[str, Any], Path]:
    resolved_path = Path(config_path)
    with open(resolved_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML payload in {resolved_path}.")
    cfg = F7SyntheticCapPolicy.from_payload(payload)
    return cfg, payload, resolved_path


def _normalize_synth_flag(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(0).astype(int).astype(bool)


def resolve_f7_synthetic_targets_from_real_counts(
    *,
    real_counts: Mapping[Any, int],
    policy: F7SyntheticCapPolicy,
) -> dict[str, Any]:
    normalized_counts = {
        str(label): int(count)
        for label, count in dict(real_counts or {}).items()
    }
    if not normalized_counts:
        raise ValueError("F7 target resolution requires at least one observed real class count.")

    max_real = max(normalized_counts.values())
    majority_labels = sorted(
        label for label, count in normalized_counts.items() if int(count) == int(max_real)
    )

    targets_by_class: dict[str, int] = {}
    per_class: dict[str, dict[str, Any]] = {}
    for label in sorted(normalized_counts):
        n_real = int(normalized_counts[label])
        is_majority = label in majority_labels
        cap_from_real = 0 if is_majority else int(
            math.floor(float(policy.max_fraction_of_real_per_minority_class) * float(n_real))
        )
        cap_from_majority = 0 if is_majority else max(0, int(max_real) - int(n_real))
        allowed_synth_max = 0 if is_majority else min(int(cap_from_real), int(cap_from_majority))
        targets_by_class[label] = int(allowed_synth_max)
        per_class[label] = {
            "n_real": int(n_real),
            "is_majority_reference_class": bool(is_majority),
            "cap_from_real_fraction": int(cap_from_real),
            "cap_from_majority_ceiling": int(cap_from_majority),
            "allowed_synth_max": int(allowed_synth_max),
            "target_synth": int(allowed_synth_max),
        }

    return {
        "policy_id": policy.policy_id,
        "split_id": policy.split_id,
        "majority_reference_real_count": int(max_real),
        "majority_reference_classes": majority_labels,
        "targets_by_class": targets_by_class,
        "per_class": per_class,
    }


def resolve_f7_synthetic_targets(
    *,
    train_df: pd.DataFrame,
    policy: F7SyntheticCapPolicy,
) -> dict[str, Any]:
    if policy.condition_col not in train_df.columns:
        raise ValueError(f"Missing condition_col '{policy.condition_col}' in train_df.")

    work = train_df.copy()
    if policy.synth_flag_col in work.columns:
        synth_mask = _normalize_synth_flag(work[policy.synth_flag_col])
    else:
        synth_mask = pd.Series(False, index=work.index)

    real_counts = (
        work.loc[~synth_mask, policy.condition_col]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return resolve_f7_synthetic_targets_from_real_counts(real_counts=real_counts, policy=policy)


def summarize_f7_synthetic_cap(
    *,
    train_df: pd.DataFrame,
    policy: F7SyntheticCapPolicy,
) -> dict[str, Any]:
    if policy.condition_col not in train_df.columns:
        raise ValueError(f"Missing condition_col '{policy.condition_col}' in train_df.")

    work = train_df.copy()
    if policy.synth_flag_col in work.columns:
        synth_mask = _normalize_synth_flag(work[policy.synth_flag_col])
    else:
        synth_mask = pd.Series(False, index=work.index)

    real_df = work.loc[~synth_mask].copy()
    synth_df = work.loc[synth_mask].copy()

    real_counts = (
        real_df[policy.condition_col]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    synth_counts = (
        synth_df[policy.condition_col]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    class_labels = sorted(set(real_counts) | set(synth_counts))
    if not class_labels:
        raise ValueError("Synthetic-cap validation requires at least one observed class in train_df.")

    target_summary = resolve_f7_synthetic_targets_from_real_counts(
        real_counts=real_counts,
        policy=policy,
    )
    max_real = int(target_summary["majority_reference_real_count"])
    majority_labels = list(target_summary["majority_reference_classes"])

    per_class: dict[str, dict[str, Any]] = {}
    is_valid = True
    violations: list[str] = []

    for label in class_labels:
        n_real = int(real_counts.get(label, 0))
        n_synth = int(synth_counts.get(label, 0))
        n_final = n_real + n_synth
        synth_fraction_final = (float(n_synth) / float(n_final)) if n_final > 0 else 0.0
        is_majority = label in majority_labels

        class_target = dict(target_summary["per_class"].get(label) or {})
        cap_from_real = int(class_target.get("cap_from_real_fraction", 0))
        cap_from_majority = int(class_target.get("cap_from_majority_ceiling", 0))
        allowed_synth_max = int(class_target.get("allowed_synth_max", 0))

        class_ok = True
        class_violations: list[str] = []

        if is_majority and policy.require_zero_synth_in_majority_classes and n_synth != 0:
            class_ok = False
            class_violations.append("majority_class_has_synth")

        if (not is_majority) and n_synth > cap_from_real:
            class_ok = False
            class_violations.append("exceeds_fractional_cap")

        if (
            (not is_majority)
            and policy.forbid_minorities_from_exceeding_majority_real_count
            and n_final > int(max_real)
        ):
            class_ok = False
            class_violations.append("exceeds_majority_real_count")

        per_class[label] = {
            "n_real": n_real,
            "n_synth": n_synth,
            "n_final": n_final,
            "synth_fraction_final": synth_fraction_final,
            "is_majority_reference_class": bool(is_majority),
            "cap_from_real_fraction": int(cap_from_real),
            "cap_from_majority_ceiling": int(cap_from_majority),
            "allowed_synth_max": int(allowed_synth_max),
            "passes_f7_cap": bool(class_ok),
            "violations": class_violations,
        }

        if not class_ok:
            is_valid = False
            violations.extend(f"{label}:{reason}" for reason in class_violations)

    return {
        "policy_id": policy.policy_id,
        "split_id": policy.split_id,
        "condition_col": policy.condition_col,
        "synth_flag_col": policy.synth_flag_col,
        "majority_reference_real_count": int(max_real),
        "majority_reference_classes": majority_labels,
        "per_class": per_class,
        "is_valid_campaign_ready": bool(is_valid),
        "violations": violations,
    }


def validate_f7_synthetic_cap(
    *,
    train_df: pd.DataFrame,
    policy: F7SyntheticCapPolicy,
) -> dict[str, Any]:
    summary = summarize_f7_synthetic_cap(train_df=train_df, policy=policy)
    if not bool(summary["is_valid_campaign_ready"]):
        joined = ", ".join(summary["violations"]) or "unknown_violation"
        raise ValueError(f"Dataset does not satisfy F7 synthetic cap policy: {joined}")
    return summary
