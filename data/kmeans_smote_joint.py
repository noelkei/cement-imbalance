from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass(frozen=True)
class KMeansSmoteJointConfig:
    synthetic_policy_config_id: str = "kmeans_smote_joint_base_v1"
    synthetic_seed: int = 42
    condition_col: str = "type"
    target_mode: str = "train_majority"
    target_value: float | int | None = None
    metric_space_mode: str = "robust"
    cluster_k_mode: str = "silhouette_auto"
    cluster_k_fixed: int | None = None
    cluster_k_max: int = 12
    min_cluster_size: int = 8
    silhouette_sample_size: int = 5000
    silhouette_min_accept: float = 0.02
    silhouette_tol: float = 1.0e-4
    neighbor_k_mode: str = "auto"
    neighbor_k_value: int | None = None
    neighbor_k_max: int = 5
    lambda_distribution: str = "uniform"
    redistribute_singletons: bool = True
    error_if_no_eligible_cluster: bool = True

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "KMeansSmoteJointConfig":
        payload = dict(payload or {})
        contract_cfg = dict(payload.get("contract") or {})
        policy_cfg = dict(payload.get("policy") or {})
        target_cfg = dict(policy_cfg.get("target_policy") or {})
        metric_cfg = dict(policy_cfg.get("metric_space") or {})
        clustering_cfg = dict(policy_cfg.get("clustering") or {})
        neighbors_cfg = dict(policy_cfg.get("neighbors") or {})
        sampling_cfg = dict(policy_cfg.get("sampling") or {})

        cfg = cls(
            synthetic_policy_config_id=str(
                contract_cfg.get("synthetic_policy_config_id", cls.synthetic_policy_config_id)
            ),
            synthetic_seed=int(policy_cfg.get("synthetic_seed", cls.synthetic_seed)),
            condition_col=str(policy_cfg.get("condition_col", cls.condition_col)),
            target_mode=str(target_cfg.get("mode", cls.target_mode)),
            target_value=target_cfg.get("value", cls.target_value),
            metric_space_mode=str(metric_cfg.get("mode", cls.metric_space_mode)).lower(),
            cluster_k_mode=str(
                clustering_cfg.get("k_mode", clustering_cfg.get("mode", cls.cluster_k_mode))
            ).lower(),
            cluster_k_fixed=(
                None
                if clustering_cfg.get("k_fixed", clustering_cfg.get("k_value", cls.cluster_k_fixed)) is None
                else int(clustering_cfg.get("k_fixed", clustering_cfg.get("k_value", cls.cluster_k_fixed)))
            ),
            cluster_k_max=int(clustering_cfg.get("k_max", cls.cluster_k_max)),
            min_cluster_size=int(clustering_cfg.get("min_cluster_size", cls.min_cluster_size)),
            silhouette_sample_size=int(
                clustering_cfg.get("silhouette_sample_size", cls.silhouette_sample_size)
            ),
            silhouette_min_accept=float(
                clustering_cfg.get("silhouette_min_accept", cls.silhouette_min_accept)
            ),
            silhouette_tol=float(clustering_cfg.get("silhouette_tol", cls.silhouette_tol)),
            neighbor_k_mode=str(neighbors_cfg.get("k_mode", cls.neighbor_k_mode)).lower(),
            neighbor_k_value=(
                None
                if neighbors_cfg.get("k_value", cls.neighbor_k_value) is None
                else int(neighbors_cfg.get("k_value", cls.neighbor_k_value))
            ),
            neighbor_k_max=int(neighbors_cfg.get("k_max", cls.neighbor_k_max)),
            lambda_distribution=str(
                sampling_cfg.get("lambda_distribution", cls.lambda_distribution)
            ).lower(),
            redistribute_singletons=bool(
                sampling_cfg.get("redistribute_singletons", cls.redistribute_singletons)
            ),
            error_if_no_eligible_cluster=bool(
                sampling_cfg.get("error_if_no_eligible_cluster", cls.error_if_no_eligible_cluster)
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.target_mode not in {"train_majority", "ratio_to_majority", "absolute_per_class"}:
            raise ValueError(
                f"Unsupported target_mode '{self.target_mode}'. "
                "Use 'train_majority', 'ratio_to_majority', or 'absolute_per_class'."
            )
        if self.target_mode == "ratio_to_majority":
            if self.target_value is None or float(self.target_value) <= 0:
                raise ValueError("ratio_to_majority requires target_value > 0.")
        if self.target_mode == "absolute_per_class":
            if self.target_value is None or int(self.target_value) <= 0:
                raise ValueError("absolute_per_class requires target_value > 0.")
        if self.metric_space_mode not in {"robust", "standard", "none"}:
            raise ValueError(
                f"Unsupported metric_space_mode '{self.metric_space_mode}'. "
                "Use 'robust', 'standard', or 'none'."
            )
        if self.cluster_k_mode not in {"silhouette_auto", "fixed"}:
            raise ValueError(
                f"Unsupported cluster_k_mode '{self.cluster_k_mode}'. "
                "Use 'silhouette_auto' or 'fixed'."
            )
        if self.cluster_k_mode == "fixed":
            if self.cluster_k_fixed is None or int(self.cluster_k_fixed) < 1:
                raise ValueError("cluster_k_mode='fixed' requires cluster_k_fixed >= 1.")
        if int(self.cluster_k_max) < 1:
            raise ValueError("cluster_k_max must be >= 1.")
        if int(self.min_cluster_size) < 2:
            raise ValueError("min_cluster_size must be >= 2.")
        if int(self.silhouette_sample_size) < 2:
            raise ValueError("silhouette_sample_size must be >= 2.")
        if float(self.silhouette_tol) < 0:
            raise ValueError("silhouette_tol must be >= 0.")
        if self.neighbor_k_mode not in {"auto", "fixed"}:
            raise ValueError(
                f"Unsupported neighbor_k_mode '{self.neighbor_k_mode}'. "
                "Use 'auto' or 'fixed'."
            )
        if self.neighbor_k_mode == "fixed":
            if self.neighbor_k_value is None or int(self.neighbor_k_value) < 1:
                raise ValueError("neighbor_k_mode='fixed' requires neighbor_k_value >= 1.")
        if int(self.neighbor_k_max) < 1:
            raise ValueError("neighbor_k_max must be >= 1.")
        if self.lambda_distribution != "uniform":
            raise ValueError(
                f"Unsupported lambda_distribution '{self.lambda_distribution}'. "
                "Only 'uniform' is supported in v1."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_value"] = self.target_value
        return payload


def load_kmeans_smote_joint_config(
    *,
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[KMeansSmoteJointConfig, dict[str, Any], Optional[Path]]:
    raw_payload: dict[str, Any] = {}
    resolved_path: Optional[Path] = None
    if config_path is not None:
        resolved_path = Path(config_path)
        with open(resolved_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid YAML payload in {resolved_path}.")
        raw_payload = loaded
    if config is not None:
        raw_payload = dict(config)
    cfg = KMeansSmoteJointConfig.from_payload(raw_payload)
    return cfg, raw_payload, resolved_path


def _fit_metric_scaler(joint_values: np.ndarray, *, mode: str):
    if mode == "none":
        return None, joint_values.astype(np.float64, copy=True)
    if mode == "robust":
        scaler = RobustScaler()
    elif mode == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported metric space mode: {mode}")
    normalized = scaler.fit_transform(joint_values)
    normalized = np.asarray(normalized, dtype=np.float64)
    normalized[~np.isfinite(normalized)] = 0.0
    return scaler, normalized


def _resolve_target_count(*, observed_count: int, majority_count: int, config: KMeansSmoteJointConfig) -> int:
    if config.target_mode == "train_majority":
        return int(majority_count)
    if config.target_mode == "ratio_to_majority":
        return int(np.ceil(float(config.target_value) * float(majority_count)))
    if config.target_mode == "absolute_per_class":
        return int(config.target_value)
    raise ValueError(f"Unsupported target_mode '{config.target_mode}'.")


def _resolve_neighbor_k(*, cluster_size: int, config: KMeansSmoteJointConfig) -> int:
    if cluster_size <= 1:
        raise ValueError("cluster_size must be >= 2 to resolve neighbor_k.")
    if config.neighbor_k_mode == "fixed":
        return min(int(config.neighbor_k_value), cluster_size - 1)
    return min(int(config.neighbor_k_max), cluster_size - 1)


def _allocate_integer_counts(
    weights: np.ndarray,
    total: int,
) -> np.ndarray:
    if total <= 0:
        return np.zeros(len(weights), dtype=int)
    raw = weights * float(total)
    base = np.floor(raw).astype(int)
    remainder = int(total - int(base.sum()))
    if remainder > 0:
        frac = raw - base
        order = np.argsort(-frac, kind="mergesort")
        for idx in order[:remainder]:
            base[int(idx)] += 1
    return base


def _evaluate_k_grid(
    joint_norm: np.ndarray,
    *,
    n_class: int,
    config: KMeansSmoteJointConfig,
) -> tuple[int, dict[str, Any], Optional[str]]:
    report: dict[str, Any] = {
        "candidate_k_values": [],
        "candidate_scores": {},
        "status": "not_needed",
    }

    if config.cluster_k_mode == "fixed":
        k_selected = int(config.cluster_k_fixed)
        if k_selected <= 1:
            report["status"] = "fixed_k_equals_1"
            return 1, report, "fixed_k_equals_1"
        return k_selected, report, None

    k_max = min(int(config.cluster_k_max), int(np.floor(n_class / max(1, config.min_cluster_size))))
    if k_max < 2:
        report["status"] = "fallback_single_cluster_small_class"
        return 1, report, "class_too_small_for_multi_cluster"

    candidate_scores: list[tuple[int, float]] = []
    for k in range(2, k_max + 1):
        report["candidate_k_values"].append(int(k))
        try:
            labels = KMeans(n_clusters=k, random_state=config.synthetic_seed, n_init=10).fit_predict(joint_norm)
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                report["candidate_scores"][str(k)] = None
                continue
            sample_size = min(int(config.silhouette_sample_size), int(n_class))
            score = silhouette_score(
                joint_norm,
                labels,
                sample_size=sample_size,
                random_state=config.synthetic_seed,
            )
            if not np.isfinite(score):
                report["candidate_scores"][str(k)] = None
                continue
            score_float = float(score)
            report["candidate_scores"][str(k)] = score_float
            candidate_scores.append((int(k), score_float))
        except Exception:
            report["candidate_scores"][str(k)] = None

    if not candidate_scores:
        report["status"] = "fallback_single_cluster_no_valid_candidates"
        return 1, report, "no_valid_silhouette_candidates"

    best_score = max(score for _, score in candidate_scores)
    if best_score < float(config.silhouette_min_accept):
        report["status"] = "fallback_single_cluster_low_silhouette"
        report["best_silhouette"] = float(best_score)
        return 1, report, "silhouette_below_min_accept"

    best_candidates = [
        (k, score)
        for k, score in candidate_scores
        if abs(score - best_score) <= float(config.silhouette_tol)
    ]
    best_k = min(k for k, _ in best_candidates)
    report["status"] = "selected"
    report["best_silhouette"] = float(best_score)
    return int(best_k), report, None


def _build_condition_label_maps(class_values: list[int], mapping: Mapping[str, int] | None) -> tuple[dict[int, str], dict[str, int]]:
    label_to_id: dict[str, int] = {}
    if mapping:
        label_to_id = {str(label): int(idx) for label, idx in mapping.items()}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    for cls in class_values:
        id_to_label.setdefault(int(cls), f"type_{int(cls)}")
    return id_to_label, {label: idx for idx, label in id_to_label.items()}


def _validate_source_frames(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    *,
    condition_col: str,
) -> None:
    required_x = {"post_cleaning_index", condition_col}
    required_y = {"post_cleaning_index"}
    missing_x = [col for col in required_x if col not in X_train.columns]
    missing_y = [col for col in required_y if col not in y_train.columns]
    if missing_x:
        raise ValueError(f"X_train is missing required columns: {missing_x}")
    if missing_y:
        raise ValueError(f"y_train is missing required columns: {missing_y}")
    if "is_synth" in X_train.columns or "is_synth" in y_train.columns:
        raise ValueError("kmeans_smote_joint v1 requires a non-synthetic source bundle (is_synth absent in X/y).")


def generate_kmeans_smote_joint_samples(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    r_train: pd.DataFrame | None = None,
    *,
    condition_col: str = "type",
    config: KMeansSmoteJointConfig,
    condition_value_to_label_map: Mapping[int, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _validate_source_frames(X_train, y_train, condition_col=condition_col)

    X_train = X_train.sort_values("post_cleaning_index").reset_index(drop=True).copy()
    y_train = y_train.sort_values("post_cleaning_index").reset_index(drop=True).copy()

    merged = X_train.merge(
        y_train,
        on="post_cleaning_index",
        how="inner",
        validate="one_to_one",
        suffixes=("", "__y"),
    )
    if len(merged) != len(X_train) or len(merged) != len(y_train):
        raise ValueError("X_train and y_train do not align 1:1 on post_cleaning_index.")

    x_cols = [c for c in X_train.columns if c not in {"post_cleaning_index", condition_col, "is_synth"}]
    y_cols = [c for c in y_train.columns if c not in {"post_cleaning_index", "is_synth"}]
    if not x_cols:
        raise ValueError("No X feature columns available after excluding metadata.")
    if not y_cols:
        raise ValueError("No y feature columns available after excluding metadata.")

    counts = X_train[condition_col].astype(int).value_counts().sort_index()
    majority = int(counts.max())
    class_values = [int(cls) for cls in counts.index.tolist()]
    label_map, label_to_id = _build_condition_label_maps(
        class_values=class_values,
        mapping=None if condition_value_to_label_map is not None else None,
    )
    if condition_value_to_label_map is not None:
        label_map = {int(cls): str(label) for cls, label in condition_value_to_label_map.items()}
        for cls in class_values:
            label_map.setdefault(int(cls), f"type_{int(cls)}")
        label_to_id = {label: cls for cls, label in label_map.items()}

    rng = np.random.default_rng(int(config.synthetic_seed))
    next_index = int(X_train["post_cleaning_index"].max()) + 1
    synth_x_parts: list[pd.DataFrame] = []
    synth_y_parts: list[pd.DataFrame] = []
    class_reports: dict[str, Any] = {}

    for cls in class_values:
        class_label = label_map[int(cls)]
        class_df = merged[merged[condition_col].astype(int) == int(cls)].copy()
        observed_count = int(len(class_df))
        target_count = _resolve_target_count(
            observed_count=observed_count,
            majority_count=majority,
            config=config,
        )
        n_to_generate = max(0, int(target_count) - int(observed_count))

        joint_values = class_df[x_cols + y_cols].to_numpy(dtype=np.float64, copy=True)
        scaler, joint_norm = _fit_metric_scaler(joint_values, mode=config.metric_space_mode)
        k_selected, silhouette_report, fallback_reason = _evaluate_k_grid(
            joint_norm,
            n_class=observed_count,
            config=config,
        )

        if k_selected <= 1:
            labels = np.zeros(observed_count, dtype=int)
        else:
            labels = KMeans(
                n_clusters=int(k_selected),
                random_state=config.synthetic_seed,
                n_init=10,
            ).fit_predict(joint_norm)

        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        eligible_cluster_ids = [int(cluster_id) for cluster_id, size in cluster_sizes.items() if int(size) >= 2]

        if n_to_generate > 0 and not eligible_cluster_ids:
            message = (
                f"Class={class_label} requires {n_to_generate} synthetic rows but has no eligible clusters "
                "with size >= 2."
            )
            if config.error_if_no_eligible_cluster:
                raise RuntimeError(message)

        if not config.redistribute_singletons and n_to_generate > 0 and len(eligible_cluster_ids) != len(cluster_sizes):
            raise RuntimeError(
                f"Class={class_label} contains ineligible clusters and redistribution is disabled."
            )

        eligible_sizes = np.array([int(cluster_sizes.loc[cluster_id]) for cluster_id in eligible_cluster_ids], dtype=float)
        if eligible_sizes.size > 0 and n_to_generate > 0:
            cluster_alloc = _allocate_integer_counts(eligible_sizes / eligible_sizes.sum(), n_to_generate)
        else:
            cluster_alloc = np.zeros(len(eligible_cluster_ids), dtype=int)

        allocated_by_cluster = {
            str(cluster_id): int(cluster_alloc[pos])
            for pos, cluster_id in enumerate(eligible_cluster_ids)
            if int(cluster_alloc[pos]) > 0
        }

        neighbor_k_by_cluster: dict[str, int] = {}
        generated_blocks: list[np.ndarray] = []
        for pos, cluster_id in enumerate(eligible_cluster_ids):
            n_cluster_generate = int(cluster_alloc[pos])
            if n_cluster_generate <= 0:
                continue

            cluster_mask = labels == int(cluster_id)
            cluster_norm = joint_norm[cluster_mask]
            cluster_size = int(cluster_norm.shape[0])
            neighbor_k = _resolve_neighbor_k(cluster_size=cluster_size, config=config)
            neighbor_k_by_cluster[str(cluster_id)] = int(neighbor_k)

            nn = NearestNeighbors(n_neighbors=int(neighbor_k) + 1, metric="euclidean")
            nn.fit(cluster_norm)
            neighbor_indices = nn.kneighbors(cluster_norm, return_distance=False)

            for _ in range(n_cluster_generate):
                anchor_pos = int(rng.integers(0, cluster_size))
                anchor = cluster_norm[anchor_pos]
                candidates = [int(idx) for idx in neighbor_indices[anchor_pos].tolist() if int(idx) != anchor_pos]
                if not candidates:
                    raise RuntimeError(
                        f"Class={class_label}, cluster={cluster_id} has no valid neighbors for anchor={anchor_pos}."
                    )
                neighbor_pos = int(rng.choice(candidates))
                lam = float(rng.random())
                synth_norm = anchor + lam * (cluster_norm[neighbor_pos] - anchor)
                generated_blocks.append(np.asarray(synth_norm, dtype=np.float64))

        if generated_blocks:
            generated_norm = np.vstack(generated_blocks)
            generated_values = (
                scaler.inverse_transform(generated_norm)
                if scaler is not None
                else generated_norm
            )
            if not np.isfinite(generated_values).all():
                raise RuntimeError(f"Non-finite synthetic values produced for class={class_label}.")
            generated_values = np.asarray(generated_values, dtype=np.float64)
            idx_block = np.arange(next_index, next_index + len(generated_values), dtype=int)
            next_index += len(generated_values)

            X_syn = pd.DataFrame(generated_values[:, : len(x_cols)], columns=x_cols)
            X_syn.insert(0, condition_col, int(cls))
            X_syn.insert(0, "post_cleaning_index", idx_block)

            y_syn = pd.DataFrame(generated_values[:, len(x_cols):], columns=y_cols)
            y_syn.insert(0, "post_cleaning_index", idx_block)

            synth_x_parts.append(X_syn)
            synth_y_parts.append(y_syn)

        class_reports[class_label] = {
            "class_id": int(cls),
            "class_label": class_label,
            "observed_count": int(observed_count),
            "majority_count": int(majority),
            "target_count": int(target_count),
            "n_to_generate": int(n_to_generate),
            "k_selected": int(k_selected),
            "silhouette": silhouette_report,
            "fallback_reason": fallback_reason,
            "cluster_sizes": {str(int(cluster_id)): int(size) for cluster_id, size in cluster_sizes.items()},
            "eligible_cluster_ids": [int(cluster_id) for cluster_id in eligible_cluster_ids],
            "allocated_by_cluster": allocated_by_cluster,
            "neighbor_k_by_cluster": neighbor_k_by_cluster,
            "generated_count": int(n_to_generate),
        }

    if synth_x_parts:
        X_synth = pd.concat(synth_x_parts, axis=0, ignore_index=True).sort_values("post_cleaning_index").reset_index(drop=True)
        y_synth = pd.concat(synth_y_parts, axis=0, ignore_index=True).sort_values("post_cleaning_index").reset_index(drop=True)
    else:
        X_synth = X_train.iloc[0:0].copy()
        y_synth = y_train.iloc[0:0].copy()

    report = {
        "synthetic_policy_family": "kmeans_smote_joint",
        "synthetic_seed": int(config.synthetic_seed),
        "synthetic_policy_config_id": config.synthetic_policy_config_id,
        "metric_space_mode": config.metric_space_mode,
        "target_mode": config.target_mode,
        "target_value": config.target_value,
        "majority_count": int(majority),
        "x_cols": list(x_cols),
        "y_cols": list(y_cols),
        "condition_col": condition_col,
        "condition_value_to_label_map": {str(int(cls)): label_map[int(cls)] for cls in class_values},
        "class_reports": class_reports,
        "added_by_class": {
            class_label: int(payload["generated_count"])
            for class_label, payload in class_reports.items()
        },
        "resolved_target_by_class": {
            class_label: int(payload["target_count"])
            for class_label, payload in class_reports.items()
        },
        "resolved_cluster_k_by_class": {
            class_label: int(payload["k_selected"])
            for class_label, payload in class_reports.items()
        },
        "resolved_neighbor_k_by_class": {
            class_label: dict(payload["neighbor_k_by_cluster"])
            for class_label, payload in class_reports.items()
        },
        "silhouette_by_class": {
            class_label: dict(payload["silhouette"].get("candidate_scores", {}))
            for class_label, payload in class_reports.items()
        },
        "cluster_min_size": int(config.min_cluster_size),
    }
    return X_synth, y_synth, report


def augment_canonical_bundle_with_kmeans_smote(
    *,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    y_test: pd.DataFrame,
    r_train: pd.DataFrame,
    r_val: pd.DataFrame,
    r_test: pd.DataFrame,
    config: KMeansSmoteJointConfig,
    condition_col: str = "type",
    condition_value_to_label_map: Mapping[int, str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    X_synth, y_synth, report = generate_kmeans_smote_joint_samples(
        X_train=X_train,
        y_train=y_train,
        r_train=r_train,
        condition_col=condition_col,
        config=config,
        condition_value_to_label_map=condition_value_to_label_map,
    )

    X_train_aug = X_train.copy()
    X_train_aug.insert(2, "is_synth", False)
    y_train_aug = y_train.copy()
    y_train_aug.insert(1, "is_synth", False)

    if len(X_synth) > 0:
        X_synth = X_synth.copy()
        X_synth.insert(2, "is_synth", True)
        y_synth = y_synth.copy()
        y_synth.insert(1, "is_synth", True)
        X_train_aug = pd.concat([X_train_aug, X_synth], axis=0, ignore_index=True)
        y_train_aug = pd.concat([y_train_aug, y_synth], axis=0, ignore_index=True)

    X_train_aug = X_train_aug.sort_values("post_cleaning_index").reset_index(drop=True)
    y_train_aug = y_train_aug.sort_values("post_cleaning_index").reset_index(drop=True)

    r_train_aug = r_train.copy()
    r_train_aug["is_synth"] = False
    if len(X_synth) > 0:
        synth_removed = pd.DataFrame(columns=r_train_aug.columns)
        synth_removed["post_cleaning_index"] = X_synth["post_cleaning_index"].values
        if "split" in synth_removed.columns:
            synth_removed["split"] = "train"
        if "split_id" in synth_removed.columns:
            synth_removed["split_id"] = r_train_aug["split_id"].iloc[0] if len(r_train_aug) else None
        if "split_row_id" in synth_removed.columns:
            synth_removed["split_row_id"] = [f"synth_train_kmeans_{i}" for i in range(len(synth_removed))]
        synth_removed["is_synth"] = True
        r_train_aug = pd.concat([r_train_aug, synth_removed], axis=0, ignore_index=True, sort=False)

    augmented = {
        "X_train": X_train_aug,
        "X_val": X_val.copy(),
        "X_test": X_test.copy(),
        "y_train": y_train_aug,
        "y_val": y_val.copy(),
        "y_test": y_test.copy(),
        "r_train": r_train_aug,
        "r_val": r_val.copy(),
        "r_test": r_test.copy(),
    }
    report["generated_rows_total"] = int(len(X_synth))
    return augmented, report
