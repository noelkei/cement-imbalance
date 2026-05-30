from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from data.utils import load_column_mapping_by_group


@dataclass(frozen=True)
class GroupProjectionSpec:
    key: str
    ilr_feature_names: list[str]
    component_names: list[str]
    input_indices: list[int]


@dataclass(frozen=True)
class FinalSemanticSurfaceSpec:
    direct_feature_names: list[str]
    groups: list[GroupProjectionSpec]
    final_feature_names: list[str]
    direct_output_indices: list[int]
    group_output_indices: dict[str, list[int]]


@lru_cache(maxsize=1)
def load_compositional_component_names() -> dict[str, list[str]]:
    _, grouped = load_column_mapping_by_group(verbose=False)
    return {
        "chem": [str(value) for value in (grouped.get("chem") or {}).values()],
        "phase": [str(value) for value in (grouped.get("phase") or {}).values()],
    }


def inverse_ilr_to_normalized_components(
    ilr_array: np.ndarray,
    *,
    n_components: int,
) -> np.ndarray:
    ilr_array = np.asarray(ilr_array, dtype=np.float32)
    if ilr_array.ndim != 2:
        raise ValueError("ILR array must be 2D.")
    if ilr_array.shape[1] != n_components - 1:
        raise ValueError(
            f"ILR array width {ilr_array.shape[1]} does not match n_components - 1 = {n_components - 1}."
        )

    n_samples = ilr_array.shape[0]
    logz = np.zeros((n_samples, n_components), dtype=np.float64)
    for j in reversed(range(n_components - 1)):
        coef = np.sqrt((j + 1) / (j + 2))
        mean_log_rest = np.mean(logz[:, j + 1 :], axis=1)
        logz[:, j] = mean_log_rest + (ilr_array[:, j].astype(np.float64, copy=False) / coef)
    row_max = np.max(logz, axis=1, keepdims=True)
    exp_shifted = np.exp(logz - row_max)
    exp_sum = exp_shifted.sum(axis=1, keepdims=True)
    normalized = exp_shifted / np.clip(exp_sum, 1e-12, None)
    return normalized.astype(np.float32, copy=False)


def build_final_semantic_surface_spec(
    input_feature_names: list[str],
    *,
    component_names_by_group: Mapping[str, list[str]] | None = None,
) -> FinalSemanticSurfaceSpec:
    component_names_by_group = (
        {str(k): [str(v) for v in values] for k, values in component_names_by_group.items()}
        if component_names_by_group is not None
        else load_compositional_component_names()
    )
    direct_feature_names: list[str] = []
    groups: list[GroupProjectionSpec] = []
    used_indices: set[int] = set()

    for group_key in ("chem", "phase"):
        component_names = list(component_names_by_group.get(group_key) or [])
        if not component_names:
            continue
        ilr_feature_names = [f"ilr_{group_key}_{idx + 1}" for idx in range(len(component_names) - 1)]
        input_indices = [idx for idx, name in enumerate(input_feature_names) if name in ilr_feature_names]
        if not input_indices:
            continue
        discovered_names = [str(input_feature_names[idx]) for idx in input_indices]
        groups.append(
            GroupProjectionSpec(
                key=str(group_key),
                ilr_feature_names=discovered_names,
                component_names=component_names,
                input_indices=input_indices,
            )
        )
        used_indices.update(input_indices)

    for idx, name in enumerate(input_feature_names):
        if idx not in used_indices:
            direct_feature_names.append(str(name))

    final_feature_names = list(direct_feature_names)
    direct_output_indices = list(range(len(direct_feature_names)))
    group_output_indices: dict[str, list[int]] = {}
    cursor = len(final_feature_names)
    for group in groups:
        start = cursor
        final_feature_names.extend(group.component_names)
        cursor = len(final_feature_names)
        group_output_indices[group.key] = list(range(start, cursor))

    return FinalSemanticSurfaceSpec(
        direct_feature_names=direct_feature_names,
        groups=groups,
        final_feature_names=final_feature_names,
        direct_output_indices=direct_output_indices,
        group_output_indices=group_output_indices,
    )


def distribute_event_deltas_to_surfaces(
    *,
    signed_event_deltas: np.ndarray,
    input_feature_names: list[str],
    original_input: np.ndarray,
    perturbed_inputs: np.ndarray,
    component_names_by_group: Mapping[str, list[str]] | None = None,
    fallback_input_indices: list[int] | None = None,
) -> dict[str, Any]:
    signed_event_deltas = np.asarray(signed_event_deltas, dtype=np.float32)
    original_input = np.asarray(original_input, dtype=np.float32)
    perturbed_inputs = np.asarray(perturbed_inputs, dtype=np.float32)
    if signed_event_deltas.ndim != 2:
        raise ValueError("signed_event_deltas must be 2D.")
    if original_input.ndim != 2:
        raise ValueError("original_input must be 2D.")
    if perturbed_inputs.ndim != 3:
        raise ValueError("perturbed_inputs must be 3D (events, samples, features).")
    event_count, n_samples, input_width = perturbed_inputs.shape
    if original_input.shape != (n_samples, input_width):
        raise ValueError("original_input shape does not match perturbed_inputs.")
    if signed_event_deltas.shape != (n_samples, event_count):
        raise ValueError("signed_event_deltas shape does not match perturbed event count.")
    if len(input_feature_names) != input_width:
        raise ValueError("input_feature_names width does not match input tensors.")

    spec = build_final_semantic_surface_spec(
        input_feature_names,
        component_names_by_group=component_names_by_group,
    )
    direct_input_indices = [
        idx for idx, name in enumerate(input_feature_names) if name in set(spec.direct_feature_names)
    ]
    input_signed = np.zeros((n_samples, input_width), dtype=np.float32)
    input_abs = np.zeros((n_samples, input_width), dtype=np.float32)
    final_signed = np.zeros((n_samples, len(spec.final_feature_names)), dtype=np.float32)
    final_abs = np.zeros((n_samples, len(spec.final_feature_names)), dtype=np.float32)

    original_expand = original_input[None, :, :]
    abs_input_delta = np.abs(perturbed_inputs - original_expand)
    input_weight_sums = abs_input_delta.sum(axis=2, keepdims=True)

    fallback_input_indices = list(fallback_input_indices or [])
    for event_idx in range(event_count):
        zero_mask = input_weight_sums[event_idx, :, 0] <= 1e-12
        if zero_mask.any() and event_idx < input_width:
            abs_input_delta[event_idx, zero_mask, event_idx] = 1.0
            input_weight_sums[event_idx, zero_mask, 0] = 1.0
        elif zero_mask.any() and event_idx < len(fallback_input_indices):
            fallback_idx = int(fallback_input_indices[event_idx])
            abs_input_delta[event_idx, zero_mask, fallback_idx] = 1.0
            input_weight_sums[event_idx, zero_mask, 0] = 1.0

    input_weights = abs_input_delta / np.clip(input_weight_sums, 1e-12, None)

    direct_index_to_output = {
        input_feature_names.index(name): out_idx
        for out_idx, name in enumerate(spec.direct_feature_names)
    }

    for event_idx in range(event_count):
        signed_event = signed_event_deltas[:, event_idx].astype(np.float32, copy=False)
        abs_event = np.abs(signed_event)
        input_signed += input_weights[event_idx] * signed_event[:, None]
        input_abs += input_weights[event_idx] * abs_event[:, None]

        final_magnitude = np.zeros((n_samples, len(spec.final_feature_names)), dtype=np.float32)
        for input_idx in direct_input_indices:
            output_idx = direct_index_to_output[input_idx]
            final_magnitude[:, output_idx] = abs_input_delta[event_idx, :, input_idx]

        for group in spec.groups:
            orig_group = original_input[:, group.input_indices]
            pert_group = perturbed_inputs[event_idx][:, group.input_indices]
            orig_comp = inverse_ilr_to_normalized_components(
                orig_group,
                n_components=len(group.component_names),
            )
            pert_comp = inverse_ilr_to_normalized_components(
                pert_group,
                n_components=len(group.component_names),
            )
            comp_delta = np.abs(pert_comp - orig_comp)
            output_indices = spec.group_output_indices[group.key]
            final_magnitude[:, output_indices] = comp_delta

        final_weight_sums = final_magnitude.sum(axis=1, keepdims=True)
        zero_mask = final_weight_sums[:, 0] <= 1e-12
        if zero_mask.any():
            if event_idx < input_width and event_idx in direct_index_to_output:
                fallback_output_idx = direct_index_to_output[event_idx]
                final_magnitude[zero_mask, fallback_output_idx] = 1.0
                final_weight_sums[zero_mask, 0] = 1.0
            else:
                final_magnitude[zero_mask, 0] = 1.0
                final_weight_sums[zero_mask, 0] = 1.0
        final_weights = final_magnitude / np.clip(final_weight_sums, 1e-12, None)
        final_signed += final_weights * signed_event[:, None]
        final_abs += final_weights * abs_event[:, None]

    return {
        "surface_spec": spec,
        "input_signed": input_signed,
        "input_abs": input_abs,
        "final_signed": final_signed,
        "final_abs": final_abs,
    }
