"""Validation physical."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.plant.operating_point import ClarifierParameters
    from surrogate_optimization.surrogate.projection import NetworkLayout
import math
import numpy as np


def reduce_mechanistic_responses(
    responses: np.ndarray,
    layer_count: int,
    *,
    layer_volumes_m3: np.ndarray | None = None,
) -> np.ndarray:
    """Map full mechanistic responses to ``(m,c_1,...,c_N,g_E,g_U,M_cl)``.

    The full response remains the immutable generation/checkpoint format.  The
    returned array is the statistical response and replaces all layer-wise TSS
    coordinates by their volume-weighted clarifier solids inventory.
    """
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES

    count = int(layer_count)
    if count < 1 or count != layer_count:
        raise ValueError("layer_count must be a positive integer")
    values = np.asarray(responses, dtype=np.float64)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    shared_count = (N_STAGES + 3) * N_COMPONENTS
    expected = shared_count + count
    if (
        values.ndim != 2
        or values.shape[1] != expected
        or (not np.all(np.isfinite(values)))
    ):
        raise ValueError(
            f"mechanistic responses must be finite with {expected} coordinates"
        )
    if layer_volumes_m3 is None:
        volumes = np.full(count, 6000.0 / count)
    else:
        volumes = np.asarray(layer_volumes_m3, dtype=np.float64)
    if (
        volumes.shape != (count,)
        or not np.all(np.isfinite(volumes))
        or np.any(volumes <= 0.0)
    ):
        raise ValueError(
            "layer_volumes_m3 must contain one positive finite volume per layer"
        )
    if not np.all(volumes == volumes[0]):
        raise ValueError(
            "layer_volumes_m3 must be equal because the reduced projection uses an equal-volume layer envelope"
        )
    inventory = values[:, shared_count:] @ volumes
    reduced = np.concatenate((values[:, :shared_count], inventory[:, None]), axis=1)
    return reduced[0] if single else reduced


def _operators(
    controls: np.ndarray,
    influent: np.ndarray,
    layout: NetworkLayout,
    overflow_tss_closure: float | None = None,
):
    from surrogate_optimization.plant.definitions import INVARIANT_MATRIX
    from surrogate_optimization.plant.definitions import TSS_VECTOR
    from surrogate_optimization.surrogate.projection import build_network_operators

    return build_network_operators(
        influent,
        internal_recycle=float(controls[4]),
        return_recycle=float(controls[5]),
        waste_fraction=float(controls[6]),
        invariant_operator=INVARIANT_MATRIX,
        tss_weights=TSS_VECTOR,
        layout=layout,
        overflow_tss_closure=overflow_tss_closure,
    )


def response_coordinate_names(layout: NetworkLayout) -> tuple[str, ...]:
    from surrogate_optimization.plant.definitions import COMPONENTS

    names = [f"mixer:{name}" for name in COMPONENTS]
    for stage in range(layout.stage_count):
        names.extend((f"reactor_{stage + 1}:{name}" for name in COMPONENTS))
    names.extend((f"overflow_flow:{name}" for name in COMPONENTS))
    names.extend((f"underflow_flow:{name}" for name in COMPONENTS))
    names.append("clarifier_inventory:TSS_mass")
    return tuple(names)


def violation_record(
    method: str,
    case: str,
    state: np.ndarray,
    controls: np.ndarray,
    influent: np.ndarray,
    layout: NetworkLayout,
    equality_scale: np.ndarray,
    inequality_scale: np.ndarray,
    state_scale: np.ndarray,
    overflow_tss_closure: float | None = None,
) -> dict[str, object]:
    from surrogate_optimization.plant.definitions import INVARIANT_MATRIX
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import TSS_VECTOR
    from surrogate_optimization.plant.model import PHYSICAL_BALANCE_TOLERANCE

    operators = _operators(
        controls, influent, layout, overflow_tss_closure=overflow_tss_closure
    )
    equality_physical = operators.equality_matrix @ state - operators.equality_rhs
    equality = equality_physical / equality_scale
    inequality_physical = operators.inequality_matrix @ state
    inequality = inequality_physical / inequality_scale
    particulate_inequality = inequality[: len(layout.particulate_indices)]
    inventory_inequality = inequality[len(layout.particulate_indices) :]
    negative = np.maximum(-state / state_scale, 0.0)
    invariant_count = INVARIANT_MATRIX.shape[0]
    family_slices = {
        "mixer_component": slice(0, N_COMPONENTS),
        "reactor_invariant": slice(
            N_COMPONENTS, N_COMPONENTS + layout.stage_count * invariant_count
        ),
        "clarifier_component": slice(
            N_COMPONENTS + layout.stage_count * invariant_count,
            2 * N_COMPONENTS + layout.stage_count * invariant_count,
        ),
        "soluble_passthrough": slice(
            2 * N_COMPONENTS + layout.stage_count * invariant_count,
            2 * N_COMPONENTS
            + layout.stage_count * invariant_count
            + len(layout.soluble_indices),
        ),
    }
    if overflow_tss_closure is not None:
        family_slices["overflow_tss_closure"] = slice(
            equality_physical.size - 1, equality_physical.size
        )
    family_maxima = {
        name: float(np.max(np.abs(equality[index])))
        for name, index in family_slices.items()
    }
    final = state[layout.reactor_slice(layout.stage_count - 1)]
    overflow = state[layout.overflow_flow_slice]
    underflow = state[layout.underflow_flow_slice]
    q_u = float(controls[5] + controls[6])
    external_terms = np.vstack(
        (
            INVARIANT_MATRIX @ influent,
            -(INVARIANT_MATRIX @ overflow),
            -(controls[6] / q_u) * (INVARIANT_MATRIX @ underflow),
        )
    )
    external_scaled = np.abs(np.sum(external_terms, axis=0)) / np.maximum(
        1.0, np.max(np.abs(external_terms), axis=0)
    )
    family_maxima["external_invariant"] = float(np.max(external_scaled))
    base_equality_count = equality.size - int(overflow_tss_closure is not None)
    combined_mass = np.concatenate(
        (np.abs(equality[:base_equality_count]), external_scaled)
    )
    names = response_coordinate_names(layout)
    negative_indices = np.flatnonzero(negative > 1e-10)
    nonlinear_status = "not_applicable_to_reduced_response"
    nonlinear_max = math.nan
    rate_negativity = math.nan
    return {
        "case": case,
        "method": method,
        "mass_conservation_violation_max": float(np.max(combined_mass)),
        "mass_conservation_violation_count": int(
            np.count_nonzero(combined_mass > PHYSICAL_BALANCE_TOLERANCE)
        ),
        **{f"mass_{name}_max": value for name, value in family_maxima.items()},
        "mass_physical_residual_max": float(np.max(np.abs(equality_physical))),
        "network_inequality_violation_max": float(np.max(np.maximum(inequality, 0.0))),
        "network_inequality_violation_count": int(np.count_nonzero(inequality > 1e-08)),
        "particulate_densification_violation_max": float(
            np.max(np.maximum(particulate_inequality, 0.0))
        ),
        "clarifier_inventory_bound_violation_max": float(
            np.max(np.maximum(inventory_inequality, 0.0))
        ),
        "overflow_tss_closure_mg_L": math.nan
        if overflow_tss_closure is None
        else float(overflow_tss_closure),
        "overflow_tss_closure_residual_g_m3": math.nan
        if overflow_tss_closure is None
        else float(
            TSS_VECTOR @ overflow / (1.0 - float(controls[6]))
            - float(overflow_tss_closure)
        ),
        "overflow_tss_closure_scaled_residual": math.nan
        if overflow_tss_closure is None
        else float(equality[-1]),
        "nonnegativity_violation_max": float(np.max(negative)),
        "nonnegativity_violation_count": int(np.count_nonzero(negative > 1e-10)),
        "negative_coordinates": ";".join((names[index] for index in negative_indices)),
        "minimum_coordinate": float(np.min(state)),
        "nonlinear_balance_residual_max": nonlinear_max,
        "rate_nonnegativity_violation_max": rate_negativity,
        "nonlinear_audit_status": nonlinear_status,
    }


def clarifier_for_layers(layer_count: int) -> ClarifierParameters:
    """Return the fixed 4 m/6000 m3 vessel discretized into ``layer_count`` cells."""
    from surrogate_optimization.plant.operating_point import ClarifierParameters

    return ClarifierParameters(
        layer_count=layer_count,
        feed_layer=(layer_count - 1) // 2,
        layer_volume=6000.0 / layer_count,
    )


def _response_blocks(layout: NetworkLayout) -> dict[str, np.ndarray]:
    blocks: dict[str, np.ndarray] = {
        "mixer": np.arange(layout.mixer_slice.start, layout.mixer_slice.stop)
    }
    for stage in range(layout.stage_count):
        block = layout.reactor_slice(stage)
        blocks[f"reactor_{stage + 1}"] = np.arange(block.start, block.stop)
    blocks["clarifier_overflow"] = np.arange(
        layout.overflow_flow_slice.start, layout.overflow_flow_slice.stop
    )
    blocks["clarifier_underflow"] = np.arange(
        layout.underflow_flow_slice.start, layout.underflow_flow_slice.stop
    )
    blocks["clarifier_inventory"] = np.asarray([layout.inventory_index])
    blocks["clarifier_complete"] = np.arange(
        layout.overflow_flow_slice.start, layout.inventory_slice.stop
    )
    blocks["complete_response"] = np.arange(layout.state_size)
    return blocks


def _prediction_metric_rows(
    method: str,
    prediction: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
    layout: NetworkLayout,
) -> list[dict[str, object]]:
    names = response_coordinate_names(layout)
    rows: list[dict[str, object]] = []
    error = prediction - reference
    standardized = error / scale
    for block_name, indices in _response_blocks(layout).items():
        block_error = error[:, indices]
        block_standardized = standardized[:, indices]
        target = reference[:, indices]
        target_centered = target - np.mean(target, axis=0)
        ss_res = np.sum(np.square(block_error), axis=0)
        ss_tot = np.sum(np.square(target_centered), axis=0)
        coordinate_r2 = np.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, np.nan)
        rows.append(
            {
                "method": method,
                "block": block_name,
                "coordinate": "ALL",
                "sample_count": int(reference.shape[0]),
                "coordinate_count": int(indices.size),
                "rmse": float(np.sqrt(np.mean(np.square(block_error)))),
                "mae": float(np.mean(np.abs(block_error))),
                "bias": float(np.mean(block_error)),
                "nrmse": float(np.sqrt(np.mean(np.square(block_standardized)))),
                "nmae": float(np.mean(np.abs(block_standardized))),
                "r2_mean": float(np.nanmean(coordinate_r2)),
            }
        )
        for local, coordinate in enumerate(indices):
            coordinate_error = block_error[:, local]
            coordinate_standardized = block_standardized[:, local]
            rows.append(
                {
                    "method": method,
                    "block": block_name,
                    "coordinate": names[int(coordinate)],
                    "sample_count": int(reference.shape[0]),
                    "coordinate_count": 1,
                    "rmse": float(np.sqrt(np.mean(np.square(coordinate_error)))),
                    "mae": float(np.mean(np.abs(coordinate_error))),
                    "bias": float(np.mean(coordinate_error)),
                    "nrmse": float(
                        np.sqrt(np.mean(np.square(coordinate_standardized)))
                    ),
                    "nmae": float(np.mean(np.abs(coordinate_standardized))),
                    "r2_mean": float(coordinate_r2[local]),
                }
            )
    return rows


def _overflow_closure_metric_rows(
    prediction: np.ndarray, reference: np.ndarray, development_reference: np.ndarray
) -> list[dict[str, object]]:
    """Return concentration/log metrics with development-frozen tail strata."""
    from surrogate_optimization.config import OVERFLOW_TSS_HIGH_QUANTILE
    from surrogate_optimization.config import OVERFLOW_TSS_LOW_QUANTILE

    predicted = np.asarray(prediction, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    development = np.asarray(development_reference, dtype=np.float64)
    if (
        predicted.shape != exact.shape
        or predicted.ndim != 1
        or development.ndim != 1
        or (not np.all(np.isfinite(predicted)))
        or (not np.all(np.isfinite(exact)))
        or (not np.all(np.isfinite(development)))
        or np.any(predicted <= 0.0)
        or np.any(exact <= 0.0)
        or np.any(development <= 0.0)
    ):
        raise ValueError(
            "overflow closure metric inputs must be finite positive vectors"
        )
    low = float(np.quantile(development, OVERFLOW_TSS_LOW_QUANTILE))
    high = float(np.quantile(development, OVERFLOW_TSS_HIGH_QUANTILE))
    strata = (
        ("all", np.ones(len(exact), dtype=bool), np.ones(len(development), dtype=bool)),
        ("low_q25", exact <= low, development <= low),
        ("upper_q90", exact >= high, development >= high),
    )
    rows: list[dict[str, object]] = []
    for stratum, mask, development_mask in strata:
        for scale_name, values, mechanistic_responses, development_values in (
            ("concentration", predicted, exact, development),
            ("log", np.log(predicted), np.log(exact), np.log(development)),
        ):
            count = int(np.count_nonzero(mask))
            if count:
                error = values[mask] - mechanistic_responses[mask]
                centered = mechanistic_responses[mask] - np.mean(
                    mechanistic_responses[mask]
                )
                ss_tot = float(np.sum(np.square(centered)))
                frozen_scale = max(
                    1e-12, float(np.std(development_values[development_mask], ddof=0))
                )
                rmse = float(np.sqrt(np.mean(np.square(error))))
                mae = float(np.mean(np.abs(error)))
                bias = float(np.mean(error))
                r2 = (
                    math.nan
                    if ss_tot <= 0.0
                    else float(1.0 - np.sum(np.square(error)) / ss_tot)
                )
            else:
                rmse = mae = bias = r2 = frozen_scale = math.nan
            rows.append(
                {
                    "method": "log_overflow_closure",
                    "block": "clarifier_overflow_tss"
                    if scale_name == "concentration"
                    else "clarifier_overflow_tss_log",
                    "coordinate": "TSS"
                    if scale_name == "concentration"
                    else "log(TSS/1mgL)",
                    "stratum": stratum,
                    "sample_count": count,
                    "coordinate_count": 1,
                    "rmse": rmse,
                    "mae": mae,
                    "bias": bias,
                    "nrmse": rmse / frozen_scale if count else math.nan,
                    "nmae": mae / frozen_scale if count else math.nan,
                    "r2_mean": r2,
                    "development_low_q25_mg_L": low,
                    "development_upper_q90_mg_L": high,
                }
            )
    return rows
