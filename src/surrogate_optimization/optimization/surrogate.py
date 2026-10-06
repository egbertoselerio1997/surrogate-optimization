"""Optimization surrogate."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.optimization.types import EngineeringLimits
    from surrogate_optimization.optimization.types import FeasibilityRecord
    from surrogate_optimization.optimization.types import FinalCandidateRecord
    from surrogate_optimization.surrogate.regression import LogOverflowTSSClosure
    from surrogate_optimization.surrogate.projection import NetworkLayout
    from surrogate_optimization.optimization.types import OuterRefinementRecord
    from surrogate_optimization.surrogate.projection import ProjectionResult
    from surrogate_optimization.surrogate.regression import QuadraticSurrogate
    from surrogate_optimization.optimization.types import SurrogateCase
    from surrogate_optimization.optimization.types import SurrogateNLP
    from surrogate_optimization.optimization.types import SurrogateNLPAssets
    from surrogate_optimization.optimization.types import SurrogateRouteResult
    from surrogate_optimization.optimization.types import SurrogateSolverSettings
    from surrogate_optimization.optimization.types import SurrogateStartResult
    from surrogate_optimization.optimization.types import SymbolicNetworkOperators
    from surrogate_optimization.optimization.types import TrustDiagnosticCallbacks
    from surrogate_optimization.optimization.types import _ExactEvaluation
from surrogate_optimization.plant.definitions import COMPOSITE_MATRIX
from surrogate_optimization.plant.definitions import INVARIANT_MATRIX
from surrogate_optimization.plant.definitions import TSS_VECTOR
from surrogate_optimization.config import engineering_parameters
from dataclasses import replace
from hashlib import sha256
from scipy import linalg
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from time import perf_counter
from typing import Any
from typing import Callable
import casadi as ca
import json
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
TrustRowCallback = Callable[[Any, Any, Any, Any], Any]
_ENGINEERING_PARAMETERS = engineering_parameters()
EXPRESSION_GAP = 1e-08
START_SEED = 271828
DEFAULT_OBJECTIVE_WEIGHTS = np.asarray([0.5, 0.15, 0.2, 0.05, 0.05, 0.05])
EXACT_QP_SINGLE_START_PROTOCOL = "seven_variable_exact_qp_single_start"
EXACT_QP_CENTER_START: tuple[float, ...] = (0.5,) * 7
LOCAL_CONVERGENCE_PROTOCOL = "exact_qp_two_scale_accelerated_feasible_poll"


def _vector(
    value: npt.ArrayLike, size: int, name: str, *, positive: bool = False
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {size}.")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    return array.copy()


def _matrix(value: npt.ArrayLike, shape: tuple[int, int], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite matrix with shape {shape}.")
    return array.copy()


def _flat(value: Any) -> FloatArray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _float_or_nan(value: Any) -> float:
    """Restore strict-JSON ``null`` failure placeholders as IEEE NaN."""
    return float("nan") if value is None else float(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _maximum_positive(values: npt.ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.max(np.maximum(array, 0.0), initial=0.0))


def _scaled_upper_residual(residual: Any, positive_scale: float) -> Any:
    """Return a dimensionless upper-inequality row without changing its sign."""
    scale = float(positive_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("an upper-constraint row scale must be finite and positive.")
    return residual / scale


def _normalized_limit_residual(value: Any, nonnegative_limit: float) -> Any:
    """Normalize ``value <= limit`` by its positive limit when available.

    A zero limit denotes an exact-zero diagnostic. Unit scaling is then the
    only finite fixed scale that preserves the feasible set without inserting
    an arbitrary floor into the scientific limit.
    """
    limit = float(nonnegative_limit)
    if not np.isfinite(limit) or limit < 0.0:
        raise ValueError("an upper-constraint limit must be finite and nonnegative.")
    scale = limit if limit > 0.0 else 1.0
    return _scaled_upper_residual(value - limit, scale)


def _safe_name(value: str) -> str:
    cleaned = "".join(
        (character if character.isalnum() else "_" for character in value)
    )
    return cleaned or "surrogate"


def regularized_leverage_contract(
    model: QuadraticSurrogate,
    development_controls: npt.ArrayLike,
    development_influents: npt.ArrayLike,
) -> tuple[FloatArray, float]:
    """Return ``M_R^{-1}`` and the maximum development-row leverage."""
    from surrogate_optimization.optimization.types import SurrogateNLPError

    controls = np.asarray(development_controls, dtype=np.float64)
    influents = np.asarray(development_influents, dtype=np.float64)
    if (
        controls.ndim != 2
        or influents.ndim != 2
        or controls.shape[0] != influents.shape[0]
    ):
        raise ValueError(
            "development input blocks must be two-dimensional with equal rows."
        )
    design = np.asarray(
        model.feature_map.transform(controls, influents), dtype=np.float64
    )
    rows, feature_count = design.shape
    penalty = np.ones(feature_count, dtype=np.float64)
    penalty[0] = 0.0
    matrix = design.T @ design + rows * float(model.ridge_penalty) * np.diag(penalty)
    try:
        factor = linalg.cho_factor(matrix, lower=True, check_finite=True)
        precision = linalg.cho_solve(factor, np.eye(feature_count), check_finite=True)
    except linalg.LinAlgError as exc:
        raise SurrogateNLPError(
            "the regularized feature matrix is not positive definite; a positive ridge penalty or full-column-rank design is required."
        ) from exc
    precision = 0.5 * (precision + precision.T)
    leverage = np.einsum("ij,jk,ik->i", design, precision, design)
    if not np.all(np.isfinite(leverage)) or np.min(leverage) < -1e-10:
        raise SurrogateNLPError(
            "regularized leverage calculation failed its finite/PSD check."
        )
    return (precision, float(np.max(leverage)))


def build_surrogate_assets(
    model: QuadraticSurrogate,
    development_controls: npt.ArrayLike,
    development_influents: npt.ArrayLike,
    development_responses: npt.ArrayLike,
    *,
    layout: NetworkLayout | None = None,
    invariant_operator: npt.ArrayLike = INVARIANT_MATRIX,
    tss_weights: npt.ArrayLike = TSS_VECTOR,
    quality_operator: npt.ArrayLike = COMPOSITE_MATRIX,
    correction_rms_threshold: float = 0.5,
    trust_callbacks: TrustDiagnosticCallbacks | None = None,
    split_rms_threshold: float | None = None,
    reactor_rms_threshold: float | None = None,
    engineering: EngineeringLimits | None = None,
    overflow_closure: LogOverflowTSSClosure | None = None,
    development_overflow_tss_closure: npt.ArrayLike | None = None,
) -> SurrogateNLPAssets:
    """Fit the projection scales, leverage contract, and quality normalization.

    Optional split/reactor callbacks must return *already scaled* residual
    rows.  Their thresholds are the frozen RMS limits computed from the
    development/out-of-fold calculations described by the supplement.
    """
    from surrogate_optimization.optimization.types import EngineeringLimits
    from surrogate_optimization.optimization.types import SurrogateNLPAssets
    from surrogate_optimization.optimization.types import TrustDiagnosticCallbacks
    from surrogate_optimization.optimization.types import TrustThresholds
    from surrogate_optimization.surrogate.projection import NetworkLayout
    from surrogate_optimization.surrogate.projection import fit_network_row_scales

    layout = layout or NetworkLayout()
    controls = np.asarray(development_controls, dtype=np.float64)
    influents = np.asarray(development_influents, dtype=np.float64)
    mechanistic_responses = np.asarray(development_responses, dtype=np.float64)
    if controls.ndim != 2 or controls.shape[1] != 7:
        raise ValueError("development_controls must have seven columns.")
    if influents.shape != (controls.shape[0], layout.component_count):
        raise ValueError("development_influents have inconsistent dimensions.")
    if mechanistic_responses.shape != (controls.shape[0], layout.state_size):
        raise ValueError("development_responses have inconsistent dimensions.")
    engineering_limits = engineering or EngineeringLimits()
    if overflow_closure is None:
        closure_prediction = None
        if development_overflow_tss_closure is not None:
            raise ValueError(
                "development_overflow_tss_closure requires an overflow closure model"
            )
    else:
        closure_prediction = np.asarray(
            development_overflow_tss_closure
            if development_overflow_tss_closure is not None
            else overflow_closure.predict(controls, influents),
            dtype=np.float64,
        )
        if (
            closure_prediction.shape != (len(controls),)
            or not np.all(np.isfinite(closure_prediction))
            or np.any(closure_prediction <= 0.0)
        ):
            raise ValueError("development overflow-TSS closure predictions are invalid")
    row_scales = fit_network_row_scales(
        mechanistic_responses,
        influents,
        internal_recycle=controls[:, 4],
        return_recycle=controls[:, 5],
        waste_fraction=controls[:, 6],
        invariant_operator=invariant_operator,
        tss_weights=tss_weights,
        layout=layout,
        clarifier_volume_m3=engineering_limits.clarifier_volume_m3,
        minimum_scale=1.0,
        overflow_tss_closure=closure_prediction,
    )
    precision, leverage_limit = regularized_leverage_contract(
        model, controls, influents
    )
    quality_matrix = np.asarray(quality_operator, dtype=np.float64)
    q_effluent = 1.0 - controls[:, 6]
    effluent = (
        mechanistic_responses[:, layout.overflow_flow_slice] / q_effluent[:, None]
    )
    quality_values = effluent @ quality_matrix.T
    quality_scale = np.maximum(1.0, np.std(quality_values, axis=0, ddof=0))
    thresholds = TrustThresholds(
        correction_rms=float(correction_rms_threshold),
        regularized_leverage=leverage_limit,
        split_rms=split_rms_threshold,
        reactor_rms=reactor_rms_threshold,
    )
    return SurrogateNLPAssets(
        model=model,
        layout=layout,
        invariant_operator=np.asarray(invariant_operator, dtype=np.float64),
        tss_weights=np.asarray(tss_weights, dtype=np.float64),
        row_scales=row_scales,
        leverage_precision=precision,
        trust_thresholds=thresholds,
        quality_operator=quality_matrix,
        quality_scale=quality_scale,
        overflow_closure=overflow_closure,
        trust_callbacks=trust_callbacks or TrustDiagnosticCallbacks(),
        engineering=engineering_limits,
    )


def symbolic_quadratic_prediction(
    model: QuadraticSurrogate, controls: ca.MX, influent: ca.MX
) -> tuple[ca.MX, ca.MX]:
    """Construct the exact fitted feature serialization and raw response."""
    feature_map = model.feature_map
    if controls.numel() != 7 or influent.numel() != feature_map.influent_count:
        raise ValueError("symbolic predictor inputs have inconsistent dimensions.")
    decision_z = (controls - ca.DM(feature_map.decision_center)) / ca.DM(
        feature_map.decision_scale
    )
    influent_z = (influent - ca.DM(feature_map.influent_center)) / ca.DM(
        feature_map.influent_scale
    )
    terms: list[Any] = [decision_z, influent_z]
    terms.extend(
        (
            decision_z[j] * decision_z[k]
            for j in range(feature_map.decision_count)
            for k in range(j, feature_map.decision_count)
        )
    )
    terms.extend(
        (
            influent_z[j] * influent_z[k]
            for j in range(feature_map.influent_count)
            for k in range(j, feature_map.influent_count)
        )
    )
    terms.extend(
        (
            decision_z[j] * influent_z[k]
            for j in range(feature_map.decision_count)
            for k in range(feature_map.influent_count)
        )
    )
    unscaled = ca.vertcat(*terms)
    standardized = (unscaled - ca.DM(feature_map.term_center)) / ca.DM(
        feature_map.term_scale
    )
    phi = ca.vertcat(1.0, standardized)
    raw_standardized = ca.DM(model.coefficients) @ phi
    raw = ca.DM(model.response_center) + ca.DM(model.response_scale) * raw_standardized
    return (raw, phi)


def symbolic_network_operators(
    controls: ca.MX, influent: ca.MX, assets: SurrogateNLPAssets
) -> SymbolicNetworkOperators:
    """Assemble the symbolic model H/b/G operators."""
    from surrogate_optimization.optimization.types import SymbolicNetworkOperators

    layout = assets.layout
    component_count = layout.component_count
    state_count = layout.state_size
    invariant = assets.invariant_operator
    invariant_count = invariant.shape[0]
    equality_count = assets.equality_count
    equality = ca.MX.zeros(equality_count, state_count)
    rhs = ca.MX.zeros(equality_count, 1)
    identity = ca.DM.eye(component_count)
    internal, returned, waste = (controls[4], controls[5], controls[6])
    underflow = returned + waste
    primary = 1.0 + internal + returned
    clarifier = 1.0 + returned
    effluent = 1.0 - waste
    row = 0
    equality[row : row + component_count, layout.mixer_slice] = primary * identity
    equality[
        row : row + component_count, layout.reactor_slice(layout.stage_count - 1)
    ] = -internal * identity
    equality[row : row + component_count, layout.underflow_flow_slice] = (
        -returned / underflow * identity
    )
    rhs[row : row + component_count] = influent
    row += component_count
    previous = layout.mixer_slice
    invariant_dm = ca.DM(invariant)
    for stage in range(layout.stage_count):
        current = layout.reactor_slice(stage)
        equality[row : row + invariant_count, current] = invariant_dm
        equality[row : row + invariant_count, previous] = -invariant_dm
        row += invariant_count
        previous = current
    final_reactor = layout.reactor_slice(layout.stage_count - 1)
    equality[row : row + component_count, layout.overflow_flow_slice] = identity
    equality[row : row + component_count, layout.underflow_flow_slice] = identity
    equality[row : row + component_count, final_reactor] = -clarifier * identity
    row += component_count
    for component in layout.soluble_indices:
        equality[row, layout.underflow_flow_slice.start + component] = 1.0
        equality[row, final_reactor.start + component] = -underflow
        row += 1
    overflow_closure = getattr(assets, "overflow_closure", None)
    if overflow_closure is not None:
        log_prediction, _ = symbolic_quadratic_prediction(
            overflow_closure.model, controls, influent
        )
        overflow_tss = overflow_closure.reference_concentration * ca.exp(
            log_prediction[0]
        )
        equality[row, layout.overflow_flow_slice] = ca.DM(assets.tss_weights).T
        rhs[row] = effluent * overflow_tss
        row += 1
    if row != equality_count:
        raise AssertionError("symbolic equality-row construction is inconsistent.")
    inequality = ca.MX.zeros(layout.inequality_count, state_count)
    row = 0
    for component in layout.particulate_indices:
        inequality[row, final_reactor.start + component] = underflow
        inequality[row, layout.underflow_flow_slice.start + component] = -1.0
        row += 1
    endpoint_layer_volume = assets.engineering.clarifier_volume_m3 / layout.layer_count
    remaining_volume = assets.engineering.clarifier_volume_m3 - endpoint_layer_volume
    tss = ca.DM(assets.tss_weights).T
    inequality[row, layout.overflow_flow_slice] = underflow * remaining_volume * tss
    inequality[row, layout.underflow_flow_slice] = (
        effluent * endpoint_layer_volume * tss
    )
    inequality[row, layout.inventory_index] = -effluent * underflow
    row += 1
    inequality[row, layout.overflow_flow_slice] = (
        -underflow * endpoint_layer_volume * tss
    )
    inequality[row, layout.underflow_flow_slice] = -effluent * remaining_volume * tss
    inequality[row, layout.inventory_index] = effluent * underflow
    row += 1
    if row != layout.inequality_count:
        raise AssertionError("symbolic inequality-row construction is inconsistent.")
    return SymbolicNetworkOperators(
        equality, rhs, inequality, primary, clarifier, underflow, effluent
    )


def _engineering_expressions(
    controls: ca.MX, state: ca.MX, assets: SurrogateNLPAssets
) -> tuple[ca.MX, tuple[str, ...], ca.MX]:
    layout = assets.layout
    limits = assets.engineering
    tss = ca.DM(assets.tss_weights)
    final = state[layout.reactor_slice(layout.stage_count - 1)]
    overflow = state[layout.overflow_flow_slice]
    underflow_flow = state[layout.underflow_flow_slice]
    clarifier_inventory = state[layout.inventory_index]
    returned, waste = (controls[5], controls[6])
    q_underflow = returned + waste
    q_effluent = 1.0 - waste
    q_clarifier = 1.0 + returned
    feed_tss = ca.dot(tss, final)
    underflow_tss = ca.dot(tss, underflow_flow) / q_underflow
    external_loss = (
        ca.dot(tss, overflow) + waste * ca.dot(tss, underflow_flow) / q_underflow
    )
    stage_volume = limits.fresh_flow_m3_d * controls[0] / (24.0 * layout.stage_count)
    reactor_inventory = 0.0
    for stage in range(layout.stage_count):
        reactor_inventory += stage_volume * ca.dot(
            tss, state[layout.reactor_slice(stage)]
        )
    inventory = reactor_inventory + clarifier_inventory
    sor = limits.fresh_flow_m3_d * q_effluent / limits.clarifier_area_m2
    slr = (
        0.001
        * limits.fresh_flow_m3_d
        * q_clarifier
        * feed_tss
        / limits.clarifier_area_m2
    )
    rows: list[Any] = [
        _scaled_upper_residual(
            limits.external_loss_min_g_m3 - external_loss, limits.external_loss_min_g_m3
        ),
        _scaled_upper_residual(
            underflow_tss - limits.underflow_tss_upper_g_m3,
            limits.underflow_tss_upper_g_m3,
        ),
        _scaled_upper_residual(
            limits.feed_tss_min_g_m3 - feed_tss, limits.feed_tss_min_g_m3
        ),
    ]
    names = ["external_solids_loss_guard", "underflow_tss_upper", "feed_tss_lower"]
    srt = inventory / (limits.fresh_flow_m3_d * external_loss)
    quantities = ca.vertcat(
        srt, sor, slr, underflow_tss, feed_tss, external_loss, inventory
    )
    return (ca.vertcat(*rows), tuple(names), quantities)


def _objective_expressions(
    controls: ca.MX,
    state: ca.MX,
    objective_weights: ca.MX,
    quality_weights: ca.MX,
    assets: SurrogateNLPAssets,
) -> tuple[ca.MX, ca.MX]:
    layout = assets.layout
    lower = assets.theta_lower
    span = assets.theta_span
    q_effluent = 1.0 - controls[6]
    effluent = state[layout.overflow_flow_slice] / q_effluent
    quality_composites = ca.DM(assets.quality_operator) @ effluent
    quality = ca.dot(quality_weights, quality_composites / ca.DM(assets.quality_scale))
    hrt = (controls[0] - lower[0]) / span[0]
    aeration = controls[0] * ca.sum1(controls[1:4]) / (assets.theta_upper[0] * 3.0)
    internal = (controls[4] - lower[4]) / span[4]
    returned = (controls[5] - lower[5]) / span[5]
    q_underflow = controls[5] + controls[6]
    underflow_tss = (
        ca.dot(ca.DM(assets.tss_weights), state[layout.underflow_flow_slice])
        / q_underflow
    )
    wasting = (
        controls[6]
        * underflow_tss
        / (assets.theta_upper[6] * assets.engineering.underflow_tss_upper_g_m3)
    )
    components = ca.vertcat(quality, hrt, aeration, internal, returned, wasting)
    return (ca.dot(objective_weights, components), components)


def _callback_rows(
    callback: TrustRowCallback,
    controls: ca.MX,
    raw: ca.MX,
    projected: ca.MX,
    influent: ca.MX,
    name: str,
) -> ca.MX:
    try:
        result = callback(controls, raw, projected, influent)
        if isinstance(result, (tuple, list)):
            result = ca.vertcat(*result)
        rows = ca.vec(result)
    except Exception as exc:
        raise ValueError(
            f"{name} trust callback failed during symbolic construction: {exc}"
        ) from exc
    if rows.numel() < 1:
        raise ValueError(f"{name} trust callback returned no rows.")
    return rows


def _trust_expressions(
    controls: ca.MX,
    raw: ca.MX,
    projected: ca.MX,
    u: ca.MX,
    influent: ca.MX,
    phi: ca.MX,
    assets: SurrogateNLPAssets,
) -> tuple[ca.MX, tuple[str, ...], ca.MX]:
    thresholds = assets.trust_thresholds
    correction_squared = ca.dot(u, u) / assets.layout.state_size
    leverage = ca.mtimes([phi.T, ca.DM(assets.leverage_precision), phi])
    rows: list[Any] = [
        _normalized_limit_residual(correction_squared, thresholds.correction_rms**2),
        _normalized_limit_residual(leverage, thresholds.regularized_leverage),
    ]
    values: list[Any] = [correction_squared, leverage]
    names = ["correction", "leverage"]
    specifications = (
        ("split", assets.trust_callbacks.split_rows, thresholds.split_rms),
        ("reactor", assets.trust_callbacks.reactor_rows, thresholds.reactor_rms),
    )
    for name, callback, threshold in specifications:
        if callback is None or threshold is None:
            continue
        residual = _callback_rows(callback, controls, raw, projected, influent, name)
        squared = ca.dot(residual, residual) / residual.numel()
        rows.append(_normalized_limit_residual(squared, threshold**2))
        values.append(squared)
        names.append(name)
    for specification in assets.trust_callbacks.additional:
        residual = _callback_rows(
            specification.callback,
            controls,
            raw,
            projected,
            influent,
            specification.name,
        )
        squared = ca.dot(residual, residual) / residual.numel()
        rows.append(_normalized_limit_residual(squared, specification.rms_threshold**2))
        values.append(squared)
        names.append(specification.name)
    return (ca.vertcat(*rows), tuple(names), ca.vertcat(*values))


def _update_overflow_closure_digest(
    digest: Any, closure: LogOverflowTSSClosure | None
) -> None:
    if closure is None:
        digest.update(b"\x00no-log-overflow-closure\x00")
        return
    digest.update(b"\x00log-overflow-closure-v1\x00")
    feature_map = closure.model.feature_map
    for value in (
        feature_map.decision_center,
        feature_map.decision_scale,
        feature_map.influent_center,
        feature_map.influent_scale,
        feature_map.term_center,
        feature_map.term_scale,
        closure.model.response_center,
        closure.model.response_scale,
        closure.model.coefficients,
        np.asarray(
            [
                feature_map.variance_relative_tolerance,
                closure.model.ridge_penalty,
                closure.reference_concentration,
            ]
        ),
    ):
        digest.update(np.asarray(value, dtype="<f8").tobytes(order="C"))


def surrogate_exact_qp_resume_contract(
    assets: SurrogateNLPAssets, case: SurrogateCase, settings: SurrogateSolverSettings
) -> str:
    """Fingerprint the direct seven-variable, single-start protocol.

    This token is deliberately disjoint from
    :func:`surrogate_start_resume_contract`: a checkpoint produced by the
    embedded-KKT continuation route can never be mistaken for a direct
    exact-QP local-optimization result. IPOPT-only settings are omitted
    because this protocol does not construct or call an IPOPT solver.
    """
    digest = sha256()
    digest.update(b"surrogate-surrogate-exact-qp-single-start-resume-v1\x00")
    digest.update(EXACT_QP_SINGLE_START_PROTOCOL.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(case.case_id.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(
        np.asarray(case.parameter_vector(assets), dtype="<f8").tobytes(order="C")
    )
    digest.update(np.asarray(EXACT_QP_CENTER_START, dtype="<f8").tobytes(order="C"))
    _update_overflow_closure_digest(digest, assets.overflow_closure)
    digest.update(
        json.dumps(
            {
                "final_upper_tolerance": settings.final_upper_tolerance,
                "outer_maximum_iterations": settings.outer_maximum_iterations,
                "outer_function_tolerance": settings.outer_function_tolerance,
                "perform_outer_refinement": settings.perform_outer_refinement,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def build_surrogate_expression_graph(
    assets: SurrogateNLPAssets,
    tau: float,
    *,
    settings: SurrogateSolverSettings | None = None,
    name: str = "surrogate",
) -> SurrogateNLP:
    """Build the fixed expression graph used by the exact cold-QP outer search."""
    from surrogate_optimization.optimization.types import SurrogateNLP
    from surrogate_optimization.optimization.types import SurrogateSolverSettings

    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive.")
    settings = settings or SurrogateSolverSettings()
    safe = _safe_name(f"{name}_{tau:.0e}")
    n_state = assets.layout.state_size
    n_equality = assets.equality_count
    n_q = assets.projection_inequality_count
    theta_slice = slice(0, 7)
    displacement_slice = slice(theta_slice.stop, theta_slice.stop + n_state)
    equality_multiplier_slice = slice(
        displacement_slice.stop, displacement_slice.stop + n_equality
    )
    inequality_multiplier_slice = slice(
        equality_multiplier_slice.stop, equality_multiplier_slice.stop + n_q
    )
    variable_count = inequality_multiplier_slice.stop
    variable = ca.MX.sym(f"{safe}_variable", variable_count)
    parameter_count = assets.layout.component_count + 6 + assets.quality_count
    parameter = ca.MX.sym(f"{safe}_parameter", parameter_count)
    normalized_theta = variable[theta_slice]
    controls = ca.DM(assets.theta_lower) + ca.DM(assets.theta_span) * normalized_theta
    u = variable[displacement_slice]
    lambda_equality = variable[equality_multiplier_slice]
    lambda_inequality = variable[inequality_multiplier_slice]
    influent = parameter[: assets.layout.component_count]
    objective_weights = parameter[
        assets.layout.component_count : assets.layout.component_count + 6
    ]
    quality_weights = parameter[assets.layout.component_count + 6 :]
    raw, phi = symbolic_quadratic_prediction(assets.model, controls, influent)
    network = symbolic_network_operators(controls, influent, assets)
    state_scale = ca.DM(assets.model.response_scale)
    equality_scale = ca.DM(assets.row_scales.equality)
    inequality_scale = ca.DM(assets.row_scales.inequality)
    projected = raw + state_scale * u
    scaled_equality = (
        ca.diag(1.0 / equality_scale) @ network.equality_matrix @ ca.diag(state_scale)
    )
    required_equality = (
        network.equality_rhs - network.equality_matrix @ raw
    ) / equality_scale
    scaled_network_inequality = (
        ca.diag(1.0 / inequality_scale)
        @ network.inequality_matrix
        @ ca.diag(state_scale)
    )
    projection_inequality = ca.vertcat(-ca.DM.eye(n_state), scaled_network_inequality)
    projection_rhs = ca.vertcat(
        raw / state_scale, -(network.inequality_matrix @ raw) / inequality_scale
    )
    lower_equality = scaled_equality @ u - required_equality
    lower_inequality = projection_inequality @ u - projection_rhs
    stationarity = (
        u
        + scaled_equality.T @ lambda_equality
        + projection_inequality.T @ lambda_inequality
    )
    slack = projection_rhs - projection_inequality @ u
    gap = 0.5 * ca.dot(stationarity, stationarity) + ca.dot(lambda_inequality, slack)
    normalized_gap = gap / (n_state + n_q)
    engineering, engineering_names, quantities = _engineering_expressions(
        controls, projected, assets
    )
    trust, trust_names, trust_values = _trust_expressions(
        controls, raw, projected, u, influent, phi, assets
    )
    objective, components = _objective_expressions(
        controls, projected, objective_weights, quality_weights, assets
    )
    inequality = ca.vertcat(
        lower_inequality, -normalized_gap, normalized_gap - tau, engineering, trust
    )
    constraints = ca.vertcat(lower_equality, inequality)
    lower_bounds = np.full(variable_count, -np.inf, dtype=np.float64)
    upper_bounds = np.full(variable_count, np.inf, dtype=np.float64)
    lower_bounds[theta_slice] = 0.0
    upper_bounds[theta_slice] = 1.0
    lower_bounds[inequality_multiplier_slice] = 0.0
    constraint_lower = np.concatenate(
        (np.zeros(n_equality), np.full(int(inequality.numel()), -np.inf))
    )
    constraint_upper = np.zeros(int(constraints.numel()), dtype=np.float64)
    solver = None
    evaluation_normalized = ca.MX.sym(f"{safe}_normalized", 7)
    evaluation_theta = (
        ca.DM(assets.theta_lower) + ca.DM(assets.theta_span) * evaluation_normalized
    )
    evaluation_influent = parameter[: assets.layout.component_count]
    evaluation_raw, evaluation_phi = symbolic_quadratic_prediction(
        assets.model, evaluation_theta, evaluation_influent
    )
    evaluation_network = symbolic_network_operators(
        evaluation_theta, evaluation_influent, assets
    )
    exact_state = ca.MX.sym(f"{safe}_exact_state", n_state)
    exact_u = (exact_state - evaluation_raw) / state_scale
    exact_engineering, _, exact_quantities = _engineering_expressions(
        evaluation_theta, exact_state, assets
    )
    exact_trust, _, exact_trust_values = _trust_expressions(
        evaluation_theta,
        evaluation_raw,
        exact_state,
        exact_u,
        evaluation_influent,
        evaluation_phi,
        assets,
    )
    exact_objective, exact_components = _objective_expressions(
        evaluation_theta, exact_state, objective_weights, quality_weights, assets
    )
    network_outputs = (
        evaluation_network.equality_matrix,
        evaluation_network.equality_rhs,
        evaluation_network.inequality_matrix,
    )
    return SurrogateNLP(
        assets=assets,
        tau=float(tau),
        name=safe,
        variable_count=variable_count,
        equality_count=n_equality,
        inequality_count=int(inequality.numel()),
        theta_slice=theta_slice,
        displacement_slice=displacement_slice,
        equality_multiplier_slice=equality_multiplier_slice,
        inequality_multiplier_slice=inequality_multiplier_slice,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        constraint_lower_bounds=constraint_lower,
        constraint_upper_bounds=constraint_upper,
        solver=solver,
        objective_function=ca.Function(
            f"{safe}_objective", [variable, parameter], [objective]
        ),
        raw_function=ca.Function(
            f"{safe}_raw", [evaluation_normalized, parameter], [evaluation_raw]
        ),
        feature_function=ca.Function(
            f"{safe}_feature", [evaluation_normalized, parameter], [evaluation_phi]
        ),
        state_function=ca.Function(f"{safe}_state", [variable, parameter], [projected]),
        network_function=ca.Function(
            f"{safe}_network", [evaluation_normalized, parameter], list(network_outputs)
        ),
        equality_function=ca.Function(
            f"{safe}_equality", [variable, parameter], [lower_equality]
        ),
        inequality_function=ca.Function(
            f"{safe}_inequality", [variable, parameter], [inequality]
        ),
        gap_function=ca.Function(
            f"{safe}_gap", [variable, parameter], [normalized_gap]
        ),
        engineering_function=ca.Function(
            f"{safe}_engineering", [variable, parameter], [engineering]
        ),
        trust_function=ca.Function(
            f"{safe}_trust", [variable, parameter], [trust, trust_values]
        ),
        upper_from_state_function=ca.Function(
            f"{safe}_upper_exact",
            [evaluation_normalized, parameter, exact_state],
            [
                exact_objective,
                exact_engineering,
                exact_trust,
                exact_components,
                exact_quantities,
                exact_trust_values,
            ],
        ),
        objective_components_function=ca.Function(
            f"{safe}_components", [variable, parameter], [components]
        ),
        engineering_quantity_function=ca.Function(
            f"{safe}_quantities", [variable, parameter], [quantities]
        ),
        engineering_names=engineering_names,
        trust_names=trust_names,
    )


def unpack_primal(
    problem: SurrogateNLP, primal: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    value = _vector(primal, problem.variable_count, "primal")
    return (
        value[problem.theta_slice],
        value[problem.displacement_slice],
        value[problem.equality_multiplier_slice],
        value[problem.inequality_multiplier_slice],
    )


def evaluate_surrogate_problem(
    problem: SurrogateNLP, primal: npt.ArrayLike, case: SurrogateCase
) -> dict[str, Any]:
    value = _vector(primal, problem.variable_count, "primal")
    parameter = case.parameter_vector(problem.assets)
    normalized, displacement, lambda_eq, lambda_ineq = unpack_primal(problem, value)
    controls = problem.assets.theta_lower + problem.assets.theta_span * normalized
    trust_output = problem.trust_function(value, parameter)
    return {
        "objective": float(problem.objective_function(value, parameter)),
        "normalized_controls": normalized,
        "controls": controls,
        "displacement": displacement,
        "lambda_eq": lambda_eq,
        "lambda_ineq": lambda_ineq,
        "raw": _flat(problem.raw_function(normalized, parameter)),
        "projected": _flat(problem.state_function(value, parameter)),
        "equality": _flat(problem.equality_function(value, parameter)),
        "inequality": _flat(problem.inequality_function(value, parameter)),
        "normalized_gap": float(problem.gap_function(value, parameter)),
        "engineering": _flat(problem.engineering_function(value, parameter)),
        "trust": _flat(trust_output[0]),
        "trust_values": _flat(trust_output[1]),
        "objective_components": _flat(
            problem.objective_components_function(value, parameter)
        ),
        "engineering_quantities": _flat(
            problem.engineering_quantity_function(value, parameter)
        ),
    }


def cold_reproject(
    assets: SurrogateNLPAssets,
    case: SurrogateCase,
    normalized_controls: npt.ArrayLike,
    *,
    raise_on_failure: bool = False,
) -> ProjectionResult:
    """Resolve a newly initialized OSQP projection at one control vector."""
    from surrogate_optimization.surrogate.projection import PhysicalProjector
    from surrogate_optimization.surrogate.projection import build_network_operators

    normalized = _vector(normalized_controls, 7, "normalized_controls")
    if np.any(normalized < 0.0) or np.any(normalized > 1.0):
        raise ValueError("normalized_controls must lie in [0, 1].")
    influent = _vector(case.influent, assets.layout.component_count, "case influent")
    controls = assets.theta_lower + assets.theta_span * normalized
    raw = np.asarray(assets.model.predict(controls, influent), dtype=np.float64)
    operators = build_network_operators(
        influent,
        internal_recycle=float(controls[4]),
        return_recycle=float(controls[5]),
        waste_fraction=float(controls[6]),
        invariant_operator=assets.invariant_operator,
        tss_weights=assets.tss_weights,
        layout=assets.layout,
        clarifier_volume_m3=assets.engineering.clarifier_volume_m3,
        overflow_tss_closure=None
        if assets.overflow_closure is None
        else float(assets.overflow_closure.predict(controls, influent)),
    )
    projector = PhysicalProjector(
        assets.model.response_scale,
        assets.row_scales.equality,
        assets.row_scales.inequality,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        maximum_iterations=100000,
        polish=True,
    )
    return projector.project(
        raw,
        operators.equality_matrix,
        operators.equality_rhs,
        operators.inequality_matrix,
        warm_start=None,
        raise_on_failure=raise_on_failure,
    )


def initial_primal_from_projection(
    problem: SurrogateNLP, case: SurrogateCase, normalized_controls: npt.ArrayLike
) -> tuple[FloatArray, ProjectionResult]:
    from surrogate_optimization.optimization.types import SurrogateNLPError

    normalized = _vector(normalized_controls, 7, "normalized_controls")
    projection = cold_reproject(
        problem.assets, case, normalized, raise_on_failure=False
    )
    if not projection.accepted:
        raise SurrogateNLPError(
            "the initial cold projection failed its independent KKT audit."
        )
    primal = np.concatenate(
        (
            normalized,
            projection.displacement,
            projection.equality_multipliers,
            projection.inequality_multipliers,
        )
    )
    return (primal, projection)


def _exact_evaluation(
    problem: SurrogateNLP, case: SurrogateCase, normalized: FloatArray
) -> _ExactEvaluation:
    from surrogate_optimization.optimization.types import _ExactEvaluation

    projection = cold_reproject(
        problem.assets, case, normalized, raise_on_failure=False
    )
    parameter = case.parameter_vector(problem.assets)
    outputs = problem.upper_from_state_function(normalized, parameter, projection.state)
    return _ExactEvaluation(
        normalized=normalized.copy(),
        projection=projection,
        objective=float(outputs[0]),
        engineering=_flat(outputs[1]),
        trust=_flat(outputs[2]),
        components=_flat(outputs[3]),
        quantities=_flat(outputs[4]),
        trust_values=_flat(outputs[5]),
    )


def _feasibility_record(
    evaluation: _ExactEvaluation, settings: SurrogateSolverSettings
) -> FeasibilityRecord:
    from surrogate_optimization.optimization.types import FeasibilityRecord

    normalized = evaluation.normalized
    control = max(_maximum_positive(-normalized), _maximum_positive(normalized - 1.0))
    engineering = _maximum_positive(evaluation.engineering)
    trust = _maximum_positive(evaluation.trust)
    finite = all(
        (
            np.all(np.isfinite(np.asarray(value)))
            for value in (
                normalized,
                evaluation.projection.state,
                evaluation.objective,
                evaluation.engineering,
                evaluation.trust,
            )
        )
    )
    maximum = max(control, engineering, trust)
    feasible = bool(
        finite
        and evaluation.projection.accepted
        and (maximum <= settings.final_upper_tolerance)
    )
    return FeasibilityRecord(
        finite=finite,
        cold_projection=True,
        projection_accepted=bool(evaluation.projection.accepted),
        control_bound_residual=control,
        engineering_residual=engineering,
        trust_residual=trust,
        maximum_upper_residual=maximum,
        feasible=feasible,
    )


def audit_exact_candidate(
    problem: SurrogateNLP,
    case: SurrogateCase,
    normalized_controls: npt.ArrayLike,
    *,
    settings: SurrogateSolverSettings | None = None,
) -> FinalCandidateRecord:
    """Cold-reproject and retain feasibility, lower KKT, and status records."""
    from surrogate_optimization.optimization.types import FinalCandidateRecord
    from surrogate_optimization.optimization.types import StationarityRecord
    from surrogate_optimization.optimization.types import SurrogateSolverSettings

    settings = settings or SurrogateSolverSettings()
    normalized = _vector(normalized_controls, 7, "normalized_controls")
    evaluation = _exact_evaluation(problem, case, normalized)
    feasibility = _feasibility_record(evaluation, settings)
    stationarity = StationarityRecord(
        classification="stationarity_unresolved",
        resolved=False,
        stationary=False,
        lower_qp_kkt_passed=bool(evaluation.projection.accepted),
        upper_stationarity_residual=None,
        reason="A cold lower-QP KKT audit is available, but the rank, conditioning, strict-complementarity, active-set perturbation, total-sensitivity, and independently reconstructed upper-multiplier contract has not been established for this endpoint.",
    )
    status = (
        "validated_feasible_stationarity_unresolved"
        if feasibility.feasible
        else "final_feasibility_failed"
    )
    controls = problem.assets.theta_lower + problem.assets.theta_span * normalized
    raw = np.asarray(
        problem.assets.model.predict(controls, case.influent), dtype=np.float64
    )
    return FinalCandidateRecord(
        normalized_controls=normalized,
        controls=controls,
        raw=raw,
        projected=evaluation.projection.state.copy(),
        displacement=evaluation.projection.displacement.copy(),
        objective=evaluation.objective,
        objective_components=evaluation.components,
        engineering_rows=evaluation.engineering,
        engineering_quantities=evaluation.quantities,
        trust_rows=evaluation.trust,
        trust_values=evaluation.trust_values,
        projection=evaluation.projection,
        feasibility=feasibility,
        stationarity=stationarity,
        status=status,
    )


def _outer_refine(
    problem: SurrogateNLP,
    case: SurrogateCase,
    initial: FloatArray,
    settings: SurrogateSolverSettings,
) -> tuple[FinalCandidateRecord | None, OuterRefinementRecord]:
    """Run the seven-variable exact-QP active-set refinement.

    The import is local because :mod:`surrogate_optimization.optimization.active_set` builds on the
    public data structures in this module.  The implementation has no
    finite-difference fallback: an unstable lower active set is retained as
    an explicit stationarity-unresolved result.
    """
    from surrogate_optimization.optimization.active_set import (
        ActiveSetRefinementSettings,
    )
    from surrogate_optimization.optimization.active_set import ExactQPActiveSetRefiner
    from surrogate_optimization.optimization.types import FinalCandidateRecord
    from surrogate_optimization.optimization.types import OuterRefinementRecord
    from surrogate_optimization.optimization.types import StationarityRecord
    from surrogate_optimization.optimization.types import _ExactEvaluation

    if not settings.perform_outer_refinement:
        return (
            None,
            OuterRefinementRecord(
                attempted=False,
                solver_success=False,
                status="disabled",
                iterations=0,
                evaluations=0,
                elapsed_seconds=0.0,
                initial_objective=None,
                final_objective=None,
            ),
        )
    active_settings = ActiveSetRefinementSettings(
        upper_acceptance_tolerance=settings.final_upper_tolerance,
        maximum_iterations=settings.outer_maximum_iterations,
        function_tolerance=settings.outer_function_tolerance,
    )
    exact = ExactQPActiveSetRefiner(
        problem.assets,
        case,
        problem=problem,
        settings=active_settings,
        name=f"{problem.name}_outer",
    ).refine(initial)
    initial_objective = (
        None if exact.initial is None else float(exact.initial.objective)
    )
    final_objective = None if exact.final is None else float(exact.final.objective)
    lower_audit = (
        exact.final.lower_active_set.as_dict()
        if exact.final is not None
        else None
        if exact.derivative_audit is None
        else exact.derivative_audit.as_dict()
    )
    upper_audit = None if exact.upper_kkt is None else exact.upper_kkt.as_dict()
    record = OuterRefinementRecord(
        attempted=True,
        solver_success=exact.solver_success,
        status=exact.solver_status,
        iterations=exact.iterations,
        evaluations=exact.distinct_trials,
        elapsed_seconds=exact.elapsed_seconds,
        initial_objective=initial_objective,
        final_objective=final_objective,
        cold_qp_resolutions=exact.cold_qp_resolutions,
        derivative_error=exact.derivative_error,
        lower_active_set=lower_audit,
        upper_kkt=upper_audit,
        projection_reproduction_residual=exact.state_reproduction_residual,
        projection_reproduction_passed=exact.state_reproduction_passed,
    )
    if exact.final is None or exact.upper_kkt is None:
        return (None, record)
    trial = exact.final
    parameter = case.parameter_vector(problem.assets)
    outputs = problem.upper_from_state_function(
        trial.normalized_controls, parameter, trial.projected_state
    )
    evaluation = _ExactEvaluation(
        normalized=trial.normalized_controls.copy(),
        projection=trial.projection,
        objective=float(outputs[0]),
        engineering=_flat(outputs[1]),
        trust=_flat(outputs[2]),
        components=_flat(outputs[3]),
        quantities=_flat(outputs[4]),
        trust_values=_flat(outputs[5]),
    )
    feasibility = _feasibility_record(evaluation, settings)
    reproduction_passed = exact.state_reproduction_passed is True
    feasibility = replace(
        feasibility,
        feasible=bool(feasibility.feasible and reproduction_passed),
        projection_reproduction_residual=exact.state_reproduction_residual,
        projection_reproduction_passed=exact.state_reproduction_passed,
    )
    upper = exact.upper_kkt
    stationary = bool(upper.stationary and reproduction_passed)
    unresolved = not stationary
    stationarity = StationarityRecord(
        classification=upper.classification
        if reproduction_passed
        else "projection_reproduction_failed",
        resolved=not unresolved,
        stationary=stationary,
        lower_qp_kkt_passed=bool(
            trial.projection.accepted and trial.lower_active_set.stable
        ),
        upper_stationarity_residual=float(upper.stationarity_residual),
        reason=upper.reason
        if reproduction_passed
        else "independent cold-QP state reproduction failed",
    )
    status = (
        "projection_reproduction_failed"
        if not reproduction_passed
        else "validated_stationary"
        if feasibility.feasible and stationarity.stationary
        else "validated_feasible_stationarity_unresolved"
        if feasibility.feasible
        else "final_feasibility_failed"
    )
    return (
        FinalCandidateRecord(
            normalized_controls=trial.normalized_controls.copy(),
            controls=trial.physical_controls.copy(),
            raw=trial.raw_state.copy(),
            projected=trial.projected_state.copy(),
            displacement=trial.projection.displacement.copy(),
            objective=evaluation.objective,
            objective_components=evaluation.components,
            engineering_rows=evaluation.engineering,
            engineering_quantities=evaluation.quantities,
            trust_rows=evaluation.trust,
            trust_values=evaluation.trust_values,
            projection=trial.projection,
            feasibility=feasibility,
            stationarity=stationarity,
            status=status,
            lower_active_set=trial.lower_active_set.as_dict(),
            upper_kkt=upper.as_dict(),
        ),
        record,
    )


def _derivative_free_exact_qp_refine(
    problem: SurrogateNLP,
    case: SurrogateCase,
    initial: FloatArray,
    settings: SurrogateSolverSettings,
    active_record: OuterRefinementRecord,
) -> tuple[FinalCandidateRecord | None, OuterRefinementRecord]:
    """Continue an unavailable active-set derivative with exact-QP COBYQA.

    Bounds are unrelaxable in COBYQA and every distinct in-box point is
    evaluated by a newly initialized projection QP. Objective and nonlinear
    constraint callbacks share only the resulting value cache. The selected
    visited point is then cold-replayed independently, and analytical
    active-set/upper-KKT auditing is attempted again only at that endpoint.
    """
    from surrogate_optimization.optimization.active_set import ActiveSetDerivativeError
    from surrogate_optimization.optimization.active_set import ActiveSetRefinementError
    from surrogate_optimization.optimization.active_set import (
        ActiveSetRefinementSettings,
    )
    from surrogate_optimization.optimization.active_set import ExactQPActiveSetRefiner
    from surrogate_optimization.optimization.types import OuterRefinementRecord
    from surrogate_optimization.optimization.types import StationarityRecord
    from surrogate_optimization.optimization.types import SurrogateNLPError
    from surrogate_optimization.optimization.types import _DerivativeFreeEvaluation

    started = perf_counter()
    normalized_initial = _vector(initial, 7, "normalized_start")
    upper_count = len(problem.engineering_names) + len(problem.trust_names)
    cache: dict[bytes, _DerivativeFreeEvaluation] = {}
    cold_qp_resolutions = 0
    evaluation_errors: list[str] = []

    def evaluate(value: npt.ArrayLike) -> _DerivativeFreeEvaluation:
        nonlocal cold_qp_resolutions
        normalized = _vector(value, 7, "derivative-free normalized controls")
        normalized = np.clip(normalized, 0.0, 1.0)
        key = np.ascontiguousarray(normalized, dtype=np.float64).tobytes()
        if key in cache:
            return cache[key]
        if (
            settings.maximum_wall_time is not None
            and perf_counter() - started >= settings.maximum_wall_time
        ):
            raise TimeoutError("derivative-free exact-QP wall-time limit reached")
        cold_qp_resolutions += 1
        try:
            candidate = audit_exact_candidate(
                problem, case, normalized, settings=settings
            )
            finite = bool(
                candidate.feasibility.finite
                and np.isfinite(candidate.objective)
                and np.all(np.isfinite(candidate.engineering_rows))
                and np.all(np.isfinite(candidate.trust_rows))
            )
            if not finite:
                raise SurrogateNLPError(
                    "exact-QP derivative-free evaluation was non-finite"
                )
            item = _DerivativeFreeEvaluation(normalized, candidate, None)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            item = _DerivativeFreeEvaluation(normalized, None, message)
            evaluation_errors.append(message)
        cache[key] = item
        return item

    def usable(item: _DerivativeFreeEvaluation) -> bool:
        return bool(
            item.candidate is not None
            and item.candidate.projection.accepted
            and item.candidate.feasibility.finite
        )

    def objective(value: npt.ArrayLike) -> float:
        item = evaluate(value)
        if usable(item):
            assert item.candidate is not None
            return float(item.candidate.objective)
        distance = float(
            np.linalg.norm(item.normalized_controls - normalized_initial) ** 2
        )
        return 1000000.0 + distance

    def upper_constraints(value: npt.ArrayLike) -> FloatArray:
        item = evaluate(value)
        if usable(item):
            assert item.candidate is not None
            rows = np.concatenate(
                (item.candidate.engineering_rows, item.candidate.trust_rows)
            )
            if rows.shape != (upper_count,):
                raise AssertionError(
                    "derivative-free upper constraints have inconsistent dimensions"
                )
            return rows
        return np.ones(upper_count, dtype=np.float64)

    solver_success = False
    solver_status = "not_started"
    iterations = 0
    proposed = normalized_initial.copy()
    try:
        optimized = minimize(
            objective,
            normalized_initial,
            method="COBYQA",
            bounds=Bounds(np.zeros(7), np.ones(7)),
            constraints=NonlinearConstraint(
                upper_constraints, np.full(upper_count, -np.inf), np.zeros(upper_count)
            ),
            options={
                "maxiter": settings.outer_maximum_iterations,
                "maxfev": settings.outer_maximum_iterations,
                "initial_tr_radius": 0.25,
                "final_tr_radius": max(
                    settings.outer_function_tolerance, settings.final_upper_tolerance
                ),
                "feasibility_tol": settings.final_upper_tolerance,
                "scale": False,
                "disp": False,
            },
        )
        proposed = np.clip(
            _vector(optimized.x, 7, "derivative-free optimizer endpoint"), 0.0, 1.0
        )
        solver_success = bool(optimized.success)
        solver_status = str(optimized.message)
        iterations = int(optimized.nit)
    except Exception as exc:
        solver_status = f"{type(exc).__name__}: {exc}"
    try:
        evaluate(proposed)
    except Exception as exc:
        evaluation_errors.append(f"{type(exc).__name__}: {exc}")
    candidates = [
        item.candidate
        for item in cache.values()
        if usable(item) and item.candidate is not None
    ]
    feasible = [item for item in candidates if item.feasibility.feasible]
    pool = feasible or candidates
    selected_cached: FinalCandidateRecord | None = None
    if pool:
        if feasible:
            best_objective = min((item.objective for item in pool))
            tie = 1e-10 * max(1.0, abs(best_objective))
            selected_cached = min(
                (item for item in pool if item.objective <= best_objective + tie),
                key=lambda item: tuple(item.normalized_controls.tolist()),
            )
        else:
            selected_cached = min(
                pool,
                key=lambda item: (
                    item.feasibility.maximum_upper_residual,
                    item.objective,
                    *item.normalized_controls.tolist(),
                ),
            )
    final: FinalCandidateRecord | None = None
    reproduction_residual: float | None = None
    reproduction_passed: bool | None = None
    endpoint_lower = active_record.lower_active_set
    endpoint_upper: dict[str, Any] | None = None
    endpoint_stationary = False
    endpoint_stationarity_residual: float | None = None
    endpoint_derivative_error: str | None = None
    if selected_cached is not None:
        cold_qp_resolutions += 1
        try:
            replay = audit_exact_candidate(
                problem, case, selected_cached.normalized_controls, settings=settings
            )
            reproduction_residual = float(
                np.linalg.norm(
                    (replay.projected - selected_cached.projected)
                    / problem.assets.model.response_scale,
                    ord=np.inf,
                )
            )
            reproduction_passed = bool(
                replay.projection.accepted
                and np.isfinite(reproduction_residual)
                and (reproduction_residual <= 1e-08)
            )
            final = replay
        except Exception as exc:
            endpoint_derivative_error = f"endpoint replay: {type(exc).__name__}: {exc}"
    if final is not None and reproduction_passed is True:
        active_settings = ActiveSetRefinementSettings(
            upper_acceptance_tolerance=settings.final_upper_tolerance,
            maximum_iterations=settings.outer_maximum_iterations,
            function_tolerance=settings.outer_function_tolerance,
        )
        endpoint_refiner = ExactQPActiveSetRefiner(
            problem.assets,
            case,
            problem=problem,
            settings=active_settings,
            name=f"{problem.name}_derivative_free_endpoint",
        )
        try:
            endpoint_trial = endpoint_refiner.evaluate(
                final.normalized_controls,
                force_cold=True,
                independent_final_replay=True,
            )
            upper_audit = endpoint_refiner.audit_upper_kkt(endpoint_trial)
            endpoint_lower = endpoint_trial.lower_active_set.as_dict()
            endpoint_upper = upper_audit.as_dict()
            endpoint_stationary = bool(upper_audit.stationary)
            endpoint_stationarity_residual = float(upper_audit.stationarity_residual)
        except ActiveSetDerivativeError as exc:
            endpoint_lower = None if exc.audit is None else exc.audit.as_dict()
            endpoint_derivative_error = f"endpoint active-set audit: {exc}"
        except ActiveSetRefinementError as exc:
            endpoint_derivative_error = f"endpoint active-set audit: {exc}"
        except Exception as exc:
            endpoint_derivative_error = (
                f"endpoint active-set audit: {type(exc).__name__}: {exc}"
            )
        finally:
            cold_qp_resolutions += endpoint_refiner.cold_qp_resolutions
    derivative_messages = [
        item
        for item in (active_record.derivative_error, endpoint_derivative_error)
        if item
    ]
    derivative_error = "; ".join(dict.fromkeys(derivative_messages)) or None
    if final is not None:
        final_feasibility = replace(
            final.feasibility,
            feasible=bool(final.feasibility.feasible and reproduction_passed is True),
            projection_reproduction_residual=reproduction_residual,
            projection_reproduction_passed=reproduction_passed,
        )
        stationary = bool(
            final_feasibility.feasible
            and endpoint_stationary
            and (endpoint_upper is not None)
        )
        if stationary:
            stationarity = StationarityRecord(
                classification="first_order_kkt_stationary_feasible",
                resolved=True,
                stationary=True,
                lower_qp_kkt_passed=True,
                upper_stationarity_residual=endpoint_stationarity_residual,
                reason="the derivative-free local endpoint passed independent lower active-set and upper KKT audits",
            )
            final_status = "validated_stationary"
        else:
            normalized_solver_status = solver_status.lower()
            budget_limited = bool(
                not solver_success
                and "maximum number" in normalized_solver_status
                and (
                    "evaluation" in normalized_solver_status
                    or "iteration" in normalized_solver_status
                )
            )
            nonconverged_label = (
                "budget_limited_derivative_free_feasible_incumbent_stationarity_unresolved"
                if budget_limited
                else "nonconverged_derivative_free_feasible_incumbent_stationarity_unresolved"
            )
            stationarity = StationarityRecord(
                classification=nonconverged_label
                if final_feasibility.feasible and (not solver_success)
                else "derivative_free_local_candidate_stationarity_unresolved"
                if final_feasibility.feasible
                else "derivative_free_local_candidate_feasibility_failed",
                resolved=False,
                stationary=False,
                lower_qp_kkt_passed=bool(final.projection.accepted),
                upper_stationarity_residual=endpoint_stationarity_residual,
                reason=(
                    "COBYQA reached its iteration/evaluation budget; this is the best feasible exact-QP point visited, not an established local optimum. "
                    if budget_limited
                    else "COBYQA did not report convergence; this is the best feasible exact-QP point visited, not an established local optimum. "
                    if not solver_success
                    else "COBYQA converged to a derivative-free exact-QP local candidate. "
                )
                + "Stationarity remains unresolved because the independent "
                + "active-set/KKT audit did not pass: "
                + str(derivative_error or solver_status),
            )
            final_status = (
                "validated_feasible_budget_limited_derivative_free_incumbent_stationarity_unresolved"
                if final_feasibility.feasible and budget_limited
                else "validated_feasible_nonconverged_derivative_free_incumbent_stationarity_unresolved"
                if final_feasibility.feasible and (not solver_success)
                else "validated_feasible_derivative_free_local_candidate_stationarity_unresolved"
                if final_feasibility.feasible
                else "projection_reproduction_failed"
                if reproduction_passed is False
                else "final_feasibility_failed"
            )
        final = replace(
            final,
            feasibility=final_feasibility,
            stationarity=stationarity,
            status=final_status,
            lower_active_set=endpoint_lower,
            upper_kkt=endpoint_upper,
        )
    initial_item = cache.get(
        np.ascontiguousarray(normalized_initial, dtype=np.float64).tobytes()
    )
    initial_objective = (
        None
        if initial_item is None or initial_item.candidate is None
        else float(initial_item.candidate.objective)
    )
    fallback_status = solver_status
    if evaluation_errors:
        fallback_status = (
            f"{fallback_status}; failed exact-QP evaluations={len(evaluation_errors)}"
        )
    record = OuterRefinementRecord(
        attempted=True,
        solver_success=solver_success,
        status="derivative_free_local_candidate"
        if final is not None and solver_success
        else "derivative_free_budget_limited_candidate"
        if final is not None
        and "maximum number" in solver_status.lower()
        and (
            "evaluation" in solver_status.lower()
            or "iteration" in solver_status.lower()
        )
        else "derivative_free_nonconverged_candidate"
        if final is not None
        else "derivative_free_local_optimization_failed",
        iterations=iterations,
        evaluations=len(cache),
        elapsed_seconds=active_record.elapsed_seconds + (perf_counter() - started),
        initial_objective=initial_objective,
        final_objective=None if final is None else float(final.objective),
        cold_qp_resolutions=active_record.cold_qp_resolutions + cold_qp_resolutions,
        derivative_error=derivative_error,
        lower_active_set=endpoint_lower,
        upper_kkt=endpoint_upper,
        projection_reproduction_residual=reproduction_residual,
        projection_reproduction_passed=reproduction_passed,
        method="exact_qp_derivative_free_cobyqa",
        fallback_used=True,
        fallback_method="COBYQA",
        fallback_solver_success=solver_success,
        fallback_status=fallback_status,
        fallback_iterations=iterations,
        fallback_evaluations=len(cache),
    )
    return (final, record)


def solve_surrogate_exact_qp_local(
    assets: SurrogateNLPAssets,
    case: SurrogateCase,
    *,
    settings: SurrogateSolverSettings | None = None,
    problem: SurrogateNLP | None = None,
    completed_result: SurrogateStartResult | None = None,
    name: str = "surrogate_exact_qp_local",
    progress_callback: Callable[[SurrogateStartResult], None] | None = None,
) -> SurrogateRouteResult:
    """Solve one case by one seven-variable exact-QP local optimization.

    The sole initial point is the deterministic center of the normalized
    operating box. This route builds only the expression graph needed by the
    exact-QP active-set evaluator (``compile_solver=False``): it never
    constructs an embedded-KKT IPOPT solver and has no continuation stages.

    Numerical projection, active-set, derivative, or endpoint failures are
    returned as one structured, stationarity-unresolved start result. They
    therefore remain visible to scientific reporting without aborting other
    influent cases. Invalid inputs and stale checkpoints still raise.
    """
    from surrogate_optimization.optimization.types import OuterRefinementRecord
    from surrogate_optimization.optimization.types import StationarityRecord
    from surrogate_optimization.optimization.types import SurrogateRouteResult
    from surrogate_optimization.optimization.types import SurrogateSolverSettings
    from surrogate_optimization.optimization.types import SurrogateStartResult

    settings = settings or SurrogateSolverSettings()
    if not settings.perform_outer_refinement:
        raise ValueError(
            "the exact-QP single-start protocol requires perform_outer_refinement=True"
        )
    center = np.asarray(EXACT_QP_CENTER_START, dtype=np.float64)
    resume_contract = surrogate_exact_qp_resume_contract(assets, case, settings)
    if completed_result is not None:
        if not isinstance(completed_result, SurrogateStartResult):
            raise TypeError("completed_result must be a SurrogateStartResult.")
        if completed_result.start_index != 0:
            raise ValueError(
                "the completed exact-QP result must have start_index zero."
            )
        if not np.array_equal(completed_result.initial_normalized_controls, center):
            raise ValueError(
                "the completed exact-QP result does not use the center start."
            )
        if completed_result.stages:
            raise ValueError(
                "an exact-QP single-start checkpoint cannot contain continuation stages."
            )
        if completed_result.protocol != EXACT_QP_SINGLE_START_PROTOCOL:
            raise ValueError(
                "the completed result uses a different surrogate protocol."
            )
        if completed_result.resume_contract != resume_contract:
            raise ValueError(
                "the completed result has a stale exact-QP resume contract."
            )
        result = completed_result
    else:
        if problem is None:
            problem = build_surrogate_expression_graph(
                assets, EXPRESSION_GAP, settings=settings, name=f"{name}_expressions"
            )
        else:
            if problem.assets is not assets:
                raise ValueError("the reusable exact-QP problem uses different assets.")
            if problem.tau != EXPRESSION_GAP:
                raise ValueError(
                    "the reusable exact-QP problem must use the final gap value."
                )
            if problem.solver is not None:
                raise ValueError(
                    "the reusable exact-QP expression problem must not contain an IPOPT solver."
                )
        refinement_error: str | None = None
        try:
            final, refinement = _outer_refine(problem, case, center, settings)
        except Exception as exc:
            refinement_error = f"{type(exc).__name__}: {exc}"
            final = None
            refinement = OuterRefinementRecord(
                attempted=True,
                solver_success=False,
                status="unexpected_exact_qp_exception",
                iterations=0,
                evaluations=0,
                elapsed_seconds=0.0,
                initial_objective=None,
                final_objective=None,
                derivative_error=refinement_error,
            )
        if final is None:
            try:
                final, refinement = _derivative_free_exact_qp_refine(
                    problem, case, center, settings, refinement
                )
            except Exception as exc:
                fallback_error = f"{type(exc).__name__}: {exc}"
                refinement_error = (
                    fallback_error
                    if refinement_error is None
                    else f"{refinement_error}; derivative-free fallback: {fallback_error}"
                )
                refinement = replace(
                    refinement,
                    status="unexpected_derivative_free_exception",
                    derivative_error=refinement.derivative_error or refinement_error,
                    method="exact_qp_derivative_free_cobyqa",
                    fallback_used=True,
                    fallback_method="COBYQA",
                    fallback_solver_success=False,
                    fallback_status=fallback_error,
                )
        if final is None:
            try:
                incumbent = audit_exact_candidate(
                    problem, case, center, settings=settings
                )
            except Exception as exc:
                audit_error = f"{type(exc).__name__}: {exc}"
                refinement_error = (
                    audit_error
                    if refinement_error is None
                    else f"{refinement_error}; center audit: {audit_error}"
                )
            else:
                reason = refinement.derivative_error or refinement.status
                final = replace(
                    incumbent,
                    stationarity=StationarityRecord(
                        classification="stationarity_unresolved",
                        resolved=False,
                        stationary=False,
                        lower_qp_kkt_passed=bool(incumbent.projection.accepted),
                        upper_stationarity_residual=None,
                        reason=f"The primary seven-variable exact-QP local optimizer did not establish endpoint stationarity; the independently cold-projected center incumbent is retained: {reason}",
                    ),
                    status="validated_feasible_stationarity_unresolved"
                    if incumbent.feasibility.feasible
                    else incumbent.status,
                    lower_active_set=refinement.lower_active_set,
                    upper_kkt=refinement.upper_kkt,
                )
                if refinement_error is None:
                    refinement_error = refinement.derivative_error
        result = SurrogateStartResult(
            start_index=0,
            initial_normalized_controls=center.copy(),
            stages=(),
            outer_refinement=refinement,
            final=final,
            status=final.status
            if final is not None
            else "exact_qp_local_optimization_unresolved",
            error=refinement_error,
            resume_contract=resume_contract,
            protocol=EXACT_QP_SINGLE_START_PROTOCOL,
        )
        if progress_callback is not None:
            progress_callback(result)
    selected = result if result.feasible else None
    return SurrogateRouteResult(
        starts=(result,),
        selected=selected,
        status="selected_stationary"
        if result.stationary
        else "selected_stationarity_unresolved"
        if result.feasible
        else "no_validated_feasible_start",
        protocol=EXACT_QP_SINGLE_START_PROTOCOL,
    )


IMPLEMENTATION_LIMITATIONS = "Exact-QP analytical derivatives are available only when the lower active set passes the LICQ, conditioning, strict-complementarity, and local perturbation audits. When they do not, deterministic value-only COBYQA continues with a cold exact projection QP at every distinct trial. Its independently replayed endpoint remains stationarity-unresolved unless the endpoint active-set and upper-KKT audits pass. No finite-difference derivative is substituted for a failed analytical audit."
