"""Surrogate projection."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from scipy import linalg
from scipy import sparse
from scipy.optimize import lsq_linear
import numpy as np
import numpy.typing as npt
import osqp


@dataclass(frozen=True)
class NetworkLayout:
    """Coordinate layout for ``chi=(m,c_1,...,c_N,g_E,g_U,M_cl)``.

    ``layer_count`` remains part of the mechanistic Clarifier geometry, but
    the statistical response contains only its total solids inventory.  The
    envelope implemented here assumes that total Clarifier volume is divided
    equally among those mechanistic layers.
    """

    stage_count: int = 5
    component_count: int = 20
    layer_count: int = 10
    soluble_indices: tuple[int, ...] = tuple(range(10))
    particulate_indices: tuple[int, ...] = tuple(range(10, 20))

    def __post_init__(self) -> None:
        from surrogate_optimization.surrogate.regression import SurrogateValidationError

        if self.stage_count < 1 or self.component_count < 1 or self.layer_count < 3:
            raise SurrogateValidationError(
                "network dimensions are outside the supported range."
            )
        soluble = tuple((int(index) for index in self.soluble_indices))
        particulate = tuple((int(index) for index in self.particulate_indices))
        if sorted(soluble + particulate) != list(range(self.component_count)):
            raise SurrogateValidationError(
                "soluble_indices and particulate_indices must partition all components exactly once."
            )

    @property
    def state_size(self) -> int:
        return (self.stage_count + 3) * self.component_count + 1

    @property
    def equality_count_without_invariants(self) -> int:
        return 2 * self.component_count + len(self.soluble_indices)

    @property
    def inequality_count(self) -> int:
        return len(self.particulate_indices) + 2

    @property
    def mixer_slice(self) -> slice:
        return slice(0, self.component_count)

    def reactor_slice(self, stage: int) -> slice:
        if not 0 <= stage < self.stage_count:
            raise IndexError("reactor stage is out of range")
        start = (stage + 1) * self.component_count
        return slice(start, start + self.component_count)

    @property
    def overflow_flow_slice(self) -> slice:
        start = (self.stage_count + 1) * self.component_count
        return slice(start, start + self.component_count)

    @property
    def underflow_flow_slice(self) -> slice:
        start = (self.stage_count + 2) * self.component_count
        return slice(start, start + self.component_count)

    @property
    def inventory_index(self) -> int:
        return (self.stage_count + 3) * self.component_count

    @property
    def inventory_slice(self) -> slice:
        return slice(self.inventory_index, self.inventory_index + 1)


@dataclass(frozen=True)
class NetworkOperators:
    layout: NetworkLayout
    equality_matrix: FloatArray
    equality_rhs: FloatArray
    inequality_matrix: FloatArray
    primary_flow: float
    clarifier_flow: float
    underflow: float
    effluent_flow: float
    clarifier_volume_m3: float


def build_network_operators(
    influent: npt.ArrayLike,
    *,
    internal_recycle: float,
    return_recycle: float,
    waste_fraction: float,
    invariant_operator: npt.ArrayLike,
    tss_weights: npt.ArrayLike,
    layout: NetworkLayout | None = None,
    clarifier_volume_m3: float = 6000.0,
    overflow_tss_closure: float | None = None,
) -> NetworkOperators:
    """Assemble the model's ordered H chi=b and G chi<=0 matrices.

    ``clarifier_volume_m3`` is divided equally over ``layout.layer_count``;
    unequal layer volumes are outside this reduced projection contract.
    """
    from surrogate_optimization.surrogate.regression import SurrogateValidationError
    from surrogate_optimization.surrogate.regression import _finite_array

    layout = layout or NetworkLayout()
    component_count = layout.component_count
    state_size = layout.state_size
    x = _finite_array(influent, name="influent", ndim=1)
    invariant = _finite_array(invariant_operator, name="invariant_operator", ndim=2)
    tss = _finite_array(tss_weights, name="tss_weights", ndim=1)
    if (
        x.size != component_count
        or invariant.shape[1] != component_count
        or tss.size != component_count
    ):
        raise SurrogateValidationError("network component dimensions are inconsistent.")
    if (
        invariant.shape[0] == 0
        or np.linalg.matrix_rank(invariant) != invariant.shape[0]
    ):
        raise SurrogateValidationError("invariant_operator must have full row rank.")
    if (
        internal_recycle < 0.0
        or return_recycle < 0.0
        or (not 0.0 < waste_fraction < 1.0)
    ):
        raise SurrogateValidationError(
            "recycle ratios must be nonnegative and 0 < waste_fraction < 1."
        )
    if not np.isfinite(clarifier_volume_m3) or clarifier_volume_m3 <= 0.0:
        raise SurrogateValidationError(
            "clarifier_volume_m3 must be positive and finite."
        )
    underflow = float(return_recycle + waste_fraction)
    if underflow <= 0.0:
        raise SurrogateValidationError("return plus waste flow must be positive.")
    primary_flow = float(1.0 + internal_recycle + return_recycle)
    clarifier_flow = float(1.0 + return_recycle)
    effluent_flow = float(1.0 - waste_fraction)
    invariant_count = invariant.shape[0]
    has_overflow_closure = overflow_tss_closure is not None
    if has_overflow_closure and (
        not np.isfinite(float(overflow_tss_closure))
        or float(overflow_tss_closure) <= 0.0
    ):
        raise SurrogateValidationError(
            "overflow_tss_closure must be finite and strictly positive"
        )
    equality_count = (
        component_count
        + layout.stage_count * invariant_count
        + component_count
        + len(layout.soluble_indices)
        + int(has_overflow_closure)
    )
    equality = np.zeros((equality_count, state_size), dtype=np.float64)
    rhs = np.zeros(equality_count, dtype=np.float64)
    identity = np.eye(component_count, dtype=np.float64)
    row = 0
    equality[row : row + component_count, layout.mixer_slice] = primary_flow * identity
    equality[
        row : row + component_count, layout.reactor_slice(layout.stage_count - 1)
    ] = -float(internal_recycle) * identity
    equality[row : row + component_count, layout.underflow_flow_slice] = (
        -float(return_recycle) / underflow * identity
    )
    rhs[row : row + component_count] = x
    row += component_count
    previous = layout.mixer_slice
    for stage in range(layout.stage_count):
        current = layout.reactor_slice(stage)
        equality[row : row + invariant_count, current] = invariant
        equality[row : row + invariant_count, previous] = -invariant
        row += invariant_count
        previous = current
    final_reactor = layout.reactor_slice(layout.stage_count - 1)
    equality[row : row + component_count, layout.overflow_flow_slice] = identity
    equality[row : row + component_count, layout.underflow_flow_slice] = identity
    equality[row : row + component_count, final_reactor] = -clarifier_flow * identity
    row += component_count
    for component in layout.soluble_indices:
        equality[row, layout.underflow_flow_slice.start + component] = 1.0
        equality[row, final_reactor.start + component] = -underflow
        row += 1
    if has_overflow_closure:
        equality[row, layout.overflow_flow_slice] = tss
        rhs[row] = effluent_flow * float(overflow_tss_closure)
        row += 1
    if row != equality_count:
        raise AssertionError("equality row assembly is inconsistent")
    inequality = np.zeros((layout.inequality_count, state_size), dtype=np.float64)
    row = 0
    for component in layout.particulate_indices:
        inequality[row, final_reactor.start + component] = underflow
        inequality[row, layout.underflow_flow_slice.start + component] = -1.0
        row += 1
    endpoint_layer_volume = float(clarifier_volume_m3) / layout.layer_count
    remaining_volume = float(clarifier_volume_m3) - endpoint_layer_volume
    inequality[row, layout.overflow_flow_slice] = underflow * remaining_volume * tss
    inequality[row, layout.underflow_flow_slice] = (
        effluent_flow * endpoint_layer_volume * tss
    )
    inequality[row, layout.inventory_index] = -effluent_flow * underflow
    row += 1
    inequality[row, layout.overflow_flow_slice] = (
        -underflow * endpoint_layer_volume * tss
    )
    inequality[row, layout.underflow_flow_slice] = (
        -effluent_flow * remaining_volume * tss
    )
    inequality[row, layout.inventory_index] = effluent_flow * underflow
    row += 1
    if row != layout.inequality_count:
        raise AssertionError("inequality row assembly is inconsistent")
    return NetworkOperators(
        layout=layout,
        equality_matrix=equality,
        equality_rhs=rhs,
        inequality_matrix=inequality,
        primary_flow=primary_flow,
        clarifier_flow=clarifier_flow,
        underflow=underflow,
        effluent_flow=effluent_flow,
        clarifier_volume_m3=float(clarifier_volume_m3),
    )


def no_conversion_feasible_state(
    influent: npt.ArrayLike, *, operators: NetworkOperators, tss_weights: npt.ArrayLike
) -> FloatArray:
    """Construct the analytical feasible point used to prove QP nonemptiness."""
    from surrogate_optimization.surrogate.regression import SurrogateValidationError
    from surrogate_optimization.surrogate.regression import _finite_array

    x = _finite_array(influent, name="influent", ndim=1)
    tss = _finite_array(tss_weights, name="tss_weights", ndim=1)
    layout = operators.layout
    if x.size != layout.component_count or tss.size != layout.component_count:
        raise SurrogateValidationError(
            "no-conversion state has inconsistent component dimensions."
        )
    state = np.empty(layout.state_size, dtype=np.float64)
    state[layout.mixer_slice] = x
    for stage in range(layout.stage_count):
        state[layout.reactor_slice(stage)] = x
    state[layout.overflow_flow_slice] = operators.effluent_flow * x
    state[layout.underflow_flow_slice] = operators.underflow * x
    state[layout.inventory_index] = operators.clarifier_volume_m3 * float(tss @ x)
    return state


@dataclass(frozen=True)
class NetworkRowScales:
    equality: FloatArray
    inequality: FloatArray


def _flow_vector(value: npt.ArrayLike, row_count: int, *, name: str) -> FloatArray:
    from surrogate_optimization.surrogate.regression import SurrogateValidationError
    from surrogate_optimization.surrogate.regression import _finite_array

    array = _finite_array(value, name=name)
    if array.ndim == 0:
        return np.full(row_count, float(array), dtype=np.float64)
    if array.ndim != 1 or array.size != row_count:
        raise SurrogateValidationError(
            f"{name} must be scalar or have one value per state row."
        )
    return array


def fit_network_row_scales(
    states: npt.ArrayLike,
    influents: npt.ArrayLike,
    *,
    internal_recycle: npt.ArrayLike,
    return_recycle: npt.ArrayLike,
    waste_fraction: npt.ArrayLike,
    invariant_operator: npt.ArrayLike,
    tss_weights: npt.ArrayLike,
    layout: NetworkLayout | None = None,
    clarifier_volume_m3: float = 6000.0,
    minimum_scale: float = 1e-12,
    overflow_tss_closure: npt.ArrayLike | None = None,
) -> NetworkRowScales:
    """Fit D_b and D_g from the named physical terms in the model."""
    from surrogate_optimization.surrogate.regression import SurrogateValidationError
    from surrogate_optimization.surrogate.regression import _finite_array

    layout = layout or NetworkLayout()
    state_matrix = _finite_array(states, name="states", ndim=2)
    influent_matrix = _finite_array(influents, name="influents", ndim=2)
    invariant = _finite_array(invariant_operator, name="invariant_operator", ndim=2)
    tss = _finite_array(tss_weights, name="tss_weights", ndim=1)
    row_count = state_matrix.shape[0]
    if state_matrix.shape[1] != layout.state_size:
        raise SurrogateValidationError(f"states must have {layout.state_size} columns.")
    if influent_matrix.shape != (row_count, layout.component_count):
        raise SurrogateValidationError(
            "influents do not match the state rows and component count."
        )
    if (
        invariant.shape[1] != layout.component_count
        or tss.size != layout.component_count
    ):
        raise SurrogateValidationError(
            "invariant or TSS component dimensions are inconsistent."
        )
    if minimum_scale <= 0.0:
        raise SurrogateValidationError("minimum_scale must be positive.")
    if not np.isfinite(clarifier_volume_m3) or clarifier_volume_m3 <= 0.0:
        raise SurrogateValidationError(
            "clarifier_volume_m3 must be positive and finite."
        )
    r_internal = _flow_vector(internal_recycle, row_count, name="internal_recycle")
    r_return = _flow_vector(return_recycle, row_count, name="return_recycle")
    waste = _flow_vector(waste_fraction, row_count, name="waste_fraction")
    if (
        np.any(r_internal < 0.0)
        or np.any(r_return < 0.0)
        or np.any((waste <= 0.0) | (waste >= 1.0))
    ):
        raise SurrogateValidationError(
            "row-scale flow inputs are outside their physical ranges."
        )
    q_primary = 1.0 + r_internal + r_return
    q_clarifier = 1.0 + r_return
    q_underflow = r_return + waste
    q_effluent = 1.0 - waste
    closure = None
    if overflow_tss_closure is not None:
        closure = _flow_vector(
            overflow_tss_closure, row_count, name="overflow_tss_closure"
        )
        if np.any(closure <= 0.0):
            raise SurrogateValidationError(
                "overflow_tss_closure must be strictly positive"
            )
    invariant_count = invariant.shape[0]
    equality_count = (
        layout.component_count
        + layout.stage_count * invariant_count
        + layout.component_count
        + len(layout.soluble_indices)
        + int(closure is not None)
    )
    equality_mean_square = np.zeros((row_count, equality_count), dtype=np.float64)
    equality_term_count = np.zeros(equality_count, dtype=np.float64)
    inequality_mean_square = np.zeros(
        (row_count, layout.inequality_count), dtype=np.float64
    )
    inequality_term_count = np.zeros(layout.inequality_count, dtype=np.float64)
    mixer = state_matrix[:, layout.mixer_slice]
    final_reactor = state_matrix[:, layout.reactor_slice(layout.stage_count - 1)]
    overflow = state_matrix[:, layout.overflow_flow_slice]
    underflow_flow = state_matrix[:, layout.underflow_flow_slice]
    clarifier_inventory = state_matrix[:, layout.inventory_index]
    row = 0
    mixer_terms = (
        q_primary[:, None] * mixer,
        -influent_matrix,
        -r_internal[:, None] * final_reactor,
        -(r_return / q_underflow)[:, None] * underflow_flow,
    )
    for term in mixer_terms:
        equality_mean_square[:, row : row + layout.component_count] += np.square(term)
    equality_term_count[row : row + layout.component_count] = len(mixer_terms)
    row += layout.component_count
    previous = mixer
    for stage in range(layout.stage_count):
        current = state_matrix[:, layout.reactor_slice(stage)]
        current_invariant = current @ invariant.T
        previous_invariant = previous @ invariant.T
        equality_mean_square[:, row : row + invariant_count] = np.square(
            current_invariant
        ) + np.square(previous_invariant)
        equality_term_count[row : row + invariant_count] = 2.0
        row += invariant_count
        previous = current
    clarifier_terms = (overflow, underflow_flow, -q_clarifier[:, None] * final_reactor)
    for term in clarifier_terms:
        equality_mean_square[:, row : row + layout.component_count] += np.square(term)
    equality_term_count[row : row + layout.component_count] = len(clarifier_terms)
    row += layout.component_count
    for component in layout.soluble_indices:
        equality_mean_square[:, row] = np.square(
            underflow_flow[:, component]
        ) + np.square(q_underflow * final_reactor[:, component])
        equality_term_count[row] = 2.0
        row += 1
    if closure is not None:
        overflow_tss_flow = overflow @ tss
        predicted_tss_flow = q_effluent * closure
        equality_mean_square[:, row] = np.square(overflow_tss_flow) + np.square(
            predicted_tss_flow
        )
        equality_term_count[row] = 2.0
        row += 1
    if row != equality_count:
        raise AssertionError("equality scale serialization is inconsistent")
    row = 0
    for component in layout.particulate_indices:
        inequality_mean_square[:, row] = np.square(
            q_underflow * final_reactor[:, component]
        ) + np.square(underflow_flow[:, component])
        inequality_term_count[row] = 2.0
        row += 1
    endpoint_layer_volume = float(clarifier_volume_m3) / layout.layer_count
    remaining_volume = float(clarifier_volume_m3) - endpoint_layer_volume
    overflow_tss_flow = overflow @ tss
    underflow_tss_flow = underflow_flow @ tss
    lower_inventory_terms = (
        q_underflow * remaining_volume * overflow_tss_flow,
        q_effluent * endpoint_layer_volume * underflow_tss_flow,
        -q_effluent * q_underflow * clarifier_inventory,
    )
    for term in lower_inventory_terms:
        inequality_mean_square[:, row] += np.square(term)
    inequality_term_count[row] = len(lower_inventory_terms)
    row += 1
    upper_inventory_terms = (
        q_effluent * q_underflow * clarifier_inventory,
        -q_underflow * endpoint_layer_volume * overflow_tss_flow,
        -q_effluent * remaining_volume * underflow_tss_flow,
    )
    for term in upper_inventory_terms:
        inequality_mean_square[:, row] += np.square(term)
    inequality_term_count[row] = len(upper_inventory_terms)
    row += 1
    if row != layout.inequality_count:
        raise AssertionError("inequality scale serialization is inconsistent")
    equality_scale = np.sqrt(
        np.mean(equality_mean_square / equality_term_count[None, :], axis=0)
    )
    inequality_scale = np.sqrt(
        np.mean(inequality_mean_square / inequality_term_count[None, :], axis=0)
    )
    equality_scale = np.maximum(minimum_scale, equality_scale)
    inequality_scale = np.maximum(minimum_scale, inequality_scale)
    if not np.all(np.isfinite(equality_scale)) or not np.all(
        np.isfinite(inequality_scale)
    ):
        raise SurrogateValidationError("a fitted network row scale is non-finite.")
    return NetworkRowScales(equality=equality_scale, inequality=inequality_scale)


@dataclass(frozen=True)
class ProjectionWarmStart:
    displacement: FloatArray
    dual: FloatArray


@dataclass(frozen=True)
class ProjectionDiagnostics:
    status: str
    status_value: int
    iterations: int
    equality_rank_tolerance: float
    equality_smallest_singular_value: float
    equality_condition_number: float
    equality_residual: float
    inequality_residual: float
    nonnegativity_residual: float
    dual_feasibility_residual: float
    stationarity_residual: float
    complementarity_residual: float
    retried_cold: bool
    active_inequality_count: int = 0
    multipliers_reconstructed: bool = False
    solver_attempts: int = 1
    fallback_used: bool = False

    def as_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectionResult:
    state: FloatArray
    displacement: FloatArray
    equality_multipliers: FloatArray
    inequality_multipliers: FloatArray
    inequality_slack: FloatArray
    diagnostics: ProjectionDiagnostics
    accepted: bool

    @property
    def warm_start(self) -> ProjectionWarmStart:
        dual = np.concatenate((self.equality_multipliers, self.inequality_multipliers))
        return ProjectionWarmStart(self.displacement.copy(), dual)


class PhysicalProjector:
    """Solve and independently accept the strictly convex network QP."""

    def __init__(
        self,
        state_scale: npt.ArrayLike,
        equality_scale: npt.ArrayLike,
        inequality_scale: npt.ArrayLike,
        *,
        absolute_tolerance: float = 1e-08,
        relative_tolerance: float = 1e-08,
        maximum_iterations: int = 100000,
        polish: bool = True,
        equality_acceptance_tolerance: float = 1e-08,
        inequality_acceptance_tolerance: float = 1e-08,
        nonnegativity_acceptance_tolerance: float = 1e-10,
        active_set_tolerance: float = 1e-07,
    ) -> None:
        from surrogate_optimization.surrogate.regression import SurrogateValidationError
        from surrogate_optimization.surrogate.regression import _finite_array

        self.state_scale = _finite_array(state_scale, name="state_scale", ndim=1).copy()
        self.equality_scale = _finite_array(
            equality_scale, name="equality_scale", ndim=1
        ).copy()
        self.inequality_scale = _finite_array(
            inequality_scale, name="inequality_scale", ndim=1
        ).copy()
        if (
            np.any(self.state_scale <= 0.0)
            or np.any(self.equality_scale <= 0.0)
            or np.any(self.inequality_scale <= 0.0)
        ):
            raise SurrogateValidationError(
                "all projection scales must be strictly positive."
            )
        if (
            absolute_tolerance <= 0.0
            or relative_tolerance <= 0.0
            or maximum_iterations < 1
        ):
            raise SurrogateValidationError(
                "OSQP tolerances and iteration limit must be positive."
            )
        if not np.isfinite(active_set_tolerance) or active_set_tolerance <= 0.0:
            raise SurrogateValidationError(
                "active_set_tolerance must be finite and positive."
            )
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.maximum_iterations = int(maximum_iterations)
        self.polish = bool(polish)
        self.equality_acceptance_tolerance = float(equality_acceptance_tolerance)
        self.inequality_acceptance_tolerance = float(inequality_acceptance_tolerance)
        self.nonnegativity_acceptance_tolerance = float(
            nonnegativity_acceptance_tolerance
        )
        self.active_set_tolerance = float(active_set_tolerance)

    @staticmethod
    def _positive_part_norm(values: FloatArray) -> float:
        if values.size == 0:
            return 0.0
        return float(np.max(np.maximum(values, 0.0)))

    def _solve_once(
        self,
        *,
        scaled_equality: FloatArray,
        equality_rhs: FloatArray,
        scaled_inequality: FloatArray,
        inequality_rhs: FloatArray,
        warm_start: ProjectionWarmStart | None,
        rho: float | None = None,
        adaptive_rho: bool | None = None,
        maximum_iterations: int | None = None,
    ) -> object:
        from surrogate_optimization.surrogate.regression import SurrogateValidationError
        from surrogate_optimization.surrogate.regression import _finite_array

        variable_count = self.state_scale.size
        constraint_matrix = sparse.csc_matrix(
            np.vstack((scaled_equality, scaled_inequality)), dtype=np.float64
        )
        lower = np.concatenate(
            (equality_rhs, np.full(inequality_rhs.size, -np.inf, dtype=np.float64))
        )
        upper = np.concatenate((equality_rhs, inequality_rhs))
        solver = osqp.OSQP()
        setup_options: dict[str, object] = {}
        if rho is not None:
            setup_options["rho"] = float(rho)
        if adaptive_rho is not None:
            setup_options["adaptive_rho"] = bool(adaptive_rho)
        solver.setup(
            P=sparse.eye(variable_count, format="csc", dtype=np.float64),
            q=np.zeros(variable_count, dtype=np.float64),
            A=constraint_matrix,
            l=lower,
            u=upper,
            eps_abs=self.absolute_tolerance,
            eps_rel=self.relative_tolerance,
            max_iter=self.maximum_iterations
            if maximum_iterations is None
            else int(maximum_iterations),
            polishing=self.polish,
            verbose=False,
            **setup_options,
        )
        if warm_start is not None:
            displacement = _finite_array(
                warm_start.displacement, name="warm displacement", ndim=1
            )
            dual = _finite_array(warm_start.dual, name="warm dual", ndim=1)
            if (
                displacement.size != variable_count
                or dual.size != constraint_matrix.shape[0]
            ):
                raise SurrogateValidationError(
                    "projection warm-start dimensions are inconsistent."
                )
            solver.warm_start(x=displacement, y=dual)
        return solver.solve(raise_error=False)

    def _reconstruct_multipliers(
        self,
        displacement: FloatArray,
        scaled_equality: FloatArray,
        scaled_inequality: FloatArray,
        inequality_rhs: FloatArray,
    ) -> tuple[FloatArray, FloatArray, int]:
        """Recover an independently audited KKT multiplier representation.

        BVLS is used because its active-set termination directly controls the
        residual of this small dense bounded problem.  The previous TRF solve
        could declare first-order convergence while leaving a stationarity
        residual above the projection gate, and a subsequent minimum-norm QP
        could introduce another unrelated numerical failure.  No
        solver-reported projection-QP dual is used.
        """
        from surrogate_optimization.surrogate.regression import ProjectionError

        equality_count = scaled_equality.shape[0]
        inequality_count = scaled_inequality.shape[0]
        inequality_value = scaled_inequality @ displacement - inequality_rhs
        active = np.flatnonzero(inequality_value >= -self.active_set_tolerance)
        multiplier_matrix = np.column_stack(
            (scaled_equality.T, scaled_inequality[active].T)
        )
        lower = np.concatenate(
            (
                np.full(equality_count, -np.inf, dtype=np.float64),
                np.zeros(active.size, dtype=np.float64),
            )
        )
        upper = np.full(lower.size, np.inf, dtype=np.float64)
        least_squares = lsq_linear(
            multiplier_matrix,
            -displacement,
            bounds=(lower, upper),
            method="bvls",
            tol=1e-12,
            max_iter=10000,
        )
        if not least_squares.success or not np.all(np.isfinite(least_squares.x)):
            raise ProjectionError(
                "active-set multiplier reconstruction did not solve its bounded least-squares problem."
            )
        multipliers = np.asarray(least_squares.x, dtype=np.float64)
        multipliers[equality_count:] = np.maximum(multipliers[equality_count:], 0.0)
        equality_multipliers = multipliers[:equality_count]
        inequality_multipliers = np.zeros(inequality_count, dtype=np.float64)
        inequality_multipliers[active] = multipliers[equality_count:]
        return (equality_multipliers, inequality_multipliers, int(active.size))

    def project(
        self,
        raw_state: npt.ArrayLike,
        equality_matrix: npt.ArrayLike,
        equality_rhs: npt.ArrayLike,
        inequality_matrix: npt.ArrayLike,
        *,
        warm_start: ProjectionWarmStart | None = None,
        raise_on_failure: bool = True,
    ) -> ProjectionResult:
        from surrogate_optimization.surrogate.regression import ProjectionError
        from surrogate_optimization.surrogate.regression import SurrogateValidationError
        from surrogate_optimization.surrogate.regression import _finite_array

        raw = _finite_array(raw_state, name="raw_state", ndim=1)
        equality = _finite_array(equality_matrix, name="equality_matrix", ndim=2)
        rhs = _finite_array(equality_rhs, name="equality_rhs", ndim=1)
        inequality = _finite_array(inequality_matrix, name="inequality_matrix", ndim=2)
        variable_count = raw.size
        if variable_count != self.state_scale.size:
            raise SurrogateValidationError("raw state and state_scale sizes differ.")
        if (
            equality.shape != (self.equality_scale.size, variable_count)
            or rhs.size != equality.shape[0]
        ):
            raise SurrogateValidationError(
                "equality dimensions do not match their fitted scales."
            )
        if inequality.shape != (self.inequality_scale.size, variable_count):
            raise SurrogateValidationError(
                "inequality dimensions do not match their fitted scales."
            )
        scaled_equality = (
            equality * self.state_scale[None, :] / self.equality_scale[:, None]
        )
        required_equality = (rhs - equality @ raw) / self.equality_scale
        physical_scaled_inequality = inequality / self.inequality_scale[:, None]
        scaled_network_inequality = (
            physical_scaled_inequality * self.state_scale[None, :]
        )
        network_inequality_rhs = -(physical_scaled_inequality @ raw)
        scaled_inequality = np.vstack(
            (-np.eye(variable_count), scaled_network_inequality)
        )
        inequality_rhs = np.concatenate(
            (raw / self.state_scale, network_inequality_rhs)
        )
        singular_values = linalg.svdvals(scaled_equality, check_finite=True)
        if singular_values.size == 0:
            raise ProjectionError(
                "the physical projection requires at least one equality row."
            )
        rank_tolerance = (
            max(scaled_equality.shape) * np.finfo(np.float64).eps * singular_values[0]
        )
        if singular_values[-1] <= rank_tolerance:
            raise ProjectionError("scaled equality matrix is not full row rank.")
        equality_condition = float(singular_values[0] / singular_values[-1])
        if warm_start is not None:
            self._solve_once(
                scaled_equality=scaled_equality,
                equality_rhs=required_equality,
                scaled_inequality=scaled_inequality,
                inequality_rhs=inequality_rhs,
                warm_start=warm_start,
            )

        def audited_result(solver_result: object, attempt: int) -> ProjectionResult:
            from surrogate_optimization.surrogate.regression import ProjectionError

            displacement = np.asarray(
                solver_result.x
                if solver_result.x is not None
                else np.full(variable_count, np.nan),
                dtype=np.float64,
            )
            state = raw + self.state_scale * displacement
            slack = inequality_rhs - scaled_inequality @ displacement
            multipliers_reconstructed = False
            active_inequality_count = 0
            if np.all(np.isfinite(displacement)):
                try:
                    equality_dual, inequality_dual, active_inequality_count = (
                        self._reconstruct_multipliers(
                            displacement,
                            scaled_equality,
                            scaled_inequality,
                            inequality_rhs,
                        )
                    )
                    multipliers_reconstructed = True
                except ProjectionError:
                    equality_dual = np.full(scaled_equality.shape[0], np.nan)
                    inequality_dual = np.full(scaled_inequality.shape[0], np.nan)
            else:
                equality_dual = np.full(scaled_equality.shape[0], np.nan)
                inequality_dual = np.full(scaled_inequality.shape[0], np.nan)
            arrays_finite = all(
                (
                    np.all(np.isfinite(array))
                    for array in (
                        displacement,
                        state,
                        slack,
                        equality_dual,
                        inequality_dual,
                    )
                )
            )
            if arrays_finite:
                equality_residual = float(
                    np.linalg.norm(
                        scaled_equality @ displacement - required_equality, ord=np.inf
                    )
                )
                inequality_residual = self._positive_part_norm(
                    physical_scaled_inequality @ state
                )
                nonnegativity_residual = self._positive_part_norm(
                    -state / self.state_scale
                )
                dual_residual = self._positive_part_norm(-inequality_dual)
                equality_stationarity = scaled_equality.T @ equality_dual
                inequality_stationarity = scaled_inequality.T @ inequality_dual
                stationarity = (
                    displacement + equality_stationarity + inequality_stationarity
                )
                stationarity_residual = float(
                    np.max(
                        np.abs(stationarity)
                        / (
                            1.0
                            + np.abs(displacement)
                            + np.abs(equality_stationarity)
                            + np.abs(inequality_stationarity)
                        )
                    )
                )
                complementarity_residual = float(
                    np.linalg.norm(inequality_dual * slack, ord=np.inf)
                )
            else:
                equality_residual = inequality_residual = nonnegativity_residual = (
                    np.inf
                )
                dual_residual = stationarity_residual = complementarity_residual = (
                    np.inf
                )
            accepted = (
                arrays_finite
                and multipliers_reconstructed
                and (equality_residual <= self.equality_acceptance_tolerance)
                and (inequality_residual <= self.inequality_acceptance_tolerance)
                and (nonnegativity_residual <= self.nonnegativity_acceptance_tolerance)
                and (dual_residual <= self.inequality_acceptance_tolerance)
                and (stationarity_residual <= self.inequality_acceptance_tolerance)
                and (complementarity_residual <= self.inequality_acceptance_tolerance)
            )
            diagnostics = ProjectionDiagnostics(
                status=str(solver_result.info.status),
                status_value=int(solver_result.info.status_val),
                iterations=int(solver_result.info.iter),
                equality_rank_tolerance=float(rank_tolerance),
                equality_smallest_singular_value=float(singular_values[-1]),
                equality_condition_number=equality_condition,
                equality_residual=equality_residual,
                inequality_residual=inequality_residual,
                nonnegativity_residual=nonnegativity_residual,
                dual_feasibility_residual=dual_residual,
                stationarity_residual=stationarity_residual,
                complementarity_residual=complementarity_residual,
                retried_cold=warm_start is not None or attempt > 1,
                active_inequality_count=active_inequality_count,
                multipliers_reconstructed=multipliers_reconstructed,
                solver_attempts=attempt,
                fallback_used=attempt > 1,
            )
            return ProjectionResult(
                state=state,
                displacement=displacement,
                equality_multipliers=equality_dual,
                inequality_multipliers=inequality_dual,
                inequality_slack=slack,
                diagnostics=diagnostics,
                accepted=accepted,
            )

        cold_attempts = (
            {},
            {"rho": 0.01, "adaptive_rho": False, "maximum_iterations": 200000},
            {"rho": 10.0, "adaptive_rho": False, "maximum_iterations": 200000},
        )
        attempted_results: list[ProjectionResult] = []
        for attempt, options in enumerate(cold_attempts, start=1):
            solver_result = self._solve_once(
                scaled_equality=scaled_equality,
                equality_rhs=required_equality,
                scaled_inequality=scaled_inequality,
                inequality_rhs=inequality_rhs,
                warm_start=None,
                **options,
            )
            candidate = audited_result(solver_result, attempt)
            attempted_results.append(candidate)
            if candidate.accepted:
                return candidate

        def failure_score(result: ProjectionResult) -> float:
            diagnostics = result.diagnostics
            scaled_residuals = (
                diagnostics.equality_residual / self.equality_acceptance_tolerance,
                diagnostics.inequality_residual / self.inequality_acceptance_tolerance,
                diagnostics.nonnegativity_residual
                / self.nonnegativity_acceptance_tolerance,
                diagnostics.dual_feasibility_residual
                / self.inequality_acceptance_tolerance,
                diagnostics.stationarity_residual
                / self.inequality_acceptance_tolerance,
                diagnostics.complementarity_residual
                / self.inequality_acceptance_tolerance,
            )
            return float(max(scaled_residuals))

        final_result = min(attempted_results, key=failure_score)
        final_result = replace(
            final_result,
            diagnostics=replace(
                final_result.diagnostics,
                retried_cold=True,
                solver_attempts=len(attempted_results),
                fallback_used=True,
            ),
        )
        if raise_on_failure:
            diagnostics = final_result.diagnostics
            raise ProjectionError(
                f"physical QP failed independent acceptance: status={diagnostics.status!r}, r_E={diagnostics.equality_residual:.3e}, r_G={diagnostics.inequality_residual:.3e}, r_+={diagnostics.nonnegativity_residual:.3e}, r_stat={diagnostics.stationarity_residual:.3e}."
            )
        return final_result
