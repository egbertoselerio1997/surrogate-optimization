"""Optimization types."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
    from surrogate_optimization.surrogate.regression import LogOverflowTSSClosure
    from surrogate_optimization.surrogate.projection import NetworkLayout
    from surrogate_optimization.surrogate.projection import NetworkRowScales
    from surrogate_optimization.surrogate.projection import ProjectionResult
    from surrogate_optimization.surrogate.regression import QuadraticSurrogate
    from surrogate_optimization.optimization.surrogate import TrustRowCallback
from surrogate_optimization.config import DECISION_LOWER
from surrogate_optimization.config import DECISION_UPPER
from surrogate_optimization.optimization.surrogate import DEFAULT_OBJECTIVE_WEIGHTS
from surrogate_optimization.optimization.surrogate import EXACT_QP_SINGLE_START_PROTOCOL
from surrogate_optimization.optimization.surrogate import _ENGINEERING_PARAMETERS
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from scipy import linalg
from typing import Any
from typing import Mapping
import casadi as ca
import numpy as np


class SurrogateNLPError(RuntimeError):
    """Raised when the surrogate route cannot satisfy its contract."""


@dataclass(frozen=True)
class EngineeringLimits:
    """Physical constants and retained case-study engineering limits."""

    fresh_flow_m3_d: float = 10000.0
    clarifier_area_m2: float = 1500.0
    clarifier_volume_m3: float = 6000.0
    external_loss_min_g_m3: float = float(
        _ENGINEERING_PARAMETERS["external_solids_loss_min_g_m3"]
    )
    underflow_tss_upper_g_m3: float = float(
        _ENGINEERING_PARAMETERS["underflow_tss_max_g_m3"]
    )
    feed_tss_min_g_m3: float = float(_ENGINEERING_PARAMETERS["feed_tss_min_g_m3"])

    def __post_init__(self) -> None:
        positive = (
            self.fresh_flow_m3_d,
            self.clarifier_area_m2,
            self.clarifier_volume_m3,
            self.external_loss_min_g_m3,
            self.underflow_tss_upper_g_m3,
            self.feed_tss_min_g_m3,
        )
        if not all((np.isfinite(value) and value > 0.0 for value in positive)):
            raise ValueError(
                "engineering constants and finite limits must be positive."
            )


@dataclass(frozen=True)
class TrustThresholds:
    """Frozen development-supported limits used by the upper problem."""

    correction_rms: float
    regularized_leverage: float
    split_rms: float | None = None
    reactor_rms: float | None = None

    def __post_init__(self) -> None:
        required = (self.correction_rms, self.regularized_leverage)
        if not all((np.isfinite(value) and value >= 0.0 for value in required)):
            raise ValueError("native trust thresholds must be finite and nonnegative.")
        for value in (self.split_rms, self.reactor_rms):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(
                    "optional trust thresholds must be finite and nonnegative."
                )


@dataclass(frozen=True)
class NamedTrustRows:
    """One additional scaled-residual family.

    The callback receives ``(controls, raw, projected, influent)`` and returns a
    CasADi-compatible vector of already scaled residual rows.  Its constraint
    is ``mean(rows**2) <= rms_threshold**2``.
    """

    name: str
    callback: TrustRowCallback
    rms_threshold: float

    def __post_init__(self) -> None:
        if not self.name or not callable(self.callback):
            raise ValueError("a named trust row requires a name and callable.")
        if not np.isfinite(self.rms_threshold) or self.rms_threshold < 0.0:
            raise ValueError(
                "a trust-row RMS threshold must be finite and nonnegative."
            )


@dataclass(frozen=True)
class TrustDiagnosticCallbacks:
    """Optional particulate-split and smooth-reactor residual rows."""

    split_rows: TrustRowCallback | None = None
    reactor_rows: TrustRowCallback | None = None
    additional: tuple[NamedTrustRows, ...] = ()


@dataclass(frozen=True)
class SurrogateCase:
    influent: FloatArray
    case_id: str = "nominal"
    objective_weights: FloatArray = field(
        default_factory=lambda: DEFAULT_OBJECTIVE_WEIGHTS.copy()
    )
    quality_weights: FloatArray | None = None

    def parameter_vector(self, assets: "SurrogateNLPAssets") -> FloatArray:
        from surrogate_optimization.optimization.surrogate import _vector

        influent = _vector(
            self.influent, assets.layout.component_count, "case influent"
        )
        objective = _vector(self.objective_weights, 6, "objective_weights")
        if np.any(objective < 0.0) or not np.isclose(
            np.sum(objective), 1.0, atol=1e-12
        ):
            raise ValueError("objective_weights must be nonnegative and sum to one.")
        if self.quality_weights is None:
            quality = np.full(assets.quality_count, 1.0 / assets.quality_count)
        else:
            quality = _vector(
                self.quality_weights, assets.quality_count, "quality_weights"
            )
            if np.any(quality < 0.0) or not np.isclose(
                np.sum(quality), 1.0, atol=1e-12
            ):
                raise ValueError("quality_weights must be nonnegative and sum to one.")
        return np.concatenate((influent, objective, quality))


@dataclass(frozen=True)
class SurrogateNLPAssets:
    """Frozen numerical data shared by all influent cases and gap stages."""

    model: QuadraticSurrogate
    layout: NetworkLayout
    invariant_operator: FloatArray
    tss_weights: FloatArray
    row_scales: NetworkRowScales
    leverage_precision: FloatArray
    trust_thresholds: TrustThresholds
    quality_operator: FloatArray
    quality_scale: FloatArray
    overflow_closure: LogOverflowTSSClosure | None = None
    trust_callbacks: TrustDiagnosticCallbacks = field(
        default_factory=TrustDiagnosticCallbacks
    )
    engineering: EngineeringLimits = field(default_factory=EngineeringLimits)
    theta_lower: FloatArray = field(default_factory=lambda: DECISION_LOWER.copy())
    theta_upper: FloatArray = field(default_factory=lambda: DECISION_UPPER.copy())

    def __post_init__(self) -> None:
        from surrogate_optimization.surrogate.projection import NetworkRowScales
        from surrogate_optimization.optimization.surrogate import _matrix
        from surrogate_optimization.optimization.surrogate import _vector

        if self.layout.layer_count < 3:
            raise ValueError(
                "the surrogate route requires at least three Clarifier layers."
            )
        if self.model.feature_map.decision_count != 7:
            raise ValueError(
                "the surrogate surrogate route requires exactly seven controls."
            )
        if self.model.feature_map.influent_count != self.layout.component_count:
            raise ValueError(
                "surrogate influent coordinates must equal the component count."
            )
        if self.model.response_center.size != self.layout.state_size:
            raise ValueError(
                "surrogate response coordinates do not match the network layout."
            )
        component_count = self.layout.component_count
        invariant = np.asarray(self.invariant_operator, dtype=np.float64)
        if (
            invariant.ndim != 2
            or invariant.shape[1] != component_count
            or invariant.shape[0] == 0
            or (not np.all(np.isfinite(invariant)))
            or (np.linalg.matrix_rank(invariant) != invariant.shape[0])
        ):
            raise ValueError("invariant_operator must be finite and full row rank.")
        equality_count = (
            2 * component_count
            + self.layout.stage_count * invariant.shape[0]
            + len(self.layout.soluble_indices)
            + int(self.overflow_closure is not None)
        )
        equality_scale = _vector(
            self.row_scales.equality,
            equality_count,
            "equality row scales",
            positive=True,
        )
        inequality_scale = _vector(
            self.row_scales.inequality,
            self.layout.inequality_count,
            "inequality row scales",
            positive=True,
        )
        tss = _vector(self.tss_weights, component_count, "tss_weights")
        if np.any(tss < 0.0) or not np.any(tss > 0.0):
            raise ValueError("tss_weights must be nonnegative and nonzero.")
        if self.overflow_closure is not None:
            closure_map = self.overflow_closure.model.feature_map
            if (
                closure_map.decision_count != self.model.feature_map.decision_count
                or closure_map.influent_count != self.model.feature_map.influent_count
            ):
                raise ValueError(
                    "overflow closure input dimensions do not match the surrogate"
                )
        feature_count = self.model.feature_map.feature_count
        leverage = _matrix(
            self.leverage_precision,
            (feature_count, feature_count),
            "leverage_precision",
        )
        if not np.allclose(leverage, leverage.T, rtol=1e-10, atol=1e-12):
            raise ValueError("leverage_precision must be symmetric.")
        minimum_eigenvalue = float(linalg.eigvalsh(leverage, check_finite=True)[0])
        if minimum_eigenvalue < -1e-10 * max(1.0, float(np.linalg.norm(leverage, 2))):
            raise ValueError("leverage_precision must be positive semidefinite.")
        quality = np.asarray(self.quality_operator, dtype=np.float64)
        if (
            quality.ndim != 2
            or quality.shape[0] == 0
            or quality.shape[1] != component_count
            or (not np.all(np.isfinite(quality)))
        ):
            raise ValueError("quality_operator must have one column per component.")
        quality_scale = _vector(
            self.quality_scale, quality.shape[0], "quality_scale", positive=True
        )
        lower = _vector(self.theta_lower, 7, "theta_lower")
        upper = _vector(self.theta_upper, 7, "theta_upper")
        if np.any(upper <= lower):
            raise ValueError("every control must have a positive operating span.")
        if lower[5] + lower[6] <= 0.0 or upper[6] >= 1.0:
            raise ValueError("control bounds must keep q_U positive and q_E positive.")
        pairs = (
            (self.trust_callbacks.split_rows, self.trust_thresholds.split_rms, "split"),
            (
                self.trust_callbacks.reactor_rows,
                self.trust_thresholds.reactor_rms,
                "reactor",
            ),
        )
        for callback, threshold, name in pairs:
            if (callback is None) != (threshold is None):
                raise ValueError(
                    f"{name} trust rows and their threshold must be supplied together."
                )
        names = [item.name for item in self.trust_callbacks.additional]
        if len(set(names)) != len(names) or any(
            (name in {"correction", "leverage", "split", "reactor"} for name in names)
        ):
            raise ValueError(
                "additional trust diagnostic names must be unique and reserved-name free."
            )
        object.__setattr__(self, "invariant_operator", invariant.copy())
        object.__setattr__(self, "tss_weights", tss)
        object.__setattr__(
            self, "row_scales", NetworkRowScales(equality_scale, inequality_scale)
        )
        object.__setattr__(self, "leverage_precision", 0.5 * (leverage + leverage.T))
        object.__setattr__(self, "quality_operator", quality.copy())
        object.__setattr__(self, "quality_scale", quality_scale)
        object.__setattr__(self, "theta_lower", lower)
        object.__setattr__(self, "theta_upper", upper)

    @property
    def quality_count(self) -> int:
        return int(self.quality_operator.shape[0])

    @property
    def equality_count(self) -> int:
        return int(self.row_scales.equality.size)

    @property
    def network_inequality_count(self) -> int:
        return int(self.row_scales.inequality.size)

    @property
    def projection_inequality_count(self) -> int:
        return self.layout.state_size + self.network_inequality_count

    @property
    def theta_span(self) -> FloatArray:
        return self.theta_upper - self.theta_lower


@dataclass(frozen=True)
class SymbolicNetworkOperators:
    equality_matrix: ca.MX
    equality_rhs: ca.MX
    inequality_matrix: ca.MX
    primary_flow: ca.MX
    clarifier_flow: ca.MX
    underflow: ca.MX
    effluent_flow: ca.MX


@dataclass(frozen=True)
class SurrogateSolverSettings:
    maximum_iterations: int = 2500
    tolerance: float = 1e-09
    acceptable_tolerance: float = 1e-08
    stage_feasibility_tolerance: float = 1e-07
    final_upper_tolerance: float = 1e-06
    outer_maximum_iterations: int = 250
    outer_function_tolerance: float = 1e-10
    perform_outer_refinement: bool = True
    print_level: int = 0
    maximum_wall_time: float | None = None

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1 or self.outer_maximum_iterations < 1:
            raise ValueError("solver iteration limits must be positive.")
        for value in (
            self.tolerance,
            self.acceptable_tolerance,
            self.stage_feasibility_tolerance,
            self.final_upper_tolerance,
            self.outer_function_tolerance,
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("solver tolerances must be finite and positive.")
        if self.maximum_wall_time is not None and (
            not np.isfinite(self.maximum_wall_time) or self.maximum_wall_time <= 0.0
        ):
            raise ValueError("maximum_wall_time must be positive when supplied.")

    def ipopt_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "print_time": False,
            "ipopt.print_level": self.print_level,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.maximum_iterations,
            "ipopt.tol": self.tolerance,
            "ipopt.acceptable_tol": self.acceptable_tolerance,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.bound_relax_factor": 0.0,
            "ipopt.honor_original_bounds": "yes",
            "ipopt.warm_start_init_point": "yes",
            "ipopt.warm_start_bound_push": 1e-09,
            "ipopt.warm_start_bound_frac": 1e-09,
            "ipopt.warm_start_slack_bound_push": 1e-09,
            "ipopt.warm_start_slack_bound_frac": 1e-09,
            "ipopt.warm_start_mult_bound_push": 1e-09,
        }
        if self.maximum_wall_time is not None:
            options["ipopt.max_wall_time"] = float(self.maximum_wall_time)
        return options


@dataclass
class SurrogateNLP:
    assets: SurrogateNLPAssets
    tau: float
    name: str
    variable_count: int
    equality_count: int
    inequality_count: int
    theta_slice: slice
    displacement_slice: slice
    equality_multiplier_slice: slice
    inequality_multiplier_slice: slice
    lower_bounds: FloatArray
    upper_bounds: FloatArray
    constraint_lower_bounds: FloatArray
    constraint_upper_bounds: FloatArray
    solver: Any | None
    objective_function: ca.Function
    raw_function: ca.Function
    feature_function: ca.Function
    state_function: ca.Function
    network_function: ca.Function
    equality_function: ca.Function
    inequality_function: ca.Function
    gap_function: ca.Function
    engineering_function: ca.Function
    trust_function: ca.Function
    upper_from_state_function: ca.Function
    objective_components_function: ca.Function
    engineering_quantity_function: ca.Function
    engineering_names: tuple[str, ...]
    trust_names: tuple[str, ...]

    @property
    def parameter_count(self) -> int:
        return self.assets.layout.component_count + 6 + self.assets.quality_count


@dataclass(frozen=True)
class FeasibilityRecord:
    finite: bool
    cold_projection: bool
    projection_accepted: bool
    control_bound_residual: float
    engineering_residual: float
    trust_residual: float
    maximum_upper_residual: float
    feasible: bool
    projection_reproduction_residual: float | None = None
    projection_reproduction_passed: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeasibilityRecord":
        from surrogate_optimization.optimization.surrogate import _float_or_nan
        from surrogate_optimization.optimization.surrogate import _optional_float

        return cls(
            finite=bool(value.get("finite", False)),
            cold_projection=bool(value.get("cold_projection", False)),
            projection_accepted=bool(value.get("projection_accepted", False)),
            control_bound_residual=_float_or_nan(value.get("control_bound_residual")),
            engineering_residual=_float_or_nan(value.get("engineering_residual")),
            trust_residual=_float_or_nan(value.get("trust_residual")),
            maximum_upper_residual=_float_or_nan(value.get("maximum_upper_residual")),
            feasible=bool(value.get("feasible", False)),
            projection_reproduction_residual=_optional_float(
                value.get("projection_reproduction_residual")
            ),
            projection_reproduction_passed=None
            if value.get("projection_reproduction_passed") is None
            else bool(value["projection_reproduction_passed"]),
        )


@dataclass(frozen=True)
class StationarityRecord:
    classification: str
    resolved: bool
    stationary: bool
    lower_qp_kkt_passed: bool
    upper_stationarity_residual: float | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StationarityRecord":
        from surrogate_optimization.optimization.surrogate import _optional_float

        return cls(
            classification=str(value.get("classification", "stationarity_unresolved")),
            resolved=bool(value.get("resolved", False)),
            stationary=bool(value.get("stationary", False)),
            lower_qp_kkt_passed=bool(value.get("lower_qp_kkt_passed", False)),
            upper_stationarity_residual=_optional_float(
                value.get("upper_stationarity_residual")
            ),
            reason=str(value.get("reason", "checkpoint did not record a reason")),
        )


@dataclass(frozen=True)
class OuterRefinementRecord:
    attempted: bool
    solver_success: bool
    status: str
    iterations: int
    evaluations: int
    elapsed_seconds: float
    initial_objective: float | None
    final_objective: float | None
    cold_qp_resolutions: int = 0
    derivative_error: str | None = None
    lower_active_set: dict[str, Any] | None = None
    upper_kkt: dict[str, Any] | None = None
    projection_reproduction_residual: float | None = None
    projection_reproduction_passed: bool | None = None
    method: str = "exact_qp_active_set_slsqp"
    fallback_used: bool = False
    fallback_method: str | None = None
    fallback_solver_success: bool | None = None
    fallback_status: str | None = None
    fallback_iterations: int = 0
    fallback_evaluations: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OuterRefinementRecord":
        from surrogate_optimization.optimization.surrogate import _float_or_nan
        from surrogate_optimization.optimization.surrogate import _optional_float

        lower = value.get("lower_active_set")
        upper = value.get("upper_kkt")
        return cls(
            attempted=bool(value.get("attempted", False)),
            solver_success=bool(value.get("solver_success", False)),
            status=str(value.get("status", "unknown")),
            iterations=int(value.get("iterations", 0)),
            evaluations=int(value.get("evaluations", 0)),
            elapsed_seconds=_float_or_nan(value.get("elapsed_seconds")),
            initial_objective=_optional_float(value.get("initial_objective")),
            final_objective=_optional_float(value.get("final_objective")),
            cold_qp_resolutions=int(value.get("cold_qp_resolutions", 0)),
            derivative_error=None
            if value.get("derivative_error") is None
            else str(value["derivative_error"]),
            lower_active_set=None if lower is None else dict(lower),
            upper_kkt=None if upper is None else dict(upper),
            projection_reproduction_residual=_optional_float(
                value.get("projection_reproduction_residual")
            ),
            projection_reproduction_passed=None
            if value.get("projection_reproduction_passed") is None
            else bool(value["projection_reproduction_passed"]),
            method=str(value.get("method", "exact_qp_active_set_slsqp")),
            fallback_used=bool(value.get("fallback_used", False)),
            fallback_method=None
            if value.get("fallback_method") is None
            else str(value["fallback_method"]),
            fallback_solver_success=None
            if value.get("fallback_solver_success") is None
            else bool(value["fallback_solver_success"]),
            fallback_status=None
            if value.get("fallback_status") is None
            else str(value["fallback_status"]),
            fallback_iterations=int(value.get("fallback_iterations", 0)),
            fallback_evaluations=int(value.get("fallback_evaluations", 0)),
        )


@dataclass(frozen=True)
class FinalCandidateRecord:
    normalized_controls: FloatArray
    controls: FloatArray
    raw: FloatArray
    projected: FloatArray
    displacement: FloatArray
    objective: float
    objective_components: FloatArray
    engineering_rows: FloatArray
    engineering_quantities: FloatArray
    trust_rows: FloatArray
    trust_values: FloatArray
    projection: ProjectionResult
    feasibility: FeasibilityRecord
    stationarity: StationarityRecord
    status: str
    lower_active_set: dict[str, Any] | None = None
    upper_kkt: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "normalized_controls": self.normalized_controls.tolist(),
            "controls": self.controls.tolist(),
            "raw": self.raw.tolist(),
            "projected": self.projected.tolist(),
            "displacement": self.displacement.tolist(),
            "objective": self.objective,
            "objective_components": self.objective_components.tolist(),
            "engineering_rows": self.engineering_rows.tolist(),
            "engineering_quantities": self.engineering_quantities.tolist(),
            "trust_rows": self.trust_rows.tolist(),
            "trust_values": self.trust_values.tolist(),
            "projection": {
                "accepted": self.projection.accepted,
                "state": self.projection.state.tolist(),
                "displacement": self.projection.displacement.tolist(),
                "equality_multipliers": self.projection.equality_multipliers.tolist(),
                "inequality_multipliers": self.projection.inequality_multipliers.tolist(),
                "inequality_slack": self.projection.inequality_slack.tolist(),
                "diagnostics": self.projection.diagnostics.as_dict(),
            },
            "feasibility": self.feasibility.as_dict(),
            "stationarity": self.stationarity.as_dict(),
            "status": self.status,
            "lower_active_set": self.lower_active_set,
            "upper_kkt": self.upper_kkt,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FinalCandidateRecord":
        from surrogate_optimization.surrogate.projection import ProjectionDiagnostics
        from surrogate_optimization.surrogate.projection import ProjectionResult
        from surrogate_optimization.optimization.surrogate import _float_or_nan

        projection_value = value["projection"]
        diagnostics_value = projection_value["diagnostics"]
        diagnostics = ProjectionDiagnostics(
            status=str(diagnostics_value.get("status", "unknown")),
            status_value=int(diagnostics_value.get("status_value", 0)),
            iterations=int(diagnostics_value.get("iterations", 0)),
            equality_rank_tolerance=_float_or_nan(
                diagnostics_value.get("equality_rank_tolerance")
            ),
            equality_smallest_singular_value=_float_or_nan(
                diagnostics_value.get("equality_smallest_singular_value")
            ),
            equality_condition_number=_float_or_nan(
                diagnostics_value.get("equality_condition_number")
            ),
            equality_residual=_float_or_nan(diagnostics_value.get("equality_residual")),
            inequality_residual=_float_or_nan(
                diagnostics_value.get("inequality_residual")
            ),
            nonnegativity_residual=_float_or_nan(
                diagnostics_value.get("nonnegativity_residual")
            ),
            dual_feasibility_residual=_float_or_nan(
                diagnostics_value.get("dual_feasibility_residual")
            ),
            stationarity_residual=_float_or_nan(
                diagnostics_value.get("stationarity_residual")
            ),
            complementarity_residual=_float_or_nan(
                diagnostics_value.get("complementarity_residual")
            ),
            retried_cold=bool(diagnostics_value.get("retried_cold", False)),
            active_inequality_count=int(
                diagnostics_value.get("active_inequality_count", 0)
            ),
            multipliers_reconstructed=bool(
                diagnostics_value.get("multipliers_reconstructed", False)
            ),
            solver_attempts=int(diagnostics_value.get("solver_attempts", 1)),
            fallback_used=bool(diagnostics_value.get("fallback_used", False)),
        )
        projection = ProjectionResult(
            state=np.asarray(projection_value.get("state"), dtype=np.float64).reshape(
                -1
            ),
            displacement=np.asarray(
                projection_value.get("displacement"), dtype=np.float64
            ).reshape(-1),
            equality_multipliers=np.asarray(
                projection_value.get("equality_multipliers"), dtype=np.float64
            ).reshape(-1),
            inequality_multipliers=np.asarray(
                projection_value.get("inequality_multipliers"), dtype=np.float64
            ).reshape(-1),
            inequality_slack=np.asarray(
                projection_value.get("inequality_slack"), dtype=np.float64
            ).reshape(-1),
            diagnostics=diagnostics,
            accepted=bool(projection_value.get("accepted", False)),
        )
        lower = value.get("lower_active_set")
        upper = value.get("upper_kkt")
        return cls(
            normalized_controls=np.asarray(
                value.get("normalized_controls"), dtype=np.float64
            ).reshape(-1),
            controls=np.asarray(value.get("controls"), dtype=np.float64).reshape(-1),
            raw=np.asarray(value.get("raw"), dtype=np.float64).reshape(-1),
            projected=np.asarray(value.get("projected"), dtype=np.float64).reshape(-1),
            displacement=np.asarray(
                value.get("displacement"), dtype=np.float64
            ).reshape(-1),
            objective=_float_or_nan(value.get("objective")),
            objective_components=np.asarray(
                value.get("objective_components"), dtype=np.float64
            ).reshape(-1),
            engineering_rows=np.asarray(
                value.get("engineering_rows"), dtype=np.float64
            ).reshape(-1),
            engineering_quantities=np.asarray(
                value.get("engineering_quantities"), dtype=np.float64
            ).reshape(-1),
            trust_rows=np.asarray(value.get("trust_rows"), dtype=np.float64).reshape(
                -1
            ),
            trust_values=np.asarray(
                value.get("trust_values"), dtype=np.float64
            ).reshape(-1),
            projection=projection,
            feasibility=FeasibilityRecord.from_dict(value["feasibility"]),
            stationarity=StationarityRecord.from_dict(value["stationarity"]),
            status=str(value.get("status", "unknown")),
            lower_active_set=None if lower is None else dict(lower),
            upper_kkt=None if upper is None else dict(upper),
        )


@dataclass(frozen=True)
class SurrogateStartResult:
    start_index: int
    initial_normalized_controls: FloatArray
    stages: tuple[()]
    outer_refinement: OuterRefinementRecord
    final: FinalCandidateRecord | None
    status: str
    error: str | None = None
    resume_contract: str | None = None
    protocol: str = EXACT_QP_SINGLE_START_PROTOCOL

    @property
    def feasible(self) -> bool:
        return self.final is not None and self.final.feasibility.feasible

    @property
    def stationary(self) -> bool:
        return self.final is not None and self.final.stationarity.stationary

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "initial_normalized_controls": self.initial_normalized_controls.tolist(),
            "stages": [],
            "outer_refinement": self.outer_refinement.as_dict(),
            "final": None if self.final is None else self.final.as_dict(),
            "status": self.status,
            "error": self.error,
            "resume_contract": self.resume_contract,
            "protocol": self.protocol,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SurrogateStartResult":
        if value.get("stages"):
            raise ValueError("surrogate checkpoints cannot contain continuation stages")
        final_value = value.get("final")
        return cls(
            start_index=int(value["start_index"]),
            initial_normalized_controls=np.asarray(
                value.get("initial_normalized_controls"), dtype=np.float64
            ).reshape(-1),
            stages=(),
            outer_refinement=OuterRefinementRecord.from_dict(
                value.get("outer_refinement", {})
            ),
            final=None
            if final_value is None
            else FinalCandidateRecord.from_dict(final_value),
            status=str(value.get("status", "unknown")),
            error=None if value.get("error") is None else str(value["error"]),
            resume_contract=None
            if value.get("resume_contract") is None
            else str(value["resume_contract"]),
            protocol=str(value.get("protocol", EXACT_QP_SINGLE_START_PROTOCOL)),
        )

    def __post_init__(self):
        if self.stages:
            raise ValueError("the surrogate route has no continuation stages")


@dataclass(frozen=True)
class SurrogateRouteResult:
    starts: tuple[SurrogateStartResult, ...]
    selected: SurrogateStartResult | None
    status: str
    protocol: str = EXACT_QP_SINGLE_START_PROTOCOL

    def as_dict(self) -> dict[str, Any]:
        return {
            "starts": [result.as_dict() for result in self.starts],
            "selected_start": None
            if self.selected is None
            else self.selected.start_index,
            "status": self.status,
            "protocol": self.protocol,
        }


@dataclass(frozen=True)
class _ExactEvaluation:
    normalized: FloatArray
    projection: ProjectionResult
    objective: float
    engineering: FloatArray
    trust: FloatArray
    components: FloatArray
    quantities: FloatArray
    trust_values: FloatArray


@dataclass(frozen=True)
class _DerivativeFreeEvaluation:
    normalized_controls: FloatArray
    candidate: FinalCandidateRecord | None
    error: str | None
