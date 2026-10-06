"""Optimization mechanistic."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.plant.operating_point import ClarifierParameters
from surrogate_optimization.plant.model import CLARIFIER
from surrogate_optimization.plant.model import UNDERFLOW_TSS_MAX_G_M3
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from hashlib import sha256
from scipy.optimize import lsq_linear
from time import perf_counter
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Mapping
import casadi as ca
import json
import numpy as np
import numpy.typing as npt
import re

FloatArray = npt.NDArray[np.float64]
DECISION_NAMES: tuple[str, ...] = ("H", "a_3", "a_4", "a_5", "r_I", "r_R", "w")
DECISION_LOWER = np.asarray([6.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.001])
DECISION_UPPER = np.asarray([36.0, 1.0, 1.0, 1.0, 4.0, 1.25, 0.05])
DECISION_SPAN = DECISION_UPPER - DECISION_LOWER
DEFAULT_OBJECTIVE_WEIGHTS = np.asarray([0.5, 0.15, 0.2, 0.05, 0.05, 0.05])
QUALITY_WEIGHTS = np.full(4, 0.25)
CONTINUATION_SCHEDULE: tuple[tuple[float, float], ...] = (
    (1e-06, 10.0),
    (1e-07, 3.0),
    (1e-08, 1.0),
)


class MechanisticOptimizationError(RuntimeError):
    """Raised when a smooth-model contract cannot be satisfied."""


def _finite_vector(value: npt.ArrayLike, size: int, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}.")
    return result.copy()


def _finite_matrix(value: npt.ArrayLike, columns: int, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[1] != columns
        or (not np.all(np.isfinite(result)))
    ):
        raise ValueError(f"{name} must be a finite matrix with {columns} columns.")
    return result.copy()


def _maximum(values: Any) -> Any:
    return (
        ca.mmax(values) if isinstance(values, (ca.MX, ca.SX, ca.DM)) else np.max(values)
    )


def _hypot(a: Any, b: Any) -> Any:
    return ca.hypot(a, b) if isinstance(a, (ca.MX, ca.SX, ca.DM)) else np.hypot(a, b)


def smooth_maximum(a: Any, b: Any, *, epsilon: float, scale: float) -> Any:
    """Return the model square-root smooth maximum."""
    if epsilon <= 0.0 or scale <= 0.0:
        raise ValueError("epsilon and scale must be positive.")
    return 0.5 * (a + b + _hypot(a - b, epsilon * scale))


def smooth_minimum(a: Any, b: Any, *, epsilon: float, scale: float) -> Any:
    """Return the model square-root smooth minimum."""
    if epsilon <= 0.0 or scale <= 0.0:
        raise ValueError("epsilon and scale must be positive.")
    return 0.5 * (a + b - _hypot(a - b, epsilon * scale))


def smooth_positive_part(value: Any, *, epsilon: float, scale: float) -> Any:
    """Cancellation-safe evaluation of ``(v + hypot(v, eps*S))/2``.

    The negative branch is the algebraically identical rational expression
    required by the supplement.  It avoids returning a spurious exact zero for
    a large negative argument.
    """
    if epsilon <= 0.0 or scale <= 0.0:
        raise ValueError("epsilon and scale must be positive.")
    width = epsilon * scale
    root = _hypot(value, width)
    if isinstance(value, (ca.MX, ca.SX, ca.DM)):
        positive = 0.5 * (value + root)
        negative = width * width / (2.0 * (root - value))
        return ca.if_else(value >= 0.0, positive, negative, True)
    values = np.asarray(value, dtype=np.float64)
    roots = np.hypot(values, width)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 0.5 * (values[nonnegative] + roots[nonnegative])
    result[~nonnegative] = (
        width * width / (2.0 * (roots[~nonnegative] - values[~nonnegative]))
    )
    return float(result) if result.ndim == 0 else result


def smooth_division(
    numerator: Any, denominator: Any, *, epsilon: float, scale: float
) -> Any:
    """Return the declared denominator-safe quotient ``ab/(b^2+(eps*S)^2)``."""
    if epsilon <= 0.0 or scale <= 0.0:
        raise ValueError("epsilon and scale must be positive.")
    return (
        numerator * denominator / (denominator * denominator + (epsilon * scale) ** 2)
    )


def receiver_transition(solids: Any, *, threshold: float, half_width: float) -> Any:
    """C2 receiver switch with the exact quintic transition polynomial."""
    if half_width <= 0.0:
        raise ValueError("receiver half-width must be positive.")
    coordinate = (solids - threshold + half_width) / (2.0 * half_width)
    polynomial = 6.0 * coordinate**5 - 15.0 * coordinate**4 + 10.0 * coordinate**3
    if isinstance(solids, (ca.MX, ca.SX, ca.DM)):
        return ca.if_else(
            solids <= threshold - half_width,
            0.0,
            ca.if_else(solids >= threshold + half_width, 1.0, polynomial),
        )
    values = np.asarray(solids, dtype=np.float64)
    result = np.where(
        values <= threshold - half_width,
        0.0,
        np.where(values >= threshold + half_width, 1.0, polynomial),
    )
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class SmoothScales:
    """Development-only scales for every regularized kinetic/settling operation."""

    nox: float
    fermentable_acetate: float
    hydrolysis: float
    polyphosphate: float
    pha: float
    capacity: float
    settling_delta: float
    velocity: float
    flux: float

    def __post_init__(self) -> None:
        values = np.asarray(tuple(self.__dict__.values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("all smoothing scales must be finite and positive.")


@dataclass(frozen=True)
class MechanisticAssets:
    """All development-fitted quantities used by the direct smooth route."""

    clarifier: ClarifierParameters
    smoothing: SmoothScales
    state_center: FloatArray
    state_scale: FloatArray
    feed_scale: float
    balance_scale: FloatArray
    quality_scale: FloatArray
    envelope_scale: FloatArray
    engineering_scale: FloatArray
    decision_center: FloatArray
    decision_scale: FloatArray
    influent_center: FloatArray
    influent_scale: FloatArray

    @property
    def layer_count(self) -> int:
        return self.clarifier.layer_count

    @property
    def state_count(self) -> int:
        from surrogate_optimization.plant.definitions import N_COMPONENTS
        from surrogate_optimization.plant.definitions import N_STAGES

        return N_STAGES * N_COMPONENTS + self.layer_count

    @property
    def response_count(self) -> int:
        from surrogate_optimization.plant.definitions import N_COMPONENTS
        from surrogate_optimization.plant.definitions import N_STAGES

        return (N_STAGES + 3) * N_COMPONENTS + self.layer_count

    def __post_init__(self) -> None:
        from surrogate_optimization.plant.definitions import N_COMPONENTS

        if self.layer_count < 3:
            raise ValueError("layer_count must be at least three.")
        checks = (
            (self.state_center, self.state_count, "state_center", False),
            (self.state_scale, self.state_count, "state_scale", True),
            (self.balance_scale, self.state_count, "balance_scale", True),
            (self.quality_scale, 4, "quality_scale", True),
            (self.envelope_scale, 2 * (self.layer_count - 2), "envelope_scale", True),
            (self.engineering_scale, 2, "engineering_scale", True),
            (self.decision_center, 7, "decision_center", False),
            (self.decision_scale, 7, "decision_scale", True),
            (self.influent_center, N_COMPONENTS, "influent_center", False),
            (self.influent_scale, N_COMPONENTS, "influent_scale", True),
        )
        for value, size, name, positive in checks:
            array = _finite_vector(value, size, name)
            if positive and np.any(array <= 0.0):
                raise ValueError(f"{name} must be strictly positive.")
            object.__setattr__(self, name, array)
        if not np.isfinite(self.feed_scale) or self.feed_scale <= 0.0:
            raise ValueError("feed_scale must be finite and positive.")


@dataclass(frozen=True)
class MechanisticCase:
    """One influent case and its predeclared objective/engineering settings."""

    influent: FloatArray
    weights: FloatArray = field(
        default_factory=lambda: DEFAULT_OBJECTIVE_WEIGHTS.copy()
    )
    underflow_tss_limit: float = UNDERFLOW_TSS_MAX_G_M3
    case_id: str = "nominal"

    def __post_init__(self) -> None:
        from surrogate_optimization.plant.definitions import N_COMPONENTS

        influent = _finite_vector(self.influent, N_COMPONENTS, "influent")
        weights = _finite_vector(self.weights, 6, "weights")
        if np.any(influent < 0.0):
            raise ValueError("influent must be nonnegative.")
        if np.any(weights < 0.0) or not np.isclose(
            np.sum(weights), 1.0, atol=1e-12, rtol=0.0
        ):
            raise ValueError("objective weights must be nonnegative and sum to one.")
        if not np.isfinite(self.underflow_tss_limit) or self.underflow_tss_limit <= 0.0:
            raise ValueError("underflow_tss_limit must be positive and finite.")
        if not self.case_id:
            raise ValueError("case_id must not be empty.")
        object.__setattr__(self, "influent", influent)
        object.__setattr__(self, "weights", weights)

    def parameter_vector(self) -> FloatArray:
        return np.concatenate((self.influent, self.weights, [self.underflow_tss_limit]))


@dataclass(frozen=True)
class SolverSettings:
    """IPOPT and independent direct-route acceptance settings."""

    maximum_iterations: int = 2500
    tolerance: float = 1e-08
    constraint_tolerance: float = 1e-08
    dual_tolerance: float = 1e-06
    complementarity_tolerance: float = 1e-06
    active_tolerance: float = 1e-07
    equality_acceptance: float = 1e-08
    inequality_acceptance: float = 1e-06
    stationarity_acceptance: float = 1e-06
    maximum_wall_time: float | None = None

    def solver_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "print_time": False,
            "error_on_fail": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.linear_solver": "mumps",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.hessian_approximation": "exact",
            "ipopt.bound_relax_factor": 0.0,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.warm_start_bound_push": 1e-09,
            "ipopt.warm_start_bound_frac": 1e-09,
            "ipopt.warm_start_slack_bound_push": 1e-09,
            "ipopt.warm_start_slack_bound_frac": 1e-09,
            "ipopt.warm_start_mult_bound_push": 1e-09,
            "ipopt.max_iter": int(self.maximum_iterations),
            "ipopt.tol": float(self.tolerance),
            "ipopt.constr_viol_tol": float(self.constraint_tolerance),
            "ipopt.dual_inf_tol": float(self.dual_tolerance),
            "ipopt.compl_inf_tol": float(self.complementarity_tolerance),
        }
        if self.maximum_wall_time is not None:
            if not np.isfinite(self.maximum_wall_time) or self.maximum_wall_time <= 0.0:
                raise ValueError("maximum_wall_time must be positive when supplied.")
            options["ipopt.max_wall_time"] = float(self.maximum_wall_time)
        return options


def _rms_scale(values: FloatArray, unit: float = 1.0) -> float:
    return max(float(unit), float(np.sqrt(np.mean(np.square(values)))))


def _fit_coordinate_center_scale(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    center = np.mean(values, axis=0)
    scale = np.sqrt(np.mean(np.square(values - center), axis=0))
    reference = np.maximum(1.0, np.max(np.abs(values), axis=0))
    if np.any(scale <= 1e-12 * reference):
        raise MechanisticOptimizationError(
            "a development input coordinate fails the nonzero-variance rule."
        )
    return (center, scale)


def extract_reduced_states(
    mechanistic_responses: npt.ArrayLike, layer_count: int
) -> FloatArray:
    """Extract ``(c_1,...,c_5,s_1,...,s_L)`` from complete responses."""
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES

    response_count = (N_STAGES + 3) * N_COMPONENTS + layer_count
    matrix = _finite_matrix(
        mechanistic_responses, response_count, "mechanistic_responses"
    )
    layer_start = (N_STAGES + 3) * N_COMPONENTS
    return np.concatenate(
        (
            matrix[:, N_COMPONENTS : (N_STAGES + 1) * N_COMPONENTS],
            matrix[:, layer_start:],
        ),
        axis=1,
    )


def _smooth_rates(state: Any, scales: SmoothScales, epsilon: float) -> Any:
    """Twenty-component ASM2d-TSN rate vector with exactly the guards."""
    from surrogate_optimization.plant.definitions import COMPONENT_INDEX
    from surrogate_optimization.plant.definitions import N_PROCESSES
    from surrogate_optimization.plant.definitions import PARAMETERS

    p, ix = (PARAMETERS, COMPONENT_INDEX)
    c = state
    so, sf, sa = (c[ix["S_O"]], c[ix["S_F"]], c[ix["S_A"]])
    snh4, sno2, sno3 = (c[ix["S_NH4"]], c[ix["S_NO2"]], c[ix["S_NO3"]])
    spo4, salk = (c[ix["S_PO4"]], c[ix["S_ALK"]])
    xs, xh, xpao = (c[ix["X_S"]], c[ix["X_H"]], c[ix["X_PAO"]])
    xpp, xpha = (c[ix["X_PP"]], c[ix["X_PHA"]])
    xaob, xnob = (c[ix["X_AOB"]], c[ix["X_NOB"]])
    xmep, xmeoh = (c[ix["X_MeP"]], c[ix["X_MeOH"]])
    snox, carbon = (sno2 + sno3, sf + sa)
    alpha2 = smooth_division(sno2, snox, epsilon=epsilon, scale=scales.nox)
    alpha3 = smooth_division(sno3, snox, epsilon=epsilon, scale=scales.nox)
    alpha_f = smooth_division(
        sf, carbon, epsilon=epsilon, scale=scales.fermentable_acetate
    )
    alpha_a = smooth_division(
        sa, carbon, epsilon=epsilon, scale=scales.fermentable_acetate
    )
    theta_x = smooth_division(
        xs * xh, p["K_X"] * xh + xs, epsilon=epsilon, scale=scales.hydrolysis
    )
    psi_pp = smooth_division(
        xpp * xpao, p["K_PP"] * xpao + xpp, epsilon=epsilon, scale=scales.polyphosphate
    )
    psi_pha = smooth_division(
        xpha * xpao, p["K_PHA"] * xpao + xpha, epsilon=epsilon, scale=scales.pha
    )
    capacity_delta = smooth_positive_part(
        p["K_max"] * xpao - xpp, epsilon=epsilon, scale=scales.capacity
    )
    capacity = capacity_delta / (p["K_IPP"] * xpao + capacity_delta)

    def monod(value: Any, half: float) -> Any:
        return value / (half + value)

    def inhibit(value: Any, half: float) -> Any:
        return half / (half + value)

    lh = (
        monod(snh4, p["K_NH4_H"])
        * monod(spo4, p["K_PO4_H"])
        * monod(salk, p["K_ALK_H"])
    )
    lp = (
        monod(snh4, p["K_NH4_PAO"])
        * monod(spo4, p["K_PO4_PAO"])
        * monod(salk, p["K_ALK_PAO"])
    )
    ln = monod(spo4, p["K_PO4_nit"]) * monod(salk, p["K_ALK_nit"])
    mo_hyd, io_hyd = (monod(so, p["K_O_hyd"]), inhibit(so, p["K_O_hyd"]))
    mo_h, io_h = (monod(so, p["K_O_H"]), inhibit(so, p["K_O_H"]))
    mo_p, io_p = (monod(so, p["K_O_PAO"]), inhibit(so, p["K_O_PAO"]))
    alk_p = monod(salk, p["K_ALK_PAO"])
    common_pp = p["q_PP"] * monod(spo4, p["K_PS"]) * alk_p * capacity
    rates = [
        p["K_H"] * mo_hyd * theta_x,
        p["eta_hyd_NO2"]
        * p["K_H"]
        * io_hyd
        * monod(sno2, p["K_NO2_hyd"])
        * alpha2
        * theta_x,
        p["eta_hyd_NO3"]
        * p["K_H"]
        * io_hyd
        * monod(sno3, p["K_NO3_hyd"])
        * alpha3
        * theta_x,
        p["eta_hyd_fe"] * p["K_H"] * io_hyd * inhibit(snox, p["K_NOx_hyd"]) * theta_x,
        p["mu_H"] * mo_h * monod(sf, p["K_F"]) * alpha_f * lh * xh,
        p["mu_H"] * mo_h * monod(sa, p["K_A"]) * alpha_a * lh * xh,
        p["mu_H"]
        * io_h
        * monod(sf, p["K_F"])
        * alpha_f
        * lh
        * p["eta_H_NO3"]
        * monod(sno3, p["K_NO3_H"])
        * alpha3
        * xh,
        p["mu_H"]
        * io_h
        * monod(sf, p["K_F"])
        * alpha_f
        * lh
        * p["eta_H_NO2"]
        * monod(sno2, p["K_NO2_H"])
        * alpha2
        * xh,
        p["mu_H"]
        * io_h
        * monod(sa, p["K_A"])
        * alpha_a
        * lh
        * p["eta_H_NO3"]
        * monod(sno3, p["K_NO3_H"])
        * alpha3
        * xh,
        p["mu_H"]
        * io_h
        * monod(sa, p["K_A"])
        * alpha_a
        * lh
        * p["eta_H_NO2"]
        * monod(sno2, p["K_NO2_H"])
        * alpha2
        * xh,
        p["q_fe"]
        * io_h
        * inhibit(snox, p["K_NOx_H"])
        * monod(sf, p["K_fe"])
        * monod(salk, p["K_ALK_H"])
        * xh,
        p["b_H"] * xh,
        p["q_PHA"]
        * monod(sa, p["K_A"])
        * io_p
        * inhibit(snox, p["K_NOx_PAO"])
        * alk_p
        * psi_pp,
        common_pp * mo_p * psi_pha,
        common_pp
        * io_p
        * p["eta_PAO_NO3"]
        * monod(sno3, p["K_NO3_PAO"])
        * alpha3
        * psi_pha,
        common_pp
        * io_p
        * p["eta_PAO_NO2"]
        * monod(sno2, p["K_NO2_PAO"])
        * alpha2
        * psi_pha,
        p["mu_PAO"] * mo_p * lp * psi_pha,
        p["mu_PAO"]
        * io_p
        * lp
        * p["eta_PAO_NO3"]
        * monod(sno3, p["K_NO3_PAO"])
        * alpha3
        * psi_pha,
        p["mu_PAO"]
        * io_p
        * lp
        * p["eta_PAO_NO2"]
        * monod(sno2, p["K_NO2_PAO"])
        * alpha2
        * psi_pha,
        p["b_PAO"] * alk_p * xpao,
        p["b_PP"] * alk_p * xpp,
        p["b_PHA"] * alk_p * xpha,
        p["mu_AOB"] * monod(so, p["K_O_AOB"]) * monod(snh4, p["K_NH4_AOB"]) * ln * xaob,
        p["mu_NOB"]
        * monod(so, p["K_O_NOB"])
        * monod(sno2, p["K_NO2_NOB"])
        * ln
        * monod(snh4, p["K_NH4_NOB"])
        * xnob,
        p["b_AOB"] * xaob,
        p["b_NOB"] * xnob,
        p["k_PRE"] * spo4 * xmeoh * monod(salk, p["K_ALK_PRE"]),
        p["k_RED"] * p["i_PMeP"] * monod(salk, p["K_ALK_chem"]) * xmep,
    ]
    if isinstance(state, (ca.MX, ca.SX, ca.DM)):
        return ca.vertcat(*rates)
    result = np.asarray(rates, dtype=np.float64)
    if result.shape != (N_PROCESSES,) or not np.all(np.isfinite(result)):
        raise FloatingPointError("smooth process rates are non-finite.")
    return result


def smooth_process_rates(
    state: npt.ArrayLike, scales: SmoothScales, *, epsilon: float = 1e-08
) -> FloatArray:
    """Evaluate the smooth kinetic rates for one nonnegative state."""
    from surrogate_optimization.plant.definitions import N_COMPONENTS

    values = _finite_vector(state, N_COMPONENTS, "state")
    if np.any(values < 0.0):
        raise ValueError("state must be nonnegative.")
    return np.asarray(_smooth_rates(values, scales, float(epsilon)), dtype=float)


def _smooth_clarifier_fluxes(
    layers: Any,
    feed_tss: Any,
    controls: Any,
    scales: SmoothScales,
    clarifier: ClarifierParameters,
    epsilon: float,
    receiver_half_width: float,
) -> Any:
    gravity: list[Any] = []
    floor = clarifier.nonsettleable_fraction * feed_tss
    for layer in range(clarifier.layer_count):
        delta = smooth_positive_part(
            layers[layer] - floor, epsilon=epsilon, scale=scales.settling_delta
        )
        raw_velocity = clarifier.theoretical_settling_velocity * (
            ca.exp(-clarifier.hindered_coefficient * delta)
            - ca.exp(-clarifier.low_concentration_coefficient * delta)
            if isinstance(delta, (ca.MX, ca.SX, ca.DM))
            else np.exp(-clarifier.hindered_coefficient * delta)
            - np.exp(-clarifier.low_concentration_coefficient * delta)
        )
        velocity = smooth_maximum(
            0.0,
            smooth_minimum(
                clarifier.maximum_settling_velocity,
                raw_velocity,
                epsilon=epsilon,
                scale=scales.velocity,
            ),
            epsilon=epsilon,
            scale=scales.velocity,
        )
        gravity.append(layers[layer] * velocity)
    r_r, waste = (controls[5], controls[6])
    q_u, q_e = (r_r + waste, 1.0 - waste)
    v_e = clarifier.fresh_flow * q_e / clarifier.area
    v_u = clarifier.fresh_flow * q_u / clarifier.area
    flux: list[Any] = [-v_e * layers[0]]
    for upper in range(clarifier.layer_count - 1):
        lower = upper + 1
        weight = receiver_transition(
            layers[lower],
            threshold=clarifier.flux_threshold,
            half_width=receiver_half_width,
        )
        minimum = smooth_minimum(
            gravity[upper], gravity[lower], epsilon=epsilon, scale=scales.flux
        )
        limited = (1.0 - weight) * gravity[upper] + weight * smooth_positive_part(
            minimum, epsilon=epsilon, scale=scales.flux
        )
        flux.append(
            -v_e * layers[lower] + limited
            if upper < clarifier.feed_layer
            else v_u * layers[upper] + limited
        )
    flux.append(v_u * layers[-1])
    if isinstance(layers, (ca.MX, ca.SX, ca.DM)):
        return ca.vertcat(*flux)
    return np.asarray(flux, dtype=np.float64)


def smooth_clarifier_fluxes(
    layers: npt.ArrayLike,
    feed_tss: float,
    controls: npt.ArrayLike,
    scales: SmoothScales,
    clarifier: ClarifierParameters = CLARIFIER,
    *,
    epsilon: float = 1e-08,
    receiver_half_width: float = 1.0,
) -> FloatArray:
    values = _finite_vector(layers, clarifier.layer_count, "layers")
    controls = _finite_vector(controls, 7, "controls")
    if np.any(values < 0.0) or feed_tss < 0.0:
        raise ValueError("layers and feed_tss must be nonnegative.")
    return np.asarray(
        _smooth_clarifier_fluxes(
            values,
            float(feed_tss),
            controls,
            scales,
            clarifier,
            float(epsilon),
            float(receiver_half_width),
        ),
        dtype=float,
    )


def _reconstruct_response(
    controls: Any, influent: Any, state: Any, feed_tss: Any, layer_count: int
) -> Any:
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES
    from surrogate_optimization.plant.definitions import PARTICULATE

    reactors = [
        state[i * N_COMPONENTS : (i + 1) * N_COMPONENTS] for i in range(N_STAGES)
    ]
    layers = state[N_STAGES * N_COMPONENTS :]
    final = reactors[-1]
    c_e: list[Any] = []
    c_u: list[Any] = []
    particulate = set(map(int, PARTICULATE.tolist()))
    for component in range(N_COMPONENTS):
        if component in particulate:
            fraction = final[component] / feed_tss
            c_e.append(layers[0] * fraction)
            c_u.append(layers[-1] * fraction)
        else:
            c_e.append(final[component])
            c_u.append(final[component])
    if isinstance(state, (ca.MX, ca.SX, ca.DM)):
        c_e_v, c_u_v = (ca.vertcat(*c_e), ca.vertcat(*c_u))
    else:
        c_e_v, c_u_v = (np.asarray(c_e), np.asarray(c_u))
    r_i, r_r, waste = (controls[4], controls[5], controls[6])
    q_p, q_u, q_e = (1.0 + r_i + r_r, r_r + waste, 1.0 - waste)
    mixer = (influent + r_i * final + r_r * c_u_v) / q_p
    blocks = (mixer, *reactors, q_e * c_e_v, q_u * c_u_v, layers)
    return (
        ca.vertcat(*blocks)
        if isinstance(state, (ca.MX, ca.SX, ca.DM))
        else np.concatenate(blocks)
    )


def _smooth_reactor_residual(
    controls: Any, response: Any, assets: MechanisticAssets, epsilon: float
) -> Any:
    """Evaluate only the smooth reactor rows from a response prefix.

    Both the complete mechanistic response and the reduced surrogate response
    begin with ``(m,c_1,...,c_N)``.  Keeping these rows independent of the
    Clarifier layer coordinates lets the reduced surrogate retain the reactor
    trust diagnostic without reconstructing a fictitious layer profile.
    """
    from surrogate_optimization.plant.definitions import COMPONENT_INDEX
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES
    from surrogate_optimization.plant.definitions import STOICHIOMETRIC_MATRIX

    mixer = response[:N_COMPONENTS]
    reactors = [
        response[(stage + 1) * N_COMPONENTS : (stage + 2) * N_COMPONENTS]
        for stage in range(N_STAGES)
    ]
    dilution = 120.0 * (1.0 + controls[4] + controls[5]) / controls[0]
    residuals: list[Any] = []
    upstream = mixer
    for stage, reactor in enumerate(reactors):
        source = (
            ca.DM(STOICHIOMETRIC_MATRIX).T
            @ _smooth_rates(reactor, assets.smoothing, epsilon)
            if isinstance(response, (ca.MX, ca.SX, ca.DM))
            else STOICHIOMETRIC_MATRIX.T
            @ _smooth_rates(reactor, assets.smoothing, epsilon)
        )
        if stage >= 2:
            aeration = controls[stage - 1]
            oxygen = 47.0 * aeration * (8.5 - reactor[COMPONENT_INDEX["S_O"]])
            if isinstance(response, (ca.MX, ca.SX, ca.DM)):
                source = source + ca.vertcat(oxygen, ca.MX.zeros(N_COMPONENTS - 1))
            else:
                source = np.asarray(source, dtype=float).copy()
                source[COMPONENT_INDEX["S_O"]] += oxygen
        residuals.append(dilution * (upstream - reactor) + source)
        upstream = reactor
    if isinstance(response, (ca.MX, ca.SX, ca.DM)):
        return ca.vertcat(*residuals)
    return np.concatenate(tuple((np.asarray(item, dtype=float) for item in residuals)))


def _smooth_response_residual(
    controls: Any,
    influent: Any,
    state: Any,
    feed_tss: Any,
    assets: MechanisticAssets,
    epsilon: float,
    receiver_half_width: float,
) -> tuple[Any, Any]:
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES

    response = _reconstruct_response(
        controls, influent, state, feed_tss, assets.layer_count
    )
    layers = state[N_STAGES * N_COMPONENTS :]
    reactor_residual = _smooth_reactor_residual(controls, response, assets, epsilon)
    flux = _smooth_clarifier_fluxes(
        layers,
        feed_tss,
        controls,
        assets.smoothing,
        assets.clarifier,
        epsilon,
        receiver_half_width,
    )
    layer_residuals: list[Any] = []
    q_c = 1.0 + controls[5]
    for layer in range(assets.layer_count):
        value = (
            assets.clarifier.area
            * (flux[layer] - flux[layer + 1])
            / assets.clarifier.layer_volume
        )
        if layer == assets.clarifier.feed_layer:
            value += (
                assets.clarifier.fresh_flow
                * q_c
                * feed_tss
                / assets.clarifier.layer_volume
            )
        layer_residuals.append(value)
    if isinstance(state, (ca.MX, ca.SX, ca.DM)):
        return (response, ca.vertcat(reactor_residual, *layer_residuals))
    return (response, np.concatenate((reactor_residual, np.asarray(layer_residuals))))


def evaluate_smooth_response(
    controls: npt.ArrayLike,
    influent: npt.ArrayLike,
    state: npt.ArrayLike,
    feed_tss: float,
    assets: MechanisticAssets,
    *,
    epsilon: float = 1e-08,
    receiver_half_width: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Return the complete response and raw smooth balance residual."""
    from surrogate_optimization.plant.definitions import N_COMPONENTS

    controls = _finite_vector(controls, 7, "controls")
    feed = _finite_vector(influent, N_COMPONENTS, "influent")
    reduced = _finite_vector(state, assets.state_count, "state")
    if feed_tss <= 0.0 or not np.isfinite(feed_tss):
        raise ValueError("feed_tss must be positive and finite.")
    response, residual = _smooth_response_residual(
        controls,
        feed,
        reduced,
        float(feed_tss),
        assets,
        float(epsilon),
        float(receiver_half_width),
    )
    return (np.asarray(response, dtype=float), np.asarray(residual, dtype=float))


def _engineering_values(
    controls: Any, response: Any, assets: MechanisticAssets
) -> tuple[Any, Any, Any]:
    from surrogate_optimization.plant.definitions import COMPOSITE_MATRIX
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES
    from surrogate_optimization.plant.definitions import TSS_VECTOR

    layer_start = (N_STAGES + 3) * N_COMPONENTS
    reactors = [
        response[(i + 1) * N_COMPONENTS : (i + 2) * N_COMPONENTS]
        for i in range(N_STAGES)
    ]
    g_e = response[(N_STAGES + 1) * N_COMPONENTS : (N_STAGES + 2) * N_COMPONENTS]
    g_u = response[(N_STAGES + 2) * N_COMPONENTS : layer_start]
    layers = response[layer_start:]
    hrt, r_r, waste = (controls[0], controls[5], controls[6])
    q_c, q_u, q_e = (1.0 + r_r, r_r + waste, 1.0 - waste)
    tss = (
        ca.DM(TSS_VECTOR) if isinstance(response, (ca.MX, ca.SX, ca.DM)) else TSS_VECTOR
    )
    c_e, c_u = (g_e / q_e, g_u / q_u)
    feed = (
        ca.dot(tss, reactors[-1])
        if isinstance(response, (ca.MX, ca.SX, ca.DM))
        else float(tss @ reactors[-1])
    )
    underflow = (
        ca.dot(tss, c_u)
        if isinstance(response, (ca.MX, ca.SX, ca.DM))
        else float(tss @ c_u)
    )
    effluent = (
        ca.dot(tss, g_e)
        if isinstance(response, (ca.MX, ca.SX, ca.DM))
        else float(tss @ g_e)
    )
    boundary = effluent + waste * underflow
    stage_volume = assets.clarifier.fresh_flow * hrt / (24.0 * N_STAGES)
    inventory = sum(
        (
            stage_volume
            * (
                ca.dot(tss, reactor)
                if isinstance(response, (ca.MX, ca.SX, ca.DM))
                else float(tss @ reactor)
            )
            for reactor in reactors
        )
    )
    inventory += assets.clarifier.layer_volume * (
        ca.sum1(layers)
        if isinstance(response, (ca.MX, ca.SX, ca.DM))
        else float(np.sum(layers))
    )
    sor = assets.clarifier.fresh_flow * q_e / assets.clarifier.area
    slr = 0.001 * assets.clarifier.fresh_flow * q_c * feed / assets.clarifier.area
    composites = (
        ca.DM(COMPOSITE_MATRIX) @ c_e
        if isinstance(response, (ca.MX, ca.SX, ca.DM))
        else COMPOSITE_MATRIX @ c_e
    )
    if isinstance(response, (ca.MX, ca.SX, ca.DM)):
        reported = ca.vertcat(
            inventory,
            boundary,
            inventory / (assets.clarifier.fresh_flow * boundary),
            sor,
            slr,
            underflow,
            feed,
            composites,
        )
        raw_constraints = ca.vertcat(underflow - 15000.0, 1.0 - boundary)
    else:
        reported = np.concatenate(
            (
                [
                    inventory,
                    boundary,
                    inventory / (assets.clarifier.fresh_flow * boundary),
                    sor,
                    slr,
                    underflow,
                    feed,
                ],
                np.asarray(composites),
            )
        )
        raw_constraints = np.asarray([underflow - 15000.0, 1.0 - boundary])
    return (reported, raw_constraints, composites)


def engineering_quantities(
    controls: npt.ArrayLike, response: npt.ArrayLike, assets: MechanisticAssets
) -> dict[str, float]:
    controls = _finite_vector(controls, 7, "controls")
    complete = _finite_vector(response, assets.response_count, "response")
    values, _, _ = _engineering_values(controls, complete, assets)
    names = (
        "solids_inventory",
        "external_solids_loss",
        "srt_d",
        "sor_m_d",
        "slr_kg_m2_d",
        "underflow_tss_g_m3",
        "feed_tss_g_m3",
        "effluent_cod",
        "effluent_tn",
        "effluent_tp",
        "effluent_tss",
    )
    return dict(zip(names, np.asarray(values, dtype=float), strict=True))


def objective_components(
    controls: npt.ArrayLike, response: npt.ArrayLike, assets: MechanisticAssets
) -> FloatArray:
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES
    from surrogate_optimization.plant.definitions import TSS_VECTOR

    controls = _finite_vector(controls, 7, "controls")
    complete = _finite_vector(response, assets.response_count, "response")
    _, _, composites = _engineering_values(controls, complete, assets)
    q_u = controls[5] + controls[6]
    underflow = (
        complete[(N_STAGES + 2) * N_COMPONENTS : (N_STAGES + 3) * N_COMPONENTS] / q_u
    )
    underflow_tss = float(TSS_VECTOR @ underflow)
    quality = float(
        np.dot(QUALITY_WEIGHTS, np.asarray(composites) / assets.quality_scale)
    )
    return np.asarray(
        [
            quality,
            (controls[0] - DECISION_LOWER[0]) / DECISION_SPAN[0],
            controls[0] * float(np.sum(controls[1:4])) / (DECISION_UPPER[0] * 3.0),
            controls[4] / DECISION_UPPER[4],
            (controls[5] - DECISION_LOWER[5]) / DECISION_SPAN[5],
            controls[6] * underflow_tss / (DECISION_UPPER[6] * 15000.0),
        ]
    )


def _objective_symbolic(
    controls: ca.MX, response: ca.MX, weights: ca.MX, assets: MechanisticAssets
) -> tuple[ca.MX, ca.MX]:
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES
    from surrogate_optimization.plant.definitions import TSS_VECTOR

    _, _, composites = _engineering_values(controls, response, assets)
    q_u = controls[5] + controls[6]
    g_u = response[(N_STAGES + 2) * N_COMPONENTS : (N_STAGES + 3) * N_COMPONENTS]
    underflow_tss = ca.dot(ca.DM(TSS_VECTOR), g_u / q_u)
    components = ca.vertcat(
        ca.dot(ca.DM(QUALITY_WEIGHTS / assets.quality_scale), composites),
        (controls[0] - DECISION_LOWER[0]) / DECISION_SPAN[0],
        controls[0] * ca.sum1(controls[1:4]) / (DECISION_UPPER[0] * 3.0),
        controls[4] / DECISION_UPPER[4],
        (controls[5] - DECISION_LOWER[5]) / DECISION_SPAN[5],
        controls[6] * underflow_tss / (DECISION_UPPER[6] * 15000.0),
    )
    return (ca.dot(weights, components), components)


def _term_scale(square_sum: FloatArray, term_count: FloatArray) -> FloatArray:
    return np.maximum(1.0, np.sqrt(np.mean(square_sum / term_count[None, :], axis=0)))


def _casadi_name(value: str) -> str:
    """Return a deterministic identifier accepted by CasADi."""
    result = re.sub("[^A-Za-z0-9_]", "_", value)
    result = re.sub("_+", "_", result).strip("_")
    if not result or not result[0].isalpha():
        result = f"{result}"
    return result


def fit_mechanistic_assets(
    development_controls: npt.ArrayLike,
    development_influents: npt.ArrayLike,
    development_responses: npt.ArrayLike,
    *,
    clarifier: ClarifierParameters = CLARIFIER,
) -> MechanisticAssets:
    """Fit every smooth/direct scale using development rows only."""
    from surrogate_optimization.plant.definitions import COMPONENT_INDEX
    from surrogate_optimization.plant.definitions import COMPOSITE_MATRIX
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES
    from surrogate_optimization.plant.definitions import PARAMETERS
    from surrogate_optimization.plant.definitions import STOICHIOMETRIC_MATRIX
    from surrogate_optimization.plant.definitions import TSS_VECTOR
    from surrogate_optimization.plant.model import settling_velocity

    controls = _finite_matrix(development_controls, 7, "development_controls")
    influents = _finite_matrix(
        development_influents, N_COMPONENTS, "development_influents"
    )
    mechanistic_responses = _finite_matrix(
        development_responses,
        (N_STAGES + 3) * N_COMPONENTS + clarifier.layer_count,
        "development_responses",
    )
    rows = controls.shape[0]
    if rows < 2 or influents.shape[0] != rows or mechanistic_responses.shape[0] != rows:
        raise ValueError(
            "development blocks must contain the same number of at least two rows."
        )
    states = extract_reduced_states(mechanistic_responses, clarifier.layer_count)
    reactors = states[:, : N_STAGES * N_COMPONENTS].reshape(
        rows, N_STAGES, N_COMPONENTS
    )
    layers = states[:, N_STAGES * N_COMPONENTS :]
    final = reactors[:, -1]
    feed_tss = final @ TSS_VECTOR
    if np.any(feed_tss <= 0.0):
        raise MechanisticOptimizationError("development feed TSS must be positive.")
    pooled = reactors.reshape(-1, N_COMPONENTS)
    ix, p = (COMPONENT_INDEX, PARAMETERS)
    nox = pooled[:, ix["S_NO2"]] + pooled[:, ix["S_NO3"]]
    carbon = pooled[:, ix["S_F"]] + pooled[:, ix["S_A"]]
    hydrolysis = p["K_X"] * pooled[:, ix["X_H"]] + pooled[:, ix["X_S"]]
    pp = p["K_PP"] * pooled[:, ix["X_PAO"]] + pooled[:, ix["X_PP"]]
    pha = p["K_PHA"] * pooled[:, ix["X_PAO"]] + pooled[:, ix["X_PHA"]]
    capacity = p["K_max"] * pooled[:, ix["X_PAO"]] - pooled[:, ix["X_PP"]]
    settling_delta = layers - clarifier.nonsettleable_fraction * feed_tss[:, None]
    exact_gravity = np.empty_like(layers)
    for row in range(rows):
        exact_gravity[row] = layers[row] * settling_velocity(
            layers[row], feed_tss[row], clarifier
        )
    smoothing = SmoothScales(
        nox=_rms_scale(nox),
        fermentable_acetate=_rms_scale(carbon),
        hydrolysis=_rms_scale(hydrolysis),
        polyphosphate=_rms_scale(pp),
        pha=_rms_scale(pha),
        capacity=_rms_scale(capacity),
        settling_delta=_rms_scale(settling_delta),
        velocity=max(1.0, clarifier.maximum_settling_velocity),
        flux=_rms_scale(exact_gravity),
    )
    state_center = np.median(states, axis=0)
    state_scale = np.maximum(1.0, np.median(np.abs(states - state_center), axis=0))
    feed_scale = _rms_scale(feed_tss)
    decision_center, decision_scale = _fit_coordinate_center_scale(controls)
    influent_center, influent_scale = _fit_coordinate_center_scale(influents)
    shell = MechanisticAssets(
        clarifier=clarifier,
        smoothing=smoothing,
        state_center=state_center,
        state_scale=state_scale,
        feed_scale=feed_scale,
        balance_scale=np.ones(states.shape[1]),
        quality_scale=np.ones(4),
        envelope_scale=np.ones(2 * (clarifier.layer_count - 2)),
        engineering_scale=np.ones(2),
        decision_center=decision_center,
        decision_scale=decision_scale,
        influent_center=influent_center,
        influent_scale=influent_scale,
    )
    balance_square = np.zeros((rows, shell.state_count))
    balance_terms = np.zeros(shell.state_count)
    envelope_square = np.zeros((rows, 2 * (clarifier.layer_count - 2)))
    engineering_square = np.zeros((rows, 2))
    for row in range(rows):
        row_controls = controls[row]
        response, _ = evaluate_smooth_response(
            row_controls, influents[row], states[row], feed_tss[row], shell
        )
        mixer = response[:N_COMPONENTS]
        dilution = 120.0 * (1.0 + row_controls[4] + row_controls[5]) / row_controls[0]
        upstream = mixer
        offset = 0
        for stage in range(N_STAGES):
            current = reactors[row, stage]
            hydraulic = dilution * (upstream - current)
            reaction = STOICHIOMETRIC_MATRIX.T @ smooth_process_rates(
                current, smoothing
            )
            balance_square[row, offset : offset + N_COMPONENTS] = np.square(
                hydraulic
            ) + np.square(reaction)
            count = np.full(N_COMPONENTS, 2.0)
            if stage >= 2:
                oxygen = (
                    47.0
                    * row_controls[stage - 1]
                    * (8.5 - current[COMPONENT_INDEX["S_O"]])
                )
                balance_square[row, offset + COMPONENT_INDEX["S_O"]] += oxygen * oxygen
                count[COMPONENT_INDEX["S_O"]] += 1.0
            balance_terms[offset : offset + N_COMPONENTS] = count
            upstream = current
            offset += N_COMPONENTS
        flux = smooth_clarifier_fluxes(
            layers[row], feed_tss[row], row_controls, smoothing, clarifier
        )
        factor = clarifier.area / clarifier.layer_volume
        for layer in range(clarifier.layer_count):
            balance_square[row, offset + layer] = (factor * flux[layer]) ** 2 + (
                factor * flux[layer + 1]
            ) ** 2
            balance_terms[offset + layer] = 2.0
            if layer == clarifier.feed_layer:
                feed_term = (
                    clarifier.fresh_flow
                    * (1.0 + row_controls[5])
                    * feed_tss[row]
                    / clarifier.layer_volume
                )
                balance_square[row, offset + layer] += feed_term * feed_term
                balance_terms[offset + layer] += 1.0
        column = 0
        for layer in range(1, clarifier.layer_count - 1):
            envelope_square[row, column] = (
                layers[row, 0] ** 2 + layers[row, layer] ** 2
            ) / 2.0
            column += 1
        for layer in range(1, clarifier.layer_count - 1):
            envelope_square[row, column] = (
                layers[row, layer] ** 2 + layers[row, -1] ** 2
            ) / 2.0
            column += 1
        reported, _, _ = _engineering_values(row_controls, response, shell)
        _, boundary, _, _, _, underflow = np.asarray(reported)[:6]
        engineering_square[row] = (
            (underflow**2 + 15000.0**2) / 2.0,
            (1.0 + boundary**2) / 2.0,
        )
    balance_scale = _term_scale(balance_square, balance_terms)
    envelope_scale = np.maximum(1.0, np.sqrt(np.mean(envelope_square, axis=0)))
    engineering_scale = np.maximum(1.0, np.sqrt(np.mean(engineering_square, axis=0)))
    q_e = 1.0 - controls[:, 6]
    c_e = (
        mechanistic_responses[
            :, (N_STAGES + 1) * N_COMPONENTS : (N_STAGES + 2) * N_COMPONENTS
        ]
        / q_e[:, None]
    )
    composites = c_e @ COMPOSITE_MATRIX.T
    quality_scale = np.std(composites, axis=0, ddof=0)
    reference = np.maximum(1.0, np.max(np.abs(composites), axis=0))
    if np.any(quality_scale <= 1e-12 * reference):
        raise MechanisticOptimizationError(
            "a development quality coordinate fails the nonzero-variance rule."
        )
    return MechanisticAssets(
        clarifier=clarifier,
        smoothing=smoothing,
        state_center=state_center,
        state_scale=state_scale,
        feed_scale=feed_scale,
        balance_scale=balance_scale,
        quality_scale=quality_scale,
        envelope_scale=envelope_scale,
        engineering_scale=engineering_scale,
        decision_center=decision_center,
        decision_scale=decision_scale,
        influent_center=influent_center,
        influent_scale=influent_scale,
    )


@dataclass(frozen=True)
class BranchClassification:
    receiver: tuple[str, ...]
    storage_capacity: tuple[str, ...]
    settling_floor: tuple[str, ...]
    settling_cap: tuple[str, ...]
    flux_minimum: tuple[str, ...]
    ambiguous: bool
    minimum_normalized_margin: float


def branches_match(left: BranchClassification, right: BranchClassification) -> bool:
    """Compare categorical branches without comparing their numeric margins."""
    return bool(
        left.receiver == right.receiver
        and left.storage_capacity == right.storage_capacity
        and (left.settling_floor == right.settling_floor)
        and (left.settling_cap == right.settling_cap)
        and (left.flux_minimum == right.flux_minimum)
    )


def classify_branches(
    state: npt.ArrayLike, assets: MechanisticAssets
) -> BranchClassification:
    """Classify every nonsmooth reference branch and its declared margin."""
    from surrogate_optimization.plant.definitions import COMPONENT_INDEX
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES
    from surrogate_optimization.plant.definitions import PARAMETERS
    from surrogate_optimization.plant.definitions import TSS_VECTOR

    values = _finite_vector(state, assets.state_count, "state")
    reactors = values[: N_STAGES * N_COMPONENTS].reshape(N_STAGES, N_COMPONENTS)
    layers = values[N_STAGES * N_COMPONENTS :]
    feed = float(TSS_VECTOR @ reactors[-1])
    p, ix, c = (PARAMETERS, COMPONENT_INDEX, assets.clarifier)
    receiver: list[str] = []
    receiver_margins: list[float] = []
    for lower in range(1, c.layer_count):
        receiver.append("limited" if layers[lower] > c.flux_threshold else "unlimited")
        receiver_margins.append(abs(layers[lower] - c.flux_threshold) / 1.0)
    storage: list[str] = []
    storage_margins: list[float] = []
    for reactor in reactors:
        delta = p["K_max"] * reactor[ix["X_PAO"]] - reactor[ix["X_PP"]]
        storage.append("positive" if delta > 0.0 else "zero")
        storage_margins.append(abs(delta) / (1e-06 * assets.smoothing.capacity))
    gravity: list[float] = []
    floors: list[str] = []
    caps: list[str] = []
    settling_margins: list[float] = []
    for layer in layers:
        delta = max(layer - c.nonsettleable_fraction * feed, 0.0)
        raw = c.theoretical_settling_velocity * (
            np.exp(-c.hindered_coefficient * delta)
            - np.exp(-c.low_concentration_coefficient * delta)
        )
        floors.append("floor" if raw <= 0.0 else "above_floor")
        caps.append("cap" if raw >= c.maximum_settling_velocity else "below_cap")
        settling_margins.extend(
            (
                abs(layer - c.nonsettleable_fraction * feed)
                / (1e-06 * assets.smoothing.settling_delta),
                abs(raw) / (1e-06 * assets.smoothing.velocity),
                abs(raw - c.maximum_settling_velocity)
                / (1e-06 * assets.smoothing.velocity),
            )
        )
        gravity.append(layer * min(c.maximum_settling_velocity, max(0.0, raw)))
    flux_branch: list[str] = []
    flux_margins: list[float] = []
    for upper in range(c.layer_count - 1):
        flux_branch.append("upper" if gravity[upper] <= gravity[upper + 1] else "lower")
        flux_margins.append(
            abs(gravity[upper] - gravity[upper + 1]) / (1e-06 * assets.smoothing.flux)
        )
    margins = np.asarray(
        receiver_margins + storage_margins + settling_margins + flux_margins
    )
    minimum = float(np.min(margins, initial=np.inf))
    return BranchClassification(
        receiver=tuple(receiver),
        storage_capacity=tuple(storage),
        settling_floor=tuple(floors),
        settling_cap=tuple(caps),
        flux_minimum=tuple(flux_branch),
        ambiguous=bool(minimum <= 1.0),
        minimum_normalized_margin=minimum,
    )


@dataclass
class MechanisticNLP:
    """Compiled direct smooth NLP for one continuation pair."""

    assets: MechanisticAssets
    epsilon: float
    receiver_half_width: float
    settings: SolverSettings
    variable_count: int
    equality_count: int
    inequality_count: int
    lower_bounds: FloatArray
    upper_bounds: FloatArray
    solver: ca.Function | None
    objective_function: ca.Function
    equality_function: ca.Function
    inequality_function: ca.Function
    gradient_function: ca.Function
    equality_jacobian_function: ca.Function
    inequality_jacobian_function: ca.Function
    response_function: ca.Function
    residual_function: ca.Function
    engineering_function: ca.Function
    components_function: ca.Function

    @property
    def constraint_lower_bounds(self) -> FloatArray:
        return np.concatenate(
            (np.zeros(self.equality_count), np.full(self.inequality_count, -np.inf))
        )

    @property
    def constraint_upper_bounds(self) -> FloatArray:
        return np.zeros(self.equality_count + self.inequality_count)


def build_mechanistic_nlp(
    assets: MechanisticAssets,
    *,
    epsilon: float = 1e-08,
    receiver_half_width: float = 1.0,
    settings: SolverSettings | None = None,
    compile_solver: bool = True,
    name: str | None = None,
) -> MechanisticNLP:
    """Build the seven-control, ``NF+L+1``-state direct mechanistic NLP."""
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES
    from surrogate_optimization.plant.definitions import TSS_VECTOR

    if epsilon <= 0.0 or receiver_half_width <= 0.0:
        raise ValueError("continuation widths must be positive.")
    settings = settings or SolverSettings()
    state_count = assets.state_count
    variable_count = 7 + state_count + 1
    variable = ca.MX.sym("mechanistic_v", variable_count)
    parameter = ca.MX.sym("mechanistic_case", N_COMPONENTS + 7)
    normalized_controls = variable[:7]
    scaled_state = variable[7 : 7 + state_count]
    scaled_feed = variable[-1]
    controls = ca.DM(DECISION_LOWER) + ca.DM(DECISION_SPAN) * normalized_controls
    state = ca.DM(assets.state_center) + ca.DM(assets.state_scale) * scaled_state
    feed_tss = assets.feed_scale * scaled_feed
    influent, weights = (
        parameter[:N_COMPONENTS],
        parameter[N_COMPONENTS : N_COMPONENTS + 6],
    )
    underflow_limit = parameter[-1]
    response, residual = _smooth_response_residual(
        controls, influent, state, feed_tss, assets, epsilon, receiver_half_width
    )
    equality = ca.vertcat(
        residual / ca.DM(assets.balance_scale),
        (
            feed_tss
            - ca.dot(
                ca.DM(TSS_VECTOR),
                state[(N_STAGES - 1) * N_COMPONENTS : N_STAGES * N_COMPONENTS],
            )
        )
        / assets.feed_scale,
    )
    layers = state[N_STAGES * N_COMPONENTS :]
    envelope_rows: list[ca.MX] = []
    for layer in range(1, assets.layer_count - 1):
        envelope_rows.append(layers[0] - layers[layer])
    for layer in range(1, assets.layer_count - 1):
        envelope_rows.append(layers[layer] - layers[-1])
    envelope = ca.vertcat(*envelope_rows) / ca.DM(assets.envelope_scale)
    reported, raw_engineering, _ = _engineering_values(controls, response, assets)
    raw_engineering[0] = reported[5] - underflow_limit
    engineering = raw_engineering / ca.DM(assets.engineering_scale)
    inequality = ca.vertcat(envelope, engineering)
    objective, components = _objective_symbolic(controls, response, weights, assets)
    lower = np.concatenate(
        (
            np.zeros(7),
            -assets.state_center / assets.state_scale,
            [1.0 / assets.feed_scale],
        )
    )
    upper = np.concatenate((np.ones(7), np.full(state_count + 1, np.inf)))
    constraint = ca.vertcat(equality, inequality)
    solver_name = _casadi_name(
        name
        or f"mechanistic_L{assets.layer_count}_{epsilon:.0e}_{receiver_half_width:g}"
    )
    solver = None
    if compile_solver:
        solver = ca.nlpsol(
            solver_name,
            "ipopt",
            {"x": variable, "p": parameter, "f": objective, "g": constraint},
            settings.solver_options(),
        )
    return MechanisticNLP(
        assets=assets,
        epsilon=float(epsilon),
        receiver_half_width=float(receiver_half_width),
        settings=settings,
        variable_count=variable_count,
        equality_count=int(equality.numel()),
        inequality_count=int(inequality.numel()),
        lower_bounds=lower,
        upper_bounds=upper,
        solver=solver,
        objective_function=ca.Function(
            f"{solver_name}_objective", [variable, parameter], [objective]
        ),
        equality_function=ca.Function(
            f"{solver_name}_equality", [variable, parameter], [equality]
        ),
        inequality_function=ca.Function(
            f"{solver_name}_inequality", [variable, parameter], [inequality]
        ),
        gradient_function=ca.Function(
            f"{solver_name}_gradient",
            [variable, parameter],
            [ca.gradient(objective, variable)],
        ),
        equality_jacobian_function=ca.Function(
            f"{solver_name}_eq_jac",
            [variable, parameter],
            [ca.jacobian(equality, variable)],
        ),
        inequality_jacobian_function=ca.Function(
            f"{solver_name}_ineq_jac",
            [variable, parameter],
            [ca.jacobian(inequality, variable)],
        ),
        response_function=ca.Function(
            f"{solver_name}_response", [variable, parameter], [response]
        ),
        residual_function=ca.Function(
            f"{solver_name}_residual", [variable, parameter], [residual]
        ),
        engineering_function=ca.Function(
            f"{solver_name}_engineering", [variable, parameter], [reported]
        ),
        components_function=ca.Function(
            f"{solver_name}_components", [variable, parameter], [components]
        ),
    )


def _flat(value: Any) -> FloatArray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def unpack_primal(
    problem: MechanisticNLP, primal: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, float]:
    point = _finite_vector(primal, problem.variable_count, "primal")
    assets = problem.assets
    controls = DECISION_LOWER + DECISION_SPAN * point[:7]
    state = assets.state_center + assets.state_scale * point[7 : 7 + assets.state_count]
    feed_tss = assets.feed_scale * point[-1]
    return (controls, state, float(feed_tss))


def evaluate_direct(
    problem: MechanisticNLP, primal: npt.ArrayLike, case: MechanisticCase
) -> dict[str, Any]:
    point = _finite_vector(primal, problem.variable_count, "primal")
    parameter = case.parameter_vector()
    controls, state, feed_tss = unpack_primal(problem, point)
    return {
        "objective": float(problem.objective_function(point, parameter)),
        "equality": _flat(problem.equality_function(point, parameter)),
        "inequality": _flat(problem.inequality_function(point, parameter)),
        "response": _flat(problem.response_function(point, parameter)),
        "raw_residual": _flat(problem.residual_function(point, parameter)),
        "engineering": _flat(problem.engineering_function(point, parameter)),
        "objective_components": _flat(problem.components_function(point, parameter)),
        "normalized_controls": point[:7].copy(),
        "controls": controls,
        "state": state,
        "feed_tss": feed_tss,
    }


@dataclass(frozen=True)
class KKTDiagnostics:
    finite: bool
    equality_residual: float
    inequality_residual: float
    bound_residual: float
    stationarity_residual: float
    dual_feasibility_residual: float
    complementarity_residual: float
    active_inequality_count: int
    feasible: bool
    stationary: bool
    equality_multipliers: FloatArray
    inequality_multipliers: FloatArray
    lower_bound_multipliers: FloatArray
    upper_bound_multipliers: FloatArray


def _independent_multipliers(
    gradient: FloatArray, equality_jacobian: FloatArray, active_jacobian: FloatArray
) -> tuple[FloatArray, FloatArray]:
    equality_count = equality_jacobian.shape[0]
    active_count = active_jacobian.shape[0]
    matrix = np.column_stack((equality_jacobian.T, active_jacobian.T))
    lower = np.concatenate((np.full(equality_count, -np.inf), np.zeros(active_count)))
    upper = np.full(equality_count + active_count, np.inf)
    if matrix.shape[1] == 0:
        return (np.empty(0), np.empty(0))
    result = lsq_linear(
        matrix,
        -gradient,
        bounds=(lower, upper),
        method="trf",
        tol=1e-12,
        lsmr_tol=1e-12,
        max_iter=10000,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise MechanisticOptimizationError(
            f"independent bounded multiplier reconstruction did not converge: {result.message}"
        )
    coefficients = np.asarray(result.x, dtype=float)
    return (coefficients[:equality_count], coefficients[equality_count:])


def independent_kkt_diagnostics(
    problem: MechanisticNLP, primal: npt.ArrayLike, case: MechanisticCase
) -> KKTDiagnostics:
    """Recompute primal and minimum-residual multiplier/KKT checks from scratch."""
    point = _finite_vector(primal, problem.variable_count, "primal")
    parameter = case.parameter_vector()
    equality = _flat(problem.equality_function(point, parameter))
    inequality = _flat(problem.inequality_function(point, parameter))
    gradient = _flat(problem.gradient_function(point, parameter))
    eq_jac = np.asarray(
        problem.equality_jacobian_function(point, parameter), dtype=float
    )
    ineq_jac = np.asarray(
        problem.inequality_jacobian_function(point, parameter), dtype=float
    )
    finite_lower = np.isfinite(problem.lower_bounds)
    finite_upper = np.isfinite(problem.upper_bounds)
    lower_rows = np.flatnonzero(
        finite_lower
        & (point - problem.lower_bounds <= problem.settings.active_tolerance)
    )
    upper_rows = np.flatnonzero(
        finite_upper
        & (problem.upper_bounds - point <= problem.settings.active_tolerance)
    )
    active_general = np.flatnonzero(inequality >= -problem.settings.active_tolerance)
    active_jacobian = np.vstack(
        (
            ineq_jac[active_general],
            -np.eye(problem.variable_count)[lower_rows],
            np.eye(problem.variable_count)[upper_rows],
        )
    )
    eta, active_mu = _independent_multipliers(gradient, eq_jac, active_jacobian)
    general_mu = np.zeros(problem.inequality_count)
    lower_mu = np.zeros(problem.variable_count)
    upper_mu = np.zeros(problem.variable_count)
    cursor = 0
    general_mu[active_general] = active_mu[cursor : cursor + active_general.size]
    cursor += active_general.size
    lower_mu[lower_rows] = active_mu[cursor : cursor + lower_rows.size]
    cursor += lower_rows.size
    upper_mu[upper_rows] = active_mu[cursor : cursor + upper_rows.size]
    stationarity = (
        gradient + eq_jac.T @ eta + ineq_jac.T @ general_mu - lower_mu + upper_mu
    )
    lower_constraint = problem.lower_bounds[finite_lower] - point[finite_lower]
    upper_constraint = point[finite_upper] - problem.upper_bounds[finite_upper]
    equality_residual = float(np.max(np.abs(equality), initial=0.0))
    inequality_residual = float(np.max(np.maximum(inequality, 0.0), initial=0.0))
    bound_residual = max(
        float(np.max(np.maximum(lower_constraint, 0.0), initial=0.0)),
        float(np.max(np.maximum(upper_constraint, 0.0), initial=0.0)),
    )
    stationarity_residual = float(np.max(np.abs(stationarity), initial=0.0))
    dual_residual = max(
        float(np.max(np.maximum(-general_mu, 0.0), initial=0.0)),
        float(np.max(np.maximum(-lower_mu, 0.0), initial=0.0)),
        float(np.max(np.maximum(-upper_mu, 0.0), initial=0.0)),
    )
    complementarity = max(
        float(np.max(np.abs(general_mu * inequality), initial=0.0)),
        float(np.max(np.abs(lower_mu[finite_lower] * lower_constraint), initial=0.0)),
        float(np.max(np.abs(upper_mu[finite_upper] * upper_constraint), initial=0.0)),
    )
    arrays: Iterable[FloatArray] = (
        point,
        equality,
        inequality,
        gradient,
        eq_jac,
        ineq_jac,
        eta,
        general_mu,
        lower_mu,
        upper_mu,
    )
    finite = bool(all((np.all(np.isfinite(value)) for value in arrays)))
    feasible = bool(
        finite
        and equality_residual <= problem.settings.equality_acceptance
        and (inequality_residual <= problem.settings.inequality_acceptance)
        and (bound_residual <= problem.settings.inequality_acceptance)
    )
    stationary = bool(
        feasible
        and stationarity_residual <= problem.settings.stationarity_acceptance
        and (dual_residual <= problem.settings.dual_tolerance)
        and (complementarity <= problem.settings.complementarity_tolerance)
    )
    return KKTDiagnostics(
        finite=finite,
        equality_residual=equality_residual,
        inequality_residual=inequality_residual,
        bound_residual=bound_residual,
        stationarity_residual=stationarity_residual,
        dual_feasibility_residual=dual_residual,
        complementarity_residual=complementarity,
        active_inequality_count=int(active_jacobian.shape[0]),
        feasible=feasible,
        stationary=stationary,
        equality_multipliers=eta,
        inequality_multipliers=general_mu,
        lower_bound_multipliers=lower_mu,
        upper_bound_multipliers=upper_mu,
    )


def ordered_normalized_starts(seed: int = 271828) -> FloatArray:
    return np.full((1, 7), 0.5)


def nearest_development_index(
    normalized_controls: npt.ArrayLike,
    influent: npt.ArrayLike,
    development_controls: npt.ArrayLike,
    development_influents: npt.ArrayLike,
    assets: MechanisticAssets,
) -> int:
    from surrogate_optimization.plant.definitions import N_COMPONENTS

    z = _finite_vector(normalized_controls, 7, "normalized_controls")
    x = _finite_vector(influent, N_COMPONENTS, "influent")
    controls = _finite_matrix(development_controls, 7, "development_controls")
    influents = _finite_matrix(
        development_influents, N_COMPONENTS, "development_influents"
    )
    if controls.shape[0] != influents.shape[0] or controls.shape[0] == 0:
        raise ValueError("development input blocks must have equal nonzero row counts.")
    query_controls = DECISION_LOWER + DECISION_SPAN * z
    query = np.concatenate(
        (
            (query_controls - assets.decision_center) / assets.decision_scale,
            (x - assets.influent_center) / assets.influent_scale,
        )
    )
    rows = np.concatenate(
        (
            (controls - assets.decision_center) / assets.decision_scale,
            (influents - assets.influent_center) / assets.influent_scale,
        ),
        axis=1,
    )
    return int(np.argmin(np.sum(np.square(rows - query[None, :]), axis=1)))


def mechanistic_initial_point(
    normalized_controls: npt.ArrayLike,
    case: MechanisticCase,
    development_controls: npt.ArrayLike,
    development_influents: npt.ArrayLike,
    development_responses: npt.ArrayLike,
    assets: MechanisticAssets,
) -> tuple[FloatArray, int]:
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES
    from surrogate_optimization.plant.definitions import TSS_VECTOR

    z = _finite_vector(normalized_controls, 7, "normalized_controls")
    mechanistic_responses = _finite_matrix(
        development_responses, assets.response_count, "development_responses"
    )
    index = nearest_development_index(
        z, case.influent, development_controls, development_influents, assets
    )
    state = extract_reduced_states(
        mechanistic_responses[index : index + 1], assets.layer_count
    )[0]
    state = np.maximum(state, 1e-08)
    feed = max(
        1.0,
        float(
            TSS_VECTOR @ state[(N_STAGES - 1) * N_COMPONENTS : N_STAGES * N_COMPONENTS]
        ),
    )
    scaled_state = (state - assets.state_center) / assets.state_scale
    return (np.concatenate((z, scaled_state, [feed / assets.feed_scale])), index)


@dataclass(frozen=True)
class ContinuationStageResult:
    epsilon: float
    receiver_half_width: float
    status: str
    solver_success: bool
    elapsed_seconds: float
    iterations: int
    primal: FloatArray
    feasible: bool
    error: str | None = None
    constraint_multipliers: FloatArray | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "primal": self.primal.tolist(),
            "constraint_multipliers": None
            if self.constraint_multipliers is None
            else self.constraint_multipliers.tolist(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContinuationStageResult":
        return cls(
            epsilon=float(value["epsilon"]),
            receiver_half_width=float(value["receiver_half_width"]),
            status=str(value["status"]),
            solver_success=bool(value["solver_success"]),
            elapsed_seconds=float(value["elapsed_seconds"]),
            iterations=int(value["iterations"]),
            primal=np.asarray(value["primal"], dtype=float),
            feasible=bool(value["feasible"]),
            error=None if value.get("error") is None else str(value["error"]),
            constraint_multipliers=None
            if value.get("constraint_multipliers") is None
            else np.asarray(value["constraint_multipliers"], dtype=float),
        )


@dataclass(frozen=True)
class MechanisticStartResult:
    start_index: int
    initial_normalized_controls: FloatArray
    resume_contract: str
    nearest_development_row: int
    stages: tuple[ContinuationStageResult, ...]
    objective: float
    normalized_controls: FloatArray
    controls: FloatArray
    state: FloatArray
    feed_tss: float
    response: FloatArray
    engineering: FloatArray
    objective_components: FloatArray
    branch: BranchClassification | None
    kkt: KKTDiagnostics | None
    feasible: bool
    stationary: bool
    status: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "initial_normalized_controls": self.initial_normalized_controls.tolist(),
            "resume_contract": self.resume_contract,
            "nearest_development_row": self.nearest_development_row,
            "stages": [stage.as_dict() for stage in self.stages],
            "objective": self.objective,
            "normalized_controls": self.normalized_controls.tolist(),
            "controls": self.controls.tolist(),
            "state": self.state.tolist(),
            "feed_tss": self.feed_tss,
            "response": self.response.tolist(),
            "engineering": self.engineering.tolist(),
            "objective_components": self.objective_components.tolist(),
            "branch": None
            if self.branch is None
            else {
                **self.branch.__dict__,
                "receiver": list(self.branch.receiver),
                "storage_capacity": list(self.branch.storage_capacity),
                "settling_floor": list(self.branch.settling_floor),
                "settling_cap": list(self.branch.settling_cap),
                "flux_minimum": list(self.branch.flux_minimum),
            },
            "kkt": None
            if self.kkt is None
            else {
                **self.kkt.__dict__,
                "equality_multipliers": self.kkt.equality_multipliers.tolist(),
                "inequality_multipliers": self.kkt.inequality_multipliers.tolist(),
                "lower_bound_multipliers": self.kkt.lower_bound_multipliers.tolist(),
                "upper_bound_multipliers": self.kkt.upper_bound_multipliers.tolist(),
            },
            "feasible": self.feasible,
            "stationary": self.stationary,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MechanisticStartResult":

        def checkpoint_float(item: Any) -> float:
            return np.nan if item is None else float(item)

        branch_value = value.get("branch")
        branch = (
            None
            if branch_value is None
            else BranchClassification(
                receiver=tuple((str(item) for item in branch_value["receiver"])),
                storage_capacity=tuple(
                    (str(item) for item in branch_value["storage_capacity"])
                ),
                settling_floor=tuple(
                    (str(item) for item in branch_value["settling_floor"])
                ),
                settling_cap=tuple(
                    (str(item) for item in branch_value["settling_cap"])
                ),
                flux_minimum=tuple(
                    (str(item) for item in branch_value["flux_minimum"])
                ),
                ambiguous=bool(branch_value["ambiguous"]),
                minimum_normalized_margin=float(
                    branch_value["minimum_normalized_margin"]
                ),
            )
        )
        kkt_value = value.get("kkt")
        kkt = (
            None
            if kkt_value is None
            else KKTDiagnostics(
                finite=bool(kkt_value["finite"]),
                equality_residual=float(kkt_value["equality_residual"]),
                inequality_residual=float(kkt_value["inequality_residual"]),
                bound_residual=float(kkt_value["bound_residual"]),
                stationarity_residual=float(kkt_value["stationarity_residual"]),
                dual_feasibility_residual=float(kkt_value["dual_feasibility_residual"]),
                complementarity_residual=float(kkt_value["complementarity_residual"]),
                active_inequality_count=int(kkt_value["active_inequality_count"]),
                feasible=bool(kkt_value["feasible"]),
                stationary=bool(kkt_value["stationary"]),
                equality_multipliers=np.asarray(
                    kkt_value["equality_multipliers"], dtype=float
                ),
                inequality_multipliers=np.asarray(
                    kkt_value["inequality_multipliers"], dtype=float
                ),
                lower_bound_multipliers=np.asarray(
                    kkt_value["lower_bound_multipliers"], dtype=float
                ),
                upper_bound_multipliers=np.asarray(
                    kkt_value["upper_bound_multipliers"], dtype=float
                ),
            )
        )
        return cls(
            start_index=int(value["start_index"]),
            initial_normalized_controls=np.asarray(
                value["initial_normalized_controls"], dtype=float
            ),
            resume_contract=str(value["resume_contract"]),
            nearest_development_row=int(value["nearest_development_row"]),
            stages=tuple(
                (ContinuationStageResult.from_dict(item) for item in value["stages"])
            ),
            objective=checkpoint_float(value["objective"]),
            normalized_controls=np.asarray(value["normalized_controls"], dtype=float),
            controls=np.asarray(value["controls"], dtype=float),
            state=np.asarray(value["state"], dtype=float),
            feed_tss=checkpoint_float(value["feed_tss"]),
            response=np.asarray(value["response"], dtype=float),
            engineering=np.asarray(value["engineering"], dtype=float),
            objective_components=np.asarray(value["objective_components"], dtype=float),
            branch=branch,
            kkt=kkt,
            feasible=bool(value["feasible"]),
            stationary=bool(value["stationary"]),
            status=str(value["status"]),
            error=None if value.get("error") is None else str(value["error"]),
        )


@dataclass(frozen=True)
class MechanisticRouteResult:
    starts: tuple[MechanisticStartResult, ...]
    selected: MechanisticStartResult | None
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "starts": [start.as_dict() for start in self.starts],
            "selected_start": None
            if self.selected is None
            else self.selected.start_index,
            "status": self.status,
        }


def mechanistic_start_resume_contract(
    assets: MechanisticAssets, case: MechanisticCase, settings: SolverSettings
) -> str:
    """Bind a cached direct start to all fixed case and numerical inputs."""
    digest = sha256()
    digest.update(json.dumps(asdict(settings), sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(asdict(assets.clarifier), sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(asdict(assets.smoothing), sort_keys=True).encode("utf-8"))
    digest.update(np.ascontiguousarray(case.parameter_vector(), dtype="<f8").tobytes())
    for value in (
        assets.state_center,
        assets.state_scale,
        assets.balance_scale,
        assets.quality_scale,
        assets.envelope_scale,
        assets.engineering_scale,
        assets.decision_center,
        assets.decision_scale,
        assets.influent_center,
        assets.influent_scale,
    ):
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    digest.update(np.asarray(assets.feed_scale, dtype="<f8").tobytes())
    return digest.hexdigest()


def _solve_mechanistic_stage(
    problem: MechanisticNLP,
    case: MechanisticCase,
    initial: FloatArray,
    dual_warm_start: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
) -> tuple[ContinuationStageResult, FloatArray, tuple[FloatArray, FloatArray] | None]:
    if problem.solver is None:
        raise MechanisticOptimizationError(
            "the NLP was built with compile_solver=False."
        )
    started = perf_counter()
    try:
        arguments: dict[str, Any] = dict(
            x0=initial,
            p=case.parameter_vector(),
            lbx=problem.lower_bounds,
            ubx=problem.upper_bounds,
            lbg=problem.constraint_lower_bounds,
            ubg=problem.constraint_upper_bounds,
        )
        if dual_warm_start is not None:
            arguments["lam_x0"] = _finite_vector(
                dual_warm_start[0], problem.variable_count, "IPOPT bound warm start"
            )
            arguments["lam_g0"] = _finite_vector(
                dual_warm_start[1],
                problem.equality_count + problem.inequality_count,
                "IPOPT constraint warm start",
            )
        solution = problem.solver(**arguments)
        elapsed = perf_counter() - started
        primal = _flat(solution["x"])
        stats = problem.solver.stats()
        evaluated = evaluate_direct(problem, primal, case)
        feasible = bool(
            np.max(np.abs(evaluated["equality"]), initial=0.0)
            <= problem.settings.equality_acceptance
            and np.max(np.maximum(evaluated["inequality"], 0.0), initial=0.0)
            <= problem.settings.inequality_acceptance
            and np.all(
                primal >= problem.lower_bounds - problem.settings.inequality_acceptance
            )
            and np.all(
                primal <= problem.upper_bounds + problem.settings.inequality_acceptance
            )
        )
        stage = ContinuationStageResult(
            epsilon=problem.epsilon,
            receiver_half_width=problem.receiver_half_width,
            status=str(stats.get("return_status", "unknown")),
            solver_success=bool(stats.get("success", False)),
            elapsed_seconds=elapsed,
            iterations=int(stats.get("iter_count", 0)),
            primal=primal,
            feasible=feasible,
            constraint_multipliers=_flat(solution["lam_g"]),
        )
        dual = (
            (_flat(solution["lam_x"]), _flat(solution["lam_g"])) if feasible else None
        )
        return (stage, primal, dual)
    except Exception as exc:
        stage = ContinuationStageResult(
            epsilon=problem.epsilon,
            receiver_half_width=problem.receiver_half_width,
            status="solver_exception",
            solver_success=False,
            elapsed_seconds=perf_counter() - started,
            iterations=0,
            primal=initial.copy(),
            feasible=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        return (stage, initial.copy(), None)


def solve_mechanistic_case(
    assets: MechanisticAssets,
    case: MechanisticCase,
    development_controls: npt.ArrayLike,
    development_influents: npt.ArrayLike,
    development_responses: npt.ArrayLike,
    *,
    settings: SolverSettings | None = None,
    starts: npt.ArrayLike | None = None,
    completed_starts: Mapping[int, MechanisticStartResult] | None = None,
    progress_callback: Callable[[MechanisticStartResult], None] | None = None,
) -> MechanisticRouteResult:
    """Solve a primary center start or a supplied recovery start through the three smoothing stages."""
    settings = settings or SolverSettings()
    resume_contract = mechanistic_start_resume_contract(assets, case, settings)
    normalized_starts = (
        ordered_normalized_starts()
        if starts is None
        else np.asarray(starts, dtype=float)
    )
    if (
        normalized_starts.ndim != 2
        or normalized_starts.shape[1:] != (7,)
        or normalized_starts.shape[0] < 1
        or (normalized_starts.shape[0] != 1)
        or (not np.all(np.isfinite(normalized_starts)))
    ):
        requirement = "exactly one" if True else "exactly one"
        raise ValueError(
            f"starts must contain {requirement} finite seven-control rows."
        )
    completed = dict(completed_starts or {})
    for index, result in completed.items():
        stage_pairs = tuple(
            ((stage.epsilon, stage.receiver_half_width) for stage in result.stages)
        )
        declared_prefix = CONTINUATION_SCHEDULE[: len(stage_pairs)]
        if (
            not isinstance(index, int)
            or not isinstance(result, MechanisticStartResult)
            or index != result.start_index
            or (not 0 <= index < len(normalized_starts))
            or (
                not np.array_equal(
                    result.initial_normalized_controls, normalized_starts[index]
                )
            )
            or (result.resume_contract != resume_contract)
            or (len(stage_pairs) > len(CONTINUATION_SCHEDULE))
            or (stage_pairs != declared_prefix)
            or (result.feasible and len(stage_pairs) != len(CONTINUATION_SCHEDULE))
        ):
            raise ValueError(
                "each completed direct start must match its declared index and controls exactly"
            )
    problems = tuple(
        (
            build_mechanistic_nlp(
                assets,
                epsilon=epsilon,
                receiver_half_width=half_width,
                settings=settings,
                name=f"mechanistic_{case.case_id}_{index}",
            )
            for index, (epsilon, half_width) in enumerate(CONTINUATION_SCHEDULE)
        )
    )
    results_by_index: dict[int, MechanisticStartResult] = dict(completed)
    for start_index, normalized in enumerate(normalized_starts):
        if start_index in results_by_index:
            continue
        try:
            primal, nearest = mechanistic_initial_point(
                normalized,
                case,
                development_controls,
                development_influents,
                development_responses,
                assets,
            )
        except Exception as exc:
            result = MechanisticStartResult(
                start_index=start_index,
                initial_normalized_controls=np.asarray(normalized).copy(),
                resume_contract=resume_contract,
                nearest_development_row=-1,
                stages=(),
                objective=np.nan,
                normalized_controls=np.asarray(normalized).copy(),
                controls=DECISION_LOWER + DECISION_SPAN * normalized,
                state=np.full(assets.state_count, np.nan),
                feed_tss=np.nan,
                response=np.full(assets.response_count, np.nan),
                engineering=np.full(11, np.nan),
                objective_components=np.full(6, np.nan),
                branch=None,
                kkt=None,
                feasible=False,
                stationary=False,
                status="initialization_exception",
                error=f"{type(exc).__name__}: {exc}",
            )
            results_by_index[start_index] = result
            if progress_callback is not None:
                progress_callback(result)
            continue
        stages: list[ContinuationStageResult] = []
        dual_warm_start: tuple[FloatArray, FloatArray] | None = None
        for problem in problems:
            stage, primal, dual_warm_start = _solve_mechanistic_stage(
                problem, case, primal, dual_warm_start
            )
            stages.append(stage)
            if not stage.feasible:
                break
        if len(stages) != len(problems) or not stages[-1].feasible:
            result = MechanisticStartResult(
                start_index=start_index,
                initial_normalized_controls=np.asarray(normalized).copy(),
                resume_contract=resume_contract,
                nearest_development_row=nearest,
                stages=tuple(stages),
                objective=np.nan,
                normalized_controls=np.asarray(normalized).copy(),
                controls=DECISION_LOWER + DECISION_SPAN * normalized,
                state=np.full(assets.state_count, np.nan),
                feed_tss=np.nan,
                response=np.full(assets.response_count, np.nan),
                engineering=np.full(11, np.nan),
                objective_components=np.full(6, np.nan),
                branch=None,
                kkt=None,
                feasible=False,
                stationary=False,
                status="continuation_failed",
            )
            results_by_index[start_index] = result
            if progress_callback is not None:
                progress_callback(result)
            continue
        final_problem = problems[-1]
        try:
            evaluated = evaluate_direct(final_problem, primal, case)
            kkt = independent_kkt_diagnostics(final_problem, primal, case)
            branch = classify_branches(evaluated["state"], assets)
        except Exception as exc:
            result = MechanisticStartResult(
                start_index=start_index,
                initial_normalized_controls=np.asarray(normalized).copy(),
                resume_contract=resume_contract,
                nearest_development_row=nearest,
                stages=tuple(stages),
                objective=np.nan,
                normalized_controls=np.asarray(normalized).copy(),
                controls=DECISION_LOWER + DECISION_SPAN * normalized,
                state=np.full(assets.state_count, np.nan),
                feed_tss=np.nan,
                response=np.full(assets.response_count, np.nan),
                engineering=np.full(11, np.nan),
                objective_components=np.full(6, np.nan),
                branch=None,
                kkt=None,
                feasible=False,
                stationary=False,
                status="final_audit_exception",
                error=f"{type(exc).__name__}: {exc}",
            )
            results_by_index[start_index] = result
            if progress_callback is not None:
                progress_callback(result)
            continue
        stationary = bool(kkt.stationary and (not branch.ambiguous))
        status = (
            "first_order_kkt_stationary_feasible"
            if stationary
            else "validated_feasible_stationarity_unresolved"
            if kkt.feasible
            else "final_feasibility_failed"
        )
        result = MechanisticStartResult(
            start_index=start_index,
            initial_normalized_controls=np.asarray(normalized).copy(),
            resume_contract=resume_contract,
            nearest_development_row=nearest,
            stages=tuple(stages),
            objective=float(evaluated["objective"]),
            normalized_controls=evaluated["normalized_controls"],
            controls=evaluated["controls"],
            state=evaluated["state"],
            feed_tss=float(evaluated["feed_tss"]),
            response=evaluated["response"],
            engineering=evaluated["engineering"],
            objective_components=evaluated["objective_components"],
            branch=branch,
            kkt=kkt,
            feasible=kkt.feasible,
            stationary=stationary,
            status=status,
        )
        results_by_index[start_index] = result
        if progress_callback is not None:
            progress_callback(result)
    results = [results_by_index[index] for index in range(len(normalized_starts))]
    stationary = [item for item in results if item.stationary]
    feasible = [item for item in results if item.feasible]
    pool = stationary or feasible
    if not pool:
        return MechanisticRouteResult(
            tuple(results), None, "no_validated_feasible_start"
        )
    minimum = min((item.objective for item in pool))
    tolerance = 1e-10 * max(1.0, abs(minimum))
    tied = [item for item in pool if item.objective <= minimum + tolerance]
    selected = min(
        tied, key=lambda item: (*item.normalized_controls.tolist(), item.start_index)
    )
    return MechanisticRouteResult(
        tuple(results),
        selected,
        "selected_stationary" if stationary else "selected_stationarity_unresolved",
    )


def engineering_feasible(
    controls: npt.ArrayLike,
    response: npt.ArrayLike,
    assets: MechanisticAssets,
    *,
    tolerance: float = 1e-06,
    nonnegativity_tolerance: float = 1e-10,
) -> bool:
    """Evaluate retained engineering gates and physical response bounds."""
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES

    controls = _finite_vector(controls, 7, "controls")
    complete = _finite_vector(response, assets.response_count, "response")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    if not np.isfinite(nonnegativity_tolerance) or nonnegativity_tolerance <= 0.0:
        raise ValueError("nonnegativity_tolerance must be finite and positive.")
    reported, raw, _ = _engineering_values(controls, complete, assets)
    layer_start = (N_STAGES + 3) * N_COMPONENTS
    layers = complete[layer_start:]
    envelope = np.concatenate((layers[0] - layers[1:-1], layers[1:-1] - layers[-1]))
    return bool(
        np.max(np.asarray(raw), initial=-np.inf) <= tolerance
        and np.max(envelope, initial=-np.inf) <= tolerance
        and (np.min(complete) >= -nonnegativity_tolerance)
        and (float(np.asarray(reported)[6]) >= 1.0 - tolerance)
    )


def _engineering_feasible(
    controls: FloatArray, response: FloatArray, assets: MechanisticAssets
) -> bool:
    """Backward-compatible internal alias for the public feasibility audit."""
    return engineering_feasible(controls, response, assets)
