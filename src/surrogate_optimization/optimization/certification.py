"""Optimization certification."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.workflow.types import AnalysisBundle
    from surrogate_optimization.optimization.types import FinalCandidateRecord
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
    from surrogate_optimization.optimization.types import SurrogateCase
    from surrogate_optimization.optimization.types import SurrogateNLP
    from surrogate_optimization.optimization.types import SurrogateNLPAssets
    from surrogate_optimization.optimization.types import SurrogateRouteResult
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from scipy import linalg
from time import perf_counter
from typing import Any
import json
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SurrogateCertificationSettings:
    """Frozen finite-resolution local-convergence audit settings.

    This audit is deliberately separate from the optimizer.  It can therefore
    be applied to an already selected endpoint without repeating the global
    data, fit, or single-start search stages.  A passing poll is evidence of
    local convergence at the declared resolution; it is not a KKT or global
    optimality certificate.
    """

    poll_radii: tuple[float, ...] = (0.001, 0.0001)
    absolute_decrease_tolerance: float = 1e-08
    relative_decrease_tolerance: float = 1e-08
    feasibility_tolerance: float = 1e-06
    direction_rank_tolerance: float = 1e-10
    maximum_evaluations: int = 10000
    acceleration_growth_factor: float = 2.0
    maximum_acceleration_probes: int = 16

    def __post_init__(self) -> None:
        if not self.poll_radii:
            raise ValueError("poll_radii must not be empty.")
        radii = np.asarray(self.poll_radii, dtype=np.float64)
        if (
            not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
            or np.any(np.diff(radii) >= 0.0)
        ):
            raise ValueError("poll_radii must be finite, positive, and decreasing.")
        for value in (
            self.absolute_decrease_tolerance,
            self.relative_decrease_tolerance,
            self.feasibility_tolerance,
            self.direction_rank_tolerance,
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "certification tolerances must be finite and positive."
                )
        if self.maximum_evaluations < 1:
            raise ValueError("maximum_evaluations must be positive.")
        if (
            not np.isfinite(self.acceleration_growth_factor)
            or self.acceleration_growth_factor <= 1.0
        ):
            raise ValueError("acceleration_growth_factor must be finite and above one.")
        if self.maximum_acceleration_probes < 1:
            raise ValueError("maximum_acceleration_probes must be positive.")


@dataclass(frozen=True)
class LocalConvergenceCertificate:
    protocol: str
    classification: str
    locally_converged: bool
    first_order_certified: bool
    stationarity_resolved: bool
    initial_objective: float
    final_objective: float
    initial_normalized_controls: FloatArray
    final_normalized_controls: FloatArray
    evaluations: int
    cold_qp_resolutions: int
    accepted_improvements: int
    elapsed_seconds: float
    termination_reason: str
    poll_levels: tuple[dict[str, Any], ...]
    lower_active_set: dict[str, Any] | None = None
    upper_kkt: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "initial_normalized_controls": self.initial_normalized_controls.tolist(),
            "final_normalized_controls": self.final_normalized_controls.tolist(),
            "poll_levels": [dict(item) for item in self.poll_levels],
        }


@dataclass(frozen=True)
class SurrogateCertificationResult:
    candidate: FinalCandidateRecord | None
    certificate: LocalConvergenceCertificate

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": None if self.candidate is None else self.candidate.as_dict(),
            "certificate": self.certificate.as_dict(),
        }


def _local_poll_directions(*, include_simplex: bool) -> FloatArray:
    """Return a deterministic poll set in seven normalized dimensions."""
    axes = np.concatenate((np.eye(7), -np.eye(7)), axis=0)
    if not include_simplex:
        return axes
    pairwise = []
    for left in range(7):
        for right in range(left + 1, 7):
            for left_sign in (-1.0, 1.0):
                for right_sign in (-1.0, 1.0):
                    direction = np.zeros(7)
                    direction[left] = left_sign / np.sqrt(2.0)
                    direction[right] = right_sign / np.sqrt(2.0)
                    pairwise.append(direction)
    simplex = linalg.helmert(8, full=False).T
    simplex /= np.linalg.norm(simplex, axis=1, keepdims=True)
    return np.concatenate((axes, np.asarray(pairwise), simplex), axis=0)


def certify_surrogate_local_convergence(
    assets: SurrogateNLPAssets,
    case: SurrogateCase,
    initial_candidate: FinalCandidateRecord,
    *,
    settings: SurrogateCertificationSettings | None = None,
    problem: SurrogateNLP | None = None,
    name: str = "surrogate_local_certificate",
) -> SurrogateCertificationResult:
    """Polish and audit one selected exact-QP surrogate endpoint.

    A stable lower active set followed by a passing upper KKT audit is the
    strong first-order certificate.  Otherwise a deterministic feasible poll
    is completed at each declared normalized-control radius.  The latter is a
    finite-resolution local-convergence certificate only: it deliberately
    leaves classical stationarity unresolved at a nonsmooth or degenerate
    endpoint.
    """
    from surrogate_optimization.optimization.active_set import ActiveSetDerivativeError
    from surrogate_optimization.optimization.active_set import (
        ActiveSetRefinementSettings,
    )
    from surrogate_optimization.optimization.active_set import ExactQPActiveSetRefiner
    from surrogate_optimization.optimization.surrogate import EXPRESSION_GAP
    from surrogate_optimization.optimization.surrogate import LOCAL_CONVERGENCE_PROTOCOL
    from surrogate_optimization.optimization.surrogate import _vector
    from surrogate_optimization.optimization.surrogate import audit_exact_candidate
    from surrogate_optimization.optimization.surrogate import (
        build_surrogate_expression_graph,
    )
    from surrogate_optimization.optimization.types import StationarityRecord
    from surrogate_optimization.optimization.types import SurrogateNLPError
    from surrogate_optimization.optimization.types import SurrogateSolverSettings

    configuration = settings or SurrogateCertificationSettings()
    started = perf_counter()
    solver_settings = SurrogateSolverSettings(
        final_upper_tolerance=configuration.feasibility_tolerance,
        maximum_wall_time=None,
    )
    if problem is None:
        problem = build_surrogate_expression_graph(
            assets, EXPRESSION_GAP, settings=solver_settings, name=f"{name}_expressions"
        )
    elif problem.assets is not assets:
        raise ValueError("the certification problem uses different surrogate assets.")
    initial_controls = _vector(
        initial_candidate.normalized_controls, 7, "initial candidate controls"
    )
    initial_objective = float(initial_candidate.objective)
    evaluations = 0
    cold_qp_resolutions = 0
    accepted_improvements = 0
    poll_levels: list[dict[str, Any]] = []
    cache: dict[bytes, tuple[FinalCandidateRecord | None, str | None]] = {}

    def evaluate(
        controls: npt.ArrayLike,
    ) -> tuple[FinalCandidateRecord | None, str | None]:
        nonlocal evaluations, cold_qp_resolutions
        value = np.clip(_vector(controls, 7, "poll controls"), 0.0, 1.0)
        key = np.ascontiguousarray(value, dtype=np.float64).tobytes()
        if key in cache:
            return cache[key]
        if evaluations >= configuration.maximum_evaluations:
            return (None, "certification evaluation budget exhausted")
        evaluations += 1
        cold_qp_resolutions += 1
        try:
            candidate = audit_exact_candidate(
                problem, case, value, settings=solver_settings
            )
            if (
                not candidate.projection.accepted
                or not candidate.feasibility.finite
                or (not np.isfinite(candidate.objective))
            ):
                raise SurrogateNLPError("poll trial failed the exact-QP finite audit")
            result = (candidate, None)
        except Exception as exc:
            result = (None, f"{type(exc).__name__}: {exc}")
        cache[key] = result
        return result

    current, initial_error = evaluate(initial_controls)
    if current is None or not current.feasibility.feasible:
        reason = initial_error or "initial endpoint failed upper feasibility"
        certificate = LocalConvergenceCertificate(
            protocol=LOCAL_CONVERGENCE_PROTOCOL,
            classification="initial_candidate_invalid",
            locally_converged=False,
            first_order_certified=False,
            stationarity_resolved=False,
            initial_objective=initial_objective,
            final_objective=float("nan")
            if current is None
            else float(current.objective),
            initial_normalized_controls=initial_controls,
            final_normalized_controls=initial_controls
            if current is None
            else current.normalized_controls.copy(),
            evaluations=evaluations,
            cold_qp_resolutions=cold_qp_resolutions,
            accepted_improvements=0,
            elapsed_seconds=perf_counter() - started,
            termination_reason=reason,
            poll_levels=(),
        )
        return SurrogateCertificationResult(None, certificate)
    initial_kkt_error: str | None = None
    initial_lower_active_set: dict[str, Any] | None = None
    initial_upper_kkt: dict[str, Any] | None = None
    initial_refiner = ExactQPActiveSetRefiner(
        assets,
        case,
        problem=problem,
        settings=ActiveSetRefinementSettings(
            upper_acceptance_tolerance=configuration.feasibility_tolerance
        ),
        name=f"{name}_initial_endpoint",
    )
    try:
        initial_trial = initial_refiner.evaluate(
            current.normalized_controls, force_cold=True, independent_final_replay=True
        )
        initial_upper = initial_refiner.audit_upper_kkt(initial_trial)
        initial_lower_active_set = initial_trial.lower_active_set.as_dict()
        initial_upper_kkt = initial_upper.as_dict()
        initial_kkt_passed = bool(initial_upper.stationary)
    except ActiveSetDerivativeError as exc:
        initial_lower_active_set = None if exc.audit is None else exc.audit.as_dict()
        initial_kkt_error = str(exc)
        initial_kkt_passed = False
    except Exception as exc:
        initial_kkt_error = f"{type(exc).__name__}: {exc}"
        initial_kkt_passed = False
    finally:
        cold_qp_resolutions += initial_refiner.cold_qp_resolutions
    if initial_kkt_passed:
        reason = "the independently reconstructed lower active set and upper KKT audit passed at the cold-replayed endpoint"
        current = replace(
            current,
            stationarity=StationarityRecord(
                classification="first_order_kkt_stationary_feasible",
                resolved=True,
                stationary=True,
                lower_qp_kkt_passed=True,
                upper_stationarity_residual=None
                if initial_upper_kkt is None
                else float(initial_upper_kkt["stationarity_residual"]),
                reason=reason,
            ),
            status="validated_stationary",
            lower_active_set=initial_lower_active_set,
            upper_kkt=initial_upper_kkt,
        )
        certificate = LocalConvergenceCertificate(
            protocol=LOCAL_CONVERGENCE_PROTOCOL,
            classification="exact_active_set_kkt",
            locally_converged=True,
            first_order_certified=True,
            stationarity_resolved=True,
            initial_objective=initial_objective,
            final_objective=float(current.objective),
            initial_normalized_controls=initial_controls,
            final_normalized_controls=current.normalized_controls.copy(),
            evaluations=evaluations,
            cold_qp_resolutions=cold_qp_resolutions,
            accepted_improvements=0,
            elapsed_seconds=perf_counter() - started,
            termination_reason=reason,
            poll_levels=(),
            lower_active_set=initial_lower_active_set,
            upper_kkt=initial_upper_kkt,
        )
        return SurrogateCertificationResult(current, certificate)
    termination_reason = "all declared poll levels completed without sufficient descent"
    all_levels_passed = False
    validation_round = 0
    while True:
        round_passed = True
        earlier_level_became_stale = False
        for level_index, radius in enumerate(configuration.poll_radii):
            directions = _local_poll_directions(
                include_simplex=level_index == len(configuration.poll_radii) - 1
            )
            level_evaluations = 0
            level_failures: list[str] = []
            level_improvements = 0
            level_poll_improvements = 0
            acceleration_requests = 0
            acceleration_unique_evaluations = 0
            acceleration_accepted_improvements = 0
            acceleration_maximum_accepted_multiplier = 1.0
            acceleration_stops: dict[str, int] = {}
            last_feasible_directions: list[FloatArray] = []
            last_best_decrease = 0.0
            complete_no_descent_poll = False
            while True:
                center = current.normalized_controls.copy()
                center_objective = float(current.objective)
                decrease_tolerance = max(
                    configuration.absolute_decrease_tolerance,
                    configuration.relative_decrease_tolerance * abs(center_objective),
                )
                feasible_trials: list[FinalCandidateRecord] = []
                feasible_directions: list[FloatArray] = []
                sweep_complete = True
                seen: set[bytes] = set()
                for direction in directions:
                    trial_controls = np.clip(
                        center + float(radius) * direction, 0.0, 1.0
                    )
                    displacement = trial_controls - center
                    norm = float(np.linalg.norm(displacement))
                    if norm <= np.finfo(float).eps:
                        continue
                    key = np.ascontiguousarray(
                        trial_controls, dtype=np.float64
                    ).tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    trial, error = evaluate(trial_controls)
                    level_evaluations += 1
                    if error is not None or trial is None:
                        level_failures.append(
                            error or "unknown exact-QP evaluation failure"
                        )
                        sweep_complete = False
                        continue
                    if trial.feasibility.feasible:
                        feasible_trials.append(trial)
                        feasible_directions.append(displacement / norm)
                last_feasible_directions = feasible_directions
                if not sweep_complete:
                    if any(("budget exhausted" in item for item in level_failures)):
                        termination_reason = "certification evaluation budget exhausted"
                    elif level_failures:
                        termination_reason = "one or more required poll QPs failed"
                    break
                improving = [
                    trial
                    for trial in feasible_trials
                    if float(trial.objective) < center_objective - decrease_tolerance
                ]
                if not improving:
                    last_best_decrease = max(
                        (
                            center_objective - float(trial.objective)
                            for trial in feasible_trials
                        ),
                        default=float("nan"),
                    )
                    complete_no_descent_poll = True
                    break
                best_objective = min((float(trial.objective) for trial in improving))
                tie = 1e-12 * max(1.0, abs(best_objective))
                selected = min(
                    (
                        trial
                        for trial in improving
                        if float(trial.objective) <= best_objective + tie
                    ),
                    key=lambda item: tuple(item.normalized_controls.tolist()),
                )
                current = selected
                accepted_improvements += 1
                level_improvements += 1
                level_poll_improvements += 1
                base_step = current.normalized_controls - center
                nonzero = np.abs(base_step) > np.finfo(float).eps
                maximum_multiplier = float("inf")
                if np.any(nonzero):
                    positive = base_step > np.finfo(float).eps
                    negative = base_step < -np.finfo(float).eps
                    bounds: list[FloatArray] = []
                    if np.any(positive):
                        bounds.append((1.0 - center[positive]) / base_step[positive])
                    if np.any(negative):
                        bounds.append(-center[negative] / base_step[negative])
                    if bounds:
                        maximum_multiplier = float(np.min(np.concatenate(bounds)))
                multiplier = float(configuration.acceleration_growth_factor)
                previous_controls = current.normalized_controls.copy()
                for _ in range(configuration.maximum_acceleration_probes):
                    bounded_multiplier = min(multiplier, maximum_multiplier)
                    if bounded_multiplier <= 1.0 + 1e-12:
                        acceleration_stops["box_boundary"] = (
                            acceleration_stops.get("box_boundary", 0) + 1
                        )
                        break
                    trial_controls = np.clip(
                        center + bounded_multiplier * base_step, 0.0, 1.0
                    )
                    if np.array_equal(trial_controls, previous_controls):
                        acceleration_stops["duplicate_or_boundary"] = (
                            acceleration_stops.get("duplicate_or_boundary", 0) + 1
                        )
                        break
                    evaluations_before = evaluations
                    accelerated, acceleration_error = evaluate(trial_controls)
                    acceleration_requests += 1
                    acceleration_unique_evaluations += evaluations - evaluations_before
                    if acceleration_error is not None or accelerated is None:
                        stop = (
                            "evaluation_budget_exhausted"
                            if acceleration_error is not None
                            and "budget exhausted" in acceleration_error
                            else "exact_qp_failure"
                        )
                        acceleration_stops[stop] = acceleration_stops.get(stop, 0) + 1
                        break
                    if not accelerated.feasibility.feasible:
                        acceleration_stops["upper_infeasible"] = (
                            acceleration_stops.get("upper_infeasible", 0) + 1
                        )
                        break
                    accelerated_decrease_tolerance = max(
                        configuration.absolute_decrease_tolerance,
                        configuration.relative_decrease_tolerance
                        * abs(float(current.objective)),
                    )
                    if (
                        not float(accelerated.objective)
                        < float(current.objective) - accelerated_decrease_tolerance
                    ):
                        acceleration_stops["no_sufficient_descent"] = (
                            acceleration_stops.get("no_sufficient_descent", 0) + 1
                        )
                        break
                    current = accelerated
                    previous_controls = accelerated.normalized_controls.copy()
                    accepted_improvements += 1
                    level_improvements += 1
                    acceleration_accepted_improvements += 1
                    acceleration_maximum_accepted_multiplier = max(
                        acceleration_maximum_accepted_multiplier,
                        float(bounded_multiplier),
                    )
                    if bounded_multiplier >= maximum_multiplier - 1e-12:
                        acceleration_stops["box_boundary"] = (
                            acceleration_stops.get("box_boundary", 0) + 1
                        )
                        break
                    multiplier *= configuration.acceleration_growth_factor
                else:
                    acceleration_stops["expansion_limit"] = (
                        acceleration_stops.get("expansion_limit", 0) + 1
                    )
            feasible_rank = (
                int(
                    np.linalg.matrix_rank(
                        np.vstack(last_feasible_directions),
                        tol=configuration.direction_rank_tolerance,
                    )
                )
                if last_feasible_directions
                else 0
            )
            coverage_passed = bool(last_feasible_directions)
            level_passed = bool(
                complete_no_descent_poll and (not level_failures) and coverage_passed
            )
            poll_levels.append(
                {
                    "validation_round": validation_round,
                    "radius": float(radius),
                    "direction_set": "signed_coordinate_pairwise_plus_helmert_simplex"
                    if level_index == len(configuration.poll_radii) - 1
                    else "signed_coordinate",
                    "direction_count": int(len(directions)),
                    "evaluations": level_evaluations,
                    "feasible_direction_count": len(last_feasible_directions),
                    "feasible_direction_rank": feasible_rank,
                    "required_direction_rank": None,
                    "rank_is_diagnostic_only": True,
                    "feasible_direction_coverage_passed": coverage_passed,
                    "accepted_improvements": level_improvements,
                    "poll_accepted_improvements": level_poll_improvements,
                    "acceleration_evaluation_requests": acceleration_requests,
                    "acceleration_unique_evaluations": acceleration_unique_evaluations,
                    "acceleration_accepted_improvements": acceleration_accepted_improvements,
                    "acceleration_maximum_accepted_multiplier": acceleration_maximum_accepted_multiplier,
                    "acceleration_stops": dict(sorted(acceleration_stops.items())),
                    "best_objective_decrease_in_final_sweep": last_best_decrease,
                    "complete_no_descent_poll": complete_no_descent_poll,
                    "passed": level_passed,
                    "failures": level_failures,
                }
            )
            if not level_passed:
                round_passed = False
                if complete_no_descent_poll and (not level_failures):
                    termination_reason = (
                        "the declared poll contained no feasible nonzero displacement"
                    )
                break
            if level_index > 0 and level_improvements:
                earlier_level_became_stale = True
        if not round_passed:
            break
        if earlier_level_became_stale:
            validation_round += 1
            continue
        all_levels_passed = True
        break
    if current is None:
        raise AssertionError("local certification lost its feasible incumbent")
    cold_qp_resolutions += 1
    try:
        final_replay = audit_exact_candidate(
            problem, case, current.normalized_controls, settings=solver_settings
        )
        replay_error = None
    except Exception as exc:
        final_replay = None
        replay_error = f"{type(exc).__name__}: {exc}"
    if final_replay is None or not final_replay.feasibility.feasible:
        all_levels_passed = False
        termination_reason = replay_error or "final endpoint replay failed feasibility"
    else:
        current = final_replay
    lower_active_set: dict[str, Any] | None = None
    upper_kkt: dict[str, Any] | None = None
    first_order_certified = False
    kkt_error: str | None = initial_kkt_error
    if final_replay is not None:
        active_settings = ActiveSetRefinementSettings(
            upper_acceptance_tolerance=configuration.feasibility_tolerance
        )
        refiner = ExactQPActiveSetRefiner(
            assets,
            case,
            problem=problem,
            settings=active_settings,
            name=f"{name}_endpoint",
        )
        try:
            trial = refiner.evaluate(
                current.normalized_controls,
                force_cold=True,
                independent_final_replay=True,
            )
            upper = refiner.audit_upper_kkt(trial)
            lower_active_set = trial.lower_active_set.as_dict()
            upper_kkt = upper.as_dict()
            first_order_certified = bool(upper.stationary)
        except ActiveSetDerivativeError as exc:
            lower_active_set = None if exc.audit is None else exc.audit.as_dict()
            kkt_error = str(exc)
        except Exception as exc:
            kkt_error = f"{type(exc).__name__}: {exc}"
        finally:
            cold_qp_resolutions += refiner.cold_qp_resolutions
    locally_converged = bool(first_order_certified or all_levels_passed)
    if first_order_certified:
        classification = "exact_active_set_kkt"
        termination_reason = (
            "the independently replayed endpoint passed the lower and upper KKT audits"
        )
        stationarity = StationarityRecord(
            classification="first_order_kkt_stationary_feasible",
            resolved=True,
            stationary=True,
            lower_qp_kkt_passed=True,
            upper_stationarity_residual=None
            if upper_kkt is None
            else float(upper_kkt["stationarity_residual"]),
            reason=termination_reason,
        )
        status = "validated_stationary"
    elif all_levels_passed:
        classification = "finite_resolution_feasible_poll"
        stationarity = StationarityRecord(
            classification="poll_converged_stationarity_unresolved",
            resolved=False,
            stationary=False,
            lower_qp_kkt_passed=bool(current.projection.accepted),
            upper_stationarity_residual=None,
            reason="The exact-QP endpoint passed the complete two-scale feasible no-descent poll. This establishes finite-resolution local convergence, not classical stationarity.",
        )
        status = "validated_feasible_poll_converged_stationarity_unresolved"
    else:
        classification = (
            "poll_budget_limited"
            if "budget" in termination_reason
            else "poll_inconclusive"
        )
        stationarity = StationarityRecord(
            classification=f"{classification}_stationarity_unresolved",
            resolved=False,
            stationary=False,
            lower_qp_kkt_passed=bool(current.projection.accepted),
            upper_stationarity_residual=None,
            reason=f"{termination_reason}; endpoint KKT audit: {kkt_error or 'did not pass'}",
        )
        status = f"validated_feasible_{classification}_stationarity_unresolved"
    current = replace(
        current,
        stationarity=stationarity,
        status=status,
        lower_active_set=lower_active_set,
        upper_kkt=upper_kkt,
    )
    certificate = LocalConvergenceCertificate(
        protocol=LOCAL_CONVERGENCE_PROTOCOL,
        classification=classification,
        locally_converged=locally_converged,
        first_order_certified=first_order_certified,
        stationarity_resolved=first_order_certified,
        initial_objective=initial_objective,
        final_objective=float(current.objective),
        initial_normalized_controls=initial_controls,
        final_normalized_controls=current.normalized_controls.copy(),
        evaluations=evaluations,
        cold_qp_resolutions=cold_qp_resolutions,
        accepted_improvements=accepted_improvements,
        elapsed_seconds=perf_counter() - started,
        termination_reason=termination_reason,
        poll_levels=tuple(poll_levels),
        lower_active_set=lower_active_set,
        upper_kkt=upper_kkt,
    )
    return SurrogateCertificationResult(current, certificate)


def _surrogate_certification_contract(
    *,
    source_id: str,
    analysis_id: str,
    case_id: str,
    influent: np.ndarray,
    route_artifact_digest: str,
    settings: SurrogateCertificationSettings,
) -> str:
    from surrogate_optimization.runtime.protocols import COMPARISON_PROTOCOL

    digest = sha256()
    digest.update(COMPARISON_PROTOCOL.encode())
    digest.update(source_id.encode())
    digest.update(analysis_id.encode())
    digest.update(case_id.encode())
    digest.update(route_artifact_digest.encode())
    digest.update(np.ascontiguousarray(influent, dtype="<f8").tobytes())
    digest.update(json.dumps(asdict(settings), sort_keys=True).encode())
    return digest.hexdigest()


def _run_surrogate_certification(
    case_directory: Path,
    *,
    case_id: str,
    influent: np.ndarray,
    result: SurrogateRouteResult,
    analysis: AnalysisBundle,
    source_id: str,
    analysis_id: str,
    problem: Any,
) -> tuple[FinalCandidateRecord | None, dict[str, Any]]:
    """Certify/polish one retained surrogate endpoint without rerunning search."""
    from surrogate_optimization.optimization.types import FinalCandidateRecord
    from surrogate_optimization.optimization.types import SurrogateCase
    from surrogate_optimization.runtime.artifacts import atomic_json
    from surrogate_optimization.runtime.artifacts import atomic_npz
    from surrogate_optimization.runtime.contracts import _artifact_hashes
    from surrogate_optimization.runtime.contracts import _artifacts_match
    from surrogate_optimization.runtime.contracts import _load_json_object
    from surrogate_optimization.runtime.contracts import file_digest

    settings = SurrogateCertificationSettings()
    route_path = case_directory / "surrogate.json"
    contract = _surrogate_certification_contract(
        source_id=source_id,
        analysis_id=analysis_id,
        case_id=case_id,
        influent=influent,
        route_artifact_digest=file_digest(route_path),
        settings=settings,
    )
    payload_path = case_directory / "surrogate_local_convergence.json"
    candidate_path = case_directory / "surrogate_certified.npz"
    marker_path = case_directory / "surrogate_local_convergence_complete.json"
    if marker_path.is_file() and payload_path.is_file():
        marker = _load_json_object(
            marker_path, description="surrogate certification marker"
        )
        payload = _load_json_object(payload_path, description="surrogate certification")
        if (
            marker.get("certification_contract") == contract
            and payload.get("certification_contract") == contract
            and _artifacts_match(case_directory, marker.get("artifacts", {}))
        ):
            candidate_value = payload.get("candidate")
            candidate = (
                None
                if candidate_value is None
                else FinalCandidateRecord.from_dict(candidate_value)
            )
            return (candidate, payload)
    if result.selected is None or result.selected.final is None:
        candidate_path.unlink(missing_ok=True)
        payload = {
            "stage": "surrogate_local_convergence",
            "case": case_id,
            "certification_contract": contract,
            "selected": False,
            "locally_converged": False,
            "first_order_certified": False,
            "status": "no_surrogate_candidate",
            "settings": asdict(settings),
            "candidate": None,
            "certificate": None,
        }
        atomic_json(payload_path, payload)
        atomic_json(
            marker_path,
            {
                "stage": "surrogate_local_convergence",
                "certification_contract": contract,
                "selected": False,
                "artifacts": _artifact_hashes(
                    case_directory, (payload_path, route_path)
                ),
            },
        )
        return (None, payload)
    certification = certify_surrogate_local_convergence(
        analysis.surrogate_assets,
        SurrogateCase(influent=influent, case_id=case_id),
        result.selected.final,
        settings=settings,
        problem=problem,
        name=f"study_surrogate_certificate_{case_id}",
    )
    candidate = certification.candidate
    certificate = certification.certificate
    payload = {
        "stage": "surrogate_local_convergence",
        "case": case_id,
        "certification_contract": contract,
        "selected": candidate is not None,
        "locally_converged": bool(certificate.locally_converged),
        "first_order_certified": bool(certificate.first_order_certified),
        "status": certificate.classification,
        "settings": asdict(settings),
        "candidate": None if candidate is None else candidate.as_dict(),
        "certificate": certificate.as_dict(),
    }
    atomic_json(payload_path, payload, nonfinite_to_none=True)
    artifacts: list[Path] = [payload_path, route_path]
    if candidate is not None:
        atomic_npz(
            candidate_path,
            normalized_controls=candidate.normalized_controls,
            controls=candidate.controls,
            raw=candidate.raw,
            projected=candidate.projected,
            objective=np.asarray(candidate.objective),
            objective_components=candidate.objective_components,
        )
        artifacts.append(candidate_path)
    atomic_json(
        marker_path,
        {
            "stage": "surrogate_local_convergence",
            "certification_contract": contract,
            "selected": candidate is not None,
            "locally_converged": bool(certificate.locally_converged),
            "first_order_certified": bool(certificate.first_order_certified),
            "artifacts": _artifact_hashes(case_directory, tuple(artifacts)),
        },
    )
    return (candidate, payload)
