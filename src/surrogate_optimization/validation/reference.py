"""Validation reference."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.workflow.types import AnalysisBundle
    from surrogate_optimization.optimization.types import FinalCandidateRecord
    from surrogate_optimization.optimization.mechanistic import MechanisticRouteResult
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any
from typing import Mapping
import numpy as np
import pandas as pd


def _unavailable_violation(method: str, case: str, reason: str) -> dict[str, Any]:
    return {
        "case": case,
        "method": method,
        "audit_available": False,
        "audit_unavailable_reason": reason,
        "mass_conservation_violation_max": np.nan,
        "mass_conservation_violation_count": 0,
        "nonnegativity_violation_max": np.nan,
        "nonnegativity_violation_count": 0,
    }


def _physical_record(
    method: str,
    case: str,
    response: np.ndarray,
    controls: np.ndarray,
    influent: np.ndarray,
    analysis: AnalysisBundle,
) -> dict[str, Any]:
    from surrogate_optimization.validation.physical import violation_record

    if response.shape != (analysis.surrogate_assets.layout.state_size,) or not np.all(
        np.isfinite(response)
    ):
        return _unavailable_violation(
            method, case, "response unavailable or non-finite"
        )
    overflow_tss_closure = None
    if analysis.overflow_closure is not None:
        overflow_tss_closure = float(
            analysis.overflow_closure.predict(controls, influent)
        )
    record = violation_record(
        method,
        case,
        response,
        controls,
        influent,
        analysis.surrogate_assets.layout,
        analysis.surrogate_assets.row_scales.equality,
        analysis.surrogate_assets.row_scales.inequality,
        analysis.model.response_scale,
        overflow_tss_closure=overflow_tss_closure,
    )
    record["audit_available"] = True
    record["audit_unavailable_reason"] = None
    return record


def _casewise_exact_reference(
    controls: np.ndarray,
    influent: np.ndarray,
    analysis: AnalysisBundle,
    *,
    retained_reference_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate a decision with the exact nonsmooth reference model."""
    from surrogate_optimization.optimization.mechanistic import (
        DEFAULT_OBJECTIVE_WEIGHTS,
    )
    from surrogate_optimization.optimization.mechanistic import (
        branches_match as smooth_branches_match,
    )
    from surrogate_optimization.optimization.mechanistic import classify_branches
    from surrogate_optimization.optimization.mechanistic import engineering_feasible
    from surrogate_optimization.optimization.mechanistic import engineering_quantities
    from surrogate_optimization.optimization.mechanistic import objective_components
    from surrogate_optimization.plant.model import assemble_target
    from surrogate_optimization.plant.model import (
        diagnostics as mechanistic_diagnostics,
    )
    from surrogate_optimization.plant.model import generation_scale
    from surrogate_optimization.plant.model import solve_steady_state
    from surrogate_optimization.plant.model import unpack_state
    from surrogate_optimization.plant.operating_point import OperatingPoint
    from surrogate_optimization.runtime.contracts import _load_json_object

    controls = np.asarray(controls, dtype=np.float64)
    feed = np.asarray(influent, dtype=np.float64)
    clarifier = analysis.mechanistic_assets.clarifier
    state_size = analysis.mechanistic_assets.state_count
    response_size = analysis.mechanistic_assets.response_count
    started = perf_counter()
    first_state: np.ndarray | None = None
    second_state: np.ndarray | None = None
    first_accepted = False
    second_accepted: bool | None = None
    source = "adaptive_exact_solve"
    errors: list[str] = []
    retained_original_elapsed_seconds: float | None = None
    if retained_reference_path is not None and retained_reference_path.is_file():
        try:
            with np.load(retained_reference_path, allow_pickle=False) as stored:
                retained_theta = np.asarray(stored["controls"], dtype=np.float64)
                if np.array_equal(retained_theta, controls):
                    first_state = np.asarray(stored["state"], dtype=np.float64)
                    second_state = np.asarray(stored["state_start_2"], dtype=np.float64)
                    if (
                        first_state.shape == (state_size,)
                        and second_state.shape == (state_size,)
                        and np.all(np.isfinite(first_state))
                        and np.all(np.isfinite(second_state))
                    ):
                        source = "retained_two_start_exact_replay"
                        equivalence_path = retained_reference_path.with_name(
                            retained_reference_path.name.replace(
                                "_reference.npz", "_equivalence.json"
                            )
                        )
                        if equivalence_path.is_file():
                            equivalence = _load_json_object(
                                equivalence_path,
                                description="retained exact-reference timing",
                            )
                            replay = equivalence.get("reference_replay")
                            if isinstance(replay, Mapping):
                                retained_elapsed = replay.get("elapsed_seconds")
                                if retained_elapsed is not None:
                                    retained_original_elapsed_seconds = float(
                                        retained_elapsed
                                    )
                    else:
                        first_state = second_state = None
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"retained replay: {type(exc).__name__}: {exc}")
            first_state = second_state = None
    operating = OperatingPoint(*map(float, controls))
    if first_state is None:
        try:
            first = solve_steady_state(
                operating,
                feed,
                starts=(1,),
                clarifier=clarifier,
                logarithmic_only=True,
                require_physical_audit=True,
            )
            first_accepted = bool(first.accepted)
            if first.accepted:
                first_state = np.asarray(first.state, dtype=np.float64)
            else:
                errors.append(f"start 1 rejected: {first.message}")
        except Exception as exc:
            errors.append(f"start 1: {type(exc).__name__}: {exc}")
    else:
        first_accepted = True
    first_branch = (
        classify_branches(first_state, analysis.mechanistic_assets)
        if first_state is not None
        else None
    )
    second_required = True
    if second_state is not None:
        second_accepted = True
    elif second_required:
        try:
            second = solve_steady_state(
                operating,
                feed,
                starts=(2,),
                clarifier=clarifier,
                logarithmic_only=True,
                require_physical_audit=True,
            )
            second_accepted = bool(second.accepted)
            if second.accepted:
                second_state = np.asarray(second.state, dtype=np.float64)
            else:
                errors.append(f"start 2 rejected: {second.message}")
        except Exception as exc:
            second_accepted = False
            errors.append(f"start 2: {type(exc).__name__}: {exc}")
    selected_state = first_state if first_state is not None else second_state
    if selected_state is None:
        return (
            np.full(response_size, np.nan),
            np.full(state_size, np.nan),
            np.full(state_size, np.nan),
            {
                "accepted": False,
                "status": "solver_failed",
                "source": source,
                "start_1_accepted": first_accepted,
                "start_2_required": second_required,
                "start_2_accepted": second_accepted,
                "elapsed_seconds": perf_counter() - started,
                "errors": errors,
            },
        )
    selected_diagnostics = mechanistic_diagnostics(
        selected_state,
        operating,
        feed,
        clarifier=clarifier,
        require_physical_audit=True,
    )
    physical_passed = bool(selected_diagnostics.get("passed", False))
    selected_branch = classify_branches(selected_state, analysis.mechanistic_assets)
    branch_agreement: bool | None = None
    generation_difference: float | None = None
    state_scale_difference: float | None = None
    first_diagnostics: dict[str, Any] | None = None
    second_diagnostics: dict[str, Any] | None = None
    second_branch = None
    if first_state is not None:
        first_diagnostics = mechanistic_diagnostics(
            first_state,
            operating,
            feed,
            clarifier=clarifier,
            require_physical_audit=True,
        )
    if second_state is not None:
        second_diagnostics = mechanistic_diagnostics(
            second_state,
            operating,
            feed,
            clarifier=clarifier,
            require_physical_audit=True,
        )
        comparison_state = first_state if first_state is not None else selected_state
        reactors, _ = unpack_state(comparison_state, clarifier)
        scale = generation_scale(feed, reactors[-1], clarifier)
        generation_difference = float(
            np.max(np.abs(comparison_state - second_state) / scale)
        )
        state_scale_difference = float(
            np.max(
                np.abs(comparison_state - second_state)
                / analysis.mechanistic_assets.state_scale
            )
        )
        first_comparison_branch = classify_branches(
            comparison_state, analysis.mechanistic_assets
        )
        second_branch = classify_branches(second_state, analysis.mechanistic_assets)
        branch_agreement = smooth_branches_match(first_comparison_branch, second_branch)
        physical_passed = bool(
            physical_passed and second_diagnostics.get("passed", False)
        )
    replay_agreement = bool(
        first_accepted
        and second_accepted is True
        and (second_state is not None)
        and (generation_difference is not None)
        and (generation_difference <= 1e-06)
        and (state_scale_difference is not None)
        and (state_scale_difference <= 1e-06)
        and (
            branch_agreement is True
            or selected_branch.ambiguous
            or (second_branch is not None and second_branch.ambiguous)
        )
    )
    accepted = bool(first_accepted and physical_passed and replay_agreement)
    response = assemble_target(selected_state, operating, feed, clarifier)
    reference_components = objective_components(
        controls, response, analysis.mechanistic_assets
    )
    reference_objective = float(DEFAULT_OBJECTIVE_WEIGHTS @ reference_components)
    quantities = engineering_quantities(controls, response, analysis.mechanistic_assets)
    reference_engineering_feasible = engineering_feasible(
        controls, response, analysis.mechanistic_assets
    )
    boundary_ambiguous = bool(
        selected_branch.ambiguous
        or (second_branch is not None and second_branch.ambiguous)
    )
    status = (
        "valid_branch_boundary"
        if accepted and boundary_ambiguous
        else "valid_interior"
        if accepted
        else "start_1_failed"
        if not first_accepted
        else "start_2_failed"
        if second_accepted is not True
        else "root_disagreement"
        if second_state is not None and (not replay_agreement)
        else "physical_audit_failed"
    )
    current_execution_elapsed_seconds = perf_counter() - started
    payload = {
        "accepted": accepted,
        "status": status,
        "source": source,
        "start_1_accepted": first_accepted,
        "start_2_required": second_required,
        "start_2_accepted": second_accepted,
        "two_start_agreement_checked": second_state is not None,
        "scaled_root_difference_generation": generation_difference,
        "scaled_root_difference_state": state_scale_difference,
        "branch_agreement": branch_agreement,
        "branch_disagreement_excused_by_boundary_ambiguity": bool(
            branch_agreement is False
            and (
                selected_branch.ambiguous
                or (second_branch is not None and second_branch.ambiguous)
            )
        ),
        "branch_ambiguous": boundary_ambiguous,
        "minimum_normalized_branch_margin": float(
            selected_branch.minimum_normalized_margin
        ),
        "branch_start_1": None if first_branch is None else asdict(first_branch),
        "branch_start_2": None if second_branch is None else asdict(second_branch),
        "physical_stability_accepted": physical_passed,
        "diagnostics_start_1": first_diagnostics,
        "diagnostics_start_2": second_diagnostics,
        "engineering_feasible": bool(reference_engineering_feasible),
        "engineering_quantities": quantities,
        "objective": reference_objective,
        "objective_components": reference_components.tolist(),
        "elapsed_seconds": retained_original_elapsed_seconds
        if retained_original_elapsed_seconds is not None
        else current_execution_elapsed_seconds,
        "current_run_reuse_overhead_seconds": current_execution_elapsed_seconds,
        "retained_original_solve_elapsed_seconds": retained_original_elapsed_seconds,
        "errors": errors,
    }
    return (
        np.asarray(response, dtype=np.float64),
        np.full(state_size, np.nan)
        if first_state is None
        else np.asarray(first_state, dtype=np.float64),
        np.full(state_size, np.nan)
        if second_state is None
        else np.asarray(second_state, dtype=np.float64),
        payload,
    )


def _mechanistic_recovery_contract(
    *,
    source_id: str,
    analysis_id: str,
    case_id: str,
    influent: np.ndarray,
    recovery_start: np.ndarray,
    route_artifact_digest: str,
) -> str:
    digest = sha256()
    digest.update(b"smooth-direct-single-failure-recovery-v1\x00")
    digest.update(source_id.encode())
    digest.update(analysis_id.encode())
    digest.update(case_id.encode())
    digest.update(route_artifact_digest.encode())
    digest.update(np.ascontiguousarray(influent, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(recovery_start, dtype="<f8").tobytes())
    return digest.hexdigest()


def _run_mechanistic_failure_recovery(
    case_directory: Path,
    *,
    case_id: str,
    influent: np.ndarray,
    result: MechanisticRouteResult,
    surrogate_candidate: FinalCandidateRecord | None,
    assets: Any,
    development_controls: np.ndarray,
    development_influents: np.ndarray,
    development_responses: np.ndarray,
    source_id: str,
    analysis_id: str,
) -> tuple[MechanisticRouteResult, dict[str, Any]]:
    """Permit one declared recovery start only after a primary direct failure."""
    from surrogate_optimization.optimization.mechanistic import MechanisticCase
    from surrogate_optimization.optimization.mechanistic import MechanisticRouteResult
    from surrogate_optimization.optimization.mechanistic import MechanisticStartResult
    from surrogate_optimization.optimization.mechanistic import SolverSettings
    from surrogate_optimization.optimization.mechanistic import solve_mechanistic_case
    from surrogate_optimization.runtime.artifacts import atomic_json
    from surrogate_optimization.runtime.checkpoints import (
        _validate_route_result_integrity,
    )
    from surrogate_optimization.runtime.contracts import _artifact_hashes
    from surrogate_optimization.runtime.contracts import _artifacts_match
    from surrogate_optimization.runtime.contracts import _load_json_object
    from surrogate_optimization.runtime.contracts import file_digest

    if result.selected is not None:
        return (
            result,
            {
                "attempted": False,
                "selected_from": "primary_center_start",
                "status": "not_required",
            },
        )
    if surrogate_candidate is None:
        return (
            result,
            {
                "attempted": False,
                "selected_from": None,
                "status": "unavailable_without_convergence_certified_surrogate_candidate",
            },
        )
    start = np.asarray(surrogate_candidate.normalized_controls, dtype=np.float64)
    route_path = case_directory / "mechanistic.json"
    contract = _mechanistic_recovery_contract(
        source_id=source_id,
        analysis_id=analysis_id,
        case_id=case_id,
        influent=influent,
        recovery_start=start,
        route_artifact_digest=file_digest(route_path),
    )
    payload_path = case_directory / "mechanistic_recovery.json"
    marker_path = case_directory / "mechanistic_recovery_complete.json"
    if marker_path.is_file() and payload_path.is_file():
        marker = _load_json_object(marker_path, description="direct recovery marker")
        payload = _load_json_object(payload_path, description="direct recovery")
        if (
            marker.get("recovery_contract") == contract
            and payload.get("recovery_contract") == contract
            and _artifacts_match(case_directory, marker.get("artifacts", {}))
        ):
            stored_result = payload.get("result")
            if stored_result is None:
                return (result, payload)
            if not isinstance(stored_result, Mapping):
                raise RuntimeError("cached direct recovery result is malformed")
            restored = tuple(
                (
                    MechanisticStartResult.from_dict(item)
                    for item in stored_result["starts"]
                )
            )
            if (
                len(restored) != 1
                or restored[0].start_index != 0
                or (not np.array_equal(restored[0].initial_normalized_controls, start))
            ):
                raise RuntimeError(
                    "cached direct recovery violates its one-start contract"
                )
            for item in restored:
                _validate_route_result_integrity(item, "mechanistic")
            selected_index = stored_result.get("selected_start")
            if selected_index not in (None, 0):
                raise RuntimeError("cached direct recovery has an invalid selection")
            selected = None if selected_index is None else restored[int(selected_index)]
            return (
                MechanisticRouteResult(
                    restored, selected, str(stored_result["status"])
                ),
                payload,
            )
    started = perf_counter()
    try:
        recovered = solve_mechanistic_case(
            assets,
            MechanisticCase(influent=influent, case_id=case_id),
            development_controls,
            development_influents,
            development_responses,
            settings=SolverSettings(maximum_wall_time=None),
            starts=start.reshape(1, 7),
        )
        if (
            len(recovered.starts) != 1
            or recovered.starts[0].start_index != 0
            or (
                not np.array_equal(
                    recovered.starts[0].initial_normalized_controls, start
                )
            )
        ):
            raise RuntimeError("fresh direct recovery violated its one-start contract")
        for item in recovered.starts:
            _validate_route_result_integrity(item, "mechanistic")
        error = None
    except Exception as exc:
        recovered = result
        error = f"{type(exc).__name__}: {exc}"
    payload = {
        "stage": "mechanistic_failure_recovery",
        "protocol": "smooth_mechanistic_single_failure_recovery",
        "recovery_contract": contract,
        "attempted": True,
        "recovery_start": start.tolist(),
        "selected_from": "single_surrogate_endpoint_recovery"
        if recovered.selected is not None
        else None,
        "status": recovered.status if error is None else "recovery_execution_failed",
        "elapsed_seconds": perf_counter() - started,
        "error": error,
        "result": None if error is not None else recovered.as_dict(),
    }
    atomic_json(payload_path, payload, nonfinite_to_none=True)
    atomic_json(
        marker_path,
        {
            "stage": "mechanistic_failure_recovery",
            "recovery_contract": contract,
            "attempted": True,
            "selected": recovered.selected is not None,
            "artifacts": _artifact_hashes(case_directory, (payload_path, route_path)),
        },
    )
    return (recovered, payload)


def _casewise_reference_contract(
    *,
    source_id: str,
    analysis_id: str,
    case_id: str,
    route: str,
    controls: np.ndarray | None,
    candidate_source_digest: str,
) -> str:
    from surrogate_optimization.runtime.protocols import COMPARISON_PROTOCOL

    digest = sha256()
    digest.update(COMPARISON_PROTOCOL.encode())
    digest.update(source_id.encode())
    digest.update(analysis_id.encode())
    digest.update(case_id.encode())
    digest.update(route.encode())
    digest.update(candidate_source_digest.encode())
    if controls is not None:
        digest.update(np.ascontiguousarray(controls, dtype="<f8").tobytes())
    return digest.hexdigest()


def _scaled_response_errors(
    response: np.ndarray, reference: np.ndarray, scale: np.ndarray
) -> dict[str, float | None]:
    if (
        response.shape != reference.shape
        or response.shape != scale.shape
        or (not np.all(np.isfinite(response)))
        or (not np.all(np.isfinite(reference)))
    ):
        return {"nrmse": None, "nmae": None, "scaled_inf": None}
    scaled = (response - reference) / scale
    return {
        "nrmse": float(np.sqrt(np.mean(scaled**2))),
        "nmae": float(np.mean(np.abs(scaled))),
        "scaled_inf": float(np.max(np.abs(scaled))),
    }


def _run_casewise_route_reference_evaluation(
    case_directory: Path,
    *,
    case_id: str,
    route: str,
    influent: np.ndarray,
    selected: Any | None,
    surrogate_candidate: FinalCandidateRecord | None,
    route_payload: Mapping[str, Any],
    certification_payload: Mapping[str, Any] | None,
    recovery_payload: Mapping[str, Any] | None,
    analysis: AnalysisBundle,
    source_id: str,
    analysis_id: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate one returned decision with the common exact reference model."""
    from surrogate_optimization.config import DECISION_LOWER
    from surrogate_optimization.config import DECISION_UPPER
    from surrogate_optimization.optimization.surrogate import cold_reproject
    from surrogate_optimization.optimization.types import SurrogateCase
    from surrogate_optimization.plant.model import assemble_target
    from surrogate_optimization.plant.operating_point import OperatingPoint
    from surrogate_optimization.runtime.artifacts import atomic_dataframe
    from surrogate_optimization.runtime.artifacts import atomic_json
    from surrogate_optimization.runtime.artifacts import atomic_npz
    from surrogate_optimization.runtime.contracts import _artifact_hashes
    from surrogate_optimization.runtime.contracts import _artifacts_match
    from surrogate_optimization.runtime.contracts import _load_json_object
    from surrogate_optimization.runtime.contracts import file_digest
    from surrogate_optimization.validation.physical import reduce_mechanistic_responses

    if route == "surrogate":
        final = surrogate_candidate
    else:
        final = selected
    controls = None if final is None else np.asarray(final.controls, dtype=np.float64)
    candidate_source = case_directory / (
        "surrogate_local_convergence.json"
        if route == "surrogate"
        else "mechanistic_recovery.json"
        if recovery_payload is not None and recovery_payload.get("attempted")
        else "mechanistic.json"
    )
    if not candidate_source.is_file():
        candidate_source = case_directory / f"{route}.json"
    contract = _casewise_reference_contract(
        source_id=source_id,
        analysis_id=analysis_id,
        case_id=case_id,
        route=route,
        controls=controls,
        candidate_source_digest=file_digest(candidate_source),
    )
    payload_path = case_directory / f"{route}_casewise_reference.json"
    arrays_path = case_directory / f"{route}_casewise_reference.npz"
    physical_path = case_directory / f"{route}_casewise_physical_violations.csv"
    marker_path = case_directory / f"{route}_casewise_reference_complete.json"
    if marker_path.is_file() and payload_path.is_file():
        marker = _load_json_object(marker_path, description="casewise reference marker")
        payload = _load_json_object(
            payload_path, description="casewise reference result"
        )
        if (
            marker.get("reference_contract") == contract
            and payload.get("reference_contract") == contract
            and _artifacts_match(case_directory, marker.get("artifacts", {}))
        ):
            physical = (
                pd.read_csv(physical_path)
                if physical_path.is_file()
                else pd.DataFrame()
            )
            return (payload, physical)
    if final is None:
        arrays_path.unlink(missing_ok=True)
        physical_path.unlink(missing_ok=True)
        payload = {
            "stage": "casewise_exact_common_reference",
            "reference_contract": contract,
            "case": case_id,
            "route": route,
            "candidate_available": False,
            "native_feasible": False,
            "exact_replay_valid": False,
            "comparison_valid": False,
            "status": f"unpaired_{route}_no_candidate",
            "native_status": route_payload.get("status"),
            "recovery": recovery_payload,
        }
        atomic_json(payload_path, payload, nonfinite_to_none=True)
        atomic_json(
            marker_path,
            {
                "stage": "casewise_exact_common_reference",
                "reference_contract": contract,
                "candidate_available": False,
                "artifacts": _artifact_hashes(
                    case_directory, (payload_path, candidate_source)
                ),
            },
        )
        return (payload, pd.DataFrame())
    if (
        controls is None
        or controls.shape != (7,)
        or (not np.all(np.isfinite(controls)))
    ):
        raise RuntimeError(f"{route} selected candidate has invalid controls")
    normalized = (controls - DECISION_LOWER) / (DECISION_UPPER - DECISION_LOWER)
    surrogate_case = SurrogateCase(influent=influent, case_id=case_id)
    raw = np.asarray(analysis.model.predict(controls, influent), dtype=np.float64)
    projection = cold_reproject(
        analysis.surrogate_assets, surrogate_case, normalized, raise_on_failure=False
    )
    projected = np.asarray(projection.state, dtype=np.float64)
    native_full_response = (
        np.asarray(final.projected, dtype=np.float64)
        if route == "surrogate"
        else np.asarray(final.response, dtype=np.float64)
    )
    native_response = (
        native_full_response
        if route == "surrogate"
        else reduce_mechanistic_responses(
            native_full_response, analysis.surrogate_assets.layout.layer_count
        )
    )
    native_objective = float(final.objective)
    expected_response_shape = (analysis.surrogate_assets.layout.state_size,)
    for label, values in (
        ("raw", raw),
        ("projected", projected),
        ("optimizer-native", native_response),
    ):
        if values.shape != expected_response_shape or not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"{route} {label} response is non-finite or has the wrong shape"
            )
    if not np.isfinite(native_objective):
        raise RuntimeError(f"{route} selected candidate has a non-finite objective")
    native_feasible = bool(
        final.feasibility.feasible if route == "surrogate" else final.feasible
    )
    if not native_feasible:
        raise RuntimeError(
            f"{route} selected candidate failed its native feasibility audit"
        )
    retained_reference = case_directory / f"{route}_reference.npz"
    reference_full, state_1, state_2, reference_payload = _casewise_exact_reference(
        controls, influent, analysis, retained_reference_path=retained_reference
    )
    reference = (
        reduce_mechanistic_responses(
            reference_full, analysis.surrogate_assets.layout.layer_count
        )
        if np.all(np.isfinite(reference_full))
        else np.full(expected_response_shape, np.nan)
    )
    exact_replay_valid = bool(
        reference_payload.get("accepted") is True and np.all(np.isfinite(reference))
    )
    reference_valid = bool(
        exact_replay_valid and reference_payload.get("engineering_feasible") is True
    )
    reference_objective = (
        float(reference_payload["objective"]) if exact_replay_valid else None
    )
    reference_components = (
        list(reference_payload["objective_components"]) if exact_replay_valid else None
    )
    response_scale = np.asarray(analysis.model.response_scale, dtype=np.float64)
    local_converged = (
        bool(certification_payload.get("locally_converged"))
        if route == "surrogate" and certification_payload is not None
        else bool(getattr(final, "stationary", False))
    )
    first_order_certified = (
        bool(certification_payload.get("first_order_certified"))
        if route == "surrogate" and certification_payload is not None
        else bool(getattr(final, "stationary", False))
    )
    time_seconds = float(route_payload.get("elapsed_seconds", np.nan))
    payload = {
        "stage": "casewise_exact_common_reference",
        "reference_contract": contract,
        "case": case_id,
        "route": route,
        "candidate_available": True,
        "native_feasible": native_feasible,
        "exact_replay_valid": exact_replay_valid,
        "comparison_valid": reference_valid,
        "status": str(reference_payload.get("status"))
        if reference_valid
        else "exact_valid_engineering_infeasible"
        if exact_replay_valid
        else f"reference_{reference_payload.get('status', 'failed')}",
        "selected_start": int(selected.start_index) if selected is not None else 0,
        "normalized_controls": normalized.tolist(),
        "controls": controls.tolist(),
        "native_status": getattr(final, "status", route_payload.get("status")),
        "native_objective": native_objective,
        "exact_reference_objective": reference_objective,
        "exact_reference_objective_components": reference_components,
        "native_minus_reference_objective": None
        if reference_objective is None
        else native_objective - reference_objective,
        "reference": reference_payload,
        "local_convergence_certified": local_converged,
        "first_order_stationarity_certified": first_order_certified,
        "local_convergence_classification": certification_payload.get("status")
        if route == "surrogate" and certification_payload is not None
        else getattr(final, "status", None),
        "branch_ambiguity_is_qualifier_not_rejection": True,
        "minimum_srt_is_descriptive_not_eligibility_gate": True,
        "projection_accepted": bool(projection.accepted),
        "prediction_error_raw": _scaled_response_errors(raw, reference, response_scale),
        "prediction_error_projected": _scaled_response_errors(
            projected, reference, response_scale
        ),
        "native_model_error": _scaled_response_errors(
            native_response, reference, response_scale
        ),
        "time_metric": "Optimization time",
        "time_unit": "s",
        "time_seconds": time_seconds if np.isfinite(time_seconds) else None,
        "recovery": recovery_payload,
    }
    atomic_json(payload_path, payload, nonfinite_to_none=True)
    atomic_npz(
        arrays_path,
        controls=controls,
        normalized_controls=normalized,
        raw=raw,
        projected=projected,
        optimizer_native=native_response,
        exact_reference=reference,
        optimizer_native_full=native_full_response
        if route == "mechanistic"
        else np.empty(0, dtype=np.float64),
        exact_reference_full=reference_full,
        exact_state_start_1=state_1,
        exact_state_start_2=state_2,
    )
    operating = OperatingPoint(*map(float, controls))
    unavailable_response = np.full(analysis.mechanistic_assets.response_count, np.nan)
    response_1 = (
        assemble_target(
            state_1, operating, influent, analysis.mechanistic_assets.clarifier
        )
        if np.all(np.isfinite(state_1))
        else unavailable_response.copy()
    )
    response_2 = (
        assemble_target(
            state_2, operating, influent, analysis.mechanistic_assets.clarifier
        )
        if np.all(np.isfinite(state_2))
        else unavailable_response.copy()
    )
    response_rows: list[tuple[str, np.ndarray]] = [
        ("raw", raw),
        ("projected", projected),
        ("optimizer_native", native_response),
        (
            "exact_mechanistic_start_1",
            reduce_mechanistic_responses(
                response_1, analysis.surrogate_assets.layout.layer_count
            )
            if np.all(np.isfinite(response_1))
            else np.full(expected_response_shape, np.nan),
        ),
        (
            "exact_mechanistic_start_2",
            reduce_mechanistic_responses(
                response_2, analysis.surrogate_assets.layout.layer_count
            )
            if np.all(np.isfinite(response_2))
            else np.full(expected_response_shape, np.nan),
        ),
    ]
    physical = pd.DataFrame(
        [
            {
                **_physical_record(
                    method, case_id, response, controls, influent, analysis
                ),
                "decision_route": route,
                "response_source": method,
            }
            for method, response in response_rows
        ]
    )
    atomic_dataframe(physical_path, physical)
    atomic_json(
        marker_path,
        {
            "stage": "casewise_exact_common_reference",
            "reference_contract": contract,
            "candidate_available": True,
            "comparison_valid": reference_valid,
            "artifacts": _artifact_hashes(
                case_directory,
                (payload_path, arrays_path, physical_path, candidate_source),
            ),
        },
    )
    return (payload, physical)


def _casewise_comparison_row(
    case_id: str, surrogate: Mapping[str, Any], direct: Mapping[str, Any]
) -> dict[str, Any]:
    from surrogate_optimization.reporting.tables import OBJECTIVE_COMPONENT_NAMES

    reasons: list[str] = []
    if not surrogate.get("candidate_available"):
        reasons.append("surrogate_no_candidate")
    if not direct.get("candidate_available"):
        reasons.append("mechanistic_no_candidate")
    if surrogate.get("candidate_available"):
        if not surrogate.get("exact_replay_valid"):
            reasons.append("surrogate_reference_invalid")
        elif not surrogate.get("comparison_valid"):
            reasons.append("surrogate_exact_engineering_infeasible")
    if direct.get("candidate_available"):
        if not direct.get("exact_replay_valid"):
            reasons.append("mechanistic_reference_invalid")
        elif not direct.get("comparison_valid"):
            reasons.append("mechanistic_exact_engineering_infeasible")
    if surrogate.get("candidate_available") and (not surrogate.get("native_feasible")):
        reasons.append("surrogate_native_infeasible")
    if direct.get("candidate_available") and (not direct.get("native_feasible")):
        reasons.append("mechanistic_native_infeasible")
    eligible = not reasons
    s_objective = surrogate.get("exact_reference_objective")
    d_objective = direct.get("exact_reference_objective")
    delta = (
        float(s_objective) - float(d_objective)
        if eligible and s_objective is not None and (d_objective is not None)
        else None
    )
    symmetric = (
        delta / max(1.0, abs(float(s_objective)), abs(float(d_objective)))
        if delta is not None
        else None
    )
    mechanistic_relative = (
        100.0 * delta / max(abs(float(d_objective)), 1e-12)
        if delta is not None
        else None
    )
    s_controls = surrogate.get("normalized_controls")
    d_controls = direct.get("normalized_controls")
    control_difference = None
    if eligible and isinstance(s_controls, list) and isinstance(d_controls, list):
        difference = np.asarray(s_controls, dtype=float) - np.asarray(
            d_controls, dtype=float
        )
        control_difference = {
            "rms": float(np.sqrt(np.mean(difference**2))),
            "maximum": float(np.max(np.abs(difference))),
        }
    component_differences = None
    s_components = surrogate.get("exact_reference_objective_components")
    d_components = direct.get("exact_reference_objective_components")
    if eligible and isinstance(s_components, list) and isinstance(d_components, list):
        component_differences = (
            np.asarray(s_components, dtype=float)
            - np.asarray(d_components, dtype=float)
        ).tolist()
    surrogate_seconds = surrogate.get("time_seconds")
    mechanistic_seconds = direct.get("time_seconds")
    row = {
        "case": case_id,
        "comparison_eligible": eligible,
        "minimum_srt_is_descriptive_not_eligibility_gate": True,
        "ineligibility_reasons": ";".join(reasons) if reasons else None,
        "surrogate_candidate_available": bool(surrogate.get("candidate_available")),
        "mechanistic_candidate_available": bool(direct.get("candidate_available")),
        "surrogate_local_convergence_certified": bool(
            surrogate.get("local_convergence_certified")
        ),
        "mechanistic_first_order_stationarity_certified": bool(
            direct.get("first_order_stationarity_certified")
        ),
        "both_local_convergence_qualified": bool(
            eligible
            and surrogate.get("local_convergence_certified")
            and direct.get("first_order_stationarity_certified")
        ),
        "surrogate_reference_status": surrogate.get("status"),
        "mechanistic_reference_status": direct.get("status"),
        "surrogate_branch_ambiguous": surrogate.get("reference", {}).get(
            "branch_ambiguous"
        ),
        "mechanistic_branch_ambiguous": direct.get("reference", {}).get(
            "branch_ambiguous"
        ),
        "J_S_reference": s_objective,
        "J_M_reference": d_objective,
        "delta_J_S_minus_M": delta,
        "symmetric_relative_difference": symmetric,
        "surrogate_penalty_percent_relative_to_direct": mechanistic_relative,
        "control_rms_difference": None
        if control_difference is None
        else control_difference["rms"],
        "control_maximum_difference": None
        if control_difference is None
        else control_difference["maximum"],
        "objective_component_differences": component_differences,
        "time_metric": "Optimization time",
        "time_unit": "s",
        "surrogate_time_seconds": surrogate_seconds,
        "mechanistic_time_seconds": mechanistic_seconds,
    }
    if component_differences is not None:
        row.update(
            {
                f"delta_component_{name}": value
                for name, value in zip(
                    OBJECTIVE_COMPONENT_NAMES, component_differences, strict=True
                )
            }
        )
    return row
