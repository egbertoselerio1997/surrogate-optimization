"""Workflow optimization."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.workflow.types import AnalysisBundle
    from surrogate_optimization.config import StudyProfile
from pathlib import Path
from typing import Any
from typing import Mapping
import json
import numpy as np
import pandas as pd


def run_optimization_stage(
    *,
    run: Path,
    profile: StudyProfile,
    design: Mapping[str, object],
    development_responses: np.ndarray,
    holdout_responses: np.ndarray,
    analysis: AnalysisBundle,
    source_files: Mapping[str, str],
) -> bool:
    from surrogate_optimization.runtime.contracts import _artifacts_match

    "Run searches and compare selected controls on one exact reference.\n\n    The frozen post-selection holdout remains reserved for descriptive\n    surrogate assessment. Smooth/reference equivalence is not rerun over the\n    rows. Instead, every available nominal/robustness decision is evaluated\n    by the same exact nonsmooth mechanistic equations.  Search failures and\n    unresolved convergence certificates are recorded casewise and never stop\n    the remaining scientific comparisons.\n    "
    from surrogate_optimization.config import DECISION_NAMES
    from surrogate_optimization.optimization.certification import (
        _run_surrogate_certification,
    )
    from surrogate_optimization.optimization.mechanistic import CONTINUATION_SCHEDULE
    from surrogate_optimization.optimization.surrogate import EXPRESSION_GAP
    from surrogate_optimization.optimization.surrogate import (
        build_surrogate_expression_graph,
    )
    from surrogate_optimization.optimization.types import SurrogateSolverSettings
    from surrogate_optimization.plant.definitions import NOMINAL_INFLUENT
    from surrogate_optimization.reporting.timing import (
        _run_robustness_case_timing_aggregation,
    )
    from surrogate_optimization.runtime.artifacts import atomic_dataframe
    from surrogate_optimization.runtime.artifacts import atomic_json
    from surrogate_optimization.runtime.checkpoints import _case_contract_id
    from surrogate_optimization.runtime.checkpoints import _run_mechanistic_route
    from surrogate_optimization.runtime.checkpoints import _run_surrogate_route
    from surrogate_optimization.runtime.contracts import _artifact_hashes
    from surrogate_optimization.runtime.contracts import _casewise_artifact_source_id
    from surrogate_optimization.runtime.contracts import assert_source_unchanged
    from surrogate_optimization.runtime.contracts import source_digest
    from surrogate_optimization.runtime.protocols import COMPARISON_PROTOCOL
    from surrogate_optimization.runtime.protocols import OPTIMIZATION_PROTOCOL
    from surrogate_optimization.validation.reference import _casewise_comparison_row
    from surrogate_optimization.validation.reference import (
        _run_casewise_route_reference_evaluation,
    )
    from surrogate_optimization.validation.reference import (
        _run_mechanistic_failure_recovery,
    )
    from surrogate_optimization.workflow.assessment import _assessment_binding
    from surrogate_optimization.workflow.assessment import (
        assessment_gate_allows_optimization,
    )
    from surrogate_optimization.workflow.study import _write_state

    if profile.robustness_count != 10 or profile.layer_count != 10:
        raise RuntimeError(
            "optimization requires ten robustness cases and ten clarifier layers"
        )
    if not assessment_gate_allows_optimization(analysis.passed):
        raise RuntimeError("optimization cannot bypass the enforced admission gate")
    source_id = source_digest(source_files)
    casewise_source_id = _casewise_artifact_source_id(run, source_id)
    analysis_id = _assessment_binding(design, development_responses, holdout_responses)
    marker_path = run / "optimization/optimization_complete.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("source_digest") != source_id
            or marker.get("input_digest") != analysis_id
        ):
            raise RuntimeError("optimization completion contract differs")
        if not _artifacts_match(run, marker.get("artifacts", {})):
            raise RuntimeError("completed optimization artifacts changed")
        return bool(marker["scientific_validation_passed"])
    case_inputs = [
        ("nominal", np.asarray(NOMINAL_INFLUENT, dtype=float)),
        *(
            (f"robustness_{index + 1:02d}", np.asarray(row, dtype=float))
            for index, row in enumerate(np.asarray(design["robustness_influents"]))
        ),
    ]
    if len(case_inputs) != 11:
        raise RuntimeError("the full study requires nominal plus ten robustness cases")
    development_controls = np.asarray(design["development_controls"], dtype=float)
    development_influents = np.asarray(design["development_influents"], dtype=float)
    selected_physical_frames: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    route_statuses: list[dict[str, Any]] = []
    shared_surrogate_problem: Any | None = None
    for case_number, (case_id, influent) in enumerate(case_inputs, start=1):
        _write_state(
            run,
            "casewise_common_reference",
            "running",
            case=case_id,
            completed_cases=case_number - 1,
            total_cases=len(case_inputs),
        )
        case_directory = run / "optimization" / case_id
        case_directory.mkdir(parents=True, exist_ok=True)
        if shared_surrogate_problem is None:
            shared_surrogate_problem = build_surrogate_expression_graph(
                analysis.surrogate_assets,
                EXPRESSION_GAP,
                settings=SurrogateSolverSettings(maximum_wall_time=None),
                name="study_surrogate_exact_qp_expressions",
            )
        surrogate, surrogate_payload = _run_surrogate_route(
            case_directory,
            case_id=case_id,
            influent=influent,
            assets=analysis.surrogate_assets,
            source_id=casewise_source_id,
            analysis_id=analysis_id,
            problem=shared_surrogate_problem,
        )
        direct, mechanistic_payload = _run_mechanistic_route(
            case_directory,
            case_id=case_id,
            influent=influent,
            assets=analysis.mechanistic_assets,
            development_controls=development_controls,
            development_influents=development_influents,
            development_responses=development_responses,
            source_id=casewise_source_id,
            analysis_id=analysis_id,
        )
        surrogate_candidate, certification = _run_surrogate_certification(
            case_directory,
            case_id=case_id,
            influent=influent,
            result=surrogate,
            analysis=analysis,
            source_id=casewise_source_id,
            analysis_id=analysis_id,
            problem=shared_surrogate_problem,
        )
        mechanistic_for_comparison, recovery = _run_mechanistic_failure_recovery(
            case_directory,
            case_id=case_id,
            influent=influent,
            result=direct,
            surrogate_candidate=surrogate_candidate
            if certification.get("locally_converged") is True
            else None,
            assets=analysis.mechanistic_assets,
            development_controls=development_controls,
            development_influents=development_influents,
            development_responses=development_responses,
            source_id=casewise_source_id,
            analysis_id=analysis_id,
        )
        surrogate_evaluation, surrogate_physical = (
            _run_casewise_route_reference_evaluation(
                case_directory,
                case_id=case_id,
                route="surrogate",
                influent=influent,
                selected=surrogate.selected,
                surrogate_candidate=surrogate_candidate,
                route_payload=surrogate_payload,
                certification_payload=certification,
                recovery_payload=None,
                analysis=analysis,
                source_id=casewise_source_id,
                analysis_id=analysis_id,
            )
        )
        mechanistic_evaluation, mechanistic_physical = (
            _run_casewise_route_reference_evaluation(
                case_directory,
                case_id=case_id,
                route="mechanistic",
                influent=influent,
                selected=mechanistic_for_comparison.selected,
                surrogate_candidate=None,
                route_payload=mechanistic_payload,
                certification_payload=None,
                recovery_payload=recovery,
                analysis=analysis,
                source_id=casewise_source_id,
                analysis_id=analysis_id,
            )
        )
        for evaluation in (surrogate_evaluation, mechanistic_evaluation):
            reference = evaluation.get("reference", {})
            raw_error = evaluation.get("prediction_error_raw", {})
            projected_error = evaluation.get("prediction_error_projected", {})
            native_error = evaluation.get("native_model_error", {})
            normalized_controls = evaluation.get("normalized_controls")
            controls = evaluation.get("controls")
            row = {
                "case": case_id,
                "route": evaluation.get("route"),
                "candidate_available": evaluation.get("candidate_available"),
                "native_feasible": evaluation.get("native_feasible"),
                "exact_replay_valid": evaluation.get("exact_replay_valid"),
                "comparison_valid": evaluation.get("comparison_valid"),
                "status": evaluation.get("status"),
                "native_status": evaluation.get("native_status"),
                "native_objective": evaluation.get("native_objective"),
                "exact_reference_objective": evaluation.get(
                    "exact_reference_objective"
                ),
                "native_minus_reference_objective": evaluation.get(
                    "native_minus_reference_objective"
                ),
                "local_convergence_certified": evaluation.get(
                    "local_convergence_certified"
                ),
                "first_order_stationarity_certified": evaluation.get(
                    "first_order_stationarity_certified"
                ),
                "local_convergence_classification": evaluation.get(
                    "local_convergence_classification"
                ),
                "reference_status": reference.get("status"),
                "reference_source": reference.get("source"),
                "reference_start_1_accepted": reference.get("start_1_accepted"),
                "reference_start_2_accepted": reference.get("start_2_accepted"),
                "reference_scaled_root_difference_generation": reference.get(
                    "scaled_root_difference_generation"
                ),
                "reference_scaled_root_difference_state": reference.get(
                    "scaled_root_difference_state"
                ),
                "reference_branch_agreement": reference.get("branch_agreement"),
                "reference_branch_ambiguous": reference.get("branch_ambiguous"),
                "reference_minimum_normalized_branch_margin": reference.get(
                    "minimum_normalized_branch_margin"
                ),
                "reference_physical_stability_accepted": reference.get(
                    "physical_stability_accepted"
                ),
                "reference_engineering_feasible": reference.get("engineering_feasible"),
                "raw_reference_nrmse": raw_error.get("nrmse"),
                "raw_reference_nmae": raw_error.get("nmae"),
                "raw_reference_scaled_inf": raw_error.get("scaled_inf"),
                "projected_reference_nrmse": projected_error.get("nrmse"),
                "projected_reference_nmae": projected_error.get("nmae"),
                "projected_reference_scaled_inf": projected_error.get("scaled_inf"),
                "optimizer_native_reference_nrmse": native_error.get("nrmse"),
                "optimizer_native_reference_nmae": native_error.get("nmae"),
                "optimizer_native_reference_scaled_inf": native_error.get("scaled_inf"),
                "time_metric": "Optimization time",
                "time_unit": "s",
                "time_seconds": evaluation.get("time_seconds"),
            }
            if isinstance(normalized_controls, list) and len(normalized_controls) == 7:
                row.update(
                    {
                        f"normalized_{name}": value
                        for name, value in zip(
                            DECISION_NAMES, normalized_controls, strict=True
                        )
                    }
                )
            if isinstance(controls, list) and len(controls) == 7:
                row.update(
                    {
                        name: value
                        for name, value in zip(DECISION_NAMES, controls, strict=True)
                    }
                )
            reference_rows.append(row)
        comparison = _casewise_comparison_row(
            case_id, surrogate_evaluation, mechanistic_evaluation
        )
        comparison_rows.append(comparison)
        case_comparison_path = case_directory / "common_reference_comparison.json"
        atomic_json(case_comparison_path, comparison, nonfinite_to_none=True)
        case_frames = [
            frame
            for frame in (surrogate_physical, mechanistic_physical)
            if not frame.empty
        ]
        case_violations = (
            pd.concat(case_frames, ignore_index=True, sort=False)
            if case_frames
            else pd.DataFrame()
        )
        if not case_violations.empty:
            selected_physical_frames.append(case_violations)
        new_artifacts = (
            case_directory / "surrogate_local_convergence_complete.json",
            case_directory / "surrogate_casewise_reference_complete.json",
            case_directory / "mechanistic_casewise_reference_complete.json",
            case_comparison_path,
        )
        case_contract = _case_contract_id(source_id, analysis_id, case_id, influent)
        route_rows = [
            {
                "case": case_id,
                "route": "surrogate",
                "primary_status": surrogate.status,
                "selected": surrogate_evaluation.get("candidate_available"),
                "comparison_valid": surrogate_evaluation.get("comparison_valid"),
                "locally_converged": surrogate_evaluation.get(
                    "local_convergence_certified"
                ),
                "first_order_stationary": surrogate_evaluation.get(
                    "first_order_stationarity_certified"
                ),
                "primary_attempts": 1,
                "recovery_attempts": 0,
            },
            {
                "case": case_id,
                "route": "mechanistic",
                "primary_status": direct.status,
                "selected": mechanistic_evaluation.get("candidate_available"),
                "comparison_valid": mechanistic_evaluation.get("comparison_valid"),
                "locally_converged": mechanistic_evaluation.get(
                    "local_convergence_certified"
                ),
                "first_order_stationary": mechanistic_evaluation.get(
                    "first_order_stationarity_certified"
                ),
                "primary_attempts": 1,
                "recovery_attempts": int(bool(recovery.get("attempted"))),
            },
        ]
        route_statuses.extend(route_rows)
        assert_source_unchanged(source_files)
        atomic_json(
            case_directory / "casewise_comparison_complete.json",
            {
                "stage": "casewise_exact_common_reference",
                "case": case_id,
                "case_contract": case_contract,
                "routes": route_rows,
                "comparison_eligible": comparison["comparison_eligible"],
                "artifacts": _artifact_hashes(run, new_artifacts),
            },
        )
    comparison_frame = pd.DataFrame(comparison_rows)
    reference_frame = pd.DataFrame(reference_rows)
    atomic_dataframe(
        run / "metrics" / "case_common_reference_comparison.csv", comparison_frame
    )
    atomic_dataframe(
        run / "metrics" / "selected_candidate_reference_evaluation.csv", reference_frame
    )
    _write_state(
        run,
        "robustness_timing_aggregation",
        "running",
        case_count=profile.robustness_count,
        nominal_case_included=False,
    )
    _run_robustness_case_timing_aggregation(
        run, source_files=source_files, analysis_id=analysis_id
    )
    selected_physical = (
        pd.concat(selected_physical_frames, ignore_index=True, sort=False)
        if selected_physical_frames
        else pd.DataFrame(columns=("case", "method"))
    )
    atomic_dataframe(
        run / "metrics" / "selected_response_physical_audit.csv", selected_physical
    )
    assessment_physical = pd.read_csv(
        run / "metrics" / "physical_violations_assessment.csv"
    )
    combined_frames = []
    for scope, frame in (
        ("post_selection_holdout", assessment_physical),
        ("selected_decision_common_reference", selected_physical),
    ):
        item = frame.copy()
        item.insert(0, "analysis_scope", scope)
        combined_frames.append(item)
    atomic_dataframe(
        run / "metrics" / "physical_violations_all_analysis.csv",
        pd.concat(combined_frames, ignore_index=True, sort=False),
    )
    available_reference = reference_frame["candidate_available"].fillna(False)
    exact_replay_valid = reference_frame.loc[
        available_reference, "exact_replay_valid"
    ].fillna(False)
    reference_valid = reference_frame.loc[
        available_reference, "comparison_valid"
    ].fillna(False)
    surrogate_certified = reference_frame.loc[
        reference_frame["route"].eq("surrogate") & available_reference,
        "local_convergence_certified",
    ].fillna(False)
    scientific_passed = bool(
        len(reference_frame) == 2 * len(case_inputs)
        and int(available_reference.sum()) == 2 * len(case_inputs)
        and bool(reference_valid.all())
        and (len(surrogate_certified) == len(case_inputs))
        and bool(surrogate_certified.all())
        and (int(comparison_frame["comparison_eligible"].sum()) == len(case_inputs))
    )
    selected_count = int(available_reference.sum())
    paired_count = int(comparison_frame["comparison_eligible"].sum())
    final_status_path = run / "optimization" / "final_status.json"
    atomic_json(
        final_status_path,
        {
            "case_count": len(case_inputs),
            "route_count": len(route_statuses),
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
            "validation_protocol": COMPARISON_PROTOCOL,
            "required_attempts_per_route": 1,
            "required_starts_per_route": 1,
            "surrogate_ipopt_continuation_stage_count": 0,
            "mechanistic_smoothing_continuation_stage_count": len(
                CONTINUATION_SCHEDULE
            ),
            "wall_time_ceiling": None,
            "untouched_holdout_equivalence_executed": False,
            "selected_decision_count": selected_count,
            "exact_reference_valid_selected_decision_count": int(
                exact_replay_valid.sum()
            ),
            "exact_reference_engineering_feasible_selected_decision_count": int(
                reference_valid.sum()
            ),
            "paired_common_reference_case_count": paired_count,
            "surrogate_locally_converged_count": int(surrogate_certified.sum()),
            "casewise_reference_validation_passed": bool(
                selected_count == 2 * len(case_inputs)
                and len(exact_replay_valid) == 2 * len(case_inputs)
                and exact_replay_valid.all()
            ),
            "all_pairs_comparison_eligible": paired_count == len(case_inputs),
            "scientific_validation_passed": scientific_passed,
            "routes": route_statuses,
        },
        nonfinite_to_none=True,
    )
    assert_source_unchanged(source_files)
    final_paths = (
        run / "metrics" / "selected_response_physical_audit.csv",
        run / "metrics" / "physical_violations_all_analysis.csv",
        run / "metrics" / "case_common_reference_comparison.csv",
        run / "metrics" / "selected_candidate_reference_evaluation.csv",
        final_status_path,
        run / "metrics" / "robustness_case_timing_complete.json",
        run / "metrics" / "robustness_case_timing.csv",
        run / "metrics" / "robustness_case_timing_summary.json",
        *(
            run / "optimization" / case_id / "casewise_comparison_complete.json"
            for case_id, _ in case_inputs
        ),
    )
    atomic_json(
        run / "optimization" / "optimization_complete.json",
        {
            "stage": "study_optimization_replay_reporting",
            "source_digest": source_id,
            "input_digest": analysis_id,
            "case_count": 11,
            "route_count": 22,
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
            "required_attempts_per_route": 1,
            "selected_decision_count": selected_count,
            "paired_common_reference_case_count": paired_count,
            "untouched_holdout_equivalence_executed": False,
            "scientific_validation_passed": scientific_passed,
            "artifacts": _artifact_hashes(run, final_paths),
        },
    )
    return scientific_passed
