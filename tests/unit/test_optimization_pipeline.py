from __future__ import annotations
import surrogate_optimization.workflow.types as module_workflow_types
import surrogate_optimization.workflow.optimization as module_workflow_optimization
import surrogate_optimization.validation.reference as module_validation_reference
import surrogate_optimization.runtime.protocols as module_runtime_protocols
import surrogate_optimization.runtime.contracts as module_runtime_contracts
import surrogate_optimization.runtime.checkpoints as module_runtime_checkpoints
import surrogate_optimization.runtime.artifacts as module_runtime_artifacts
import surrogate_optimization.reporting.timing as module_reporting_timing
import surrogate_optimization.reporting.tables as module_reporting_tables
import surrogate_optimization.plant.model as module_plant_model
import surrogate_optimization.optimization.surrogate as module_optimization_surrogate
import surrogate_optimization.optimization.mechanistic as module_optimization_mechanistic
import surrogate_optimization.optimization.certification as module_optimization_certification
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from surrogate_optimization.config import PRODUCTION_PROFILE
from surrogate_optimization.config import DECISION_LOWER
from surrogate_optimization.config import DECISION_UPPER
from surrogate_optimization.validation.physical import reduce_mechanistic_responses
from surrogate_optimization.plant.definitions import INFLUENT_LOWER
from surrogate_optimization.plant.definitions import INFLUENT_UPPER
from surrogate_optimization.surrogate.projection import NetworkLayout
from surrogate_optimization.surrogate.projection import ProjectionDiagnostics
from surrogate_optimization.surrogate.projection import ProjectionResult
from surrogate_optimization.optimization.mechanistic import BranchClassification
from surrogate_optimization.optimization.mechanistic import (
    ContinuationStageResult as DirectStage,
)
from surrogate_optimization.optimization.mechanistic import MechanisticRouteResult
from surrogate_optimization.optimization.mechanistic import MechanisticStartResult
from surrogate_optimization.optimization.mechanistic import KKTDiagnostics
from surrogate_optimization.optimization.surrogate import EXACT_QP_CENTER_START
from surrogate_optimization.optimization.surrogate import EXACT_QP_SINGLE_START_PROTOCOL
from surrogate_optimization.optimization.types import FeasibilityRecord
from surrogate_optimization.optimization.types import FinalCandidateRecord
from surrogate_optimization.optimization.types import OuterRefinementRecord
from surrogate_optimization.optimization.types import StationarityRecord
from surrogate_optimization.optimization.types import SurrogateRouteResult
from surrogate_optimization.optimization.types import SurrogateStartResult

RESPONSE_COUNT = PRODUCTION_PROFILE.surrogate_response_count
MECHANISTIC_RESPONSE_COUNT = PRODUCTION_PROFILE.mechanistic_response_count
REDUCED_STATE_COUNT = 5 * 20 + PRODUCTION_PROFILE.layer_count


def _branch() -> BranchClassification:
    return BranchClassification((), (), (), (), (), False, 1.0)


def _projection(response: np.ndarray) -> ProjectionResult:
    diagnostics = ProjectionDiagnostics(
        status="solved",
        status_value=1,
        iterations=4,
        equality_rank_tolerance=1e-12,
        equality_smallest_singular_value=1.0,
        equality_condition_number=1.0,
        equality_residual=1e-12,
        inequality_residual=0.0,
        nonnegativity_residual=0.0,
        dual_feasibility_residual=0.0,
        stationarity_residual=1e-12,
        complementarity_residual=1e-12,
        retried_cold=False,
        active_inequality_count=2,
        multipliers_reconstructed=True,
    )
    return ProjectionResult(
        state=np.asarray(response, dtype=float),
        displacement=np.zeros(RESPONSE_COUNT),
        equality_multipliers=np.zeros(1),
        inequality_multipliers=np.zeros(1),
        inequality_slack=np.ones(1),
        diagnostics=diagnostics,
        accepted=True,
    )


def _surrogate_start(index: int, normalized: np.ndarray) -> SurrogateStartResult:
    raw = np.full(RESPONSE_COUNT, 1.0 + 0.01 * index)
    projected = np.full(RESPONSE_COUNT, 2.0 + 0.01 * index)
    controls = DECISION_LOWER + (DECISION_UPPER - DECISION_LOWER) * normalized
    final = FinalCandidateRecord(
        normalized_controls=normalized.copy(),
        controls=controls,
        raw=raw,
        projected=projected,
        displacement=projected - raw,
        objective=1.0 + index,
        objective_components=np.full(6, 1.0 / 6.0),
        engineering_rows=np.full(7, -1.0),
        engineering_quantities=np.ones(7),
        trust_rows=np.full(4, -1.0),
        trust_values=np.zeros(4),
        projection=_projection(projected),
        feasibility=FeasibilityRecord(
            finite=True,
            cold_projection=True,
            projection_accepted=True,
            control_bound_residual=0.0,
            engineering_residual=0.0,
            trust_residual=0.0,
            maximum_upper_residual=0.0,
            feasible=True,
            projection_reproduction_residual=1e-12,
            projection_reproduction_passed=True,
        ),
        stationarity=StationarityRecord(
            classification="first_order_kkt_stationary_feasible",
            resolved=True,
            stationary=True,
            lower_qp_kkt_passed=True,
            upper_stationarity_residual=1e-12,
            reason="mocked independent audit",
        ),
        status="validated_stationary",
    )
    lower_active_set = {"stable": True, "active_count": 2}
    upper_kkt = {"feasible": True, "stationary": True}
    final = FinalCandidateRecord(
        **{
            **final.__dict__,
            "lower_active_set": lower_active_set,
            "upper_kkt": upper_kkt,
        }
    )
    return SurrogateStartResult(
        start_index=index,
        initial_normalized_controls=normalized.copy(),
        stages=(),
        outer_refinement=OuterRefinementRecord(
            attempted=True,
            solver_success=True,
            status="success",
            iterations=2,
            evaluations=3,
            elapsed_seconds=0.01,
            initial_objective=final.objective + 0.1,
            final_objective=final.objective,
            projection_reproduction_residual=1e-12,
            projection_reproduction_passed=True,
            lower_active_set=lower_active_set,
            upper_kkt=upper_kkt,
        ),
        final=final,
        status="validated_stationary",
        resume_contract="mock-resume-contract",
        protocol=EXACT_QP_SINGLE_START_PROTOCOL,
    )


def _mechanistic_start(index: int, normalized: np.ndarray) -> MechanisticStartResult:
    controls = DECISION_LOWER + (DECISION_UPPER - DECISION_LOWER) * normalized
    stages = tuple(
        (
            DirectStage(
                epsilon=epsilon,
                receiver_half_width=half_width,
                status="Solve_Succeeded",
                solver_success=True,
                elapsed_seconds=0.01,
                iterations=2,
                primal=np.zeros(8),
                feasible=True,
            )
            for epsilon, half_width in ((1e-06, 10.0), (1e-07, 3.0), (1e-08, 1.0))
        )
    )
    kkt = KKTDiagnostics(
        finite=True,
        equality_residual=1e-12,
        inequality_residual=0.0,
        bound_residual=0.0,
        stationarity_residual=1e-12,
        dual_feasibility_residual=0.0,
        complementarity_residual=0.0,
        active_inequality_count=2,
        feasible=True,
        stationary=True,
        equality_multipliers=np.zeros(1),
        inequality_multipliers=np.zeros(1),
        lower_bound_multipliers=np.zeros(1),
        upper_bound_multipliers=np.zeros(1),
    )
    return MechanisticStartResult(
        start_index=index,
        initial_normalized_controls=normalized.copy(),
        resume_contract="mock-resume-contract",
        nearest_development_row=0,
        stages=stages,
        objective=2.0 + index,
        normalized_controls=normalized.copy(),
        controls=controls,
        state=np.ones(REDUCED_STATE_COUNT),
        feed_tss=100.0,
        response=np.full(MECHANISTIC_RESPONSE_COUNT, 3.0 + 0.01 * index),
        engineering=np.ones(11),
        objective_components=np.full(6, 1.0 / 6.0),
        branch=_branch(),
        kkt=kkt,
        feasible=True,
        stationary=True,
        status="first_order_kkt_stationary_feasible",
        error=None,
    )


def _surrogate_solver(*_args: object, **kwargs: object) -> SurrogateRouteResult:
    normalized = np.asarray(EXACT_QP_CENTER_START, dtype=float)
    completed = kwargs.get("completed_result")
    callback = kwargs["progress_callback"]
    if completed is None:
        result = _surrogate_start(0, normalized)
        callback(result)
    else:
        result = completed
    return SurrogateRouteResult(
        (result,),
        result,
        "selected_stationary",
        protocol=EXACT_QP_SINGLE_START_PROTOCOL,
    )


def _mechanistic_solver(*_args: object, **kwargs: object) -> MechanisticRouteResult:
    starts = np.asarray(kwargs["starts"], dtype=float)
    completed = dict(kwargs.get("completed_starts") or {})
    callback = kwargs["progress_callback"]
    for index, normalized in enumerate(starts):
        if index in completed:
            continue
        result = _mechanistic_start(index, normalized)
        completed[index] = result
        callback(result)
    ordered = tuple((completed[index] for index in range(len(starts))))
    return MechanisticRouteResult(ordered, ordered[0], "selected_stationary")


def _fixture() -> tuple[
    dict[str, np.ndarray], np.ndarray, np.ndarray, module_workflow_types.AnalysisBundle
]:
    midpoint = 0.5 * (DECISION_LOWER + DECISION_UPPER)
    influent = 0.5 * (INFLUENT_LOWER + INFLUENT_UPPER)
    development_candidate_count, holdout_candidate_count = (3, 2)
    design = {
        "development_controls": np.tile(midpoint, (development_candidate_count, 1)),
        "development_influents": np.tile(influent, (development_candidate_count, 1)),
        "holdout_controls": np.tile(midpoint, (holdout_candidate_count, 1)),
        "holdout_influents": np.tile(influent, (holdout_candidate_count, 1)),
        "robustness_influents": np.vstack(
            [influent + 0.001 * index for index in range(10)]
        ),
    }
    development_responses = np.ones(
        (development_candidate_count, MECHANISTIC_RESPONSE_COUNT)
    )
    holdout_responses = np.ones((holdout_candidate_count, MECHANISTIC_RESPONSE_COUNT))
    layout = NetworkLayout(layer_count=PRODUCTION_PROFILE.layer_count)

    def predict(_theta: np.ndarray, _influent: np.ndarray) -> np.ndarray:
        return np.ones(RESPONSE_COUNT)

    model = SimpleNamespace(
        response_count=MECHANISTIC_RESPONSE_COUNT,
        response_scale=np.ones(RESPONSE_COUNT),
        predict=predict,
    )
    surrogate_assets = SimpleNamespace(
        layout=layout,
        row_scales=SimpleNamespace(equality=np.ones(1), inequality=np.ones(1)),
    )
    mechanistic_assets = SimpleNamespace(
        response_count=RESPONSE_COUNT,
        state_count=REDUCED_STATE_COUNT,
        state_scale=np.ones(REDUCED_STATE_COUNT),
        feed_scale=1.0,
    )
    analysis = module_workflow_types.AnalysisBundle(
        passed=True,
        model=model,
        mechanistic_assets=mechanistic_assets,
        surrogate_assets=surrogate_assets,
        assessment=None,
        gate={"passed": True},
    )
    return (design, development_responses, holdout_responses, analysis)


def _mock_timing(run: Path, *_args: object, **_kwargs: object) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "case": f"robustness_{case:02d}",
                "route": route,
                "metric": "Time",
                "unit": "s",
                "time_seconds": 1.0,
            }
            for case in range(1, 11)
            for route in ("surrogate", "mechanistic")
        ]
    )
    module_runtime_artifacts.atomic_dataframe(
        run / "metrics/robustness_case_timing.csv", frame
    )
    module_runtime_artifacts.atomic_json(
        run / "metrics/robustness_case_timing_summary.json",
        {
            "protocol": module_runtime_protocols.TIMING_PROTOCOL,
            "metric": "Time",
            "unit": "s",
            "robustness_case_count": 10,
        },
    )
    module_runtime_artifacts.atomic_json(
        run / "metrics/robustness_case_timing_complete.json",
        {"stage": "robustness_case_timing_aggregation", "case_count": 10},
    )
    return frame


class ProductionOptimizationHookTests(unittest.TestCase):
    def test_timing_is_aggregated_from_ten_robustness_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "metrics").mkdir()
            for index in range(1, 11):
                case_id = f"robustness_{index:02d}"
                case = run / "optimization" / case_id
                case.mkdir(parents=True)
                module_runtime_artifacts.atomic_json(
                    case / "surrogate.json",
                    {"elapsed_seconds": float(index), "status": "selected"},
                )
                module_runtime_artifacts.atomic_json(
                    case / "mechanistic.json",
                    {"elapsed_seconds": float(index + 10), "status": "selected"},
                )
                module_runtime_artifacts.atomic_json(
                    case / "surrogate_local_convergence.json",
                    {"certificate": {"elapsed_seconds": 0.5}},
                )
                for route in ("surrogate", "mechanistic"):
                    module_runtime_artifacts.atomic_json(
                        case / f"{route}_casewise_reference.json",
                        {
                            "candidate_available": True,
                            "comparison_valid": True,
                            "recovery": {"attempted": False},
                        },
                    )
                comparison = case / "common_reference_comparison.json"
                module_runtime_artifacts.atomic_json(comparison, {"case": case_id})
                module_runtime_artifacts.atomic_json(
                    case / "casewise_comparison_complete.json",
                    {
                        "case": case_id,
                        "artifacts": {
                            comparison.relative_to(
                                run
                            ).as_posix(): module_runtime_contracts.file_digest(
                                comparison
                            )
                        },
                    },
                )
            frame = module_reporting_timing._run_robustness_case_timing_aggregation(
                run, source_files={"unit": "source"}, analysis_id="analysis"
            )
            self.assertEqual(len(frame), 20)
            means = frame.groupby("route")["time_seconds"].mean()
            self.assertAlmostEqual(means["surrogate"], 5.5)
            self.assertAlmostEqual(means["mechanistic"], 15.5)
            self.assertTrue(frame["metric"].eq("Time").all())
            self.assertTrue(frame["unit"].eq("s").all())
            self.assertNotIn("complete_optimization_seconds", frame)
            summary = json.loads(
                (run / "metrics/robustness_case_timing_summary.json").read_text()
            )
            self.assertEqual(summary["robustness_case_count"], 10)
            self.assertEqual(summary["metric"], "Time")
            self.assertEqual(summary["unit"], "s")

    def test_full_hook_runs_11_cases_and_cross_evaluates_both_routes(self) -> None:
        self.assertEqual(
            module_runtime_protocols.COMPARISON_PROTOCOL,
            "casewise_exact_common_reference_no_minimum_srt",
        )
        design, development_responses, holdout_responses, analysis = _fixture()
        surrogate_completed_on_entry: list[set[int]] = []
        mechanistic_completed_on_entry: list[set[int]] = []

        def surrogate_solver(*args: object, **kwargs: object):
            surrogate_completed_on_entry.append(
                set() if kwargs.get("completed_result") is None else {0}
            )
            return _surrogate_solver(*args, **kwargs)

        def mechanistic_solver(*args: object, **kwargs: object):
            mechanistic_completed_on_entry.append(
                set((kwargs.get("completed_starts") or {}).keys())
            )
            return _mechanistic_solver(*args, **kwargs)

        def certify(
            case_directory: Path, *, result: SurrogateRouteResult, **_kwargs: object
        ):
            self.assertIsNotNone(result.selected)
            assert result.selected is not None and result.selected.final is not None
            candidate = result.selected.final
            payload = {
                "selected": True,
                "locally_converged": True,
                "first_order_certified": True,
                "status": "exact_active_set_kkt",
                "candidate": candidate.as_dict(),
                "certificate": {"elapsed_seconds": 0.01},
            }
            module_runtime_artifacts.atomic_json(
                case_directory / "surrogate_local_convergence_complete.json",
                {"locally_converged": True},
            )
            return (candidate, payload)

        def casewise_reference(
            case_directory: Path,
            *,
            case_id: str,
            route: str,
            selected: object,
            surrogate_candidate: FinalCandidateRecord | None,
            route_payload: dict[str, object],
            **_kwargs: object,
        ):
            candidate = surrogate_candidate if route == "surrogate" else selected
            self.assertIsNotNone(candidate)
            assert candidate is not None
            objective = float(candidate.objective)
            normalized = np.asarray(candidate.normalized_controls, dtype=float)
            payload = {
                "case": case_id,
                "route": route,
                "candidate_available": True,
                "native_feasible": True,
                "exact_replay_valid": True,
                "comparison_valid": True,
                "status": "valid_interior",
                "native_status": candidate.status,
                "native_objective": objective,
                "exact_reference_objective": objective,
                "native_minus_reference_objective": 0.0,
                "normalized_controls": normalized.tolist(),
                "exact_reference_objective_components": np.full(6, 1.0 / 6.0).tolist(),
                "local_convergence_certified": True,
                "first_order_stationarity_certified": True,
                "time_metric": "Time",
                "time_unit": "s",
                "time_seconds": route_payload["elapsed_seconds"],
                "reference": {
                    "status": "valid_interior",
                    "branch_ambiguous": False,
                    "engineering_feasible": True,
                },
            }
            physical = pd.DataFrame(
                [
                    {
                        "case": f"{case_id}:{route}",
                        "method": method,
                        "decision_route": route,
                        "response_source": method,
                        "audit_available": True,
                        "mass_conservation_violation_max": 1e-12,
                        "mass_conservation_violation_count": 0,
                        "nonnegativity_violation_max": 0.0,
                        "nonnegativity_violation_count": 0,
                    }
                    for method in (
                        "raw",
                        "projected",
                        "optimizer_native",
                        "exact_mechanistic_start_1",
                        "exact_mechanistic_start_2",
                    )
                ]
            )
            module_runtime_artifacts.atomic_json(
                case_directory / f"{route}_casewise_reference_complete.json",
                {"candidate_available": True, "comparison_valid": True},
            )
            return (payload, physical)

        def report(
            run: Path, *, output_directory: Path, expected_cases: tuple[str, ...]
        ):
            self.assertEqual(
                expected_cases,
                ("nominal", *(f"robustness_{index:02d}" for index in range(1, 11))),
            )
            output_directory.mkdir(parents=True, exist_ok=True)
            module_runtime_artifacts.atomic_json(
                output_directory / "report_manifest.json",
                {"expected_cases": list(expected_cases)},
            )
            return SimpleNamespace(warnings=())

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "study"
            (run / "metrics").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "case": "holdout_0000",
                        "method": method,
                        "mass_conservation_violation_max": 0.0,
                        "mass_conservation_violation_count": 0,
                        "nonnegativity_violation_max": 0.0,
                        "nonnegativity_violation_count": 0,
                    }
                    for method in ("raw", "projected", "mechanistic")
                ]
            ).to_csv(run / "metrics/physical_violations_assessment.csv", index=False)
            with (
                patch.object(
                    module_optimization_surrogate,
                    "build_surrogate_expression_graph",
                    return_value=SimpleNamespace(assets=analysis.surrogate_assets),
                ) as graph_builder,
                patch.object(
                    module_optimization_certification,
                    "_run_surrogate_certification",
                    side_effect=certify,
                ) as certification,
                patch.object(
                    module_validation_reference,
                    "_run_casewise_route_reference_evaluation",
                    side_effect=casewise_reference,
                ) as reference_evaluation,
                patch.object(
                    module_optimization_surrogate,
                    "solve_surrogate_exact_qp_local",
                    side_effect=surrogate_solver,
                ) as surrogate_solver,
                patch.object(
                    module_optimization_mechanistic,
                    "solve_mechanistic_case",
                    side_effect=mechanistic_solver,
                ) as mechanistic_solver,
                patch.object(
                    module_reporting_timing,
                    "_run_robustness_case_timing_aggregation",
                    side_effect=_mock_timing,
                ) as timing_benchmark,
                patch.object(
                    module_reporting_tables,
                    "write_reporting_tables",
                    side_effect=report,
                ) as reporting,
                patch.object(module_runtime_contracts, "assert_source_unchanged"),
            ):
                passed = module_workflow_optimization.run_optimization_stage(
                    run=run,
                    profile=PRODUCTION_PROFILE,
                    design=design,
                    development_responses=development_responses,
                    holdout_responses=holdout_responses,
                    analysis=analysis,
                    source_files={"mock": "source"},
                )
                self.assertTrue(passed)
                self.assertEqual(surrogate_solver.call_count, 11)
                self.assertEqual(mechanistic_solver.call_count, 11)
                self.assertEqual(certification.call_count, 11)
                self.assertEqual(reference_evaluation.call_count, 22)
                graph_builder.assert_called_once()
                for call in surrogate_solver.call_args_list:
                    self.assertNotIn("starts", call.kwargs)
                    self.assertIs(call.kwargs["problem"], graph_builder.return_value)
                    self.assertIsNone(call.kwargs["settings"].maximum_wall_time)
                for call in mechanistic_solver.call_args_list:
                    np.testing.assert_array_equal(
                        np.asarray(call.kwargs["starts"]), np.full((1, 7), 0.5)
                    )
                    self.assertNotIn("allow_reduced_starts", call.kwargs)
                    self.assertIsNone(call.kwargs["settings"].maximum_wall_time)
                self.assertEqual(surrogate_completed_on_entry, [set()] * 11)
                self.assertEqual(mechanistic_completed_on_entry, [set()] * 11)
                self.assertEqual(timing_benchmark.call_count, 1)
                self.assertEqual(reporting.call_count, 0)
                selected = pd.read_csv(
                    run / "metrics/selected_response_physical_audit.csv"
                )
                self.assertEqual(len(selected), 11 * 2 * 5)
                self.assertEqual(
                    selected.groupby("method").size().to_dict(),
                    {
                        method: 22
                        for method in (
                            "raw",
                            "projected",
                            "optimizer_native",
                            "exact_mechanistic_start_1",
                            "exact_mechanistic_start_2",
                        )
                    },
                )
                comparison = pd.read_csv(
                    run / "metrics/case_common_reference_comparison.csv"
                )
                reference = pd.read_csv(
                    run / "metrics/selected_candidate_reference_evaluation.csv"
                )
                self.assertEqual(len(comparison), 11)
                self.assertTrue(comparison["comparison_eligible"].all())
                self.assertEqual(len(reference), 22)
                self.assertTrue(reference["comparison_valid"].all())
                status = json.loads(
                    (run / "optimization/final_status.json").read_text(encoding="utf-8")
                )
                self.assertEqual(status["case_count"], 11)
                self.assertEqual(status["route_count"], 22)
                self.assertEqual(status["required_starts_per_route"], 1)
                self.assertEqual(status["required_attempts_per_route"], 1)
                self.assertEqual(status["surrogate_ipopt_continuation_stage_count"], 0)
                self.assertFalse(status["untouched_holdout_equivalence_executed"])
                self.assertEqual(status["selected_decision_count"], 22)
                self.assertTrue(status["scientific_validation_passed"])
                for case_id in (
                    "nominal",
                    *(f"robustness_{index:02d}" for index in range(1, 11)),
                ):
                    case = run / "optimization" / case_id
                    self.assertTrue(
                        (case / "casewise_comparison_complete.json").is_file()
                    )
                    self.assertTrue(
                        (case / "surrogate_local_convergence_complete.json").is_file()
                    )
                    for route in ("surrogate", "mechanistic"):
                        payload = json.loads(
                            (case / f"{route}.json").read_text(encoding="utf-8")
                        )
                        self.assertEqual(len(payload["starts"]), 1)
                        self.assertEqual(payload["optimization_attempt_count"], 1)
                        self.assertIsNone(payload["maximum_wall_time"])
                        self.assertEqual(
                            len(
                                list(
                                    (case / "checkpoints").glob(f"{route}_start_*.json")
                                )
                            ),
                            1,
                        )
                        self.assertTrue(
                            (
                                case / f"{route}_casewise_reference_complete.json"
                            ).is_file()
                        )
                self.assertFalse(
                    any((path.name.endswith(".tmp") for path in run.rglob("*")))
                )
                surrogate_solver.reset_mock()
                mechanistic_solver.reset_mock()
                graph_builder.reset_mock()
                certification.reset_mock()
                reference_evaluation.reset_mock()
                timing_benchmark.reset_mock()
                reporting.reset_mock()
                self.assertTrue(
                    module_workflow_optimization.run_optimization_stage(
                        run=run,
                        profile=PRODUCTION_PROFILE,
                        design=design,
                        development_responses=development_responses,
                        holdout_responses=holdout_responses,
                        analysis=analysis,
                        source_files={"mock": "source"},
                    )
                )
                surrogate_solver.assert_not_called()
                mechanistic_solver.assert_not_called()
                graph_builder.assert_not_called()
                certification.assert_not_called()
                reference_evaluation.assert_not_called()
                timing_benchmark.assert_not_called()
                self.assertEqual(reporting.call_count, 0)

    def test_surrogate_route_resumes_the_single_atomic_attempt(self) -> None:
        center = np.asarray(EXACT_QP_CENTER_START, dtype=float)
        phase = {"first": True}
        observed_completed: list[set[int]] = []

        def interrupted(*_args: object, **kwargs: object):
            completed = kwargs.get("completed_result")
            observed_completed.append(set() if completed is None else {0})
            callback = kwargs["progress_callback"]
            if phase["first"]:
                phase["first"] = False
                callback(_surrogate_start(0, center))
                raise RuntimeError("simulated interruption")
            result = completed
            self.assertIsNotNone(result)
            return SurrogateRouteResult(
                (result,),
                result,
                "selected_stationary",
                protocol=EXACT_QP_SINGLE_START_PROTOCOL,
            )

        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "nominal"
            module_runtime_artifacts.atomic_json(
                case / "checkpoints/surrogate_start_00.json",
                {
                    "route_contract": "legacy-nine-start-contract",
                    "protocol": "embedded_kkt_continuation_multistart",
                    "start_index": 0,
                    "normalized_start": center,
                    "result": _surrogate_start(0, center).as_dict(),
                },
                nonfinite_to_none=True,
            )
            with patch.object(
                module_optimization_surrogate,
                "solve_surrogate_exact_qp_local",
                side_effect=interrupted,
            ) as solver:
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    module_runtime_checkpoints._run_surrogate_route(
                        case,
                        case_id="nominal",
                        influent=0.5 * (INFLUENT_LOWER + INFLUENT_UPPER),
                        assets=object(),
                        source_id="source",
                        analysis_id="analysis",
                    )
                self.assertEqual(
                    sorted(
                        (path.name for path in (case / "checkpoints").glob("*.json"))
                    ),
                    ["surrogate_start_00.json"],
                )
                result, payload = module_runtime_checkpoints._run_surrogate_route(
                    case,
                    case_id="nominal",
                    influent=0.5 * (INFLUENT_LOWER + INFLUENT_UPPER),
                    assets=object(),
                    source_id="source",
                    analysis_id="analysis",
                )
                self.assertEqual(observed_completed, [set(), {0}])
                self.assertEqual(len(result.starts), 1)
                self.assertEqual(len(payload["starts"]), 1)
                self.assertEqual(payload["protocol"], EXACT_QP_SINGLE_START_PROTOCOL)
                self.assertEqual(payload["starts"][0]["stages"], [])
                self.assertTrue((case / "surrogate_complete.json").is_file())
                self.assertFalse(
                    any((path.name.endswith(".tmp") for path in case.rglob("*")))
                )
                solver.reset_mock()
                restored, restored_payload = (
                    module_runtime_checkpoints._run_surrogate_route(
                        case,
                        case_id="nominal",
                        influent=0.5 * (INFLUENT_LOWER + INFLUENT_UPPER),
                        assets=object(),
                        source_id="source",
                        analysis_id="analysis",
                    )
                )
                solver.assert_not_called()
                self.assertEqual(len(restored.starts), 1)
                self.assertEqual(restored_payload["selected_start"], 0)

    def test_mechanistic_route_resumes_the_single_atomic_attempt(self) -> None:
        starts = np.full((1, 7), 0.5)
        phase = {"first": True}
        observed_completed: list[set[int]] = []

        def interrupted(*_args: object, **kwargs: object):
            completed = dict(kwargs.get("completed_starts") or {})
            observed_completed.append(set(completed))
            callback = kwargs["progress_callback"]
            if phase["first"]:
                phase["first"] = False
                callback(_mechanistic_start(0, starts[0]))
                raise RuntimeError("simulated direct interruption")
            for index in range(1):
                if index not in completed:
                    result = _mechanistic_start(index, starts[index])
                    completed[index] = result
                    callback(result)
            ordered = tuple((completed[index] for index in range(1)))
            return MechanisticRouteResult(ordered, ordered[0], "selected_stationary")

        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "nominal"
            with patch.object(
                module_optimization_mechanistic,
                "solve_mechanistic_case",
                side_effect=interrupted,
            ) as solver:
                arguments = dict(
                    case_id="nominal",
                    influent=0.5 * (INFLUENT_LOWER + INFLUENT_UPPER),
                    assets=object(),
                    development_controls=np.zeros((1, 7)),
                    development_influents=np.zeros((1, 20)),
                    development_responses=np.zeros((1, MECHANISTIC_RESPONSE_COUNT)),
                    source_id="source",
                    analysis_id="analysis",
                )
                with self.assertRaisesRegex(
                    RuntimeError, "simulated direct interruption"
                ):
                    module_runtime_checkpoints._run_mechanistic_route(case, **arguments)
                self.assertEqual(
                    sorted(
                        (path.name for path in (case / "checkpoints").glob("*.json"))
                    ),
                    ["mechanistic_start_00.json"],
                )
                result, payload = module_runtime_checkpoints._run_mechanistic_route(
                    case, **arguments
                )
                self.assertEqual(observed_completed, [set(), {0}])
                self.assertEqual(len(result.starts), 1)
                self.assertEqual(len(payload["starts"]), 1)
                self.assertTrue((case / "mechanistic_complete.json").is_file())
                solver.reset_mock()
                restored, restored_payload = (
                    module_runtime_checkpoints._run_mechanistic_route(case, **arguments)
                )
                solver.assert_not_called()
                self.assertEqual(len(restored.starts), 1)
                self.assertEqual(restored_payload["selected_start"], 0)

    def test_branch_boundary_is_a_qualifier_when_exact_replays_agree(self) -> None:
        controls = 0.5 * (DECISION_LOWER + DECISION_UPPER)
        influent = 0.5 * (INFLUENT_LOWER + INFLUENT_UPPER)
        state = np.ones(3)
        analysis = SimpleNamespace(
            mechanistic_assets=SimpleNamespace(
                clarifier=object(),
                state_count=3,
                response_count=4,
                state_scale=np.ones(3),
            )
        )
        ambiguous = BranchClassification((), (), (), (), (), True, 0.0)
        solved = SimpleNamespace(accepted=True, state=state, message="accepted")
        with (
            patch.object(
                module_plant_model, "solve_steady_state", side_effect=(solved, solved)
            ) as solve,
            patch.object(
                module_optimization_mechanistic,
                "classify_branches",
                return_value=ambiguous,
            ),
            patch.object(
                module_plant_model, "diagnostics", return_value={"passed": True}
            ),
            patch.object(
                module_plant_model,
                "unpack_state",
                return_value=(np.ones((1, 20)), np.ones(1)),
            ),
            patch.object(
                module_plant_model, "generation_scale", return_value=np.ones(3)
            ),
            patch.object(
                module_optimization_mechanistic, "branches_match", return_value=True
            ),
            patch.object(
                module_plant_model, "assemble_target", return_value=np.ones(4)
            ),
            patch.object(
                module_optimization_mechanistic,
                "objective_components",
                return_value=np.ones(6),
            ),
            patch.object(
                module_optimization_mechanistic,
                "engineering_quantities",
                return_value=np.ones(7),
            ),
            patch.object(
                module_optimization_mechanistic,
                "engineering_feasible",
                return_value=True,
            ),
        ):
            response, start_1, start_2, payload = (
                module_validation_reference._casewise_exact_reference(
                    controls, influent, analysis
                )
            )
        self.assertTrue(payload["accepted"])
        self.assertEqual(payload["status"], "valid_branch_boundary")
        self.assertTrue(payload["branch_ambiguous"])
        self.assertTrue(payload["start_2_required"])
        self.assertTrue(payload["two_start_agreement_checked"])
        self.assertEqual(
            [call.kwargs["starts"] for call in solve.call_args_list], [(1,), (2,)]
        )
        np.testing.assert_array_equal(response, np.ones(4))
        np.testing.assert_array_equal(start_1, state)
        np.testing.assert_array_equal(start_2, state)

    def test_valid_exact_replay_retains_objective_when_engineering_is_infeasible(
        self,
    ) -> None:
        _, _, _, analysis = _fixture()
        analysis.mechanistic_assets.clarifier = object()
        normalized = np.asarray(EXACT_QP_CENTER_START, dtype=float)
        selected = _mechanistic_start(0, normalized)
        route_payload = MechanisticRouteResult(
            (selected,), selected, "selected_stationary"
        ).as_dict()
        route_payload.update(
            {"route_contract": "direct-contract", "elapsed_seconds": 0.5}
        )
        reference = np.full(MECHANISTIC_RESPONSE_COUNT, 4.0)
        replay = (
            reference,
            np.ones(REDUCED_STATE_COUNT),
            np.ones(REDUCED_STATE_COUNT),
            {
                "accepted": True,
                "status": "valid_interior",
                "branch_ambiguous": False,
                "engineering_feasible": False,
                "objective": 12.5,
                "objective_components": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "elapsed_seconds": 0.2,
            },
        )

        def physical(method: str, case: str, *_args: object, **_kwargs: object):
            return {
                "case": case,
                "method": method,
                "audit_available": True,
                "mass_conservation_violation_max": 0.0,
                "mass_conservation_violation_count": 0,
                "nonnegativity_violation_max": 0.0,
                "nonnegativity_violation_count": 0,
            }

        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "nominal"
            module_runtime_artifacts.atomic_json(
                case / "mechanistic.json", route_payload, nonfinite_to_none=True
            )
            with (
                patch.object(
                    module_optimization_surrogate,
                    "cold_reproject",
                    return_value=_projection(np.full(RESPONSE_COUNT, 2.0)),
                ),
                patch.object(
                    module_validation_reference,
                    "_casewise_exact_reference",
                    return_value=replay,
                ),
                patch.object(
                    module_plant_model, "assemble_target", return_value=reference
                ),
                patch.object(
                    module_validation_reference,
                    "_physical_record",
                    side_effect=physical,
                ),
            ):
                payload, violations = (
                    module_validation_reference._run_casewise_route_reference_evaluation(
                        case,
                        case_id="nominal",
                        route="mechanistic",
                        influent=0.5 * (INFLUENT_LOWER + INFLUENT_UPPER),
                        selected=selected,
                        surrogate_candidate=None,
                        route_payload=route_payload,
                        certification_payload=None,
                        recovery_payload=None,
                        analysis=analysis,
                        source_id="source",
                        analysis_id="analysis",
                    )
                )
            self.assertTrue(payload["candidate_available"])
            self.assertTrue(payload["exact_replay_valid"])
            self.assertFalse(payload["comparison_valid"])
            self.assertEqual(payload["status"], "exact_valid_engineering_infeasible")
            self.assertEqual(payload["exact_reference_objective"], 12.5)
            self.assertEqual(
                payload["exact_reference_objective_components"],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            self.assertAlmostEqual(
                payload["native_minus_reference_objective"], selected.objective - 12.5
            )
            self.assertEqual(len(violations), 5)
            marker = json.loads(
                (case / "mechanistic_casewise_reference_complete.json").read_text()
            )
            self.assertFalse(marker["comparison_valid"])
            with np.load(case / "mechanistic_casewise_reference.npz") as arrays:
                np.testing.assert_array_equal(
                    arrays["exact_reference"],
                    reduce_mechanistic_responses(
                        reference, PRODUCTION_PROFILE.layer_count
                    ),
                )
                np.testing.assert_array_equal(arrays["exact_reference_full"], reference)

    def test_mechanistic_recovery_runs_only_after_primary_failure(self) -> None:
        normalized = np.asarray(EXACT_QP_CENTER_START, dtype=float)
        primary_start = _mechanistic_start(0, normalized)
        primary = MechanisticRouteResult(
            (primary_start,), primary_start, "selected_stationary"
        )
        surrogate_candidate = _surrogate_start(0, normalized).final
        assert surrogate_candidate is not None
        common = {
            "case_id": "nominal",
            "influent": 0.5 * (INFLUENT_LOWER + INFLUENT_UPPER),
            "surrogate_candidate": surrogate_candidate,
            "assets": object(),
            "development_controls": np.zeros((1, 7)),
            "development_influents": np.zeros((1, 20)),
            "development_responses": np.zeros((1, MECHANISTIC_RESPONSE_COUNT)),
            "source_id": "source",
            "analysis_id": "analysis",
        }
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "nominal"
            with patch.object(
                module_optimization_mechanistic, "solve_mechanistic_case"
            ) as solve:
                returned, payload = (
                    module_validation_reference._run_mechanistic_failure_recovery(
                        case, result=primary, **common
                    )
                )
            self.assertIs(returned, primary)
            self.assertFalse(payload["attempted"])
            self.assertEqual(payload["status"], "not_required")
            solve.assert_not_called()
            case.mkdir(parents=True, exist_ok=True)
            module_runtime_artifacts.atomic_json(
                case / "mechanistic.json", {"route": "mechanistic"}
            )
            failed = MechanisticRouteResult((), None, "no_validated_feasible_start")
            recovered_start = _mechanistic_start(0, normalized)
            recovered = MechanisticRouteResult(
                (recovered_start,), recovered_start, "selected_stationary"
            )
            with patch.object(
                module_optimization_mechanistic,
                "solve_mechanistic_case",
                return_value=recovered,
            ) as solve:
                returned, payload = (
                    module_validation_reference._run_mechanistic_failure_recovery(
                        case, result=failed, **common
                    )
                )
            self.assertIs(returned, recovered)
            self.assertTrue(payload["attempted"])
            self.assertEqual(
                payload["selected_from"], "single_surrogate_endpoint_recovery"
            )
            solve.assert_called_once()
            np.testing.assert_array_equal(
                solve.call_args.kwargs["starts"], normalized.reshape(1, 7)
            )
            self.assertNotIn("allow_reduced_starts", solve.call_args.kwargs)
            self.assertTrue((case / "mechanistic_recovery_complete.json").is_file())
