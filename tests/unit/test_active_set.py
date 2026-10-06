from __future__ import annotations
import surrogate_optimization.optimization.surrogate as module_optimization_surrogate
import json
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from surrogate_optimization.config import DECISION_LOWER
from surrogate_optimization.config import DECISION_UPPER
from surrogate_optimization.surrogate.regression import LeastSquaresDiagnostics
from surrogate_optimization.surrogate.projection import NetworkLayout
from surrogate_optimization.surrogate.projection import NetworkRowScales
from surrogate_optimization.surrogate.regression import QuadraticFeatureMap
from surrogate_optimization.surrogate.regression import QuadraticSurrogate
from surrogate_optimization.optimization.active_set import ActiveSetDerivativeError
from surrogate_optimization.optimization.active_set import ActiveSetRefinementSettings
from surrogate_optimization.optimization.active_set import ExactQPActiveSetRefiner
from surrogate_optimization.optimization.active_set import UpperKKTAudit
from surrogate_optimization.optimization.surrogate import EXACT_QP_CENTER_START
from surrogate_optimization.optimization.surrogate import EXACT_QP_SINGLE_START_PROTOCOL
from surrogate_optimization.optimization.surrogate import LOCAL_CONVERGENCE_PROTOCOL
from surrogate_optimization.optimization.types import EngineeringLimits
from surrogate_optimization.optimization.types import SurrogateCase
from surrogate_optimization.optimization.certification import (
    SurrogateCertificationSettings,
)
from surrogate_optimization.optimization.types import SurrogateNLPAssets
from surrogate_optimization.optimization.types import SurrogateSolverSettings
from surrogate_optimization.optimization.types import SurrogateStartResult
from surrogate_optimization.optimization.types import TrustThresholds
from surrogate_optimization.optimization.surrogate import _outer_refine
from surrogate_optimization.optimization.surrogate import audit_exact_candidate
from surrogate_optimization.optimization.surrogate import (
    build_surrogate_expression_graph,
)
from surrogate_optimization.optimization.certification import (
    certify_surrogate_local_convergence,
)
from surrogate_optimization.optimization.surrogate import cold_reproject
from surrogate_optimization.optimization.surrogate import solve_surrogate_exact_qp_local
from surrogate_optimization.optimization.surrogate import (
    surrogate_exact_qp_resume_contract,
)


def _toy_assets(
    *, weak_active_set: bool = False
) -> tuple[SurrogateNLPAssets, SurrogateCase]:
    """Return a small, physical one-stage network with a constant surrogate."""
    layout = NetworkLayout(
        stage_count=1,
        component_count=2,
        layer_count=3,
        soluble_indices=(0,),
        particulate_indices=(1,),
    )
    controls = 0.5 * (DECISION_LOWER + DECISION_UPPER)
    internal, returned, waste = (controls[4], controls[5], controls[6])
    underflow = returned + waste
    effluent = 1.0 - waste
    influent = np.asarray([10.0, 10.0])
    if weak_active_set:
        response_center = np.concatenate(
            (
                influent,
                influent,
                effluent * influent,
                underflow * influent,
                np.asarray([10.0]),
            )
        )
    else:
        final = np.asarray([10.0, 10.0])
        underflow_flow = np.asarray([underflow * 10.0, underflow * 15.0])
        overflow_flow = (1.0 + returned) * final - underflow_flow
        primary = 1.0 + internal + returned
        mixer = np.asarray(
            [
                10.0,
                (10.0 + internal * final[1] + returned / underflow * underflow_flow[1])
                / primary,
            ]
        )
        response_center = np.concatenate(
            (mixer, final, overflow_flow, underflow_flow, np.asarray([10.0]))
        )
    nonconstant = QuadraticFeatureMap.expected_feature_count(7, 2) - 1
    feature_map = QuadraticFeatureMap(
        decision_center=controls,
        decision_scale=0.5 * (DECISION_UPPER - DECISION_LOWER),
        influent_center=influent,
        influent_scale=np.ones(2),
        term_center=np.zeros(nonconstant),
        term_scale=np.ones(nonconstant),
    )
    diagnostics = LeastSquaresDiagnostics(
        sample_count=100,
        feature_count=nonconstant + 1,
        response_count=layout.state_size,
        rank_tolerance=1.0,
        smallest_singular_value=1.0,
        largest_singular_value=1.0,
        condition_number=1.0,
        optimality_residual=0.0,
        coefficient_agreement=0.0,
        acceptance_threshold=1.0,
    )
    model = QuadraticSurrogate(
        feature_map=feature_map,
        response_center=response_center,
        response_scale=np.ones(layout.state_size),
        coefficients=np.zeros((layout.state_size, nonconstant + 1)),
        diagnostics=diagnostics,
        ridge_penalty=1.0,
    )
    assets = SurrogateNLPAssets(
        model=model,
        layout=layout,
        invariant_operator=np.asarray([[1.0, 0.0]]),
        tss_weights=np.asarray([0.0, 1.0]),
        row_scales=NetworkRowScales(np.ones(6), np.ones(3)),
        leverage_precision=np.zeros((nonconstant + 1, nonconstant + 1)),
        trust_thresholds=TrustThresholds(0.5, 1.0),
        quality_operator=np.asarray([[0.0, 1.0]]),
        quality_scale=np.ones(1),
        engineering=EngineeringLimits(
            fresh_flow_m3_d=1.0,
            clarifier_area_m2=1.0,
            clarifier_volume_m3=1.0,
            external_loss_min_g_m3=0.1,
            underflow_tss_upper_g_m3=100.0,
            feed_tss_min_g_m3=0.1,
        ),
    )
    return (assets, SurrogateCase(influent=influent, case_id="active_set_unit"))


class ExactQPActiveSetTests(unittest.TestCase):
    def test_primary_exact_qp_protocol_is_one_center_start_without_ipopt(self) -> None:
        assets, case = _toy_assets()
        settings = SurrogateSolverSettings(outer_maximum_iterations=40)
        reported: list[SurrogateStartResult] = []
        original_build = module_optimization_surrogate.build_surrogate_expression_graph
        with patch(
            "surrogate_optimization.optimization.surrogate.build_surrogate_expression_graph",
            wraps=original_build,
        ) as build:
            result = solve_surrogate_exact_qp_local(
                assets, case, settings=settings, progress_callback=reported.append
            )
        self.assertEqual(result.protocol, EXACT_QP_SINGLE_START_PROTOCOL)
        self.assertEqual(len(result.starts), 1)
        self.assertEqual(reported, [result.starts[0]])
        start = result.starts[0]
        self.assertEqual(start.start_index, 0)
        np.testing.assert_array_equal(
            start.initial_normalized_controls, np.asarray(EXACT_QP_CENTER_START)
        )
        self.assertEqual(start.stages, ())
        self.assertEqual(start.protocol, EXACT_QP_SINGLE_START_PROTOCOL)
        self.assertEqual(
            start.resume_contract,
            surrogate_exact_qp_resume_contract(assets, case, settings),
        )
        self.assertTrue(start.outer_refinement.attempted)
        self.assertGreater(start.outer_refinement.cold_qp_resolutions, 0)
        self.assertIsNotNone(start.final)
        build.assert_called_once()
        self.assertNotIn("compile_solver", build.call_args.kwargs)
        self.assertEqual(build.call_args.args[1], 1e-08)
        self.assertEqual(result.as_dict()["protocol"], EXACT_QP_SINGLE_START_PROTOCOL)
        self.assertEqual(
            result.as_dict()["starts"][0]["protocol"], EXACT_QP_SINGLE_START_PROTOCOL
        )

    def test_primary_exact_qp_protocol_uses_derivative_free_fallback(self) -> None:
        assets, case = _toy_assets(weak_active_set=True)
        result = solve_surrogate_exact_qp_local(
            assets, case, settings=SurrogateSolverSettings(outer_maximum_iterations=40)
        )
        start = result.starts[0]
        self.assertEqual(start.stages, ())
        self.assertEqual(
            start.outer_refinement.status, "derivative_free_budget_limited_candidate"
        )
        self.assertEqual(
            start.outer_refinement.method, "exact_qp_derivative_free_cobyqa"
        )
        self.assertTrue(start.outer_refinement.fallback_used)
        self.assertEqual(start.outer_refinement.fallback_method, "COBYQA")
        self.assertGreater(start.outer_refinement.fallback_evaluations, 1)
        self.assertGreaterEqual(
            start.outer_refinement.cold_qp_resolutions,
            start.outer_refinement.fallback_evaluations,
        )
        self.assertIsNotNone(start.outer_refinement.derivative_error)
        self.assertIsNotNone(start.final)
        assert start.final is not None
        self.assertLess(start.final.objective, start.outer_refinement.initial_objective)
        self.assertFalse(start.final.stationarity.stationary)
        self.assertEqual(
            start.final.stationarity.classification,
            "budget_limited_derivative_free_feasible_incumbent_stationarity_unresolved",
        )
        self.assertIn(
            "not an established local optimum", start.final.stationarity.reason
        )

    def test_primary_exact_qp_checkpoint_resume_performs_no_recomputation(self) -> None:
        assets, case = _toy_assets(weak_active_set=True)
        settings = SurrogateSolverSettings(outer_maximum_iterations=10)
        first = solve_surrogate_exact_qp_local(assets, case, settings=settings)
        checkpoint = first.starts[0]
        with patch(
            "surrogate_optimization.optimization.surrogate.build_surrogate_expression_graph",
            side_effect=AssertionError("a valid completed result was recomputed"),
        ) as build:
            resumed = solve_surrogate_exact_qp_local(
                assets, case, settings=settings, completed_result=checkpoint
            )
        build.assert_not_called()
        self.assertIs(resumed.starts[0], checkpoint)
        stale = replace(checkpoint, resume_contract="stale")
        with self.assertRaisesRegex(ValueError, "stale exact-QP resume contract"):
            solve_surrogate_exact_qp_local(
                assets, case, settings=settings, completed_result=stale
            )

    def test_primary_exact_qp_protocol_reuses_one_expression_problem(self) -> None:
        assets, case = _toy_assets(weak_active_set=True)
        settings = SurrogateSolverSettings(outer_maximum_iterations=10)
        problem = build_surrogate_expression_graph(
            assets,
            1e-08,
            settings=settings,
            name="active_set_reusable_expression_graph",
        )
        with patch(
            "surrogate_optimization.optimization.surrogate.build_surrogate_expression_graph",
            side_effect=AssertionError("the reusable expression graph was rebuilt"),
        ) as build:
            result = solve_surrogate_exact_qp_local(
                assets, case, settings=settings, problem=problem
            )
        build.assert_not_called()
        self.assertEqual(result.starts[0].stages, ())
        with self.assertRaisesRegex(ValueError, "final gap value"):
            solve_surrogate_exact_qp_local(
                assets, case, settings=settings, problem=replace(problem, tau=1e-06)
            )

    def test_exact_sensitivity_matches_an_external_central_difference(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_expression_graph(
            assets, 1e-08, name="active_set_gradient"
        )
        refiner = ExactQPActiveSetRefiner(
            assets, case, problem=problem, name="active_set_gradient"
        )
        normalized = np.full(7, 0.5)
        trial = refiner.evaluate(normalized)
        self.assertTrue(trial.projection.accepted)
        self.assertTrue(trial.lower_active_set.stable)
        self.assertEqual(trial.lower_active_set.active_indices, ())
        self.assertEqual(len(trial.lower_active_set.perturbations), 14)
        self.assertEqual(refiner.cold_qp_resolutions, 15)
        self.assertLessEqual(trial.sensitivity.solve_residual, 1e-08)
        step = 2e-06
        finite_difference = np.empty(7)
        parameter = case.parameter_vector(assets)
        for coordinate in range(7):
            plus = normalized.copy()
            minus = normalized.copy()
            plus[coordinate] += step
            minus[coordinate] -= step
            plus_projection = cold_reproject(assets, case, plus)
            minus_projection = cold_reproject(assets, case, minus)
            plus_value = float(
                problem.upper_from_state_function(
                    plus, parameter, plus_projection.state
                )[0]
            )
            minus_value = float(
                problem.upper_from_state_function(
                    minus, parameter, minus_projection.state
                )[0]
            )
            finite_difference[coordinate] = (plus_value - minus_value) / (2.0 * step)
        np.testing.assert_allclose(
            trial.objective_gradient_normalized,
            finite_difference,
            rtol=2e-06,
            atol=2e-08,
        )

    def test_weakly_active_lower_constraints_fail_explicitly(self) -> None:
        assets, case = _toy_assets(weak_active_set=True)
        problem = build_surrogate_expression_graph(
            assets, 1e-08, name="active_set_weak"
        )
        refiner = ExactQPActiveSetRefiner(
            assets, case, problem=problem, name="active_set_weak"
        )
        with self.assertRaises(ActiveSetDerivativeError) as captured:
            refiner.evaluate(np.full(7, 0.5))
        audit = captured.exception.audit
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertFalse(audit.stable)
        self.assertFalse(audit.strict_complementarity_passed)
        self.assertAlmostEqual(audit.minimum_active_multiplier, 0.0, places=12)
        self.assertIn("active lower multiplier", audit.reason)
        result = refiner.refine(np.full(7, 0.5))
        self.assertEqual(result.status, "active_set_derivative_unavailable")
        self.assertIsNotNone(result.derivative_error)
        self.assertIsNone(result.final)

    def test_refinement_has_independent_final_qp_and_upper_kkt_audit(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_expression_graph(
            assets, 1e-08, name="active_set_refinement"
        )
        refiner = ExactQPActiveSetRefiner(
            assets,
            case,
            problem=problem,
            settings=ActiveSetRefinementSettings(maximum_iterations=40),
            name="active_set_refinement",
        )
        result = refiner.refine(np.full(7, 0.5))
        self.assertIsNotNone(result.final)
        self.assertIsNotNone(result.upper_kkt)
        assert result.final is not None
        assert result.upper_kkt is not None
        self.assertTrue(result.final.independent_final_replay)
        self.assertTrue(result.final.projection.accepted)
        self.assertTrue(result.upper_kkt.feasible)
        self.assertTrue(result.state_reproduction_passed)
        self.assertLessEqual(result.state_reproduction_residual, 1e-08)
        self.assertLess(result.final.objective, result.initial.objective)
        self.assertGreater(result.cold_qp_resolutions, result.distinct_trials)
        self.assertIn(
            result.status,
            {"validated_stationary", "validated_feasible_stationarity_unresolved"},
        )
        json.dumps(result.as_dict())

    def test_surrogate_outer_path_serializes_active_set_and_upper_kkt(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_expression_graph(
            assets, 1e-08, name="active_set_integration"
        )
        final, refinement = _outer_refine(
            problem,
            case,
            np.full(7, 0.5),
            SurrogateSolverSettings(outer_maximum_iterations=40),
        )
        self.assertIsNotNone(final)
        assert final is not None
        self.assertTrue(final.feasibility.feasible)
        self.assertIsNotNone(final.lower_active_set)
        self.assertIsNotNone(final.upper_kkt)
        self.assertEqual(
            final.stationarity.upper_stationarity_residual,
            final.upper_kkt["stationarity_residual"],
        )
        self.assertGreater(refinement.cold_qp_resolutions, refinement.evaluations)
        self.assertEqual(refinement.lower_active_set, final.lower_active_set)
        self.assertEqual(refinement.upper_kkt, final.upper_kkt)
        if final.status == "validated_feasible_stationarity_unresolved":
            self.assertFalse(final.stationarity.resolved)
        serialized = final.as_dict()
        self.assertIn("lower_active_set", serialized)
        self.assertIn("upper_kkt", serialized)
        json.dumps(serialized)

    def test_checkpoint_records_round_trip_and_restore_null_as_nan(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_expression_graph(
            assets, 1e-08, name="active_set_roundtrip"
        )
        final, refinement = _outer_refine(
            problem,
            case,
            np.full(7, 0.5),
            SurrogateSolverSettings(outer_maximum_iterations=40),
        )
        assert final is not None
        original = SurrogateStartResult(
            start_index=0,
            initial_normalized_controls=np.full(7, 0.5),
            stages=(),
            outer_refinement=refinement,
            final=final,
            status=final.status,
            resume_contract="unit-contract",
        )
        payload = original.as_dict()
        payload["final"]["projection"]["state"][0] = None
        restored = SurrogateStartResult.from_dict(payload)
        self.assertEqual(restored.resume_contract, "unit-contract")
        self.assertTrue(np.isnan(restored.final.projection.state[0]))
        np.testing.assert_array_equal(restored.final.projected, final.projected)
        np.testing.assert_array_equal(
            restored.final.projection.inequality_multipliers,
            final.projection.inequality_multipliers,
        )
        json.dumps(restored.as_dict())

    def test_refinement_prefers_stationary_cached_trial_before_final_replay(
        self,
    ) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_expression_graph(
            assets, 1e-08, name="active_set_inner_selection"
        )
        refiner = ExactQPActiveSetRefiner(
            assets, case, problem=problem, name="active_set_inner_selection"
        )
        stationary_controls = np.full(7, 0.5)
        lower_objective_controls = np.full(7, 0.4)
        stationary_trial = refiner.evaluate(stationary_controls)
        lower_trial = refiner.evaluate(lower_objective_controls)
        self.assertLess(lower_trial.objective, stationary_trial.objective)

        def audit(trial: object) -> UpperKKTAudit:
            stationary = np.array_equal(trial.normalized_controls, stationary_controls)
            return UpperKKTAudit(
                active_indices=(),
                active_names=(),
                multipliers=np.zeros(trial.upper_constraints.size),
                primal_residual=0.0,
                dual_feasibility_residual=0.0,
                stationarity_residual=0.0 if stationary else 0.001,
                complementarity_residual=0.0,
                feasible=True,
                stationary=stationary,
                classification="first_order_kkt_stationary_feasible"
                if stationary
                else "validated_feasible_stationarity_unresolved",
                reason="unit audit",
            )

        with (
            patch(
                "surrogate_optimization.optimization.active_set.minimize",
                return_value=SimpleNamespace(
                    x=lower_objective_controls, success=True, message="unit", nit=1
                ),
            ),
            patch.object(refiner, "audit_upper_kkt", side_effect=audit),
        ):
            result = refiner.refine(stationary_controls)
        np.testing.assert_array_equal(
            result.final.normalized_controls, stationary_controls
        )
        self.assertTrue(result.stationary)

    def test_independent_state_reproduction_is_an_acceptance_gate(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_expression_graph(
            assets, 1e-08, name="active_set_reproduction"
        )
        refiner = ExactQPActiveSetRefiner(
            assets, case, problem=problem, name="active_set_reproduction"
        )
        original_evaluate = refiner.evaluate

        def shifted_final(
            value: np.ndarray,
            *,
            force_cold: bool = False,
            independent_final_replay: bool = False,
        ) -> object:
            trial = original_evaluate(
                value,
                force_cold=force_cold,
                independent_final_replay=independent_final_replay,
            )
            if force_cold:
                return replace(
                    trial,
                    projected_state=trial.projected_state
                    + 2e-08 * assets.model.response_scale,
                )
            return trial

        with (
            patch(
                "surrogate_optimization.optimization.active_set.minimize",
                return_value=SimpleNamespace(
                    x=np.full(7, 0.5), success=True, message="unit", nit=1
                ),
            ),
            patch.object(refiner, "evaluate", side_effect=shifted_final),
        ):
            result = refiner.refine(np.full(7, 0.5))
        self.assertEqual(result.status, "projection_reproduction_failed")
        self.assertFalse(result.state_reproduction_passed)
        self.assertGreater(result.state_reproduction_residual, 1e-08)
        self.assertFalse(result.feasible)

    def test_degenerate_endpoint_passes_complete_two_scale_poll(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_expression_graph(
            assets, 1e-08, name="certificate_flat"
        )
        controls = np.full(7, 0.5)
        initial = audit_exact_candidate(problem, case, controls)

        def exact_candidate(
            _problem: object, _case: object, normalized: np.ndarray, **_kwargs: object
        ):
            value = np.asarray(normalized, dtype=float)
            feasible = bool(np.all(value[1:] == controls[1:]))
            return replace(
                initial,
                normalized_controls=value.copy(),
                controls=assets.theta_lower + assets.theta_span * value,
                objective=float(np.sum((value - controls) ** 2)),
                feasibility=replace(initial.feasibility, feasible=feasible),
            )

        class DegenerateRefiner:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.cold_qp_resolutions = 0

            def evaluate(self, *_args: object, **_kwargs: object) -> object:
                raise ActiveSetDerivativeError("unit degenerate active set")

        with (
            patch(
                "surrogate_optimization.optimization.surrogate.audit_exact_candidate",
                side_effect=exact_candidate,
            ),
            patch(
                "surrogate_optimization.optimization.active_set.ExactQPActiveSetRefiner",
                DegenerateRefiner,
            ),
        ):
            result = certify_surrogate_local_convergence(
                assets, case, initial, problem=problem
            )
        certificate = result.certificate
        self.assertEqual(certificate.protocol, LOCAL_CONVERGENCE_PROTOCOL)
        self.assertEqual(
            certificate.protocol, "exact_qp_two_scale_accelerated_feasible_poll"
        )
        self.assertEqual(SurrogateCertificationSettings().maximum_evaluations, 10000)
        self.assertEqual(
            SurrogateCertificationSettings().acceleration_growth_factor, 2.0
        )
        self.assertEqual(
            SurrogateCertificationSettings().maximum_acceleration_probes, 16
        )
        self.assertEqual(certificate.classification, "finite_resolution_feasible_poll")
        self.assertTrue(certificate.locally_converged)
        self.assertFalse(certificate.first_order_certified)
        self.assertFalse(certificate.stationarity_resolved)
        self.assertEqual(certificate.accepted_improvements, 0)
        self.assertEqual(len(certificate.poll_levels), 2)
        self.assertEqual(
            [level["direction_count"] for level in certificate.poll_levels], [14, 106]
        )
        self.assertTrue(all((level["passed"] for level in certificate.poll_levels)))
        self.assertTrue(
            all(
                (
                    level["feasible_direction_rank"] == 1
                    for level in certificate.poll_levels
                )
            )
        )
        self.assertTrue(
            all(
                (
                    level["required_direction_rank"] is None
                    for level in certificate.poll_levels
                )
            )
        )
        self.assertTrue(
            all((level["rank_is_diagnostic_only"] for level in certificate.poll_levels))
        )
        self.assertTrue(
            all(
                (
                    level["feasible_direction_coverage_passed"]
                    for level in certificate.poll_levels
                )
            )
        )
        self.assertTrue(
            all(
                (
                    level["acceleration_evaluation_requests"] == 0
                    for level in certificate.poll_levels
                )
            )
        )
        self.assertTrue(
            all(
                (
                    level["acceleration_accepted_improvements"] == 0
                    for level in certificate.poll_levels
                )
            )
        )
        self.assertIsNotNone(result.candidate)
        assert result.candidate is not None
        self.assertEqual(
            result.candidate.status,
            "validated_feasible_poll_converged_stationarity_unresolved",
        )
        self.assertFalse(result.candidate.stationarity.resolved)

    def test_poll_accepts_improvements_and_repeats_radius_before_passing(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_expression_graph(
            assets, 1e-08, name="certificate_repeat"
        )
        controls = np.full(7, 0.5)
        optimum = controls.copy()
        optimum[0] = 0.502
        initial = audit_exact_candidate(problem, case, controls)

        def exact_candidate(
            _problem: object, _case: object, normalized: np.ndarray, **_kwargs: object
        ):
            value = np.asarray(normalized, dtype=float)
            return replace(
                initial,
                normalized_controls=value.copy(),
                controls=assets.theta_lower + assets.theta_span * value,
                objective=float(np.sum((value - optimum) ** 2)),
            )

        class DegenerateRefiner:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.cold_qp_resolutions = 0

            def evaluate(self, *_args: object, **_kwargs: object) -> object:
                raise ActiveSetDerivativeError("unit nonsmooth endpoint")

        settings = SurrogateCertificationSettings(
            poll_radii=(0.001,), maximum_evaluations=1000
        )
        with (
            patch(
                "surrogate_optimization.optimization.surrogate.audit_exact_candidate",
                side_effect=exact_candidate,
            ),
            patch(
                "surrogate_optimization.optimization.active_set.ExactQPActiveSetRefiner",
                DegenerateRefiner,
            ),
        ):
            result = certify_surrogate_local_convergence(
                assets, case, initial, settings=settings, problem=problem
            )
        self.assertEqual(
            result.certificate.classification, "finite_resolution_feasible_poll"
        )
        self.assertEqual(result.certificate.accepted_improvements, 2)
        level = result.certificate.poll_levels[0]
        self.assertEqual(level["accepted_improvements"], 2)
        self.assertEqual(level["poll_accepted_improvements"], 1)
        self.assertEqual(level["acceleration_accepted_improvements"], 1)
        self.assertEqual(level["acceleration_evaluation_requests"], 2)
        self.assertEqual(level["acceleration_unique_evaluations"], 2)
        self.assertEqual(level["acceleration_maximum_accepted_multiplier"], 2.0)
        self.assertEqual(level["acceleration_stops"], {"no_sufficient_descent": 1})
        self.assertEqual(level["direction_count"], 106)
        self.assertTrue(level["complete_no_descent_poll"])
        np.testing.assert_allclose(
            result.certificate.final_normalized_controls, optimum, atol=1e-14
        )

    def test_fine_scale_improvement_revalidates_the_coarse_scale(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_expression_graph(
            assets, 1e-08, name="certificate_revalidate"
        )
        controls = np.full(7, 0.5)
        optimum = controls.copy()
        optimum[0] += 0.0001
        initial = audit_exact_candidate(problem, case, controls)

        def exact_candidate(
            _problem: object, _case: object, normalized: np.ndarray, **_kwargs: object
        ):
            value = np.asarray(normalized, dtype=float)
            return replace(
                initial,
                normalized_controls=value.copy(),
                controls=assets.theta_lower + assets.theta_span * value,
                objective=float(np.sum((value - optimum) ** 2)),
            )

        class DegenerateRefiner:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.cold_qp_resolutions = 0

            def evaluate(self, *_args: object, **_kwargs: object) -> object:
                raise ActiveSetDerivativeError("unit nonsmooth endpoint")

        settings = SurrogateCertificationSettings(
            absolute_decrease_tolerance=1e-12, relative_decrease_tolerance=1e-12
        )
        with (
            patch(
                "surrogate_optimization.optimization.surrogate.audit_exact_candidate",
                side_effect=exact_candidate,
            ),
            patch(
                "surrogate_optimization.optimization.active_set.ExactQPActiveSetRefiner",
                DegenerateRefiner,
            ),
        ):
            result = certify_surrogate_local_convergence(
                assets, case, initial, settings=settings, problem=problem
            )
        self.assertTrue(result.certificate.locally_converged)
        self.assertEqual(result.certificate.accepted_improvements, 1)
        levels = result.certificate.poll_levels
        self.assertEqual(
            [(level["validation_round"], level["radius"]) for level in levels],
            [(0, 0.001), (0, 0.0001), (1, 0.001), (1, 0.0001)],
        )
        self.assertEqual(
            [level["accepted_improvements"] for level in levels], [0, 1, 0, 0]
        )
        self.assertTrue(all((level["passed"] for level in levels)))
        np.testing.assert_allclose(
            result.certificate.final_normalized_controls, optimum, atol=1e-14
        )

    def test_cached_revalidation_completes_at_the_exact_evaluation_cap(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_expression_graph(
            assets, 1e-08, name="certificate_cached_cap"
        )
        controls = np.full(7, 0.5)
        optimum = controls.copy()
        optimum[0] += 0.0001
        initial = audit_exact_candidate(problem, case, controls)

        def exact_candidate(
            _problem: object, _case: object, normalized: np.ndarray, **_kwargs: object
        ):
            value = np.asarray(normalized, dtype=float)
            return replace(
                initial,
                normalized_controls=value.copy(),
                controls=assets.theta_lower + assets.theta_span * value,
                objective=float(np.sum((value - optimum) ** 2)),
            )

        class DegenerateRefiner:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.cold_qp_resolutions = 0

            def evaluate(self, *_args: object, **_kwargs: object) -> object:
                raise ActiveSetDerivativeError("unit nonsmooth endpoint")

        axis = np.zeros(7)
        axis[0] = 1.0
        settings = SurrogateCertificationSettings(
            poll_radii=(0.0002, 0.0001),
            absolute_decrease_tolerance=1e-12,
            relative_decrease_tolerance=1e-12,
            maximum_evaluations=6,
        )
        with (
            patch(
                "surrogate_optimization.optimization.surrogate.audit_exact_candidate",
                side_effect=exact_candidate,
            ),
            patch(
                "surrogate_optimization.optimization.certification._local_poll_directions",
                return_value=np.vstack((axis, -axis)),
            ),
            patch(
                "surrogate_optimization.optimization.active_set.ExactQPActiveSetRefiner",
                DegenerateRefiner,
            ),
        ):
            result = certify_surrogate_local_convergence(
                assets, case, initial, settings=settings, problem=problem
            )
        self.assertEqual(result.certificate.evaluations, settings.maximum_evaluations)
        self.assertEqual(
            result.certificate.classification, "finite_resolution_feasible_poll"
        )
        self.assertTrue(result.certificate.locally_converged)
        self.assertNotIn("budget", result.certificate.termination_reason)
        self.assertEqual(
            [level["validation_round"] for level in result.certificate.poll_levels],
            [0, 0, 1, 1],
        )
        self.assertTrue(
            all((level["passed"] for level in result.certificate.poll_levels))
        )

    def test_poll_budget_exhaustion_is_explicit_and_not_certified(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_expression_graph(
            assets, 1e-08, name="certificate_budget"
        )
        controls = np.full(7, 0.5)
        initial = audit_exact_candidate(problem, case, controls)

        def exact_candidate(
            _problem: object, _case: object, normalized: np.ndarray, **_kwargs: object
        ):
            value = np.asarray(normalized, dtype=float)
            return replace(
                initial,
                normalized_controls=value.copy(),
                controls=assets.theta_lower + assets.theta_span * value,
                objective=1.0,
            )

        class DegenerateRefiner:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.cold_qp_resolutions = 0

            def evaluate(self, *_args: object, **_kwargs: object) -> object:
                raise ActiveSetDerivativeError("unit degenerate active set")

        with (
            patch(
                "surrogate_optimization.optimization.surrogate.audit_exact_candidate",
                side_effect=exact_candidate,
            ),
            patch(
                "surrogate_optimization.optimization.active_set.ExactQPActiveSetRefiner",
                DegenerateRefiner,
            ),
        ):
            result = certify_surrogate_local_convergence(
                assets,
                case,
                initial,
                settings=SurrogateCertificationSettings(maximum_evaluations=2),
                problem=problem,
            )
        self.assertEqual(result.certificate.classification, "poll_budget_limited")
        self.assertFalse(result.certificate.locally_converged)
        self.assertFalse(result.certificate.first_order_certified)
        self.assertEqual(result.certificate.evaluations, 2)
        self.assertIn("budget", result.certificate.termination_reason)
        self.assertIsNotNone(result.candidate)
        assert result.candidate is not None
        self.assertTrue(result.candidate.feasibility.feasible)
        self.assertIn("poll_budget_limited", result.candidate.status)
