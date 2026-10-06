"""Staged unit tests for the coupled closed-loop mechanistic model."""

from __future__ import annotations
import unittest
import numpy as np
from surrogate_optimization.plant.model import CLARIFIER
from surrogate_optimization.plant.definitions import COMPONENT_INDEX
from surrogate_optimization.plant.definitions import INVARIANT_MATRIX
from surrogate_optimization.plant.definitions import NOMINAL_INFLUENT
from surrogate_optimization.plant.definitions import N_COMPONENTS
from surrogate_optimization.plant.definitions import N_LAYERS
from surrogate_optimization.plant.definitions import N_PROCESSES
from surrogate_optimization.plant.definitions import STATE_SIZE
from surrogate_optimization.plant.definitions import STOICHIOMETRIC_MATRIX
from surrogate_optimization.plant.definitions import TARGET_SIZE
from surrogate_optimization.plant.definitions import TSS_VECTOR
from surrogate_optimization.plant.operating_point import OperatingPoint
from surrogate_optimization.plant.model import assemble_target
from surrogate_optimization.plant.model import audit_mechanistic_matrices
from surrogate_optimization.plant.model import clarifier_fluxes
from surrogate_optimization.plant.model import clarifier_rhs
from surrogate_optimization.plant.model import coupled_rhs
from surrogate_optimization.plant.model import initial_state
from surrogate_optimization.plant.model import mixer_state
from surrogate_optimization.plant.model import process_rates
from surrogate_optimization.plant.model import reconstruct_clarifier
from surrogate_optimization.plant.model import settling_velocity
from surrogate_optimization.plant.model import solve_steady_state
from surrogate_optimization.plant.model import zero_state_solution


class MatrixAndKineticsTests(unittest.TestCase):
    def test_dimensions_and_invariants(self) -> None:
        self.assertEqual(STOICHIOMETRIC_MATRIX.shape, (N_PROCESSES, N_COMPONENTS))
        self.assertEqual(INVARIANT_MATRIX.shape, (5, N_COMPONENTS))
        audit = audit_mechanistic_matrices()
        self.assertTrue(audit["passed"], audit)
        self.assertLessEqual(
            float(np.max(np.abs(INVARIANT_MATRIX @ STOICHIOMETRIC_MATRIX.T))), 1e-12
        )

    def test_rates_are_complete_finite_and_nonnegative(self) -> None:
        rates = process_rates(NOMINAL_INFLUENT)
        self.assertEqual(rates.shape, (N_PROCESSES,))
        self.assertTrue(np.all(np.isfinite(rates)))
        self.assertTrue(np.all(rates >= 0.0))
        ix = COMPONENT_INDEX
        c = NOMINAL_INFLUENT
        p11_expected = (
            3.0
            * (0.2 / (0.2 + c[ix["S_O"]]))
            * (0.5 / (0.5 + c[ix["S_NO2"]] + c[ix["S_NO3"]]))
            * c[ix["S_F"]]
            / (4.0 + c[ix["S_F"]])
            * c[ix["S_ALK"]]
            / (0.1 + c[ix["S_ALK"]])
            * c[ix["X_H"]]
        )
        self.assertAlmostEqual(rates[10], p11_expected, places=12)


class ClarifierAndRecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.operating = OperatingPoint(18.0, 0.6, 0.6, 0.6, 2.0, 0.75, 0.02)
        self.feed = NOMINAL_INFLUENT.copy()
        self.feed_tss = float(TSS_VECTOR @ self.feed)
        self.layers = self.feed_tss * np.asarray(
            [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0]
        )

    def test_settling_bounds_and_flux_dimensions(self) -> None:
        velocity = settling_velocity(self.layers, self.feed_tss)
        self.assertTrue(np.all(velocity >= 0.0))
        self.assertTrue(np.all(velocity <= CLARIFIER.maximum_settling_velocity))
        flux = clarifier_fluxes(self.layers, self.feed_tss, self.operating)
        self.assertEqual(flux.shape, (N_LAYERS + 1,))
        self.assertAlmostEqual(
            flux[0],
            -CLARIFIER.fresh_flow
            * self.operating.q_effluent
            / CLARIFIER.area
            * self.layers[0],
        )

    def test_layer_balance_telescopes_to_external_tss_balance(self) -> None:
        rhs = clarifier_rhs(self.layers, self.feed_tss, self.operating)
        accumulated = CLARIFIER.layer_volume * rhs.sum()
        external = CLARIFIER.fresh_flow * (
            self.operating.q_clarifier * self.feed_tss
            - self.operating.q_effluent * self.layers[0]
            - self.operating.q_underflow * self.layers[-1]
        )
        self.assertAlmostEqual(accumulated, external, places=6)

    def test_component_reconstruction_and_mixer_close(self) -> None:
        ce, cu = reconstruct_clarifier(self.feed, self.layers)
        np.testing.assert_allclose(ce[:10], self.feed[:10])
        np.testing.assert_allclose(cu[:10], self.feed[:10])
        self.assertAlmostEqual(float(TSS_VECTOR @ ce), self.layers[0])
        self.assertAlmostEqual(float(TSS_VECTOR @ cu), self.layers[-1])
        mixer = mixer_state(NOMINAL_INFLUENT, self.feed, cu, self.operating)
        balance = (
            self.operating.q_process * mixer
            - NOMINAL_INFLUENT
            - self.operating.internal_recycle * self.feed
            - self.operating.return_sludge * cu
        )
        np.testing.assert_allclose(balance, 0.0, atol=1e-12)

    def test_coupled_state_and_target_dimensions(self) -> None:
        state = initial_state(NOMINAL_INFLUENT, start=2)
        self.assertEqual(state.shape, (STATE_SIZE,))
        derivative = coupled_rhs(state, self.operating, NOMINAL_INFLUENT)
        self.assertEqual(derivative.shape, (STATE_SIZE,))
        self.assertTrue(np.all(np.isfinite(derivative)))
        target = assemble_target(state, self.operating, NOMINAL_INFLUENT)
        self.assertEqual(target.shape, (TARGET_SIZE,))


class SteadySolverSmokeTests(unittest.TestCase):
    def test_exact_zero_fixture_exercises_full_solve_route(self) -> None:
        result = zero_state_solution()
        self.assertTrue(result.accepted, result.diagnostics)
        self.assertLessEqual(float(result.diagnostics["scaled_residual_inf"]), 1e-08)
        self.assertEqual(result.target.shape, (TARGET_SIZE,))

    def test_loaded_nominal_state_converges_and_is_stable(self) -> None:
        result = solve_steady_state(
            OperatingPoint(21.0, 0.5, 0.5, 0.5, 2.0, 0.75, 0.02),
            NOMINAL_INFLUENT,
            starts=(1,),
        )
        self.assertTrue(result.accepted, result.diagnostics)
        self.assertEqual(result.route, "scaled-bdf")
        self.assertLessEqual(float(result.diagnostics["scaled_residual_inf"]), 1e-08)
        self.assertLessEqual(
            float(result.diagnostics["largest_real_eigenvalue"]), 1e-08
        )

    def test_high_throughflow_box_corner_converges(self) -> None:
        result = solve_steady_state(
            OperatingPoint(6.0, 1.0, 1.0, 1.0, 4.0, 1.25, 0.05),
            NOMINAL_INFLUENT,
            starts=(1,),
        )
        self.assertTrue(result.accepted, result.diagnostics)

    def test_zero_aeration_low_waste_corner_uses_positive_fallback(self) -> None:
        result = solve_steady_state(
            OperatingPoint(36.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.001),
            NOMINAL_INFLUENT,
            starts=(1,),
        )
        self.assertTrue(result.accepted, result.diagnostics)
        self.assertEqual(result.route, "log-bdf")
        self.assertGreaterEqual(float(result.diagnostics["minimum_state"]), -1e-10)

    def test_slow_solids_mode_closes_the_plant_boundary(self) -> None:
        controls = np.array(
            [
                7.601629632672628,
                0.531404810145,
                0.06715255853995941,
                0.6228666363318475,
                0.01572435395674802,
            ]
        )
        influent = np.array(
            [
                0.26290134172788404,
                151.5854617729845,
                37.24872795296685,
                24.010258042743573,
                2.718802559303634,
                1.0455364709367772,
                0.08009791773972903,
                11.835171908208736,
                69.36974364687532,
                2.4450814032412236,
                74.6781761479309,
                247.41435225227497,
                42.011655432467435,
                54.24148051596899,
                7.1726225681351545,
                29.302221935565996,
                5.883801048317794,
                1.809217675497095,
                7.079367158597445,
                2.82512504618499,
            ]
        )
        result = solve_steady_state(
            OperatingPoint(
                controls[0], controls[1], controls[1], controls[1], *controls[2:]
            ),
            influent,
        )
        self.assertTrue(result.accepted, result.diagnostics)
        self.assertLessEqual(
            float(result.diagnostics["plant_boundary_residual"]), 1e-08
        )
