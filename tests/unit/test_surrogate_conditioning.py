from __future__ import annotations
import surrogate_optimization.optimization.surrogate as module_optimization_surrogate
from types import SimpleNamespace
import unittest
import casadi as ca
import numpy as np
from surrogate_optimization.surrogate.regression import LogOverflowTSSClosure
from surrogate_optimization.surrogate.projection import NetworkLayout
from surrogate_optimization.surrogate.projection import build_network_operators
from surrogate_optimization.optimization.types import EngineeringLimits
from surrogate_optimization.optimization.types import NamedTrustRows
from surrogate_optimization.optimization.types import TrustDiagnosticCallbacks
from surrogate_optimization.optimization.types import TrustThresholds
from surrogate_optimization.optimization.surrogate import symbolic_network_operators


class _RecordingSolver:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **arguments: object) -> dict[str, ca.DM]:
        self.calls.append(arguments)
        return {
            "x": ca.DM(np.asarray(arguments["x0"], dtype=float)),
            "lam_x": ca.DM([1.0, 2.0, 3.0]),
            "lam_g": ca.DM([-4.0, 5.0]),
        }

    @staticmethod
    def stats() -> dict[str, object]:
        return {"success": True, "return_status": "Solve_Succeeded", "iter_count": 3}


class _Case:
    @staticmethod
    def parameter_vector(_assets: object) -> np.ndarray:
        return np.asarray([42.0])


class SurrogateConditioningTests(unittest.TestCase):
    def test_all_trust_rows_are_relative_dimensionless_residuals(self) -> None:

        def rows(first: float, second: float):
            return lambda *_arguments: ca.vertcat(first, second)

        assets = SimpleNamespace(
            layout=SimpleNamespace(state_size=2),
            leverage_precision=np.eye(2),
            trust_thresholds=TrustThresholds(
                correction_rms=2.0,
                regularized_leverage=4.0,
                split_rms=3.0,
                reactor_rms=5.0,
            ),
            trust_callbacks=TrustDiagnosticCallbacks(
                split_rows=rows(3.0, 0.0),
                reactor_rows=rows(10.0, 0.0),
                additional=(NamedTrustRows("extra", rows(22.0, 0.0), 11.0),),
            ),
        )
        constraints, names, values = module_optimization_surrogate._trust_expressions(
            ca.DM.zeros(7),
            ca.DM.zeros(2),
            ca.DM.zeros(2),
            ca.DM([2.0, 0.0]),
            ca.DM.zeros(2),
            ca.DM([2.0, 0.0]),
            assets,
        )
        self.assertEqual(names, ("correction", "leverage", "split", "reactor", "extra"))
        np.testing.assert_allclose(
            np.asarray(values).reshape(-1), [2.0, 4.0, 4.5, 50.0, 242.0]
        )
        np.testing.assert_allclose(
            np.asarray(constraints).reshape(-1), [-0.5, 0.0, -0.5, 1.0, 1.0]
        )
        self.assertEqual(
            module_optimization_surrogate._normalized_limit_residual(0.25, 0.0), 0.25
        )

    def test_engineering_rows_use_only_positive_fixed_scales(self) -> None:
        layout = NetworkLayout(
            stage_count=1,
            component_count=2,
            layer_count=3,
            soluble_indices=(0,),
            particulate_indices=(1,),
        )
        limits = EngineeringLimits(
            fresh_flow_m3_d=100.0,
            clarifier_area_m2=10.0,
            clarifier_volume_m3=30.0,
            external_loss_min_g_m3=2.0,
            underflow_tss_upper_g_m3=1000.0,
            feed_tss_min_g_m3=10.0,
        )
        assets = SimpleNamespace(
            layout=layout, engineering=limits, tss_weights=np.asarray([0.0, 1.0])
        )
        controls = ca.DM([24.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.1])
        state = np.zeros(layout.state_size)
        state[layout.reactor_slice(0)] = [0.0, 100.0]
        state[layout.overflow_flow_slice] = [0.0, 10.0]
        state[layout.underflow_flow_slice] = [0.0, 200.0]
        state[layout.inventory_index] = 3000.0
        constraints, names, quantities = (
            module_optimization_surrogate._engineering_expressions(
                controls, ca.DM(state), assets
            )
        )
        external_loss = 10.0 + 0.1 * 200.0 / 0.6
        inventory = 100.0 * 100.0 + 10.0 * 300.0
        expected = np.asarray(
            [
                (2.0 - external_loss) / 2.0,
                (200.0 / 0.6 - 1000.0) / 1000.0,
                (10.0 - 100.0) / 10.0,
            ]
        )
        self.assertEqual(
            names,
            ("external_solids_loss_guard", "underflow_tss_upper", "feed_tss_lower"),
        )
        np.testing.assert_allclose(np.asarray(constraints).reshape(-1), expected)
        self.assertTrue(np.all(np.isfinite(np.asarray(quantities))))

    def test_symbolic_and_numeric_reduced_network_operators_match(self) -> None:
        layout = NetworkLayout(
            stage_count=1,
            component_count=2,
            layer_count=3,
            soluble_indices=(0,),
            particulate_indices=(1,),
        )
        invariant = np.asarray([[1.0, 0.0]])
        tss = np.asarray([0.0, 1.0])
        assets = SimpleNamespace(
            layout=layout,
            invariant_operator=invariant,
            tss_weights=tss,
            equality_count=6,
            engineering=SimpleNamespace(clarifier_volume_m3=30.0),
        )
        theta_symbol = ca.MX.sym("network_theta", 7)
        feed_symbol = ca.MX.sym("network_feed", 2)
        symbolic = symbolic_network_operators(theta_symbol, feed_symbol, assets)
        evaluate = ca.Function(
            "reduced_network_operator_test",
            [theta_symbol, feed_symbol],
            [
                symbolic.equality_matrix,
                symbolic.equality_rhs,
                symbolic.inequality_matrix,
            ],
        )
        controls = np.asarray([24.0, 0.2, 0.3, 0.4, 1.0, 0.5, 0.1])
        feed = np.asarray([2.0, 10.0])
        symbolic_values = evaluate(controls, feed)
        numeric = build_network_operators(
            feed,
            internal_recycle=controls[4],
            return_recycle=controls[5],
            waste_fraction=controls[6],
            invariant_operator=invariant,
            tss_weights=tss,
            layout=layout,
            clarifier_volume_m3=30.0,
        )
        np.testing.assert_allclose(symbolic_values[0], numeric.equality_matrix)
        np.testing.assert_allclose(
            np.asarray(symbolic_values[1]).reshape(-1), numeric.equality_rhs
        )
        np.testing.assert_allclose(symbolic_values[2], numeric.inequality_matrix)
        rng = np.random.default_rng(20260827)
        closure_controls = rng.uniform(0.1, 1.0, size=(120, 7))
        closure_influents = rng.uniform(1.0, 20.0, size=(120, 2))
        closure_target = np.exp(
            0.2 + 0.1 * closure_controls[:, 4] - 0.02 * closure_influents[:, 1]
        )
        closure = LogOverflowTSSClosure.fit_ridge(
            closure_controls, closure_influents, closure_target, ridge_penalty=0.0001
        )
        closure_assets = SimpleNamespace(
            layout=layout,
            invariant_operator=invariant,
            tss_weights=tss,
            equality_count=7,
            engineering=SimpleNamespace(clarifier_volume_m3=30.0),
            overflow_closure=closure,
        )
        symbolic_closure = symbolic_network_operators(
            theta_symbol, feed_symbol, closure_assets
        )
        evaluate_closure = ca.Function(
            "reduced_network_operator_closure_test",
            [theta_symbol, feed_symbol],
            [
                symbolic_closure.equality_matrix,
                symbolic_closure.equality_rhs,
                symbolic_closure.inequality_matrix,
            ],
        )
        closure_value = float(closure.predict(controls, feed))
        numeric_closure = build_network_operators(
            feed,
            internal_recycle=controls[4],
            return_recycle=controls[5],
            waste_fraction=controls[6],
            invariant_operator=invariant,
            tss_weights=tss,
            layout=layout,
            clarifier_volume_m3=30.0,
            overflow_tss_closure=closure_value,
        )
        symbolic_closure_values = evaluate_closure(controls, feed)
        np.testing.assert_allclose(
            symbolic_closure_values[0], numeric_closure.equality_matrix
        )
        np.testing.assert_allclose(
            np.asarray(symbolic_closure_values[1]).reshape(-1),
            numeric_closure.equality_rhs,
        )
        np.testing.assert_allclose(
            symbolic_closure_values[2], numeric_closure.inequality_matrix
        )
