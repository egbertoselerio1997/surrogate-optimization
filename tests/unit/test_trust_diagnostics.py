import surrogate_optimization.surrogate.projection as module_surrogate_projection
import surrogate_optimization.optimization.trust as module_optimization_trust
import surrogate_optimization.optimization.mechanistic as module_optimization_mechanistic
import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from surrogate_optimization.plant.operating_point import ClarifierParameters
from surrogate_optimization.plant.definitions import INVARIANT_MATRIX
from surrogate_optimization.plant.definitions import N_COMPONENTS
from surrogate_optimization.plant.definitions import N_STAGES
from surrogate_optimization.plant.definitions import TSS_VECTOR
from surrogate_optimization.surrogate.projection import NetworkLayout
from surrogate_optimization.surrogate.projection import build_network_operators
from surrogate_optimization.surrogate.projection import no_conversion_feasible_state
from surrogate_optimization.optimization.mechanistic import fit_mechanistic_assets


class _Projector:
    states: list[np.ndarray] = []
    accepted: list[bool] = []

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self._row = 0

    def project(self, *args, **kwargs):
        del args, kwargs
        row = self._row
        self._row += 1
        return SimpleNamespace(state=self.states[row], accepted=self.accepted[row])


class TrustCalibrationProjectionTests(unittest.TestCase):
    def test_parallel_projection_matches_serial_and_resumes(self) -> None:
        layout = NetworkLayout(layer_count=3)
        rng = np.random.default_rng(81)
        rows = 4
        lower = np.asarray([6.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.001])
        upper = np.asarray([36.0, 1.0, 1.0, 1.0, 4.0, 1.25, 0.05])
        controls = lower + rng.uniform(0.2, 0.8, size=(rows, 7)) * (upper - lower)
        influents = rng.uniform(0.5, 2.0, size=(rows, N_COMPONENTS))

        def feasible(controls, influent):
            operators = build_network_operators(
                influent,
                internal_recycle=controls[4],
                return_recycle=controls[5],
                waste_fraction=controls[6],
                invariant_operator=INVARIANT_MATRIX,
                tss_weights=TSS_VECTOR,
                layout=layout,
            )
            return no_conversion_feasible_state(
                influent, operators=operators, tss_weights=TSS_VECTOR
            )

        mechanistic_responses = np.vstack(
            [
                feasible(controls, influent)
                for controls, influent in zip(controls, influents, strict=True)
            ]
        )
        full_targets = np.column_stack(
            (
                mechanistic_responses[:, :-1],
                np.tile(np.asarray([100.0, 200.0, 300.0]), (rows, 1)),
            )
        )
        mechanistic_assets = fit_mechanistic_assets(
            controls,
            influents,
            full_targets,
            clarifier=ClarifierParameters(
                layer_count=3, feed_layer=1, layer_volume=2000.0
            ),
        )
        raw = mechanistic_responses + rng.normal(
            0.0, 0.001, size=mechanistic_responses.shape
        )
        model = SimpleNamespace(
            response_scale=np.maximum(1.0, np.std(mechanistic_responses, axis=0))
        )
        arguments = (
            model,
            controls,
            influents,
            mechanistic_responses,
            raw,
            mechanistic_assets,
        )
        serial = module_optimization_trust.calibrate_trust_diagnostics(
            *arguments, layout=layout, parallel_workers=1, batch_size=2
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            parallel = module_optimization_trust.calibrate_trust_diagnostics(
                *arguments,
                layout=layout,
                parallel_workers=2,
                batch_size=2,
                checkpoint_directory=checkpoint,
                checkpoint_contract="whole-development-test",
            )
            with patch.object(
                module_optimization_trust,
                "_trust_projection_batch",
                side_effect=AssertionError("completed batches were recomputed"),
            ):
                resumed = module_optimization_trust.calibrate_trust_diagnostics(
                    *arguments,
                    layout=layout,
                    parallel_workers=1,
                    batch_size=2,
                    checkpoint_directory=checkpoint,
                    checkpoint_contract="whole-development-test",
                )
        for observed in (parallel, resumed):
            np.testing.assert_allclose(
                observed.out_of_fold_projected,
                serial.out_of_fold_projected,
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_array_equal(
                observed.out_of_fold_projection_accepted,
                serial.out_of_fold_projection_accepted,
            )
            np.testing.assert_allclose(
                observed.development_values,
                serial.development_values,
                rtol=0.0,
                atol=1e-12,
            )
            self.assertEqual(observed.correction_limit, serial.correction_limit)
            self.assertEqual(observed.split_limit, serial.split_limit)
            self.assertEqual(observed.reactor_limit, serial.reactor_limit)

    def _calibrate(self, states: list[np.ndarray], accepted: list[bool]):
        layout = NetworkLayout(layer_count=3)
        row_count = len(states)
        _Projector.states = states
        _Projector.accepted = accepted
        model = SimpleNamespace(response_scale=np.ones(layout.state_size))
        controls = np.zeros((row_count, 7))
        influents = np.ones((row_count, N_COMPONENTS))
        mechanistic_responses = np.ones((row_count, layout.state_size))
        raw = np.ones_like(mechanistic_responses)
        row_scales = SimpleNamespace(equality=np.ones(1), inequality=np.ones(1))
        operators = SimpleNamespace(
            equality_matrix=np.zeros((1, layout.state_size)),
            equality_rhs=np.zeros(1),
            inequality_matrix=np.zeros((1, layout.state_size)),
        )
        mechanistic_assets = SimpleNamespace(
            balance_scale=np.ones(N_STAGES * N_COMPONENTS + layout.layer_count),
            clarifier=SimpleNamespace(
                layer_volume=600.0, layer_count=layout.layer_count
            ),
        )
        balance = np.zeros(N_STAGES * N_COMPONENTS)
        with (
            patch.object(module_surrogate_projection, "PhysicalProjector", _Projector),
            patch.object(
                module_surrogate_projection,
                "fit_network_row_scales",
                return_value=row_scales,
            ),
            patch.object(
                module_surrogate_projection,
                "build_network_operators",
                return_value=operators,
            ),
            patch.object(
                module_optimization_mechanistic,
                "_smooth_reactor_residual",
                return_value=balance,
            ),
        ):
            return module_optimization_trust.calibrate_trust_diagnostics(
                model,
                controls,
                influents,
                mechanistic_responses,
                raw,
                mechanistic_assets,
                layout=layout,
            )

    def test_finite_rejected_projection_is_retained_and_exposed(self) -> None:
        layout = NetworkLayout(layer_count=3)
        states = [np.ones(layout.state_size), np.full(layout.state_size, 2.0)]
        result = self._calibrate(states, [False, True])
        self.assertEqual(result.development_values.shape, (2, 3))
        self.assertFalse(hasattr(result, "flux_limit"))
        self.assertFalse(hasattr(result.callbacks, "flux_rows"))
        np.testing.assert_array_equal(
            result.out_of_fold_projection_accepted, np.array([False, True])
        )
        np.testing.assert_allclose(result.out_of_fold_projected, states)

    def test_nonfinite_or_wrong_shape_projection_remains_fatal(self) -> None:
        layout = NetworkLayout(layer_count=3)
        invalid_states = (
            np.full(layout.state_size, np.nan),
            np.ones(layout.state_size - 1),
        )
        for state in invalid_states:
            with (
                self.subTest(shape=state.shape),
                self.assertRaisesRegex(RuntimeError, "invalid state"),
            ):
                self._calibrate([state], [False])
