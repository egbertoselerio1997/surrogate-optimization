import surrogate_optimization.plant.definitions as module_plant_definitions
import surrogate_optimization.config as module_config
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
from tests.support.profiles import REDUCED_PROFILE
from surrogate_optimization.config import StudyProfile
from surrogate_optimization.validation.assessment import (
    assess_raw_projected_mechanistic,
)
from surrogate_optimization.data.design import clarifier_for
from surrogate_optimization.surrogate.training import (
    cross_validate_log_overflow_closure,
)
from surrogate_optimization.data.design import create_design
from surrogate_optimization.validation.physical import reduce_mechanistic_responses
from surrogate_optimization.plant.operating_point import OperatingPoint
from surrogate_optimization.plant.definitions import INVARIANT_MATRIX
from surrogate_optimization.plant.definitions import NOMINAL_INFLUENT
from surrogate_optimization.plant.definitions import TSS_VECTOR
from surrogate_optimization.plant.model import solve_steady_state
from surrogate_optimization.surrogate.regression import LeastSquaresDiagnostics
from surrogate_optimization.surrogate.projection import NetworkLayout
from surrogate_optimization.surrogate.regression import QuadraticFeatureMap
from surrogate_optimization.surrogate.regression import QuadraticSurrogate
from surrogate_optimization.surrogate.projection import build_network_operators
from surrogate_optimization.surrogate.projection import no_conversion_feasible_state


class StudyContractTests(unittest.TestCase):
    def test_log_overflow_cross_validation_is_complete_and_out_of_fold(self) -> None:
        layout = NetworkLayout(
            stage_count=1,
            component_count=2,
            layer_count=3,
            soluble_indices=(0,),
            particulate_indices=(1,),
        )
        rng = np.random.default_rng(20260827)
        controls = rng.uniform(0.1, 0.9, size=(120, 7))
        controls[:, 6] *= 0.05
        influents = rng.uniform(1.0, 20.0, size=(120, 2))
        exact = np.exp(0.25 + 0.1 * controls[:, 0] - 0.03 * influents[:, 1])
        mechanistic_responses = np.ones((120, layout.state_size))
        mechanistic_responses[:, layout.overflow_flow_slice.start + 1] = (
            1.0 - controls[:, 6]
        ) * exact
        with (
            patch.object(
                module_plant_definitions, "TSS_VECTOR", np.asarray([0.0, 1.0])
            ),
            patch.object(module_config, "RIDGE_GRID", np.asarray([0.0001])),
        ):
            result = cross_validate_log_overflow_closure(
                controls, influents, mechanistic_responses, layout=layout
            )
        self.assertEqual(len(result.scores), 5)
        self.assertEqual(set(result.fold_membership), {1, 2, 3, 4, 5})
        self.assertEqual(result.out_of_fold_tss.shape, (120,))
        self.assertTrue(np.all(np.isfinite(result.out_of_fold_tss)))
        self.assertTrue(np.all(result.out_of_fold_tss > 0.0))
        np.testing.assert_allclose(result.exact_overflow_tss, exact)

    def test_parallel_holdout_projection_matches_serial_and_resumes(self) -> None:
        profile = StudyProfile(
            name="parallel_unit",
            development_candidate_count=4,
            holdout_candidate_count=3,
            robustness_count=1,
            layer_count=3,
            development_seed=1,
            holdout_seed=2,
            robustness_seed=3,
            parallel_workers=2,
            scientifically_eligible=False,
            enforce_admission_gate=False,
        )
        layout = NetworkLayout(layer_count=profile.layer_count)
        rng = np.random.default_rng(441)
        development_controls = np.tile(
            np.asarray([18.0, 0.2, 0.3, 0.4, 2.0, 0.75, 0.02]),
            (profile.development_candidate_count, 1),
        )
        holdout_controls = development_controls[
            : profile.holdout_candidate_count
        ].copy()
        development_influents = rng.uniform(
            0.5, 2.0, size=(profile.development_candidate_count, 20)
        )
        holdout_influents = rng.uniform(
            0.5, 2.0, size=(profile.holdout_candidate_count, 20)
        )

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

        development_responses = np.vstack(
            [
                feasible(controls, influent)
                for controls, influent in zip(
                    development_controls, development_influents, strict=True
                )
            ]
        )
        holdout_responses = np.vstack(
            [
                feasible(controls, influent)
                for controls, influent in zip(
                    holdout_controls, holdout_influents, strict=True
                )
            ]
        )
        feature_controls = rng.uniform(0.0, 1.0, size=(450, 7))
        feature_influents = rng.uniform(0.1, 3.0, size=(450, 20))
        feature_map = QuadraticFeatureMap.fit(feature_controls, feature_influents)
        diagnostics = LeastSquaresDiagnostics(
            sample_count=450,
            feature_count=feature_map.feature_count,
            response_count=layout.state_size,
            rank_tolerance=1e-12,
            smallest_singular_value=1.0,
            largest_singular_value=2.0,
            condition_number=2.0,
            optimality_residual=0.0,
            coefficient_agreement=0.0,
            acceptance_threshold=1e-12,
        )
        model = QuadraticSurrogate(
            feature_map=feature_map,
            response_center=development_responses[0],
            response_scale=np.maximum(1.0, np.std(development_responses, axis=0)),
            coefficients=np.zeros((layout.state_size, feature_map.feature_count)),
            diagnostics=diagnostics,
            ridge_penalty=0.1,
        )
        arguments = (
            model,
            development_controls,
            development_influents,
            development_responses,
            holdout_controls,
            holdout_influents,
            holdout_responses,
            profile,
        )
        serial = assess_raw_projected_mechanistic(
            *arguments, parallel_workers=1, batch_size=2
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            parallel = assess_raw_projected_mechanistic(
                *arguments,
                parallel_workers=2,
                batch_size=2,
                checkpoint_directory=checkpoint,
                checkpoint_contract="whole-holdout-test",
            )
            with patch(
                "surrogate_optimization.validation.assessment._holdout_projection_batch",
                side_effect=AssertionError("completed batches were recomputed"),
            ):
                resumed = assess_raw_projected_mechanistic(
                    *arguments,
                    parallel_workers=1,
                    batch_size=2,
                    checkpoint_directory=checkpoint,
                    checkpoint_contract="whole-holdout-test",
                )
        for observed in (parallel, resumed):
            np.testing.assert_allclose(observed.raw, serial.raw, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(
                observed.projected, serial.projected, rtol=0.0, atol=1e-12
            )
            np.testing.assert_allclose(
                observed.projected_targets,
                serial.projected_targets,
                rtol=0.0,
                atol=1e-12,
            )
            pd.testing.assert_frame_equal(observed.metrics, serial.metrics)
            pd.testing.assert_frame_equal(observed.violations, serial.violations)
            pd.testing.assert_frame_equal(observed.feasibility, serial.feasibility)
            pd.testing.assert_frame_equal(
                observed.qp_diagnostics.drop(columns="elapsed_ns"),
                serial.qp_diagnostics.drop(columns="elapsed_ns"),
            )
            self.assertEqual(
                list(
                    zip(
                        observed.qp_diagnostics["row"],
                        observed.qp_diagnostics["projection_input"],
                    )
                ),
                [
                    (0, "raw_prediction"),
                    (0, "mechanistic_target"),
                    (1, "raw_prediction"),
                    (1, "mechanistic_target"),
                    (2, "raw_prediction"),
                    (2, "mechanistic_target"),
                ],
            )

    def test_requested_verification_dimensions_and_independent_blocks(self) -> None:
        design = create_design(REDUCED_PROFILE)
        self.assertEqual(design["development_controls"].shape, (400, 7))
        self.assertEqual(design["development_influents"].shape, (400, 20))
        self.assertEqual(design["holdout_controls"].shape, (100, 7))
        self.assertEqual(design["robustness_influents"].shape, (5, 20))
        self.assertFalse(
            np.array_equal(
                design["development_influents"][:100], design["holdout_influents"]
            )
        )
        feature_map = QuadraticFeatureMap.fit(
            design["development_controls"], design["development_influents"]
        )
        self.assertEqual(feature_map.feature_count, 406)
        self.assertEqual(NetworkLayout(layer_count=5).state_size, 161)
        self.assertEqual(REDUCED_PROFILE.mechanistic_response_count, 165)
        self.assertEqual(REDUCED_PROFILE.surrogate_response_count, 161)

    def test_layer_resolved_target_reduces_to_one_inventory_coordinate(self) -> None:
        full = np.arange(REDUCED_PROFILE.mechanistic_response_count, dtype=float)
        reduced = reduce_mechanistic_responses(full, REDUCED_PROFILE.layer_count)
        self.assertEqual(reduced.shape, (REDUCED_PROFILE.surrogate_response_count,))
        np.testing.assert_array_equal(reduced[:160], full[:160])
        self.assertEqual(reduced[-1], 1200.0 * np.sum(full[160:165]))

    def test_reduction_rejects_unequal_layer_volumes(self) -> None:
        full = np.arange(REDUCED_PROFILE.mechanistic_response_count, dtype=float)
        with self.assertRaisesRegex(ValueError, "must be equal"):
            reduce_mechanistic_responses(
                full,
                REDUCED_PROFILE.layer_count,
                layer_volumes_m3=np.asarray([1000.0, 1100.0, 1200.0, 1300.0, 1400.0]),
            )

    def test_five_layer_independent_aeration_mechanism_closes(self) -> None:
        clarifier = clarifier_for(REDUCED_PROFILE)
        operating = OperatingPoint(18.0, 0.2, 0.3, 0.4, 2.0, 0.75, 0.02)
        result = solve_steady_state(
            operating, NOMINAL_INFLUENT, starts=(1,), clarifier=clarifier
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.state.shape, (105,))
        self.assertEqual(result.target.shape, (165,))
        self.assertLessEqual(result.diagnostics["plant_boundary_residual"], 1e-08)
        self.assertGreaterEqual(result.diagnostics["minimum_state"], -1e-10)
