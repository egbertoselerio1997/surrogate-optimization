from surrogate_optimization.paths import REPOSITORY_ROOT
import surrogate_optimization.workflow.optimization as module_workflow_optimization
import surrogate_optimization.workflow.generation as module_workflow_generation
import surrogate_optimization.workflow.assessment as module_workflow_assessment
import surrogate_optimization.validation.assessment as module_validation_assessment
import surrogate_optimization.surrogate.training as module_surrogate_training
import surrogate_optimization.runtime.protocols as module_runtime_protocols
import surrogate_optimization.runtime.contracts as module_runtime_contracts
import surrogate_optimization.runtime.artifacts as module_runtime_artifacts
import surrogate_optimization.plant.definitions as module_plant_definitions
import surrogate_optimization.optimization.trust as module_optimization_trust
import surrogate_optimization.optimization.surrogate as module_optimization_surrogate
import surrogate_optimization.optimization.mechanistic as module_optimization_mechanistic
import surrogate_optimization.data.generation as module_data_generation
import surrogate_optimization.data.design as module_data_design
import surrogate_optimization.config as module_config
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from surrogate_optimization.config import PRODUCTION_PROFILE
from tests.support.profiles import REDUCED_PROFILE
from surrogate_optimization.validation.assessment import AssessmentResult
from surrogate_optimization.config import StudyProfile
from surrogate_optimization.data.design import create_design
from surrogate_optimization.surrogate.regression import LeastSquaresDiagnostics
from surrogate_optimization.surrogate.regression import QuadraticFeatureMap
from surrogate_optimization.surrogate.regression import QuadraticSurrogate

ROOT = REPOSITORY_ROOT


def tiny_profile() -> StudyProfile:
    return StudyProfile(
        name="tiny_test",
        development_candidate_count=5,
        holdout_candidate_count=2,
        robustness_count=1,
        layer_count=3,
        development_seed=1,
        holdout_seed=2,
        robustness_seed=3,
        parallel_workers=1,
        scientifically_eligible=False,
        enforce_admission_gate=False,
    )


def tiny_design(profile: StudyProfile) -> dict[str, object]:
    decision = 0.5 * (module_config.DECISION_LOWER + module_config.DECISION_UPPER)
    influent = 0.5 * (
        module_plant_definitions.INFLUENT_LOWER
        + module_plant_definitions.INFLUENT_UPPER
    )
    return {
        "development_controls": np.tile(
            decision, (profile.development_candidate_count, 1)
        ),
        "development_influents": np.tile(
            influent, (profile.development_candidate_count, 1)
        ),
        "holdout_controls": np.tile(decision, (profile.holdout_candidate_count, 1)),
        "holdout_influents": np.tile(influent, (profile.holdout_candidate_count, 1)),
        "robustness_influents": np.tile(influent, (profile.robustness_count, 1)),
        "generators": {
            "development": {"seed": profile.development_seed},
            "holdout": {"seed": profile.holdout_seed},
            "robustness": {"seed": profile.robustness_seed},
        },
    }


def accepted_diagnostics(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row": np.arange(count),
            "accepted": np.ones(count, dtype=bool),
            "root_difference_inf": np.full(count, 1e-08),
            "branch_agreement": np.ones(count, dtype=bool),
            "mass_residual_start_1": np.full(count, 1e-10),
            "mass_residual_start_2": np.full(count, 1e-10),
            "state_negativity_start_1": np.zeros(count),
            "state_negativity_start_2": np.zeros(count),
            "rate_negativity_start_1": np.zeros(count),
            "rate_negativity_start_2": np.zeros(count),
            "largest_real_eigenvalue_start_1": np.full(count, -1.0),
            "largest_real_eigenvalue_start_2": np.full(count, -1.0),
            "stability_agreement_start_1": np.full(count, 1e-08),
            "stability_agreement_start_2": np.full(count, 1e-08),
            "feed_tss_start_1": np.full(count, 100.0),
            "feed_tss_start_2": np.full(count, 100.0),
            "external_solids_loss_start_1": np.full(count, 10.0),
            "external_solids_loss_start_2": np.full(count, 10.0),
        }
    )


def publish_mock_generation_block(
    output: Path,
    controls: np.ndarray,
    influents: np.ndarray,
    profile: StudyProfile,
    *,
    accepted: bool = True,
) -> module_data_generation.MechanisticBlockResult:
    count = len(controls)
    mechanistic_responses = np.full((count, profile.response_count), 2.0)
    diagnostics = accepted_diagnostics(count)
    if not accepted:
        diagnostics.loc[0, "accepted"] = False
    provenance = pd.DataFrame(
        {
            "accepted_slot": np.arange(count),
            "source_candidate_id": [f"candidate-{index}" for index in range(count)],
            "source_candidate_index": np.arange(count),
        }
    )
    attempt_rows = []
    for index in range(count):
        relative = f"rows/row_{index:06d}.npz"
        checkpoint = output / relative
        module_runtime_artifacts.atomic_npz(checkpoint, value=np.asarray([index]))
        attempt_rows.append(
            {
                "candidate_id": f"candidate-{index}",
                "accepted": bool(diagnostics.loc[index, "accepted"]),
                "rejection_reason": "accepted"
                if bool(diagnostics.loc[index, "accepted"])
                else "branch_disagreement",
                "checkpoint_path": relative,
                "checkpoint_sha256": module_runtime_contracts.file_digest(checkpoint),
            }
        )
    attempts = pd.DataFrame(attempt_rows)
    module_runtime_artifacts.atomic_npz(
        output / "accepted_mechanistic_responses.npz",
        mechanistic_responses=mechanistic_responses,
        states_start_1=np.zeros((count, 1)),
        states_start_2=np.zeros((count, 1)),
    )
    module_runtime_artifacts.atomic_npz(
        output / "accepted_inputs.npz",
        controls=controls,
        influents=influents,
        source_candidate_id=provenance["source_candidate_id"].to_numpy(str),
    )
    module_runtime_artifacts.atomic_dataframe(
        output / "accepted_diagnostics.csv", diagnostics
    )
    module_runtime_artifacts.atomic_dataframe(output / "all_attempts.csv", attempts)
    module_runtime_artifacts.atomic_dataframe(
        output / "accepted_provenance.csv", provenance
    )
    module_runtime_artifacts.atomic_dataframe(
        output / "candidate_checkpoint_summary.csv", attempts
    )
    module_runtime_artifacts.atomic_json(
        output / "generation_summary.json",
        {"candidate_count": count, "accepted_count": count, "rejected_count": 0},
    )
    return module_data_generation.MechanisticBlockResult(
        controls=np.asarray(controls),
        influents=np.asarray(influents),
        mechanistic_responses=mechanistic_responses,
        diagnostics=diagnostics,
        attempts=attempts,
        provenance=provenance,
    )


def assessment_fixture(raw_nrmse: float = 0.8) -> AssessmentResult:
    holdout_candidate_count, response_count = (2, 3)
    metrics = pd.DataFrame(
        [
            {
                "method": "raw",
                "block": "complete_response",
                "coordinate": "ALL",
                "nrmse": raw_nrmse,
            }
        ]
    )
    qp = pd.DataFrame(
        [
            {"row": row, "projection_input": kind, "accepted": True}
            for row in range(holdout_candidate_count)
            for kind in ("raw_prediction", "mechanistic_target")
        ]
    )
    violations = pd.DataFrame(
        [
            {
                "case": f"holdout_{row:04d}",
                "method": method,
                "mass_conservation_violation_max": 2.0 if method == "raw" else 1e-09,
                "nonnegativity_violation_max": 3.0 if method == "raw" else 1e-12,
            }
            for row in range(holdout_candidate_count)
            for method in ("raw", "projected", "mechanistic")
        ]
    )
    feasibility = pd.DataFrame(
        {"row": range(holdout_candidate_count), "bound_passed": [True, True]}
    )
    values = np.zeros((holdout_candidate_count, response_count))
    return AssessmentResult(
        metrics=metrics,
        violations=violations,
        qp_diagnostics=qp,
        feasibility=feasibility,
        raw=values.copy(),
        projected=values.copy(),
        projected_targets=values.copy(),
    )


TRUST_LIMITS = {
    "correction": 0.4,
    "regularized_leverage": 1.0,
    "particulate_split": 1.0,
    "reactor_residual": 1.0,
}


class RunContractTests(unittest.TestCase):
    def test_full_profile_attempts_eight_thousand_plus_two_thousand(self) -> None:
        module_runtime_contracts.validate_production_profile(PRODUCTION_PROFILE)
        self.assertEqual(PRODUCTION_PROFILE.development_candidate_count, 8000)
        self.assertEqual(PRODUCTION_PROFILE.holdout_candidate_count, 2000)
        self.assertEqual(
            PRODUCTION_PROFILE.development_candidate_count
            + PRODUCTION_PROFILE.holdout_candidate_count,
            10000,
        )
        self.assertEqual(PRODUCTION_PROFILE.robustness_count, 10)
        self.assertEqual(PRODUCTION_PROFILE.layer_count, 10)
        self.assertTrue(PRODUCTION_PROFILE.scientifically_eligible)
        self.assertTrue(PRODUCTION_PROFILE.enforce_admission_gate)

    def test_full_design_is_distinct_from_smoke_test(self) -> None:
        self.assertNotEqual(
            PRODUCTION_PROFILE.development_seed, REDUCED_PROFILE.development_seed
        )
        self.assertNotEqual(
            PRODUCTION_PROFILE.holdout_seed, REDUCED_PROFILE.holdout_seed
        )
        design = create_design(PRODUCTION_PROFILE)
        module_data_design.validate_design(design, PRODUCTION_PROFILE)
        self.assertEqual(design["development_controls"].shape, (8000, 7))
        self.assertEqual(design["development_influents"].shape, (8000, 20))
        self.assertEqual(design["holdout_controls"].shape, (2000, 7))
        self.assertEqual(design["holdout_influents"].shape, (2000, 20))
        self.assertEqual(design["robustness_influents"].shape, (10, 20))

    def test_json_profile_matches_executable_profile(self) -> None:
        payload = json.loads((ROOT / "config/parameters.json").read_text())
        profile = payload["profiles"]["production"]
        self.assertEqual(profile["development_candidate_count"], 8000)
        self.assertEqual(profile["holdout_candidate_count"], 2000)
        self.assertTrue(profile["continue_after_study_admission_gate_failure"])
        self.assertEqual(
            module_runtime_protocols.ASSESSMENT_GATE_EXECUTION_POLICY,
            "advisory_continue",
        )

    def test_run_directory_rejects_unsafe_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accepted = module_runtime_contracts.resolve_run_directory(
                "production_trial1", root
            )
            self.assertEqual(accepted.parent, root.resolve())
            for value in ("../escape", "a/b", "a\\b", "", "CON"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    module_runtime_contracts.resolve_run_directory(value, root)

    def test_contract_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            first = {"source_digest": "a", "fixed_dataset_total": 5000}
            module_runtime_contracts.establish_contract(run, first)
            module_runtime_contracts.establish_contract(run, dict(first))
            with self.assertRaisesRegex(RuntimeError, "choose a new run id"):
                module_runtime_contracts.establish_contract(
                    run, {**first, "source_digest": "b"}
                )

    def test_atomic_helpers_publish_no_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module_runtime_artifacts.atomic_json(
                root / "item.json", {"answer": np.int64(42)}
            )
            module_runtime_artifacts.atomic_npz(root / "item.npz", values=np.arange(3))
            module_runtime_artifacts.atomic_dataframe(
                root / "item.csv", pd.DataFrame({"x": [1, 2]})
            )
            self.assertEqual(
                json.loads((root / "item.json").read_text()), {"answer": 42}
            )
            with np.load(root / "item.npz", allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored["values"], np.arange(3))
            self.assertEqual(pd.read_csv(root / "item.csv")["x"].tolist(), [1, 2])
            self.assertFalse(any((path.suffix == ".tmp" for path in root.iterdir())))

    def test_generation_reuses_validated_block_manifests(self) -> None:
        profile = tiny_profile()
        design = tiny_design(profile)
        source_files = {"unit-test": "bound-source"}

        def generator(controls, influents, supplied_profile, output, *, block=None):
            return publish_mock_generation_block(
                output, controls, influents, supplied_profile
            )

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with (
                patch.object(
                    module_data_generation,
                    "generate_mechanistic_block_from_fixed_design",
                    side_effect=generator,
                ) as mocked,
                patch.object(module_runtime_contracts, "assert_source_unchanged"),
            ):
                first = module_workflow_generation.run_generation(
                    run, design, profile=profile, source_files=source_files
                )
                self.assertEqual(mocked.call_count, 2)
                second = module_workflow_generation.run_generation(
                    run, design, profile=profile, source_files=source_files
                )
                self.assertEqual(mocked.call_count, 2)
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            summary = pd.read_csv(run / "metrics/mechanistic_generation_summary.csv")
            self.assertTrue(summary["reused_complete_checkpoint"].all())
            self.assertTrue((run / "datasets/effective_design.npz").is_file())

    def test_generation_gate_rejects_failed_row_before_publishing_manifest(
        self,
    ) -> None:
        profile = tiny_profile()
        design = tiny_design(profile)

        def generator(controls, influents, supplied_profile, output, *, block=None):
            return publish_mock_generation_block(
                output, controls, influents, supplied_profile, accepted=False
            )

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with (
                patch.object(
                    module_data_generation,
                    "generate_mechanistic_block_from_fixed_design",
                    side_effect=generator,
                ),
                patch.object(module_runtime_contracts, "assert_source_unchanged"),
            ):
                with self.assertRaisesRegex(RuntimeError, "unaccepted"):
                    module_workflow_generation.run_generation(
                        run,
                        design,
                        profile=profile,
                        source_files={"unit-test": "bound-source"},
                    )
            self.assertFalse(
                (run / "datasets/development/block_complete.json").exists()
            )

    def test_accepted_subset_becomes_effective_design_without_mutating_base(
        self,
    ) -> None:
        profile = tiny_profile()
        design = tiny_design(profile)
        base_development = design["development_controls"].copy()

        def generator(controls, influents, supplied_profile, output, *, block=None):
            accepted_controls = np.asarray(controls).copy()
            accepted_influents = np.asarray(influents).copy()
            if block == "development":
                accepted_controls = np.delete(accepted_controls, 1, axis=0)
                accepted_influents = np.delete(accepted_influents, 1, axis=0)
            return publish_mock_generation_block(
                output, accepted_controls, accepted_influents, supplied_profile
            )

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with (
                patch.object(
                    module_data_generation,
                    "generate_mechanistic_block_from_fixed_design",
                    side_effect=generator,
                ),
                patch.object(module_runtime_contracts, "assert_source_unchanged"),
            ):
                result = module_workflow_generation.run_generation(
                    run,
                    design,
                    profile=profile,
                    source_files={"unit-test": "bound-source"},
                )
            np.testing.assert_array_equal(
                design["development_controls"], base_development
            )
            self.assertEqual(len(result.design["development_controls"]), 4)
            with np.load(run / "datasets/effective_design.npz") as stored:
                np.testing.assert_array_equal(
                    stored["development_controls"],
                    result.design["development_controls"],
                )

    def test_admission_gate_is_development_only_and_tracks_holdout(self) -> None:
        gate = module_workflow_assessment.evaluate_admission_gate(
            assessment_fixture(),
            correction_limit=0.4,
            trust_limits=TRUST_LIMITS,
            development_oof_projection_accepted=np.ones(5, dtype=bool),
            development_oof_complete_nrmse=0.5,
            development_oof_inventory_nrmse=0.5,
            holdout_candidate_count=2,
        )
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["physical_audit_maxima"]["raw"]["passed"])
        self.assertTrue(gate["physical_audit_maxima"]["projected"]["passed"])
        self.assertTrue(gate["physical_audit_maxima"]["mechanistic"]["passed"])
        self.assertEqual(gate["admission_gate_scope"], "development_only")
        self.assertFalse(gate["post_selection_holdout_checks_are_admission_gates"])
        holdout_failure = assessment_fixture()
        holdout_failure.qp_diagnostics.loc[:, "accepted"] = False
        holdout_failure.feasibility.loc[:, "bound_passed"] = False
        holdout_failure.violations.loc[
            holdout_failure.violations["method"].isin(["projected", "mechanistic"]),
            "mass_conservation_violation_max",
        ] = 2.0
        descriptive = module_workflow_assessment.evaluate_admission_gate(
            holdout_failure,
            correction_limit=0.4,
            trust_limits=TRUST_LIMITS,
            development_oof_projection_accepted=np.ones(5, dtype=bool),
            development_oof_complete_nrmse=0.5,
            development_oof_inventory_nrmse=0.5,
            holdout_candidate_count=2,
        )
        self.assertTrue(descriptive["passed"])
        self.assertFalse(descriptive["all_projection_qp_audits_passed"])
        self.assertFalse(descriptive["all_finite_distance_bounds_passed"])
        self.assertFalse(descriptive["projected_physical_audits_passed"])
        self.assertFalse(descriptive["mechanistic_physical_audits_passed"])
        failed = module_workflow_assessment.evaluate_admission_gate(
            assessment_fixture(),
            correction_limit=0.51,
            trust_limits={**TRUST_LIMITS, "correction": 0.51},
            development_oof_projection_accepted=np.ones(5, dtype=bool),
            development_oof_complete_nrmse=0.5,
            development_oof_inventory_nrmse=0.5,
            holdout_candidate_count=2,
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["execution_policy"], "advisory_continue")
        self.assertTrue(failed["optimization_permitted"])
        self.assertEqual(
            failed["failure_action"],
            "record advisory failure and continue without refitting",
        )
        self.assertTrue(
            module_workflow_assessment.assessment_gate_allows_optimization(False)
        )

    def test_development_oof_projection_rejection_is_advisory(self) -> None:
        gate = module_workflow_assessment.evaluate_admission_gate(
            assessment_fixture(),
            correction_limit=0.4,
            trust_limits=TRUST_LIMITS,
            development_oof_projection_accepted=np.array(
                [True, False, True], dtype=bool
            ),
            development_oof_complete_nrmse=0.5,
            development_oof_inventory_nrmse=0.5,
            holdout_candidate_count=2,
        )
        self.assertFalse(gate["all_development_oof_projection_qp_audits_passed"])
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["optimization_permitted"])

    def test_inventory_coordinate_has_its_own_development_oof_gate(self) -> None:
        gate = module_workflow_assessment.evaluate_admission_gate(
            assessment_fixture(),
            correction_limit=0.4,
            trust_limits=TRUST_LIMITS,
            development_oof_projection_accepted=np.ones(5, dtype=bool),
            development_oof_complete_nrmse=0.5,
            development_oof_inventory_nrmse=1.01,
            holdout_candidate_count=2,
        )
        self.assertFalse(gate["development_oof_clarifier_inventory_nrmse_below_one"])
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["post_selection_holdout_is_confirmatory"])

    def test_assessment_persists_development_projection_acceptance(self) -> None:
        profile = tiny_profile()
        design = tiny_design(profile)
        response_count = profile.surrogate_response_count
        mechanistic_response_count = profile.mechanistic_response_count
        model = SimpleNamespace(
            response_count=response_count,
            response_scale=np.ones(response_count),
            feature_map=SimpleNamespace(
                transform=lambda controls, influents: np.ones((len(controls), 1))
            ),
        )
        callbacks = SimpleNamespace(
            split_rows=lambda controls, raw, projected, influent: np.array([3.0, 4.0]),
            reactor_rows=lambda controls, raw, projected, influent: np.array(
                [6.0, 8.0]
            ),
        )
        accepted = np.array([True, False, True, True, True], dtype=bool)
        trust = SimpleNamespace(
            correction_limit=0.4,
            split_limit=1.0,
            reactor_limit=1.0,
            callbacks=callbacks,
            development_values=np.zeros((profile.development_candidate_count, 3)),
            out_of_fold_projected=np.zeros(
                (profile.development_candidate_count, response_count)
            ),
            out_of_fold_projection_accepted=accepted,
            split_scale=np.ones(1),
        )
        surrogate_assets = SimpleNamespace(
            leverage_precision=np.ones((1, 1)),
            trust_thresholds=SimpleNamespace(regularized_leverage=1.0),
            trust_callbacks=callbacks,
        )
        assessment = assessment_fixture()
        assessment = AssessmentResult(
            metrics=assessment.metrics,
            violations=assessment.violations,
            qp_diagnostics=assessment.qp_diagnostics,
            feasibility=assessment.feasibility,
            raw=np.zeros((profile.holdout_candidate_count, response_count)),
            projected=np.zeros((profile.holdout_candidate_count, response_count)),
            projected_targets=np.zeros(
                (profile.holdout_candidate_count, response_count)
            ),
            overflow_tss_closure=np.ones(profile.holdout_candidate_count),
        )
        overflow_closure = SimpleNamespace(
            predict=lambda controls, influents: np.ones(
                1 if np.asarray(controls).ndim == 1 else len(controls)
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with (
                patch.object(
                    module_surrogate_training,
                    "fit_or_resume_ridge",
                    return_value=(
                        model,
                        np.zeros((profile.development_candidate_count, response_count)),
                        "ridge-input",
                    ),
                ),
                patch.object(
                    module_surrogate_training,
                    "fit_or_resume_log_overflow_closure",
                    return_value=(
                        overflow_closure,
                        np.ones(profile.development_candidate_count),
                        "closure-input",
                    ),
                ),
                patch.object(
                    module_surrogate_training,
                    "overflow_tss_from_response",
                    return_value=np.ones(profile.development_candidate_count),
                ),
                patch.object(
                    module_optimization_mechanistic,
                    "fit_mechanistic_assets",
                    return_value=SimpleNamespace(),
                ),
                patch.object(
                    module_optimization_trust,
                    "calibrate_trust_diagnostics",
                    return_value=trust,
                ),
                patch.object(
                    module_optimization_surrogate,
                    "build_surrogate_assets",
                    return_value=surrogate_assets,
                ),
                patch.object(
                    module_validation_assessment,
                    "assess_raw_projected_mechanistic",
                    return_value=assessment,
                ),
                patch.object(module_runtime_contracts, "assert_source_unchanged"),
                patch.object(
                    module_runtime_contracts, "_artifact_hashes", return_value={}
                ) as artifact_hashes,
            ):
                result = module_workflow_assessment.run_assessment(
                    run,
                    design,
                    np.zeros(
                        (
                            profile.development_candidate_count,
                            mechanistic_response_count,
                        )
                    ),
                    np.zeros(
                        (profile.holdout_candidate_count, mechanistic_response_count)
                    ),
                    profile=profile,
                    source_files={"unit-test": "bound-source"},
                )
            trust_frame = pd.read_csv(run / "metrics/trust_development_oof.csv")
            np.testing.assert_array_equal(
                trust_frame["projection_qp_accepted"].to_numpy(dtype=bool), accepted
            )
            with np.load(run / "models/trust_calibration.npz") as stored:
                np.testing.assert_array_equal(
                    stored["out_of_fold_projection_accepted"], accepted
                )
            with np.load(
                run / "datasets/development/surrogate_responses.npz"
            ) as stored:
                self.assertEqual(
                    stored["responses"].shape,
                    (
                        profile.development_candidate_count,
                        profile.surrogate_response_count,
                    ),
                )
                self.assertEqual(
                    str(stored["schema"].item()),
                    module_runtime_protocols.RESPONSE_SCHEMA,
                )
            holdout_trust = pd.read_csv(
                run / "metrics/trust_post_selection_holdout.csv"
            )
            self.assertEqual(
                list(holdout_trust.columns),
                [
                    "row",
                    "correction",
                    "regularized_leverage",
                    "particulate_split",
                    "reactor_residual",
                ],
            )
            np.testing.assert_allclose(
                holdout_trust[
                    [
                        "correction",
                        "regularized_leverage",
                        "particulate_split",
                        "reactor_residual",
                    ]
                ].to_numpy(),
                np.tile(
                    [0.0, 1.0, np.sqrt(12.5), np.sqrt(50.0)],
                    (profile.holdout_candidate_count, 1),
                ),
            )
            published_paths = artifact_hashes.call_args.args[1]
            self.assertIn(
                run / "metrics/trust_post_selection_holdout.csv", published_paths
            )
            self.assertFalse(result.passed)
            self.assertFalse(
                result.gate["all_development_oof_projection_qp_audits_passed"]
            )
            self.assertTrue(result.gate["optimization_permitted"])

    def test_ridge_bundle_round_trip_and_tamper_detection(self) -> None:
        rows, responses = (10, 2)
        feature_map = QuadraticFeatureMap(
            decision_center=np.zeros(1),
            decision_scale=np.ones(1),
            influent_center=np.zeros(1),
            influent_scale=np.ones(1),
            term_center=np.zeros(5),
            term_scale=np.ones(5),
        )
        diagnostics = LeastSquaresDiagnostics(
            sample_count=rows,
            feature_count=6,
            response_count=responses,
            rank_tolerance=1e-12,
            smallest_singular_value=1.0,
            largest_singular_value=2.0,
            condition_number=2.0,
            optimality_residual=1e-12,
            coefficient_agreement=1e-12,
            acceptance_threshold=1e-10,
        )
        model = QuadraticSurrogate(
            feature_map=feature_map,
            response_center=np.zeros(responses),
            response_scale=np.ones(responses),
            coefficients=np.zeros((responses, 6)),
            diagnostics=diagnostics,
            ridge_penalty=float(module_config.RIDGE_GRID[-1]),
        )
        scores = pd.DataFrame(
            [
                {
                    "fold": fold,
                    "gamma": gamma,
                    "raw_nrmse": 1.0,
                    "selected": bool(gamma == module_config.RIDGE_GRID[-1]),
                }
                for fold in range(1, 6)
                for gamma in module_config.RIDGE_GRID
            ]
        )
        membership = np.tile(np.arange(1, 6), 2)
        result = SimpleNamespace(
            model=model,
            scores=scores,
            fold_membership=membership,
            out_of_fold_raw=np.zeros((rows, responses)),
            elapsed_seconds=1.0,
        )
        controls = np.zeros((rows, 1))
        influents = np.zeros((rows, 1))
        mechanistic_responses = np.zeros((rows, responses))
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            old_sources = {"unit-test": "old-source"}
            new_sources = {"unit-test": "new-source"}
            old_source_id = module_runtime_contracts.source_digest(old_sources)
            input_id = module_surrogate_training._ridge_input_digest(
                controls, influents, mechanistic_responses
            )
            module_surrogate_training.save_ridge(
                run, result, input_id=input_id, source_id=old_source_id
            )
            preserved_paths = (
                run / "models/ridge_complete.json",
                run / "models/ridge_surrogate.npz",
                run / "metrics/ridge_cross_validation.csv",
                run / "metrics/ridge_fold_membership.csv",
            )
            before = {
                path: module_runtime_contracts.file_digest(path)
                for path in preserved_paths
            }
            restored = module_surrogate_training._load_ridge(
                run,
                controls=controls,
                influents=influents,
                mechanistic_responses=mechanistic_responses,
                input_id=input_id,
                source_id=old_source_id,
            )
            self.assertIsNotNone(restored)
            np.testing.assert_array_equal(restored[1], result.out_of_fold_raw)
            with (
                patch.object(
                    module_runtime_contracts,
                    "_checkpoint_source_is_authorized",
                    return_value=True,
                ),
                patch.object(
                    module_surrogate_training, "cross_validate_ridge"
                ) as refit,
            ):
                carried_model, carried_oof, carried_input = (
                    module_surrogate_training.fit_or_resume_ridge(
                        run,
                        controls,
                        influents,
                        mechanistic_responses,
                        source_files=new_sources,
                    )
                )
            refit.assert_not_called()
            self.assertEqual(carried_input, input_id)
            self.assertEqual(carried_model.ridge_penalty, model.ridge_penalty)
            np.testing.assert_array_equal(carried_oof, result.out_of_fold_raw)
            self.assertEqual(
                before,
                {
                    path: module_runtime_contracts.file_digest(path)
                    for path in preserved_paths
                },
            )
            with (run / "metrics/ridge_cross_validation.csv").open("a") as stream:
                stream.write("tampered")
            self.assertIsNone(
                module_surrogate_training._load_ridge(
                    run,
                    controls=controls,
                    influents=influents,
                    mechanistic_responses=mechanistic_responses,
                    input_id=input_id,
                    source_id=old_source_id,
                )
            )

    def test_optimization_hook_rejects_wrong_physical_geometry(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "ten robustness cases and ten clarifier layers"
        ):
            module_workflow_optimization.run_optimization_stage(
                run=Path("unused"),
                profile=tiny_profile(),
                design={},
                development_responses=np.empty((0, 0)),
                holdout_responses=np.empty((0, 0)),
                analysis=None,
                source_files={},
            )
