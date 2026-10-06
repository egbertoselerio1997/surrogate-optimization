import surrogate_optimization.data.generation as module_data_generation
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from surrogate_optimization.config import StudyProfile
from surrogate_optimization.data.design import create_design
from surrogate_optimization.plant.definitions import N_COMPONENTS
from surrogate_optimization.plant.definitions import N_STAGES


def _profile() -> StudyProfile:
    return StudyProfile(
        name="replacement_unit",
        development_candidate_count=3,
        holdout_candidate_count=2,
        robustness_count=1,
        layer_count=3,
        development_seed=101,
        holdout_seed=202,
        robustness_seed=303,
        parallel_workers=1,
        scientifically_eligible=False,
        enforce_admission_gate=False,
    )


def _record(
    candidate: module_data_generation._Candidate, accepted: bool
) -> dict[str, object]:
    record: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "candidate_index": candidate.candidate_index,
        "accepted": accepted,
        "attempt_status": "accepted" if accepted else "rejected",
        "error_type": "",
        "error_message": "",
        "elapsed_seconds": 1.0,
        "root_difference_inf": 1e-08,
        "branch_agreement": accepted,
        "branch_classification": "{}",
        "route_start_1": "holdout",
        "route_start_2": "holdout",
    }
    for name in (
        "minimum_state_start_1",
        "minimum_state_start_2",
        "state_negativity_start_1",
        "state_negativity_start_2",
        "rate_negativity_start_1",
        "rate_negativity_start_2",
        "mass_residual_start_1",
        "mass_residual_start_2",
        "largest_real_eigenvalue_start_1",
        "largest_real_eigenvalue_start_2",
        "stability_agreement_start_1",
        "stability_agreement_start_2",
        "feed_tss_start_1",
        "feed_tss_start_2",
        "external_solids_loss_start_1",
        "external_solids_loss_start_2",
    ):
        record[name] = 0.0
    return record


def _checkpoint(
    candidate: module_data_generation._Candidate,
    profile: StudyProfile,
    contract_hash: str,
    *,
    accepted: bool,
    marker: float,
) -> None:
    state_size = N_STAGES * N_COMPONENTS + profile.layer_count
    module_data_generation._write_attempt(
        candidate,
        contract_hash=contract_hash,
        target=np.full(profile.response_count, marker),
        first=np.full(state_size, marker),
        second=np.full(state_size, marker),
        record=_record(candidate, accepted),
    )


class CandidateGenerationTests(unittest.TestCase):
    def test_rejection_reason_is_deterministic_and_retains_overlaps(self) -> None:
        record = module_data_generation._annotate_rejection(
            {
                "accepted": False,
                "attempt_status": "rejected",
                "mass_residual_start_1": 2e-08,
                "stability_agreement_start_1": 2e-06,
                "state_negativity_start_1": 2e-10,
                "feed_tss_start_1": 0.5,
                "root_difference_inf": 2e-06,
                "branch_agreement": False,
            }
        )
        self.assertEqual(record["rejection_reason"], "mass_or_residual")
        self.assertEqual(
            record["rejection_reasons"],
            "mass_or_residual;stability;nonnegativity;domain;root_distance;branch_disagreement",
        )
        self.assertTrue(record["rejected_branch_disagreement"])

    def test_reuses_fixed_candidates_and_excludes_rejected_row(self) -> None:
        profile = _profile()
        design = create_design(profile)
        controls = design["development_controls"]
        influents = design["development_influents"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "development"
            output.mkdir(parents=True)
            np.savez_compressed(output / "mechanistic_responses.npz", legacy=[1.0])
            (output / "mechanistic_diagnostics.csv").write_text(
                "legacy,value\ntrue,1\n", encoding="utf-8"
            )
            legacy_hashes = {
                path: module_data_generation._file_digest(path)
                for path in (
                    output / "mechanistic_responses.npz",
                    output / "mechanistic_diagnostics.csv",
                )
            }
            base_contract = module_data_generation._base_contract_hash(
                controls, influents, profile
            )
            base_candidates = [
                module_data_generation._Candidate(
                    "development",
                    index,
                    controls[index],
                    influents[index],
                    output / "rows" / f"row_{index:06d}.npz",
                )
                for index in range(3)
            ]
            for index, candidate in enumerate(base_candidates):
                _checkpoint(
                    candidate,
                    profile,
                    base_contract,
                    accepted=index != 1,
                    marker=float(index + 1),
                )
            original_hashes = {
                item.checkpoint: module_data_generation._file_digest(item.checkpoint)
                for item in base_candidates
            }
            result = (
                module_data_generation.generate_mechanistic_block_from_fixed_design(
                    controls, influents, profile, output, block="development"
                )
            )
            self.assertEqual(
                result.mechanistic_responses.shape, (2, profile.response_count)
            )
            np.testing.assert_array_equal(
                result.mechanistic_responses[:, 0], [1.0, 3.0]
            )
            np.testing.assert_array_equal(result.controls[0], controls[0])
            np.testing.assert_array_equal(result.controls[1], controls[2])
            self.assertEqual(
                result.provenance["source_candidate_index"].tolist(), [0, 2]
            )
            self.assertEqual(len(result.attempts), 3)
            self.assertFalse(result.attempts.iloc[1]["accepted"])
            for path, digest in original_hashes.items():
                self.assertEqual(module_data_generation._file_digest(path), digest)
            for path, digest in legacy_hashes.items():
                self.assertEqual(module_data_generation._file_digest(path), digest)
            self.assertTrue((output / "accepted_mechanistic_responses.npz").is_file())
            self.assertTrue((output / "accepted_diagnostics.csv").is_file())
            self.assertTrue((output / "all_attempts.csv").is_file())
            self.assertTrue((output / "accepted_inputs.npz").is_file())
            self.assertTrue((output / "accepted_provenance.csv").is_file())
            self.assertTrue((output / "candidate_checkpoint_summary.csv").is_file())
            self.assertTrue((output / "generation_summary.json").is_file())
            self.assertTrue((output / "mechanistic_responses.npz").exists())
            self.assertTrue((output / "mechanistic_diagnostics.csv").exists())
            replayed = (
                module_data_generation.generate_mechanistic_block_from_fixed_design(
                    controls, influents, profile, output, block="development"
                )
            )
            np.testing.assert_array_equal(
                replayed.mechanistic_responses, result.mechanistic_responses
            )
            for path, digest in original_hashes.items():
                self.assertEqual(module_data_generation._file_digest(path), digest)
            for path, digest in legacy_hashes.items():
                self.assertEqual(module_data_generation._file_digest(path), digest)

    def test_rejected_fixed_candidate_does_not_trigger_another_round(self) -> None:
        profile = _profile()
        design = create_design(profile)
        controls = design["holdout_controls"]
        influents = design["holdout_influents"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "holdout"
            base_contract = module_data_generation._base_contract_hash(
                controls, influents, profile
            )
            candidates = [
                module_data_generation._Candidate(
                    "holdout",
                    index,
                    controls[index],
                    influents[index],
                    output / "rows" / f"row_{index:06d}.npz",
                )
                for index in range(2)
            ]
            _checkpoint(
                candidates[0], profile, base_contract, accepted=True, marker=1.0
            )
            _checkpoint(
                candidates[1], profile, base_contract, accepted=False, marker=2.0
            )
            result = (
                module_data_generation.generate_mechanistic_block_from_fixed_design(
                    controls, influents, profile, output, block="holdout"
                )
            )
            self.assertEqual(len(result.attempts), 2)
            self.assertEqual(result.provenance["source_candidate_index"].tolist(), [0])
            np.testing.assert_array_equal(result.mechanistic_responses[:, 0], [1.0])
            summary = json.loads(
                (output / "generation_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["candidate_count"], 2)
            self.assertEqual(summary["accepted_count"], 1)
            self.assertEqual(summary["rejected_count"], 1)
            self.assertFalse((output / "attempts" / "replacement").exists())
