from __future__ import annotations
import surrogate_optimization.reporting.tables as module_reporting_tables
import surrogate_optimization.plant.model as module_plant_model
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from surrogate_optimization.reporting.tables import FAILURE_CLASSES
from surrogate_optimization.reporting.tables import PENDING_CLASS
from surrogate_optimization.reporting.tables import build_reporting_tables

THETA = np.asarray([18.0, 0.2, 0.3, 0.4, 2.0, 0.75, 0.02])


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _mechanistic_payload(*, with_selection: bool = True) -> dict[str, object]:
    start = {
        "start_index": 0,
        "nearest_development_row": 0,
        "status": "first_order_kkt_stationary_feasible",
        "objective": 0.75,
        "feasible": True,
        "stationary": True,
        "controls": THETA.tolist(),
        "stages": [
            {
                "epsilon": 1e-06,
                "receiver_half_width": 10.0,
                "status": "Solve_Succeeded",
                "solver_success": True,
                "elapsed_seconds": 2.0,
                "iterations": 4,
                "feasible": True,
            }
        ],
        "kkt": {"active_inequality_count": 2},
    }
    return {
        "status": "selected_stationary"
        if with_selection
        else "no_validated_feasible_start",
        "selected_start": 0 if with_selection else None,
        "starts": [start],
        "elapsed_seconds": 3.0,
        "preflight_stage_wall_time_seconds": 600.0,
    }


def _make_run(root: Path, robustness_count: int = 2) -> None:
    rng = np.random.default_rng(4)
    development_candidate_count = 12
    controls = np.tile(THETA, (development_candidate_count, 1))
    controls += rng.normal(0.0, 0.01, size=controls.shape)
    controls[:, 0] = np.clip(controls[:, 0], 6.0, 36.0)
    controls[:, 1:4] = np.clip(controls[:, 1:4], 0.0, 1.0)
    controls[:, 4] = np.clip(controls[:, 4], 0.0, 4.0)
    controls[:, 5] = np.clip(controls[:, 5], 0.25, 1.25)
    controls[:, 6] = np.clip(controls[:, 6], 0.001, 0.05)
    influents = rng.uniform(1.0, 20.0, size=(development_candidate_count, 20))
    mechanistic_responses = rng.uniform(
        1.0, 100.0, size=(development_candidate_count, 165)
    )
    (root / "datasets" / "development").mkdir(parents=True)
    np.savez_compressed(
        root / "datasets" / "effective_design.npz",
        development_controls=controls,
        development_influents=influents,
        robustness_influents=rng.uniform(1.0, 20.0, size=(robustness_count, 20)),
    )
    np.savez_compressed(
        root / "datasets" / "development" / "accepted_mechanistic_responses.npz",
        mechanistic_responses=mechanistic_responses,
    )
    (root / "models").mkdir(parents=True)
    np.savez_compressed(
        root / "models" / "ridge_surrogate.npz", response_scale=np.ones(161)
    )
    (root / "metrics").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "case": "holdout_0000",
                "method": "raw",
                "mass_conservation_violation_max": 2.0,
                "mass_conservation_violation_count": 3,
                "nonnegativity_violation_max": 0.25,
                "nonnegativity_violation_count": 1,
                "minimum_coordinate": -0.25,
            },
            {
                "case": "holdout_0000",
                "method": "projected",
                "mass_conservation_violation_max": 1e-11,
                "mass_conservation_violation_count": 0,
                "nonnegativity_violation_max": 0.0,
                "nonnegativity_violation_count": 0,
                "minimum_coordinate": 0.0,
            },
        ]
    ).to_csv(root / "metrics" / "physical_violations_assessment.csv", index=False)
    _write_json(
        root / "metrics" / "trust_limits.json",
        {
            "correction": 0.5,
            "regularized_leverage": 2.0,
            "particulate_split": 0.1,
            "reactor_residual": 1.0,
        },
    )


class ReportingSnapshotTests(unittest.TestCase):
    def test_route_status_time_is_the_primary_optimization_duration(self) -> None:
        for route in ("surrogate", "mechanistic"):
            with self.subTest(route=route):
                snapshot = module_reporting_tables.RouteSnapshot(
                    case="nominal",
                    route=route,
                    artifact_state="complete",
                    outcome="selected",
                    payload={"elapsed_seconds": 12.0},
                    starts=(),
                    selected_start=None,
                    selected=None,
                    selected_arrays={},
                    reference_arrays={},
                    casewise_reference=None,
                    certification={"certificate": {"elapsed_seconds": 90.0}},
                    recovery={"attempted": True, "elapsed_seconds": 120.0},
                )
                self.assertEqual(module_reporting_tables._route_elapsed(snapshot), 12.0)

    def test_reporting_geometry_separates_surrogate_and_mechanistic_widths(
        self,
    ) -> None:
        geometry = module_reporting_tables.StudyGeometry(
            layer_count=5, layer_volume_m3=1200.0
        )
        self.assertEqual(geometry.surrogate_response_count, 161)
        self.assertEqual(geometry.response_count, 161)
        self.assertEqual(geometry.mechanistic_response_count, 165)
        self.assertEqual(geometry.mechanistic_state_count, 105)
        full = np.arange(165, dtype=float)
        full[-5:] = np.arange(1.0, 6.0)
        reduced = module_reporting_tables._as_reduced_response(
            full, geometry, allow_mechanistic=True
        )
        self.assertIsNotNone(reduced)
        assert reduced is not None
        self.assertEqual(reduced.shape, (161,))
        np.testing.assert_array_equal(reduced[:160], full[:160])
        self.assertEqual(reduced[-1], 1200.0 * sum(range(1, 6)))
        self.assertIsNone(
            module_reporting_tables._as_reduced_response(
                full, geometry, allow_mechanistic=False
            )
        )

    def test_shared_engineering_uses_scalar_clarifier_inventory(self) -> None:
        geometry = module_reporting_tables.StudyGeometry(
            layer_count=5, layer_volume_m3=1200.0
        )
        response = np.zeros(geometry.surrogate_response_count)
        response[geometry.inventory_index] = 5432.0
        quantities, _, _ = module_reporting_tables._response_quantities(
            THETA, response, geometry, np.ones(4)
        )
        self.assertEqual(quantities["clarifier_solids_inventory"], 5432.0)
        self.assertEqual(quantities["solids_inventory"], 5432.0)

    def test_trust_reporting_has_four_reduced_response_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            table = module_reporting_tables._trust_table(run, (), [])
            self.assertEqual(
                tuple(table["diagnostic"]), module_reporting_tables.TRUST_DIAGNOSTICS
            )
            self.assertNotIn("clarifier_flux", set(table["diagnostic"]))

    def test_trust_uses_only_post_selection_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            _write_json(
                run / "inputs" / "contract.json",
                {
                    "schema_version": 10,
                    "response_schema": {"name": "clarifier_inventory"},
                    "profile": {"layer_count": 5},
                },
            )
            legacy = pd.DataFrame(
                [
                    {
                        "correction": 99.0,
                        "regularized_leverage": 99.0,
                        "particulate_split": 99.0,
                        "reactor_residual": 99.0,
                        "clarifier_flux": 99.0,
                    }
                ]
            )
            legacy.to_csv(run / "metrics" / "trust_untouched_test.csv", index=False)
            warnings: list[str] = []
            table = module_reporting_tables._trust_table(run, (), warnings)
            self.assertTrue((table["holdout_candidate_count"] == 0).all())
            pd.DataFrame(
                [
                    {
                        "correction": 0.1,
                        "regularized_leverage": 1.0,
                        "particulate_split": 0.2,
                        "reactor_residual": 0.3,
                    },
                    {
                        "correction": 0.4,
                        "regularized_leverage": 1.5,
                        "particulate_split": 0.5,
                        "reactor_residual": 0.6,
                    },
                ]
            ).to_csv(run / "metrics" / "trust_post_selection_holdout.csv", index=False)
            table = module_reporting_tables._trust_table(run, (), [])
            self.assertTrue((table["holdout_candidate_count"] == 2).all())
            correction = table.set_index("diagnostic").loc["correction"]
            self.assertEqual(correction["holdout_p95"], 0.4)

    def test_physical_summary_reports_reduced_inequality_families(self) -> None:
        detail = pd.DataFrame(
            [
                {
                    "analysis_scope": "post_selection_holdout",
                    "case": "holdout_0000",
                    "method": "raw",
                    "mass_conservation_violation_max": 0.0,
                    "mass_conservation_violation_count": 0,
                    "nonnegativity_violation_max": 0.0,
                    "nonnegativity_violation_count": 0,
                    "particulate_densification_violation_max": 0.25,
                    "clarifier_inventory_bound_violation_max": 0.75,
                }
            ]
        )
        summary = module_reporting_tables._physical_summary(detail)
        row = summary[
            (summary["analysis_scope"] == "all_analysis") & (summary["method"] == "raw")
        ].iloc[0]
        self.assertEqual(row["particulate_densification_violation_max"], 0.25)
        self.assertEqual(row["clarifier_inventory_bound_violation_max"], 0.75)
        self.assertNotIn("mass_tss_endpoint_max", summary.columns)

    def test_scope_specific_nonlinear_audit_uses_only_saved_full_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            geometry = module_reporting_tables.StudyGeometry(5, 1200.0)
            states = np.ones((1, geometry.mechanistic_state_count))
            states[0, -5:] = np.arange(1.0, 6.0)
            alternate = states.copy()
            alternate[0, -5:] = np.arange(5.0, 0.0, -1.0)
            np.savez_compressed(
                run / "datasets" / "development" / "accepted_mechanistic_responses.npz",
                mechanistic_responses=np.ones((1, geometry.mechanistic_response_count)),
                states_start_1=states,
                states_start_2=alternate,
            )
            np.savez_compressed(
                run / "datasets" / "development" / "accepted_inputs.npz",
                controls=THETA[None, :],
                influents=np.ones((1, 20)),
            )
            pd.DataFrame(
                [
                    {
                        "accepted_slot": 0,
                        "largest_real_eigenvalue_start_1": -0.2,
                        "largest_real_eigenvalue_start_2": -0.1,
                        "stability_agreement_start_1": 1e-08,
                        "stability_agreement_start_2": 2e-08,
                        "locally_stable_start_1": True,
                        "locally_stable_start_2": True,
                    }
                ]
            ).to_csv(
                run / "datasets" / "development" / "accepted_diagnostics.csv",
                index=False,
            )
            exact_diagnostics = {
                "diagnostics_start_1": {
                    "largest_real_eigenvalue": -0.3,
                    "stability_eigenvalue_agreement": 1e-08,
                    "locally_stable": True,
                },
                "diagnostics_start_2": {
                    "largest_real_eigenvalue": 0.0,
                    "stability_eigenvalue_agreement": 2e-06,
                    "locally_stable": False,
                },
            }
            snapshot = module_reporting_tables.RouteSnapshot(
                case="nominal",
                route="surrogate",
                artifact_state="complete",
                outcome="selected",
                payload=None,
                starts=(),
                selected_start=0,
                selected={"final": {"controls": THETA.tolist()}},
                selected_arrays={
                    "controls": THETA,
                    "exact_state_start_1": states[0],
                    "exact_state_start_2": alternate[0],
                },
                reference_arrays={},
                casewise_reference={
                    "candidate_available": True,
                    "reference": exact_diagnostics,
                },
                certification=None,
                recovery=None,
            )

            def balance_audit(state, *_args, **_kwargs):
                descending = bool(state[-1] < state[-2])
                return {
                    "balance_family_maxima": {
                        "clarifier_layer": 2e-08 if descending else 5e-09
                    },
                    "balance_family_violation_counts": {
                        "clarifier_layer": 1 if descending else 0
                    },
                }

            physical = pd.DataFrame({"method": ["raw", "projected"]})
            with patch.object(
                module_plant_model,
                "mechanistic_balance_audit",
                side_effect=balance_audit,
            ) as audit:
                table = module_reporting_tables._scope_specific_nonlinear_audit(
                    run, (snapshot,), ("nominal",), geometry, physical, []
                ).set_index("source")
            self.assertEqual(audit.call_count, 4)
            for source in ("raw_reduced", "projected_reduced"):
                self.assertEqual(
                    table.loc[source, "applicability"], "not_applicable_no_layer_state"
                )
                self.assertTrue(np.isnan(table.loc[source, "layer_residual_max"]))
            generation = table.loc["exact_mechanistic_generation"]
            self.assertEqual(generation["record_count"], 2)
            self.assertEqual(generation["audited_record_count"], 2)
            self.assertEqual(generation["layer_residual_violation_count"], 1)
            replay = table.loc["exact_mechanistic_replay"]
            self.assertEqual(replay["record_count"], 2)
            self.assertEqual(replay["layer_envelope_violation_count"], 6)
            self.assertEqual(replay["stability_violation_count"], 1)

    def test_physical_summary_excludes_unavailable_placeholders(self) -> None:
        detail = pd.DataFrame(
            [
                {
                    "analysis_scope": "selected_decision_common_reference",
                    "case": "nominal:surrogate",
                    "method": "optimizer_native",
                    "audit_available": True,
                    "mass_conservation_violation_max": 2e-08,
                    "mass_conservation_violation_count": 2,
                    "nonnegativity_violation_max": 3e-10,
                    "nonnegativity_violation_count": 1,
                    "minimum_coordinate": -3e-10,
                    "network_inequality_violation_count": 4,
                },
                {
                    "analysis_scope": "selected_decision_common_reference",
                    "case": "robustness_01:surrogate",
                    "method": "optimizer_native",
                    "audit_available": False,
                    "mass_conservation_violation_max": np.nan,
                    "mass_conservation_violation_count": 0,
                    "nonnegativity_violation_max": np.nan,
                    "nonnegativity_violation_count": 0,
                    "minimum_coordinate": np.nan,
                    "network_inequality_violation_count": 0,
                },
                {
                    "analysis_scope": "post_selection_holdout",
                    "case": "holdout_0000",
                    "method": "optimizer_native",
                    "audit_available": np.nan,
                    "mass_conservation_violation_max": 1e-09,
                    "mass_conservation_violation_count": 0,
                    "nonnegativity_violation_max": 0.0,
                    "nonnegativity_violation_count": 0,
                    "minimum_coordinate": 0.0,
                    "network_inequality_violation_count": 0,
                },
            ]
        )
        summary = module_reporting_tables._physical_summary(detail)
        selected = summary[
            (summary["analysis_scope"] == "selected_decision_common_reference")
            & (summary["method"] == "optimizer_native")
        ].iloc[0]
        self.assertEqual(selected["availability"], "partially_available")
        self.assertEqual(selected["record_count"], 2)
        self.assertEqual(selected["audited_record_count"], 1)
        self.assertEqual(selected["unavailable_record_count"], 1)
        self.assertEqual(selected["audit_coverage_fraction"], 0.5)
        self.assertEqual(selected["mass_conservation_violation_count"], 2)
        self.assertEqual(selected["nonnegativity_violation_count"], 1)
        only_placeholder = detail.iloc[[1]].copy()
        unavailable = module_reporting_tables._physical_summary(only_placeholder)
        unavailable = unavailable[
            (unavailable["analysis_scope"] == "selected_decision_common_reference")
            & (unavailable["method"] == "optimizer_native")
        ].iloc[0]
        self.assertEqual(unavailable["availability"], "not_available")
        self.assertEqual(unavailable["audited_record_count"], 0)
        self.assertEqual(unavailable["unavailable_record_count"], 1)
        self.assertTrue(np.isnan(unavailable["mass_conservation_violation_count"]))
        self.assertTrue(np.isnan(unavailable["nonnegativity_violation_count"]))

    def test_physical_summary_treats_legacy_rows_as_audited(self) -> None:
        legacy = pd.DataFrame(
            [
                {
                    "analysis_scope": "untouched_test",
                    "case": "holdout_0000",
                    "method": "mechanistic",
                    "mass_conservation_violation_max": 1e-09,
                    "mass_conservation_violation_count": 0,
                    "nonnegativity_violation_max": 0.0,
                    "nonnegativity_violation_count": 0,
                }
            ]
        )
        summary = module_reporting_tables._physical_summary(legacy)
        row = summary[
            (summary["analysis_scope"] == "post_selection_holdout")
            & (summary["method"] == "mechanistic")
        ].iloc[0]
        self.assertEqual(row["availability"], "available")
        self.assertEqual(row["audited_record_count"], 1)
        self.assertEqual(row["unavailable_record_count"], 0)
        self.assertEqual(row["mass_conservation_violation_count"], 0)

    def test_route_status_uses_declared_single_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            path = run / "optimization" / "nominal" / "mechanistic.json"
            current = _mechanistic_payload()
            current["optimization_attempt_count"] = 1
            _write_json(path, current)
            row = build_reporting_tables(run)["route_status"]
            row = row[
                (row["case"] == "nominal") & (row["route"] == "mechanistic")
            ].iloc[0]
            self.assertEqual(row["starts_expected"], 1)
            legacy = _mechanistic_payload()
            _write_json(path, legacy)
            row = build_reporting_tables(run)["route_status"]
            row = row[
                (row["case"] == "nominal") & (row["route"] == "mechanistic")
            ].iloc[0]
            self.assertEqual(row["starts_expected"], 1)

    def test_fixed_generation_tables_and_effective_artifacts_are_preferred(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            with np.load(
                run / "datasets" / "effective_design.npz", allow_pickle=False
            ) as stored:
                effective = {
                    name: np.asarray(stored[name]).copy() for name in stored.files
                }
            effective["development_controls"] = (
                effective["development_controls"] + 0.125
            )
            np.savez_compressed(run / "datasets" / "effective_design.npz", **effective)
            accepted_targets = np.full((12, 165), 7.0)
            np.savez_compressed(
                run / "datasets" / "development" / "accepted_mechanistic_responses.npz",
                mechanistic_responses=accepted_targets,
            )
            for block in ("development", "holdout"):
                directory = run / "datasets" / block
                directory.mkdir(parents=True, exist_ok=True)
                controls = np.asarray(
                    [
                        [6.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.001],
                        [36.0, 1.0, 1.0, 1.0, 4.0, 1.25, 0.05],
                    ]
                )
                influents = np.vstack((np.ones(20), np.full(20, 2.0)))
                np.savez_compressed(
                    directory / "accepted_inputs.npz",
                    controls=controls,
                    influents=influents,
                )
                attempts = pd.DataFrame(
                    {
                        "candidate_id": [
                            f"{block}:r000000:c000000",
                            f"{block}:r000000:c000001",
                            f"{block}:r000000:c000002",
                        ],
                        "accepted": [True, False, True],
                        "rejection_reason": [
                            "accepted",
                            "branch_disagreement",
                            "accepted",
                        ],
                        "rejected_solver_exception": [False, False, False],
                        "rejected_mass_or_residual": [False, False, False],
                        "rejected_stability": [False, False, False],
                        "rejected_nonnegativity": [False, False, False],
                        "rejected_domain": [False, False, False],
                        "rejected_root_distance": [False, False, False],
                        "rejected_branch_disagreement": [False, True, False],
                        "rejected_other_solver_rejection": [False, False, False],
                        "elapsed_seconds": [1.0, 2.0, 3.0],
                    }
                )
                attempts.to_csv(directory / "all_attempts.csv", index=False)
                pd.DataFrame(
                    {
                        "accepted_slot": [0, 1],
                        "source_candidate_id": [
                            f"{block}:r000000:c000000",
                            f"{block}:r000000:c000002",
                        ],
                        "source_candidate_index": [0, 2],
                    }
                ).to_csv(directory / "accepted_provenance.csv", index=False)
                pd.DataFrame(
                    {
                        "candidate_id": [
                            f"{block}:r000000:c000000",
                            f"{block}:r000000:c000001",
                        ],
                        "preserved_without_rewrite": [True, True],
                    }
                ).to_csv(directory / "candidate_checkpoint_summary.csv", index=False)
                _write_json(
                    directory / "generation_summary.json",
                    {"candidate_count": 3, "accepted_count": 2, "rejected_count": 1},
                )
            warnings: list[str] = []
            loaded_design = module_reporting_tables._effective_design(run, warnings)
            loaded_targets = module_reporting_tables._accepted_development(
                run, warnings
            )
            np.testing.assert_array_equal(
                loaded_design["development_controls"], effective["development_controls"]
            )
            np.testing.assert_array_equal(
                loaded_targets["mechanistic_responses"], accepted_targets
            )
            bundle = build_reporting_tables(run)
            summary = bundle["generation_summary"].set_index("block")
            self.assertEqual(
                summary.loc["development", "candidate_attempt_denominator"], 3
            )
            self.assertEqual(summary.loc["development", "accepted_row_denominator"], 2)
            self.assertEqual(summary.loc["development", "rejected_candidate_count"], 1)
            self.assertTrue(summary.loc["development", "accepted_slots_fully_traced"])
            self.assertTrue(
                summary.loc[
                    "development", "candidate_design_is_single_strength_one_lhs"
                ]
            )
            self.assertFalse(summary.loc["development", "rejected_candidates_replaced"])
            reasons = bundle["generation_rejection_reasons"]
            branch = reasons[
                (reasons["block"] == "development")
                & (reasons["rejection_reason"] == "branch_disagreement")
            ].iloc[0]
            self.assertEqual(branch["count"], 1)
            self.assertEqual(len(bundle["generation_accepted_coverage"]), 54)
            self.assertEqual(len(bundle["generation_attempt_ledger"]), 6)
            self.assertEqual(len(bundle["generation_accepted_provenance"]), 4)
            self.assertEqual(len(bundle["generation_checkpoint_summary"]), 4)

    def test_timing_tables_use_declared_artifact_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            np.savez_compressed(
                run / "models" / "ridge_surrogate.npz", response_scale=np.ones(161)
            )
            pd.DataFrame(
                [
                    {
                        "case": "robustness_01",
                        "route": "surrogate",
                        "candidate_available": True,
                        "metric": "Optimization time",
                        "unit": "s",
                        "time_seconds": 10.0,
                    },
                    {
                        "case": "robustness_02",
                        "route": "surrogate",
                        "candidate_available": True,
                        "metric": "Optimization time",
                        "unit": "s",
                        "time_seconds": 20.0,
                    },
                    {
                        "case": "robustness_01",
                        "route": "mechanistic",
                        "candidate_available": True,
                        "metric": "Optimization time",
                        "unit": "s",
                        "time_seconds": 12.0,
                    },
                    {
                        "case": "robustness_02",
                        "route": "mechanistic",
                        "candidate_available": False,
                        "metric": "Optimization time",
                        "unit": "s",
                        "time_seconds": 18.0,
                    },
                ]
            ).to_csv(run / "metrics" / "robustness_case_timing.csv", index=False)
            bundle = build_reporting_tables(run)
            timing = bundle["timing_summary"].set_index("route")
            self.assertTrue(timing["metric"].eq("Optimization time").all())
            self.assertTrue(timing["unit"].eq("s").all())
            self.assertAlmostEqual(timing.loc["surrogate", "mean"], 15.0)
            self.assertAlmostEqual(timing.loc["mechanistic", "mean"], 15.0)
            workload = bundle["timing_workload"].set_index("route")
            self.assertEqual(
                workload.loc["mechanistic", "candidate_available_count"], 1
            )

    def test_incomplete_cases_and_zero_count_failure_classes_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run)
            _write_json(
                run / "optimization" / "nominal" / "mechanistic.json",
                _mechanistic_payload(),
            )
            from surrogate_optimization.runtime.contracts import file_digest

            payload_path = (
                run / "optimization/nominal/mechanistic_casewise_reference.json"
            )
            _write_json(
                payload_path,
                {
                    "reference_contract": "fixture",
                    "candidate_available": True,
                    "comparison_valid": True,
                    "status": "valid_interior",
                },
            )
            _write_json(
                payload_path.with_name("mechanistic_casewise_reference_complete.json"),
                {
                    "reference_contract": "fixture",
                    "artifacts": {payload_path.name: file_digest(payload_path)},
                },
            )
            bundle = build_reporting_tables(run)
            self.assertEqual(
                bundle.expected_cases, ("nominal", "robustness_01", "robustness_02")
            )
            self.assertEqual(len(bundle["route_status"]), 6)
            self.assertEqual(len(bundle["case_status"]), 3)
            nominal = bundle["case_status"].set_index("case").loc["nominal"]
            self.assertEqual(nominal["selected_n"], 1)
            self.assertEqual(nominal["mechanistic_disposition"], "validated result")
            pending = bundle["case_status"].set_index("case").loc["robustness_01"]
            self.assertEqual(pending["surrogate_disposition"], PENDING_CLASS)
            controls = bundle["selected_controls"]
            direct = controls[
                (controls["case"] == "nominal") & (controls["route"] == "mechanistic")
            ].iloc[0]
            self.assertAlmostEqual(direct["H"], 18.0)
            self.assertEqual(len(bundle["nominal_controls"]), 2)
            self.assertEqual(len(bundle["scenario_controls"]), 4)
            failure = bundle["failure_accounting"]
            self.assertEqual(len(failure), 2 * (len(FAILURE_CLASSES) + 1))
            projection = failure[
                (failure["route"] == "mechanistic")
                & (failure["classification"] == "projection failure")
            ].iloc[0]
            self.assertEqual(projection["count"], 0)
            self.assertEqual(projection["denominator"], 3)
            physical = bundle["physical_violation_summary"].set_index(
                ["analysis_scope", "method"]
            )
            self.assertEqual(
                physical.loc[("post_selection_holdout", "raw"), "record_count"], 1
            )
            self.assertEqual(
                physical.loc[
                    ("post_selection_holdout", "raw"),
                    "mass_conservation_violation_count",
                ],
                3,
            )
            self.assertEqual(
                physical.loc[
                    ("selected_decision_common_reference", "exact_mechanistic_start_1"),
                    "availability",
                ],
                "not_available",
            )

    def test_selected_response_audit_is_reconstructed_and_tables_write_atomically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            case = run / "optimization" / "nominal"
            _write_json(case / "mechanistic.json", _mechanistic_payload())
            case.mkdir(parents=True, exist_ok=True)
            response = np.linspace(1.0, 10.0, 165)
            np.savez_compressed(
                case / "mechanistic_selected.npz",
                controls=THETA,
                response=response,
                state=np.ones(105),
            )
            pd.DataFrame(
                [
                    {
                        "case": "nominal:direct",
                        "method": "optimizer_native",
                        "decision_route": "mechanistic",
                        "response_source": "optimizer_native",
                        "audit_available": True,
                        "mass_conservation_violation_max": 1e-09,
                        "mass_conservation_violation_count": 0,
                        "nonnegativity_violation_max": 0.0,
                        "nonnegativity_violation_count": 0,
                    }
                ]
            ).to_csv(
                run / "metrics" / "selected_response_physical_audit.csv", index=False
            )
            bundle = build_reporting_tables(run)
            detail = bundle["physical_violation_detail"]
            selected = detail[
                (detail["analysis_scope"] == "selected_decision_common_reference")
                & (detail["method"] == "optimizer_native")
            ]
            self.assertEqual(len(selected), 1)
            self.assertTrue(
                np.isfinite(selected.iloc[0]["mass_conservation_violation_max"])
            )
            summary = bundle["physical_violation_summary"].set_index(
                ["analysis_scope", "method"]
            )
            self.assertEqual(
                summary.loc[
                    ("selected_decision_common_reference", "optimizer_native"),
                    "availability",
                ],
                "available",
            )
            self.assertEqual(len(bundle["process_profiles"]), 38)
            profiles = bundle["process_profiles"]
            inventory = profiles[profiles["location"].eq("clarifier_inventory")]
            layers = profiles[profiles["location"].str.startswith("clarifier_layer_")]
            self.assertEqual(len(inventory), 1)
            self.assertEqual(set(inventory["quantity"]), {"TSS_mass"})
            self.assertEqual(len(layers), 5)
            self.assertEqual(set(layers["response_method"]), {"smooth"})
            output = Path(temporary) / "report"
            written = bundle.write(output)
            self.assertTrue(written["physical_violation_summary"].is_file())
            self.assertTrue(written["scope_specific_nonlinear_audit"].is_file())
            self.assertTrue(written["manifest"].is_file())
            manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["expected_cases"], ["nominal"])
            self.assertEqual(
                manifest["table_rows"]["physical_violation_summary"],
                len(bundle["physical_violation_summary"]),
            )
            self.assertEqual(
                manifest["table_rows"]["scope_specific_nonlinear_audit"],
                len(bundle["scope_specific_nonlinear_audit"]),
            )
