"""Real 80/20 candidate execution with unchanged physical and solver settings."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import unittest
from unittest.mock import patch

from surrogate_optimization.runtime.contracts import _artifacts_match
from surrogate_optimization.runtime import checkpoints
from surrogate_optimization.workflow.study import _execute_pipeline
from tests.support.profiles import INTEGRATION_PROFILE


class ReducedWorkloadTests(unittest.TestCase):
    def test_real_pipeline_reports_all_cases_and_resumes(self):
        run_id = os.environ.get("SURROGATE_OPTIMIZATION_VALIDATION_RUN_ID") or (
            "validation_refactor_"
            + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
        summary = _execute_pipeline(run_id, "complete", profile=INTEGRATION_PROFILE)
        self.assertIn(summary.status, ("complete", "complete_with_validation_failures"))
        run = summary.result_directory
        self.assertEqual(run.parent.name, "results")
        contract = json.loads((run / "run_contract.json").read_text())
        self.assertEqual(contract["profile"]["development_candidate_count"], 80)
        self.assertEqual(contract["profile"]["holdout_candidate_count"], 20)
        self.assertEqual(contract["profile"]["layer_count"], 10)
        final = json.loads((run / "optimization/final_status.json").read_text())
        self.assertEqual(final["case_count"], 11)
        self.assertEqual(final["route_count"], 22)
        report = json.loads((run / "report/manifest.json").read_text())
        self.assertEqual(len(report["figures"]), 28)
        self.assertTrue(_artifacts_match(run, report["artifacts"]))
        all_paths = [p.resolve() for p in run.rglob("*") if p.is_file()]
        self.assertTrue(all(run.resolve() in p.parents for p in all_paths))
        with patch.object(
            checkpoints,
            "_run_surrogate_route",
            side_effect=AssertionError("completed search repeated"),
        ):
            resumed = _execute_pipeline(run_id, "complete", profile=INTEGRATION_PROFILE)
        self.assertEqual(resumed, summary)
        print(f"Validated reduced workload: {run}", flush=True)
