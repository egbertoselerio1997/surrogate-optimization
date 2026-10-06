"""Tables and chart metadata use one presentation vocabulary."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from surrogate_optimization.reporting.figures import comparison
from surrogate_optimization.reporting.labels import (
    CONTROL_LABELS,
    LOCATION_LABELS,
    ROUTE_LABELS,
    presentation_table,
    scenario_label,
    chart_metric_label,
)
from surrogate_optimization.reporting.tables import ReportingBundle


class ReportingLabelsTests(unittest.TestCase):
    def test_chart_and_table_names_match_without_changing_values(self):
        frame = pd.DataFrame(
            {
                "case": ["nominal", "robustness_10"],
                "route": ["surrogate", "mechanistic"],
                "decision_route": ["surrogate", "mechanistic"],
                "response_method": ["raw", "projected"],
                "location": ["reactor_3", "clarifier_overflow"],
                "value": [0.125, 15.0],
            }
        )
        result = presentation_table(frame)
        pd.testing.assert_frame_equal(result[frame.columns], frame)
        self.assertEqual(result.scenario_label.tolist(), ["N", "S10"])
        self.assertEqual(result.route_label.tolist(), ["Extended ICSOR", "Smooth NLP"])
        self.assertEqual(result.response_label.tolist(), ["Raw", "Projected"])
        self.assertEqual(result.location_label.tolist(), ["R3", "Overflow"])
        self.assertIs(comparison.ROUTE_LABEL, ROUTE_LABELS)
        self.assertEqual(
            comparison.LOCATION_LABELS,
            tuple(LOCATION_LABELS[name] for name in comparison.LOCATIONS),
        )
        pd.testing.assert_frame_equal(presentation_table(result), result)

    def test_bundle_publishes_shared_vocabulary_and_metric_definitions(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            frame = pd.DataFrame(
                {
                    "case": ["nominal"],
                    "route": ["surrogate"],
                    "metric": ["Optimization time"],
                    "time_seconds": [2.5],
                }
            )
            bundle = ReportingBundle(run, ("nominal",), {"timing": frame}, ())
            written = bundle.write()
            exported = pd.read_csv(written["timing"])
            self.assertEqual(
                exported.route_label.iloc[0], comparison.ROUTE_LABEL["surrogate"]
            )
            self.assertEqual(exported.metric.iloc[0], "Optimization time")
            manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
            vocabulary = manifest["presentation"]
            self.assertEqual(vocabulary["column_labels"]["H"], CONTROL_LABELS["H"])
            self.assertEqual(vocabulary["scenarios"]["robustness_01"], "S1")
            self.assertEqual(vocabulary["optimization_time_unit"], "s")
            guide = written["guide"].read_text(encoding="utf-8")
            self.assertIn("Mean location R²", guide)
            self.assertIn("development-scale response errors", guide)
            self.assertNotIn("primary", guide.lower())

    def test_chart_metric_names(self):
        self.assertEqual(
            chart_metric_label("holdout_projected_cod_mean_location_r2"),
            "holdout Projected COD Mean location R²",
        )
        self.assertEqual(
            chart_metric_label("smooth_nlp_mean_seconds"),
            "Smooth NLP mean optimization time (s)",
        )

    def test_all_scenario_labels_and_missing_values(self):
        self.assertEqual(
            [scenario_label(f"robustness_{i:02d}") for i in range(1, 11)],
            [f"S{i}" for i in range(1, 11)],
        )
        frame = pd.DataFrame({"case": [None], "response_method": ["unavailable"]})
        result = presentation_table(frame)
        self.assertIsNone(result.scenario_label.iloc[0])
        self.assertEqual(result.response_label.iloc[0], "unavailable")
