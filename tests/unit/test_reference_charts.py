"""Chart layouts, complete scenario coverage, and atomic eight-PNG publication."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from surrogate_optimization.reporting.figures import comparison, package
from surrogate_optimization.reporting.figures.data import (
    ChartDataError,
    load_chart_data,
    boolean_values,
)
from tests.support.chart_fixtures import complete_chart_run


class ReferenceChartTests(unittest.TestCase):
    def test_exact_layouts_have_no_titles_and_include_all_scenarios(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            complete_chart_run(run)
            data = load_chart_data(run)
            captured = {}

            def capture(figure, path, **_kwargs):
                captured[path.stem] = figure
                self.assertTrue(all(not axis.get_title() for axis in figure.axes))
                self.assertIsNone(figure._suptitle)

            with patch.object(comparison, "export_figure", side_effect=capture):
                comparison.render_figures(
                    data,
                    run,
                )
            self.assertEqual(set(captured), set(comparison.FIGURES))
            self.assertEqual(len(captured["q01_holdout_accuracy_overview"].axes), 9)
            heatmaps = [
                axis
                for axis in captured["q02_holdout_accuracy_by_location"].axes
                if axis.images
            ]
            self.assertEqual(len(heatmaps), 3)
            self.assertEqual(
                heatmaps[0].images[0].get_clim(), heatmaps[1].images[0].get_clim()
            )
            parity = captured["q03_holdout_parity_all_locations"]
            axes = parity.axes[:4]
            self.assertTrue(
                all(a.get_xscale() == a.get_yscale() == "log" for a in axes)
            )
            self.assertEqual(len({id(a.collections[0].norm) for a in axes}), 1)
            self.assertTrue(
                any(
                    "Median — Mixer" == text.get_text()
                    for legend in parity.legends
                    for text in legend.texts
                )
            )
            effluent = captured["q04_effluent_and_removal_parity"].axes
            self.assertEqual(len(effluent), 8)
            self.assertTrue(
                all(len(a.collections[0].get_offsets()) == 11 for a in effluent)
            )
            self.assertNotEqual(
                tuple(effluent[0].collections[0].get_facecolor()[0]),
                tuple(effluent[4].collections[0].get_facecolor()[0]),
            )
            controls = captured["q05_effluent_and_operating_values"]
            self.assertEqual(len(controls.axes), 12)
            self.assertFalse(controls.axes[-1].axison)
            self.assertEqual(controls.axes[5].get_ylim(), (0.0, 1.0))
            self.assertEqual(controls.axes[9].get_ylim(), (0.0, 1.0))
            profiles = captured["q06_treatment_train_profiles"].axes
            for row in range(4):
                self.assertEqual(
                    profiles[2 * row].get_ylim(), profiles[2 * row + 1].get_ylim()
                )
                self.assertEqual(len(profiles[2 * row].lines), 11)
                self.assertEqual(len(profiles[2 * row].lines[0].get_ydata()), 8)
            objectives = captured["q07_objective_quality_economic"].axes
            self.assertEqual(len(objectives), 3)
            self.assertTrue(
                any(bar.get_hatch() == "///" for bar in objectives[2].patches)
            )
            timing = captured["q08_optimization_time"].axes[0]
            self.assertEqual(timing.get_yscale(), "log")
            self.assertEqual(timing.get_ylabel(), "Optimization time (s; log scale)")
            self.assertEqual(len(timing.patches), 22)
            self.assertEqual(
                [t.get_text() for t in timing.get_xticklabels()],
                ["N", *[f"S{i}" for i in range(1, 11)]],
            )

    def test_package_publishes_only_eight_pngs_and_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            complete_chart_run(run)
            output = run / "report/figures"
            output.mkdir(parents=True)
            (output / "emulation").mkdir()
            obsolete = output / "emulation/all_output_parity.svg"
            obsolete.write_text("old generated figure")
            note = output / "notes.txt"
            note.write_text("user notes")
            package.generate_figures(run, output)
            self.assertEqual(
                {p.stem for p in output.rglob("*.png")}, set(comparison.FIGURES)
            )
            self.assertFalse(list(output.rglob("*.svg")))
            self.assertFalse(list(output.rglob("*.pdf")))
            self.assertEqual(note.read_text(), "user notes")
            index = pd.read_csv(output / "chart_index.csv")
            self.assertEqual(
                list(index.columns),
                ["results_sequence", "source_questions", "presentation_role", "png"],
            )
            self.assertEqual(index["results_sequence"].tolist(), list(range(1, 9)))
            self.assertTrue(
                all((output / name).stat().st_size > 0 for name in index["png"])
            )
            readme = (output / "README.md").read_text(encoding="utf-8")
            self.assertTrue(all(name in readme for name in index["png"]))
            self.assertNotIn("primary", readme.lower())
            self.assertNotIn("timing exclusions", readme.lower())

    def test_missing_scenario_blocks_publication_instead_of_omitting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            complete_chart_run(run)
            (
                run / "optimization/robustness_03/mechanistic_casewise_reference.npz"
            ).unlink()
            output = run / "report/figures"
            with self.assertRaisesRegex(ChartDataError, "robustness_03/mechanistic"):
                package.generate_figures(run, output)
            self.assertFalse(output.exists())

    def test_unavailable_audited_state_and_invalid_booleans_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            complete_chart_run(run)
            status = run / "optimization/nominal/surrogate_casewise_reference.json"
            status.write_text('{"candidate_available": false}')
            with self.assertRaisesRegex(ChartDataError, "nominal/surrogate"):
                load_chart_data(run)
        with self.assertRaisesRegex(ChartDataError, "Boolean"):
            boolean_values(pd.Series(["maybe"]), description="fixture")

    def test_metrics_average_coordinate_r_squared_and_keep_negative_values(self):
        truth = np.asarray([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        prediction = np.asarray([[2.0, 100.0], [3.0, 200.0], [4.0, 300.0]])
        nrmse, nmae, r_squared = comparison.coordinate_normalized_score(
            truth, prediction
        )
        self.assertAlmostEqual(nrmse, np.sqrt(0.125))
        self.assertAlmostEqual(nmae, 0.25)
        self.assertAlmostEqual(r_squared, 0.25)
        self.assertLess(
            comparison.coordinate_normalized_score(truth, prediction + 1000)[2], 0
        )
