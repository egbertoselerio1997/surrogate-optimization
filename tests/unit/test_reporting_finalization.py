"""Final reporting binds its inputs, renders unavailable data, and resumes safely."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from surrogate_optimization.reporting import finalization
from surrogate_optimization.reporting.figures import comparison, package
from surrogate_optimization.reporting.figures.common import export_figure
import matplotlib.pyplot as plt
from surrogate_optimization.runtime.artifacts import atomic_json
from surrogate_optimization.runtime.contracts import (
    _artifacts_match,
    file_digest,
    source_digest,
)


class ReportingFinalizationTests(unittest.TestCase):
    def _fixture(self, run):
        sources = {"scientific.py": "frozen-source"}
        artifact = run / "metrics/scientific_audit.json"
        atomic_json(artifact, {"passed": False})
        atomic_json(
            run / "optimization/optimization_complete.json",
            {
                "source_digest": source_digest(sources),
                "artifacts": {"metrics/scientific_audit.json": file_digest(artifact)},
            },
        )
        return sources

    def _renderers(self):
        stack = ExitStack()

        def charts(run, output):
            output.mkdir(parents=True, exist_ok=True)
            for stem in comparison.FIGURES:
                (output / f"{stem}.png").write_bytes(b"figure fixture")
            for name in comparison.SIDECARS:
                (output / name).write_text("fixture metadata")
            return {}

        stack.enter_context(
            patch.object(package, "generate_figures", side_effect=charts)
        )

        def tables(run, **_kwargs):
            atomic_json(run / "report/tables/report_manifest.json", {"tables": []})
            (run / "report/tables/README.md").write_text("Table guide")
            return SimpleNamespace(tables={})

        stack.enter_context(
            patch.object(finalization, "write_reporting_tables", side_effect=tables)
        )
        stack.enter_context(patch.object(finalization, "assert_source_unchanged"))
        return stack

    def test_completion_and_resume_preserve_truthful_scientific_status(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            sources = self._fixture(run)
            with self._renderers():
                result = finalization.finalize_reporting(
                    run, source_files=sources, scientific_passed=False
                )
                self.assertEqual(result["status"], "complete_with_validation_failures")
                self.assertEqual(len(result["figures"]), 8)
                self.assertTrue(all(row["data_available"] for row in result["figures"]))
                self.assertTrue(_artifacts_match(run, result["artifacts"]))
                with patch.object(
                    package,
                    "generate_figures",
                    side_effect=AssertionError("render repeated"),
                ):
                    resumed = finalization.finalize_reporting(
                        run, source_files=sources, scientific_passed=False
                    )
                self.assertEqual(resumed, result)
                self.assertFalse(
                    json.loads((run / "complete.json").read_text())[
                        "scientific_validation_passed"
                    ]
                )

    def test_reporting_change_rebuilds_without_rewriting_numerical_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            sources = self._fixture(run)
            numerical = run / "optimization/optimization_complete.json"
            before = file_digest(numerical)
            with self._renderers():
                finalization.finalize_reporting(
                    run, source_files=sources, scientific_passed=False
                )
                with patch.object(
                    finalization,
                    "reporting_sources",
                    return_value={"plot.py": "changed"},
                ):
                    result = finalization.finalize_reporting(
                        run, source_files=sources, scientific_passed=False
                    )
                self.assertEqual(
                    result["binding"]["reporting_sources"], {"plot.py": "changed"}
                )
            self.assertEqual(file_digest(numerical), before)

    def test_corrupt_report_does_not_resume_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            sources = self._fixture(run)
            with self._renderers():
                finalization.finalize_reporting(
                    run, source_files=sources, scientific_passed=False
                )
                (run / "report/figures/q01_holdout_accuracy_overview.png").write_bytes(
                    b"corrupt"
                )
                with self.assertRaisesRegex(
                    RuntimeError, "reporting artifacts changed"
                ):
                    finalization.finalize_reporting(
                        run, source_files=sources, scientific_passed=False
                    )

    def test_failed_render_does_not_publish_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            sources = self._fixture(run)
            with (
                self._renderers(),
                patch.object(
                    package,
                    "generate_figures",
                    side_effect=ValueError("corrupt input"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "corrupt input"):
                    finalization.finalize_reporting(
                        run, source_files=sources, scientific_passed=False
                    )
            self.assertFalse((run / "complete.json").exists())

    def test_png_export_is_nonempty(self):
        with tempfile.TemporaryDirectory() as directory:
            figure, axis = plt.subplots()
            axis.plot([1, 2], [3, 4])
            path = Path(directory) / "figure.png"
            export_figure(figure, path)
            plt.close(figure)
            self.assertGreater(path.stat().st_size, 0)
