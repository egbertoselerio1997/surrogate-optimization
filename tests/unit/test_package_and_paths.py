"""Installed interfaces, source bindings, and repository-owned outputs."""

from __future__ import annotations
from contextlib import redirect_stdout
from io import StringIO
import importlib
import json
import pkgutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import surrogate_optimization
from surrogate_optimization.cli import run_study as cli
from surrogate_optimization.paths import REPOSITORY_ROOT, RESULTS_ROOT
from surrogate_optimization.runtime.artifacts import atomic_json
from surrogate_optimization.runtime.contracts import (
    _artifacts_match,
    _build_contract,
    establish_contract,
    file_digest,
    resolve_run_directory,
    source_file_digests,
    validate_production_profile,
)
from surrogate_optimization.config import PRODUCTION_PROFILE
from surrogate_optimization.reporting.finalization import reporting_sources
from tests.support.profiles import INTEGRATION_PROFILE


class InstalledPackageTests(unittest.TestCase):
    def test_all_modules_import_without_creating_outputs(self):
        before = sorted(RESULTS_ROOT.glob("*")) if RESULTS_ROOT.exists() else []
        for item in pkgutil.walk_packages(
            surrogate_optimization.__path__, surrogate_optimization.__name__ + "."
        ):
            importlib.import_module(item.name)
        self.assertEqual(
            sorted(RESULTS_ROOT.glob("*")) if RESULTS_ROOT.exists() else [], before
        )

    def test_cli_help_works_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "surrogate_optimization.cli.run_study",
                    "--help",
                ],
                cwd=directory,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("--candidate-count", result.stdout)
            self.assertFalse((Path(directory) / "results").exists())
        self.assertEqual(
            resolve_run_directory("production_001"), RESULTS_ROOT / "production_001"
        )

    def test_cli_rejects_reduced_counts_before_execution(self):
        with patch.object(cli, "run_study") as execute, redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["--candidate-count", "100"])
            execute.assert_not_called()

    def test_production_validator_rejects_integration_profile(self):
        validate_production_profile(PRODUCTION_PROFILE)
        with self.assertRaises(RuntimeError):
            validate_production_profile(INTEGRATION_PROFILE)

    def test_safe_run_ids_do_not_require_historical_prefix(self):
        self.assertEqual(
            resolve_run_directory("experiment-001"), RESULTS_ROOT / "experiment-001"
        )
        for value in ("../escape", "a/b", "a\\b", "", "CON", "LPT1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_run_directory(value)


class SourceAndArtifactTests(unittest.TestCase):
    def test_source_manifests_cover_workers_and_separate_reporting(self):
        scientific = source_file_digests()
        reporting = reporting_sources()
        all_sources = {
            p.relative_to(REPOSITORY_ROOT).as_posix()
            for p in (REPOSITORY_ROOT / "src/surrogate_optimization").rglob("*.py")
        }
        self.assertEqual((set(scientific) | set(reporting)) & all_sources, all_sources)
        self.assertFalse(set(scientific) & set(reporting))
        self.assertIn("src/surrogate_optimization/data/random_design.py", scientific)
        self.assertIn("src/surrogate_optimization/runtime/parallel.py", scientific)
        self.assertIn("config/parameters.json", scientific)

    def test_contract_resume_is_immutable_and_rejects_historical_folders(self):
        contract = _build_contract(
            "fixture", PRODUCTION_PROFILE, {"science.py": "digest"}
        )
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            establish_contract(run, contract)
            digest = file_digest(run / "run_contract.json")
            establish_contract(run, contract)
            self.assertEqual(file_digest(run / "run_contract.json"), digest)
            with self.assertRaises(RuntimeError):
                establish_contract(run, {**contract, "schema_version": 12})
            self.assertEqual(file_digest(run / "run_contract.json"), digest)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            old = run / "inputs/contract.json"
            atomic_json(old, {"schema_version": 12})
            with self.assertRaises(RuntimeError):
                establish_contract(run, contract)
            self.assertEqual(json.loads(old.read_text()), {"schema_version": 12})

    def test_artifact_hashes_reject_tampering_and_parent_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            atomic_json(run / "value.json", {"value": 1})
            expected = {"value.json": file_digest(run / "value.json")}
            self.assertTrue(_artifacts_match(run, expected))
            self.assertFalse(
                _artifacts_match(
                    run,
                    {"../run/value.json": expected["value.json"], "../outside": "x"},
                )
            )
            atomic_json(run / "value.json", {"value": 2})
            self.assertFalse(_artifacts_match(run, expected))
