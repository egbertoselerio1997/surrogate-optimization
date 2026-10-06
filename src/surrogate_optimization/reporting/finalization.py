"""Bind tables and figures to validated numerical artifacts before completion."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Mapping
from surrogate_optimization.paths import REPOSITORY_ROOT
from surrogate_optimization.runtime.artifacts import atomic_bytes, atomic_json
from surrogate_optimization.runtime.contracts import (
    _artifacts_match,
    _artifact_hashes,
    file_digest,
    source_digest,
    assert_source_unchanged,
)
from surrogate_optimization.reporting.tables import write_reporting_tables


def reporting_sources() -> dict[str, str]:
    directory = REPOSITORY_ROOT / "src/surrogate_optimization/reporting"
    return {
        p.relative_to(REPOSITORY_ROOT).as_posix(): file_digest(p)
        for p in sorted(directory.rglob("*.py"))
    }


def finalize_reporting(
    run: Path, *, source_files: Mapping[str, str], scientific_passed: bool
) -> dict:
    from surrogate_optimization.reporting.figures import comparison, package

    (run / "complete.json").unlink(missing_ok=True)
    computation_digest = source_digest(source_files)
    numerical_marker = run / "optimization/optimization_complete.json"
    numerical = json.loads(numerical_marker.read_text(encoding="utf-8"))
    if numerical["source_digest"] != computation_digest or not _artifacts_match(
        run, numerical["artifacts"]
    ):
        raise RuntimeError("numerical artifacts are incompatible or changed")
    reporting_manifest = reporting_sources()
    binding = {
        "computation_digest": computation_digest,
        "reporting_sources": reporting_manifest,
        "numerical_marker_sha256": file_digest(numerical_marker),
    }
    manifest_path = run / "report/manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("binding") == binding:
            if not _artifacts_match(run, previous.get("artifacts", {})):
                raise RuntimeError("completed reporting artifacts changed")
            atomic_json(
                run / "complete.json",
                {
                    "schema_version": 1,
                    "status": previous["status"],
                    "scientific_validation_passed": scientific_passed,
                    "report_manifest_sha256": file_digest(manifest_path),
                },
            )
            return previous
    bundle = write_reporting_tables(
        run,
        output_directory=run / "report/tables",
        expected_cases=("nominal", *(f"robustness_{i:02d}" for i in range(1, 11))),
    )
    figures = []
    expected_paths = [run / "report/tables" / f"{name}.csv" for name in bundle.tables]
    expected_paths.append(run / "report/tables/report_manifest.json")
    expected_paths.append(run / "report/tables/README.md")
    output = run / "report/figures"
    package.generate_figures(run, output)
    for stem in comparison.FIGURES:
        path = output / f"{stem}.png"
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required figure was not written: {path}")
        figures.append(
            {
                "path": path.relative_to(run).as_posix(),
                "data_available": True,
                "reason": None,
            }
        )
        expected_paths.append(path)
    expected_paths.extend(output / name for name in comparison.SIDECARS)
    assert_source_unchanged(source_files)
    if reporting_sources() != reporting_manifest:
        raise RuntimeError("reporting source changed during rendering")
    status = "complete" if scientific_passed else "complete_with_validation_failures"
    atomic_bytes(
        run / "README.md",
        f"# Run results\n\nStatus: `{status}`. Scientific validation passed: `{scientific_passed}`.\n\nInputs and candidate checkpoints are in `inputs` and `datasets`. Fitted models are in `models`; predictions and audits are in `predictions` and `metrics`. Case searches and exact replays are in `optimization`. Tables and figures are in `report`; `report/manifest.json` records their hashes.\n".encode(),
    )
    expected_paths.append(run / "README.md")
    paths = tuple(sorted(expected_paths))
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required reporting artifact was not written: {path}")
    result = {
        "schema_version": 1,
        "binding": binding,
        "status": status,
        "scientific_validation_passed": scientific_passed,
        "figures": figures,
        "artifacts": _artifact_hashes(run, paths),
    }
    atomic_json(manifest_path, result)
    atomic_json(
        run / "complete.json",
        {
            "schema_version": 1,
            "status": status,
            "scientific_validation_passed": scientific_passed,
            "report_manifest_sha256": file_digest(manifest_path),
        },
    )
    return result
