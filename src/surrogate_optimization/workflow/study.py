"""Resumable generation, assessment, optimization, and reporting."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import os
from surrogate_optimization.config import PRODUCTION_PROFILE, StudyProfile
from surrogate_optimization.runtime.artifacts import atomic_json
from surrogate_optimization.runtime.contracts import (
    _build_contract,
    assert_source_unchanged,
    establish_contract,
    resolve_run_directory,
    source_file_digests,
    validate_production_profile,
)


@dataclass(frozen=True)
class RunSummary:
    result_directory: Path
    completed_stage: str
    status: str
    scientific_validation_passed: bool | None


def _prepare_run_directories(run: Path) -> None:
    for relative in (
        "inputs",
        "datasets",
        "models",
        "predictions",
        "metrics",
        "optimization",
        "report/tables",
        "report/figures",
    ):
        (run / relative).mkdir(parents=True, exist_ok=True)


def _write_state(run: Path, stage: str, status: str, **details: object) -> None:
    atomic_json(
        run / "run_state.json",
        {"stage": stage, "status": status, "pid": os.getpid(), **details},
    )


def run_study(run_id: str, through: str = "complete") -> RunSummary:
    """Execute the fixed production profile inside this checkout's results folder."""
    validate_production_profile(PRODUCTION_PROFILE)
    return _execute_pipeline(run_id, through, profile=PRODUCTION_PROFILE)


def _execute_pipeline(
    run_id: str, through: str, *, profile: StudyProfile
) -> RunSummary:
    """Shared internal execution; reduced profiles are used only by integration tests."""
    from surrogate_optimization.data.design import load_or_create_design
    from surrogate_optimization.workflow.generation import run_generation
    from surrogate_optimization.workflow.assessment import run_assessment
    from surrogate_optimization.workflow.optimization import run_optimization_stage
    from surrogate_optimization.reporting.finalization import finalize_reporting

    if through not in {"generation", "assessment", "complete"}:
        raise ValueError("through must be generation, assessment, or complete")
    run = resolve_run_directory(run_id)
    source_files = source_file_digests()
    contract = _build_contract(run_id, profile, source_files)
    run.mkdir(parents=True, exist_ok=True)
    establish_contract(run, contract)
    _prepare_run_directories(run)
    try:
        design = load_or_create_design(run, profile)
        assert_source_unchanged(source_files)
        _write_state(run, "generation", "running")
        generation = run_generation(
            run, design, profile=profile, source_files=source_files
        )
        if through == "generation":
            _write_state(run, "generation", "complete")
            return RunSummary(run, through, "complete", None)
        _write_state(run, "assessment", "running")
        analysis = run_assessment(
            run,
            generation.design,
            generation.development_responses,
            generation.holdout_responses,
            profile=profile,
            source_files=source_files,
        )
        if through == "assessment":
            status = (
                "complete" if analysis.passed else "complete_with_advisory_failures"
            )
            _write_state(
                run, "assessment", status, admission_gate_passed=bool(analysis.passed)
            )
            return RunSummary(run, through, status, bool(analysis.passed))
        _write_state(run, "optimization", "running")
        optimization_passed = run_optimization_stage(
            run=run,
            profile=profile,
            design=generation.design,
            development_responses=generation.development_responses,
            holdout_responses=generation.holdout_responses,
            analysis=analysis,
            source_files=source_files,
        )
        assert_source_unchanged(source_files)
        scientific_passed = bool(analysis.passed and optimization_passed)
        _write_state(run, "reporting", "running")
        finalize_reporting(
            run, source_files=source_files, scientific_passed=scientific_passed
        )
        status = (
            "complete" if scientific_passed else "complete_with_validation_failures"
        )
        _write_state(
            run,
            "complete",
            status,
            admission_gate_passed=bool(analysis.passed),
            optimization_validation_passed=bool(optimization_passed),
            scientific_validation_passed=scientific_passed,
        )
        return RunSummary(run, through, status, scientific_passed)
    except Exception as exc:
        state_path = run / "run_state.json"
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.is_file()
            else {}
        )
        (run / "complete.json").unlink(missing_ok=True)
        _write_state(
            run,
            str(state.get("stage", "unknown")),
            "failed",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        raise
