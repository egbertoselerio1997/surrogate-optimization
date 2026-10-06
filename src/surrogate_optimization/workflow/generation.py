"""Workflow generation."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.workflow.types import GenerationResult
    from surrogate_optimization.data.generation import MechanisticBlockResult
    from surrogate_optimization.config import StudyProfile
from pathlib import Path
from time import perf_counter
from typing import Mapping
import json
import numpy as np
import pandas as pd


def _validate_generation_block(
    mechanistic_responses: np.ndarray,
    diagnostics: pd.DataFrame,
    *,
    block: str,
    attempted_count: int,
    profile: StudyProfile,
) -> int:
    from surrogate_optimization.plant.model import PHYSICAL_BALANCE_TOLERANCE

    accepted_count = len(mechanistic_responses)
    if (
        accepted_count < 1
        or accepted_count > attempted_count
        or mechanistic_responses.shape
        != (accepted_count, profile.mechanistic_response_count)
        or (not np.all(np.isfinite(mechanistic_responses)))
    ):
        raise RuntimeError(f"{block} target block is incomplete or non-finite")
    required = {
        "row",
        "accepted",
        "root_difference_inf",
        "branch_agreement",
        "mass_residual_start_1",
        "mass_residual_start_2",
        "state_negativity_start_1",
        "state_negativity_start_2",
        "rate_negativity_start_1",
        "rate_negativity_start_2",
        "largest_real_eigenvalue_start_1",
        "largest_real_eigenvalue_start_2",
        "stability_agreement_start_1",
        "stability_agreement_start_2",
        "feed_tss_start_1",
        "feed_tss_start_2",
        "external_solids_loss_start_1",
        "external_solids_loss_start_2",
    }
    missing = required - set(diagnostics.columns)
    if missing:
        raise RuntimeError(f"{block} diagnostics omit columns: {sorted(missing)}")
    rows = np.asarray(diagnostics["row"], dtype=int)
    if len(diagnostics) != accepted_count or not np.array_equal(
        np.sort(rows), np.arange(accepted_count)
    ):
        raise RuntimeError(
            f"{block} diagnostics do not cover every accepted row exactly once"
        )
    accepted = (
        diagnostics["accepted"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    branches = (
        diagnostics["branch_agreement"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    if accepted.isna().any() or not bool(accepted.all()):
        raise RuntimeError(f"{block} contains an unaccepted fixed mechanistic row")
    if branches.isna().any() or not bool(branches.all()):
        raise RuntimeError(f"{block} contains a two-start branch disagreement")
    numeric_columns = list(required - {"row", "accepted", "branch_agreement"})
    numeric = diagnostics[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.all(np.isfinite(numeric.to_numpy())):
        raise RuntimeError(f"{block} mechanistic audits contain non-finite values")
    checks = (
        ("root_difference_inf", np.less_equal, 1e-06),
        ("mass_residual_start_1", np.less_equal, PHYSICAL_BALANCE_TOLERANCE),
        ("mass_residual_start_2", np.less_equal, PHYSICAL_BALANCE_TOLERANCE),
        ("state_negativity_start_1", np.less_equal, 1e-10),
        ("state_negativity_start_2", np.less_equal, 1e-10),
        ("rate_negativity_start_1", np.less_equal, 1e-12),
        ("rate_negativity_start_2", np.less_equal, 1e-12),
        ("largest_real_eigenvalue_start_1", np.less_equal, -1e-08),
        ("largest_real_eigenvalue_start_2", np.less_equal, -1e-08),
        ("stability_agreement_start_1", np.less_equal, 1e-06),
        ("stability_agreement_start_2", np.less_equal, 1e-06),
        ("feed_tss_start_1", np.greater_equal, 1.0),
        ("feed_tss_start_2", np.greater_equal, 1.0),
        ("external_solids_loss_start_1", np.greater_equal, 1.0),
        ("external_solids_loss_start_2", np.greater_equal, 1.0),
    )
    failures = [
        name
        for name, comparison, limit in checks
        if not bool(np.all(comparison(numeric[name].to_numpy(), limit)))
    ]
    if failures:
        raise RuntimeError(f"{block} failed mechanistic generation gates: {failures}")
    return accepted_count


def _load_generation_checkpoint(
    run: Path,
    *,
    block: str,
    count: int,
    profile: StudyProfile,
    source_id: str,
    design_id: str,
) -> tuple[MechanisticBlockResult, float] | None:
    from surrogate_optimization.config import DECISION_LOWER
    from surrogate_optimization.config import DECISION_UPPER
    from surrogate_optimization.data.generation import MechanisticBlockResult
    from surrogate_optimization.plant.definitions import INFLUENT_LOWER
    from surrogate_optimization.plant.definitions import INFLUENT_UPPER
    from surrogate_optimization.runtime.contracts import _artifacts_match
    from surrogate_optimization.runtime.contracts import (
        _checkpoint_source_is_authorized,
    )
    from surrogate_optimization.runtime.contracts import array_digest

    marker_path = run / "datasets" / block / "block_complete.json"
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_source_id = str(marker.get("source_digest", ""))
        if (
            not _checkpoint_source_is_authorized(
                run,
                stage=f"generation/{block}",
                checkpoint=marker_path,
                observed_source_id=marker_source_id,
                current_source_id=source_id,
            )
            or marker.get("design_digest") != design_id
            or marker.get("block") != block
            or (int(marker.get("row_count", -1)) != count)
            or (not _artifacts_match(run, marker.get("artifacts", {})))
        ):
            raise RuntimeError("completed checkpoint is inconsistent or changed")
        output = run / "datasets" / block
        target_path = output / "accepted_mechanistic_responses.npz"
        input_path = output / "accepted_inputs.npz"
        with np.load(target_path, allow_pickle=False) as stored:
            mechanistic_responses = np.asarray(
                stored["mechanistic_responses"], dtype=float
            )
        with np.load(input_path, allow_pickle=False) as stored:
            controls = np.asarray(stored["controls"], dtype=float)
            influents = np.asarray(stored["influents"], dtype=float)
            source_candidate_id = np.asarray(stored["source_candidate_id"], dtype=str)
        diagnostics = pd.read_csv(output / "accepted_diagnostics.csv")
        attempts = pd.read_csv(output / "all_attempts.csv")
        provenance = pd.read_csv(output / "accepted_provenance.csv")
        accepted_count = _validate_generation_block(
            mechanistic_responses,
            diagnostics,
            block=block,
            attempted_count=count,
            profile=profile,
        )
        if (
            int(marker.get("accepted_count", -1)) != accepted_count
            or controls.shape != (accepted_count, 7)
            or influents.shape != (accepted_count, 20)
            or (not np.all(np.isfinite(controls)))
            or (not np.all(np.isfinite(influents)))
            or np.any(controls < DECISION_LOWER)
            or np.any(controls > DECISION_UPPER)
            or np.any(influents < INFLUENT_LOWER)
            or np.any(influents > INFLUENT_UPPER)
            or (len(attempts) != count)
            or (
                int(
                    _boolean_series(
                        attempts["accepted"], description="attempt ledger"
                    ).sum()
                )
                != accepted_count
            )
            or (len(provenance) != accepted_count)
            or (len(source_candidate_id) != accepted_count)
            or (
                not np.array_equal(
                    source_candidate_id,
                    provenance["source_candidate_id"].to_numpy(dtype=str),
                )
            )
            or (
                marker.get("effective_input_digest")
                != array_digest(
                    controls=np.asarray(controls, dtype="<f8"),
                    influents=np.asarray(influents, dtype="<f8"),
                )
            )
        ):
            raise RuntimeError("completed checkpoint is inconsistent or changed")
        _validate_attempt_checkpoint_hashes(output, attempts)
        return (
            MechanisticBlockResult(
                controls=controls,
                influents=influents,
                mechanistic_responses=mechanistic_responses,
                diagnostics=diagnostics,
                attempts=attempts,
                provenance=provenance,
            ),
            float(marker["elapsed_seconds"]),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise RuntimeError("completed checkpoint is corrupt")


def _validate_attempt_checkpoint_hashes(output: Path, attempts: pd.DataFrame) -> None:
    from surrogate_optimization.runtime.contracts import file_digest

    required = {"checkpoint_path", "checkpoint_sha256", "candidate_id", "accepted"}
    if required - set(attempts.columns) or attempts.empty:
        raise RuntimeError("generation attempt ledger is incomplete")
    if attempts["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("generation attempt ledger contains duplicate candidates")
    root = output.resolve()
    for row in attempts.itertuples(index=False):
        path = (output / str(row.checkpoint_path)).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError(
                "generation attempt checkpoint is missing or outside its block"
            )
        if file_digest(path) != str(row.checkpoint_sha256):
            raise RuntimeError(f"generation attempt checkpoint changed: {path.name}")


def _boolean_series(values: pd.Series, *, description: str) -> pd.Series:
    converted = values.astype(str).str.lower().map({"true": True, "false": False})
    if converted.isna().any():
        raise RuntimeError(f"{description} contains invalid Boolean values")
    return converted.astype(bool)


def _write_generation_audits(output: Path, result: MechanisticBlockResult) -> None:
    from surrogate_optimization.config import DECISION_LOWER
    from surrogate_optimization.config import DECISION_NAMES
    from surrogate_optimization.config import DECISION_UPPER
    from surrogate_optimization.plant.definitions import COMPONENTS
    from surrogate_optimization.plant.definitions import INFLUENT_LOWER
    from surrogate_optimization.plant.definitions import INFLUENT_UPPER
    from surrogate_optimization.runtime.artifacts import atomic_dataframe

    physical = np.column_stack((result.controls, result.influents))
    lower = np.concatenate((DECISION_LOWER, INFLUENT_LOWER))
    upper = np.concatenate((DECISION_UPPER, INFLUENT_UPPER))
    names = tuple(DECISION_NAMES) + tuple(COMPONENTS)
    coverage = pd.DataFrame(
        [
            {
                "coordinate_index": index,
                "coordinate_group": "decision" if index < 7 else "influent",
                "coordinate": names[index],
                "declared_lower": lower[index],
                "declared_upper": upper[index],
                "accepted_minimum": float(np.min(physical[:, index])),
                "accepted_maximum": float(np.max(physical[:, index])),
                "accepted_span_fraction": float(
                    np.ptp(physical[:, index]) / (upper[index] - lower[index])
                ),
                "outside_declared_box_count": int(
                    np.count_nonzero(
                        (physical[:, index] < lower[index])
                        | (physical[:, index] > upper[index])
                    )
                ),
            }
            for index in range(physical.shape[1])
        ]
    )
    attempts = result.attempts.copy()
    if "rejection_reason" not in attempts.columns:
        raise RuntimeError("generation attempt ledger omits rejection_reason")
    accepted = _boolean_series(attempts["accepted"], description="attempt ledger")
    reasons = attempts["rejection_reason"].astype(str)
    if bool(
        (accepted & reasons.ne("accepted") | ~accepted & reasons.eq("accepted")).any()
    ):
        raise RuntimeError(
            "generation rejection reasons disagree with acceptance flags"
        )
    rejected = attempts.loc[attempts["rejection_reason"] != "accepted"]
    if rejected.empty:
        rejection_summary = pd.DataFrame(
            columns=("rejection_reason", "attempt_count", "fraction_of_all_attempts")
        )
    else:
        rejection_summary = (
            rejected.groupby("rejection_reason", dropna=False)
            .size()
            .rename("attempt_count")
            .reset_index()
            .sort_values("rejection_reason", kind="stable")
            .reset_index(drop=True)
        )
        rejection_summary["fraction_of_all_attempts"] = rejection_summary[
            "attempt_count"
        ] / len(attempts)
    atomic_dataframe(output / "accepted_coordinate_coverage.csv", coverage)
    atomic_dataframe(output / "rejection_reason_summary.csv", rejection_summary)


def _generation_publication_paths(output: Path) -> tuple[Path, ...]:
    names = (
        "accepted_mechanistic_responses.npz",
        "accepted_inputs.npz",
        "accepted_diagnostics.csv",
        "all_attempts.csv",
        "accepted_provenance.csv",
        "candidate_checkpoint_summary.csv",
        "generation_summary.json",
        "accepted_coordinate_coverage.csv",
        "rejection_reason_summary.csv",
    )
    return tuple((output / name for name in names))


def _run_generation_block(
    run: Path,
    design: Mapping[str, object],
    *,
    block: str,
    profile: StudyProfile,
    source_files: Mapping[str, str],
    design_id: str,
) -> tuple[MechanisticBlockResult, float, bool]:
    from surrogate_optimization.data.generation import (
        generate_mechanistic_block_from_fixed_design,
    )
    from surrogate_optimization.runtime.artifacts import atomic_json
    from surrogate_optimization.runtime.contracts import _artifact_hashes
    from surrogate_optimization.runtime.contracts import array_digest
    from surrogate_optimization.runtime.contracts import assert_source_unchanged
    from surrogate_optimization.runtime.contracts import source_digest

    count = (
        profile.development_candidate_count
        if block == "development"
        else profile.holdout_candidate_count
    )
    source_id = source_digest(source_files)
    checkpoint = _load_generation_checkpoint(
        run,
        block=block,
        count=count,
        profile=profile,
        source_id=source_id,
        design_id=design_id,
    )
    if checkpoint is not None:
        return (*checkpoint, True)
    started = perf_counter()
    result = generate_mechanistic_block_from_fixed_design(
        np.asarray(design[f"{block}_controls"]),
        np.asarray(design[f"{block}_influents"]),
        profile,
        run / "datasets" / block,
        block=block,
    )
    elapsed = perf_counter() - started
    accepted_count = _validate_generation_block(
        result.mechanistic_responses,
        result.diagnostics,
        block=block,
        attempted_count=count,
        profile=profile,
    )
    _validate_attempt_checkpoint_hashes(run / "datasets" / block, result.attempts)
    _write_generation_audits(run / "datasets" / block, result)
    assert_source_unchanged(source_files)
    paths = _generation_publication_paths(run / "datasets" / block)
    if not all((path.is_file() for path in paths)):
        raise RuntimeError(f"{block} generator did not publish its required artifacts")
    atomic_json(
        run / "datasets" / block / "block_complete.json",
        {
            "stage": "mechanistic_generation",
            "block": block,
            "source_digest": source_id,
            "design_digest": design_id,
            "row_count": count,
            "accepted_count": accepted_count,
            "target_shape": list(result.mechanistic_responses.shape),
            "attempt_count": len(result.attempts),
            "rejected_attempt_count": int(
                (
                    ~_boolean_series(
                        result.attempts["accepted"], description="attempt ledger"
                    )
                ).sum()
            ),
            "effective_input_digest": array_digest(
                controls=np.asarray(result.controls, dtype="<f8"),
                influents=np.asarray(result.influents, dtype="<f8"),
            ),
            "elapsed_seconds": elapsed,
            "artifacts": _artifact_hashes(run, paths),
        },
    )
    return (result, elapsed, False)


def run_generation(
    run: Path,
    design: Mapping[str, object],
    *,
    profile: StudyProfile,
    source_files: Mapping[str, str],
) -> GenerationResult:
    from surrogate_optimization.data.design import _design_digest
    from surrogate_optimization.runtime.artifacts import _json_ready
    from surrogate_optimization.runtime.artifacts import atomic_dataframe
    from surrogate_optimization.runtime.artifacts import atomic_json
    from surrogate_optimization.runtime.artifacts import atomic_npz
    from surrogate_optimization.runtime.contracts import (
        _checkpoint_source_is_authorized,
    )
    from surrogate_optimization.runtime.contracts import _load_json_object
    from surrogate_optimization.runtime.contracts import assert_source_unchanged
    from surrogate_optimization.runtime.contracts import file_digest
    from surrogate_optimization.runtime.contracts import source_digest
    from surrogate_optimization.runtime.protocols import DESIGN_ARRAYS
    from surrogate_optimization.workflow.types import GenerationResult

    design_id = _design_digest(design)
    blocks: dict[str, tuple[MechanisticBlockResult, float, bool]] = {}
    for block in ("development", "holdout"):
        blocks[block] = _run_generation_block(
            run,
            design,
            block=block,
            profile=profile,
            source_files=source_files,
            design_id=design_id,
        )
    summary = pd.DataFrame(
        [
            {
                "block": block,
                "candidate_rows": profile.development_candidate_count
                if block == "development"
                else profile.holdout_candidate_count,
                "accepted_rows": len(result[0].diagnostics),
                "total_attempts": len(result[0].attempts),
                "excluded_rejected_attempts": int(
                    (
                        ~_boolean_series(
                            result[0].attempts["accepted"], description="attempt ledger"
                        )
                    ).sum()
                ),
                "elapsed_seconds": result[1],
                "reused_complete_checkpoint": result[2],
            }
            for block, result in blocks.items()
        ]
    )
    summary_path = run / "metrics" / "mechanistic_generation_summary.csv"
    source_id = source_digest(source_files)
    marker_source_ids = {
        str(
            _load_json_object(
                run / "datasets" / block / "block_complete.json",
                description=f"{block} generation marker",
            ).get("source_digest", "")
        )
        for block in blocks
    }
    carried_forward = bool(
        all((result[2] for result in blocks.values()))
        and marker_source_ids != {source_id}
    )
    if carried_forward:
        if len(marker_source_ids) != 1 or not _checkpoint_source_is_authorized(
            run,
            stage="generation/summary",
            checkpoint=summary_path,
            observed_source_id=next(iter(marker_source_ids)),
            current_source_id=source_id,
        ):
            raise RuntimeError("generation summary is not authorized for reuse")
    else:
        atomic_dataframe(summary_path, summary)
    effective_design = dict(design)
    for block, result in blocks.items():
        effective_design[f"{block}_controls"] = result[0].controls
        effective_design[f"{block}_influents"] = result[0].influents
    effective_path = run / "datasets" / "effective_design.npz"
    effective_arrays = {
        name: np.asarray(effective_design[name]) for name in DESIGN_ARRAYS
    }
    if effective_path.is_file():
        with np.load(effective_path, allow_pickle=False) as stored:
            if set(stored.files) != set(DESIGN_ARRAYS) or any(
                (
                    not np.array_equal(stored[name], effective_arrays[name])
                    for name in DESIGN_ARRAYS
                )
            ):
                raise RuntimeError(
                    "existing effective design differs from accepted inputs"
                )
    else:
        atomic_npz(effective_path, **effective_arrays)
    effective_id = _design_digest(effective_design)
    effective_manifest = {
        "schema": 1,
        "base_design_digest": design_id,
        "effective_design_digest": effective_id,
        "source_digest": source_id,
        "artifact": {
            effective_path.relative_to(run).as_posix(): file_digest(effective_path)
        },
        "blocks": {
            block: {
                "accepted_input_artifact": (
                    run / "datasets" / block / "accepted_inputs.npz"
                )
                .relative_to(run)
                .as_posix(),
                "accepted_input_artifact_sha256": file_digest(
                    run / "datasets" / block / "accepted_inputs.npz"
                ),
                "accepted_row_count": len(result[0].mechanistic_responses),
            }
            for block, result in blocks.items()
        },
    }
    effective_manifest_path = run / "datasets" / "effective_design_manifest.json"
    if effective_manifest_path.is_file():
        existing_manifest = _load_json_object(
            effective_manifest_path, description="effective design manifest"
        )
        expected_manifest = _json_ready(effective_manifest)
        existing_source_id = str(existing_manifest.get("source_digest", ""))
        existing_without_source = dict(existing_manifest)
        expected_without_source = dict(expected_manifest)
        existing_without_source.pop("source_digest", None)
        expected_without_source.pop("source_digest", None)
        if (
            existing_without_source != expected_without_source
            or not _checkpoint_source_is_authorized(
                run,
                stage="generation/effective_design",
                checkpoint=effective_manifest_path,
                observed_source_id=existing_source_id,
                current_source_id=source_id,
            )
        ):
            raise RuntimeError("existing effective design manifest differs")
    else:
        atomic_json(effective_manifest_path, effective_manifest)
    assert_source_unchanged(source_files)
    return GenerationResult(
        design=effective_design,
        development_responses=blocks["development"][0].mechanistic_responses,
        holdout_responses=blocks["holdout"][0].mechanistic_responses,
    )
