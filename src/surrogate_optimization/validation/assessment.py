"""Validation assessment."""

from __future__ import annotations

_HOLDOUT_PROJECTION_CONTEXT = None
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.runtime.parallel import BatchProgress
    from surrogate_optimization.surrogate.regression import LogOverflowTSSClosure
    from surrogate_optimization.surrogate.projection import NetworkLayout
    from surrogate_optimization.surrogate.regression import QuadraticSurrogate
    from surrogate_optimization.config import StudyProfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Callable
from typing import Mapping
import json
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssessmentResult:
    metrics: pd.DataFrame
    violations: pd.DataFrame
    qp_diagnostics: pd.DataFrame
    feasibility: pd.DataFrame
    raw: np.ndarray
    projected: np.ndarray
    projected_targets: np.ndarray
    overflow_tss_closure: np.ndarray | None = None


def _initialize_holdout_projection_worker(
    raw: np.ndarray,
    mechanistic_responses: np.ndarray,
    controls: np.ndarray,
    influents: np.ndarray,
    layout: NetworkLayout,
    state_scale: np.ndarray,
    equality_scale: np.ndarray,
    inequality_scale: np.ndarray,
    overflow_tss_closure: np.ndarray | None,
) -> None:
    from surrogate_optimization.surrogate.projection import PhysicalProjector

    global _HOLDOUT_PROJECTION_CONTEXT
    _HOLDOUT_PROJECTION_CONTEXT = (
        np.asarray(raw, dtype=np.float64),
        np.asarray(mechanistic_responses, dtype=np.float64),
        np.asarray(controls, dtype=np.float64),
        np.asarray(influents, dtype=np.float64),
        layout,
        np.asarray(state_scale, dtype=np.float64),
        np.asarray(equality_scale, dtype=np.float64),
        np.asarray(inequality_scale, dtype=np.float64),
        None
        if overflow_tss_closure is None
        else np.asarray(overflow_tss_closure, dtype=np.float64),
        PhysicalProjector(
            state_scale,
            equality_scale,
            inequality_scale,
            absolute_tolerance=1e-12,
            relative_tolerance=1e-12,
        ),
    )


def _holdout_projection_batch(bounds: tuple[int, int]) -> Mapping[str, np.ndarray]:
    from surrogate_optimization.validation.physical import _operators
    from surrogate_optimization.validation.physical import violation_record

    if _HOLDOUT_PROJECTION_CONTEXT is None:
        raise RuntimeError("holdout-projection worker was not initialized")
    (
        raw,
        mechanistic_responses,
        controls,
        influents,
        layout,
        state_scale,
        equality_scale,
        inequality_scale,
        overflow_tss_closure,
        projector,
    ) = _HOLDOUT_PROJECTION_CONTEXT
    start, stop = bounds
    count = stop - start
    projected = np.empty((count, layout.state_size), dtype=np.float64)
    projected_targets = np.empty_like(projected)
    qp_rows: list[str] = []
    feasibility_rows: list[str] = []
    violation_rows: list[str] = []
    for local, row in enumerate(range(start, stop)):
        closure_value = (
            None if overflow_tss_closure is None else float(overflow_tss_closure[row])
        )
        operators = _operators(
            controls[row], influents[row], layout, overflow_tss_closure=closure_value
        )
        started = perf_counter_ns()
        projection = projector.project(
            raw[row],
            operators.equality_matrix,
            operators.equality_rhs,
            operators.inequality_matrix,
            raise_on_failure=False,
        )
        qp_elapsed = perf_counter_ns() - started
        projected[local] = projection.state
        target_projection = projector.project(
            mechanistic_responses[row],
            operators.equality_matrix,
            operators.equality_rhs,
            operators.inequality_matrix,
            raise_on_failure=False,
        )
        projected_targets[local] = target_projection.state
        for kind, result, elapsed in (
            ("raw_prediction", projection, qp_elapsed),
            ("mechanistic_target", target_projection, math.nan),
        ):
            qp_rows.append(
                json.dumps(
                    {
                        "row": row,
                        "projection_input": kind,
                        "accepted": bool(result.accepted),
                        "elapsed_ns": elapsed,
                        **result.diagnostics.as_dict(),
                    }
                )
            )
        raw_distance = float(
            np.linalg.norm((raw[row] - mechanistic_responses[row]) / state_scale)
        )
        projected_distance = float(
            np.linalg.norm(
                (projected[local] - mechanistic_responses[row]) / state_scale
            )
        )
        target_feasibility = float(
            np.linalg.norm(
                (projected_targets[local] - mechanistic_responses[row]) / state_scale
            )
        )
        upper_bound = raw_distance + target_feasibility
        feasibility_rows.append(
            json.dumps(
                {
                    "row": row,
                    "target_feasibility_distance": target_feasibility,
                    "raw_distance": raw_distance,
                    "projected_distance": projected_distance,
                    "finite_feasibility_bound": upper_bound,
                    "bound_slack": upper_bound - projected_distance,
                    "bound_passed": bool(projected_distance <= upper_bound + 1e-10),
                    "raw_projection_qp_passed": bool(projection.accepted),
                    "target_projection_qp_passed": bool(target_projection.accepted),
                }
            )
        )
        for method, state in (
            ("raw", raw[row]),
            ("projected", projected[local]),
            ("mechanistic", mechanistic_responses[row]),
        ):
            violation_rows.append(
                json.dumps(
                    violation_record(
                        method,
                        f"holdout_{row:04d}",
                        state,
                        controls[row],
                        influents[row],
                        layout,
                        equality_scale,
                        inequality_scale,
                        state_scale,
                        closure_value,
                    )
                )
            )
    return {
        "projected": projected,
        "projected_targets": projected_targets,
        "qp_json": np.asarray(qp_rows),
        "feasibility_json": np.asarray(feasibility_rows),
        "violations_json": np.asarray(violation_rows),
    }


def _validate_holdout_projection_batch(
    start: int, stop: int, payload: Mapping[str, np.ndarray]
) -> None:
    count = stop - start
    projected = np.asarray(payload["projected"])
    projected_targets = np.asarray(payload["projected_targets"])
    if (
        projected.ndim != 2
        or projected.shape[0] != count
        or projected_targets.shape != projected.shape
        or (np.asarray(payload["qp_json"]).shape != (2 * count,))
        or (np.asarray(payload["feasibility_json"]).shape != (count,))
        or (np.asarray(payload["violations_json"]).shape != (3 * count,))
    ):
        raise ValueError("holdout-projection batch payload has invalid dimensions")
    qp = [json.loads(str(value)) for value in np.asarray(payload["qp_json"])]
    feasibility = [
        json.loads(str(value)) for value in np.asarray(payload["feasibility_json"])
    ]
    violations = [
        json.loads(str(value)) for value in np.asarray(payload["violations_json"])
    ]
    expected_rows = list(range(start, stop))
    if [int(item.get("row", -1)) for item in qp] != [
        row for row in expected_rows for _ in range(2)
    ] or [item.get("projection_input") for item in qp] != [
        kind for _ in expected_rows for kind in ("raw_prediction", "mechanistic_target")
    ]:
        raise ValueError("holdout-projection QP row ordering is invalid")
    if [int(item.get("row", -1)) for item in feasibility] != expected_rows:
        raise ValueError("holdout-projection feasibility row ordering is invalid")
    if [item.get("case") for item in violations] != [
        f"holdout_{row:04d}" for row in expected_rows for _ in range(3)
    ] or [item.get("method") for item in violations] != [
        method for _ in expected_rows for method in ("raw", "projected", "mechanistic")
    ]:
        raise ValueError("holdout-projection violation row ordering is invalid")


def assess_raw_projected_mechanistic(
    model: QuadraticSurrogate,
    development_controls: np.ndarray,
    development_influents: np.ndarray,
    development_responses: np.ndarray,
    holdout_controls: np.ndarray,
    holdout_influents: np.ndarray,
    holdout_responses: np.ndarray,
    profile: StudyProfile,
    *,
    overflow_closure: LogOverflowTSSClosure | None = None,
    development_overflow_tss_closure: np.ndarray | None = None,
    parallel_workers: int = 1,
    batch_size: int = 64,
    checkpoint_directory: Path | None = None,
    checkpoint_contract: str | None = None,
    progress: Callable[[BatchProgress], None] | None = None,
) -> AssessmentResult:
    from surrogate_optimization.plant.definitions import INVARIANT_MATRIX
    from surrogate_optimization.plant.definitions import TSS_VECTOR
    from surrogate_optimization.runtime.parallel import run_resumable_batches
    from surrogate_optimization.surrogate.projection import NetworkLayout
    from surrogate_optimization.surrogate.projection import fit_network_row_scales
    from surrogate_optimization.surrogate.training import overflow_tss_from_response
    from surrogate_optimization.validation.physical import _overflow_closure_metric_rows
    from surrogate_optimization.validation.physical import _prediction_metric_rows

    layout = NetworkLayout(layer_count=profile.layer_count)
    state_scale = model.response_scale
    holdout_closure = (
        None
        if overflow_closure is None
        else np.asarray(
            overflow_closure.predict(holdout_controls, holdout_influents),
            dtype=np.float64,
        )
    )
    if overflow_closure is not None:
        development_closure = np.asarray(
            development_overflow_tss_closure
            if development_overflow_tss_closure is not None
            else overflow_closure.predict(development_controls, development_influents),
            dtype=np.float64,
        )
        if development_closure.shape != (len(development_controls),):
            raise ValueError("development overflow-TSS closure predictions are invalid")
    else:
        development_closure = None
    row_scales = fit_network_row_scales(
        development_responses,
        development_influents,
        internal_recycle=development_controls[:, 4],
        return_recycle=development_controls[:, 5],
        waste_fraction=development_controls[:, 6],
        invariant_operator=INVARIANT_MATRIX,
        tss_weights=TSS_VECTOR,
        layout=layout,
        minimum_scale=1.0,
        overflow_tss_closure=development_closure,
    )
    raw = model.predict(holdout_controls, holdout_influents)
    if checkpoint_directory is not None and (not checkpoint_contract):
        raise ValueError(
            "checkpoint_contract is required when holdout checkpoints are enabled"
        )
    batches = run_resumable_batches(
        stage="whole_system_holdout_projection_audit",
        row_count=len(raw),
        batch_size=batch_size,
        parallel_workers=parallel_workers,
        checkpoint_directory=checkpoint_directory,
        contract_digest=checkpoint_contract or "unpersisted",
        payload_names=(
            "projected",
            "projected_targets",
            "qp_json",
            "feasibility_json",
            "violations_json",
        ),
        worker=_holdout_projection_batch,
        validate=_validate_holdout_projection_batch,
        initializer=_initialize_holdout_projection_worker,
        initargs=(
            raw,
            holdout_responses,
            holdout_controls,
            holdout_influents,
            layout,
            state_scale,
            row_scales.equality,
            row_scales.inequality,
            holdout_closure,
        ),
        progress=progress,
    )
    projected = np.vstack([batch["projected"] for batch in batches])
    projected_targets = np.vstack([batch["projected_targets"] for batch in batches])
    qp_rows = [
        json.loads(str(record)) for batch in batches for record in batch["qp_json"]
    ]
    feasibility_rows = [
        json.loads(str(record))
        for batch in batches
        for record in batch["feasibility_json"]
    ]
    violations = [
        json.loads(str(record))
        for batch in batches
        for record in batch["violations_json"]
    ]
    metrics: list[dict[str, object]] = []
    for method, values in (("raw", raw), ("projected", projected)):
        metrics.extend(
            _prediction_metric_rows(
                method, values, holdout_responses, model.response_scale, layout
            )
        )
    if holdout_closure is not None:
        exact_overflow = overflow_tss_from_response(
            holdout_responses, holdout_controls, layout
        )
        development_exact_overflow = overflow_tss_from_response(
            development_responses, development_controls, layout
        )
        metrics.extend(
            _overflow_closure_metric_rows(
                holdout_closure, exact_overflow, development_exact_overflow
            )
        )
    correction = (projected - raw) / model.response_scale
    metrics.append(
        {
            "method": "projection_correction",
            "block": "complete_response",
            "coordinate": "ALL",
            "sample_count": len(raw),
            "coordinate_count": layout.state_size,
            "rmse": math.nan,
            "mae": math.nan,
            "bias": math.nan,
            "nrmse": float(np.sqrt(np.mean(np.square(correction)))),
            "nmae": float(np.mean(np.abs(correction))),
            "r2_mean": math.nan,
        }
    )
    return AssessmentResult(
        metrics=pd.DataFrame(metrics),
        violations=pd.DataFrame(violations),
        qp_diagnostics=pd.DataFrame(qp_rows),
        feasibility=pd.DataFrame(feasibility_rows),
        raw=raw,
        projected=projected,
        projected_targets=projected_targets,
        overflow_tss_closure=holdout_closure,
    )
