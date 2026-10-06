"""Runtime checkpoints."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.optimization.mechanistic import MechanisticRouteResult
    from surrogate_optimization.optimization.types import SurrogateRouteResult
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any
from typing import Mapping
import json
import numpy as np


def _route_contract_id(
    *,
    source_id: str,
    analysis_id: str,
    case_id: str,
    influent: np.ndarray,
    route: str,
    protocol: str,
    settings: Any,
    starts: np.ndarray,
) -> str:
    digest = sha256()
    digest.update(source_id.encode())
    digest.update(analysis_id.encode())
    digest.update(case_id.encode())
    digest.update(route.encode())
    digest.update(protocol.encode())
    digest.update(json.dumps(asdict(settings), sort_keys=True).encode())
    digest.update(np.ascontiguousarray(influent, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(starts, dtype="<f8").tobytes())
    return digest.hexdigest()


def _start_controls(result: Any, route: str) -> np.ndarray:
    del route
    return np.asarray(result.initial_normalized_controls, dtype=float)


def _validate_route_result_integrity(result: Any, route: str) -> None:
    """Reject non-finite computational results while permitting clean failures."""
    controls = _start_controls(result, route)
    if controls.shape != (7,) or not np.all(np.isfinite(controls)):
        raise RuntimeError(f"{route} attempt returned invalid initial controls")
    if route == "surrogate":
        final = result.final
        if final is None:
            return
        arrays = (
            final.normalized_controls,
            final.controls,
            final.raw,
            final.projected,
            final.displacement,
            final.objective_components,
            final.engineering_rows,
            final.engineering_quantities,
            final.trust_rows,
            final.trust_values,
        )
        scalars = (final.objective,)
    elif route == "mechanistic":
        if not bool(result.feasible):
            return
        arrays = (
            result.normalized_controls,
            result.controls,
            result.state,
            result.response,
            result.engineering,
            result.objective_components,
        )
        scalars = (result.objective, result.feed_tss)
    else:
        raise ValueError(f"unknown optimization route {route!r}")
    if any(
        (not np.all(np.isfinite(np.asarray(value, dtype=float))) for value in arrays)
    ):
        raise RuntimeError(f"{route} attempt returned a non-finite candidate")
    if not np.all(np.isfinite(np.asarray(scalars, dtype=float))):
        raise RuntimeError(f"{route} attempt returned a non-finite objective/state")


def _read_completed_starts(
    case_directory: Path,
    *,
    route: str,
    route_contract: str,
    route_protocol: str,
    starts: np.ndarray,
) -> tuple[dict[int, Any], float]:
    from surrogate_optimization.optimization.mechanistic import MechanisticStartResult
    from surrogate_optimization.optimization.types import SurrogateStartResult

    result_type = (
        SurrogateStartResult if route == "surrogate" else MechanisticStartResult
    )
    completed: dict[int, Any] = {}
    elapsed = 0.0
    for index in range(len(starts)):
        path = case_directory / "checkpoints" / f"{route}_start_{index:02d}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("route_contract") != route_contract:
                continue
            if (
                payload.get("protocol") != route_protocol
                or int(payload.get("start_index", -1)) != index
                or (
                    not np.array_equal(
                        np.asarray(payload.get("normalized_start"), dtype=float),
                        starts[index],
                    )
                )
            ):
                raise RuntimeError(
                    f"current {route} attempt checkpoint is inconsistent"
                )
            result = result_type.from_dict(payload["result"])
            if result.start_index != index or not np.array_equal(
                _start_controls(result, route), starts[index]
            ):
                raise RuntimeError(f"current {route} attempt result is inconsistent")
            _validate_route_result_integrity(result, route)
            completed[index] = result
            elapsed = max(
                elapsed, float(payload.get("cumulative_route_wall_seconds", 0.0))
            )
        except RuntimeError:
            raise
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"current {route} attempt checkpoint is corrupt: {path.name}"
            ) from exc
    return (completed, elapsed)


def _publish_partial_starts(
    case_directory: Path, route: str, completed: Mapping[int, Any]
) -> None:
    from surrogate_optimization.runtime.artifacts import atomic_json

    atomic_json(
        case_directory / f"{route}_starts.partial.json",
        [completed[index].as_dict() for index in sorted(completed)],
        nonfinite_to_none=True,
    )


def _write_start_checkpoint(
    case_directory: Path,
    *,
    route: str,
    route_contract: str,
    route_protocol: str,
    starts: np.ndarray,
    result: Any,
    cumulative_elapsed: float,
) -> None:
    from surrogate_optimization.runtime.artifacts import atomic_json

    index = int(result.start_index)
    if not 0 <= index < len(starts) or not np.array_equal(
        _start_controls(result, route), starts[index]
    ):
        raise RuntimeError(f"{route} solver returned a mismatched start index/control")
    _validate_route_result_integrity(result, route)
    atomic_json(
        case_directory / "checkpoints" / f"{route}_start_{index:02d}.json",
        {
            "schema": 1,
            "route": route,
            "route_contract": route_contract,
            "protocol": route_protocol,
            "start_index": index,
            "normalized_start": starts[index],
            "cumulative_route_wall_seconds": cumulative_elapsed,
            "result": result.as_dict(),
        },
        nonfinite_to_none=True,
    )


def _load_complete_route(
    case_directory: Path,
    *,
    route: str,
    route_contract: str,
    route_protocol: str,
    starts: np.ndarray,
) -> Any | None:
    from surrogate_optimization.optimization.mechanistic import MechanisticRouteResult
    from surrogate_optimization.optimization.mechanistic import MechanisticStartResult
    from surrogate_optimization.optimization.types import SurrogateRouteResult
    from surrogate_optimization.optimization.types import SurrogateStartResult
    from surrogate_optimization.runtime.contracts import _artifacts_match

    marker_path = case_directory / f"{route}_complete.json"
    payload_path = case_directory / f"{route}.json"
    if not marker_path.is_file() or not payload_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("route_contract") != route_contract:
            return None
        if marker.get("protocol") != route_protocol:
            raise RuntimeError(f"current {route} completion marker has wrong protocol")
        if not _artifacts_match(case_directory, marker.get("artifacts", {})):
            raise RuntimeError(f"current {route} completed artifacts changed")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if (
            payload.get("route_contract") != route_contract
            or payload.get("protocol") != route_protocol
        ):
            raise RuntimeError(f"current {route} result payload is inconsistent")
        result_type = (
            SurrogateStartResult if route == "surrogate" else MechanisticStartResult
        )
        restored = tuple((result_type.from_dict(item) for item in payload["starts"]))
        if len(restored) != len(starts) or [
            item.start_index for item in restored
        ] != list(range(len(starts))):
            raise RuntimeError(f"current {route} result has wrong attempt count")
        if any(
            (
                not np.array_equal(_start_controls(item, route), starts[index])
                for index, item in enumerate(restored)
            )
        ):
            raise RuntimeError(f"current {route} result has wrong initial controls")
        for item in restored:
            _validate_route_result_integrity(item, route)
        selected_index = payload.get("selected_start")
        selected = None if selected_index is None else restored[int(selected_index)]
        result_class = (
            SurrogateRouteResult if route == "surrogate" else MechanisticRouteResult
        )
        if route == "surrogate":
            result = result_class(
                restored, selected, str(payload["status"]), protocol=route_protocol
            )
        else:
            result = result_class(restored, selected, str(payload["status"]))
        return (result, payload)
    except RuntimeError:
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"current {route} completion checkpoint is corrupt") from exc


def _publish_complete_route(
    case_directory: Path,
    *,
    route: str,
    route_contract: str,
    route_protocol: str,
    result: Any,
    elapsed_seconds: float,
) -> dict[str, Any]:
    from surrogate_optimization.runtime.artifacts import atomic_json
    from surrogate_optimization.runtime.contracts import _artifact_hashes

    start_count = len(result.starts)
    if start_count != 1:
        raise RuntimeError(f"{route} protocol requires exactly one local attempt")
    for item in result.starts:
        _validate_route_result_integrity(item, route)
    payload = result.as_dict()
    payload.update(
        {
            "route": route,
            "route_contract": route_contract,
            "protocol": route_protocol,
            "optimization_attempt_count": start_count,
            "primary_start_count": start_count,
            "maximum_wall_time": None,
            "elapsed_seconds": elapsed_seconds,
        }
    )
    payload_path = case_directory / f"{route}.json"
    atomic_json(payload_path, payload, nonfinite_to_none=True)
    start_paths = tuple(
        (
            case_directory / "checkpoints" / f"{route}_start_{index:02d}.json"
            for index in range(start_count)
        )
    )
    if not all((path.is_file() for path in start_paths)):
        raise RuntimeError(f"{route} is missing its atomic attempt checkpoint")
    paths = (payload_path, *start_paths)
    atomic_json(
        case_directory / f"{route}_complete.json",
        {
            "stage": f"{route}_single_local_attempt",
            "route_contract": route_contract,
            "protocol": route_protocol,
            "start_count": start_count,
            "optimization_attempt_count": start_count,
            "selected_start": payload["selected_start"],
            "status": payload["status"],
            "artifacts": _artifact_hashes(case_directory, paths),
        },
    )
    return payload


def _run_surrogate_route(
    case_directory: Path,
    *,
    case_id: str,
    influent: np.ndarray,
    assets: Any,
    source_id: str,
    analysis_id: str,
    problem: Any | None = None,
) -> tuple[SurrogateRouteResult, dict[str, Any]]:
    from surrogate_optimization.optimization.surrogate import EXACT_QP_CENTER_START
    from surrogate_optimization.optimization.surrogate import (
        EXACT_QP_SINGLE_START_PROTOCOL,
    )
    from surrogate_optimization.optimization.surrogate import (
        solve_surrogate_exact_qp_local,
    )
    from surrogate_optimization.optimization.types import SurrogateCase
    from surrogate_optimization.optimization.types import SurrogateSolverSettings
    from surrogate_optimization.optimization.types import SurrogateStartResult

    settings = SurrogateSolverSettings(maximum_wall_time=None)
    starts = np.asarray(EXACT_QP_CENTER_START, dtype=float).reshape(1, 7)
    contract = _route_contract_id(
        source_id=source_id,
        analysis_id=analysis_id,
        case_id=case_id,
        influent=influent,
        route="surrogate",
        protocol=EXACT_QP_SINGLE_START_PROTOCOL,
        settings=settings,
        starts=starts,
    )
    restored = _load_complete_route(
        case_directory,
        route="surrogate",
        route_contract=contract,
        route_protocol=EXACT_QP_SINGLE_START_PROTOCOL,
        starts=starts,
    )
    if restored is not None:
        return restored
    completed, prior_elapsed = _read_completed_starts(
        case_directory,
        route="surrogate",
        route_contract=contract,
        route_protocol=EXACT_QP_SINGLE_START_PROTOCOL,
        starts=starts,
    )
    _publish_partial_starts(case_directory, "surrogate", completed)
    started = perf_counter()

    def checkpoint(result: SurrogateStartResult) -> None:
        completed[result.start_index] = result
        _write_start_checkpoint(
            case_directory,
            route="surrogate",
            route_contract=contract,
            route_protocol=EXACT_QP_SINGLE_START_PROTOCOL,
            starts=starts,
            result=result,
            cumulative_elapsed=prior_elapsed + perf_counter() - started,
        )
        _publish_partial_starts(case_directory, "surrogate", completed)

    result = solve_surrogate_exact_qp_local(
        assets,
        SurrogateCase(influent=influent, case_id=case_id),
        settings=settings,
        problem=problem,
        name="study_surrogate_exact_qp",
        progress_callback=checkpoint,
        completed_result=completed.get(0),
    )
    if (
        len(result.starts) != 1
        or result.protocol != EXACT_QP_SINGLE_START_PROTOCOL
        or result.starts[0].stages
    ):
        raise RuntimeError(
            "surrogate route violated the single exact-QP attempt contract"
        )
    if 0 not in completed:
        raise RuntimeError("surrogate exact-QP attempt was not checkpointed")
    elapsed = prior_elapsed + perf_counter() - started
    payload = _publish_complete_route(
        case_directory,
        route="surrogate",
        route_contract=contract,
        route_protocol=EXACT_QP_SINGLE_START_PROTOCOL,
        result=result,
        elapsed_seconds=elapsed,
    )
    return (result, payload)


def _run_mechanistic_route(
    case_directory: Path,
    *,
    case_id: str,
    influent: np.ndarray,
    assets: Any,
    development_controls: np.ndarray,
    development_influents: np.ndarray,
    development_responses: np.ndarray,
    source_id: str,
    analysis_id: str,
) -> tuple[MechanisticRouteResult, dict[str, Any]]:
    from surrogate_optimization.optimization.mechanistic import MechanisticCase
    from surrogate_optimization.optimization.mechanistic import MechanisticStartResult
    from surrogate_optimization.optimization.mechanistic import SolverSettings
    from surrogate_optimization.optimization.mechanistic import (
        ordered_normalized_starts as mechanistic_center_start,
    )
    from surrogate_optimization.optimization.mechanistic import solve_mechanistic_case
    from surrogate_optimization.runtime.protocols import (
        MECHANISTIC_SINGLE_CENTER_PROTOCOL,
    )

    settings = SolverSettings(maximum_wall_time=None)
    starts = np.asarray(mechanistic_center_start()[0], dtype=float).reshape(1, 7)
    if not np.array_equal(starts[0], np.full(7, 0.5)):
        raise RuntimeError("direct route center start is not deterministic")
    contract = _route_contract_id(
        source_id=source_id,
        analysis_id=analysis_id,
        case_id=case_id,
        influent=influent,
        route="mechanistic",
        protocol=MECHANISTIC_SINGLE_CENTER_PROTOCOL,
        settings=settings,
        starts=starts,
    )
    restored = _load_complete_route(
        case_directory,
        route="mechanistic",
        route_contract=contract,
        route_protocol=MECHANISTIC_SINGLE_CENTER_PROTOCOL,
        starts=starts,
    )
    if restored is not None:
        return restored
    completed, prior_elapsed = _read_completed_starts(
        case_directory,
        route="mechanistic",
        route_contract=contract,
        route_protocol=MECHANISTIC_SINGLE_CENTER_PROTOCOL,
        starts=starts,
    )
    _publish_partial_starts(case_directory, "mechanistic", completed)
    started = perf_counter()

    def checkpoint(result: MechanisticStartResult) -> None:
        completed[result.start_index] = result
        _write_start_checkpoint(
            case_directory,
            route="mechanistic",
            route_contract=contract,
            route_protocol=MECHANISTIC_SINGLE_CENTER_PROTOCOL,
            starts=starts,
            result=result,
            cumulative_elapsed=prior_elapsed + perf_counter() - started,
        )
        _publish_partial_starts(case_directory, "mechanistic", completed)

    result = solve_mechanistic_case(
        assets,
        MechanisticCase(influent=influent, case_id=case_id),
        development_controls,
        development_influents,
        development_responses,
        settings=settings,
        starts=starts,
        completed_starts=completed,
        progress_callback=checkpoint,
    )
    if len(result.starts) != 1:
        raise RuntimeError("direct route violated the single-attempt contract")
    elapsed = prior_elapsed + perf_counter() - started
    payload = _publish_complete_route(
        case_directory,
        route="mechanistic",
        route_contract=contract,
        route_protocol=MECHANISTIC_SINGLE_CENTER_PROTOCOL,
        result=result,
        elapsed_seconds=elapsed,
    )
    return (result, payload)


def _case_contract_id(
    source_id: str, analysis_id: str, case_id: str, influent: np.ndarray
) -> str:
    from surrogate_optimization.runtime.protocols import OPTIMIZATION_PROTOCOL

    return sha256(
        source_id.encode()
        + analysis_id.encode()
        + OPTIMIZATION_PROTOCOL.encode()
        + case_id.encode()
        + np.ascontiguousarray(influent, dtype="<f8").tobytes()
    ).hexdigest()
