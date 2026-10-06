"""Read complete chart inputs without changing the scientific run."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from surrogate_optimization.config import DECISION_LOWER, DECISION_UPPER
from surrogate_optimization.optimization.mechanistic import DEFAULT_OBJECTIVE_WEIGHTS
from surrogate_optimization.plant.definitions import (
    COMPOSITE_MATRIX,
    NOMINAL_INFLUENT,
    TSS_VECTOR,
)

CASES = ("nominal", *(f"robustness_{i:02d}" for i in range(1, 11)))
ROUTES = ("surrogate", "mechanistic")
COMPOSITES = ("COD", "TN", "TP", "TSS")
CONTROL_NAMES = ("H", "a_3", "a_4", "a_5", "r_I", "r_R", "w")


class ChartDataError(ValueError):
    """Required data cannot support the complete eight-figure package."""


def boolean_values(values: pd.Series, *, description: str) -> pd.Series:
    converted = (
        values.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if converted.isna().any():
        raise ChartDataError(f"{description} contains invalid Boolean values")
    return converted.astype(bool)


def _npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as values:
            return {name: np.asarray(values[name]) for name in values.files}
    except (OSError, ValueError, EOFError) as exc:
        raise ChartDataError(f"cannot read chart input {path}: {exc}") from exc


def _array(
    values: dict[str, np.ndarray], names: tuple[str, ...], path: Path
) -> np.ndarray:
    for name in names:
        if name in values:
            result = np.asarray(values[name], dtype=float)
            if not np.isfinite(result).all():
                raise ChartDataError(f"{path}: {name} contains non-finite values")
            return result
    raise ChartDataError(f"{path}: missing {' or '.join(names)}")


def _response(value: np.ndarray, *, width: int, description: str) -> np.ndarray:
    if (
        value.ndim not in (1, 2)
        or value.shape[-1] not in {161, width}
        or not np.isfinite(value).all()
    ):
        raise ChartDataError(
            f"{description} must use the declared 161-coordinate or {width}-coordinate layout"
        )
    return value


def response_composites(response: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Convert outlet component flows before forming the eight location composites."""
    response, controls = np.asarray(response, float), np.asarray(controls, float)
    if (
        response.ndim != 2
        or response.shape[1] < 160
        or controls.shape != (len(response), 7)
    ):
        raise ChartDataError("response and control dimensions do not match")
    effluent_flow, underflow_flow = (
        1.0 - controls[:, 6],
        controls[:, 5] + controls[:, 6],
    )
    if np.any(effluent_flow <= 0) or np.any(underflow_flow <= 0):
        raise ChartDataError("outlet concentration conversion requires positive flows")
    locations = [
        response[:, :20],
        *[response[:, 20 * i : 20 * (i + 1)] for i in range(1, 6)],
        response[:, 120:140] / effluent_flow[:, None],
        response[:, 140:160] / underflow_flow[:, None],
    ]
    return np.stack([values @ COMPOSITE_MATRIX.T for values in locations], axis=1)


def _csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        table = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ChartDataError(f"cannot read {path}: {exc}") from exc
    for name in (
        "available",
        "candidate_available",
        "comparison_valid",
        "exact_replay_valid",
    ):
        if name in table:
            table[name] = boolean_values(table[name], description=f"{path}: {name}")
    for name in ("route", "decision_route"):
        if name in table:
            table[name] = table[name].replace({"direct": "mechanistic"})
    return table


@dataclass(frozen=True)
class ChartData:
    composites: dict[str, np.ndarray]
    quality: pd.DataFrame
    controls: pd.DataFrame
    exact: pd.DataFrame
    influent: pd.DataFrame
    profiles: dict[tuple[str, str], np.ndarray]
    timing: pd.DataFrame
    weights: np.ndarray
    sources: tuple[str, ...]


def load_chart_data(run: Path) -> ChartData:
    run = run.resolve()
    state_path = run / "run_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ChartDataError(f"cannot read target run state: {state_path}") from exc
    if not isinstance(state, dict):
        raise ChartDataError("target run state must be a JSON object")
    contract_path = run / "run_contract.json"
    if not contract_path.exists():
        contract_path = run / "inputs/contract.json"
    contract: dict[str, Any] = {}
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            raise ChartDataError("target contract must be a JSON object")
    profile = contract.get("profile", {})
    schema = contract.get("response_schema", {})
    if (
        profile.get("robustness_count", 10) != 10
        or schema.get("shared_coordinate_count", 160) != 160
    ):
        raise ChartDataError(
            "charts require ten scenarios, five reactors, and twenty components"
        )
    width = 160 + int(profile.get("layer_count", 10))
    sources: list[str] = ["run_state.json"]
    if contract_path.exists():
        sources.append(contract_path.relative_to(run).as_posix())
    design_path = run / "datasets/effective_design.npz"
    design = _npz(design_path)
    development_controls = _array(
        design, ("development_controls", "development_decisions"), design_path
    )
    holdout_controls = _array(
        design, ("holdout_controls", "test_decisions"), design_path
    )
    scenarios = _array(design, ("robustness_influents",), design_path)
    if (
        development_controls.ndim != 2
        or development_controls.shape[1] != 7
        or holdout_controls.ndim != 2
        or holdout_controls.shape[1] != 7
        or scenarios.shape != (10, 20)
    ):
        raise ChartDataError(
            "design must contain seven controls and all ten twenty-component influent scenarios"
        )
    prediction_path = run / "predictions/post_selection_holdout.npz"
    predictions = _npz(prediction_path)
    composites = {}
    for method in ("mechanistic", "raw", "projected"):
        response = _response(
            _array(predictions, (method,), prediction_path),
            width=width,
            description=method,
        )
        if (
            response.ndim != 2
            or len(response) != len(holdout_controls)
            or len(response) < 2
        ):
            raise ChartDataError(
                f"{method}: holdout responses must match the control rows"
            )
        composites[method] = response_composites(response, holdout_controls)
    if np.any(np.ptp(composites["mechanistic"], axis=0) <= 0):
        raise ChartDataError(
            "each holdout location/composite requires a positive mechanistic range"
        )
    if np.any(composites["mechanistic"] <= 0) or np.any(composites["projected"] <= 0):
        raise ChartDataError(
            "logarithmic parity requires positive mechanistic and projected composites"
        )
    development_path = run / "datasets/development/accepted_mechanistic_responses.npz"
    if not development_path.exists():
        development_path = run / "datasets/development/mechanistic_accepted_v3.npz"
    development = _npz(development_path)
    responses = _response(
        _array(development, ("mechanistic_responses", "targets"), development_path),
        width=width,
        description="development responses",
    )
    if (
        responses.ndim != 2
        or len(responses) != len(development_controls)
        or len(responses) < 2
    ):
        raise ChartDataError(
            "development controls and mechanistic responses must have matching rows"
        )
    quality_scale = np.std(
        response_composites(responses, development_controls)[:, 6, :], axis=0, ddof=0
    )
    if not np.isfinite(quality_scale).all() or np.any(quality_scale <= 0):
        raise ChartDataError(
            "development effluent composites require positive standard deviations"
        )
    sources.extend(
        path.relative_to(run).as_posix()
        for path in (design_path, prediction_path, development_path)
    )
    objective = contract.get("objective", {})
    weights = np.asarray(
        objective.get(
            "weights", contract.get("objective_weights", DEFAULT_OBJECTIVE_WEIGHTS)
        ),
        float,
    )
    bounds = contract.get("control_bounds", {})
    lower = np.asarray(
        bounds.get("lower", contract.get("decision_lower", DECISION_LOWER)), float
    )
    upper = np.asarray(
        bounds.get("upper", contract.get("decision_upper", DECISION_UPPER)), float
    )
    if (
        weights.shape != (6,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or lower.shape != (7,)
        or upper.shape != (7,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(upper <= lower)
    ):
        raise ChartDataError(
            "declared objective weights and control bounds are invalid"
        )
    underflow_reference = float(objective.get("underflow_tss_reference_g_m3", 15000.0))
    if (
        not np.isfinite(underflow_reference)
        or underflow_reference <= 0
        or upper[6] <= 0
        or upper[0] * np.sum(upper[1:4]) <= 0
    ):
        raise ChartDataError("objective reference scales must be positive")
    influent = pd.DataFrame(
        np.vstack((NOMINAL_INFLUENT, scenarios)) @ COMPOSITE_MATRIX.T,
        index=CASES,
        columns=COMPOSITES,
    )
    if np.any(influent.to_numpy() <= 0):
        raise ChartDataError(
            "removal calculation requires positive fresh-influent composites"
        )
    quality_table = _csv(run / "report/tables/selected_quality.csv")
    control_table = _csv(run / "report/tables/scenario_controls.csv")
    for table, required, name in (
        (
            quality_table,
            {"case", "decision_route", "response_method", "available", *COMPOSITES},
            "selected_quality.csv",
        ),
        (
            control_table,
            {"case", "route", "available", *CONTROL_NAMES},
            "scenario_controls.csv",
        ),
    ):
        if table is not None:
            if not required.issubset(table.columns):
                raise ChartDataError(f"{name} is missing required columns")
            sources.append(f"report/tables/{name}")
    records, quality_rows, control_rows, profiles = [], [], [], {}
    missing = []
    for case in CASES:
        for route in ROUTES:
            directory = run / "optimization" / case
            paths = [directory / f"{route}_casewise_reference.npz"]
            if route == "mechanistic":
                paths.append(directory / "direct_casewise_reference.npz")
            if (
                run.name == "article_full_10000_001"
                and case == "robustness_01"
                and route == "mechanistic"
            ):
                paths.append(
                    run
                    / "report/chart_overrides/robustness_01_direct_casewise_reference.npz"
                )
            candidates = [path for path in paths if path.is_file()]
            values = None
            for path in candidates:
                try:
                    status_path = path.with_suffix(".json")
                    if status_path.is_file():
                        status = json.loads(status_path.read_text(encoding="utf-8"))
                        for flag in ("candidate_available", "exact_replay_valid"):
                            if (
                                flag in status
                                and not boolean_values(
                                    pd.Series([status[flag]]),
                                    description=f"{status_path}: {flag}",
                                ).iloc[0]
                            ):
                                raise ChartDataError(
                                    f"{case}/{route}: audited exact response is unavailable"
                                )
                    stored = _npz(path)
                    controls = _array(stored, ("controls", "theta"), path)
                    exact = _response(
                        _array(stored, ("exact_reference",), path),
                        width=width,
                        description=f"{case}/{route} reference",
                    )
                    if controls.shape != (7,) or exact.ndim != 1:
                        raise ChartDataError(
                            f"{case}/{route}: invalid selected-decision dimensions"
                        )
                    projected = (
                        _response(
                            _array(stored, ("projected",), path),
                            width=width,
                            description=f"{case}/{route} projected",
                        )
                        if route == "surrogate"
                        else exact
                    )
                    if projected.ndim != 1:
                        raise ChartDataError(
                            f"{case}/{route}: invalid projected response dimensions"
                        )
                    values = (controls, exact, projected)
                    sources.append(path.relative_to(run).as_posix())
                    break
                except ChartDataError as exc:
                    reason = str(exc)
            if values is None:
                missing.append(
                    f"{case}/{route}: {reason if candidates else 'missing audited casewise response'}"
                )
                continue
            controls, exact, projected = values
            physical = response_composites(exact[None, :], controls[None, :])[0]
            projected_composites = response_composites(
                projected[None, :], controls[None, :]
            )[0]
            # Completed tables are preferred when their row agrees with the audited arrays.
            for method, calculated in (
                ("reference", physical[6]),
                ("projected", projected_composites[6]),
            ):
                if route != "surrogate" and method == "projected":
                    continue
                row = calculated
                if quality_table is not None:
                    selected = quality_table[
                        quality_table["case"].eq(case)
                        & quality_table["decision_route"].eq(route)
                        & quality_table["response_method"].eq(method)
                    ]
                    if len(selected) > 1:
                        raise ChartDataError(
                            f"duplicate selected-quality records for {case}/{route}/{method}"
                        )
                    if len(selected) == 1 and bool(selected["available"].iloc[0]):
                        supplied = selected[list(COMPOSITES)].to_numpy(float)[0]
                        if np.isfinite(supplied).all() and np.allclose(
                            supplied, calculated, rtol=1e-8, atol=1e-8
                        ):
                            row = supplied
                quality_rows.append(
                    {
                        "case": case,
                        "decision_route": route,
                        "response_method": method,
                        "available": True,
                        **dict(zip(COMPOSITES, row, strict=True)),
                    }
                )
            if control_table is not None:
                selected = control_table[
                    control_table["case"].eq(case) & control_table["route"].eq(route)
                ]
                if len(selected) > 1:
                    raise ChartDataError(
                        f"duplicate control records for {case}/{route}"
                    )
                if len(selected) == 1 and bool(selected["available"].iloc[0]):
                    supplied = selected[list(CONTROL_NAMES)].to_numpy(float)[0]
                    if np.isfinite(supplied).all() and np.allclose(
                        supplied, controls, rtol=1e-8, atol=1e-8
                    ):
                        controls = supplied
            control_rows.append(
                {
                    "case": case,
                    "route": route,
                    **dict(zip(CONTROL_NAMES, controls, strict=True)),
                }
            )
            parts = np.asarray(
                (
                    np.mean(physical[6] / quality_scale),
                    (controls[0] - lower[0]) / (upper[0] - lower[0]),
                    controls[0]
                    * np.sum(controls[1:4])
                    / (upper[0] * np.sum(upper[1:4])),
                    (controls[4] - lower[4]) / (upper[4] - lower[4]),
                    (controls[5] - lower[5]) / (upper[5] - lower[5]),
                    controls[6]
                    * float(TSS_VECTOR @ (exact[140:160] / (controls[5] + controls[6])))
                    / (upper[6] * underflow_reference),
                )
            )
            records.append(
                {
                    "case": case,
                    "route": route,
                    **dict(zip(COMPOSITES, physical[6], strict=True)),
                    **dict(
                        zip(
                            (
                                "quality",
                                "hrt",
                                "aeration",
                                "internal_recycle",
                                "return_sludge",
                                "wasting",
                            ),
                            parts,
                            strict=True,
                        )
                    ),
                    "objective": float(weights @ parts),
                    "economic": float(weights[1:] @ parts[1:]),
                }
            )
            liquid_profile = np.vstack(
                (influent.loc[case].to_numpy(float), physical[:7])
            )
            if np.any(liquid_profile <= 0):
                missing.append(
                    f"{case}/{route}: logarithmic treatment profiles require positive composites"
                )
            profiles[case, route] = liquid_profile
    if missing:
        raise ChartDataError(
            "cannot generate the complete eight-figure package:\n" + "\n".join(missing)
        )
    timing_path = run / "metrics/robustness_case_timing.csv"
    timing = _csv(timing_path)
    nominal_path = run / "report/tables/selected_candidate_reference_evaluation.csv"
    nominal = _csv(nominal_path)
    timing_rows = []
    for case in CASES:
        for route in ROUTES:
            table = nominal if case == "nominal" else timing
            selected = (
                table[table["case"].eq(case) & table["route"].eq(route)]
                if table is not None
                else pd.DataFrame()
            )
            if len(selected) == 1:
                metric_column = "time_metric" if case == "nominal" else "metric"
                if selected[metric_column].iloc[0] not in {"Time", "Optimization time"}:
                    raise ChartDataError(
                        f"{case}/{route}: unsupported optimization time metric"
                    )
                if (
                    selected["time_unit" if case == "nominal" else "unit"].iloc[0]
                    != "s"
                ):
                    raise ChartDataError(
                        f"{case}/{route}: optimization time must be recorded in seconds"
                    )
                duration = float(selected["time_seconds"].iloc[0])
            elif len(selected) > 1:
                raise ChartDataError(
                    f"{case}/{route}: duplicate optimization time records"
                )
            else:
                payload_path = run / "optimization" / case / f"{route}.json"
                if not payload_path.exists() and route == "mechanistic":
                    payload_path = payload_path.with_name("direct.json")
                if not payload_path.exists():
                    raise ChartDataError(f"{case}/{route}: missing optimization time")
                duration = float(
                    json.loads(payload_path.read_text())["elapsed_seconds"]
                )
                sources.append(payload_path.relative_to(run).as_posix())
            if not np.isfinite(duration) or duration <= 0:
                raise ChartDataError(
                    f"{case}/{route}: logarithmic optimization time requires a positive duration"
                )
            timing_rows.append({"case": case, "route": route, "time_seconds": duration})
    sources.extend(
        path.relative_to(run).as_posix()
        for path in (timing_path, nominal_path)
        if path.exists()
    )
    return ChartData(
        composites,
        pd.DataFrame(quality_rows),
        pd.DataFrame(control_rows).set_index(["case", "route"]),
        pd.DataFrame(records),
        influent,
        profiles,
        pd.DataFrame(timing_rows)
        .pivot(index="case", columns="route", values="time_seconds")
        .reindex(index=CASES, columns=ROUTES),
        weights,
        tuple(dict.fromkeys(sources)),
    )
