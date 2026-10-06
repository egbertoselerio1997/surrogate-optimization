"""Reporting timing."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from typing import Mapping
import numpy as np
import pandas as pd


def _robustness_timing_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {
            "count": 0,
            "total": None,
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
            "p95_nearest_rank": None,
            "maximum": None,
        }
    ordered = np.sort(finite)
    p95_index = max(0, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "count": int(len(ordered)),
        "total": float(np.sum(ordered)),
        "mean": float(np.mean(ordered)),
        "median": float(np.median(ordered)),
        "q25": float(np.quantile(ordered, 0.25)),
        "q75": float(np.quantile(ordered, 0.75)),
        "p95_nearest_rank": float(ordered[p95_index]),
        "maximum": float(ordered[-1]),
    }


def _run_robustness_case_timing_aggregation(
    run: Path, *, source_files: Mapping[str, str], analysis_id: str
) -> pd.DataFrame:
    """Summarize primary route Time over the ten robustness cases."""
    from surrogate_optimization.runtime.artifacts import atomic_dataframe
    from surrogate_optimization.runtime.artifacts import atomic_json
    from surrogate_optimization.runtime.contracts import _artifact_hashes
    from surrogate_optimization.runtime.contracts import _artifacts_match
    from surrogate_optimization.runtime.contracts import _canonical_json_digest
    from surrogate_optimization.runtime.contracts import _load_json_object
    from surrogate_optimization.runtime.contracts import file_digest
    from surrogate_optimization.runtime.contracts import source_digest
    from surrogate_optimization.runtime.protocols import TIMING_PROTOCOL

    source_id = source_digest(source_files)
    cases = tuple((f"robustness_{index:02d}" for index in range(1, 11)))
    input_paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    for case_id in cases:
        case_directory = run / "optimization" / case_id
        comparison_marker = case_directory / "casewise_comparison_complete.json"
        marker = _load_json_object(
            comparison_marker, description=f"{case_id} casewise comparison marker"
        )
        if marker.get("case") != case_id or not _artifacts_match(
            run, marker.get("artifacts", {})
        ):
            raise RuntimeError(f"completed casewise result changed: {case_id}")
        input_paths.append(comparison_marker)
        for route in ("surrogate", "mechanistic"):
            route_path = case_directory / f"{route}.json"
            reference_path = case_directory / f"{route}_casewise_reference.json"
            route_payload = _load_json_object(
                route_path, description=f"{case_id} {route} route timing"
            )
            reference_payload = _load_json_object(
                reference_path, description=f"{case_id} {route} reference timing"
            )
            input_paths.extend((route_path, reference_path))
            time_seconds = float(route_payload["elapsed_seconds"])
            if not np.isfinite(time_seconds) or time_seconds < 0.0:
                raise RuntimeError(f"{case_id} {route} has invalid Time")
            rows.append(
                {
                    "case": case_id,
                    "route": route,
                    "route_status": route_payload.get("status"),
                    "candidate_available": bool(
                        reference_payload.get("candidate_available")
                    ),
                    "comparison_valid": bool(reference_payload.get("comparison_valid")),
                    "metric": "Time",
                    "unit": "s",
                    "time_seconds": time_seconds,
                }
            )
    input_digests = {
        path.relative_to(run).as_posix(): file_digest(path)
        for path in sorted(set(input_paths))
    }
    contract = _canonical_json_digest(
        {
            "protocol": TIMING_PROTOCOL,
            "source_digest": source_id,
            "analysis_input_digest": analysis_id,
            "cases": cases,
            "inputs": input_digests,
        }
    )
    ledger_path = run / "metrics" / "robustness_case_timing.csv"
    summary_path = run / "metrics" / "robustness_case_timing_summary.json"
    marker_path = run / "metrics" / "robustness_case_timing_complete.json"
    if marker_path.is_file():
        marker = _load_json_object(marker_path, description="robustness timing marker")
        if marker.get("timing_contract") == contract and _artifacts_match(
            run, marker.get("artifacts", {})
        ):
            return pd.read_csv(ledger_path)
    ledger = pd.DataFrame(rows)
    routes = {
        str(route): _robustness_timing_summary(
            pd.to_numeric(group["time_seconds"], errors="coerce").tolist()
        )
        for route, group in ledger.groupby("route", sort=True)
    }
    atomic_dataframe(ledger_path, ledger)
    atomic_json(
        summary_path,
        {
            "timing_contract": contract,
            "protocol": TIMING_PROTOCOL,
            "metric": "Time",
            "unit": "s",
            "measurement": "primary route search only",
            "source": "completed robustness/sensitivity cases only",
            "nominal_case_included": False,
            "robustness_case_count": len(cases),
            "routes": routes,
        },
    )
    atomic_json(
        marker_path,
        {
            "stage": "robustness_case_timing_aggregation",
            "timing_contract": contract,
            "source_digest": source_id,
            "input_digest": analysis_id,
            "case_count": len(cases),
            "artifacts": _artifact_hashes(run, (ledger_path, summary_path)),
        },
    )
    return ledger
