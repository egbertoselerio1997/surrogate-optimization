"""Synthetic complete chart inputs; no mechanistic solve is claimed for fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd

from surrogate_optimization.reporting.figures.data import CASES, ROUTES
from surrogate_optimization.runtime.artifacts import (
    atomic_json,
    atomic_npz,
    atomic_dataframe,
)


def complete_chart_run(run: Path) -> None:
    def controls(count):
        values = np.tile([20.0, 0.2, 0.5, 0.7, 1.2, 0.5, 0.02], (count, 1))
        values[:, 0] += np.arange(count) / count
        return values

    def responses(values):
        count = len(values)
        concentration = 2.0 + np.arange(20)[None, :] + np.arange(count)[:, None] * 0.6
        blocks = [concentration * (1 + 0.1 * index) for index in range(6)]
        overflow = concentration * 0.5 * (1 - values[:, 6, None])
        underflow = concentration * 2 * (values[:, 5, None] + values[:, 6, None])
        full = np.column_stack(
            (*blocks, overflow, underflow, np.full((count, 10), 100.0))
        )
        reduced = np.column_stack((full[:, :160], np.full(count, 600000.0)))
        return full, reduced

    development, holdout = controls(16), controls(12)
    full, _ = responses(development)
    _, truth = responses(holdout)
    atomic_json(run / "run_state.json", {"stage": "complete", "status": "complete"})
    atomic_json(
        run / "run_contract.json",
        {
            "profile": {"layer_count": 10, "robustness_count": 10},
            "response_schema": {"shared_coordinate_count": 160},
        },
    )
    atomic_npz(
        run / "datasets/effective_design.npz",
        development_controls=development,
        holdout_controls=holdout,
        robustness_influents=np.tile(np.arange(20) + 20.0, (10, 1)),
    )
    atomic_npz(
        run / "datasets/development/accepted_mechanistic_responses.npz",
        mechanistic_responses=full,
    )
    atomic_npz(
        run / "predictions/post_selection_holdout.npz",
        mechanistic=truth,
        raw=truth * 1.12,
        projected=truth * 1.03,
    )
    durations = []
    for index, case in enumerate(CASES):
        for route_index, route in enumerate(ROUTES):
            point = controls(1)[0].copy()
            point[0] += index * 0.1 + route_index * 0.05
            point[1] = 0.0
            point[5] = 0.25
            _, exact = responses(point[None, :])
            exact *= 1 + index * 0.03 + route_index * 0.01
            directory = run / "optimization" / case
            atomic_npz(
                directory / f"{route}_casewise_reference.npz",
                controls=point,
                exact_reference=exact[0],
                projected=exact[0] * 1.02,
            )
            atomic_json(
                directory / f"{route}_casewise_reference.json",
                {"candidate_available": True, "exact_replay_valid": True},
            )
            durations.append(
                {
                    "case": case,
                    "route": route,
                    "time_seconds": float(10 + index + route_index),
                    "metric": "Optimization time",
                    "unit": "s",
                    "time_metric": "Optimization time",
                    "time_unit": "s",
                }
            )
    frame = pd.DataFrame(durations)
    atomic_dataframe(
        run / "metrics/robustness_case_timing.csv", frame[frame["case"].ne("nominal")]
    )
    atomic_dataframe(
        run / "report/tables/selected_candidate_reference_evaluation.csv", frame
    )
