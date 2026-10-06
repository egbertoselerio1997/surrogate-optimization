"""Nominal selected-decision parity with exact mechanistic replay."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from surrogate_optimization.reporting.figures.common import export_figure

FIGURES = {
    "nominal_surrogate_response_parity": ("png", "pdf"),
    "nominal_mechanistic_response_parity": ("png", "pdf"),
}


def generate_figures(run: Path, output: Path) -> dict[str, str]:
    status = pd.read_csv(
        run / "report/tables/selected_candidate_reference_evaluation.csv"
    )
    unavailable = {}
    with np.load(run / "models/ridge_surrogate.npz", allow_pickle=False) as model:
        scale = np.asarray(model["response_scale"])
    for route, stem in (
        ("surrogate", "nominal_surrogate_response_parity"),
        ("mechanistic", "nominal_mechanistic_response_parity"),
    ):
        row = status[status["case"].eq("nominal") & status["route"].eq(route)]
        if len(row) != 1:
            raise ValueError(
                "nominal reference evaluation must contain both route records"
            )
        if (
            not row["candidate_available"].astype(str).str.lower().eq("true").iloc[0]
            or not row["exact_replay_valid"].astype(str).str.lower().eq("true").iloc[0]
        ):
            unavailable[stem] = "The nominal route did not return an available decision"
            continue
        path = run / "optimization/nominal" / f"{route}_casewise_reference.npz"
        if not path.is_file():
            raise RuntimeError(
                f"available nominal decision has no reference artifact: {path}"
            )
        with np.load(path, allow_pickle=False) as data:
            if route == "surrogate":
                predicted, expected = (
                    np.asarray(data["raw"]),
                    np.asarray(data["exact_reference"]),
                )
                reduced_predicted, reduced_expected = (predicted, expected)
            else:
                predicted, expected = (
                    np.asarray(data["optimizer_native_full"]),
                    np.asarray(data["exact_reference_full"]),
                )
                reduced_predicted, reduced_expected = (
                    np.asarray(data["optimizer_native"]),
                    np.asarray(data["exact_reference"]),
                )
        if (
            reduced_predicted.shape != (161,)
            or reduced_expected.shape != (161,)
            or predicted.shape != expected.shape
        ):
            raise ValueError("nominal parity received incompatible response schemas")
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].scatter(expected, predicted, s=12, alpha=0.65)
        limits = (
            float(min(expected.min(), predicted.min())),
            float(max(expected.max(), predicted.max())),
        )
        axes[0].plot(limits, limits, "--", color="#343A40")
        axes[0].set(
            xlabel="Exact mechanistic replay", ylabel=f"{route.capitalize()} response"
        )
        blocks = [
            ("Mixer", slice(0, 20)),
            *[(f"Reactor {i}", slice(20 * i, 20 * (i + 1))) for i in range(1, 6)],
            ("Overflow", slice(120, 140)),
            ("Underflow", slice(140, 160)),
            ("Inventory", slice(160, 161)),
        ]
        errors = [
            np.sqrt(
                np.mean(
                    (
                        (reduced_predicted[index] - reduced_expected[index])
                        / scale[index]
                    )
                    ** 2
                )
            )
            for _, index in blocks
        ]
        axes[1].bar(np.arange(len(blocks)), errors)
        axes[1].set(
            xticks=np.arange(len(blocks)),
            xticklabels=[name for name, _ in blocks],
            ylabel="RMS error / development scale",
        )
        axes[1].tick_params(axis="x", rotation=45)
        figure.tight_layout()
        for extension in FIGURES[stem]:
            export_figure(
                figure, output / f"{stem}.{extension}", bbox_inches="tight", dpi=220
            )
        plt.close(figure)
    return unavailable


SIDECARS = ()
