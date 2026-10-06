"""Eight route-comparison figures from current predictions and report tables."""

from __future__ import annotations
from surrogate_optimization.runtime.artifacts import atomic_dataframe
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from surrogate_optimization.plant.definitions import COMPOSITE_MATRIX, NOMINAL_INFLUENT
from surrogate_optimization.reporting.figures.common import export_figure
from surrogate_optimization.optimization.mechanistic import DEFAULT_OBJECTIVE_WEIGHTS

GRID = "#D8DEE4"
ROUTES = ("surrogate", "mechanistic")
ROUTE_COLORS = {"surrogate": "#147D92", "mechanistic": "#7A5195"}
COMPOSITES = ("COD", "TN", "TP", "TSS")
LOCATIONS = (
    "mixer",
    "reactor_1",
    "reactor_2",
    "reactor_3",
    "reactor_4",
    "reactor_5",
    "overflow",
    "underflow",
)
FIGURES = {
    stem: ("png",)
    for stem in (
        "holdout_accuracy_overview",
        "holdout_accuracy_by_location",
        "holdout_parity_all_locations",
        "effluent_and_removal_parity",
        "effluent_and_operating_values",
        "treatment_train_profiles",
        "objective_quality_economic",
        "optimization_time",
    )
}


def save(figure: plt.Figure, output: Path, stem: str) -> None:
    figure.tight_layout()
    export_figure(
        figure, output / f"{stem}.png", bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def generate_figures(run: Path, output: Path) -> dict[str, str]:
    style()
    unavailable: dict[str, str] = {}
    with np.load(run / "datasets/effective_design.npz", allow_pickle=False) as design:
        holdout_controls = np.asarray(design["holdout_controls"])
        robustness = np.asarray(design["robustness_influents"])
    with np.load(
        run / "predictions/post_selection_holdout.npz", allow_pickle=False
    ) as data:
        composites = {
            name: response_composites(data[name], holdout_controls)
            for name in ("mechanistic", "raw", "projected")
        }
    truth = composites["mechanistic"]
    scores = {
        method: coordinate_normalized_score(
            truth.reshape(len(truth), -1), composites[method].reshape(len(truth), -1)
        )
        for method in ("raw", "projected")
    }
    figure, axes = plt.subplots(1, 3, figsize=(11, 4))
    for column, axis in enumerate(axes):
        axis.bar(
            ("Raw", "Projected"),
            [scores[m][column] for m in ("raw", "projected")],
            color=("#D97904", "#147D92"),
        )
        axis.set_ylabel(
            (
                "Coordinate-normalized RMSE",
                "Coordinate-normalized MAE",
                "Mean coordinate R²",
            )[column]
        )
    save(figure, output, "holdout_accuracy_overview")
    figure, axis = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(LOCATIONS))
    for offset, method, color in (
        (-0.18, "raw", "#D97904"),
        (0.18, "projected", "#147D92"),
    ):
        values = [
            coordinate_normalized_score(truth[:, i], composites[method][:, i])[0]
            for i in range(len(LOCATIONS))
        ]
        axis.bar(positions + offset, values, width=0.36, label=method, color=color)
    axis.set(
        xticks=positions, xticklabels=LOCATIONS, ylabel="Coordinate-normalized RMSE"
    )
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    save(figure, output, "holdout_accuracy_by_location")
    figure, axes = plt.subplots(4, 8, figsize=(24, 12))
    for quantity, component in enumerate(COMPOSITES):
        for location, name in enumerate(LOCATIONS):
            axis = axes[quantity, location]
            expected = truth[:, location, quantity]
            for method, color in (("raw", "#D97904"), ("projected", "#147D92")):
                axis.scatter(
                    expected,
                    composites[method][:, location, quantity],
                    s=5,
                    alpha=0.5,
                    color=color,
                )
            limits = (float(expected.min()), float(expected.max()))
            axis.plot(limits, limits, "--", color="#343A40")
            axis.set_title(f"{name}: {component}")
    save(figure, output, "holdout_parity_all_locations")
    tables = run / "report/tables"
    quality = pd.read_csv(tables / "selected_quality.csv")
    controls = pd.read_csv(tables / "selected_controls.csv")
    objectives = pd.read_csv(tables / "objective_decomposition.csv")
    profiles = pd.read_csv(tables / "process_profiles.csv")
    cases = ("nominal", *(f"robustness_{i:02d}" for i in range(1, 11)))
    influents = pd.DataFrame(
        np.vstack((NOMINAL_INFLUENT, robustness)) @ COMPOSITE_MATRIX.T,
        index=cases,
        columns=COMPOSITES,
    )
    available_quality = quality[quality["available"].astype(str).str.lower().eq("true")]
    parity = available_quality[
        available_quality["decision_route"].eq("surrogate")
        & available_quality["response_method"].isin(("projected", "reference"))
    ]
    pairs = parity.pivot(
        index="case", columns="response_method", values=list(COMPOSITES)
    )
    pairs = pairs.reindex(
        columns=pd.MultiIndex.from_product((COMPOSITES, ("projected", "reference")))
    ).dropna()
    if pairs.empty:
        unavailable["effluent_and_removal_parity"] = (
            "No surrogate decision has a valid exact-reference pair"
        )
    else:
        figure, axes = plt.subplots(2, 4, figsize=(16, 8))
        for column, component in enumerate(COMPOSITES):
            expected, predicted = (
                pairs[component, method].to_numpy()
                for method in ("reference", "projected")
            )
            feed = influents.loc[pairs.index, component].to_numpy()
            for row, x, y in (
                (0, expected, predicted),
                (1, 100 * (feed - expected) / feed, 100 * (feed - predicted) / feed),
            ):
                axis = axes[row, column]
                axis.scatter(x, y, color="#147D92")
                limits = (min(x.min(), y.min()), max(x.max(), y.max()))
                axis.plot(limits, limits, "--", color="#343A40")
                axis.set(
                    xlabel="Exact reference",
                    ylabel="Projected surrogate",
                    title=f"{component}: {('mg/L' if row == 0 else 'removal %')}",
                )
        save(figure, output, "effluent_and_removal_parity")
    reference = available_quality[available_quality["response_method"].eq("reference")]
    available_controls = controls[
        controls["available"].astype(str).str.lower().eq("true")
    ]
    if reference.empty and available_controls.empty:
        unavailable["effluent_and_operating_values"] = (
            "No selected decision is available"
        )
    else:
        figure, axes = plt.subplots(3, 4, figsize=(16, 11))
        for i, coordinate in enumerate(
            (*COMPOSITES, "H", "a_3", "a_4", "a_5", "r_I", "r_R", "w")
        ):
            axis = axes.flat[i]
            source = reference if coordinate in COMPOSITES else available_controls
            route_column = "decision_route" if coordinate in COMPOSITES else "route"
            for route in ROUTES:
                rows = (
                    source[source[route_column].eq(route)]
                    .set_index("case")
                    .reindex(cases)
                )
                axis.plot(
                    np.arange(len(cases)),
                    rows[coordinate],
                    "o-",
                    label=route,
                    color=ROUTE_COLORS[route],
                )
            axis.set_title(coordinate)
        axes.flat[-1].axis("off")
        axes.flat[0].legend()
        save(figure, output, "effluent_and_operating_values")
    reference_profiles = profiles[profiles["response_method"].eq("reference")]
    if not np.isfinite(reference_profiles["value"]).any():
        unavailable["treatment_train_profiles"] = (
            "No exact-reference treatment profile is available"
        )
    else:
        figure, axes = plt.subplots(2, 4, figsize=(17, 8))
        for row, route in enumerate(ROUTES):
            for column, quantity in enumerate(COMPOSITES):
                axis = axes[row, column]
                selected = reference_profiles[
                    reference_profiles["decision_route"].eq(route)
                    & reference_profiles["quantity"].eq(quantity)
                ]
                for case, values in selected.groupby("case", sort=False):
                    values = (
                        values.set_index("location")
                        .rename(
                            index={
                                "clarifier_overflow": "overflow",
                                "clarifier_underflow": "underflow",
                            }
                        )
                        .reindex(LOCATIONS)
                    )
                    axis.plot(
                        np.arange(len(LOCATIONS)),
                        values["value"],
                        label=case,
                        alpha=0.7,
                    )
                axis.set(
                    title=f"{route}: {quantity}",
                    xticks=np.arange(len(LOCATIONS)),
                    xticklabels=LOCATIONS,
                )
                axis.tick_params(axis="x", rotation=45)
        save(figure, output, "treatment_train_profiles")
    objective_rows = objectives[
        objectives["response_method"].eq("reference")
        & objectives["available"].astype(str).str.lower().eq("true")
    ]
    if objective_rows.empty:
        unavailable["objective_quality_economic"] = (
            "No validated exact-reference objective is available"
        )
    else:
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        for route in ROUTES:
            rows = (
                objective_rows[objective_rows["route"].eq(route)]
                .set_index("case")
                .reindex(cases)
            )
            economic = (
                rows[
                    ["hrt", "aeration", "internal_recycle", "return_sludge", "wasting"]
                ].to_numpy()
                @ DEFAULT_OBJECTIVE_WEIGHTS[1:]
            )
            for axis, values in zip(
                axes,
                (rows["recomputed_objective"], rows["quality"], economic),
                strict=True,
            ):
                axis.plot(
                    np.arange(len(cases)),
                    values,
                    "o-",
                    label=route,
                    color=ROUTE_COLORS[route],
                )
        for axis, title in zip(
            axes,
            ("Objective", "Quality component", "Weighted operating burden"),
            strict=True,
        ):
            axis.set_title(title)
        axes[0].legend()
        save(figure, output, "objective_quality_economic")
    timing = (
        pd.read_csv(tables / "timing_summary.csv").set_index("route").reindex(ROUTES)
    )
    if not np.isfinite(timing["mean"]).any():
        unavailable["optimization_time"] = "No primary search timing is available"
    else:
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.bar(ROUTES, timing["mean"], color=[ROUTE_COLORS[r] for r in ROUTES])
        axis.set_ylabel("Mean primary search Time (s)")
        save(figure, output, "optimization_time")
    atomic_dataframe(
        output / "figure_index.csv",
        pd.DataFrame(
            [
                {
                    "figure": stem,
                    "data_available": stem not in unavailable,
                    "reason": unavailable.get(stem),
                    "png": f"{stem}.png",
                }
                for stem in FIGURES
            ]
        ),
    )
    atomic_dataframe(
        output / "holdout_composite_metrics.csv",
        pd.DataFrame(
            [
                {
                    "method": method,
                    "nrmse": score[0],
                    "nmae": score[1],
                    "r2_mean": score[2],
                }
                for method, score in scores.items()
            ]
        ),
    )
    return unavailable


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def finite_score(
    truth: np.ndarray, prediction: np.ndarray
) -> tuple[float, float, float]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    scale = float(np.ptp(truth))
    if (
        truth.shape != prediction.shape
        or not np.all(np.isfinite(truth))
        or (not np.all(np.isfinite(prediction)))
    ):
        raise ValueError("prediction score received incompatible or non-finite arrays")
    if scale <= 0.0:
        raise ValueError("prediction score requires a nonzero truth range")
    error = prediction - truth
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 0.0 else np.nan
    return (
        float(np.sqrt(np.mean(error**2)) / scale),
        float(np.mean(np.abs(error)) / scale),
        r2,
    )


def coordinate_normalized_score(
    truth: np.ndarray, prediction: np.ndarray
) -> tuple[float, float, float]:
    """Score sample-by-coordinate data with every coordinate weighted equally."""
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if (
        truth.ndim != 2
        or truth.shape != prediction.shape
        or (not np.all(np.isfinite(truth)))
        or (not np.all(np.isfinite(prediction)))
    ):
        raise ValueError("coordinate score received incompatible or non-finite arrays")
    scales = np.ptp(truth, axis=0)
    if np.any(scales <= 0.0):
        raise ValueError(
            "coordinate score requires a nonzero truth range per coordinate"
        )
    normalized_error = (prediction - truth) / scales[None, :]
    coordinate_r2 = [
        finite_score(truth[:, index], prediction[:, index])[2]
        for index in range(truth.shape[1])
    ]
    return (
        float(np.sqrt(np.mean(normalized_error**2))),
        float(np.mean(np.abs(normalized_error))),
        float(np.nanmean(coordinate_r2)),
    )


def response_composites(response: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Return samples x locations x COD/TN/TP/TSS concentrations."""
    from surrogate_optimization.plant.definitions import COMPOSITE_MATRIX

    values = np.asarray(response, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if values.ndim != 2 or values.shape[1] < 160 or controls.shape != (len(values), 7):
        raise ValueError("unexpected response or decision shape")
    blocks = [values[:, 0:20], *[values[:, 20 * i : 20 * (i + 1)] for i in range(1, 6)]]
    overflow = values[:, 120:140] / (1.0 - controls[:, 6])[:, None]
    underflow = values[:, 140:160] / (controls[:, 5] + controls[:, 6])[:, None]
    blocks.extend((overflow, underflow))
    return np.stack([block @ COMPOSITE_MATRIX.T for block in blocks], axis=1)


SIDECARS = ("figure_index.csv", "holdout_composite_metrics.csv")
