"""The eight untitled PNG layouts specified by generate-reference-result-charts."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from surrogate_optimization.reporting.labels import (
    ROUTE_LABELS,
    LOCATION_LABELS as LOCATION_NAMES,
    CONTROL_LABELS,
    scenario_label,
    presentation_table,
    chart_metric_label,
)
from surrogate_optimization.reporting.figures.common import export_figure
from surrogate_optimization.reporting.figures.data import (
    CASES,
    ROUTES,
    COMPOSITES,
    ChartData,
)

EXTENDED = "#147D92"
MECHANISTIC = "#7A5195"
RAW = "#D97904"
REFERENCE = "#343A40"
GRID = "#D8DEE4"
DIRECT = MECHANISTIC
ROUTE_LABEL = ROUTE_LABELS
ROUTE_COLOR = {"surrogate": EXTENDED, "mechanistic": MECHANISTIC}
ROUTE_MARKER = {"surrogate": "o", "mechanistic": "s"}
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
LOCATION_LABELS = tuple(LOCATION_NAMES[location] for location in LOCATIONS)
FIGURE_INDEX = (
    (1, "Q1/Q2/Q3", "Holdout accuracy overview", "q01_holdout_accuracy_overview"),
    (2, "Q4", "Holdout accuracy by location", "q02_holdout_accuracy_by_location"),
    (3, "Q18", "Holdout parity across locations", "q03_holdout_parity_all_locations"),
    (
        4,
        "Q5/Q5R",
        "Selected-decision effluent and removal parity",
        "q04_effluent_and_removal_parity",
    ),
    (
        5,
        "Q9/Q11",
        "Exact effluent and operating values",
        "q05_effluent_and_operating_values",
    ),
    (6, "Q14/Q15/Q16/Q17", "Treatment-train profiles", "q06_treatment_train_profiles"),
    (
        7,
        "Q7/Q8/Q10",
        "Objective, quality, and economic trade-offs",
        "q07_objective_quality_economic",
    ),
    (8, "Q12", "Optimization time", "q08_optimization_time"),
)
FIGURES = {stem: ("png",) for _, _, _, stem in FIGURE_INDEX}
SIDECARS = (
    "README.md",
    "chart_index.csv",
    "chart_summary.csv",
    "holdout_composite_metrics.csv",
)


def save(fig, output, stem):
    if stem not in FIGURES:
        raise ValueError(f"unregistered figure: {stem}")
    if fig._suptitle is not None:
        fig._suptitle.remove()
    for axis in fig.axes:
        axis.set_title("")
    export_figure(fig, output / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_control_axis(axis, values):
    axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    if (
        np.min(values) >= -1e-10
        and np.max(values) <= 1 + 1e-10
        and np.ptp(values) <= 1e-6
    ):
        axis.set_ylim(0.0, 1.0)


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


def _plot_effluent_parity(
    quality: pd.DataFrame,
    influent: pd.DataFrame,
    *,
    cases: list[str],
    case_labels: dict[str, str],
    output: Path,
) -> None:
    """Combine Q5 concentration and removal parity with distinct dot colors."""
    route = "surrogate"
    method = "projected"
    subset = quality[
        quality["decision_route"].eq(route)
        & quality["response_method"].isin((method, "reference"))
        & quality["available"].astype(str).str.lower().eq("true")
    ]
    pivot = (
        subset.pivot(index="case", columns="response_method", values=list(COMPOSITES))
        .reindex(cases)
        .dropna()
    )
    if pivot.empty:
        raise RuntimeError("no complete Extended-ICSOR Q5 quality pairs")
    concentration_color = "#147D92"
    removal_color = "#D97904"
    fig, axes = plt.subplots(2, len(COMPOSITES), figsize=(19.5, 9.2), squeeze=False)
    for column, component in enumerate(COMPOSITES):
        reference = pivot[component, "reference"].to_numpy(float)
        predicted = pivot[component, method].to_numpy(float)
        low, high = (
            min(reference.min(), predicted.min()),
            max(reference.max(), predicted.max()),
        )
        pad = max(1e-09, 0.06 * (high - low))
        limits = (low - pad, high + pad)
        axis = axes[0, column]
        axis.plot(limits, limits, color=REFERENCE, lw=1.2, ls="--")
        axis.scatter(
            reference,
            predicted,
            s=42,
            color=concentration_color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for case, xx, yy in zip(pivot.index, reference, predicted, strict=True):
            axis.annotate(
                case_labels[case],
                (xx, yy),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=6,
            )
        axis.set(
            xlim=limits,
            ylim=limits,
            xlabel="Mechanistic (mg/L)",
            ylabel=f"{component} - Extended ICSOR prediction (mg/L)",
        )
        axis.set_aspect("equal", adjustable="box")
        feed = influent.loc[pivot.index, component].to_numpy(float)
        removal_reference = 100.0 * (feed - reference) / feed
        removal_predicted = 100.0 * (feed - predicted) / feed
        low, high = (
            min(removal_reference.min(), removal_predicted.min()),
            max(removal_reference.max(), removal_predicted.max()),
        )
        pad = max(0.2, 0.06 * (high - low))
        limits = (low - pad, high + pad)
        axis = axes[1, column]
        axis.plot(limits, limits, color=REFERENCE, lw=1.2, ls="--")
        axis.scatter(
            removal_reference,
            removal_predicted,
            s=42,
            color=removal_color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for case, xx, yy in zip(
            pivot.index, removal_reference, removal_predicted, strict=True
        ):
            axis.annotate(
                case_labels[case],
                (xx, yy),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=6,
            )
        axis.set(
            xlim=limits,
            ylim=limits,
            xlabel="Mechanistic removal (%)",
            ylabel=f"{component} - Extended ICSOR removal (%)",
        )
        axis.set_aspect("equal", adjustable="box")
    fig.legend(
        handles=[
            Line2D((0,), (0,), color=REFERENCE, lw=1.2, ls="--", label="Perfect match"),
            Line2D(
                (0,),
                (0,),
                marker="o",
                color="none",
                markerfacecolor=concentration_color,
                markeredgecolor="white",
                markersize=7,
                label="Effluent concentration parity",
            ),
            Line2D(
                (0,),
                (0,),
                marker="o",
                color="none",
                markerfacecolor=removal_color,
                markeredgecolor="white",
                markersize=7,
                label="Removal parity",
            ),
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, output, "q04_effluent_and_removal_parity")


def holdout_statistics(data):
    predictions = data.composites
    overall = pd.DataFrame(
        [
            dict(
                zip(
                    ("nrmse", "nmae", "r2_mean"),
                    coordinate_normalized_score(
                        predictions["mechanistic"].reshape(
                            len(predictions["mechanistic"]), -1
                        ),
                        predictions[method].reshape(len(predictions[method]), -1),
                    ),
                ),
                method=method,
            )
            for method in ("raw", "projected")
        ]
    ).set_index("method")
    locations = {
        method: np.asarray(
            [
                coordinate_normalized_score(
                    predictions["mechanistic"][:, index, :],
                    predictions[method][:, index, :],
                )
                for index in range(8)
            ]
        )
        for method in ("raw", "projected")
    }
    components = {
        method: [
            coordinate_normalized_score(
                predictions["mechanistic"][:, :, index],
                predictions[method][:, :, index],
            )
            for index in range(4)
        ]
        for method in ("raw", "projected")
    }
    return overall, locations, components


def _plot_accuracy(data, output, summary):
    overall, locations, component_scores = holdout_statistics(data)
    location_nrmse = {method: values[:, 0] for method, values in locations.items()}
    location_nmae = {method: values[:, 1] for method, values in locations.items()}
    location_r2 = {method: values[:, 2] for method, values in locations.items()}
    x = np.arange(8)
    width = 0.36
    fig = plt.figure(figsize=(15.5, 11.2))
    grid = fig.add_gridspec(
        3, 3, height_ratios=(1.0, 1.2, 1.0), hspace=0.45, wspace=0.36
    )
    aggregate_axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    for axis, (column, label, lower) in zip(
        aggregate_axes,
        (
            ("nrmse", "Coordinate-normalized RMSE (nRMSE)", True),
            ("nmae", "Coordinate-normalized MAE (nMAE)", True),
            ("r2_mean", "Mean location R²", False),
        ),
        strict=True,
    ):
        values = overall.loc[["raw", "projected"], column].to_numpy(float)
        bars = axis.bar(("Raw", "Projected"), values, color=(RAW, EXTENDED), width=0.62)
        for bar in bars:
            axis.annotate(
                f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        axis.set(
            xlabel="Prediction output",
            ylabel=f"{label} ({('lower' if lower else 'higher')} is better)",
        )
    location_axes = [fig.add_subplot(grid[1, column]) for column in range(3)]
    for axis, raw_values, projected_values, ylabel in (
        (
            location_axes[0],
            location_nrmse["raw"],
            location_nrmse["projected"],
            "Coordinate-normalized RMSE (nRMSE)",
        ),
        (
            location_axes[1],
            location_nmae["raw"],
            location_nmae["projected"],
            "Coordinate-normalized MAE (nMAE)",
        ),
        (
            location_axes[2],
            location_r2["raw"],
            location_r2["projected"],
            "Mean location R²",
        ),
    ):
        axis.bar(x - width / 2, raw_values, width, color=RAW, label="Raw")
        axis.bar(
            x + width / 2, projected_values, width, color=EXTENDED, label="Projected"
        )
        axis.set(
            xticks=x,
            xticklabels=LOCATION_LABELS,
            xlabel="System location",
            ylabel=ylabel,
        )
        axis.tick_params(axis="x", labelrotation=35)
    location_axes[0].legend(ncol=2)
    composite_axes = [fig.add_subplot(grid[2, column]) for column in range(3)]
    for axis, metric_index, ylabel in (
        (composite_axes[0], 0, "Coordinate-normalized RMSE (nRMSE)"),
        (composite_axes[1], 1, "Coordinate-normalized MAE (nMAE)"),
        (composite_axes[2], 2, "Mean location R²"),
    ):
        raw_values = [row[metric_index] for row in component_scores["raw"]]
        projected_values = [row[metric_index] for row in component_scores["projected"]]
        axis.bar(np.arange(4) - width / 2, raw_values, width, color=RAW, label="Raw")
        axis.bar(
            np.arange(4) + width / 2,
            projected_values,
            width,
            color=EXTENDED,
            label="Projected",
        )
        axis.set(
            xticks=np.arange(4),
            xticklabels=COMPOSITES,
            xlabel="Water-quality composite",
            ylabel=ylabel,
        )
    composite_axes[0].legend(ncol=2)
    save(fig, output, "q01_holdout_accuracy_overview")


def _plot_heatmaps(data, output, summary):
    composite_predictions = data.composites
    matrices = {}
    for method in ("raw", "projected"):
        matrix = np.empty((len(LOCATIONS), len(COMPOSITES)))
        for i in range(len(LOCATIONS)):
            for j in range(len(COMPOSITES)):
                matrix[i, j] = finite_score(
                    composite_predictions["mechanistic"][:, i, j],
                    composite_predictions[method][:, i, j],
                )[0]
        matrices[method] = matrix
    delta = np.divide(
        100.0 * (matrices["projected"] - matrices["raw"]),
        matrices["raw"],
        out=np.full_like(matrices["raw"], np.nan),
        where=matrices["raw"] > 0,
    )
    finite_delta = np.abs(delta[np.isfinite(delta)])
    limit = max(1.0, float(finite_delta.max()) if finite_delta.size else 1.0)
    common_max = float(
        max(np.nanmax(matrices["raw"]), np.nanmax(matrices["projected"]))
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.3))
    for axis, matrix, title in (
        (axes[0], matrices["raw"], "Raw composite nRMSE"),
        (axes[1], matrices["projected"], "Projected composite nRMSE"),
    ):
        image = axis.imshow(
            matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=common_max
        )
        fig.colorbar(
            image, ax=axis, pad=0.02, label="Coordinate-normalized RMSE (nRMSE)"
        )
        axis.set(
            yticks=np.arange(len(LOCATIONS)),
            yticklabels=LOCATION_LABELS,
            xticks=np.arange(4),
            xticklabels=COMPOSITES,
            xlabel="Water-quality composite",
            ylabel="System location",
        )
    image = axes[2].imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    fig.colorbar(image, ax=axes[2], pad=0.02, label="nRMSE change (%)")
    axes[2].set(
        yticks=np.arange(len(LOCATIONS)),
        yticklabels=LOCATION_LABELS,
        xticks=np.arange(4),
        xticklabels=COMPOSITES,
        xlabel="Water-quality composite",
        ylabel="System location",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output, "q02_holdout_accuracy_by_location")
    summary.append(
        {
            "question": 4,
            "metric": "location_composite_cells_improved",
            "value": int(np.sum(delta < 0.0)),
        }
    )


def _plot_parity(data, output, summary):
    composite_predictions = data.composites
    all_location_truth = composite_predictions["mechanistic"]
    all_location_prediction = composite_predictions["projected"]
    location_colors = plt.get_cmap("tab10")(np.arange(len(LOCATIONS)))
    location_markers = ("o", "s", "^", "v", "D", "P", "X", "*")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.2))
    density_artists: list[object] = []
    for component_index, (axis, component) in enumerate(
        zip(axes.flat, COMPOSITES, strict=True)
    ):
        truth_by_location = all_location_truth[:, :, component_index]
        prediction_by_location = all_location_prediction[:, :, component_index]
        truth = truth_by_location.ravel()
        predicted = prediction_by_location.ravel()
        nrmse, nmae, mean_r2 = coordinate_normalized_score(
            truth_by_location, prediction_by_location
        )
        low = float(min(np.min(truth), np.min(predicted)))
        high = float(max(np.max(truth), np.max(predicted)))
        if low <= 0.0:
            raise RuntimeError(
                f"Q18 {component} requires positive values for logarithmic parity axes"
            )
        limits = (low / 1.15, high * 1.15)
        density_artists.append(
            axis.hexbin(
                truth,
                predicted,
                gridsize=48,
                mincnt=1,
                cmap="viridis",
                xscale="log",
                yscale="log",
            )
        )
        axis.plot(
            limits, limits, color="#dc2626", lw=1.2, ls="--", label="Perfect match"
        )
        for location_index, (location, color, marker) in enumerate(
            zip(LOCATION_LABELS, location_colors, location_markers, strict=True)
        ):
            axis.scatter(
                np.median(truth_by_location[:, location_index]),
                np.median(prediction_by_location[:, location_index]),
                s=44 if marker != "*" else 64,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
                label=f"Median — {location}",
            )
        axis.set(
            xlim=limits,
            ylim=limits,
            xlabel=f"Mechanistic {component} (mg/L)",
            ylabel=f"Extended ICSOR prediction {component} (mg/L)",
        )
        axis.text(
            0.03,
            0.97,
            f"nRMSE = {nrmse:.3f}\nMean location R² = {mean_r2:.3f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        axis.set_aspect("equal", adjustable="box")
        summary.extend(
            (
                {
                    "question": 18,
                    "metric": f"holdout_all_locations_{component.lower()}_mean_r2",
                    "value": mean_r2,
                },
                {
                    "question": 18,
                    "metric": f"holdout_all_locations_{component.lower()}_nrmse",
                    "value": nrmse,
                },
                {
                    "question": 18,
                    "metric": f"holdout_all_locations_{component.lower()}_nmae",
                    "value": nmae,
                },
            )
        )
    maximum_bin_count = max(
        2.0, *(float(np.max(artist.get_array())) for artist in density_artists)
    )
    density_norm = LogNorm(vmin=1.0, vmax=maximum_bin_count)
    for artist in density_artists:
        artist.set_norm(density_norm)
        artist.set_clim(1.0, maximum_bin_count)
    colorbar_axis = fig.add_axes((0.91, 0.14, 0.018, 0.68))
    colorbar = fig.colorbar(density_artists[0], cax=colorbar_axis)
    colorbar.set_label("Holdout location-observations per hexagon (log scale)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=5)
    fig.subplots_adjust(
        left=0.08, right=0.87, bottom=0.13, top=0.91, wspace=0.3, hspace=0.3
    )
    save(fig, output, "q03_holdout_parity_all_locations")


def _plot_operating_values(data, output, summary):
    exact = data.exact
    controls = data.controls
    display_cases = CASES
    case_labels = {case: scenario_label(case) for case in CASES}
    x = np.arange(11)
    control_columns = ("H", "a_3", "a_4", "a_5", "r_I", "r_R", "w")
    control_titles = tuple(CONTROL_LABELS[name] for name in control_columns)
    fig = plt.figure(figsize=(20, 12))
    grid = fig.add_gridspec(3, 4, hspace=0.38, wspace=0.32)
    effluent_axes = [fig.add_subplot(grid[0, column]) for column in range(4)]
    for axis, component in zip(effluent_axes, COMPOSITES, strict=True):
        pivot = (
            exact[exact.case.isin(display_cases)]
            .pivot(index="case", columns="route", values=component)
            .reindex(display_cases)
        )
        axis.plot(x, pivot["surrogate"], marker="o", color=EXTENDED)
        axis.plot(x, pivot["mechanistic"], marker="s", color=DIRECT)
        axis.set(
            xticks=x,
            xticklabels=[case_labels[c] for c in display_cases],
            xlabel="Influent scenario",
            ylabel=f"{component} (mg/L)",
        )
        axis.tick_params(axis="x", labelrotation=35)
    control_axes = [
        fig.add_subplot(grid[row, column]) for row in (1, 2) for column in range(4)
    ]
    for axis, column, title in zip(
        control_axes, control_columns, control_titles, strict=False
    ):
        for route in ROUTES:
            values = np.asarray(
                [controls.loc[(case, route), column] for case in display_cases], float
            )
            axis.plot(x, values, marker=ROUTE_MARKER[route], color=ROUTE_COLOR[route])
        axis.set(ylabel=title, xlabel="Influent scenario")
        axis.set_xticks(x, [case_labels[c] for c in display_cases])
        axis.tick_params(axis="x", labelrotation=35)
        format_control_axis(
            axis,
            np.concatenate(
                [
                    np.asarray(
                        [controls.loc[(case, route), column] for case in display_cases],
                        float,
                    )
                    for route in ROUTES
                ]
            ),
        )
    control_axes[-1].axis("off")
    route_handles = [
        Line2D(
            (0,),
            (0,),
            marker=ROUTE_MARKER[route],
            color=ROUTE_COLOR[route],
            label=ROUTE_LABEL[route],
        )
        for route in ROUTES
    ]
    fig.legend(
        [*route_handles],
        [*[handle.get_label() for handle in route_handles]],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
    )
    fig.subplots_adjust(
        left=0.055, right=0.99, bottom=0.075, top=0.99, hspace=0.38, wspace=0.32
    )
    save(fig, output, "q05_effluent_and_operating_values")


def _plot_objectives(data, output, summary):
    exact = data.exact
    weights = data.weights
    display_cases = CASES
    robust_cases = CASES[1:]
    case_labels = {case: scenario_label(case) for case in CASES}
    x = np.arange(11)
    width = 0.37
    eligible = np.ones(10, dtype=bool)
    eligibility = pd.Series(True, index=robust_cases)

    def add_paired_bars(
        axis: plt.Axes, column: str, ylabel: str, question: int
    ) -> None:
        pivot = (
            exact[exact.case.isin(display_cases)]
            .pivot(index="case", columns="route", values=column)
            .reindex(display_cases)
        )
        axis.bar(
            x - width / 2,
            pivot["surrogate"],
            width,
            color=EXTENDED,
            label=ROUTE_LABEL["surrogate"],
        )
        axis.bar(
            x + width / 2,
            pivot["mechanistic"],
            width,
            color=DIRECT,
            label=ROUTE_LABEL["mechanistic"],
        )
        axis.set(
            xticks=x,
            xticklabels=[case_labels[c] for c in display_cases],
            xlabel="Influent scenario",
            ylabel=ylabel,
        )
        valid = pivot.reindex(robust_cases).loc[eligibility]
        summary.extend(
            (
                {
                    "question": question,
                    "metric": "paired_influent_scenarios",
                    "value": int(eligible.sum()),
                },
                {
                    "question": question,
                    "metric": "extended_icsor_lower_scenario_count",
                    "value": int((valid.surrogate < valid.mechanistic).sum()),
                },
                {
                    "question": question,
                    "metric": "smooth_nlp_lower_scenario_count",
                    "value": int((valid.mechanistic < valid.surrogate).sum()),
                },
            )
        )

    economic_columns = (
        "hrt",
        "aeration",
        "internal_recycle",
        "return_sludge",
        "wasting",
    )
    economic_labels = (
        "HRT",
        "Aeration",
        "Internal recycle",
        "Return sludge",
        "Wasting",
    )
    economic_colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2")
    fig, axes = plt.subplots(1, 3, figsize=(23, 5.8), squeeze=False)
    add_paired_bars(axes[0, 0], "objective", "Exact total objective", 7)
    add_paired_bars(axes[0, 1], "quality", "Normalized water-quality component", 8)
    economic_axis = axes[0, 2]
    for route, offset, hatch in (
        ("surrogate", -width / 2, ""),
        ("mechanistic", width / 2, "///"),
    ):
        data = (
            exact[exact.route.eq(route) & exact.case.isin(display_cases)]
            .set_index("case")
            .reindex(display_cases)
        )
        bottom = np.zeros(len(display_cases))
        for column, label, color, weight in zip(
            economic_columns, economic_labels, economic_colors, weights[1:], strict=True
        ):
            values = weight * data[column].to_numpy(float)
            economic_axis.bar(
                x + offset,
                values,
                width,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                hatch=hatch,
                label=label if route == "surrogate" else None,
            )
            bottom += values
    economic_axis.set(
        xticks=x,
        xticklabels=[case_labels[c] for c in display_cases],
        xlabel="Influent scenario",
        ylabel="Weighted economic/resource contribution",
    )
    legend_handles = [
        Patch(facecolor=EXTENDED, label=ROUTE_LABEL["surrogate"]),
        Patch(facecolor=DIRECT, label=ROUTE_LABEL["mechanistic"]),
        *[
            Patch(facecolor=color, label=label)
            for color, label in zip(economic_colors, economic_labels, strict=True)
        ],
        Patch(facecolor="white", edgecolor="black", label="Extended ICSOR: left/solid"),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="///",
            label="Smooth NLP: right/hatched",
        ),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, output, "q07_objective_quality_economic")


def _plot_optimization_time(data, output, summary):
    timing = data.timing
    robustness_timing = timing.reindex(CASES[1:])
    display_cases = CASES
    case_labels = {case: scenario_label(case) for case in CASES}
    x = np.arange(11)
    width = 0.37
    fig, axis = plt.subplots(figsize=(12, 5.7))
    axis.bar(
        x - width / 2,
        timing["surrogate"],
        width,
        color=EXTENDED,
        label=ROUTE_LABEL["surrogate"],
    )
    axis.bar(
        x + width / 2,
        timing["mechanistic"],
        width,
        color=DIRECT,
        label=ROUTE_LABEL["mechanistic"],
    )
    axis.set_yscale("log")
    axis.set(
        xticks=x,
        xticklabels=[case_labels[c] for c in display_cases],
        xlabel="Influent scenario",
        ylabel="Optimization time (s; log scale)",
    )
    axis.legend(ncol=2)
    fig.tight_layout()
    save(fig, output, "q08_optimization_time")
    summary.extend(
        (
            {
                "question": 12,
                "metric": "extended_icsor_mean_seconds",
                "value": float(robustness_timing.surrogate.mean()),
            },
            {
                "question": 12,
                "metric": "smooth_nlp_mean_seconds",
                "value": float(robustness_timing.mechanistic.mean()),
            },
            {
                "question": 12,
                "metric": "extended_icsor_faster_influent_scenario_count",
                "value": int(
                    (robustness_timing.surrogate < robustness_timing.mechanistic).sum()
                ),
            },
        )
    )


def _plot_profiles(data, output, summary):
    labels = ("Influent", "Mixer", "R1", "R2", "R3", "R4", "R5", "Effluent")
    colors = {
        "nominal": "#111827",
        **{case: plt.get_cmap("tab20")(index) for index, case in enumerate(CASES[1:])},
    }
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True, squeeze=False)
    for component_index, component in enumerate(COMPOSITES):
        for column, route in enumerate(ROUTES):
            axis = axes[component_index, column]
            for case in CASES:
                values = data.profiles[case, route][:, component_index]
                axis.plot(
                    np.arange(8),
                    values,
                    color=colors[case],
                    lw=2.2 if case == "nominal" else 1.25,
                    marker="o",
                    ms=4.2,
                    alpha=1.0 if case == "nominal" else 0.72,
                )
            axis.set_yscale("log")
            axis.set_ylabel(f"{ROUTE_LABEL[route]}\n{component} (mg/L; log scale)")
            axis.set_xlabel("Main liquid-treatment location")
            axis.set_xticks(np.arange(8), labels)
        values = np.concatenate(
            [
                data.profiles[case, route][:, component_index]
                for case in CASES
                for route in ROUTES
            ]
        )
        limits = (float(values.min()) * 0.9, float(values.max()) * 1.1)
        for axis in axes[component_index]:
            axis.set_ylim(*limits)
    handles = [
        Line2D(
            (0,),
            (0,),
            color=colors[case],
            lw=2.2 if case == "nominal" else 1.5,
            marker="o",
            label=scenario_label(case),
        )
        for index, case in enumerate(CASES)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.005),
        title="Influent scenario",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    save(fig, output, "q06_treatment_train_profiles")


def render_figures(data: ChartData, output: Path):
    style()
    summary = []
    overall, locations, components = holdout_statistics(data)
    for method in ("raw", "projected"):
        summary.extend(
            {
                "question": 1,
                "metric": f"holdout_{method}_{metric}",
                "value": float(overall.loc[method, metric]),
            }
            for metric in ("nrmse", "nmae", "r2_mean")
        )
        for component, values in zip(COMPOSITES, components[method], strict=True):
            summary.extend(
                {
                    "question": 3,
                    "metric": f"holdout_{method}_{component.lower()}_{metric}",
                    "value": float(value),
                }
                for metric, value in zip(
                    ("nrmse", "nmae", "mean_location_r2"), values, strict=True
                )
            )
    raw_error = float(overall.loc["raw", "nrmse"])
    summary.append(
        {
            "question": 1,
            "metric": "composite_projection_nrmse_improvement_percent",
            "value": 100
            * (raw_error - float(overall.loc["projected", "nrmse"]))
            / raw_error
            if raw_error
            else np.nan,
        }
    )
    _plot_accuracy(data, output, summary)
    _plot_heatmaps(data, output, summary)
    _plot_parity(data, output, summary)
    labels = {case: scenario_label(case) for case in CASES}
    _plot_effluent_parity(
        data.quality,
        data.influent,
        cases=list(CASES),
        case_labels=labels,
        output=output,
    )
    projected = (
        data.quality.query("response_method == 'projected'")
        .set_index("case")
        .reindex(CASES)
    )
    reference = (
        data.quality.query(
            "response_method == 'reference' and decision_route == 'surrogate'"
        )
        .set_index("case")
        .reindex(CASES)
    )
    for component in COMPOSITES:
        error = (
            100
            * np.abs(projected[component].to_numpy() - reference[component].to_numpy())
            / data.influent.loc[list(CASES), component].to_numpy()
        )
        summary.append(
            {
                "question": "5R",
                "metric": f"{component.lower()}_median_absolute_removal_error_percentage_points",
                "value": float(np.median(error)),
            }
        )
    _plot_operating_values(data, output, summary)
    _plot_profiles(data, output, summary)
    _plot_objectives(data, output, summary)
    _plot_optimization_time(data, output, summary)
    metrics = presentation_table(overall.reset_index())
    metrics["r2_label"] = "Mean location R²"
    summary_frame = pd.DataFrame(summary)
    summary_frame["metric_label"] = summary_frame["metric"].map(chart_metric_label)
    return metrics, summary_frame
