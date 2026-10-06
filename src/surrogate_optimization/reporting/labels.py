"""Shared presentation vocabulary for result tables and reference charts."""

from __future__ import annotations

import pandas as pd

ROUTE_LABELS = {"surrogate": "Extended ICSOR", "mechanistic": "Smooth NLP"}
RESPONSE_LABELS = {
    "raw": "Raw",
    "projected": "Projected",
    "reference": "Mechanistic",
    "mechanistic": "Mechanistic",
    "smooth": "Smooth NLP",
    "projection_correction": "Projection correction",
}
LOCATION_LABELS = {
    "influent": "Influent",
    "mixer": "Mixer",
    **{f"reactor_{i}": f"R{i}" for i in range(1, 6)},
    "overflow": "Overflow",
    "underflow": "Underflow",
    "clarifier_overflow": "Overflow",
    "clarifier_underflow": "Underflow",
    "clarifier_inventory": "Clarifier solids inventory",
    **{f"clarifier_layer_{i}": f"Clarifier layer {i}" for i in range(1, 11)},
}
CONTROL_LABELS = {
    "H": "HRT (h)",
    "a_3": "Aeration a3",
    "a_4": "Aeration a4",
    "a_5": "Aeration a5",
    "r_I": "Internal recycle rI",
    "r_R": "Return sludge rR",
    "w": "Waste fraction w",
}
COLUMN_LABELS = {
    **CONTROL_LABELS,
    "case": "Influent scenario",
    "route": "Method",
    "decision_route": "Method",
    "response_method": "Response",
    "location": "Location",
    "nrmse": "nRMSE",
    "nmae": "nMAE",
    "r2_mean": "Mean coordinate R²",
    "mean_location_r2": "Mean location R²",
    "quality": "Normalized water-quality component",
    "hrt": "HRT",
    "aeration": "Aeration",
    "internal_recycle": "Internal recycle",
    "return_sludge": "Return sludge",
    "wasting": "Wasting",
    "J_S_reference": "Extended ICSOR exact total objective",
    "J_M_reference": "Smooth NLP exact total objective",
    "surrogate_time_seconds": "Extended ICSOR optimization time (s)",
    "mechanistic_time_seconds": "Smooth NLP optimization time (s)",
    "time_seconds": "Optimization time (s)",
    "objective": "Response total objective",
    "recomputed_objective": "Selected-response total objective",
    "solver_reported_objective": "Solver objective",
    "projected_reference_nrmse_at_S": "Projected development-scale response nRMSE",
    "smooth_reference_nrmse_at_M": "Smooth NLP development-scale response nRMSE",
}


def scenario_label(case: str) -> str:
    if case == "nominal":
        return "N"
    if case.startswith("robustness_") and case[11:].isdigit():
        return f"S{int(case[11:])}"
    return case


def chart_metric_label(metric: str) -> str:
    """Readable labels for the calculated chart-summary identifiers."""
    phrases = {
        "extended_icsor_mean_seconds": "Extended ICSOR mean optimization time (s)",
        "smooth_nlp_mean_seconds": "Smooth NLP mean optimization time (s)",
        "extended_icsor_faster_influent_scenario_count": "Influent scenarios with shorter Extended ICSOR optimization time",
    }
    if metric in phrases:
        return phrases[metric]
    label = metric
    for token, display in (
        ("extended_icsor", "Extended ICSOR"),
        ("smooth_nlp", "Smooth NLP"),
        ("mean_location_r2", "Mean location R²"),
        ("r2_mean", "Mean location R²"),
        ("mean_r2", "Mean location R²"),
        ("nrmse", "nRMSE"),
        ("nmae", "nMAE"),
        ("percentage_points", "percentage points"),
        ("raw", "Raw"),
        ("projected", "Projected"),
        ("cod", "COD"),
        ("tss", "TSS"),
        ("tn", "TN"),
        ("tp", "TP"),
    ):
        label = label.replace(token, display)
    return label.replace("_", " ")


def presentation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Add display labels without changing scientific values or artifact identifiers."""
    result = frame.copy()
    mappings = {
        "route": ("route_label", ROUTE_LABELS),
        "decision_route": ("decision_route_label", ROUTE_LABELS),
        "method": ("method_label", {**RESPONSE_LABELS, "surrogate": "Extended ICSOR"}),
        "response_method": ("response_label", RESPONSE_LABELS),
        "location": ("location_label", LOCATION_LABELS),
    }
    if "case" in result:
        result["scenario_label"] = result["case"].map(
            lambda value: scenario_label(value) if isinstance(value, str) else value
        )
    for column, (label, mapping) in mappings.items():
        if column in result:
            result[label] = result[column].map(mapping).fillna(result[column])
    return result


def vocabulary() -> dict:
    return {
        "routes": ROUTE_LABELS,
        "responses": RESPONSE_LABELS,
        "scenarios": {
            "nominal": "N",
            **{f"robustness_{i:02d}": f"S{i}" for i in range(1, 11)},
        },
        "locations": LOCATION_LABELS,
        "column_labels": COLUMN_LABELS,
        "concentration_unit": "mg/L",
        "removal_unit": "%",
        "removal_error_unit": "percentage points",
        "optimization_time_unit": "s",
    }


TABLE_GUIDE = """# Result tables

Methods are **Extended ICSOR** and **Smooth NLP**. Influent scenarios are **N**
(nominal) and **S1–S10**. Display-label columns use the chart vocabulary;
identifier columns remain available for joining artifacts. Responses are **Raw**,
**Projected**, and **Mechanistic**. Reactor locations are **R1–R5**.

`selected_controls` contains selected decisions. `selected_quality` contains
effluent concentrations (mg/L). `process_profiles` contains concentrations
(mg/L) and clarifier solids inventory (g). `objective_decomposition` contains
the normalized water-quality component and unweighted resource components;
the objective weights give their weighted economic/resource contribution.
Its selected-response objective uses the row's response method. The comparison
tables and Figure 7 use the exact mechanistic replay objective.
Timing tables report **Optimization time** in seconds.

The charts' holdout composite nRMSE and nMAE use each location/composite's
holdout range; **Mean location R²** averages location scores. Prediction and
case-comparison tables retain their development-scale response errors and
coordinate R². These describe different response sets and normalizations.
`report_manifest.json` provides shared names and column labels. Failure and
status tables retain all expected cases, including unavailable decisions.
"""
