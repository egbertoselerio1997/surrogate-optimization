"""Publish exactly the eight reference figures and their calculated metadata."""

from __future__ import annotations

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import pandas as pd

from surrogate_optimization.reporting.figures import comparison
from surrogate_optimization.reporting.figures.data import (
    ChartData,
    ChartDataError,
    load_chart_data,
)
from surrogate_optimization.runtime.artifacts import atomic_bytes, atomic_dataframe


def _readme(data: ChartData) -> str:
    sources = "\n".join(f"- `{path}`" for path in data.sources)
    return (
        """# Reference result charts

Eight untitled PNG figures present the target run's calculated results. The
presentation order is recorded in `chart_index.csv`; principal comparisons
are in `chart_summary.csv` and `holdout_composite_metrics.csv`.

## Presentation plan

1. Figures 1–3 establish holdout validation: aggregate metrics, location
   heatmaps, and all-location parity.
2. Figures 4–6 assess selected decisions and process behavior: effluent and
   removal parity, operating controls, and treatment-train profiles.
3. Figures 7–8 compare objective trade-offs and optimization time.

## Figures and questions

1. `q01_holdout_accuracy_overview.png`: How does projection change accuracy?
   The 3-by-3 grid shows aggregate, location, and composite accuracy in its
   rows, with nRMSE, nMAE, and mean location R² in its columns. Raw and projected
   predictions use the same mechanistic holdout observations.
2. `q02_holdout_accuracy_by_location.png`: Where does accuracy change? Three
   heatmaps show raw nRMSE, projected nRMSE, and percentage change for each
   location/composite. Raw and projected maps share a scale. Negative change
   means improvement; a zero raw error makes percentage change undefined.
3. `q03_holdout_parity_all_locations.png`: Do projected predictions agree with
   the mechanism across locations? The four composite panels pool all holdout
   location-observations on logarithmic axes. Hexagons show counts with one
   shared logarithmic density scale. Markers named `Median — <location>` show
   location medians, not individual observations. A one-to-one line and nRMSE
   and mean location R² annotations support interpretation.
4. `q04_effluent_and_removal_parity.png`: How accurately does Extended ICSOR
   predict the selected decisions? The 2-by-4 grid compares projected and exact
   mechanistic responses at the same controls. Teal points show effluent
   concentrations; orange points show removal. Columns are COD, TN, TP, TSS.
5. `q05_effluent_and_operating_values.png`: What quality and controls do the
   selected decisions produce? The top row shows exact mechanistic effluent
   composites for both routes. The lower rows show HRT, a3, a4, a5, internal
   recycle, return sludge, and wasting. The unused panel is off and one legend
   identifies both routes. Effectively constant controls within 0–1 use plain
   0–1 limits.
6. `q06_treatment_train_profiles.png`: How does treatment progress through the
   plant? Four composite rows and two route columns show N and S1–S10 along
   Influent → Mixer → R1 → R2 → R3 → R4 → R5 → Clarifier effluent. Each row uses
   identical logarithmic limits for both routes and one scenario legend.
7. `q07_objective_quality_economic.png`: What are the objective trade-offs?
   Three panels show exact mechanistic total objective, unweighted normalized
   water quality, and stacked weighted resource contributions. Extended ICSOR
   uses solid bars and Smooth NLP uses hatched resource bars.
8. `q08_optimization_time.png`: How does optimization time vary across influent
   scenarios? Paired route bars include N and S1–S10, with logarithmic seconds
   labelled `Optimization time (s; log scale)`. Summary means cover S1–S10.

## Conventions and calculations

Extended ICSOR is `surrogate`; Smooth NLP is `mechanistic` (or `direct` in older
input artifacts). N is nominal, S1–S10 are influent scenarios, and R1–R5 are
biological reactors. Scenario displays include nominal. Concentrations use
mg/L, removal uses %, and removal differences use percentage points.

Every holdout location/composite error is divided by its own mechanistic
holdout range. nRMSE pools squared normalized errors; nMAE pools absolute
normalized errors. R² is calculated separately at each location/composite and
averaged over the coordinates represented by a panel. Negative R² is retained.

Removal is `100*(fresh influent - effluent)/fresh influent`. Figures 5–7 use
exact mechanistic responses at each route's selected controls. Quality scales
are the population standard deviations of development effluent composites.
Normalized quality is the mean of the four composites divided by those scales.
The objective uses the target's declared bounds and weights; resource stacks
already include their weights. Figure 7's quality panel is unweighted.

No scientific records are changed by chart generation. Missing required data
prevent publication of a complete package. The q01–q08 numbers denote this
presentation order; they may correspond to Figures 2–9 when a plant flowsheet
is numbered first.

## Target data sources

"""
        + sources
        + "\n"
    )


def _obsolete_paths(output: Path) -> list[Path]:
    """Only explicit filenames previously generated by this repository are owned."""
    prior = {
        "comparison": (
            "holdout_accuracy_overview",
            "holdout_accuracy_by_location",
            "holdout_parity_all_locations",
            "effluent_and_removal_parity",
            "effluent_and_operating_values",
            "treatment_train_profiles",
            "objective_quality_economic",
            "optimization_time",
        ),
        "emulation": (
            "all_output_parity",
            "coordinate_accuracy",
            "representative_outputs",
        ),
        "insights": (
            "optimization_time",
            "effluent_composite_fidelity",
            "fidelity_by_plant_component",
            "optimization_economics_and_quality",
            "optimization_tradeoff_frontier",
        ),
        "nominal_parity": (
            "nominal_surrogate_response_parity",
            "nominal_mechanistic_response_parity",
        ),
    }
    paths = [
        output / family / f"{stem}.{extension}"
        for family, stems in prior.items()
        for stem in stems
        for extension in ("png", "svg", "pdf")
    ]
    paths += [
        output / f"{stem}.{extension}"
        for stem in prior["comparison"]
        for extension in ("png", "svg", "pdf")
    ]
    retired = (
        "q01_q02_q03_holdout_accuracy",
        "q01_holdout_composite_accuracy",
        "q02_holdout_accuracy_by_response_block",
        "q03_holdout_component_accuracy",
        "q04_holdout_component_accuracy_by_stage",
        "q18_holdout_effluent_composite_parity",
        "q05_effluent_and_removal_parity",
        "q05_surrogate_effluent_prediction_vs_mechanistic",
        "q05_surrogate_percent_removal_vs_mechanistic",
        "q09_q11_effluent_and_operating_values",
        "q09_exact_effluent_composites",
        "q11_optimal_operating_values",
        "q14_cod_tn_tp_tss_main_treatment_train_profiles",
        "q14_cod_main_treatment_train_profiles",
        "q15_tn_main_treatment_train_profiles",
        "q15_tn_tp_tss_main_treatment_train_profiles",
        "q16_tp_main_treatment_train_profiles",
        "q17_tss_main_treatment_train_profiles",
        "q07_q08_q10_objective_quality_economic",
        "q07_exact_optimal_objective",
        "q08_exact_water_quality_component",
        "q10_exact_economic_component",
        "q12_primary_optimization_time",
        "q06_smooth_nlp_effluent_prediction_vs_mechanistic",
        "q06_smooth_nlp_percent_removal_vs_mechanistic",
        "q13_exact_objective_value_comparison",
    )
    paths += [
        output / f"{stem}.{extension}"
        for stem in retired
        for extension in ("png", "svg", "pdf")
    ]
    paths += [
        output / f"{stem}.{extension}"
        for stem in comparison.FIGURES
        for extension in ("svg", "pdf")
    ]
    return paths


def generate_figures(run: Path, output: Path | None = None) -> dict[str, str]:
    data = load_chart_data(run)
    output = (output or run / "report/figures").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".reference-charts-", dir=output.parent
    ) as directory:
        staging = Path(directory)
        try:
            metrics, summary = comparison.render_figures(data, staging)
        finally:
            plt.close("all")
        index = pd.DataFrame(
            [
                {
                    "results_sequence": sequence,
                    "source_questions": questions,
                    "presentation_role": role,
                    "png": f"{stem}.png",
                }
                for sequence, questions, role, stem in comparison.FIGURE_INDEX
            ]
        )
        atomic_dataframe(staging / "chart_index.csv", index)
        atomic_dataframe(staging / "chart_summary.csv", summary)
        atomic_dataframe(staging / "holdout_composite_metrics.csv", metrics)
        atomic_bytes(staging / "README.md", _readme(data).encode("utf-8"))
        expected = {f"{stem}.png" for stem in comparison.FIGURES}
        if (
            {path.name for path in staging.glob("*.png")} != expected
            or list(staging.glob("*.svg"))
            or len(index) != 8
        ):
            raise RuntimeError(
                "the chart package must contain exactly eight registered PNG figures"
            )
        for name in (*sorted(expected), *comparison.SIDECARS):
            path = staging / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"required chart output is missing: {name}")
        output.mkdir(parents=True, exist_ok=True)
        for name in (*sorted(expected), *comparison.SIDECARS):
            atomic_bytes(output / name, (staging / name).read_bytes())
        for path in _obsolete_paths(output):
            if path.is_file() and output in path.resolve().parents:
                path.unlink()
    return {}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        generate_figures(args.run.resolve(), args.output)
    except ChartDataError as exc:
        parser.exit(1, f"{exc}\n")
    print((args.output or args.run / "report/figures").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
