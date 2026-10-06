# Reference result charts

`package.py` publishes exactly eight untitled PNGs directly in a run's
`report/figures` directory. `comparison.py` implements the panel layouts;
`data.py` validates and reads the target's own arrays and audited casewise
records. `common.py` exports figures atomically. Additional chart families,
SVGs, PDFs, and unavailable-data placeholders are not generated.

The figures are, in order:

1. `q01_holdout_accuracy_overview.png`: 3-by-3 aggregate/location/composite metrics.
2. `q02_holdout_accuracy_by_location.png`: three location/composite heatmaps.
3. `q03_holdout_parity_all_locations.png`: 2-by-2 logarithmic parity with shared density.
4. `q04_effluent_and_removal_parity.png`: 2-by-4 concentration and removal parity.
5. `q05_effluent_and_operating_values.png`: 3-by-4 quality and controls.
6. `q06_treatment_train_profiles.png`: four composite rows and two route columns.
7. `q07_objective_quality_economic.png`: objective, quality, and resource stacks.
8. `q08_optimization_time.png`: paired logarithmic bars for N and S1?S10.

Each package also contains `README.md`, `chart_index.csv`, `chart_summary.csv`,
and `holdout_composite_metrics.csv`. Required data are validated before rendering;
all eight exports are verified before publication. Known obsolete generated
exports are removed after publication. Unrelated user files are preserved.

To regenerate charts without rerunning scientific calculations:

```powershell
uv run python -m surrogate_optimization.reporting.figures.package results/production_001
```

Use `--output <folder>` to preview in a separate directory. The reader supports
current artifacts and the supplied reference run's older field names. Only that
established reference run may use its existing S1 chart override when the
scientific casewise response is unavailable. Scientific records are never edited.

Tests: `tests/unit/test_reference_charts.py` and `test_reporting_finalization.py`.
