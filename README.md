# Surrogate optimization

Compare a physically constrained statistical surrogate and a smooth
mechanistic optimizer for a recycling activated-sludge plant. Both methods use
the same seven controls, objective, and engineering requirements. Selected
decisions are checked on the exact mechanistic model.

## Setup

Use Python 3.12 and run from the repository root:

```powershell
uv sync --frozen
```

## Run

```powershell
uv run python -u -m surrogate_optimization.cli.run_study `
  --run-id production_001 `
  --through complete
```

The production workload attempts 8,000 development candidates and 2,000
holdout candidates. Rejected candidates are recorded and are not replaced.
It evaluates the nominal case and ten robustness cases with both methods,
then writes exact replays, audits, tables, and eight PNG figures. A complete run can take
hours or days.

All outputs are in `results/<run-id>`. Use `--through generation` or
`--through assessment` to stop earlier. Reuse the same run ID to resume an
interrupted run with matching code, parameters, and intact checkpoints.
Use a new ID after changing the computation. Older checkpoint formats are not
supported.

`run_state.json` records execution status. A finished workload can report
`complete_with_validation_failures`; inspect the audits before interpreting
its results. `report/manifest.json` records reporting artifacts and their
hashes. The eight numbered PNGs and their chart index, numerical summary, and guide
are in `report/figures`. Missing required scenario data stop chart publication.
Optimization time is reported in seconds.

## Tests and source

```powershell
uv run python -m unittest discover -s tests -t . -v
```

This includes a real 80/20-candidate integration run with all eleven cases;
it can take substantially longer than the unit tests. It does not execute
the full production workload. For unit tests only, use `-s tests/unit`.

See [source structure](src/README.md), [parameters](config/README.md), and
[test details](tests/README.md).
