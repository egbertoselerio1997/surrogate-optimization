# Surrogate optimization package

The package compares surrogate and mechanistic optimization on a recycling activated-sludge plant.

- `plant`: physical definitions, operating points, and exact steady states.
- `data`: deterministic candidate design, generation, and provenance.
- `surrogate`: ridge fitting, overflow closure, and physical projection.
- `optimization`: searches, sensitivities, trust diagnostics, and certification.
- `validation`: assessment, physical audits, and exact replay.
- `runtime`: atomic artifacts, immutable contracts, checkpoints, and workers.
- `workflow`: execution stages.
- `reporting`: tables, figures, timing, and final manifests.
- `cli`: production command.

`config.py` loads the authoritative parameters; `paths.py` resolves repository-owned paths. Package import does not launch solvers or create outputs.
