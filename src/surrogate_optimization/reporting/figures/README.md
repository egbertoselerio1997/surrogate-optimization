# Figures

Four families produce eighteen figures:

- `comparison`: eight PNG figures for holdout accuracy, selected decisions, process profiles, objectives, and timing.
- `emulation`: three PNG/SVG figures for surrogate fidelity.
- `insights`: five PNG/SVG figures for timing, fidelity, and paired operating/quality comparisons.
- `nominal_parity`: two PNG/PDF figures comparing nominal selected decisions with exact replay.

Each module exposes `generate_figures(run, output)` and an explicit `FIGURES` registry. `common.py` provides headless, atomic export. Missing scientific decisions produce explanatory unavailable-data figures. Corrupt required inputs raise errors. There are no archived chart overrides.
