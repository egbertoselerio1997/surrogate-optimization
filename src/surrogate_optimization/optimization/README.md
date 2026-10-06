# Optimization routes

`mechanistic.py` builds the smooth model and runs its three-stage continuation. `surrogate.py` builds the expression graph and evaluates the exact cold projection at each distinct trial. `active_set.py` supplies audited sensitivities and local refinement. `certification.py` implements independent endpoint checks and two-scale feasible polls. `trust.py` calibrates diagnostics from development rows. Shared surrogate records live in `types.py`.

Each primary route starts at the normalized box center. The surrogate falls back to value-only COBYQA when exact derivatives are unavailable. A failed mechanistic primary search may receive one recovery initialized from a certified surrogate decision. Incumbents with unresolved stationarity are labeled accordingly.

Tests: `test_active_set.py`, `test_mechanistic_optimization.py`, `test_trust_diagnostics.py`, and `test_surrogate_conditioning.py`.
