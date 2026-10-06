# Physical model

`definitions.py` holds components, bounds, kinetic constants, stoichiometry, and invariants. `operating_point.py` defines seven independent controls and clarifier geometry. `model.py` implements kinetics, recycle mixing, layered settling, steady-state solving, and physical/stability audits.

Production uses twenty components, five reactor stages, and ten clarifier layers. A mechanistic response contains 170 coordinates. Solvers do not clip concentrations or reaction rates.

Tests: `test_mechanistic_model.py` and numerical/symbolic parity tests in `test_mechanistic_optimization.py`.
