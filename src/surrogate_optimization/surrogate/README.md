# Statistical surrogate and projection

`regression.py` supplies the 406-feature quadratic ridge response and the log-overflow-TSS closure. `training.py` selects both penalties with development-only five-fold cross-validation and persists validated models. `projection.py` builds the physical network operators and cold OSQP projection with independent multiplier audits.

The 161-coordinate response contains mixer/reactor states, clarifier outlet component flows, and aggregate solids inventory. Clarifier layers remain in exact mechanistic states. Holdout rows do not enter fitting, scaling, or selection.

Tests: `test_projection.py`, `test_assessment.py`, and `test_run_contract.py`.
