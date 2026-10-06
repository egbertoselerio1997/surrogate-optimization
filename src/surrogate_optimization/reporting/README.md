# Reporting

`tables.py` reads current artifacts and builds scientific tables, including failures and unavailable cases. `timing.py` aggregates optimization times across the ten robustness cases. `finalization.py` binds reporting source and validated numerical artifacts, generates figures, and publishes `complete.json`.

Outputs are `report/tables`, `report/figures`, and `report/manifest.json`. Reporting changes rebuild reporting while preserving compatible numerical checkpoints. Completion does not imply every scientific gate passed.

Tests: `test_reporting.py`, reporting finalization tests, and the reduced integration test.
