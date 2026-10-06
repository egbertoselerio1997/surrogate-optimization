# Reporting

`tables.py` reads current artifacts and builds scientific tables, including failures and unavailable cases. `timing.py` aggregates optimization times across the ten robustness cases. `finalization.py` binds reporting source and validated numerical artifacts, generates the eight reference PNG figures, and publishes `complete.json`.

Outputs are `report/tables`, `report/figures`, and `report/manifest.json`. Reporting changes rebuild reporting while preserving compatible numerical checkpoints. Completion does not imply every scientific gate passed.

Tests: `test_reporting.py`, reporting finalization tests, and the reduced integration test.

The chart package is published directly in `report/figures`. Missing required
scenario data prevent reporting completion; numerical checkpoints remain reusable.

`labels.py` supplies shared chart and table vocabulary. Tables include display
labels for methods, influent scenarios, responses, and locations alongside
artifact identifiers. Their manifest records column labels and units; the
generated table README explains metric definitions and objective components.
