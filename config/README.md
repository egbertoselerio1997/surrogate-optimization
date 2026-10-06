# Parameters

`parameters.json` is the authoritative production configuration. It records the fixed candidate counts and seeds, physical dimensions, regression/projection settings, solver limits, engineering safeguards, objective weights, and reporting conventions.

The package validates this file on load. Changing it changes the computation contract and requires a new run ID. The workbook is preserved in `data`, but the current implementation defines its physical constants in Python and does not read the workbook.
