# Reduced integration workload

`test_reduced_workload.py` runs 80 development and 20 holdout attempts through the real pipeline with unchanged physical dimensions, seeds, solver settings, and all eleven optimization cases. It checks artifact placement, manifests, case coverage, truthful scientific status, and completed-run resumption.

Run `uv run python -m unittest tests.integration.test_reduced_workload -v`. Outputs remain under the repository results folder. This is not a full production-scale equivalence test.

When the reduced sample does not provide every required chart response, the test
checks that all numerical cases finished, reporting records the missing inputs,
no completion marker is published, and resumption preserves completed searches.
