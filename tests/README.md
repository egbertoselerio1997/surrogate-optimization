# Tests

Run all checks with:

```powershell
uv run python -m unittest discover -s tests -t . -v
```

For unit tests only, use `-s tests/unit`. `unit` covers current numerical contracts, orchestration, package paths, resumption, corruption, and reporting. `integration` executes a real 80/20-candidate workload with ten clarifier layers and both routes across all eleven cases. `support` holds fixtures.

The integration test can take substantially longer than the unit suite. It writes to `results/validation_refactor_<timestamp>`; set `SURROGATE_OPTIMIZATION_VALIDATION_RUN_ID` to reuse an unchanged validation run. Reusing a completed run checks resumption. Scientific quality gates may fail for this small sample; tests require truthful status and complete artifacts.

No full 10000-candidate production run or complete historical-output comparison is part of these checks. See [refactor record](REFACTORING.md).
