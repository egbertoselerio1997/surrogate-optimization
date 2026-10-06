# Study execution

`study.run_study(run_id, through="complete")` executes the fixed production profile and returns a `RunSummary`. Generation preserves candidate attempts and creates the accepted effective design. Assessment fits models and freezes diagnostics. Optimization attempts both routes in eleven cases, certifies endpoints, and performs exact replay. Reporting then writes tables, figures, and final manifests.

All paths derive from the repository-owned run directory. Completed stages are reused after integrity checks. Execution status and scientific validation status are separate. `_execute_pipeline` accepts internal test profiles; the public API validates the full production profile.

Tests: `test_run_contract.py`, `test_optimization_pipeline.py`, and the real reduced integration test.
