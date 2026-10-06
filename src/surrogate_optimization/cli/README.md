# Command line

Run `python -m surrogate_optimization.cli.run_study --run-id production_001 --through complete`. The only supported production candidate count is 10000. Terminal stages are `generation`, `assessment`, and `complete`.

The command delegates to `workflow.study.run_study` and prints the final status and result directory. Environment defaults are `SURROGATE_OPTIMIZATION_RUN_ID` and `SURROGATE_OPTIMIZATION_CANDIDATE_COUNT`.

Tests: `tests/unit/test_package_and_paths.py`.
