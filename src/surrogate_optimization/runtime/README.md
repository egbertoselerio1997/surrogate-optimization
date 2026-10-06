# Execution integrity

`artifacts.py` publishes JSON, NPZ, CSV, and bytes atomically with Windows sharing-lock retries. `contracts.py` binds parameters, source, inputs, and artifact hashes. `checkpoints.py` resumes individual route attempts. `parallel.py` executes deterministic resumable batches. `protocols.py` names current formats and algorithms.

Only runs created by this implementation resume. A computation source change requires a new run. Reporting source is bound separately. Completed corrupt artifacts are rejected. No historical migration or predecessor-source authorization is provided.

Tests: `test_atomic_artifacts.py`, `test_package_and_paths.py`, `test_run_contract.py`, and `test_parallel_execution.py`.
