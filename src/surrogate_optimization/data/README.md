# Candidate data

`random_design.py` implements SplitMix64. `design.py` constructs midpoint-jittered Latin hypercubes and fixed influent scenarios. `generation.py` solves each candidate from two starts and records attempts, accepted responses, and original-row provenance.

Production attempts 8000 development and 2000 holdout candidates. Rejections are retained without replacement. Seeds are 100042, 100043, and 314159 for development, holdout, and robustness. Workers are top-level importable functions for Windows process spawning.

Tests: `test_sampling.py`, `test_candidate_generation.py`, and `test_run_contract.py`.
