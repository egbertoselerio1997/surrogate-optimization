# Scientific validation

`physical.py` reduces responses and records mass, sign, domain, and projection checks. `assessment.py` evaluates raw/projected holdout responses through resumable batches. `reference.py` replays selected decisions on the exact nonsmooth model from two starts and records casewise comparisons.

Mass-conservation tolerance is 1e-6. Engineering safeguards cover underflow TSS, feed TSS, and positive external solids loss. SRT, SOR, and SLR are descriptive. Scientific gate failures are recorded; malformed inputs, undefined required scales, and integrity failures stop execution.

Tests: `test_assessment.py`, `test_engineering_eligibility.py`, and `test_optimization_pipeline.py`.
