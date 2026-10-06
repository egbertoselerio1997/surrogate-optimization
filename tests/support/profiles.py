"""Small fixtures, never exposed by the production CLI."""

from dataclasses import replace
from surrogate_optimization.config import PRODUCTION_PROFILE

REDUCED_PROFILE = replace(
    PRODUCTION_PROFILE,
    name="validation",
    development_candidate_count=400,
    holdout_candidate_count=100,
    layer_count=5,
    robustness_count=5,
    development_seed=500042,
    holdout_seed=500043,
    robustness_seed=500314159,
    parallel_workers=1,
    scientifically_eligible=False,
    enforce_admission_gate=False,
)
INTEGRATION_PROFILE = replace(
    PRODUCTION_PROFILE,
    name="integration",
    development_candidate_count=80,
    holdout_candidate_count=20,
    parallel_workers=4,
    scientifically_eligible=False,
    enforce_admission_gate=False,
)
