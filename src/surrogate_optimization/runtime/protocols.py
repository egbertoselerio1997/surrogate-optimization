"""Runtime protocols."""

from __future__ import annotations
from surrogate_optimization.paths import REPOSITORY_ROOT
import re

ROOT = REPOSITORY_ROOT
DEFAULT_RUN_ID = "production_001"
CHECKPOINT_SCHEMA_VERSION = 13
RESPONSE_SCHEMA = "clarifier_inventory"
PROJECTION_SCHEMA = "system_wide_log_overflow_closure"
ASSESSMENT_GATE_EXECUTION_POLICY = "advisory_continue"
MECHANISTIC_SINGLE_CENTER_PROTOCOL = "smooth_mechanistic_single_center"
OPTIMIZATION_PROTOCOL = "single_center_local_exact_qp_no_minimum_srt"
COMPARISON_PROTOCOL = "casewise_exact_common_reference_no_minimum_srt"
TIMING_PROTOCOL = "primary_route_time"
RUN_ID_PATTERN = re.compile("^[A-Za-z0-9][A-Za-z0-9_-]*$")
AUTHORIZED_DATASET_TOTALS = (10000,)
DESIGN_ARRAYS = (
    "development_controls",
    "development_influents",
    "holdout_controls",
    "holdout_influents",
    "robustness_influents",
)
