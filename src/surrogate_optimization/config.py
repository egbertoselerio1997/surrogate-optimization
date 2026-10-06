"""Config."""

from __future__ import annotations
from surrogate_optimization.paths import REPOSITORY_ROOT
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import os

PARAMETERS_PATH = REPOSITORY_ROOT / "config" / "parameters.json"


def load_parameters(path: str | Path = PARAMETERS_PATH) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read study parameters: {source}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("study parameters must be a JSON object")
    if value.get("schema_version") != 6:
        raise RuntimeError("unsupported study-parameter schema")
    execution = value.get("execution")
    profiles = value.get("profiles")
    generation = value.get("mechanistic_generation")
    engineering = value.get("engineering")
    reporting = value.get("reporting")
    if not all(
        (
            isinstance(section, dict)
            for section in (execution, profiles, generation, engineering, reporting)
        )
    ):
        raise RuntimeError("study parameters omit a required section")
    default_profile = execution.get("default_profile")
    if default_profile != "production" or default_profile not in profiles:
        raise RuntimeError("the default profile must be production")
    profile = profiles[default_profile]
    if not isinstance(profile, dict):
        raise RuntimeError("the default profile must be a JSON object")
    if (
        profile.get("development_candidate_count"),
        profile.get("holdout_candidate_count"),
    ) != (8000, 2000):
        raise RuntimeError("the study profile must contain 8000/2000 LHS candidates")
    if profile.get("counts_are_candidate_rows") is not True:
        raise RuntimeError("study profile counts must denote attempted LHS candidates")
    if profile.get("replace_rejected_mechanistic_candidates") is not False:
        raise RuntimeError("the study profile must not replace rejected candidates")
    if generation.get("balance_tolerance") != 1e-06:
        raise RuntimeError("physical mass-conservation tolerance must be 1e-6")
    prohibited = {
        "srt_min_d",
        "srt_max_d",
        "sor_max_m_d",
        "sor_report_only_max_m_d",
        "slr_max_kg_m2_d",
    }
    present = prohibited.intersection(engineering)
    if present:
        raise RuntimeError(
            f"prohibited engineering guardrails remain: {sorted(present)}"
        )
    if reporting.get("timing_protocol") != "optimization_time":
        raise RuntimeError("unsupported optimization timing protocol")
    if (
        reporting.get("timing_metric") != "Optimization time"
        or reporting.get("timing_unit") != "s"
    ):
        raise RuntimeError("the timing metric must be Optimization time in seconds")
    return value


def production_profile_parameters() -> dict[str, Any]:
    parameters = load_parameters()
    profile_name = parameters["execution"]["default_profile"]
    return dict(parameters["profiles"][profile_name])


def physical_balance_tolerance() -> float:
    return float(load_parameters()["mechanistic_generation"]["balance_tolerance"])


def engineering_parameters() -> dict[str, Any]:
    return dict(load_parameters()["engineering"])


_PARAMETERS = load_parameters()
_MECHANISTIC_GENERATION = _PARAMETERS["mechanistic_generation"]
DECISION_NAMES = ("H", "a_3", "a_4", "a_5", "r_I", "r_R", "w")
DECISION_LOWER = np.asarray([6.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.001])
DECISION_UPPER = np.asarray([36.0, 1.0, 1.0, 1.0, 4.0, 1.25, 0.05])
RIDGE_GRID = np.asarray(_PARAMETERS["surrogate"]["ridge_grid"], dtype=float)
OVERFLOW_TSS_LOW_QUANTILE = float(
    _PARAMETERS["surrogate"]["overflow_tss_closure"]["reported_strata"][
        "low_tss_quantile"
    ]
)
OVERFLOW_TSS_HIGH_QUANTILE = float(
    _PARAMETERS["surrogate"]["overflow_tss_closure"]["reported_strata"][
        "upper_tail_quantile"
    ]
)


@dataclass(frozen=True)
class StudyProfile:
    name: str
    development_candidate_count: int
    holdout_candidate_count: int
    robustness_count: int
    layer_count: int
    development_seed: int
    holdout_seed: int
    robustness_seed: int
    parallel_workers: int
    scientifically_eligible: bool
    enforce_admission_gate: bool = True

    @property
    def response_count(self) -> int:
        """Full layer-resolved mechanistic response width (checkpoint format)."""
        return self.mechanistic_response_count

    @property
    def mechanistic_response_count(self) -> int:
        from surrogate_optimization.plant.definitions import N_COMPONENTS
        from surrogate_optimization.plant.definitions import N_STAGES

        return (N_STAGES + 3) * N_COMPONENTS + self.layer_count

    @property
    def surrogate_response_count(self) -> int:
        """Reduced operational-response width, independent of layer count."""
        from surrogate_optimization.plant.definitions import N_COMPONENTS
        from surrogate_optimization.plant.definitions import N_STAGES

        return (N_STAGES + 3) * N_COMPONENTS + 1


PRODUCTION_PROFILE = StudyProfile(
    name="production",
    development_candidate_count=8000,
    holdout_candidate_count=2000,
    robustness_count=10,
    layer_count=10,
    development_seed=100042,
    holdout_seed=100043,
    robustness_seed=314159,
    parallel_workers=max(1, min(12, (os.cpu_count() or 2) - 1)),
    scientifically_eligible=True,
    enforce_admission_gate=True,
)
