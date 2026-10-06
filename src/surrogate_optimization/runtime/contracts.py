"""Runtime contracts."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.config import StudyProfile
from surrogate_optimization.paths import RESULTS_ROOT
from dataclasses import asdict
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path
from typing import Any
from typing import Mapping
import json
import numpy as np
import platform


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_digest(files: Mapping[str, str] | None = None) -> str:
    manifest = dict(files or source_file_digests())
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def array_digest(**arrays: np.ndarray) -> str:
    digest = sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _artifact_hashes(run: Path, paths: tuple[Path, ...]) -> dict[str, str]:
    return {path.relative_to(run).as_posix(): file_digest(path) for path in paths}


def _artifacts_match(run: Path, expected: Mapping[str, str]) -> bool:
    if not isinstance(expected, Mapping) or not expected:
        return False
    root = run.resolve()
    for relative, expected_digest in expected.items():
        if not isinstance(relative, str) or Path(relative).is_absolute():
            return False
        path = (root / relative).resolve()
        if (
            root not in path.parents
            or not path.is_file()
            or file_digest(path) != expected_digest
        ):
            return False
    return True


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "pandas", "casadi", "osqp"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def profile_for_candidate_count(candidate_count: int) -> StudyProfile:
    """Return the configured fresh 80/20 study profile."""
    from surrogate_optimization.config import PRODUCTION_PROFILE
    from surrogate_optimization.config import production_profile_parameters
    from surrogate_optimization.runtime.protocols import AUTHORIZED_DATASET_TOTALS

    if candidate_count not in AUTHORIZED_DATASET_TOTALS:
        raise ValueError(
            f"dataset total must be one of {AUTHORIZED_DATASET_TOTALS}, not {candidate_count}"
        )
    configured = production_profile_parameters()
    return replace(
        PRODUCTION_PROFILE,
        development_candidate_count=int(configured["development_candidate_count"]),
        holdout_candidate_count=int(configured["holdout_candidate_count"]),
        robustness_count=int(configured["robustness_count"]),
        layer_count=int(configured["clarifier_layer_count"]),
        development_seed=int(configured["development_seed"]),
        holdout_seed=int(configured["holdout_seed"]),
        robustness_seed=int(configured["robustness_seed"]),
        scientifically_eligible=bool(configured["scientifically_eligible"]),
        enforce_admission_gate=bool(
            configured["scientific_admission_gate_enforced_for_study_eligibility"]
        ),
    )


def validate_production_profile(profile: StudyProfile) -> None:
    from surrogate_optimization.runtime.protocols import AUTHORIZED_DATASET_TOTALS

    candidate_count = (
        profile.development_candidate_count + profile.holdout_candidate_count
    )
    if candidate_count not in AUTHORIZED_DATASET_TOTALS:
        raise RuntimeError(
            f"study profile requests unauthorized dataset total {candidate_count}"
        )
    expected = {
        "name": "production",
        "development_candidate_count": 8000,
        "holdout_candidate_count": 2000,
        "robustness_count": 10,
        "layer_count": 10,
        "development_seed": 100042,
        "holdout_seed": 100043,
        "robustness_seed": 314159,
        "scientifically_eligible": True,
        "enforce_admission_gate": True,
    }
    actual = asdict(profile)
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"study profile violates the authorized contract: {mismatches}"
        )


def _build_contract(
    run_id: str, profile: StudyProfile, source_files: Mapping[str, str]
) -> dict[str, Any]:
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES
    from surrogate_optimization.runtime.protocols import (
        ASSESSMENT_GATE_EXECUTION_POLICY,
    )
    from surrogate_optimization.runtime.protocols import CHECKPOINT_SCHEMA_VERSION
    from surrogate_optimization.runtime.protocols import COMPARISON_PROTOCOL
    from surrogate_optimization.runtime.protocols import OPTIMIZATION_PROTOCOL
    from surrogate_optimization.runtime.protocols import PROJECTION_SCHEMA
    from surrogate_optimization.runtime.protocols import RESPONSE_SCHEMA
    from surrogate_optimization.runtime.protocols import TIMING_PROTOCOL

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "profile": asdict(profile),
        "fixed_dataset_total": profile.development_candidate_count
        + profile.holdout_candidate_count,
        "development_holdout_split": [
            profile.development_candidate_count,
            profile.holdout_candidate_count,
        ],
        "source_digest": source_digest(source_files),
        "source_files": dict(source_files),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime_versions": _runtime_versions(),
        "assessment_gate_execution_policy": ASSESSMENT_GATE_EXECUTION_POLICY,
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "validation_protocol": COMPARISON_PROTOCOL,
        "timing_protocol": TIMING_PROTOCOL,
        "response_schema": {
            "name": RESPONSE_SCHEMA,
            "mechanistic_response_count": profile.mechanistic_response_count,
            "surrogate_response_count": profile.surrogate_response_count,
            "shared_coordinate_count": (N_STAGES + 3) * N_COMPONENTS,
            "clarifier_inventory_formula": "sum(layer_volume_m3 * layer_tss_g_m3)",
            "clarifier_volume_m3": 6000.0,
            "holdout_role": "frozen_post_selection_descriptive",
        },
        "projection_schema": PROJECTION_SCHEMA,
        "dataset_protocol": "fixed_lhs_candidates_with_accepted_subset",
        "preflight_artifacts_permitted": False,
        "full_run_admission_gate_bypass_permitted": False,
    }


def _canonical_json_digest(value: Any) -> str:
    from surrogate_optimization.runtime.artifacts import _json_ready

    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def assert_source_unchanged(expected: Mapping[str, str]) -> None:
    current = source_file_digests()
    if current != dict(expected):
        changed = sorted(set(current) | set(expected))
        changed = [name for name in changed if current.get(name) != expected.get(name)]
        raise RuntimeError(
            f"study source changed during the run; do not mix artifacts. Changed files: {', '.join(changed)}"
        )


def establish_contract(run: Path, contract: Mapping[str, Any]) -> None:
    from surrogate_optimization.runtime.artifacts import _json_ready
    from surrogate_optimization.runtime.artifacts import atomic_json

    path = run / "run_contract.json"
    normalized = _json_ready(contract)
    if path.exists():
        if _load_json_object(path, description="run contract") != normalized:
            raise RuntimeError("existing run contract differs; choose a new run id")
        return
    if any(run.iterdir()):
        raise RuntimeError("nonempty result directory has no compatible run contract")
    atomic_json(path, normalized)


def _checkpoint_source_is_authorized(
    run: Path,
    *,
    stage: str,
    checkpoint: Path,
    observed_source_id: str,
    current_source_id: str,
) -> bool:
    return observed_source_id == current_source_id


def _casewise_artifact_source_id(run: Path, current_source_id: str) -> str:
    return current_source_id


def resolve_run_directory(run_id: str, results_root: Path = RESULTS_ROOT) -> Path:
    from surrogate_optimization.runtime.protocols import RUN_ID_PATTERN

    if not RUN_ID_PATTERN.fullmatch(run_id) or run_id.upper().split(".")[0] in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise ValueError("run id must be a safe identifier")
    root = results_root.resolve()
    run = (root / run_id).resolve()
    if run.parent != root:
        raise ValueError("run directory escapes results root")
    return run


def source_file_digests() -> dict[str, str]:
    from surrogate_optimization.runtime.protocols import ROOT

    paths = sorted(
        (
            path
            for path in (ROOT / "src" / "surrogate_optimization").rglob("*.py")
            if "reporting"
            not in path.relative_to(ROOT / "src" / "surrogate_optimization").parts
        )
    )
    paths += [
        ROOT / "config" / "parameters.json",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    return {path.relative_to(ROOT).as_posix(): file_digest(path) for path in paths}
