"""Data design."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.plant.operating_point import ClarifierParameters
    from surrogate_optimization.plant.operating_point import OperatingPoint
    from surrogate_optimization.config import StudyProfile
from pathlib import Path
from typing import Mapping
import json
import numpy as np


def clarifier_for(profile: StudyProfile) -> ClarifierParameters:
    """Preserve the 4 m / 6000 m3 Clarifier while changing numerical layers."""
    from surrogate_optimization.validation.physical import clarifier_for_layers

    return clarifier_for_layers(profile.layer_count)


def _operating(controls: np.ndarray) -> OperatingPoint:
    from surrogate_optimization.plant.operating_point import OperatingPoint

    return OperatingPoint(*map(float, controls))


def midpoint_latin_hypercube(
    count: int, dimensions: int, seed: int
) -> tuple[np.ndarray, int, int]:
    """Generate the exact midpoint-jittered, dimension-major LHS."""
    from surrogate_optimization.data.random_design import SplitMix64

    if count < 1 or dimensions < 1:
        raise ValueError("Latin-hypercube dimensions must be positive.")
    stream = SplitMix64(seed)
    coordinates = np.empty((count, dimensions), dtype=float)
    denominator = float(1 << 53)
    upper_open = np.nextafter(1.0, 0.0)
    for dimension in range(dimensions):
        permutation = list(range(count))
        for index in range(count - 1, 0, -1):
            swap = stream.randbelow(index + 1)
            permutation[index], permutation[swap] = (
                permutation[swap],
                permutation[index],
            )
        for row in range(count):
            jitter = (float(stream.next_uint64() >> 11) + 0.5) / denominator
            coordinates[row, dimension] = min(
                upper_open, (permutation[row] + jitter) / count
            )
    return (coordinates, stream.state, stream.draw_count)


def _design_block(
    count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    from surrogate_optimization.config import DECISION_LOWER
    from surrogate_optimization.config import DECISION_UPPER
    from surrogate_optimization.plant.definitions import INFLUENT_LOWER
    from surrogate_optimization.plant.definitions import INFLUENT_UPPER

    unit, final_state, draws = midpoint_latin_hypercube(count, 27, seed)
    lower = np.concatenate((DECISION_LOWER, INFLUENT_LOWER))
    upper = np.concatenate((DECISION_UPPER, INFLUENT_UPPER))
    physical = lower + unit * (upper - lower)
    return (
        physical[:, :7],
        physical[:, 7:],
        {"seed": seed, "final_state": final_state, "draw_count": draws},
    )


def create_design(profile: StudyProfile) -> dict[str, object]:
    from surrogate_optimization.plant.definitions import INFLUENT_LOWER
    from surrogate_optimization.plant.definitions import INFLUENT_UPPER
    from surrogate_optimization.plant.definitions import N_COMPONENTS

    development = _design_block(
        profile.development_candidate_count, profile.development_seed
    )
    test = _design_block(profile.holdout_candidate_count, profile.holdout_seed)
    unit, final_state, draws = midpoint_latin_hypercube(
        profile.robustness_count, N_COMPONENTS, profile.robustness_seed
    )
    robust = INFLUENT_LOWER + unit * (INFLUENT_UPPER - INFLUENT_LOWER)
    return {
        "development_controls": development[0],
        "development_influents": development[1],
        "holdout_controls": test[0],
        "holdout_influents": test[1],
        "robustness_influents": robust,
        "generators": {
            "development": development[2],
            "holdout": test[2],
            "robustness": {
                "seed": profile.robustness_seed,
                "final_state": final_state,
                "draw_count": draws,
            },
        },
    }


def _expected_design_shapes(profile: StudyProfile) -> dict[str, tuple[int, int]]:
    return {
        "development_controls": (profile.development_candidate_count, 7),
        "development_influents": (profile.development_candidate_count, 20),
        "holdout_controls": (profile.holdout_candidate_count, 7),
        "holdout_influents": (profile.holdout_candidate_count, 20),
        "robustness_influents": (profile.robustness_count, 20),
    }


def validate_design(design: Mapping[str, object], profile: StudyProfile) -> None:
    from surrogate_optimization.config import DECISION_LOWER
    from surrogate_optimization.config import DECISION_UPPER
    from surrogate_optimization.plant.definitions import INFLUENT_LOWER
    from surrogate_optimization.plant.definitions import INFLUENT_UPPER

    for name, shape in _expected_design_shapes(profile).items():
        if name not in design:
            raise RuntimeError(f"fixed design is missing {name}")
        value = np.asarray(design[name], dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise RuntimeError(f"fixed design {name} has invalid shape or values")
        lower, upper = (
            (DECISION_LOWER, DECISION_UPPER)
            if name.endswith("controls")
            else (INFLUENT_LOWER, INFLUENT_UPPER)
        )
        if np.any(value < lower) or np.any(value > upper):
            raise RuntimeError(f"fixed design {name} lies outside its declared box")
    generators = design.get("generators")
    if not isinstance(generators, Mapping):
        raise RuntimeError("fixed design is missing generator records")
    expected_seeds = {
        "development": profile.development_seed,
        "holdout": profile.holdout_seed,
        "robustness": profile.robustness_seed,
    }
    for block, seed in expected_seeds.items():
        record = generators.get(block)
        if not isinstance(record, Mapping) or int(record.get("seed", -1)) != seed:
            raise RuntimeError(f"fixed design has an invalid {block} generator record")


def _design_digest(design: Mapping[str, object]) -> str:
    from surrogate_optimization.runtime.contracts import array_digest
    from surrogate_optimization.runtime.protocols import DESIGN_ARRAYS

    return array_digest(
        **{name: np.asarray(design[name], dtype="<f8") for name in DESIGN_ARRAYS}
    )


def load_or_create_design(run: Path, profile: StudyProfile) -> dict[str, object]:
    from surrogate_optimization.runtime.artifacts import _json_ready
    from surrogate_optimization.runtime.artifacts import atomic_json
    from surrogate_optimization.runtime.artifacts import atomic_npz
    from surrogate_optimization.runtime.protocols import DESIGN_ARRAYS

    expected = create_design(profile)
    validate_design(expected, profile)
    path = run / "datasets" / "design.npz"
    records_path = run / "inputs" / "generator_records.json"
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(DESIGN_ARRAYS):
                raise RuntimeError("existing design checkpoint has unexpected arrays")
            for name in DESIGN_ARRAYS:
                if not np.array_equal(stored[name], np.asarray(expected[name])):
                    raise RuntimeError(
                        "existing design checkpoint differs from fixed design"
                    )
    else:
        atomic_npz(path, **{name: np.asarray(expected[name]) for name in DESIGN_ARRAYS})
    expected_records = _json_ready(expected["generators"])
    if records_path.is_file():
        existing_records = json.loads(records_path.read_text(encoding="utf-8"))
        if existing_records != expected_records:
            raise RuntimeError("existing generator record differs from fixed design")
    else:
        atomic_json(records_path, expected_records)
    return expected
