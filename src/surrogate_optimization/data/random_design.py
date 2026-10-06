"""Data random_design."""

from __future__ import annotations
from surrogate_optimization.plant.definitions import COMPONENTS
from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.float64]
UINT64_MODULUS = 1 << 64
UINT64_MASK = UINT64_MODULUS - 1
SPLITMIX64_INCREMENT = 11400714819323198485
SPLITMIX64_MULTIPLIER_1 = 13787848793156543929
SPLITMIX64_MULTIPLIER_2 = 10723151780598845931
FLOAT53_DENOMINATOR = 1 << 53
FLOAT52_DENOMINATOR = 1 << 52
DECISION_COLUMNS: tuple[str, ...] = ("H", "a", "r_I", "r_R", "w")
INFLUENT_COLUMNS: tuple[str, ...] = tuple(COMPONENTS)
TRAINING_COLUMNS: tuple[str, ...] = DECISION_COLUMNS + INFLUENT_COLUMNS
ROBUSTNESS_COLUMNS: tuple[str, ...] = INFLUENT_COLUMNS
DECISION_BOUNDS: dict[str, tuple[float, float]] = {
    "H": (6.0, 36.0),
    "a": (0.0, 1.0),
    "r_I": (0.0, 4.0),
    "r_R": (0.25, 1.25),
    "w": (0.001, 0.05),
}
INFLUENT_BOUNDS: dict[str, tuple[float, float]] = {
    "S_O": (0.0, 0.5),
    "S_F": (20.0, 180.0),
    "S_A": (5.0, 80.0),
    "S_NH4": (12.0, 55.0),
    "S_NO2": (0.0, 3.0),
    "S_NO3": (0.0, 8.0),
    "S_N2": (0.0, 2.0),
    "S_PO4": (2.0, 18.0),
    "S_I": (10.0, 90.0),
    "S_ALK": (1.6, 5.2),
    "X_I": (20.0, 120.0),
    "X_S": (60.0, 280.0),
    "X_H": (15.0, 100.0),
    "X_PAO": (5.0, 60.0),
    "X_PP": (2.0, 20.0),
    "X_PHA": (1.0, 30.0),
    "X_AOB": (0.5, 8.0),
    "X_NOB": (0.5, 8.0),
    "X_MeP": (0.0, 12.0),
    "X_MeOH": (0.0, 12.0),
}
TRAINING_BOUNDS: dict[str, tuple[float, float]] = {**DECISION_BOUNDS, **INFLUENT_BOUNDS}


def _unsigned_64(value: int, *, label: str) -> int:
    """Validate rather than silently wrap a declared unsigned 64-bit value."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer.")
    converted = int(value)
    if converted < 0 or converted > UINT64_MASK:
        raise ValueError(f"{label} must lie in [0, 2^64 - 1].")
    return converted


class SplitMix64:
    """Minimal SplitMix64 stream with observable state and draw count."""

    def __init__(self, seed: int) -> None:
        self._state = _unsigned_64(seed, label="seed")
        self._draw_count = 0

    @property
    def state(self) -> int:
        """Current unsigned 64-bit state, after all consumed words."""
        return self._state

    @property
    def draw_count(self) -> int:
        """Number of generated words, including rejection-sampling draws."""
        return self._draw_count

    def next_uint64(self) -> int:
        """Return the next word using arithmetic modulo ``2^64``."""
        self._state = self._state + SPLITMIX64_INCREMENT & UINT64_MASK
        value = self._state
        value = (value ^ value >> 30) * SPLITMIX64_MULTIPLIER_1 & UINT64_MASK
        value = (value ^ value >> 27) * SPLITMIX64_MULTIPLIER_2 & UINT64_MASK
        value ^= value >> 31
        self._draw_count += 1
        return value & UINT64_MASK

    def random_float53(self) -> float:
        """Return ``(U >> 11) / 2^53``, which lies in ``[0, 1)``."""
        return float(self.next_uint64() >> 11) / float(FLOAT53_DENOMINATOR)

    def random_open_float52(self) -> float:
        """Return the exact binary64 midpoint ``((U >> 12)+1/2)/2^52``."""
        return (float(self.next_uint64() >> 12) + 0.5) / float(FLOAT52_DENOMINATOR)

    def randbelow(self, upper: int) -> int:
        """Draw uniformly from ``range(upper)`` without modulo bias."""
        if isinstance(upper, bool) or not isinstance(upper, (int, np.integer)):
            raise TypeError("upper must be an integer.")
        bound = int(upper)
        if bound <= 0 or bound > UINT64_MODULUS:
            raise ValueError("upper must lie in [1, 2^64].")
        limit = UINT64_MODULUS - UINT64_MODULUS % bound
        while True:
            word = self.next_uint64()
            if word < limit:
                return word % bound
