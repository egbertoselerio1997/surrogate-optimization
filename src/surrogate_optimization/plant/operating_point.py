"""Plant operating_point."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    """Seven-control operating point specified by the model."""

    hrt_hours: float
    aeration_3: float
    aeration_4: float
    aeration_5: float
    internal_recycle: float
    return_sludge: float
    waste_sludge: float

    def __post_init__(self) -> None:
        values = np.asarray(self.as_array(), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Operating controls must be finite.")
        if self.hrt_hours <= 0.0:
            raise ValueError("hrt_hours must be positive.")
        if np.any((values[1:4] < 0.0) | (values[1:4] > 1.0)):
            raise ValueError("Stage aeration settings must lie in [0, 1].")
        if self.internal_recycle < 0.0 or self.return_sludge <= 0.0:
            raise ValueError("Recycle ratios must satisfy r_I >= 0 and r_R > 0.")
        if not 0.0 <= self.waste_sludge < 1.0:
            raise ValueError("waste_sludge must lie in [0, 1).")

    def as_array(self) -> FloatArray:
        return np.asarray(
            [
                self.hrt_hours,
                self.aeration_3,
                self.aeration_4,
                self.aeration_5,
                self.internal_recycle,
                self.return_sludge,
                self.waste_sludge,
            ],
            dtype=float,
        )

    def aeration_for_stage(self, stage: int) -> float:
        from surrogate_optimization.plant.definitions import N_STAGES

        if stage not in range(N_STAGES):
            raise ValueError("stage must be in range(5).")
        return (
            0.0
            if stage < 2
            else (self.aeration_3, self.aeration_4, self.aeration_5)[stage - 2]
        )

    @property
    def q_process(self) -> float:
        return 1.0 + self.internal_recycle + self.return_sludge

    @property
    def q_clarifier(self) -> float:
        return 1.0 + self.return_sludge

    @property
    def q_underflow(self) -> float:
        return self.return_sludge + self.waste_sludge

    @property
    def q_effluent(self) -> float:
        return 1.0 - self.waste_sludge

    @property
    def stage_dilution_rate(self) -> float:
        return 120.0 * self.q_process / self.hrt_hours


@dataclass(frozen=True)
class ClarifierParameters:
    fresh_flow: float = 10000.0
    area: float = 1500.0
    layer_volume: float = 600.0
    maximum_settling_velocity: float = 250.0
    theoretical_settling_velocity: float = 474.0
    hindered_coefficient: float = 0.000576
    low_concentration_coefficient: float = 0.00286
    nonsettleable_fraction: float = 0.00228
    flux_threshold: float = 3000.0
    layer_count: int = 10
    feed_layer: int = 4

    def __post_init__(self) -> None:
        if self.layer_count < 3:
            raise ValueError("A Clarifier requires at least three layers.")
        if self.feed_layer not in range(self.layer_count):
            raise ValueError("feed_layer must identify an existing Clarifier layer.")
        positive = (
            self.fresh_flow,
            self.area,
            self.layer_volume,
            self.maximum_settling_velocity,
            self.theoretical_settling_velocity,
            self.hindered_coefficient,
            self.low_concentration_coefficient,
            self.flux_threshold,
        )
        if any((not np.isfinite(value) or value <= 0.0 for value in positive)):
            raise ValueError(
                "Clarifier geometry and settling parameters must be positive and finite."
            )
        if not 0.0 <= self.nonsettleable_fraction <= 1.0:
            raise ValueError("nonsettleable_fraction must lie in [0, 1].")
