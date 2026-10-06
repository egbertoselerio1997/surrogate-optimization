"""Deterministic physical scales for expression-level numerical checks."""

import numpy as np
from surrogate_optimization.plant.operating_point import ClarifierParameters
from surrogate_optimization.plant.definitions import INFLUENT_LOWER, INFLUENT_UPPER
from surrogate_optimization.optimization.mechanistic import (
    MechanisticAssets,
    SmoothScales,
    DECISION_LOWER,
    DECISION_UPPER,
)


def clarifier_parameters(layer_count: int) -> ClarifierParameters:
    return ClarifierParameters(
        layer_count=layer_count,
        feed_layer=(layer_count - 1) // 2,
        layer_volume=6000.0 / layer_count,
    )


def mechanistic_assets(layer_count: int) -> MechanisticAssets:
    state_count = 100 + layer_count
    return MechanisticAssets(
        clarifier=clarifier_parameters(layer_count),
        smoothing=SmoothScales(
            10.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 250.0, 10000.0
        ),
        state_center=np.ones(state_count),
        state_scale=np.ones(state_count),
        feed_scale=100.0,
        balance_scale=np.ones(state_count),
        quality_scale=np.ones(4),
        envelope_scale=np.ones(2 * (layer_count - 2)),
        engineering_scale=np.ones(2),
        decision_center=(DECISION_LOWER + DECISION_UPPER) / 2.0,
        decision_scale=(DECISION_UPPER - DECISION_LOWER) / np.sqrt(12.0),
        influent_center=(INFLUENT_LOWER + INFLUENT_UPPER) / 2.0,
        influent_scale=(INFLUENT_UPPER - INFLUENT_LOWER) / np.sqrt(12.0),
    )
