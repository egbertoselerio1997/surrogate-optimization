"""Workflow types."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.validation.assessment import AssessmentResult
    from surrogate_optimization.surrogate.regression import LogOverflowTSSClosure
    from surrogate_optimization.surrogate.regression import QuadraticSurrogate
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class AnalysisBundle:
    passed: bool
    model: QuadraticSurrogate
    mechanistic_assets: Any
    surrogate_assets: Any
    assessment: AssessmentResult | None
    gate: dict[str, Any]
    overflow_closure: LogOverflowTSSClosure | None = None


@dataclass(frozen=True)
class GenerationResult:
    """Accepted development/test mechanistic_responses and their effective input design."""

    design: dict[str, object]
    development_responses: np.ndarray
    holdout_responses: np.ndarray

    def __iter__(self):
        """Preserve the historical two-target unpacking interface."""
        yield self.development_responses
        yield self.holdout_responses

    def __getitem__(self, index: int) -> np.ndarray:
        return (self.development_responses, self.holdout_responses)[index]
