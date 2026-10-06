"""Atomic headless export for scientific figures."""

from __future__ import annotations
from io import BytesIO
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from surrogate_optimization.runtime.artifacts import atomic_bytes


def export_figure(figure: plt.Figure, path: Path, **kwargs: object) -> None:
    stream = BytesIO()
    figure.savefig(stream, format=path.suffix.lstrip("."), **kwargs)
    atomic_bytes(path, stream.getvalue())
