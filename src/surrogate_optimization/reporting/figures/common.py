"""Atomic figure exports and inputs shared by the reporting families."""

from __future__ import annotations
from io import BytesIO
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from surrogate_optimization.runtime.artifacts import atomic_bytes


def export_figure(figure: plt.Figure, path: Path, **kwargs: object) -> None:
    stream = BytesIO()
    figure.savefig(stream, format=path.suffix.lstrip("."), **kwargs)
    atomic_bytes(path, stream.getvalue())


def unavailable_figure(
    output: Path, stem: str, formats: tuple[str, ...], reason: str
) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.axis("off")
    axis.text(
        0.5,
        0.6,
        stem.replace("_", " ").capitalize(),
        ha="center",
        transform=axis.transAxes,
    )
    axis.text(
        0.5,
        0.4,
        f"Data unavailable\n{reason}",
        ha="center",
        va="center",
        wrap=True,
        transform=axis.transAxes,
    )
    for extension in formats:
        export_figure(figure, output / f"{stem}.{extension}", bbox_inches="tight")
    plt.close(figure)


def paired_objectives(run: Path) -> pd.DataFrame:
    table = pd.read_csv(run / "report/tables/objective_decomposition.csv")
    columns = (
        "quality",
        "hrt",
        "aeration",
        "internal_recycle",
        "return_sludge",
        "wasting",
    )
    selected = table[
        table["response_method"].eq("reference")
        & table["available"].astype(str).str.lower().eq("true")
    ]
    result = selected.pivot(index="case", columns="route", values=list(columns))
    expected = pd.MultiIndex.from_product((columns, ("surrogate", "mechanistic")))
    result = result.reindex(columns=expected).dropna()
    return pd.DataFrame(
        {
            f"{route}_component_{component}": result[component, route]
            for component, route in expected
        }
    ).reset_index()
