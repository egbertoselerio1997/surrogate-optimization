"""Runtime artifacts."""

from __future__ import annotations
from pathlib import Path
from time import sleep
from typing import Any
from typing import Mapping
import json
import numpy as np
import os
import pandas as pd
import tempfile


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Publish atomically despite transient Windows scanner/file-lock races."""
    for attempt in range(7):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            transient = os.name == "nt" and getattr(exc, "winerror", None) in {5, 32}
            if not transient or attempt == 6:
                raise
            sleep(0.01 * 2**attempt)


def _json_ready(value: Any, *, nonfinite_to_none: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item, nonfinite_to_none=nonfinite_to_none)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(item, nonfinite_to_none=nonfinite_to_none) for item in value
        ]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist(), nonfinite_to_none=nonfinite_to_none)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            if nonfinite_to_none:
                return None
            raise ValueError("non-finite numbers are not permitted in JSON contracts")
        return result
    return value


def atomic_json(path: Path, value: Any, *, nonfinite_to_none: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                _json_ready(value, nonfinite_to_none=nonfinite_to_none),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
