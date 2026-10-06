"""Repository-owned configuration and result locations."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARAMETERS_PATH = REPOSITORY_ROOT / "config" / "parameters.json"
RESULTS_ROOT = REPOSITORY_ROOT / "results"
if not PARAMETERS_PATH.is_file() or not (REPOSITORY_ROOT / "pyproject.toml").is_file():
    raise RuntimeError("use an editable installation of this repository")
