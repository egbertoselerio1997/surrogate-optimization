# Unit tests

These tests preserve meaningful scientific assertions and exercise current interfaces. They cover deterministic sampling, physical equations, ridge/projection checks, independent derivatives, native solver parity, Windows spawning, immutable checkpoints, atomic writes, case failures, and reporting.

Run `uv run python -m unittest discover -s tests/unit -t . -v`.
