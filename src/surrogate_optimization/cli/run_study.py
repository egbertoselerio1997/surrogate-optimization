"""Command-line entry point for the fixed production workload."""

from __future__ import annotations
import argparse
import os
from surrogate_optimization.runtime.protocols import DEFAULT_RUN_ID
from surrogate_optimization.workflow.study import run_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=os.environ.get("SURROGATE_OPTIMIZATION_RUN_ID", DEFAULT_RUN_ID),
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        choices=(10000,),
        default=int(os.environ.get("SURROGATE_OPTIMIZATION_CANDIDATE_COUNT", "10000")),
        help="attempted development and holdout candidates (8000/2000)",
    )
    parser.add_argument(
        "--through",
        choices=("generation", "assessment", "complete"),
        default="complete",
    )
    args = parser.parse_args(argv)
    if args.candidate_count != 10000:
        parser.error("production requires exactly 10000 attempted candidates")
    summary = run_study(args.run_id, args.through)
    print(f"{summary.status}: {summary.result_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
