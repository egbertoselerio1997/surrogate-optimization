"""Data generation."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.config import StudyProfile
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from dataclasses import asdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threadpoolctl import threadpool_limits
from time import perf_counter
import json
import numpy as np
import os
import pandas as pd
import tempfile
import time


def _solve_design_row(
    payload: tuple[int, np.ndarray, np.ndarray, int],
) -> dict[str, object]:
    from surrogate_optimization.config import _MECHANISTIC_GENERATION
    from surrogate_optimization.data.design import _operating
    from surrogate_optimization.plant.model import assemble_target
    from surrogate_optimization.plant.model import branch_classification
    from surrogate_optimization.plant.model import generation_scale
    from surrogate_optimization.plant.model import solve_steady_state
    from surrogate_optimization.plant.model import unpack_state
    from surrogate_optimization.plant.operating_point import ClarifierParameters

    index, controls, influent, layer_count = payload
    clarifier = ClarifierParameters(
        layer_count=layer_count,
        feed_layer=(layer_count - 1) // 2,
        layer_volume=6000.0 / layer_count,
    )
    operating = _operating(controls)
    started = perf_counter()
    with threadpool_limits(limits=1):
        first = solve_steady_state(
            operating,
            influent,
            starts=(1,),
            clarifier=clarifier,
            max_nfev=int(_MECHANISTIC_GENERATION["polish_maximum_evaluations"]),
            tolerance=float(_MECHANISTIC_GENERATION["polish_xtol"]),
            balance_tolerance=float(_MECHANISTIC_GENERATION["balance_tolerance"]),
            minimum_relaxation_days=float(_MECHANISTIC_GENERATION["minimum_horizon_d"]),
            solids_turnovers=float(_MECHANISTIC_GENERATION["waste_turnovers"]),
            integration_rtol=float(_MECHANISTIC_GENERATION["relative_tolerance"]),
            integration_atol=float(
                _MECHANISTIC_GENERATION["dimensionless_absolute_tolerance"]
            ),
            logarithmic_only=True,
            require_physical_audit=True,
        )
        second = solve_steady_state(
            operating,
            influent,
            starts=(2,),
            clarifier=clarifier,
            max_nfev=int(_MECHANISTIC_GENERATION["polish_maximum_evaluations"]),
            tolerance=float(_MECHANISTIC_GENERATION["polish_xtol"]),
            balance_tolerance=float(_MECHANISTIC_GENERATION["balance_tolerance"]),
            minimum_relaxation_days=float(_MECHANISTIC_GENERATION["minimum_horizon_d"]),
            solids_turnovers=float(_MECHANISTIC_GENERATION["waste_turnovers"]),
            integration_rtol=float(_MECHANISTIC_GENERATION["relative_tolerance"]),
            integration_atol=float(
                _MECHANISTIC_GENERATION["dimensionless_absolute_tolerance"]
            ),
            logarithmic_only=True,
            require_physical_audit=True,
        )
    first_reactors, _ = unpack_state(first.state, clarifier)
    scale = generation_scale(influent, first_reactors[-1], clarifier)
    root_difference = float(np.max(np.abs(first.state - second.state) / scale))
    first_branches = branch_classification(first.state, clarifier)
    second_branches = branch_classification(second.state, clarifier)
    branch_agreement = first_branches == second_branches
    accepted = bool(
        first.accepted
        and second.accepted
        and (
            root_difference
            <= float(_MECHANISTIC_GENERATION["two_start_scaled_tolerance"])
        )
        and branch_agreement
    )
    return {
        "index": index,
        "accepted": accepted,
        "target": assemble_target(first.state, operating, influent, clarifier),
        "state": first.state,
        "state_start_2": second.state,
        "elapsed_seconds": perf_counter() - started,
        "root_difference_inf": root_difference,
        "start_1": first.diagnostics,
        "start_2": second.diagnostics,
        "branch_agreement": branch_agreement,
        "branch_classification": first_branches,
        "routes": [first.route, second.route],
    }


GENERATION_SCHEMA = "fixed-lhs-generation"


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Publish atomically despite transient Windows file-sharing locks."""
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            transient = (
                isinstance(error, PermissionError)
                or getattr(error, "winerror", None) in {5, 32, 33}
                or getattr(error, "errno", None) == 13
            )
            if not transient or attempt == 7:
                raise
            time.sleep(0.025 * 2**attempt)


@dataclass(frozen=True)
class MechanisticBlockResult:
    """An accepted block plus complete attempt and source provenance."""

    controls: np.ndarray
    influents: np.ndarray
    mechanistic_responses: np.ndarray
    diagnostics: pd.DataFrame
    attempts: pd.DataFrame
    provenance: pd.DataFrame


@dataclass(frozen=True)
class _Candidate:
    block: str
    candidate_index: int
    decision: np.ndarray
    influent: np.ndarray
    checkpoint: Path

    @property
    def candidate_id(self) -> str:
        return f"{self.block}:candidate_{self.candidate_index:06d}"


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
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


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_dataframe(path: Path, frame: pd.DataFrame) -> None:
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


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_block(
    block: str | None, controls: np.ndarray, profile: StudyProfile, output: Path
) -> tuple[str, int, int]:
    name = block or output.name
    if name not in {"development", "holdout"}:
        matches = []
        if len(controls) == profile.development_candidate_count:
            matches.append("development")
        if len(controls) == profile.holdout_candidate_count:
            matches.append("holdout")
        if len(matches) != 1:
            raise ValueError(
                "block must be 'development' or 'test' when its identity cannot be inferred uniquely"
            )
        name = matches[0]
    count = (
        profile.development_candidate_count
        if name == "development"
        else profile.holdout_candidate_count
    )
    seed = profile.development_seed if name == "development" else profile.holdout_seed
    if len(controls) != count:
        raise ValueError(f"{name} requires exactly {count} initial candidates")
    return (name, count, seed)


def _base_contract_hash(
    controls: np.ndarray, influents: np.ndarray, profile: StudyProfile
) -> str:
    from surrogate_optimization.runtime.contracts import source_digest

    payload = (
        json.dumps(asdict(profile), sort_keys=True).encode()
        + np.ascontiguousarray(controls, dtype="<f8").tobytes()
        + np.ascontiguousarray(influents, dtype="<f8").tobytes()
        + source_digest().encode()
    )
    return sha256(payload).hexdigest()


def _record_from_result(
    result: dict[str, object], candidate: _Candidate
) -> dict[str, object]:
    first = result["start_1"]
    second = result["start_2"]
    assert isinstance(first, dict) and isinstance(second, dict)
    record = {
        "candidate_id": candidate.candidate_id,
        "candidate_index": candidate.candidate_index,
        "accepted": bool(result["accepted"]),
        "attempt_status": "accepted" if result["accepted"] else "rejected",
        "error_type": "",
        "error_message": "",
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "root_difference_inf": float(result["root_difference_inf"]),
        "branch_agreement": bool(result["branch_agreement"]),
        "branch_classification": json.dumps(
            result["branch_classification"], sort_keys=True
        ),
        "accepted_start_1": bool(first["passed"]),
        "accepted_start_2": bool(second["passed"]),
        "scaled_residual_start_1": first["scaled_residual_inf"],
        "scaled_residual_start_2": second["scaled_residual_inf"],
        "clarifier_component_residual_start_1": first["clarifier_component_residual"],
        "clarifier_component_residual_start_2": second["clarifier_component_residual"],
        "plant_boundary_residual_start_1": first["plant_boundary_residual"],
        "plant_boundary_residual_start_2": second["plant_boundary_residual"],
        "clarifier_tss_residual_start_1": first["clarifier_tss_residual"],
        "clarifier_tss_residual_start_2": second["clarifier_tss_residual"],
        "minimum_state_start_1": first["minimum_state"],
        "minimum_state_start_2": second["minimum_state"],
        "state_negativity_start_1": first["state_negativity_max"],
        "state_negativity_start_2": second["state_negativity_max"],
        "rate_negativity_start_1": first["rate_negativity_max"],
        "rate_negativity_start_2": second["rate_negativity_max"],
        "mass_residual_start_1": first["balance_residual"],
        "mass_residual_start_2": second["balance_residual"],
        "largest_real_eigenvalue_start_1": first["largest_real_eigenvalue"],
        "largest_real_eigenvalue_start_2": second["largest_real_eigenvalue"],
        "stability_agreement_start_1": first["stability_eigenvalue_agreement"],
        "stability_agreement_start_2": second["stability_eigenvalue_agreement"],
        "feed_tss_start_1": first["feed_tss_g_m3"],
        "feed_tss_start_2": second["feed_tss_g_m3"],
        "external_solids_loss_start_1": first["external_solids_loss_g_m3"],
        "external_solids_loss_start_2": second["external_solids_loss_g_m3"],
        "layer_envelope_start_1": bool(first["layer_envelope"]),
        "layer_envelope_start_2": bool(second["layer_envelope"]),
        "finite_nonnegative_rates_start_1": bool(first["finite_nonnegative_rates"]),
        "finite_nonnegative_rates_start_2": bool(second["finite_nonnegative_rates"]),
        "locally_stable_start_1": bool(first["locally_stable"]),
        "locally_stable_start_2": bool(second["locally_stable"]),
        "soluble_passthrough_error_start_1": first["soluble_passthrough_error"],
        "soluble_passthrough_error_start_2": second["soluble_passthrough_error"],
        "route_start_1": result["routes"][0],
        "route_start_2": result["routes"][1],
    }
    return _annotate_rejection(record)


def _finite_exceeds(
    record: dict[str, object], names: tuple[str, ...], limit: float
) -> bool:
    for name in names:
        try:
            value = float(record.get(name, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > limit:
            return True
    return False


def _finite_below(
    record: dict[str, object], names: tuple[str, ...], limit: float
) -> bool:
    for name in names:
        try:
            value = float(record.get(name, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value < limit:
            return True
    return False


def _explicit_false(record: dict[str, object], names: tuple[str, ...]) -> bool:
    return any((name in record and record[name] is False for name in names))


def _annotate_rejection(record: dict[str, object]) -> dict[str, object]:
    """Attach overlapping flags and a deterministic primary rejection reason.

    Primary precedence is solver exception, mass/residual, stability,
    nonnegativity, domain, root distance, branch disagreement, then an
    otherwise-unclassified solver rejection.  ``rejection_reasons`` retains
    every applicable flag in that same order.
    """
    item = dict(record)
    if bool(item.get("accepted", False)):
        flags = {
            "solver_exception": False,
            "mass_or_residual": False,
            "stability": False,
            "nonnegativity": False,
            "domain": False,
            "root_distance": False,
            "branch_disagreement": False,
            "other_solver_rejection": False,
        }
        primary = "accepted"
        reasons = "accepted"
    else:
        solver_exception = str(item.get("attempt_status", "")) == "solver_exception"
        mass_or_residual = _finite_exceeds(
            item,
            (
                "mass_residual_start_1",
                "mass_residual_start_2",
                "scaled_residual_start_1",
                "scaled_residual_start_2",
                "clarifier_component_residual_start_1",
                "clarifier_component_residual_start_2",
                "plant_boundary_residual_start_1",
                "plant_boundary_residual_start_2",
                "clarifier_tss_residual_start_1",
                "clarifier_tss_residual_start_2",
            ),
            1e-08,
        )
        stability = _finite_exceeds(
            item,
            ("largest_real_eigenvalue_start_1", "largest_real_eigenvalue_start_2"),
            -1e-08,
        ) or _finite_exceeds(
            item, ("stability_agreement_start_1", "stability_agreement_start_2"), 1e-06
        )
        nonnegativity = (
            _finite_exceeds(
                item, ("state_negativity_start_1", "state_negativity_start_2"), 1e-10
            )
            or _finite_exceeds(
                item, ("rate_negativity_start_1", "rate_negativity_start_2"), 1e-12
            )
            or _explicit_false(
                item,
                (
                    "finite_nonnegative_rates_start_1",
                    "finite_nonnegative_rates_start_2",
                ),
            )
        )
        domain = (
            _finite_below(
                item,
                (
                    "feed_tss_start_1",
                    "feed_tss_start_2",
                    "external_solids_loss_start_1",
                    "external_solids_loss_start_2",
                ),
                1.0,
            )
            or _explicit_false(
                item, ("layer_envelope_start_1", "layer_envelope_start_2")
            )
            or _finite_exceeds(
                item,
                (
                    "soluble_passthrough_error_start_1",
                    "soluble_passthrough_error_start_2",
                ),
                1e-10,
            )
        )
        root_distance = _finite_exceeds(item, ("root_difference_inf",), 1e-06)
        branch_disagreement = item.get("branch_agreement") is False
        flags = {
            "solver_exception": solver_exception,
            "mass_or_residual": mass_or_residual,
            "stability": stability,
            "nonnegativity": nonnegativity,
            "domain": domain,
            "root_distance": root_distance,
            "branch_disagreement": branch_disagreement,
        }
        flags["other_solver_rejection"] = not any(flags.values())
        ordered = [name for name, active in flags.items() if active]
        primary = ordered[0]
        reasons = ";".join(ordered)
    for name, active in flags.items():
        item[f"rejected_{name}"] = bool(active)
    item["rejection_reason"] = primary
    item["rejection_reasons"] = reasons
    return item


def _error_record(candidate: _Candidate, error: BaseException) -> dict[str, object]:
    record: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "candidate_index": candidate.candidate_index,
        "accepted": False,
        "attempt_status": "solver_exception",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "elapsed_seconds": np.nan,
        "root_difference_inf": np.nan,
        "branch_agreement": False,
        "branch_classification": "{}",
        "route_start_1": "",
        "route_start_2": "",
    }
    for name in (
        "minimum_state_start_1",
        "minimum_state_start_2",
        "state_negativity_start_1",
        "state_negativity_start_2",
        "rate_negativity_start_1",
        "rate_negativity_start_2",
        "mass_residual_start_1",
        "mass_residual_start_2",
        "largest_real_eigenvalue_start_1",
        "largest_real_eigenvalue_start_2",
        "stability_agreement_start_1",
        "stability_agreement_start_2",
        "feed_tss_start_1",
        "feed_tss_start_2",
        "external_solids_loss_start_1",
        "external_solids_loss_start_2",
    ):
        record[name] = np.nan
    return _annotate_rejection(record)


def _load_attempt(
    candidate: _Candidate,
    *,
    expected_contract_hash: str,
    state_size: int,
    response_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]] | None:
    path = candidate.checkpoint
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            valid = bool(
                str(stored["contract_hash"].item()) == expected_contract_hash
                and np.array_equal(stored["decision"], candidate.decision)
                and np.array_equal(stored["influent"], candidate.influent)
                and (stored["target"].shape == (response_count,))
                and (stored["state_start_1"].shape == (state_size,))
                and (stored["state_start_2"].shape == (state_size,))
            )
            if not valid:
                raise RuntimeError(
                    f"immutable attempt checkpoint does not match its contract: {path}"
                )
            target = np.asarray(stored["target"], dtype=float)
            first = np.asarray(stored["state_start_1"], dtype=float)
            second = np.asarray(stored["state_start_2"], dtype=float)
            record = json.loads(str(stored["record_json"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot validate immutable attempt checkpoint: {path}"
        ) from error
    record = dict(record)
    record.update(
        {
            "candidate_id": candidate.candidate_id,
            "candidate_index": candidate.candidate_index,
            "attempt_status": "accepted"
            if bool(record.get("accepted", False))
            else "rejected",
            "error_type": str(record.get("error_type", "")),
            "error_message": str(record.get("error_message", "")),
        }
    )
    record = _annotate_rejection(record)
    accepted = bool(record.get("accepted", False))
    if accepted and (
        not (
            np.all(np.isfinite(target))
            and np.all(np.isfinite(first))
            and np.all(np.isfinite(second))
        )
    ):
        raise RuntimeError(f"accepted immutable attempt is non-finite: {path}")
    return (target, first, second, record)


def _write_attempt(
    candidate: _Candidate,
    *,
    contract_hash: str,
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    record: dict[str, object],
) -> None:
    if candidate.checkpoint.exists():
        raise RuntimeError(
            f"refusing to overwrite immutable attempt: {candidate.checkpoint}"
        )
    _atomic_npz(
        candidate.checkpoint,
        contract_hash=np.asarray(contract_hash),
        decision=np.asarray(candidate.decision, dtype=float),
        influent=np.asarray(candidate.influent, dtype=float),
        target=np.asarray(target, dtype=float),
        state_start_1=np.asarray(first, dtype=float),
        state_start_2=np.asarray(second, dtype=float),
        record_json=np.asarray(json.dumps(record, sort_keys=True)),
    )


def _solve_candidates(
    candidates: list[_Candidate], profile: StudyProfile, contract_hash: str
) -> list[tuple[_Candidate, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]]:
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES

    state_size = N_STAGES * N_COMPONENTS + profile.layer_count
    loaded: dict[
        str, tuple[_Candidate, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]
    ] = {}
    missing: list[_Candidate] = []
    for candidate in candidates:
        cached = _load_attempt(
            candidate,
            expected_contract_hash=contract_hash,
            state_size=state_size,
            response_count=profile.mechanistic_response_count,
        )
        if cached is None:
            missing.append(candidate)
        else:
            loaded[candidate.candidate_id] = (candidate, *cached)
    label = (
        f"{candidates[0].block} candidates" if candidates else "empty candidate batch"
    )
    if loaded:
        print(
            f"[{label}] reusing {len(loaded)}/{len(candidates)} immutable attempts",
            flush=True,
        )
    if missing:
        progress_interval = max(1, len(candidates) // 100)
        completed_now = 0
        batch_started = time.monotonic()
        print(
            f"[{label}] starting {len(missing)} candidate solves with {profile.parallel_workers} workers",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=profile.parallel_workers) as pool:
            futures = {
                pool.submit(
                    _solve_design_row,
                    (
                        candidate.candidate_index,
                        candidate.decision,
                        candidate.influent,
                        profile.layer_count,
                    ),
                ): candidate
                for candidate in missing
            }
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=60.0, return_when=FIRST_COMPLETED)
                if not done:
                    elapsed = time.monotonic() - batch_started
                    print(
                        f"[{label}] heartbeat: {completed_now}/{len(missing)} new checkpoints after {elapsed / 60.0:.1f} min; {len(pending)} pending",
                        flush=True,
                    )
                    continue
                for future in done:
                    candidate = futures[future]
                    try:
                        result = future.result()
                        target = np.asarray(result["target"], dtype=float)
                        first = np.asarray(result["state"], dtype=float)
                        second = np.asarray(result["state_start_2"], dtype=float)
                        record = _record_from_result(result, candidate)
                    except Exception as error:
                        target = np.full(profile.mechanistic_response_count, np.nan)
                        first = np.full(state_size, np.nan)
                        second = np.full(state_size, np.nan)
                        record = _error_record(candidate, error)
                    _write_attempt(
                        candidate,
                        contract_hash=contract_hash,
                        target=target,
                        first=first,
                        second=second,
                        record=record,
                    )
                    loaded[candidate.candidate_id] = (
                        candidate,
                        target,
                        first,
                        second,
                        record,
                    )
                    completed_now += 1
                    completed_total = len(candidates) - len(missing) + completed_now
                    if (
                        completed_total % progress_interval == 0
                        or completed_total == len(candidates)
                    ):
                        elapsed = max(time.monotonic() - batch_started, 1e-09)
                        rate = completed_now / elapsed
                        remaining = len(missing) - completed_now
                        eta = remaining / rate if rate > 0.0 else float("inf")
                        print(
                            f"[{label}] immutable attempt checkpoints {completed_total}/{len(candidates)}; elapsed {elapsed / 60.0:.1f} min; ETA {eta / 60.0:.1f} min",
                            flush=True,
                        )
    return [loaded[candidate.candidate_id] for candidate in candidates]


def generate_mechanistic_block_from_fixed_design(
    controls: np.ndarray,
    influents: np.ndarray,
    profile: StudyProfile,
    output: Path,
    *,
    block: str | None = None,
) -> MechanisticBlockResult:
    """Solve one fixed LHS candidate block and retain its accepted rows."""
    from surrogate_optimization.data.design import _design_block
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES

    controls = np.asarray(controls, dtype=float)
    influents = np.asarray(influents, dtype=float)
    output = Path(output)
    block_name, required_count, seed = _resolve_block(block, controls, profile, output)
    if controls.shape != (required_count, 7):
        raise ValueError(f"{block_name} controls have an invalid shape")
    if influents.shape != (required_count, N_COMPONENTS):
        raise ValueError(f"{block_name} influents have an invalid shape")
    if not np.all(np.isfinite(controls)) or not np.all(np.isfinite(influents)):
        raise ValueError(f"{block_name} initial candidates must be finite")
    expected_controls, expected_influents, generator = _design_block(
        required_count, seed
    )
    if not (
        np.array_equal(controls, expected_controls)
        and np.array_equal(influents, expected_influents)
    ):
        raise ValueError(
            f"{block_name} initial candidates do not match the declared design"
        )
    output.mkdir(parents=True, exist_ok=True)
    rows_directory = output / "rows"
    rows_directory.mkdir(parents=True, exist_ok=True)
    base_contract = _base_contract_hash(controls, influents, profile)
    state_size = N_STAGES * N_COMPONENTS + profile.layer_count
    base_candidates = [
        _Candidate(
            block=block_name,
            candidate_index=index,
            decision=controls[index],
            influent=influents[index],
            checkpoint=rows_directory / f"row_{index:06d}.npz",
        )
        for index in range(required_count)
    ]
    preexisting = {
        candidate.candidate_id: candidate.checkpoint.is_file()
        for candidate in base_candidates
    }
    base_results = _solve_candidates(base_candidates, profile, base_contract)
    attempts = list(base_results)
    accepted_results = [item for item in base_results if bool(item[4]["accepted"])]
    accepted_count = len(accepted_results)
    if accepted_count == 0:
        raise RuntimeError(f"{block_name} fixed LHS produced no accepted rows")
    accepted_controls = np.empty((accepted_count, 7), dtype=float)
    accepted_influents = np.empty((accepted_count, N_COMPONENTS), dtype=float)
    mechanistic_responses = np.empty(
        (accepted_count, profile.mechanistic_response_count), dtype=float
    )
    states_start_1 = np.empty((accepted_count, state_size), dtype=float)
    states_start_2 = np.empty_like(states_start_1)
    diagnostic_records: list[dict[str, object]] = []
    provenance_records: list[dict[str, object]] = []
    for slot, item in enumerate(accepted_results):
        candidate, target, first, second, attempt_record = item
        accepted_controls[slot] = candidate.decision
        accepted_influents[slot] = candidate.influent
        mechanistic_responses[slot] = target
        states_start_1[slot] = first
        states_start_2[slot] = second
        diagnostic = dict(attempt_record)
        diagnostic["row"] = slot
        diagnostic["accepted_slot"] = slot
        diagnostic["source_candidate_id"] = candidate.candidate_id
        diagnostic["source_candidate_index"] = candidate.candidate_index
        diagnostic_records.append(diagnostic)
        provenance_records.append(
            {
                "accepted_slot": slot,
                "source_candidate_id": candidate.candidate_id,
                "source_candidate_index": candidate.candidate_index,
            }
        )
    diagnostics = pd.DataFrame(diagnostic_records)
    accepted_slots_by_candidate = {
        str(record["source_candidate_id"]): int(record["accepted_slot"])
        for record in provenance_records
    }
    attempt_records = []
    for candidate, _target, _first, _second, record in attempts:
        item = dict(record)
        item["checkpoint_path"] = str(candidate.checkpoint.relative_to(output))
        item["checkpoint_sha256"] = _file_digest(candidate.checkpoint)
        item["selected_for_accepted_block"] = (
            candidate.candidate_id in accepted_slots_by_candidate
        )
        item["accepted_slot"] = accepted_slots_by_candidate.get(
            candidate.candidate_id, np.nan
        )
        attempt_records.append(item)
    attempts_frame = (
        pd.DataFrame(attempt_records)
        .sort_values(["candidate_index"], kind="stable")
        .reset_index(drop=True)
    )
    provenance = pd.DataFrame(provenance_records)
    checkpoint_summary = pd.DataFrame(
        [
            {
                "candidate_id": candidate.candidate_id,
                "candidate_index": candidate.candidate_index,
                "checkpoint_path": str(candidate.checkpoint.relative_to(output)),
                "checkpoint_sha256": _file_digest(candidate.checkpoint),
                "original_contract_hash": base_contract,
                "preexisting_checkpoint": preexisting[candidate.candidate_id],
                "accepted": bool(item[4]["accepted"]),
                "preserved_without_rewrite": preexisting[candidate.candidate_id],
            }
            for candidate, item in zip(base_candidates, base_results, strict=True)
        ]
    )
    _atomic_npz(
        output / "accepted_inputs.npz",
        controls=accepted_controls,
        influents=accepted_influents,
        source_candidate_id=provenance["source_candidate_id"].to_numpy(str),
        source_candidate_index=provenance["source_candidate_index"].to_numpy(int),
    )
    _atomic_npz(
        output / "accepted_mechanistic_responses.npz",
        contract_hash=np.asarray(base_contract),
        mechanistic_responses=mechanistic_responses,
        states_start_1=states_start_1,
        states_start_2=states_start_2,
    )
    _atomic_dataframe(output / "accepted_diagnostics.csv", diagnostics)
    _atomic_dataframe(output / "all_attempts.csv", attempts_frame)
    _atomic_dataframe(output / "accepted_provenance.csv", provenance)
    _atomic_dataframe(output / "candidate_checkpoint_summary.csv", checkpoint_summary)
    _atomic_json(
        output / "generation_summary.json",
        {
            "schema": GENERATION_SCHEMA,
            "block": block_name,
            "candidate_count": required_count,
            "accepted_count": accepted_count,
            "rejected_count": required_count - accepted_count,
            "initial_seed": int(seed),
            "initial_final_state": int(generator["final_state"]),
            "initial_draw_count": int(generator["draw_count"]),
        },
    )
    return MechanisticBlockResult(
        controls=accepted_controls,
        influents=accepted_influents,
        mechanistic_responses=mechanistic_responses,
        diagnostics=diagnostics,
        attempts=attempts_frame,
        provenance=provenance,
    )
