"""Surrogate training."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.surrogate.regression import LogOverflowTSSClosure
    from surrogate_optimization.surrogate.projection import NetworkLayout
    from surrogate_optimization.surrogate.regression import QuadraticSurrogate
from surrogate_optimization.config import _PARAMETERS
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from typing import Mapping
import json
import math
import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class RidgeSelectionResult:
    model: QuadraticSurrogate
    scores: pd.DataFrame
    fold_membership: np.ndarray
    out_of_fold_raw: np.ndarray
    elapsed_seconds: float


@dataclass(frozen=True)
class LogOverflowClosureSelectionResult:
    closure: LogOverflowTSSClosure
    scores: pd.DataFrame
    fold_membership: np.ndarray
    out_of_fold_log: np.ndarray
    out_of_fold_tss: np.ndarray
    exact_overflow_tss: np.ndarray
    elapsed_seconds: float


def _fold_permutation(count: int, seed: int = 271828) -> np.ndarray:
    from surrogate_optimization.data.random_design import SplitMix64

    stream = SplitMix64(seed)
    values = list(range(count))
    for i in range(count - 1, 0, -1):
        j = stream.randbelow(i + 1)
        values[i], values[j] = (values[j], values[i])
    return np.asarray(values, dtype=int)


def cross_validate_ridge(
    controls: np.ndarray, influents: np.ndarray, mechanistic_responses: np.ndarray
) -> RidgeSelectionResult:
    """Five-fold raw-response CV and the model one-standard-error rule."""
    from surrogate_optimization.config import RIDGE_GRID
    from surrogate_optimization.surrogate.regression import QuadraticSurrogate

    order = _fold_permutation(len(controls))
    folds = np.array_split(order, 5)
    rows: list[dict[str, float | int]] = []
    predictions: dict[float, np.ndarray] = {
        float(gamma): np.full_like(mechanistic_responses, np.nan, dtype=float)
        for gamma in RIDGE_GRID
    }
    membership = np.empty(len(controls), dtype=int)
    started = perf_counter()
    for fold_index, validation in enumerate(folds):
        membership[validation] = fold_index + 1
        fitting = np.setdiff1d(order, validation, assume_unique=True)
        for gamma in RIDGE_GRID:
            model = QuadraticSurrogate.fit_ridge(
                controls[fitting],
                influents[fitting],
                mechanistic_responses[fitting],
                ridge_penalty=float(gamma),
            )
            prediction = model.predict(controls[validation], influents[validation])
            predictions[float(gamma)][validation] = prediction
            score = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            (prediction - mechanistic_responses[validation])
                            / model.response_scale
                        )
                    )
                )
            )
            rows.append({"fold": fold_index + 1, "gamma": gamma, "raw_nrmse": score})
    scores = pd.DataFrame(rows)
    summary = scores.groupby("gamma")["raw_nrmse"].agg(["mean", "std"]).reset_index()
    summary["standard_error"] = summary["std"] / math.sqrt(5.0)
    minimum = summary.loc[summary["mean"].idxmin()]
    eligible = summary.loc[
        summary["mean"] <= minimum["mean"] + minimum["standard_error"]
    ]
    selected = float(eligible["gamma"].max())
    scores = scores.merge(summary, on="gamma", how="left")
    scores["selected"] = scores["gamma"].eq(selected)
    final = QuadraticSurrogate.fit_ridge(
        controls, influents, mechanistic_responses, ridge_penalty=selected
    )
    return RidgeSelectionResult(
        model=final,
        scores=scores,
        fold_membership=membership,
        out_of_fold_raw=predictions[selected],
        elapsed_seconds=perf_counter() - started,
    )


def overflow_tss_from_response(
    responses: npt.ArrayLike, controls: npt.ArrayLike, layout: NetworkLayout
) -> np.ndarray:
    from surrogate_optimization.plant.definitions import TSS_VECTOR

    states = np.asarray(responses, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != layout.state_size:
        raise ValueError("responses have inconsistent dimensions for overflow TSS")
    if controls.shape != (states.shape[0], 7):
        raise ValueError("controls have inconsistent dimensions for overflow TSS")
    q_effluent = 1.0 - controls[:, 6]
    if np.any(q_effluent <= 0.0):
        raise ValueError("overflow flow must be strictly positive")
    values = states[:, layout.overflow_flow_slice] @ TSS_VECTOR / q_effluent
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(
            "mechanistic overflow TSS must be finite and strictly positive"
        )
    return values


def cross_validate_log_overflow_closure(
    controls: np.ndarray,
    influents: np.ndarray,
    mechanistic_responses: np.ndarray,
    *,
    layout: NetworkLayout | None = None,
    reference_concentration: float = float(
        _PARAMETERS["surrogate"]["overflow_tss_closure"]["reference_concentration_mg_L"]
    ),
) -> LogOverflowClosureSelectionResult:
    """Select and refit the development-only quadratic log-overflow closure."""
    from surrogate_optimization.config import RIDGE_GRID
    from surrogate_optimization.surrogate.projection import NetworkLayout
    from surrogate_optimization.surrogate.regression import LogOverflowTSSClosure

    layout = layout or NetworkLayout()
    exact = overflow_tss_from_response(mechanistic_responses, controls, layout)
    log_exact = np.log(exact / float(reference_concentration))
    order = _fold_permutation(len(controls))
    folds = np.array_split(order, 5)
    rows: list[dict[str, float | int]] = []
    log_predictions = {
        float(gamma): np.full(len(controls), np.nan, dtype=np.float64)
        for gamma in RIDGE_GRID
    }
    membership = np.empty(len(controls), dtype=int)
    started = perf_counter()
    for fold_index, validation in enumerate(folds):
        membership[validation] = fold_index + 1
        fitting = np.setdiff1d(order, validation, assume_unique=True)
        for gamma in RIDGE_GRID:
            closure = LogOverflowTSSClosure.fit_ridge(
                controls[fitting],
                influents[fitting],
                exact[fitting],
                ridge_penalty=float(gamma),
                reference_concentration=reference_concentration,
            )
            prediction = np.asarray(
                closure.predict_log(controls[validation], influents[validation]),
                dtype=np.float64,
            )
            log_predictions[float(gamma)][validation] = prediction
            score = float(
                np.sqrt(np.mean(np.square(prediction - log_exact[validation])))
            )
            rows.append({"fold": fold_index + 1, "gamma": gamma, "log_rmse": score})
    scores = pd.DataFrame(rows)
    summary = scores.groupby("gamma")["log_rmse"].agg(["mean", "std"]).reset_index()
    summary["standard_error"] = summary["std"] / math.sqrt(5.0)
    minimum = summary.loc[summary["mean"].idxmin()]
    eligible = summary.loc[
        summary["mean"] <= minimum["mean"] + minimum["standard_error"]
    ]
    selected = float(eligible["gamma"].max())
    scores = scores.merge(summary, on="gamma", how="left")
    scores["selected"] = scores["gamma"].eq(selected)
    final = LogOverflowTSSClosure.fit_ridge(
        controls,
        influents,
        exact,
        ridge_penalty=selected,
        reference_concentration=reference_concentration,
    )
    selected_log = log_predictions[selected]
    selected_tss = reference_concentration * np.exp(selected_log)
    if not np.all(np.isfinite(selected_tss)) or np.any(selected_tss <= 0.0):
        raise RuntimeError("out-of-fold log-overflow predictions are invalid")
    return LogOverflowClosureSelectionResult(
        closure=final,
        scores=scores,
        fold_membership=membership,
        out_of_fold_log=selected_log,
        out_of_fold_tss=selected_tss,
        exact_overflow_tss=exact,
        elapsed_seconds=perf_counter() - started,
    )


def _ridge_input_digest(
    controls: np.ndarray, influents: np.ndarray, mechanistic_responses: np.ndarray
) -> str:
    from surrogate_optimization.runtime.contracts import array_digest

    return array_digest(
        development_controls=np.asarray(controls, dtype="<f8"),
        development_influents=np.asarray(influents, dtype="<f8"),
        development_responses=np.asarray(mechanistic_responses, dtype="<f8"),
    )


def _validate_ridge_scores(
    scores: pd.DataFrame, fold_membership: np.ndarray, row_count: int
) -> None:
    from surrogate_optimization.config import RIDGE_GRID

    required = {"fold", "gamma", "raw_nrmse", "selected"}
    if required - set(scores.columns):
        raise RuntimeError("ridge score checkpoint omits required columns")
    if len(scores) != 5 * len(RIDGE_GRID):
        raise RuntimeError(
            "ridge score checkpoint does not contain the full 5-fold grid"
        )
    if set(np.asarray(scores["fold"], dtype=int)) != {1, 2, 3, 4, 5}:
        raise RuntimeError("ridge score checkpoint has invalid fold identifiers")
    gamma = np.asarray(scores["gamma"], dtype=float)
    nrmse = np.asarray(scores["raw_nrmse"], dtype=float)
    if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(nrmse)):
        raise RuntimeError("ridge score checkpoint contains non-finite values")
    for candidate in RIDGE_GRID:
        if np.count_nonzero(np.isclose(gamma, candidate, rtol=1e-12, atol=0.0)) != 5:
            raise RuntimeError(
                "ridge score checkpoint does not cover each penalty five times"
            )
    selected = (
        scores["selected"].astype(str).str.lower().map({"true": True, "false": False})
    )
    if selected.isna().any() or int(selected.sum()) != 5:
        raise RuntimeError("ridge score checkpoint has an invalid selected penalty")
    chosen = gamma[selected.to_numpy()]
    if not np.allclose(chosen, chosen[0], rtol=0.0, atol=0.0):
        raise RuntimeError("ridge score checkpoint selects more than one penalty")
    membership = np.asarray(fold_membership, dtype=int)
    if membership.shape != (row_count,) or set(membership) != {1, 2, 3, 4, 5}:
        raise RuntimeError("ridge fold membership checkpoint is invalid")
    counts = np.bincount(membership, minlength=6)[1:]
    if int(counts.max() - counts.min()) > 1:
        raise RuntimeError("ridge folds do not form the declared balanced partition")


def save_ridge(run: Path, result: Any, *, input_id: str, source_id: str) -> None:
    from surrogate_optimization.runtime.artifacts import atomic_dataframe
    from surrogate_optimization.runtime.artifacts import atomic_json
    from surrogate_optimization.runtime.artifacts import atomic_npz
    from surrogate_optimization.runtime.contracts import _artifact_hashes

    model = result.model
    scores_path = run / "metrics" / "ridge_cross_validation.csv"
    fold_path = run / "metrics" / "ridge_fold_membership.csv"
    bundle_path = run / "models" / "ridge_surrogate.npz"
    atomic_dataframe(scores_path, result.scores)
    atomic_dataframe(
        fold_path,
        pd.DataFrame(
            {
                "row": np.arange(len(result.fold_membership)),
                "fold": result.fold_membership,
            }
        ),
    )
    atomic_npz(
        bundle_path,
        input_digest=np.asarray(input_id),
        source_digest=np.asarray(source_id),
        decision_center=model.feature_map.decision_center,
        decision_scale=model.feature_map.decision_scale,
        influent_center=model.feature_map.influent_center,
        influent_scale=model.feature_map.influent_scale,
        term_center=model.feature_map.term_center,
        term_scale=model.feature_map.term_scale,
        variance_relative_tolerance=np.asarray(
            model.feature_map.variance_relative_tolerance
        ),
        response_center=model.response_center,
        response_scale=model.response_scale,
        coefficients=model.coefficients,
        ridge_penalty=np.asarray(model.ridge_penalty),
        diagnostics_json=np.asarray(
            json.dumps(asdict(model.diagnostics), sort_keys=True)
        ),
        fold_membership=np.asarray(result.fold_membership, dtype=int),
        out_of_fold_raw=result.out_of_fold_raw,
        elapsed_seconds=np.asarray(result.elapsed_seconds),
    )
    paths = (scores_path, fold_path, bundle_path)
    atomic_json(
        run / "models" / "ridge_complete.json",
        {
            "stage": "ridge_cross_validation",
            "source_digest": source_id,
            "input_digest": input_id,
            "selected_penalty": model.ridge_penalty,
            "artifacts": _artifact_hashes(run, paths),
        },
    )


def _load_ridge(
    run: Path,
    *,
    controls: np.ndarray,
    influents: np.ndarray,
    mechanistic_responses: np.ndarray,
    input_id: str,
    source_id: str,
) -> tuple[QuadraticSurrogate, np.ndarray] | None:
    from surrogate_optimization.config import RIDGE_GRID
    from surrogate_optimization.runtime.contracts import _artifacts_match
    from surrogate_optimization.runtime.contracts import (
        _checkpoint_source_is_authorized,
    )
    from surrogate_optimization.surrogate.regression import LeastSquaresDiagnostics
    from surrogate_optimization.surrogate.regression import QuadraticFeatureMap
    from surrogate_optimization.surrogate.regression import QuadraticSurrogate

    marker_path = run / "models" / "ridge_complete.json"
    bundle_path = run / "models" / "ridge_surrogate.npz"
    scores_path = run / "metrics" / "ridge_cross_validation.csv"
    fold_path = run / "metrics" / "ridge_fold_membership.csv"
    if not all(
        (path.is_file() for path in (marker_path, bundle_path, scores_path, fold_path))
    ):
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_source_id = str(marker.get("source_digest", ""))
        if (
            not _checkpoint_source_is_authorized(
                run,
                stage="ridge",
                checkpoint=marker_path,
                observed_source_id=marker_source_id,
                current_source_id=source_id,
            )
            or marker.get("input_digest") != input_id
            or (not _artifacts_match(run, marker.get("artifacts", {})))
        ):
            return None
        with np.load(bundle_path, allow_pickle=False) as stored:
            if (
                str(stored["input_digest"].item()) != input_id
                or str(stored["source_digest"].item()) != marker_source_id
            ):
                return None
            feature_map = QuadraticFeatureMap(
                decision_center=np.asarray(stored["decision_center"], dtype=float),
                decision_scale=np.asarray(stored["decision_scale"], dtype=float),
                influent_center=np.asarray(stored["influent_center"], dtype=float),
                influent_scale=np.asarray(stored["influent_scale"], dtype=float),
                term_center=np.asarray(stored["term_center"], dtype=float),
                term_scale=np.asarray(stored["term_scale"], dtype=float),
                variance_relative_tolerance=float(
                    stored["variance_relative_tolerance"]
                ),
            )
            diagnostics = LeastSquaresDiagnostics(
                **json.loads(str(stored["diagnostics_json"].item()))
            )
            model = QuadraticSurrogate(
                feature_map=feature_map,
                response_center=np.asarray(stored["response_center"], dtype=float),
                response_scale=np.asarray(stored["response_scale"], dtype=float),
                coefficients=np.asarray(stored["coefficients"], dtype=float),
                diagnostics=diagnostics,
                ridge_penalty=float(stored["ridge_penalty"]),
            )
            oof = np.asarray(stored["out_of_fold_raw"], dtype=float)
            membership = np.asarray(stored["fold_membership"], dtype=int)
        expected_features = QuadraticFeatureMap.expected_feature_count(
            controls.shape[1], influents.shape[1]
        )
        if (
            feature_map.decision_count != controls.shape[1]
            or feature_map.influent_count != influents.shape[1]
            or feature_map.feature_count != expected_features
            or (model.response_center.shape != (mechanistic_responses.shape[1],))
            or (model.response_scale.shape != (mechanistic_responses.shape[1],))
            or (
                model.coefficients.shape
                != (mechanistic_responses.shape[1], expected_features)
            )
            or (oof.shape != mechanistic_responses.shape)
            or (not np.all(np.isfinite(oof)))
            or (
                not all(
                    (
                        np.all(np.isfinite(value))
                        for value in (
                            feature_map.decision_center,
                            feature_map.decision_scale,
                            feature_map.influent_center,
                            feature_map.influent_scale,
                            feature_map.term_center,
                            feature_map.term_scale,
                            model.response_center,
                            model.response_scale,
                            model.coefficients,
                        )
                    )
                )
            )
            or (not np.all(model.response_scale > 0.0))
            or (
                not np.any(
                    np.isclose(model.ridge_penalty, RIDGE_GRID, rtol=1e-12, atol=0.0)
                )
            )
        ):
            return None
        scores = pd.read_csv(scores_path)
        fold_frame = pd.read_csv(fold_path)
        if not np.array_equal(fold_frame["row"].to_numpy(), np.arange(len(controls))):
            return None
        if not np.array_equal(fold_frame["fold"].to_numpy(dtype=int), membership):
            return None
        _validate_ridge_scores(scores, membership, len(controls))
        return (model, oof)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def fit_or_resume_ridge(
    run: Path,
    controls: np.ndarray,
    influents: np.ndarray,
    mechanistic_responses: np.ndarray,
    *,
    source_files: Mapping[str, str],
) -> tuple[QuadraticSurrogate, np.ndarray, str]:
    from surrogate_optimization.runtime.contracts import assert_source_unchanged
    from surrogate_optimization.runtime.contracts import source_digest

    input_id = _ridge_input_digest(controls, influents, mechanistic_responses)
    source_id = source_digest(source_files)
    resumed = _load_ridge(
        run,
        controls=controls,
        influents=influents,
        mechanistic_responses=mechanistic_responses,
        input_id=input_id,
        source_id=source_id,
    )
    if resumed is not None:
        return (resumed[0], resumed[1], input_id)
    result = cross_validate_ridge(controls, influents, mechanistic_responses)
    _validate_ridge_scores(result.scores, result.fold_membership, len(controls))
    assert_source_unchanged(source_files)
    save_ridge(run, result, input_id=input_id, source_id=source_id)
    return (result.model, result.out_of_fold_raw, input_id)


def _validate_log_overflow_scores(
    scores: pd.DataFrame, fold_membership: np.ndarray, row_count: int
) -> None:
    from surrogate_optimization.config import RIDGE_GRID

    required = {"fold", "gamma", "log_rmse", "selected"}
    if required - set(scores.columns):
        raise RuntimeError("log-overflow score checkpoint omits required columns")
    if len(scores) != 5 * len(RIDGE_GRID):
        raise RuntimeError(
            "log-overflow checkpoint does not contain the full 5-fold grid"
        )
    if set(np.asarray(scores["fold"], dtype=int)) != {1, 2, 3, 4, 5}:
        raise RuntimeError("log-overflow checkpoint has invalid fold identifiers")
    gamma = np.asarray(scores["gamma"], dtype=float)
    error = np.asarray(scores["log_rmse"], dtype=float)
    if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(error)):
        raise RuntimeError("log-overflow score checkpoint contains non-finite values")
    for candidate in RIDGE_GRID:
        if np.count_nonzero(np.isclose(gamma, candidate, rtol=1e-12, atol=0.0)) != 5:
            raise RuntimeError(
                "log-overflow checkpoint does not cover each penalty five times"
            )
    selected = (
        scores["selected"].astype(str).str.lower().map({"true": True, "false": False})
    )
    if selected.isna().any() or int(selected.sum()) != 5:
        raise RuntimeError("log-overflow checkpoint has an invalid selected penalty")
    chosen = gamma[selected.to_numpy()]
    if not np.allclose(chosen, chosen[0], rtol=0.0, atol=0.0):
        raise RuntimeError("log-overflow checkpoint selects more than one penalty")
    membership = np.asarray(fold_membership, dtype=int)
    if membership.shape != (row_count,) or set(membership) != {1, 2, 3, 4, 5}:
        raise RuntimeError("log-overflow fold membership checkpoint is invalid")
    counts = np.bincount(membership, minlength=6)[1:]
    if int(counts.max() - counts.min()) > 1:
        raise RuntimeError(
            "log-overflow folds do not form the declared balanced partition"
        )


def save_log_overflow_closure(
    run: Path, result: Any, *, input_id: str, source_id: str
) -> None:
    from surrogate_optimization.runtime.artifacts import atomic_dataframe
    from surrogate_optimization.runtime.artifacts import atomic_json
    from surrogate_optimization.runtime.artifacts import atomic_npz
    from surrogate_optimization.runtime.contracts import _artifact_hashes
    from surrogate_optimization.runtime.protocols import PROJECTION_SCHEMA

    closure = result.closure
    model = closure.model
    scores_path = run / "metrics" / "log_overflow_closure_cross_validation.csv"
    fold_path = run / "metrics" / "log_overflow_closure_fold_membership.csv"
    bundle_path = run / "models" / "log_overflow_closure.npz"
    atomic_dataframe(scores_path, result.scores)
    atomic_dataframe(
        fold_path,
        pd.DataFrame(
            {
                "row": np.arange(len(result.fold_membership)),
                "fold": result.fold_membership,
            }
        ),
    )
    atomic_npz(
        bundle_path,
        input_digest=np.asarray(input_id),
        source_digest=np.asarray(source_id),
        reference_concentration=np.asarray(closure.reference_concentration),
        decision_center=model.feature_map.decision_center,
        decision_scale=model.feature_map.decision_scale,
        influent_center=model.feature_map.influent_center,
        influent_scale=model.feature_map.influent_scale,
        term_center=model.feature_map.term_center,
        term_scale=model.feature_map.term_scale,
        variance_relative_tolerance=np.asarray(
            model.feature_map.variance_relative_tolerance
        ),
        response_center=model.response_center,
        response_scale=model.response_scale,
        coefficients=model.coefficients,
        ridge_penalty=np.asarray(model.ridge_penalty),
        diagnostics_json=np.asarray(
            json.dumps(asdict(model.diagnostics), sort_keys=True)
        ),
        fold_membership=np.asarray(result.fold_membership, dtype=int),
        out_of_fold_log=result.out_of_fold_log,
        out_of_fold_tss=result.out_of_fold_tss,
        exact_overflow_tss=result.exact_overflow_tss,
        elapsed_seconds=np.asarray(result.elapsed_seconds),
    )
    paths = (scores_path, fold_path, bundle_path)
    atomic_json(
        run / "models" / "log_overflow_closure_complete.json",
        {
            "stage": "log_overflow_closure_cross_validation",
            "projection_schema": PROJECTION_SCHEMA,
            "source_digest": source_id,
            "input_digest": input_id,
            "selected_penalty": model.ridge_penalty,
            "reference_concentration_mg_L": closure.reference_concentration,
            "artifacts": _artifact_hashes(run, paths),
        },
    )


def _load_log_overflow_closure(
    run: Path,
    *,
    controls: np.ndarray,
    influents: np.ndarray,
    mechanistic_responses: np.ndarray,
    input_id: str,
    source_id: str,
    layout: NetworkLayout,
) -> tuple[LogOverflowTSSClosure, np.ndarray] | None:
    from surrogate_optimization.config import RIDGE_GRID
    from surrogate_optimization.runtime.contracts import _artifacts_match
    from surrogate_optimization.runtime.contracts import (
        _checkpoint_source_is_authorized,
    )
    from surrogate_optimization.runtime.protocols import PROJECTION_SCHEMA
    from surrogate_optimization.surrogate.regression import LeastSquaresDiagnostics
    from surrogate_optimization.surrogate.regression import LogOverflowTSSClosure
    from surrogate_optimization.surrogate.regression import QuadraticFeatureMap
    from surrogate_optimization.surrogate.regression import QuadraticSurrogate

    marker_path = run / "models" / "log_overflow_closure_complete.json"
    bundle_path = run / "models" / "log_overflow_closure.npz"
    scores_path = run / "metrics" / "log_overflow_closure_cross_validation.csv"
    fold_path = run / "metrics" / "log_overflow_closure_fold_membership.csv"
    if not all(
        (path.is_file() for path in (marker_path, bundle_path, scores_path, fold_path))
    ):
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_source_id = str(marker.get("source_digest", ""))
        if (
            marker.get("projection_schema") != PROJECTION_SCHEMA
            or not _checkpoint_source_is_authorized(
                run,
                stage="ridge",
                checkpoint=marker_path,
                observed_source_id=marker_source_id,
                current_source_id=source_id,
            )
            or marker.get("input_digest") != input_id
            or (not _artifacts_match(run, marker.get("artifacts", {})))
        ):
            return None
        with np.load(bundle_path, allow_pickle=False) as stored:
            if (
                str(stored["input_digest"].item()) != input_id
                or str(stored["source_digest"].item()) != marker_source_id
            ):
                return None
            feature_map = QuadraticFeatureMap(
                decision_center=np.asarray(stored["decision_center"], dtype=float),
                decision_scale=np.asarray(stored["decision_scale"], dtype=float),
                influent_center=np.asarray(stored["influent_center"], dtype=float),
                influent_scale=np.asarray(stored["influent_scale"], dtype=float),
                term_center=np.asarray(stored["term_center"], dtype=float),
                term_scale=np.asarray(stored["term_scale"], dtype=float),
                variance_relative_tolerance=float(
                    stored["variance_relative_tolerance"]
                ),
            )
            diagnostics = LeastSquaresDiagnostics(
                **json.loads(str(stored["diagnostics_json"].item()))
            )
            model = QuadraticSurrogate(
                feature_map=feature_map,
                response_center=np.asarray(stored["response_center"], dtype=float),
                response_scale=np.asarray(stored["response_scale"], dtype=float),
                coefficients=np.asarray(stored["coefficients"], dtype=float),
                diagnostics=diagnostics,
                ridge_penalty=float(stored["ridge_penalty"]),
            )
            closure = LogOverflowTSSClosure(
                model=model,
                reference_concentration=float(stored["reference_concentration"]),
            )
            membership = np.asarray(stored["fold_membership"], dtype=int)
            oof_log = np.asarray(stored["out_of_fold_log"], dtype=float)
            oof_tss = np.asarray(stored["out_of_fold_tss"], dtype=float)
            exact = np.asarray(stored["exact_overflow_tss"], dtype=float)
        expected_features = QuadraticFeatureMap.expected_feature_count(
            controls.shape[1], influents.shape[1]
        )
        expected_exact = overflow_tss_from_response(
            mechanistic_responses, controls, layout
        )
        if (
            feature_map.decision_count != controls.shape[1]
            or feature_map.influent_count != influents.shape[1]
            or feature_map.feature_count != expected_features
            or (model.response_center.shape != (1,))
            or (model.response_scale.shape != (1,))
            or (model.coefficients.shape != (1, expected_features))
            or (oof_log.shape != (len(controls),))
            or (oof_tss.shape != (len(controls),))
            or (exact.shape != (len(controls),))
            or (not np.all(np.isfinite(oof_log)))
            or (not np.all(np.isfinite(oof_tss)))
            or np.any(oof_tss <= 0.0)
            or (not np.allclose(exact, expected_exact, rtol=1e-12, atol=1e-12))
            or (
                not np.allclose(
                    oof_tss,
                    closure.reference_concentration * np.exp(oof_log),
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
            or (
                not np.any(
                    np.isclose(model.ridge_penalty, RIDGE_GRID, rtol=1e-12, atol=0.0)
                )
            )
        ):
            return None
        scores = pd.read_csv(scores_path)
        fold_frame = pd.read_csv(fold_path)
        if not np.array_equal(fold_frame["row"].to_numpy(), np.arange(len(controls))):
            return None
        if not np.array_equal(fold_frame["fold"].to_numpy(dtype=int), membership):
            return None
        _validate_log_overflow_scores(scores, membership, len(controls))
        return (closure, oof_tss)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def fit_or_resume_log_overflow_closure(
    run: Path,
    controls: np.ndarray,
    influents: np.ndarray,
    mechanistic_responses: np.ndarray,
    *,
    layout: NetworkLayout,
    source_files: Mapping[str, str],
) -> tuple[LogOverflowTSSClosure, np.ndarray, str]:
    from surrogate_optimization.runtime.contracts import array_digest
    from surrogate_optimization.runtime.contracts import assert_source_unchanged
    from surrogate_optimization.runtime.contracts import source_digest
    from surrogate_optimization.runtime.protocols import PROJECTION_SCHEMA

    input_id = array_digest(
        projection_schema=np.frombuffer(
            PROJECTION_SCHEMA.encode("utf-8"), dtype=np.uint8
        ),
        development_controls=np.asarray(controls, dtype="<f8"),
        development_influents=np.asarray(influents, dtype="<f8"),
        development_responses=np.asarray(mechanistic_responses, dtype="<f8"),
    )
    source_id = source_digest(source_files)
    resumed = _load_log_overflow_closure(
        run,
        controls=controls,
        influents=influents,
        mechanistic_responses=mechanistic_responses,
        input_id=input_id,
        source_id=source_id,
        layout=layout,
    )
    if resumed is not None:
        return (resumed[0], resumed[1], input_id)
    result = cross_validate_log_overflow_closure(
        controls, influents, mechanistic_responses, layout=layout
    )
    _validate_log_overflow_scores(result.scores, result.fold_membership, len(controls))
    assert_source_unchanged(source_files)
    save_log_overflow_closure(run, result, input_id=input_id, source_id=source_id)
    return (result.closure, result.out_of_fold_tss, input_id)
