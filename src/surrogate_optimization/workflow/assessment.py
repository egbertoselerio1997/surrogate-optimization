"""Workflow assessment."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from surrogate_optimization.workflow.types import AnalysisBundle
    from surrogate_optimization.validation.assessment import AssessmentResult
    from surrogate_optimization.surrogate.regression import QuadraticSurrogate
    from surrogate_optimization.config import StudyProfile
from pathlib import Path
from typing import Any
from typing import Mapping
import json
import numpy as np
import pandas as pd


def _validate_assessment(
    assessment: AssessmentResult, *, holdout_candidate_count: int, response_count: int
) -> None:
    arrays = (assessment.raw, assessment.projected, assessment.projected_targets)
    if any(
        (
            np.asarray(value).shape != (holdout_candidate_count, response_count)
            or not np.all(np.isfinite(value))
            for value in arrays
        )
    ):
        raise RuntimeError(
            "post-selection holdout predictions are incomplete or non-finite"
        )
    if assessment.overflow_tss_closure is not None:
        closure = np.asarray(assessment.overflow_tss_closure, dtype=float)
        if (
            closure.shape != (holdout_candidate_count,)
            or not np.all(np.isfinite(closure))
            or np.any(closure <= 0.0)
        ):
            raise RuntimeError(
                "post-selection overflow-TSS closure predictions are invalid"
            )
    qp = assessment.qp_diagnostics
    if len(qp) != 2 * holdout_candidate_count or set(qp["projection_input"]) != {
        "raw_prediction",
        "mechanistic_target",
    }:
        raise RuntimeError(
            "projection audit does not cover both inputs for every test row"
        )
    for kind in ("raw_prediction", "mechanistic_target"):
        subset = qp.loc[qp["projection_input"].eq(kind)]
        if len(subset) != holdout_candidate_count or set(
            np.asarray(subset["row"], dtype=int)
        ) != set(range(holdout_candidate_count)):
            raise RuntimeError(f"projection audit coverage is invalid for {kind}")
    violations = assessment.violations
    if len(violations) != 3 * holdout_candidate_count:
        raise RuntimeError(
            "physical audit does not cover raw/projected/mechanistic results"
        )
    for method in ("raw", "projected", "mechanistic"):
        if int(violations["method"].eq(method).sum()) != holdout_candidate_count:
            raise RuntimeError(f"physical audit coverage is invalid for {method}")
    if len(assessment.feasibility) != holdout_candidate_count:
        raise RuntimeError(
            "finite-distance projection bound does not cover every test row"
        )


def assessment_gate_allows_optimization(passed: bool) -> bool:
    """Return whether the recorded gate outcome permits downstream execution."""
    from surrogate_optimization.runtime.protocols import (
        ASSESSMENT_GATE_EXECUTION_POLICY,
    )

    return bool(passed or ASSESSMENT_GATE_EXECUTION_POLICY == "advisory_continue")


def _post_selection_holdout_trust_diagnostics(
    model: QuadraticSurrogate,
    surrogate_assets: Any,
    controls: np.ndarray,
    influents: np.ndarray,
    raw: np.ndarray,
    projected: np.ndarray,
) -> pd.DataFrame:
    """Evaluate the four frozen trust diagnostics on the holdout once.

    This is a descriptive post-selection calculation.  It reuses the stored
    raw and projected holdout responses and the scales/callbacks frozen from
    development; its values are not inputs to calibration or admission.
    """
    controls = np.asarray(controls, dtype=float)
    feed = np.asarray(influents, dtype=float)
    raw_values = np.asarray(raw, dtype=float)
    projected_values = np.asarray(projected, dtype=float)
    if (
        controls.ndim != 2
        or feed.ndim != 2
        or len(controls) < 1
        or (len(feed) != len(controls))
        or (raw_values.ndim != 2)
        or (raw_values.shape != projected_values.shape)
        or (len(raw_values) != len(controls))
        or (
            not all(
                (
                    np.all(np.isfinite(value))
                    for value in (controls, feed, raw_values, projected_values)
                )
            )
        )
    ):
        raise RuntimeError("post-selection holdout trust inputs are invalid")
    response_scale = np.asarray(model.response_scale, dtype=float)
    if (
        response_scale.shape != (raw_values.shape[1],)
        or not np.all(np.isfinite(response_scale))
        or np.any(response_scale <= 0.0)
    ):
        raise RuntimeError("frozen surrogate response scales are invalid")
    features = np.asarray(model.feature_map.transform(controls, feed), dtype=float)
    leverage_precision = np.asarray(surrogate_assets.leverage_precision, dtype=float)
    if (
        features.ndim != 2
        or features.shape[0] != len(controls)
        or leverage_precision.shape != (features.shape[1], features.shape[1])
        or (not np.all(np.isfinite(features)))
        or (not np.all(np.isfinite(leverage_precision)))
    ):
        raise RuntimeError("frozen regularized-leverage assets are invalid")
    callbacks = surrogate_assets.trust_callbacks
    split_rows = getattr(callbacks, "split_rows", None)
    reactor_rows = getattr(callbacks, "reactor_rows", None)
    if not callable(split_rows) or not callable(reactor_rows):
        raise RuntimeError("the two frozen residual trust callbacks are unavailable")
    values = np.empty((len(controls), 4), dtype=float)
    values[:, 0] = np.sqrt(
        np.mean(((projected_values - raw_values) / response_scale) ** 2, axis=1)
    )
    values[:, 1] = np.einsum("ij,jk,ik->i", features, leverage_precision, features)
    for row in range(len(controls)):
        split = np.asarray(
            split_rows(
                controls[row], raw_values[row], projected_values[row], feed[row]
            ),
            dtype=float,
        ).reshape(-1)
        reactor = np.asarray(
            reactor_rows(
                controls[row], raw_values[row], projected_values[row], feed[row]
            ),
            dtype=float,
        ).reshape(-1)
        if (
            split.size < 1
            or reactor.size < 1
            or (not np.all(np.isfinite(split)))
            or (not np.all(np.isfinite(reactor)))
        ):
            raise RuntimeError(
                f"frozen residual trust callbacks failed at holdout row {row}"
            )
        values[row, 2] = float(np.sqrt(np.mean(split**2)))
        values[row, 3] = float(np.sqrt(np.mean(reactor**2)))
    if not np.all(np.isfinite(values)):
        raise RuntimeError("post-selection holdout trust diagnostics are non-finite")
    frame = pd.DataFrame(
        values,
        columns=[
            "correction",
            "regularized_leverage",
            "particulate_split",
            "reactor_residual",
        ],
    )
    frame.insert(0, "row", np.arange(len(controls)))
    return frame


def evaluate_admission_gate(
    assessment: AssessmentResult,
    *,
    correction_limit: float,
    trust_limits: Mapping[str, float],
    development_oof_projection_accepted: np.ndarray,
    development_oof_complete_nrmse: float,
    development_oof_inventory_nrmse: float,
    development_oof_overflow_metrics_complete: bool = True,
    holdout_candidate_count: int,
) -> dict[str, Any]:
    from surrogate_optimization.plant.model import PHYSICAL_BALANCE_TOLERANCE
    from surrogate_optimization.runtime.protocols import (
        ASSESSMENT_GATE_EXECUTION_POLICY,
    )

    _validate_assessment(
        assessment,
        holdout_candidate_count=holdout_candidate_count,
        response_count=assessment.raw.shape[1],
    )
    complete = assessment.metrics.loc[
        assessment.metrics["method"].eq("raw")
        & assessment.metrics["block"].eq("complete_response")
        & assessment.metrics["coordinate"].eq("ALL")
    ]
    if len(complete) != 1:
        raise RuntimeError(
            "complete-response raw assessment row is missing or duplicated"
        )
    holdout_raw_nrmse = float(complete.iloc[0]["nrmse"])
    if not np.isfinite(holdout_raw_nrmse):
        raise RuntimeError(
            "post-selection holdout complete-response raw nRMSE is non-finite"
        )
    if not (
        np.isfinite(development_oof_complete_nrmse)
        and np.isfinite(development_oof_inventory_nrmse)
    ):
        raise RuntimeError("development OOF response gates must be finite")
    qp_accepted = (
        assessment.qp_diagnostics["accepted"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    feasibility = (
        assessment.feasibility["bound_passed"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    if qp_accepted.isna().any() or feasibility.isna().any():
        raise RuntimeError("projection acceptance columns are not Boolean")
    violations = assessment.violations
    physical: dict[str, dict[str, float | bool]] = {}
    for method in ("raw", "projected", "mechanistic"):
        rows = violations.loc[violations["method"].eq(method)]
        mass = float(rows["mass_conservation_violation_max"].max())
        negative = float(rows["nonnegativity_violation_max"].max())
        if not np.isfinite(mass) or not np.isfinite(negative):
            raise RuntimeError(
                f"{method} physical audit contains non-finite gate values"
            )
        physical[method] = {
            "mass_conservation_violation_max": mass,
            "nonnegativity_violation_max": negative,
            "passed": bool(mass <= PHYSICAL_BALANCE_TOLERANCE and negative <= 1e-10),
        }
    limits = {name: float(value) for name, value in trust_limits.items()}
    if set(limits) != {
        "correction",
        "regularized_leverage",
        "particulate_split",
        "reactor_residual",
    } or not all((np.isfinite(value) and value >= 0.0 for value in limits.values())):
        raise RuntimeError("four finite, nonnegative trust limits were not frozen")
    if not np.isclose(limits["correction"], correction_limit, rtol=0.0, atol=0.0):
        raise RuntimeError("the correction gate and frozen trust limit disagree")
    oof_projection_accepted = np.asarray(development_oof_projection_accepted)
    if (
        oof_projection_accepted.ndim != 1
        or oof_projection_accepted.size < 1
        or oof_projection_accepted.dtype.kind != "b"
    ):
        raise RuntimeError(
            "development OOF projection acceptance must be a nonempty Boolean vector"
        )
    admission_checks = {
        "development_oof_complete_response_nrmse_below_one": development_oof_complete_nrmse
        < 1.0,
        "development_oof_clarifier_inventory_nrmse_below_one": development_oof_inventory_nrmse
        < 1.0,
        "all_development_oof_projection_qp_audits_passed": bool(
            oof_projection_accepted.all()
        ),
        "development_oof_log_overflow_metrics_complete": bool(
            development_oof_overflow_metrics_complete
        ),
        "all_four_trust_limits_frozen": True,
        "correction_limit_at_most_0_50": bool(correction_limit <= 0.5),
    }
    holdout_diagnostics = {
        "all_projection_qp_audits_passed": bool(qp_accepted.all()),
        "all_finite_distance_bounds_passed": bool(feasibility.all()),
        "projected_physical_audits_passed": bool(physical["projected"]["passed"]),
        "mechanistic_physical_audits_passed": bool(physical["mechanistic"]["passed"]),
    }
    passed = bool(all(admission_checks.values()))
    optimization_permitted = assessment_gate_allows_optimization(passed)
    return {
        "passed": passed,
        "execution_policy": ASSESSMENT_GATE_EXECUTION_POLICY,
        "optimization_permitted": optimization_permitted,
        "post_selection_holdout_raw_complete_response_nrmse": holdout_raw_nrmse,
        "development_oof_complete_response_nrmse": development_oof_complete_nrmse,
        "development_oof_clarifier_inventory_nrmse": development_oof_inventory_nrmse,
        "post_selection_holdout_is_confirmatory": False,
        "admission_gate_scope": "development_only",
        "post_selection_holdout_checks_are_admission_gates": False,
        **admission_checks,
        **holdout_diagnostics,
        "trust_limits": limits,
        "physical_audit_maxima": physical,
        "failure_action": None
        if passed
        else "record advisory failure and continue without refitting",
    }


def _assessment_binding(
    design: Mapping[str, object],
    development_responses: np.ndarray,
    holdout_responses: np.ndarray,
) -> str:
    from surrogate_optimization.plant.definitions import N_COMPONENTS
    from surrogate_optimization.plant.definitions import N_STAGES
    from surrogate_optimization.runtime.contracts import array_digest
    from surrogate_optimization.runtime.protocols import PROJECTION_SCHEMA
    from surrogate_optimization.runtime.protocols import RESPONSE_SCHEMA
    from surrogate_optimization.validation.physical import reduce_mechanistic_responses

    shared_count = (N_STAGES + 3) * N_COMPONENTS
    layer_count = int(development_responses.shape[1] - shared_count)
    if layer_count < 1 or holdout_responses.shape[1] != shared_count + layer_count:
        raise RuntimeError("mechanistic response blocks have inconsistent dimensions")
    development_reduced = reduce_mechanistic_responses(
        development_responses, layer_count
    )
    test_reduced = reduce_mechanistic_responses(holdout_responses, layer_count)
    return array_digest(
        development_controls=np.asarray(design["development_controls"], dtype="<f8"),
        development_influents=np.asarray(design["development_influents"], dtype="<f8"),
        development_responses=np.asarray(development_responses, dtype="<f8"),
        development_reduced=np.asarray(development_reduced, dtype="<f8"),
        holdout_controls=np.asarray(design["holdout_controls"], dtype="<f8"),
        holdout_influents=np.asarray(design["holdout_influents"], dtype="<f8"),
        holdout_responses=np.asarray(holdout_responses, dtype="<f8"),
        test_reduced=np.asarray(test_reduced, dtype="<f8"),
        response_schema=np.frombuffer(RESPONSE_SCHEMA.encode("utf-8"), dtype=np.uint8),
        projection_schema=np.frombuffer(
            PROJECTION_SCHEMA.encode("utf-8"), dtype=np.uint8
        ),
    )


def _materialize_reduced_response_block(
    run: Path, *, block: str, mechanistic_targets: np.ndarray, profile: StudyProfile
) -> np.ndarray:
    """Derive and immutably checkpoint the statistical response block."""
    from surrogate_optimization.runtime.artifacts import atomic_npz
    from surrogate_optimization.runtime.contracts import array_digest
    from surrogate_optimization.runtime.protocols import RESPONSE_SCHEMA
    from surrogate_optimization.validation.physical import reduce_mechanistic_responses

    full = np.asarray(mechanistic_targets, dtype=np.float64)
    if full.ndim != 2 or full.shape[1] != profile.mechanistic_response_count:
        raise RuntimeError(
            f"{block} mechanistic responses have the wrong response width"
        )
    layer_volumes = np.full(
        profile.layer_count, 6000.0 / profile.layer_count, dtype=np.float64
    )
    reduced = reduce_mechanistic_responses(
        full, profile.layer_count, layer_volumes_m3=layer_volumes
    )
    if reduced.shape != (len(full), profile.surrogate_response_count):
        raise RuntimeError(
            f"{block} reduced response transformation has the wrong shape"
        )
    path = run / "datasets" / block / "surrogate_responses.npz"
    full_digest = array_digest(mechanistic_targets=np.asarray(full, dtype="<f8"))
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            valid = bool(
                set(stored.files)
                == {
                    "schema",
                    "mechanistic_target_digest",
                    "layer_volumes_m3",
                    "responses",
                }
                and str(stored["schema"].item()) == RESPONSE_SCHEMA
                and (str(stored["mechanistic_target_digest"].item()) == full_digest)
                and np.array_equal(stored["layer_volumes_m3"], layer_volumes)
                and np.array_equal(stored["responses"], reduced)
            )
        if not valid:
            raise RuntimeError(
                f"existing {block} reduced-response artifact is inconsistent"
            )
        return reduced
    atomic_npz(
        path,
        schema=np.asarray(RESPONSE_SCHEMA),
        mechanistic_target_digest=np.asarray(full_digest),
        layer_volumes_m3=layer_volumes,
        responses=reduced,
    )
    return reduced


def load_assessment_checkpoint(
    run: Path, *, source_id: str, input_id: str
) -> dict[str, Any] | None:
    from surrogate_optimization.runtime.contracts import _artifacts_match
    from surrogate_optimization.runtime.contracts import (
        _checkpoint_source_is_authorized,
    )
    from surrogate_optimization.runtime.protocols import (
        ASSESSMENT_GATE_EXECUTION_POLICY,
    )

    marker_path = run / "metrics" / "assessment_complete.json"
    gate_path = run / "metrics" / "admission_gate.json"
    if not marker_path.is_file() or not gate_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_source_id = str(marker.get("source_digest", ""))
        if (
            not _checkpoint_source_is_authorized(
                run,
                stage="assessment",
                checkpoint=marker_path,
                observed_source_id=marker_source_id,
                current_source_id=source_id,
            )
            or marker.get("input_digest") != input_id
            or (not _artifacts_match(run, marker.get("artifacts", {})))
        ):
            raise RuntimeError("completed checkpoint is inconsistent or changed")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            not isinstance(gate.get("passed"), bool)
            or gate.get("execution_policy") != ASSESSMENT_GATE_EXECUTION_POLICY
            or gate.get("optimization_permitted")
            != assessment_gate_allows_optimization(gate["passed"])
        ):
            raise RuntimeError("completed checkpoint is inconsistent or changed")
        return gate
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        raise RuntimeError("completed checkpoint is corrupt")


def run_assessment(
    run: Path,
    design: Mapping[str, object],
    development_responses: np.ndarray,
    holdout_responses: np.ndarray,
    *,
    profile: StudyProfile,
    source_files: Mapping[str, str],
) -> AnalysisBundle:
    from surrogate_optimization.data.design import clarifier_for
    from surrogate_optimization.optimization.mechanistic import fit_mechanistic_assets
    from surrogate_optimization.optimization.surrogate import build_surrogate_assets
    from surrogate_optimization.optimization.trust import calibrate_trust_diagnostics
    from surrogate_optimization.runtime.artifacts import atomic_dataframe
    from surrogate_optimization.runtime.artifacts import atomic_json
    from surrogate_optimization.runtime.artifacts import atomic_npz
    from surrogate_optimization.runtime.contracts import _artifact_hashes
    from surrogate_optimization.runtime.contracts import assert_source_unchanged
    from surrogate_optimization.runtime.contracts import source_digest
    from surrogate_optimization.surrogate.projection import NetworkLayout
    from surrogate_optimization.surrogate.training import (
        fit_or_resume_log_overflow_closure,
    )
    from surrogate_optimization.surrogate.training import fit_or_resume_ridge
    from surrogate_optimization.surrogate.training import overflow_tss_from_response
    from surrogate_optimization.validation.assessment import (
        assess_raw_projected_mechanistic,
    )
    from surrogate_optimization.workflow.types import AnalysisBundle

    development_controls = np.asarray(design["development_controls"])
    development_influents = np.asarray(design["development_influents"])
    if len(development_controls) < 5:
        raise RuntimeError(
            "assessment requires at least five accepted development rows"
        )
    if len(np.asarray(design["holdout_controls"])) < 1:
        raise RuntimeError("assessment requires at least one accepted holdout row")
    development_reduced = _materialize_reduced_response_block(
        run,
        block="development",
        mechanistic_targets=development_responses,
        profile=profile,
    )
    test_reduced = _materialize_reduced_response_block(
        run, block="holdout", mechanistic_targets=holdout_responses, profile=profile
    )
    input_id = _assessment_binding(design, development_responses, holdout_responses)
    source_id = source_digest(source_files)
    existing_gate = load_assessment_checkpoint(
        run, source_id=source_id, input_id=input_id
    )
    model, oof_raw, _ = fit_or_resume_ridge(
        run,
        development_controls,
        development_influents,
        development_reduced,
        source_files=source_files,
    )
    layout = NetworkLayout(layer_count=profile.layer_count)
    overflow_closure, oof_overflow_tss, _ = fit_or_resume_log_overflow_closure(
        run,
        development_controls,
        development_influents,
        development_reduced,
        layout=layout,
        source_files=source_files,
    )
    mechanistic_assets = fit_mechanistic_assets(
        development_controls,
        development_influents,
        development_responses,
        clarifier=clarifier_for(profile),
    )
    trust = calibrate_trust_diagnostics(
        model,
        development_controls,
        development_influents,
        development_reduced,
        oof_raw,
        mechanistic_assets,
        layout=layout,
        overflow_closure=overflow_closure,
        out_of_fold_overflow_tss=oof_overflow_tss,
    )
    surrogate_assets = build_surrogate_assets(
        model,
        development_controls,
        development_influents,
        development_reduced,
        layout=layout,
        correction_rms_threshold=trust.correction_limit,
        trust_callbacks=trust.callbacks,
        split_rms_threshold=trust.split_limit,
        reactor_rms_threshold=trust.reactor_limit,
        overflow_closure=overflow_closure,
        development_overflow_tss_closure=oof_overflow_tss,
    )
    features = model.feature_map.transform(development_controls, development_influents)
    leverage = np.einsum(
        "ij,jk,ik->i", features, surrogate_assets.leverage_precision, features
    )
    trust_values = np.column_stack(
        (trust.development_values[:, 0], leverage, trust.development_values[:, 1:])
    )
    development_candidate_count = len(development_controls)
    holdout_candidate_count = len(np.asarray(design["holdout_controls"]))
    if trust_values.shape != (development_candidate_count, 4) or not np.all(
        np.isfinite(trust_values)
    ):
        raise RuntimeError("four development trust diagnostics were not evaluated")
    limits = {
        "correction": float(trust.correction_limit),
        "regularized_leverage": float(
            surrogate_assets.trust_thresholds.regularized_leverage
        ),
        "particulate_split": float(trust.split_limit),
        "reactor_residual": float(trust.reactor_limit),
    }
    if existing_gate is not None:
        return AnalysisBundle(
            passed=bool(existing_gate["passed"]),
            model=model,
            mechanistic_assets=mechanistic_assets,
            surrogate_assets=surrogate_assets,
            assessment=None,
            gate=existing_gate,
            overflow_closure=overflow_closure,
        )
    trust_frame = pd.DataFrame(
        trust_values,
        columns=[
            "correction",
            "regularized_leverage",
            "particulate_split",
            "reactor_residual",
        ],
    )
    trust_frame.insert(0, "row", np.arange(development_candidate_count))
    trust_frame.insert(
        1, "projection_qp_accepted", trust.out_of_fold_projection_accepted
    )
    atomic_dataframe(run / "metrics" / "trust_development_oof.csv", trust_frame)
    atomic_json(run / "metrics" / "trust_limits.json", limits)
    atomic_npz(
        run / "models" / "trust_calibration.npz",
        development_values=trust_values,
        out_of_fold_projected=trust.out_of_fold_projected,
        out_of_fold_projection_accepted=trust.out_of_fold_projection_accepted,
        split_scale=trust.split_scale,
    )
    assessment = assess_raw_projected_mechanistic(
        model,
        development_controls,
        development_influents,
        development_reduced,
        np.asarray(design["holdout_controls"]),
        np.asarray(design["holdout_influents"]),
        test_reduced,
        profile,
        overflow_closure=overflow_closure,
        development_overflow_tss_closure=oof_overflow_tss,
    )
    _validate_assessment(
        assessment,
        holdout_candidate_count=holdout_candidate_count,
        response_count=profile.surrogate_response_count,
    )
    holdout_trust_path = run / "metrics" / "trust_post_selection_holdout.csv"
    paths = (
        run / "metrics" / "post_selection_prediction_metrics.csv",
        run / "metrics" / "physical_violations_assessment.csv",
        run / "metrics" / "projection_qp_diagnostics.csv",
        run / "metrics" / "projection_feasibility_bound.csv",
        run / "predictions" / "post_selection_holdout.npz",
        run / "metrics" / "trust_development_oof.csv",
        run / "metrics" / "trust_limits.json",
        run / "models" / "trust_calibration.npz",
        run / "metrics" / "admission_gate.json",
        run / "models" / "ridge_surrogate.npz",
        run / "models" / "ridge_complete.json",
        run / "metrics" / "ridge_cross_validation.csv",
        run / "metrics" / "ridge_fold_membership.csv",
        run / "datasets" / "development" / "surrogate_responses.npz",
        run / "datasets" / "holdout" / "surrogate_responses.npz",
        holdout_trust_path,
        run / "models" / "log_overflow_closure.npz",
        run / "models" / "log_overflow_closure_complete.json",
        run / "metrics" / "log_overflow_closure_cross_validation.csv",
        run / "metrics" / "log_overflow_closure_fold_membership.csv",
        run / "metrics" / "log_overflow_closure_development_oof.json",
    )
    atomic_dataframe(paths[0], assessment.metrics)
    atomic_dataframe(paths[1], assessment.violations)
    atomic_dataframe(paths[2], assessment.qp_diagnostics)
    atomic_dataframe(paths[3], assessment.feasibility)
    atomic_npz(
        paths[4],
        raw=assessment.raw,
        projected=assessment.projected,
        projected_targets=assessment.projected_targets,
        mechanistic=test_reduced,
        mechanistic_full=holdout_responses,
        overflow_tss_closure=assessment.overflow_tss_closure,
    )
    oof_scaled = (oof_raw - development_reduced) / model.response_scale
    development_oof_complete_nrmse = float(np.sqrt(np.mean(oof_scaled**2)))
    development_oof_inventory_nrmse = float(
        np.sqrt(np.mean(oof_scaled[:, layout.inventory_index] ** 2))
    )
    exact_development_overflow = overflow_tss_from_response(
        development_reduced, development_controls, layout
    )
    oof_closure_error = oof_overflow_tss - exact_development_overflow
    oof_log_error = np.log(oof_overflow_tss) - np.log(exact_development_overflow)
    oof_closure_metrics = {
        "sample_count": int(len(oof_overflow_tss)),
        "rmse_mg_L": float(np.sqrt(np.mean(np.square(oof_closure_error)))),
        "mae_mg_L": float(np.mean(np.abs(oof_closure_error))),
        "bias_mg_L": float(np.mean(oof_closure_error)),
        "log_rmse": float(np.sqrt(np.mean(np.square(oof_log_error)))),
        "log_bias": float(np.mean(oof_log_error)),
        "minimum_prediction_mg_L": float(np.min(oof_overflow_tss)),
        "maximum_prediction_mg_L": float(np.max(oof_overflow_tss)),
        "all_finite_and_positive": bool(
            np.all(np.isfinite(oof_overflow_tss)) and np.all(oof_overflow_tss > 0.0)
        ),
    }
    atomic_json(paths[-1], oof_closure_metrics)
    gate = evaluate_admission_gate(
        assessment,
        correction_limit=trust.correction_limit,
        trust_limits=limits,
        development_oof_projection_accepted=trust.out_of_fold_projection_accepted,
        development_oof_complete_nrmse=development_oof_complete_nrmse,
        development_oof_inventory_nrmse=development_oof_inventory_nrmse,
        development_oof_overflow_metrics_complete=bool(
            oof_closure_metrics["all_finite_and_positive"]
            and all(
                (
                    np.isfinite(float(oof_closure_metrics[name]))
                    for name in (
                        "rmse_mg_L",
                        "mae_mg_L",
                        "bias_mg_L",
                        "log_rmse",
                        "log_bias",
                    )
                )
            )
        ),
        holdout_candidate_count=holdout_candidate_count,
    )
    holdout_trust = _post_selection_holdout_trust_diagnostics(
        model,
        surrogate_assets,
        np.asarray(design["holdout_controls"]),
        np.asarray(design["holdout_influents"]),
        assessment.raw,
        assessment.projected,
    )
    atomic_dataframe(holdout_trust_path, holdout_trust)
    atomic_json(paths[8], gate)
    assert_source_unchanged(source_files)
    atomic_json(
        run / "metrics" / "assessment_complete.json",
        {
            "stage": "post_selection_holdout_assessment",
            "source_digest": source_id,
            "input_digest": input_id,
            "passed": gate["passed"],
            "artifacts": _artifact_hashes(run, paths),
        },
    )
    return AnalysisBundle(
        passed=bool(gate["passed"]),
        model=model,
        mechanistic_assets=mechanistic_assets,
        surrogate_assets=surrogate_assets,
        assessment=assessment,
        gate=gate,
        overflow_closure=overflow_closure,
    )
