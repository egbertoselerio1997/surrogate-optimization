"""Surrogate regression."""

from __future__ import annotations
from dataclasses import asdict
from dataclasses import dataclass
from scipy import linalg
from typing import Mapping
from typing import Sequence
import json
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class SurrogateValidationError(ValueError):
    """Raised when a scientific precondition for fitting is not satisfied."""


class ProjectionError(RuntimeError):
    """Raised when the physical lower problem cannot be accepted."""


def _finite_array(
    value: npt.ArrayLike, *, name: str, ndim: int | None = None
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise SurrogateValidationError(
            f"{name} must be {ndim}-dimensional; got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise SurrogateValidationError(f"{name} contains a non-finite value.")
    return array


def _sample_matrix(value: npt.ArrayLike, *, name: str) -> tuple[FloatArray, bool]:
    array = _finite_array(value, name=name)
    if array.ndim == 1:
        return (array[None, :], True)
    if array.ndim != 2:
        raise SurrogateValidationError(
            f"{name} must be a vector or a sample-by-coordinate matrix."
        )
    return (array, False)


def _fit_coordinate_scale(
    values: FloatArray, *, name: str, variance_relative_tolerance: float
) -> tuple[FloatArray, FloatArray]:
    if values.ndim != 2 or values.shape[0] < 2:
        raise SurrogateValidationError(f"{name} requires at least two sample rows.")
    center = np.mean(values, axis=0)
    scale = np.sqrt(np.mean(np.square(values - center), axis=0))
    reference = np.maximum(1.0, np.max(np.abs(values), axis=0))
    bad = np.flatnonzero(scale <= variance_relative_tolerance * reference)
    if bad.size:
        preview = ", ".join((str(int(index)) for index in bad[:10]))
        suffix = "..." if bad.size > 10 else ""
        raise SurrogateValidationError(
            f"{name} coordinates [{preview}{suffix}] fail the nonzero-variance rule."
        )
    return (center, scale)


def _upper_triangle_products(values: FloatArray) -> FloatArray:
    """Return v_j v_k in lexicographic (j, k), j <= k, order."""
    indices = np.triu_indices(values.shape[1])
    return values[:, indices[0]] * values[:, indices[1]]


@dataclass(frozen=True)
class QuadraticFeatureMap:
    """Fitted standardization and serialization for the unique quadratic basis."""

    decision_center: FloatArray
    decision_scale: FloatArray
    influent_center: FloatArray
    influent_scale: FloatArray
    term_center: FloatArray
    term_scale: FloatArray
    variance_relative_tolerance: float = 1e-12

    @property
    def decision_count(self) -> int:
        return int(self.decision_center.size)

    @property
    def influent_count(self) -> int:
        return int(self.influent_center.size)

    @property
    def nonconstant_count(self) -> int:
        return int(self.term_center.size)

    @property
    def feature_count(self) -> int:
        return self.nonconstant_count + 1

    @staticmethod
    def expected_feature_count(decision_count: int, influent_count: int) -> int:
        return (
            1
            + decision_count
            + influent_count
            + decision_count * (decision_count + 1) // 2
            + influent_count * (influent_count + 1) // 2
            + decision_count * influent_count
        )

    @staticmethod
    def _unscaled_terms(
        controls_standardized: FloatArray, influent_standardized: FloatArray
    ) -> FloatArray:
        decision_quadratic = _upper_triangle_products(controls_standardized)
        influent_quadratic = _upper_triangle_products(influent_standardized)
        interactions = np.einsum(
            "ni,nj->nij", controls_standardized, influent_standardized
        ).reshape(controls_standardized.shape[0], -1)
        return np.concatenate(
            (
                controls_standardized,
                influent_standardized,
                decision_quadratic,
                influent_quadratic,
                interactions,
            ),
            axis=1,
        )

    @classmethod
    def fit(
        cls,
        controls: npt.ArrayLike,
        influent: npt.ArrayLike,
        *,
        variance_relative_tolerance: float = 1e-12,
    ) -> "QuadraticFeatureMap":
        decision_matrix = _finite_array(controls, name="controls", ndim=2)
        influent_matrix = _finite_array(influent, name="influent", ndim=2)
        if decision_matrix.shape[0] != influent_matrix.shape[0]:
            raise SurrogateValidationError(
                "controls and influent must have the same row count."
            )
        if decision_matrix.shape[1] == 0 or influent_matrix.shape[1] == 0:
            raise SurrogateValidationError(
                "decision and influent blocks must both be nonempty."
            )
        decision_center, decision_scale = _fit_coordinate_scale(
            decision_matrix,
            name="decision",
            variance_relative_tolerance=variance_relative_tolerance,
        )
        influent_center, influent_scale = _fit_coordinate_scale(
            influent_matrix,
            name="influent",
            variance_relative_tolerance=variance_relative_tolerance,
        )
        decision_standardized = (decision_matrix - decision_center) / decision_scale
        influent_standardized = (influent_matrix - influent_center) / influent_scale
        terms = cls._unscaled_terms(decision_standardized, influent_standardized)
        term_center, term_scale = _fit_coordinate_scale(
            terms,
            name="nonconstant feature",
            variance_relative_tolerance=variance_relative_tolerance,
        )
        expected = cls.expected_feature_count(
            decision_matrix.shape[1], influent_matrix.shape[1]
        )
        if terms.shape[1] + 1 != expected:
            raise AssertionError(
                "quadratic feature serialization has an inconsistent size"
            )
        return cls(
            decision_center=decision_center,
            decision_scale=decision_scale,
            influent_center=influent_center,
            influent_scale=influent_scale,
            term_center=term_center,
            term_scale=term_scale,
            variance_relative_tolerance=float(variance_relative_tolerance),
        )

    def transform(self, controls: npt.ArrayLike, influent: npt.ArrayLike) -> FloatArray:
        decision_matrix, decision_single = _sample_matrix(controls, name="controls")
        influent_matrix, influent_single = _sample_matrix(influent, name="influent")
        if decision_single != influent_single:
            raise SurrogateValidationError(
                "controls and influent must both be vectors or both matrices."
            )
        if decision_matrix.shape != (influent_matrix.shape[0], self.decision_count):
            raise SurrogateValidationError(
                f"controls must have shape ({influent_matrix.shape[0]}, {self.decision_count})."
            )
        if influent_matrix.shape[1] != self.influent_count:
            raise SurrogateValidationError(
                f"influent must have {self.influent_count} coordinates; got {influent_matrix.shape[1]}."
            )
        decision_standardized = (
            decision_matrix - self.decision_center
        ) / self.decision_scale
        influent_standardized = (
            influent_matrix - self.influent_center
        ) / self.influent_scale
        terms = self._unscaled_terms(decision_standardized, influent_standardized)
        standardized_terms = (terms - self.term_center) / self.term_scale
        features = np.concatenate(
            (np.ones((terms.shape[0], 1), dtype=np.float64), standardized_terms), axis=1
        )
        return features[0] if decision_single else features

    def feature_names(
        self,
        decision_names: Sequence[str] | None = None,
        influent_names: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        controls = tuple(
            decision_names or (f"decision_{i}" for i in range(self.decision_count))
        )
        influents = tuple(
            influent_names or (f"influent_{i}" for i in range(self.influent_count))
        )
        if (
            len(controls) != self.decision_count
            or len(influents) != self.influent_count
        ):
            raise SurrogateValidationError(
                "feature-name blocks do not match the fitted dimensions."
            )
        labels: list[str] = list(controls) + list(influents)
        labels.extend(
            (
                f"{controls[j]}*{controls[k]}"
                for j in range(self.decision_count)
                for k in range(j, self.decision_count)
            )
        )
        labels.extend(
            (
                f"{influents[j]}*{influents[k]}"
                for j in range(self.influent_count)
                for k in range(j, self.influent_count)
            )
        )
        labels.extend(
            (
                f"{decision}*{influent}"
                for decision in controls
                for influent in influents
            )
        )
        return ("1", *(f"standardized[{label}]" for label in labels))


@dataclass(frozen=True)
class LeastSquaresDiagnostics:
    sample_count: int
    feature_count: int
    response_count: int
    rank_tolerance: float
    smallest_singular_value: float
    largest_singular_value: float
    condition_number: float
    optimality_residual: float
    coefficient_agreement: float
    acceptance_threshold: float
    augmented_condition_number: float | None = None
    condition_times_machine_epsilon: float | None = None
    effective_degrees_of_freedom: float | None = None

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class QuadraticSurrogate:
    """A fixed standardized multiresponse quadratic model."""

    feature_map: QuadraticFeatureMap
    response_center: FloatArray
    response_scale: FloatArray
    coefficients: FloatArray
    diagnostics: LeastSquaresDiagnostics
    ridge_penalty: float = 0.0

    @classmethod
    def fit(
        cls,
        controls: npt.ArrayLike,
        influent: npt.ArrayLike,
        responses: npt.ArrayLike,
        *,
        variance_relative_tolerance: float = 1e-12,
        maximum_condition_number: float = 100000000.0,
        coefficient_acceptance_factor: float = 100.0,
    ) -> "QuadraticSurrogate":
        decision_matrix = _finite_array(controls, name="controls", ndim=2)
        influent_matrix = _finite_array(influent, name="influent", ndim=2)
        response_matrix = _finite_array(responses, name="responses", ndim=2)
        row_count = decision_matrix.shape[0]
        if (
            influent_matrix.shape[0] != row_count
            or response_matrix.shape[0] != row_count
        ):
            raise SurrogateValidationError(
                "all fitting blocks must have the same row count."
            )
        feature_map = QuadraticFeatureMap.fit(
            decision_matrix,
            influent_matrix,
            variance_relative_tolerance=variance_relative_tolerance,
        )
        design = feature_map.transform(decision_matrix, influent_matrix)
        if row_count < design.shape[1]:
            raise SurrogateValidationError(
                f"the fit has {row_count} rows for {design.shape[1]} coefficients per response."
            )
        response_center, response_scale = _fit_coordinate_scale(
            response_matrix,
            name="response",
            variance_relative_tolerance=variance_relative_tolerance,
        )
        standardized_response = (response_matrix - response_center) / response_scale
        u_svd, singular_values, vt_svd = linalg.svd(
            design, full_matrices=False, check_finite=True, lapack_driver="gesdd"
        )
        epsilon = np.finfo(np.float64).eps
        rank_tolerance = max(design.shape) * epsilon * singular_values[0]
        smallest = float(singular_values[-1])
        condition = (
            float(singular_values[0] / singular_values[-1])
            if smallest > 0.0
            else np.inf
        )
        if smallest <= rank_tolerance:
            raise SurrogateValidationError(
                "the standardized quadratic design is not full column rank at the declared tolerance."
            )
        if condition > maximum_condition_number:
            raise SurrogateValidationError(
                f"design condition number {condition:.6g} exceeds {maximum_condition_number:.6g}."
            )
        q_qr, r_qr, pivot = linalg.qr(
            design, mode="economic", pivoting=True, check_finite=True
        )
        permuted_coefficients = linalg.solve_triangular(
            r_qr, q_qr.T @ standardized_response, lower=False, check_finite=True
        )
        coefficients_qr = np.empty_like(permuted_coefficients)
        coefficients_qr[pivot, :] = permuted_coefficients
        preliminary_residual = design @ coefficients_qr - standardized_response
        permuted_correction = linalg.solve_triangular(
            r_qr, q_qr.T @ -preliminary_residual, lower=False, check_finite=True
        )
        correction = np.empty_like(permuted_correction)
        correction[pivot, :] = permuted_correction
        coefficients_qr += correction
        coefficients_svd = (
            vt_svd.T / singular_values[None, :] @ (u_svd.T @ standardized_response)
        )
        residual = design @ coefficients_qr - standardized_response
        residual_norm = float(linalg.norm(residual, ord="fro"))
        optimality = float(linalg.norm(design.T @ residual, ord="fro")) / max(
            1.0, float(singular_values[0]) * residual_norm
        )
        agreement = float(
            linalg.norm(coefficients_qr - coefficients_svd, ord="fro")
        ) / max(1.0, float(linalg.norm(coefficients_svd, ord="fro")))
        threshold = float(coefficient_acceptance_factor * condition * epsilon)
        if (
            not np.all(np.isfinite(coefficients_qr))
            or max(optimality, agreement) > threshold
        ):
            raise SurrogateValidationError(
                f"QR/SVD coefficient acceptance failed: optimality={optimality:.3e}, agreement={agreement:.3e}, limit={threshold:.3e}."
            )
        diagnostics = LeastSquaresDiagnostics(
            sample_count=row_count,
            feature_count=design.shape[1],
            response_count=response_matrix.shape[1],
            rank_tolerance=float(rank_tolerance),
            smallest_singular_value=smallest,
            largest_singular_value=float(singular_values[0]),
            condition_number=condition,
            optimality_residual=optimality,
            coefficient_agreement=agreement,
            acceptance_threshold=threshold,
        )
        return cls(
            feature_map=feature_map,
            response_center=response_center,
            response_scale=response_scale,
            coefficients=coefficients_qr.T,
            diagnostics=diagnostics,
            ridge_penalty=0.0,
        )

    @classmethod
    def fit_ridge(
        cls,
        controls: npt.ArrayLike,
        influent: npt.ArrayLike,
        responses: npt.ArrayLike,
        *,
        ridge_penalty: float,
        variance_relative_tolerance: float = 1e-12,
    ) -> "QuadraticSurrogate":
        """Fit the model ridge estimator, leaving the intercept unpenalized.

        The coefficient estimate is the column-pivoted-QR solution of the
        augmented least-squares system.  A separate divide-and-conquer SVD
        solve supplies the coefficient audit; forming and solving the ridge
        normal equations is deliberately avoided.
        """
        if not np.isfinite(ridge_penalty) or ridge_penalty <= 0.0:
            raise SurrogateValidationError("ridge_penalty must be finite and positive.")
        decision_matrix = _finite_array(controls, name="controls", ndim=2)
        influent_matrix = _finite_array(influent, name="influent", ndim=2)
        response_matrix = _finite_array(responses, name="responses", ndim=2)
        rows = decision_matrix.shape[0]
        if influent_matrix.shape[0] != rows or response_matrix.shape[0] != rows:
            raise SurrogateValidationError(
                "all fitting blocks must have the same row count."
            )
        feature_map = QuadraticFeatureMap.fit(
            decision_matrix,
            influent_matrix,
            variance_relative_tolerance=variance_relative_tolerance,
        )
        design = feature_map.transform(decision_matrix, influent_matrix)
        response_center, response_scale = _fit_coordinate_scale(
            response_matrix,
            name="response",
            variance_relative_tolerance=variance_relative_tolerance,
        )
        standardized = (response_matrix - response_center) / response_scale
        feature_count = design.shape[1]
        penalty_diagonal = np.ones(feature_count, dtype=np.float64)
        penalty_diagonal[0] = 0.0
        penalty_operator = np.diag(penalty_diagonal)
        ridge_scale = np.sqrt(float(rows)) * np.sqrt(ridge_penalty)
        augmented_design = np.vstack((design, ridge_scale * penalty_operator))
        augmented_response = np.vstack(
            (standardized, np.zeros((feature_count, response_matrix.shape[1])))
        )
        q_qr, r_qr, pivot = linalg.qr(
            augmented_design, mode="economic", pivoting=True, check_finite=True
        )
        permuted_coefficients = linalg.solve_triangular(
            r_qr, q_qr.T @ augmented_response, lower=False, check_finite=True
        )
        coefficients_qr = np.empty_like(permuted_coefficients)
        coefficients_qr[pivot, :] = permuted_coefficients
        u_svd, singular_values, vt_svd = linalg.svd(
            augmented_design,
            full_matrices=False,
            check_finite=True,
            lapack_driver="gesdd",
        )
        epsilon = np.finfo(np.float64).eps
        rank_tolerance = max(augmented_design.shape) * epsilon * singular_values[0]
        svd_rank = int(np.count_nonzero(singular_values > rank_tolerance))
        coefficients_svd = (
            vt_svd.T / singular_values[None, :] @ (u_svd.T @ augmented_response)
        )
        smallest = float(singular_values[-1])
        largest = float(singular_values[0])
        condition = largest / smallest if smallest > 0.0 else np.inf
        condition_product = condition * epsilon
        if svd_rank != feature_count or condition_product > 1e-08:
            raise SurrogateValidationError(
                f"augmented ridge-system condition gate failed: rank={svd_rank}/{feature_count}, kappa*eps={condition_product:.3e} > 1.000e-08."
            )
        coefficient_scale = 1.0 + max(
            float(np.max(np.abs(coefficients_qr))),
            float(np.max(np.abs(coefficients_svd))),
        )
        agreement = (
            float(np.max(np.abs(coefficients_qr - coefficients_svd)))
            / coefficient_scale
        )
        threshold = float(100.0 * condition_product)
        if (
            not np.all(np.isfinite(coefficients_qr))
            or not np.all(np.isfinite(coefficients_svd))
            or agreement > threshold
        ):
            raise SurrogateValidationError(
                f"augmented QR/SVD coefficient acceptance failed: agreement={agreement:.3e}, limit={threshold:.3e}."
            )
        augmented_residual = augmented_design @ coefficients_qr - augmented_response
        augmented_gradient = augmented_design.T @ augmented_residual
        optimality = float(linalg.norm(augmented_gradient, ord="fro")) / max(
            1.0, largest * float(linalg.norm(augmented_residual, ord="fro"))
        )
        design_pivoted_times_r_inverse = linalg.solve_triangular(
            r_qr.T, design[:, pivot].T, lower=True, check_finite=True
        ).T
        effective_degrees_of_freedom = float(
            np.sum(np.square(design_pivoted_times_r_inverse))
        )
        diagnostics = LeastSquaresDiagnostics(
            sample_count=rows,
            feature_count=feature_count,
            response_count=response_matrix.shape[1],
            rank_tolerance=float(rank_tolerance),
            smallest_singular_value=smallest,
            largest_singular_value=largest,
            condition_number=condition,
            optimality_residual=optimality,
            coefficient_agreement=agreement,
            acceptance_threshold=threshold,
            augmented_condition_number=condition,
            condition_times_machine_epsilon=condition_product,
            effective_degrees_of_freedom=effective_degrees_of_freedom,
        )
        return cls(
            feature_map=feature_map,
            response_center=response_center,
            response_scale=response_scale,
            coefficients=coefficients_qr.T,
            diagnostics=diagnostics,
            ridge_penalty=float(ridge_penalty),
        )

    @property
    def response_count(self) -> int:
        return int(self.response_center.size)

    @property
    def effective_degrees_of_freedom(self) -> float:
        """Return the ridge hat-matrix trace (or the OLS column count)."""
        value = self.diagnostics.effective_degrees_of_freedom
        return float(self.feature_map.feature_count if value is None else value)

    def predict_standardized(
        self, controls: npt.ArrayLike, influent: npt.ArrayLike
    ) -> FloatArray:
        features = self.feature_map.transform(controls, influent)
        if features.ndim == 1:
            return self.coefficients @ features
        return features @ self.coefficients.T

    def predict(self, controls: npt.ArrayLike, influent: npt.ArrayLike) -> FloatArray:
        standardized = self.predict_standardized(controls, influent)
        return self.response_center + standardized * self.response_scale


@dataclass(frozen=True)
class LogOverflowTSSClosure:
    """Positive scalar overflow-TSS model used by the system-wide projection.

    The wrapped quadratic ridge model predicts ``log(X_E / reference)``.  The
    raw multiresponse surrogate is deliberately kept separate and unchanged.
    """

    model: QuadraticSurrogate
    reference_concentration: float = 1.0

    def __post_init__(self) -> None:
        if self.model.response_count != 1:
            raise SurrogateValidationError(
                "the log-overflow closure must contain exactly one response"
            )
        reference = float(self.reference_concentration)
        if not np.isfinite(reference) or reference <= 0.0:
            raise SurrogateValidationError(
                "reference_concentration must be finite and strictly positive"
            )
        object.__setattr__(self, "reference_concentration", reference)

    @classmethod
    def from_serialized_arrays(
        cls, arrays: Mapping[str, npt.ArrayLike]
    ) -> "LogOverflowTSSClosure":
        """Reconstruct a closure from the immutable runner NPZ fields."""
        feature_map = QuadraticFeatureMap(
            decision_center=np.asarray(arrays["decision_center"], dtype=float),
            decision_scale=np.asarray(arrays["decision_scale"], dtype=float),
            influent_center=np.asarray(arrays["influent_center"], dtype=float),
            influent_scale=np.asarray(arrays["influent_scale"], dtype=float),
            term_center=np.asarray(arrays["term_center"], dtype=float),
            term_scale=np.asarray(arrays["term_scale"], dtype=float),
            variance_relative_tolerance=float(
                np.asarray(arrays["variance_relative_tolerance"], dtype=float)
            ),
        )
        diagnostics = LeastSquaresDiagnostics(
            **json.loads(str(np.asarray(arrays["diagnostics_json"]).item()))
        )
        model = QuadraticSurrogate(
            feature_map=feature_map,
            response_center=np.asarray(arrays["response_center"], dtype=float),
            response_scale=np.asarray(arrays["response_scale"], dtype=float),
            coefficients=np.asarray(arrays["coefficients"], dtype=float),
            diagnostics=diagnostics,
            ridge_penalty=float(np.asarray(arrays["ridge_penalty"], dtype=float)),
        )
        return cls(
            model=model,
            reference_concentration=float(
                np.asarray(arrays["reference_concentration"], dtype=float)
            ),
        )

    @classmethod
    def fit_ridge(
        cls,
        controls: npt.ArrayLike,
        influent: npt.ArrayLike,
        overflow_tss: npt.ArrayLike,
        *,
        ridge_penalty: float,
        reference_concentration: float = 1.0,
        variance_relative_tolerance: float = 1e-12,
    ) -> "LogOverflowTSSClosure":
        target = _finite_array(overflow_tss, name="overflow_tss")
        if target.ndim != 1 or target.size < 2 or np.any(target <= 0.0):
            raise SurrogateValidationError(
                "overflow_tss must be a positive vector with at least two rows"
            )
        reference = float(reference_concentration)
        if not np.isfinite(reference) or reference <= 0.0:
            raise SurrogateValidationError(
                "reference_concentration must be finite and strictly positive"
            )
        log_target = np.log(target / reference)[:, None]
        model = QuadraticSurrogate.fit_ridge(
            controls,
            influent,
            log_target,
            ridge_penalty=ridge_penalty,
            variance_relative_tolerance=variance_relative_tolerance,
        )
        return cls(model=model, reference_concentration=reference)

    def predict_log(
        self, controls: npt.ArrayLike, influent: npt.ArrayLike
    ) -> FloatArray:
        prediction = np.asarray(
            self.model.predict(controls, influent), dtype=np.float64
        )
        if prediction.ndim == 1:
            if prediction.shape != (1,):
                raise SurrogateValidationError(
                    "scalar log-overflow prediction has invalid shape"
                )
            return np.asarray(float(prediction[0]))
        if prediction.ndim != 2 or prediction.shape[1] != 1:
            raise SurrogateValidationError(
                "batch log-overflow prediction has invalid shape"
            )
        return prediction[:, 0]

    def predict(self, controls: npt.ArrayLike, influent: npt.ArrayLike) -> FloatArray:
        log_prediction = self.predict_log(controls, influent)
        with np.errstate(over="raise", invalid="raise"):
            try:
                prediction = self.reference_concentration * np.exp(log_prediction)
            except FloatingPointError as exc:
                raise SurrogateValidationError(
                    "log-overflow back-transformation produced a non-finite value"
                ) from exc
        if not np.all(np.isfinite(prediction)) or np.any(prediction <= 0.0):
            raise SurrogateValidationError(
                "log-overflow prediction must be finite and strictly positive"
            )
        return np.asarray(prediction, dtype=np.float64)
