"""Independent differences check the native ten-layer expression derivatives."""

import unittest

import casadi as ca
import numpy as np

from surrogate_optimization.optimization.mechanistic import (
    MechanisticCase,
    build_mechanistic_nlp,
)
from surrogate_optimization.plant.definitions import NOMINAL_INFLUENT
from tests.support.mechanistic_fixtures import mechanistic_assets


class MechanisticDerivativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = build_mechanistic_nlp(
            mechanistic_assets(10),
            epsilon=1e-6,
            receiver_half_width=10.0,
            compile_solver=False,
            name="independent_derivatives",
        )
        cls.parameters = MechanisticCase(NOMINAL_INFLUENT).parameter_vector()
        cls.point = np.r_[np.full(7, 0.5), np.zeros(110), 1.0]
        cls.direction = np.random.default_rng(628).normal(size=cls.point.size)
        cls.direction /= np.linalg.norm(cls.direction)

    def test_constraint_jacobians_match_two_scale_directional_differences(self):
        problem = self.problem
        for value, jacobian in (
            (problem.equality_function, problem.equality_jacobian_function),
            (problem.inequality_function, problem.inequality_jacobian_function),
        ):
            analytical = (
                np.asarray(jacobian(self.point, self.parameters)) @ self.direction
            )
            for step in (1e-6, 5e-7):
                positive = np.asarray(
                    value(self.point + step * self.direction, self.parameters)
                ).ravel()
                negative = np.asarray(
                    value(self.point - step * self.direction, self.parameters)
                ).ravel()
                numerical = (positive - negative) / (2 * step)
                self.assertLessEqual(
                    float(
                        np.max(
                            np.abs(analytical - numerical) / (1 + np.abs(analytical))
                        )
                    ),
                    1e-5,
                )

    def test_objective_hessian_matches_independent_gradient_differences(self):
        symbol = ca.MX.sym("point", self.point.size)
        gradient = self.problem.gradient_function(symbol, self.parameters)
        hessian = ca.Function(
            "independent_hessian", [symbol], [ca.jacobian(gradient, symbol)]
        )
        analytical = np.asarray(hessian(self.point)) @ self.direction
        for step in (1e-6, 5e-7):
            positive = np.asarray(
                self.problem.gradient_function(
                    self.point + step * self.direction, self.parameters
                )
            ).ravel()
            negative = np.asarray(
                self.problem.gradient_function(
                    self.point - step * self.direction, self.parameters
                )
            ).ravel()
            numerical = (positive - negative) / (2 * step)
            self.assertLessEqual(
                float(
                    np.max(np.abs(analytical - numerical) / (1 + np.abs(analytical)))
                ),
                1e-5,
            )
