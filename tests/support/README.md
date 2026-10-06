# Test fixtures

`profiles.py` defines bounded fixture profiles and the 80/20 integration profile. They are internal to tests and are rejected by the production validator. Generic numerical fixtures may use fewer layers to test dimension-parametric equations; the integration profile preserves all ten production layers.

`mechanistic_fixtures.py` supplies deterministic ten-layer expression fixtures for independent Jacobian and Hessian checks.
