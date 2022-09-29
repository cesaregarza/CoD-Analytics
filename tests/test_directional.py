import numpy as np
import pytest

from cod_analytics.math.compiled_directional_functions import angular_mean_var


@pytest.mark.parametrize(
    "angles, expected_mean, expected_var",
    [
        (
            np.array([0.0, np.pi / 4, 7 / 4 * np.pi]),
            0.0,
            (1.0 + np.sqrt(2)) / 3,
        ),
        (np.array([0.0, np.pi / 2, 3 / 2 * np.pi]), 0.0, (1 / 3)),
        (
            np.array([0.0, np.pi / 4, np.pi / 2]),
            np.pi / 4,
            (1.0 + np.sqrt(2)) / 3,
        ),
        (
            np.array([0.0, 0.0, 0.0, np.pi / 2]),
            np.arctan2(1, 3),
            np.sqrt(5 / 8),
        ),
        (
            np.array([0.0, 0.0, 0.0, 0.0]),
            0,
            1.0,
        ),
    ],
    ids=[
        "0-45-315",
        "0-90-270",
        "0-45-90",
        "0-0-0-90",
        "0-0-0-0",
    ],
)
def test_angular_mean_var(
    angles: np.ndarray, expected_mean: float, expected_var: float
):
    res = angular_mean_var(angles)
    assert np.isclose(res.real, expected_mean)
    assert np.isclose(res.imag, expected_var)
