import numpy as np
import pytest

from cod_analytics.math.diff_geo import wedge, wedge_many


def float_array(input_list: list) -> np.ndarray:
    return np.array(input_list, dtype=np.float64)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (
            float_array([1, 1]),
            float_array([2, -5]),
            -7,
        ),
        (
            float_array([2, -5]),
            float_array([1, 1]),
            7,
        ),
        (
            float_array([2, 5]),
            float_array([1, 1]),
            -3,
        ),
        (
            float_array([1, 1]),
            float_array([1, 1]),
            0,
        ),
    ],
)
def test_wedge(a, b, expected):
    assert wedge(a, b) == expected


def test_wedge_many():
    a = float_array([[1, 1], [2, 5], [1, 1]])
    b = float_array([[2, -5], [1, 1], [1, 1]])
    expected = float_array([-7, -3, 0])
    assert np.all(wedge_many(a, b) == expected)
