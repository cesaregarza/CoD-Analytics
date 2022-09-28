import numpy as np
import numpy.typing as npt
import pytest

from cod_analytics.classes import TransformReference
from cod_analytics.math.homography import Homography


def test_homography_fitted_method() -> None:
    """Test the fitted_method decorator."""
    homography = Homography()

    @homography.fitted_method
    def test_method(self: Homography) -> None:
        pass

    with pytest.raises(RuntimeError):
        test_method(homography)

    homography.fitted = True
    test_method(homography)


def test_homography_fit() -> None:
    """Test the fit method."""
    homography = Homography()
    source = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    target = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    homography.fit(source, target)

    assert homography.fitted
    assert np.allclose(homography.matrix, np.eye(3))


@pytest.mark.parametrize(
    "source, target, coordinates, expected",
    [
        (  # No transform
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array([[0.5, 0.5]]),
            np.array([[0.5, 0.5]]),
        ),
        (  # Scale
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]),
            np.array([[0.5, 0.5]]),
            np.array([[1.0, 1.0]]),
        ),
        (  # Rotation
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array(
                [
                    [-np.sqrt(2) / 2 + 0.5, 0.5],
                    [0.5, -np.sqrt(2) / 2 + 0.5],
                    [np.sqrt(2) / 2 + 0.5, 0.5],
                    [0.5, np.sqrt(2) / 2 + 0.5],
                ]
            ),
            np.array([[0.5, 0.5], [0.75, 0.75]]),
            np.array([[0.5, 0.5], [0.5 + np.sqrt(2) / 4, 0.5]]),
        ),
        (  # Translation
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]),
            np.array([[0.5, 0.5]]),
            np.array([[1.5, 1.5]]),
        ),
        (  # Translation and Scale
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]),
            np.array([[0.5, 0.5]]),
            np.array([[2.0, 2.0]]),
        ),
        (  # Translation and Rotation
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array(
                [
                    [-np.sqrt(2) / 2 + 1.5, 1.5],
                    [1.5, -np.sqrt(2) / 2 + 1.5],
                    [np.sqrt(2) / 2 + 1.5, 1.5],
                    [1.5, np.sqrt(2) / 2 + 1.5],
                ]
            ),
            np.array([[0.5, 0.5], [0.75, 0.75]]),
            np.array([[1.5, 1.5], [1.5 + np.sqrt(2) / 4, 1.5]]),
        ),
        (  # Scale and Rotation
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array(
                [
                    [-np.sqrt(2) + 0.5, 0.5],
                    [0.5, -np.sqrt(2) + 0.5],
                    [np.sqrt(2) + 0.5, 0.5],
                    [0.5, np.sqrt(2) + 0.5],
                ]
            ),
            np.array([[0.5, 0.5], [0.75, 0.75]]),
            np.array([[0.5, 0.5], [0.5 + np.sqrt(2) / 2, 0.5]]),
        ),
        (  # Translation, Scale, and Rotation
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array(
                [
                    [-np.sqrt(2) + 1.5, 1.5],
                    [1.5, -np.sqrt(2) + 1.5],
                    [np.sqrt(2) + 1.5, 1.5],
                    [1.5, np.sqrt(2) + 1.5],
                ]
            ),
            np.array([[0.5, 0.5], [0.75, 0.75]]),
            np.array([[1.5, 1.5], [1.5 + np.sqrt(2) / 2, 1.5]]),
        ),
    ],
    ids=[
        "No transform",
        "Scale",
        "Rotation",
        "Translation",
        "Translation and Scale",
        "Translation and Rotation",
        "Scale and Rotation",
        "Translation, Scale, and Rotation",
    ],
)
def test_homography_transform(source, target, coordinates, expected) -> None:
    """Test the transform method."""
    homography = Homography()

    homography.fit(source, target)

    assert np.allclose(homography.transform(coordinates), expected)


@pytest.mark.parametrize(
    "source, target, coordinates, expected",
    [
        (
            TransformReference(
                map_left=0.0,
                map_right=1.0,
                map_top=1.0,
                map_bottom=0.0,
                map_rotation=0.0,
            ),
            TransformReference(
                map_left=0.0,
                map_right=1.0,
                map_top=1.0,
                map_bottom=0.0,
                map_rotation=0.0,
            ),
            np.array([[0.5, 0.5]]),
            np.array([[0.5, 0.5]]),
        ),
        (
            TransformReference(
                map_left=0.0,
                map_right=1.0,
                map_top=1.0,
                map_bottom=0.0,
                map_rotation=0.0,
            ),
            TransformReference(
                map_left=1.0,
                map_right=3.0,
                map_top=3.0,
                map_bottom=1.0,
                map_rotation=45.0,
            ),
            np.array([[0.5, 0.5], [0.75, 0.75]]),
            np.array([[2, 2], [2 + np.sqrt(2) / 2, 2]]),
        ),
    ],
    ids=[
        "No transform",
        "Translation, Scale, and Rotation",
    ],
)
def test_homography_transform_reference(
    source: TransformReference,
    target: TransformReference,
    coordinates: np.ndarray,
    expected: np.ndarray,
) -> None:
    """Test the transform method."""
    homography = Homography.from_transform_reference(source, target)

    assert np.allclose(homography.transform(coordinates), expected)
