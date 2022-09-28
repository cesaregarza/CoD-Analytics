import numpy as np
import pandas as pd
import pandas.testing as pdt
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

@pytest.mark.parametrize(
    "source, target, df, labels, expected_df",
    [
        ( # No transform
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            pd.DataFrame({
                "x": [0.5],
                "y": [0.5],
            }),
            ["x", "y"],
            pd.DataFrame({
                "x": [0.5],
                "y": [0.5],
            }),
        ),
        ( # Transform, Scale, Rotate
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array(
                [
                    [-np.sqrt(2) + 1.5, 1.5],
                    [1.5, -np.sqrt(2) + 1.5],
                    [np.sqrt(2) + 1.5, 1.5],
                    [1.5, np.sqrt(2) + 1.5],
                ]
            ),
            pd.DataFrame({
                "x": [0.5, 0.75],
                "y": [0.5, 0.75],
            }),
            ["x", "y"],
            pd.DataFrame({
                "x": [1.5, 1.5 + np.sqrt(2) / 2],
                "y": [1.5, 1.5],
            }),
        ),
        ( # Transform, Scale, Rotate, Multiple Column Sets
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            np.array(
                [
                    [-np.sqrt(2) + 1.5, 1.5],
                    [1.5, -np.sqrt(2) + 1.5],
                    [np.sqrt(2) + 1.5, 1.5],
                    [1.5, np.sqrt(2) + 1.5],
                ]
            ),
            pd.DataFrame({
                "x1": [0.5, 0.75],
                "y1": [0.5, 0.75],
                "x2": [0.5, 0.75],
                "y2": [0.5, 0.75],
            }),
            [["x1", "y1"], ["x2", "y2"]],
            pd.DataFrame({
                "x1": [1.5, 1.5 + np.sqrt(2) / 2],
                "y1": [1.5, 1.5],
                "x2": [1.5, 1.5 + np.sqrt(2) / 2],
                "y2": [1.5, 1.5],
            }),
        ),
    ],
    ids=[
        "No transform",
        "Translation, Scale, and Rotation",
        "Translation, Scale, and Rotation, Multiple Column Sets",
    ],
)
def test_homography_transform_df(
    source: np.ndarray,
    target: np.ndarray,
    df: pd.DataFrame,
    labels: list[str],
    expected_df: pd.DataFrame,
) -> None:
    """Test the transform method."""
    homography = Homography()

    homography.fit(source, target)
    out_df = homography.transform_dataframe(df, labels)

    pdt.assert_frame_equal(
        out_df, expected_df
    )