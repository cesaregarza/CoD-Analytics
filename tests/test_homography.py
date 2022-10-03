import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from cod_analytics.classes import TransformReference
from cod_analytics.math.homography import Homography


class TestHomography:
    xy_bounds = (-1, 1)
    xy_corners = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
    ]
    x_bounds = (-1, 1)
    y_bounds = (-2, 2)
    xy_rect_corners = [
        [-1, -2],
        [1, -2],
        [1, 2],
        [-1, 2],
    ]
    translate_bounds = (-5, 5)
    rotate_bounds = (-np.pi, np.pi)
    scale_bounds = (-1, 1)
    center_bounds = (-1, 1)
    size = (100, 2)

    rng = np.random.RandomState(956)
    xy = rng.uniform(*xy_bounds, size=size)
    x = rng.uniform(*x_bounds, size=size[0])
    y = rng.uniform(*y_bounds, size=size[0])
    xy_rect = np.vstack([x, y]).T
    translate = rng.uniform(*translate_bounds, size=2)
    rotate = rng.uniform(*rotate_bounds)
    scale = 2 ** rng.uniform(*scale_bounds)
    center = rng.uniform(*center_bounds, size=2)

    def transform(
        self,
        points: np.ndarray,
        translate: np.ndarray,
        rotate: float,
        scale: float,
        center: np.ndarray,
    ) -> np.ndarray:
        """Transform points using a translation, rotation, and scale."""
        rotate_array = np.array(
            [
                [np.cos(rotate), -np.sin(rotate)],
                [np.sin(rotate), np.cos(rotate)],
            ]
        )
        new_points = points - center
        new_points = new_points * scale
        new_points = new_points @ rotate_array
        new_points = new_points + center
        new_points = new_points + translate
        return new_points

    def test_homography_fitted_method(self) -> None:
        """Test the fitted_method decorator."""
        homography = Homography()

        @homography.fitted_method
        def test_method(self: Homography) -> None:
            pass

        with pytest.raises(RuntimeError):
            test_method(homography)

        homography.fitted = True
        test_method(homography)

    def test_homography_fit(self) -> None:
        """Test the fit method."""
        homography = Homography()
        source = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        target = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

        homography.fit(source, target)

        assert homography.fitted
        assert np.allclose(homography.matrix, np.eye(3))

    @pytest.mark.parametrize("scale", [True, False], ids=["S", "NS"])
    @pytest.mark.parametrize("rotate", [True, False], ids=["R", "NR"])
    @pytest.mark.parametrize("translate", [True, False], ids=["T", "NT"])
    @pytest.mark.parametrize("center", [True, False], ids=["C", "NC"])
    @pytest.mark.parametrize("square", [True, False], ids=["Sq", "Rect"])
    def test_homography_transform(
        self,
        translate: bool,
        rotate: bool,
        scale: bool,
        center: bool,
        square: bool,
    ) -> None:
        """Test the transform method."""
        translate_val = self.translate if translate else np.zeros(2)
        rotate_val = self.rotate if rotate else 0.0
        scale_val = self.scale if scale else 1.0
        center_val = self.center if center else np.zeros(2)

        # Generate XY data
        XY = (self.xy if square else self.xy_rect) + center_val

        corners = np.array(self.xy_corners if square else self.xy_rect_corners)
        transformed_corners = self.transform(
            corners, translate_val, rotate_val, scale_val, center_val
        )
        expected = self.transform(
            XY, translate_val, rotate_val, scale_val, center_val
        )
        homography = Homography()
        homography.fit(corners, transformed_corners)
        XY_transformed = homography.transform(XY)
        assert np.allclose(XY_transformed, expected)

    @pytest.mark.parametrize("scale", [True, False], ids=["S", "NS"])
    @pytest.mark.parametrize("rotate", [True, False], ids=["R", "NR"])
    @pytest.mark.parametrize("translate", [True, False], ids=["T", "NT"])
    @pytest.mark.parametrize("center", [True, False], ids=["C", "NC"])
    @pytest.mark.parametrize("square", [True, False], ids=["Sq", "Rect"])
    def test_homography_transform_reference(
        self,
        translate: bool,
        rotate: bool,
        scale: bool,
        center: bool,
        square: bool,
    ) -> None:
        translate_val = self.translate if translate else np.zeros(2)
        rotate_val = self.rotate if rotate else 0.0
        scale_val = self.scale if scale else 1.0
        center_val = self.center if center else np.zeros(2)

        # Generate XY data
        XY = (self.xy if square else self.xy_rect) + center_val
        left, right = self.xy_bounds if square else self.x_bounds
        bottom, top = self.xy_bounds if square else self.y_bounds
        source = TransformReference(
            map_left=left + center_val[0],
            map_right=right + center_val[0],
            map_bottom=bottom + center_val[1],
            map_top=top + center_val[1],
            map_rotation=0.0,
        )
        corners = np.array(self.xy_corners if square else self.xy_rect_corners)
        corners = corners + np.array([center_val[0], center_val[1]])
        transformed_corners = self.transform(
            corners,
            translate_val,
            rotate_val,
            scale_val,
            center_val,
        )
        transformed_corners_nr = self.transform(
            corners,
            translate_val,
            0.0,
            scale_val,
            center_val,
        )
        new_left, new_right = (
            transformed_corners_nr[:, 0].min(),
            transformed_corners_nr[:, 0].max(),
        )
        new_bottom, new_top = (
            transformed_corners_nr[:, 1].min(),
            transformed_corners_nr[:, 1].max(),
        )
        target = TransformReference(
            map_left=new_left,
            map_right=new_right,
            map_bottom=new_bottom,
            map_top=new_top,
            map_rotation=rotate_val * 180.0 / np.pi,
        )
        homography_expected = Homography()
        homography_expected.fit(corners, transformed_corners)
        expected = homography_expected.transform(XY)
        homography = Homography.from_transform_reference(source, target)
        XY_transformed = homography.transform(XY)
        assert np.allclose(XY_transformed, expected)

    @pytest.mark.parametrize("scale", [True, False], ids=["S", "NS"])
    @pytest.mark.parametrize("rotate", [True, False], ids=["R", "NR"])
    @pytest.mark.parametrize("translate", [True, False], ids=["T", "NT"])
    @pytest.mark.parametrize("center", [True, False], ids=["C", "NC"])
    @pytest.mark.parametrize("square", [True, False], ids=["Sq", "Rect"])
    @pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
    def test_homography_transform_df(
        self,
        translate: bool,
        rotate: bool,
        scale: bool,
        center: bool,
        square: bool,
        n: int,
    ) -> None:
        """Test the transform_df method."""
        translate_val = self.translate if translate else np.zeros(2)
        rotate_val = self.rotate if rotate else 0.0
        scale_val = self.scale if scale else 1.0
        center_val = self.center if center else np.zeros(2)

        # Generate XY data
        k = n if n > 0 else 1
        if square:
            xy_data = self.rng.uniform(
                *self.xy_bounds, size=(self.size[0], self.size[1] * k)
            ) + np.tile(center_val, k)
        else:
            x_data = self.rng.uniform(*self.x_bounds, size=(self.size[0], k))
            y_data = self.rng.uniform(*self.y_bounds, size=(self.size[0], k))
            # Interleave x and y data
            xy_data = np.empty((self.size[0], self.size[1] * k))
            xy_data[:, ::2] = x_data + center_val[0]
            xy_data[:, 1::2] = y_data + center_val[1]

        corners = np.array(self.xy_corners if square else self.xy_rect_corners)
        transformed_corners = self.transform(
            corners, translate_val, rotate_val, scale_val, center_val
        )

        if n == 0:
            labels: list[str] | list[list[str]] = ["x", "y"]
            flat_labels = ["x", "y"]
        else:
            labels = [[f"x_{i}", f"y_{i}"] for i in range(n)]
            flat_labels = [item for sublist in labels for item in sublist]

        df = pd.DataFrame(xy_data, columns=flat_labels)
        homography = Homography()
        homography.fit(corners, transformed_corners)
        df_transformed = homography.transform_dataframe(df, labels)

        if n == 0:
            expected = self.transform(
                xy_data, translate_val, rotate_val, scale_val, center_val
            )
            assert np.allclose(df_transformed.values, expected)
        else:
            for i in range(n):
                k = i * 2
                expected = self.transform(
                    xy_data[:, k : k + 2],
                    translate_val,
                    rotate_val,
                    scale_val,
                    center_val,
                )
                assert np.allclose(
                    df_transformed.iloc[:, k : k + 2].values, expected
                )
