from typing import Callable, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from cod_analytics.classes import TransformReference


class Homography:
    def __init__(self) -> None:
        """Homography class."""
        self.fitted = False

    @staticmethod
    def fitted_method(func: Callable) -> Callable:
        """Decorator to check if the homography is fitted.

        Args:
            func (Callable): Function to decorate.

        Returns:
            Callable: Decorated function.
        """

        def wrapper(self: "Homography", *args, **kwargs):
            if self.fitted:
                return func(self, *args, **kwargs)
            else:
                raise RuntimeError(
                    "Homography must be fitted before calling this method."
                )

        return wrapper

    def fit(
        self, source: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
    ) -> None:
        """Fits the homography to the source and target points.

        Args:
            source (npt.NDArray[np.float64]): Source points.
            target (npt.NDArray[np.float64]): Target points.
        """
        self.source = source
        self.target = target

        target_vector = target.reshape((-1, 1))

        source_vectors: list[list[float]] = []
        for source_point in source:
            source_vectors += [[*source_point, 1.0, 0.0, 0.0, 0.0]]
            source_vectors += [[0.0, 0.0, 0.0, *source_point, 1.0]]

        source_matrix = np.array(source_vectors)

        tform_solution, _, _, _ = np.linalg.lstsq(
            source_matrix, target_vector, rcond=None
        )
        last_row = [0, 0, 1]
        self.matrix = np.vstack([tform_solution.reshape(2, -1), last_row])
        self.fitted = True

    @fitted_method
    def transform(
        self, points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Transforms the points using the fitted homography.

        Args:
            points (npt.NDArray[np.float64]): Points to transform.

        Returns:
            npt.NDArray[np.float64]: Transformed points.
        """
        ones_vector = np.ones((points.shape[0], 1))
        points = np.hstack([points, ones_vector])
        initial_solution = self.matrix @ points.T
        return np.delete(initial_solution, 2, axis=0).T

    @fitted_method
    def transform_dataframe(
        self,
        df: pd.DataFrame,
        columns: list[str] | list[list[str]],
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Transforms the dataframe using the fitted homography.

        Given a dataframe and a list of columns, transforms the columns using
        the fitted homography. If given a list of lists, each sublist is
        treated as a separate set of columns to transform.

        Args:
            df (pd.DataFrame): Dataframe to transform.
            columns (list[str] | list[list[str]]): Columns to transform. If
                columns is a list of lists, each list is a set of columns to
                transform together.
            inplace (bool): Whether to transform the dataframe in
                place. Much faster when True. Defaults to False.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        if not inplace:
            df = df.copy()

        if isinstance(columns[0], str):
            columns_fixed = cast(list[list[str]], [columns])
        else:
            columns_fixed = cast(list[list[str]], columns)

        for column_set in columns_fixed:
            points = df.loc[:, column_set].values
            transformed_points = self.transform(points)
            df.loc[:, column_set] = transformed_points
        return df

    @staticmethod
    def rotation_matrix(
        angle: float | None, rad: bool = True
    ) -> npt.NDArray[np.float64]:
        """Returns the rotation matrix for the given angle.

        Args:
            angle (float): Angle to rotate by.
            rad (bool): Whether the angle is in radians. Defaults to
                True.

        Returns:
            npt.NDArray[np.float64]: Rotation matrix.
        """
        if angle is None:
            return np.eye(2)

        if not rad:
            angle = (angle % 360) * np.pi / 180
        else:
            angle = angle % (2 * np.pi)
        return np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )

    @staticmethod
    def from_transform_reference(
        source: TransformReference, target: TransformReference
    ) -> "Homography":
        """Creates a homography from two transform references. 

        Given two TransformReference objects, this method will separate the
        bounds and create points for the corners of the bounds. It will then
        rotate the points by the angle of the transform reference and use those
        points to create and fit a homography.

        Args:
            source (TransformReference): Source transform reference.
            target (TransformReference): Target transform reference.

        Returns:
            Homography: Homography object, already fitted.
        """

        def get_points(reference: TransformReference) -> np.ndarray:
            points = np.array(
                [
                    [reference["map_left"], reference["map_top"]],
                    [reference["map_right"], reference["map_top"]],
                    [reference["map_right"], reference["map_bottom"]],
                    [reference["map_left"], reference["map_bottom"]],
                ]
            )
            center = (
                (reference["map_left"] + reference["map_right"]) / 2,
                (reference["map_top"] + reference["map_bottom"]) / 2,
            )
            rotation = Homography.rotation_matrix(
                reference["map_rotation"], rad=False
            )
            points = (points - center) @ rotation
            return points + center

        source_points = get_points(source)
        target_points = get_points(target)

        homography = Homography()
        homography.fit(source_points, target_points)
        return homography
