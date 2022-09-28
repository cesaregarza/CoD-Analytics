from enum import Enum, auto
from functools import cached_property
from typing import TypeVar
from warnings import warn

import numba as nb
import numpy as np
import numpy.typing as npt
import pandas as pd

from cod_analytics.math.compiled_directional_functions import (
    cartesian_mean,
    cartesian_to_polar,
    cartesian_variance,
    distance_between_angles,
    polar_to_cartesian,
    project_to_unit_circle,
)

NPNUM = TypeVar("NPNUM", np.int_, np.float_)
N = TypeVar("N", int, float)


class DirectionalFormat(Enum):
    """Enum for the different directional formats."""

    CARTESIAN = auto()
    POLAR = auto()


class DirectionalStatistics:
    def __init__(
        self,
        data: npt.NDArray[NPNUM],
        format: DirectionalFormat | str = DirectionalFormat.CARTESIAN,
        center_point: tuple[N, N] = (0, 0),
        base: int | float = (2 * np.pi),
        *args,
        validate: bool = True,
    ) -> None:
        """Class that handles directional statistics for a given array of data.

        Args:
            data (npt.NDArray[N]): Array-like object containing data.
            format (DirectionalFormat | str, optional): Format of the data.
                Accepted values are DirectionalFormat.CARTESIAN,
                DirectionalFormat.POLAR, 'cartesian', 'polar'. Defaults to
                DirectionalFormat.CARTESIAN.
            center_point (npt.NDArray[N] | tuple[N, N], optional): Center point
                of the data. Defaults to (0, 0). Must be a Sequence of length 2.
            base (N, optional): base of the data. Defaults to 2 * np.pi.
            *args: Additional arguments to pass to the data constructor.
                Currently only used to enforce keyword-only arguments.
            validate (bool): Whether to validate the data.
                Keyword-only argument. Not recommended to set to False. Defaults
                to True.

        Raises:
            NotImplementedError: If data with more than 2 dimensions is
                provided.
            ValueError: If the data is 1-dimensional.
        """
        if validate:
            data_array = DirectionalStatistics.validate_data(data)
        else:
            data_array = np.array(data, dtype=np.float64)

        if data_array.shape[1] > 2:
            raise NotImplementedError(
                "Data with more than 2 dimensions is not supported"
            )
        elif data_array.shape[1] == 1:
            raise ValueError("Data must have at least 2 dimensions")

        self.center_point = np.array(center_point, dtype=np.float64)
        self.raw_data = data_array
        self.format = DirectionalStatistics.validate_format(format)
        self.base = base
        self.base_to_radians = self.base / (2 * np.pi)
        self.normalize_format()

    @cached_property
    def x(self) -> npt.NDArray[np.float64]:
        """Array of x-coorinates of the data

        Returns:
            npt.NDArray[np.float64]: Array of x-coordinates.
        """
        return self.cartesian_data[:, 0]

    @cached_property
    def y(self) -> npt.NDArray[np.float64]:
        """Array of y-coorinates of the data

        Returns:
            npt.NDArray[np.float64]: Array of y-coordinates.
        """
        return self.cartesian_data[:, 1]

    @cached_property
    def r(self) -> npt.NDArray[np.float64]:
        """Array of radial distances of the data

        Returns:
            npt.NDArray[np.float64]: Array of radial distances.
        """
        return self.polar_data[:, 0]

    @cached_property
    def theta(self) -> npt.NDArray[np.float64]:
        """Array of angular components of the data in the directional base e.g.
        360 degrees.

        Returns:
            npt.NDArray[np.float64]: Array of angular components.
        """
        return self.polar_data[:, 1] * self.base_to_radians

    @cached_property
    def radians(self) -> npt.NDArray[np.float64]:
        """Array of angular components of the data in radians

        Returns:
            npt.NDArray[np.float64]: Array of angular components.
        """
        return self.polar_data[:, 1]

    @cached_property
    def mean_cartesian(self) -> tuple[float, float]:
        """Circular mean of the data.

        Returns:
            tuple[float, float]: Mean of the data.
        """
        _mean = cartesian_mean(self.cartesian_data)
        return _mean[0], _mean[1]

    @cached_property
    def mean_theta(self) -> float:
        """Mean of the angular components of the data.

        Returns:
            float: Mean of the angular components of the data.
        """
        x, y = self.mean_cartesian
        return np.arctan2(y, x) * self.base_to_radians

    @cached_property
    def mean_radians(self) -> float:
        """Mean of the angular components of the data in radians.

        Returns:
            float: Mean of the angular components of the data in radians.
        """
        return self.mean_theta / self.base_to_radians

    @cached_property
    def var(self) -> float:
        """Variance of the data.

        Returns:
            float: Variance of the data.
        """
        return cartesian_variance(self.cartesian_data)

    @staticmethod
    def validate_data(
        data: npt.NDArray[NPNUM],
    ) -> npt.NDArray[np.float64]:
        """Validates the data and converts it to a numpy array of floats.

        Args:
            data (npt.NDArray[N]): Array-like object containing data.

        Raises:
            ValueError: If the data is ragged.
            ValueError: If data is non-numeric.
            ValueError: If data contains NaNs.

        Returns:
            npt.NDArray[np.float_]: Numpy array of floats.
        """
        # If list, convert to numpy array
        if isinstance(data, list):
            if not all(len(x) == len(data[0]) for x in data):
                raise ValueError("Ragged arrays are not supported")

        # Convert to numpy array or change dtype to float64
        array = np.array(data, dtype=np.float64)

        # Validate elements are numeric
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("Data must be numeric")

        # Raise warning if duplicate values are detected
        if len(array) != len(np.unique(array)):
            warn(
                "Duplicate values detected in data. This may cause unexpected"
                " results."
            )

        if np.isnan(array).any():
            raise ValueError("Data cannot contain NaN values")

        return array

    @staticmethod
    def validate_format(
        format: DirectionalFormat | str,
    ) -> DirectionalFormat:
        """Validates the format of the data.

        Args:
            format (DirectionalFormat | str): Format of the data.

        Raises:
            ValueError: If the format is not valid.

        Returns:
            DirectionalFormat: Validated format.
        """
        if isinstance(format, str):
            format = format.lower()

        if format in ("cartesian", DirectionalFormat.CARTESIAN):
            return DirectionalFormat.CARTESIAN
        elif format in ("polar", DirectionalFormat.POLAR):
            return DirectionalFormat.POLAR
        else:
            raise ValueError(
                "Invalid format. Accepted values are 'cartesian', 'polar', "
                "DirectionalFormat.CARTESIAN, DirectionalFormat.POLAR"
            )

    def normalize_format(self) -> None:
        """Precomputes the data of the opposite format to the provided one."""
        if self.format == DirectionalFormat.CARTESIAN:
            self.cartesian_data = self.raw_data - self.center_point
            r = np.sqrt(self.x**2 + self.y**2)
            theta = np.arctan2(self.y, self.x)
            self.polar_data = np.array([r, theta]).T
        else:
            self.polar_data = self.raw_data
            x = self.r * np.cos(self.radians)
            y = self.r * np.sin(self.radians)
            self.cartesian_data = np.array([x, y]).T + self.center_point

    def change_base(self, base: float) -> None:
        """Changes the base of the angular components of the data.

        Args:
            base (float): New base of the angular components.
        """
        self.base = base
        self.base_to_radians = base / (2 * np.pi)

    @staticmethod
    def convert(
        data: npt.NDArray[np.float64],
        input_format: DirectionalFormat | str,
        output_format: DirectionalFormat | str,
        base: float | int = 2 * np.pi,
    ) -> npt.NDArray[np.float64]:
        """Converts data from one format to another.

        Args:
            data (npt.NDArray[np.float64]): Array of data.
            input_format (DirectionalFormat | str): Format of the input data.
            output_format (DirectionalFormat | str): Format of the output data.
            base (float | int, optional): Base of the angular components.
                Defaults to 2 * np.pi.

        Raises:
            ValueError: If an invalid format is provided.

        Returns:
            npt.NDArray[np.float64]: Array of data in the output format.
        """
        in_format = DirectionalStatistics.validate_format(input_format)
        out_format = DirectionalStatistics.validate_format(output_format)

        if in_format == out_format:
            return data

        if out_format == DirectionalFormat.CARTESIAN:
            return polar_to_cartesian(data, base=base)
        elif out_format == DirectionalFormat.POLAR:
            return cartesian_to_polar(data, base=base)
        else:
            raise ValueError("Invalid output format")

    @staticmethod
    def from_dataframe(
        dataframe: pd.DataFrame,
        columns: list[str] | None = None,
        format: DirectionalFormat | str | None = None,
        x: str | None = None,
        y: str | None = None,
        r: str | None = None,
        theta: str | None = None,
        center_point: tuple[float, float] | None = None,
        base: float | int = 2 * np.pi,
    ) -> "DirectionalStatistics":
        """Creates a DirectionalStatistics object from a pandas DataFrame.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the data.
            columns (list[str] | None, optional): List of columns to use.
                Defaults to None.
            format (DirectionalFormat | str | None, optional): Format of the
                data. Defaults to None.
            x (str | None, optional): Name of the column containing the x
                values. Defaults to None.
            y (str | None, optional): Name of the column containing the y
                values. Defaults to None.
            r (str | None, optional): Name of the column containing the r
                values. Defaults to None.
            theta (str | None, optional): Name of the column containing the
                theta values. Defaults to None.
            center_point (tuple[float, float] | None, optional): Center point
                of the data. Defaults to None.
            base (float | int, optional): Base of the angular components.
                Defaults to 2 * np.pi.

        Raises:
            ValueError: If the columns are provided but the format is not.
            NotImplementedError: If data of dimensions greater than 2 is
                provided.
            ValueError: If there is an invalid combination of arguments.

        Returns:
            DirectionalStatistics: DirectionalStatistics object.
        """
        if columns is not None:
            if format is None:
                raise ValueError(
                    "If columns are provided, format must also be provided"
                )
            if len(columns) != 2:
                raise NotImplementedError("Only 2D data is currently supported")
            col_1, col_2 = columns
            format_ = DirectionalStatistics.validate_format(format)
        else:
            xy = (x is not None) and (y is not None)
            rt = (r is not None) and (theta is not None)
            if not (xy or rt):
                if len(dataframe.columns) == 2:
                    col_1, col_2 = dataframe.columns
                    format_ = DirectionalFormat.CARTESIAN
                    warn("No format was provided. Assuming cartesian format")
                else:
                    raise NotImplementedError(
                        "Only 2D data is currently supported, please provide "
                        "the names of the columns containing the x and y "
                        "or r and theta values explicitly"
                    )
            elif (x is not None) and (y is not None):
                col_1, col_2 = x, y
                format_ = DirectionalFormat.CARTESIAN
            elif (r is not None) and (theta is not None):
                col_1, col_2 = r, theta
                format_ = DirectionalFormat.POLAR
            else:
                raise ValueError(
                    "Invalid combination of arguments. Please provide either "
                    "the names of the columns containing the x and y or r and "
                    "theta values explicitly, a list of columns to use, or "
                    "pass a DataFrame with only 2 columns"
                )

        data = dataframe.loc[:, [col_1, col_2]].values
        if center_point is None:
            center_point = (0, 0)

        return DirectionalStatistics(
            data,
            format=format_,
            center_point=center_point,
            base=base,
        )
