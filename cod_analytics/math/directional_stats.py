from typing import Callable

import numba as nb
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera as pa
import pandera.typing as pat
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic_2d

from cod_analytics.math.compiled_directional_functions import (
    angular_mean_cartesian,
    angular_mean_cartesian_x,
    angular_mean_cartesian_y,
    angular_mean_var,
    cartesian_to_polar,
    polar_to_cartesian,
    project_to_unit_circle,
)
from cod_analytics.math.diff_geo import wedge_many


@nb.njit(nb.float64(nb.float64[:]))
def directional_mean(angles: npt.NDArray[np.float64]) -> float:
    """Computes the mean of a set of angles.

    Args:
        angles (npt.NDArray[np.float64]): Array of angles.

    Returns:
        float: The mean of the angles.
    """
    xs = np.cos(angles)
    ys = np.sin(angles)
    x, y = np.mean(xs), np.mean(ys)
    return np.arctan2(y, x)


class InputSchema(pa.SchemaModel):
    ax: pa.typing.Series[float] = pa.Field(nullable=False)
    ay: pa.typing.Series[float] = pa.Field(nullable=False)
    vx: pa.typing.Series[float] = pa.Field(nullable=False)
    vy: pa.typing.Series[float] = pa.Field(nullable=False)


class DirectionalStats:
    @pa.check_types
    def __init__(self, data: pat.DataFrame[InputSchema]) -> None:
        """Computes directional statistics for a set of data.

        Args:
            data (pat.DataFrame[InputSchema]): Dataframe containing the data.
                DataFrame must have the following columns:
                    ax: x component of the attacker
                    ay: y component of the attacker
                    vx: x component of the victim
                    vy: y component of the victim
        """
        self.attacker_xy: list[str] = ["ax", "ay"]
        self.victim_xy: list[str] = ["vx", "vy"]
        self.data: pd.DataFrame = data.copy()
        self.data["delta_x"] = self.data["ax"] - self.data["vx"]
        self.data["delta_y"] = self.data["ay"] - self.data["vy"]
        self.data["angle"] = np.arctan2(  # type: ignore
            -self.data["delta_y"], -self.data["delta_x"]
        )
        self.data["norm_x"] = np.cos(self.data["angle"])
        self.data["norm_y"] = np.sin(self.data["angle"])

    def generate_vector_spaces(
        self, *args, **kwargs
    ) -> "VectorSpaceResults":
        """Generates attacker and victim vector spaces for the given map. Passes
        any additional arguments via args and kwargs to SciPy's
        binned_statistic_2d.

        Returns:
            VectorSpaceResults: Results of the vector space generation.
        """
        data = self.data.copy()
        x_max = data[["ax", "vx"]].max().max()
        x_min = data[["ax", "vx"]].min().min()
        y_max = data[["ay", "vy"]].max().max()
        y_min = data[["ay", "vy"]].min().min()

        def binned_stat_part(statistic: Callable, av: str):
            if av == "a":
                val = 1
            elif av == "v":
                val = -1
            else:
                raise ValueError(f"Invalid av: {av}")
            return binned_statistic_2d(
                data[av + "y"],
                data[av + "x"],
                data["angle"] * val,
                *args,
                statistic=statistic,
                range=[[x_min, x_max], [y_min, y_max]],
                **kwargs,
            )

        try:
            a_vals, x_edges, y_edges, _ = binned_stat_part(
                angular_mean_cartesian, "a"
            )
            v_vals, _, _, _ = binned_stat_part(angular_mean_cartesian, "v")
        except TypeError:  # TODO: Remove this once SciPy 1.10.0 is released
            a_vals_x, x_edges, y_edges, _ = binned_stat_part(
                angular_mean_cartesian_x, "a"
            )
            ret = binned_stat_part(angular_mean_cartesian_x, "a")
            a_vals_x = ret.statistic
            x_edges, y_edges = ret.x_edge, ret.y_edge
            a_vals_y = binned_stat_part(angular_mean_cartesian_y, "a").statistic
            a_vals = a_vals_x + 1j * a_vals_y

            v_vals_x = binned_stat_part(angular_mean_cartesian_x, "v").statistic
            v_vals_y = binned_stat_part(angular_mean_cartesian_y, "v").statistic
            v_vals = v_vals_x + 1j * v_vals_y

        return VectorSpaceResults(a_vals, v_vals, x_edges, y_edges)

    @staticmethod
    def generate_bivector_field(
        a_vals: npt.NDArray[np.complex128],
        v_vals: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.complex128]:
        """Generates a bivector field from the given attacker and victim
        vector spaces.


        Args:
            a_vals (npt.NDArray[np.complex128]): Mean and variance of attacker
            v_vals (npt.NDArray[np.complex128]): Mean and variance of victim

        Returns:
            npt.NDArray[np.complex128]: Bivector field.
        """
        a_x = a_vals.real
        a_y = a_vals.imag
        v_x = v_vals.real
        v_y = v_vals.imag

        a_vectors = np.column_stack([a_x, a_y])
        v_vectors = np.column_stack([v_x, v_y])
        dot = np.einsum("ij, ij->i", a_vectors, v_vectors)
        wedge = wedge_many(a_vectors, v_vectors)
        return dot + 1j * wedge


class VectorSpaceResults:
    def __init__(
        self,
        a_vals: npt.NDArray[np.complex128],
        v_vals: npt.NDArray[np.complex128],
        x_edges: npt.NDArray[np.float64],
        y_edges: npt.NDArray[np.float64],
    ) -> None:
        self.a_vals = a_vals
        self.v_vals = v_vals
        self.x_edges = x_edges
        self.y_edges = y_edges

        self.a_x = a_vals.real
        self.a_y = a_vals.imag
        self.v_x = v_vals.real
        self.v_y = v_vals.imag
        self.bin_centers_x = (x_edges[1:] + x_edges[:-1]) / 2
        self.bin_centers_y = (y_edges[1:] + y_edges[:-1]) / 2
        self.bin_size_x = x_edges[1] - x_edges[0]
        self.bin_size_y = y_edges[1] - y_edges[0]

    def generate_bivector_field(self) -> npt.NDArray[np.complex128]:
        a_vectors = np.column_stack([self.a_x, self.a_y])
        v_vectors = np.column_stack([self.v_x, self.v_y])
        dot = np.einsum("ij, ij->i", a_vectors, v_vectors)
        wedge = wedge_many(a_vectors, v_vectors)
        return dot + 1j * wedge

    def plot_vector_field(self, ax: plt.Axes, av: str, **kwargs) -> None:
        if av == "a":
            x = self.a_x
            y = self.a_y
        elif av == "v":
            x = self.v_x
            y = self.v_y
        else:
            raise ValueError(f"Invalid av: {av}")
        ax.quiver(self.bin_centers_x, self.bin_centers_y, x, y, **kwargs)

    def plot_bivector_field(self, ax: plt.Axes, **kwargs) -> None:
        bivector_field = self.generate_bivector_field()
        ax.quiver(
            self.bin_centers_x,
            self.bin_centers_y,
            bivector_field.real,
            bivector_field.imag,
            **kwargs,
        )
