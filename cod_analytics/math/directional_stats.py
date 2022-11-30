from functools import partial
from typing import Callable

import numba as nb
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera as pa
import pandera.typing as pat
from matplotlib import patches
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
from cod_analytics.math.diff_geo import wedge_field


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
        self, min_points: int = 0, *args, **kwargs
    ) -> "VectorSpaceResults":
        """Generates attacker and victim vector spaces for the given map. Passes
        any additional arguments via args and kwargs to SciPy's
        binned_statistic_2d.

        Args:
            min_points (int, optional): Minimum number of points in a bin to
                include it in the results. Defaults to 0.

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
                angle = data["angle"]
            elif av == "v":
                angle = data["angle"] + np.pi
            else:
                raise ValueError(f"Invalid av: {av}")
            return binned_statistic_2d(
                data[av + "x"],
                data[av + "y"],
                angle,
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
            amcx_partial = partial(
                angular_mean_cartesian_x, min_points=min_points
            )
            amcy_partial = partial(
                angular_mean_cartesian_y, min_points=min_points
            )
            ret = binned_stat_part(amcx_partial, "a")
            a_vals_x = ret.statistic
            x_edges, y_edges = ret.x_edge, ret.y_edge
            a_vals_y = binned_stat_part(amcy_partial, "a").statistic
            a_vals = a_vals_x + 1j * a_vals_y

            v_vals_x = binned_stat_part(amcx_partial, "v").statistic
            v_vals_y = binned_stat_part(amcy_partial, "v").statistic
            v_vals = v_vals_x + 1j * v_vals_y

        return VectorSpaceResults(a_vals, v_vals, x_edges, y_edges)


class VectorSpaceResults:
    """Class for storing the results of vector space generation, including
    methods for plotting the results and computing the geometric product of
    the input vector spaces."""

    def __init__(
        self,
        a_vals: npt.NDArray[np.complex128],
        v_vals: npt.NDArray[np.complex128],
        x_edges: npt.NDArray[np.float64],
        y_edges: npt.NDArray[np.float64],
    ) -> None:
        """Initializes the VectorSpaceResults class.

        Args:
            a_vals (npt.NDArray[np.complex128]): Mean and variance of attacker
            v_vals (npt.NDArray[np.complex128]): Mean and variance of victim
            x_edges (npt.NDArray[np.float64]): x edges of the bins
            y_edges (npt.NDArray[np.float64]): y edges of the bins
        """
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

    def geometric_product_fields(self) -> npt.NDArray[np.complex128]:
        """Generates the geometric product of the attacker and victim vector.

        Given two (n, m) complex arrays, a and v, representing the angular mean
        and variance of the attacker and victim vector spaces, respectively,
        this function returns a (n, m) complex array representing the
        multivector field of the geometric product between the attacker and
        victim vector spaces.

        Returns:
            npt.NDArray[np.complex128]: Bivector field.
        """
        a_vectors = np.stack([self.a_x, self.a_y], axis=-1)
        v_vectors = np.stack([self.v_x, self.v_y], axis=-1)
        dot = np.einsum("ijk,ijk->ij", a_vectors, v_vectors)
        wedge = wedge_field(a_vectors, v_vectors)
        return dot + 1j * wedge

    def plot_vector_field(self, ax: plt.Axes, av: str, **kwargs) -> None:
        """Plots the vector field for the given attacker or victim vector.

        Kwargs are passed directly to matplotlib.pyplot.quiver, with the
        following keyword arguments as defaults:
            "scale_units": "xy",
            "angles": "xy",
            "color": "red" if av == "a" else "blue",
            "pivot": "tail" if av == "a" else "tip",
            "scale": 1 / minimum bin size,

        Args:
            ax (plt.Axes): Axes to plot on.
            av (str): "a" for attacker, "v" for victim.

        Raises:
            ValueError: If av is not "a" or "v".
        """
        default_kwargs = {
            "scale_units": "xy",
            "angles": "xy",
        }
        default_kwargs.update(kwargs)
        if av == "a":
            u = self.a_x
            v = self.a_y
            default_kwargs["pivot"] = kwargs.get("pivot", "tail")
            default_kwargs["color"] = kwargs.get("color", "red")
        elif av == "v":
            u = -self.v_x
            v = -self.v_y
            default_kwargs["pivot"] = kwargs.get("pivot", "tip")
            default_kwargs["color"] = kwargs.get("color", "blue")
        else:
            raise ValueError(f"Invalid av: {av}")

        # normalize u and v to the bin size
        if "scale" not in default_kwargs:
            min_bin = min(self.bin_size_x, self.bin_size_y)
            default_kwargs["scale"] = 1 / (min_bin)
        x, y = np.meshgrid(
            self.bin_centers_x, self.bin_centers_y, indexing="ij"
        )
        # Draw centers of bins
        ax.scatter(x, y, c="green", s=1)
        # Draw bins
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                anchor_x = x[i, j] - self.bin_size_x / 2
                anchor_y = y[i, j] - self.bin_size_y / 2
                patch = patches.Rectangle(
                    (anchor_x, anchor_y),
                    self.bin_size_x,
                    self.bin_size_y,
                    color="black",
                    fill=False,
                )
                ax.add_patch(patch)
        # Draw vector field
        ax.quiver(x, y, u, v, **default_kwargs)

    def plot_bivector_field(self, ax: plt.Axes, **kwargs) -> None:
        bivector_field = self.generate_bivector_field()
        x, y = np.meshgrid(
            self.bin_centers_x, self.bin_centers_y, indexing="ij"
        )
        u = bivector_field.real * self.bin_size_x / 2
        v = bivector_field.imag * self.bin_size_y / 2
        ax.quiver(x, y, u, v, angles="xy", **kwargs)
