import numba as nb
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera as pa
import pandera.typing as pat
from scipy.stats import binned_statistic_2d

from cod_analytics.math.compiled_directional_functions import (
    angular_mean_var,
    cartesian_to_polar,
    polar_to_cartesian,
    project_to_unit_circle,
)


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

    def generate_vector_spaces(
        self, map_id: str, *args, **kwargs
    ) -> tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Generates attacker and victim vector spaces for the given map.

        Args:
            map_id (str): Map ID.

        Returns:
            tuple:
                npt.NDArray[np.complex128]: Mean and variance of attacker
                    vectors along bins. Real part is the mean angle in radians,
                    imaginary part is the vector variance.
                npt.NDArray[np.complex128]: Mean and variance of victim vectors
                    along bins. Real part is the mean angle in radians,
                    imaginary part is the vector variance.
                npt.NDArray[np.float64]: Bin edges along the x-axis.
                npt.NDArray[np.float64]: Bin edges along the y-axis.
        """
        mask = self.data["map_id"] == map_id
        data = self.data.loc[mask, :]
        x_max = data[["ax", "vx"]].max().max()
        x_min = data[["ax", "vx"]].min().min()
        y_max = data[["ay", "vy"]].max().max()
        y_min = data[["ay", "vy"]].min().min()
        a_vals, x_edges, y_edges, _ = binned_statistic_2d(
            data["ax"],
            data["ay"],
            data["angle"],
            statistic=angular_mean_var,
            range=[[x_min, x_max], [y_min, y_max]],
            *args,
            **kwargs,
        )
        v_vals, _, _, _ = binned_statistic_2d(
            data["vx"],
            data["vy"],
            -data["angle"],
            statistic=angular_mean_var,
            range=[[x_min, x_max], [y_min, y_max]],
            *args,
            **kwargs,
        )
        return (a_vals, v_vals, x_edges, y_edges)
