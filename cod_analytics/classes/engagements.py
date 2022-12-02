from typing import Callable, ParamSpec, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cod_analytics.classes.map import MapImage
from cod_analytics.math.directional_stats import (
    DirectionalStats,
    VectorSpaceResults,
)
from cod_analytics.math.homography import HomographyCorrection
from cod_analytics.parser.parsers import parse_map_id

T = TypeVar("T")
P = ParamSpec("P")


class Engagements:
    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        self.df = dataframe

    def filter_map(
        self, map_id: str, image_path: str = None
    ) -> "MapEngagements":
        map_id = parse_map_id(map_id)
        filtered_df = self.df.loc[self.df["map_id"] == map_id]
        return MapEngagements(filtered_df, map_id, image_path)


class MapEngagements:
    def __init__(
        self, filtered_df: pd.DataFrame, map_id: str, image_path: str = None
    ) -> None:
        """MapEngagements class handles image loading and homography creation by
        using the map_id. It also handles the creation of the directional stats
        for the map. Note that any image_path provided does not need to be the
        minimap image, it will still be used to create the homography.

        Args:
            filtered_df (pd.DataFrame): Filtered dataframe containing the
                engagements from the same map.
            map_id (str): Map id.
            image_path (str, optional): Path to the map image. If not provided,
                the image will be loaded from the cod_analytics.assets
                submodule. Defaults to None.
        """
        self.df = filtered_df
        self.map_id = map_id
        self.map = MapImage(map_id, image_path)
        self.t_df = self.transform_df(self.df)

        self.dir_stats = DirectionalStats(self.t_df)
        self.generated_spaces = False
        self.vec_spaces: VectorSpaceResults | None = None

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.map.homography.transform_dataframe(
            df, [["ax", "ay"], ["vx", "vy"]]
        )

    def calculate_center(self) -> tuple[float, float]:
        maxs = self.t_df[["ax", "ay"]].max()
        mins = self.t_df[["ax", "ay"]].min()
        return (maxs["ax"] + mins["ax"]) / 2, (maxs["ay"] + mins["ay"]) / 2

    def centerize(self, center: tuple[float, float]) -> None:
        """Centerize the map.

        Args:
            center (tuple[float, float]): Center coordinates.
        """
        c1, c2 = self.calculate_center()
        return (c1 - center[0], c2 - center[1])

    def add_correction(
        self,
        translate: tuple[float, float] = (0, 0),
        rotate: float = 0,
        scale: float = 1,
        center: tuple[float, float] = None,
        rad: bool = False,
    ) -> None:
        center = self.calculate_center() if center is None else center
        correction = HomographyCorrection(
            translate=translate,
            rotate=rotate,
            scale=scale,
            center=center,
            rad=rad,
        )
        self.map.homography.add_correction(correction)
        self.t_df = self.transform_df(self.df)

    def use_calibrated_homography(self) -> None:
        """If a calibrated homography is available, use it instead of the
        default homography.
        """
        self.map.use_calibrated_homography()
        self.t_df = self.transform_df(self.df)

    @staticmethod
    def vector_space_method(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to check if the vector spaces have been generated before
        calling the decorated function.

        Args:
            func (Callable[P, T]): Function to decorate.

        Returns:
            Callable[P, T]: Decorated function.
        """

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            self: MapEngagements = cast(MapEngagements, args[0])
            if self.generated_spaces:
                return func(*args, **kwargs)
            else:
                raise RuntimeError(
                    "Vector spaces must be generated before calling this "
                    "method. Call the generate_vector_spaces method first."
                )

        return wrapper

    def generate_vector_spaces(
        self,
        min_points: int = 10,
        bins: int | list[int] | tuple[int, int] = (15, 15),
    ) -> None:
        """Generate the directional stats vector spaces for the map.

        Args:
            min_points (int, optional): Minimum number of points to generate the
                vector space. Defaults to 10.
            bins (int | list[int] | tuple[int, int], optional): Number of bins
                to use for the directional stats. Defaults to (15, 15).
        """
        self.vec_spaces = self.dir_stats.generate_vector_spaces(
            min_points=min_points, bins=bins
        )
        self.generated_spaces = True

    def initialize_plot(self, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """Initialize the plot.

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and axes.
        """
        fig, ax = plt.subplots(**kwargs)
        ax.imshow(self.map.image)
        return fig, ax

    @vector_space_method
    def plot_vector_spaces(self, ax: plt.Axes) -> None:
        """Plot the vector spaces.

        Args:
            ax (plt.Axes): Axes to plot on.
        """
        self.vec_spaces.plot_vector_field(
            ax, "v", label="killed from direction"
        )
        self.vec_spaces.plot_vector_field(ax, "a", label="kill direction")
        ax.legend()

    @vector_space_method
    def plot_geometric_product_field(self, ax: plt.Axes) -> None:
        """Plot the geometric product field.

        Args:
            ax (plt.Axes): Axes to plot on.
        """
        self.vec_spaces.plot_geometric_product_field(ax)

    @vector_space_method
    def plot_engagements(self, ax: plt.Axes, **kwargs) -> None:
        """Plot the engagements.

        Args:
            ax (plt.Axes): Axes to plot on.
        """
        default_kwargs = {
            "s": 1,
        }
        default_kwargs.update(kwargs)
        ax.scatter(self.t_df["ax"], self.t_df["ay"], c="red", **default_kwargs)
        ax.scatter(self.t_df["vx"], self.t_df["vy"], c="blue", **default_kwargs)
