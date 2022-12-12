from typing import Callable, ParamSpec, TypeVar, cast

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from matplotlib import cm, patches
import numpy as np
import pandas as pd

from cod_analytics.assets.map_images import MapRemap
from cod_analytics.classes.map import MapImage
from cod_analytics.math.directional_stats import (
    DirectionalStats,
    VectorBundle,
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
        self.vec_bundle: VectorBundle | None = None

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
    def vector_bundle_method(func: Callable[P, T]) -> Callable[P, T]:
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
                    "method. Call the generate_vector_bundle method first."
                )

        return wrapper

    def generate_vector_bundle(
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
        self.vec_bundle = self.dir_stats.generate_vector_bundle(
            min_points=min_points, bins=bins
        )
        self.generated_spaces = True

    def initialize_plot(
        self, include_title: bool = True, **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """Initialize the plot.

        Args:
            include_title (bool, optional): Whether to include the map title.

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and axes.
        """
        fig, ax = plt.subplots(**kwargs)
        ax.imshow(self.map.image)
        if include_title:
            text = plt.text(
                100,
                100,
                s=MapRemap.remap(self.map_id),
                fontsize=30,
                color="white",
            )
            text.set_path_effects(
                [PathEffects.withStroke(linewidth=5, foreground="black")]
            )
        return fig, ax

    @vector_bundle_method
    def plot_vector_spaces(self, ax: plt.Axes) -> None:
        """Plot the vector spaces.

        Args:
            ax (plt.Axes): Axes to plot on.
        """
        self.vec_bundle.plot_vector_field(
            ax, "v", label="killed from direction"
        )
        self.vec_bundle.plot_vector_field(ax, "a", label="kill direction")
        ax.legend()

    @vector_bundle_method
    def plot_geometric_product_field(self, ax: plt.Axes) -> None:
        """Plot the geometric product field.

        Args:
            ax (plt.Axes): Axes to plot on.
        """
        self.vec_bundle.plot_geometric_product_field(ax)

    def plot_engagements(self, ax: plt.Axes, **kwargs) -> None:
        """Plot the engagements.

        Args:
            ax (plt.Axes): Axes to plot on.
        """
        default_kwargs = {
            "s": 1,
        }
        default_kwargs.update(kwargs)
        ax.scatter(self.t_df["ax"], self.t_df["ay"], c="red", label="attacker", **default_kwargs)
        ax.scatter(self.t_df["vx"], self.t_df["vy"], c="blue", label="victim", **default_kwargs)
    
    @vector_bundle_method
    def plot_histogram(self, ax: plt.Axes, av:str, log: bool = False, **kwargs) -> None:
        """Plot the histogram of the engagements using the provided binning
        data.

        Args:
            ax (plt.Axes): Axes to plot on.
            av (str): Which vector space to use for the histogram. Accepts
                "a" for attacker, "v" for victim, or "av" for both.
        """
        default_kwargs = {"alpha": 0.33}
        default_kwargs.update(kwargs)
        data = []
        if "a" in av:
            data.append(self.t_df[["ax", "ay"]].values)
        
        if "v" in av:
            data.append(self.t_df[["vx", "vy"]].values)
        
        data = np.concatenate(data)
        x_edges = self.vec_bundle.x_edges
        y_edges = self.vec_bundle.y_edges
        hist, _, _ = np.histogram2d(
            data[:, 0], data[:, 1], bins=(x_edges, y_edges)
        )
        if log:
            hist = np.log(hist)
        max_val = np.max(hist)
        hist = hist / max_val
        for i in range(hist.shape[0]):
            for j in range(hist.shape[1]):
                # Skip if magnitude is 0
                if hist[i, j] == 0:
                    continue
                anchor_x = x_edges[i]
                anchor_y = y_edges[j]
                patch = patches.Rectangle(
                    (anchor_x, anchor_y),
                    self.vec_bundle.bin_size_x,
                    self.vec_bundle.bin_size_y,
                    color=cm.viridis(hist[i, j]),
                    fill=True,
                    **default_kwargs,
                )
                ax.add_patch(patch)

    @vector_bundle_method
    def plot_geometric_histogram(self, **kwargs) -> None:
        """Plot the geometric histogram of the engagements using the provided
        binning data.
        """
        default_kwargs = {
            "bins": (
                len(self.vec_bundle.x_edges) - 1,
                len(self.vec_bundle.y_edges) - 1,
            ),
            "figsize": (10, 10),
        }
        default_kwargs.update(kwargs)
        if isinstance(default_kwargs["bins"], int):
            default_kwargs["bins"] = (default_kwargs["bins"], default_kwargs["bins"])
        
        geometric_product = self.vec_bundle.geometric_product_field()
        geometric_product = geometric_product[~np.isnan(geometric_product)]
        magnitude = np.abs(geometric_product).flatten()
        angle = np.angle(geometric_product).flatten()

        hist, _, _ = np.histogram2d(angle, magnitude, bins=default_kwargs["bins"])
        R = np.linspace(0, 1, hist.shape[0])
        THETA = np.linspace(-np.pi, np.pi, hist.shape[1])

        fig, ax = plt.subplots(**default_kwargs, subplot_kw={"projection": "polar"})
        ax.grid(False)
        ax.pcolormesh(THETA, R, hist, cmap="viridis")
        ax.yaxis.grid(True, color="white", alpha=0.5)
        ax.yaxis.set_tick_params(color="white", labelcolor="white")
        ax.set_rlabel_position(90)
        return fig, ax