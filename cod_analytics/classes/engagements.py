import matplotlib.pyplot as plt
import pandas as pd

from cod_analytics.assets.map_transform_reference import (
    MapSourceOfTruthPoints,
    retrieve_minimap_image,
)
from cod_analytics.classes import TransformReference
from cod_analytics.math.directional_stats import DirectionalStats
from cod_analytics.math.homography import Homography


class EngagementClass:
    def __init__(
        self, dataframe: pd.DataFrame, map_id: str, image_path: str = None
    ) -> None:
        self.df = dataframe
        self.map_id = map_id
        if image_path is None:
            self.image = retrieve_minimap_image(map_id)
        else:
            self.image = plt.imread(image_path)

        target_reference = TransformReference(
            map_left=0,
            map_right=self.image.shape[1],
            map_top=self.image.shape[0],
            map_bottom=0,
            map_rotation=0,
        )
        source_reference = MapSourceOfTruthPoints.get(map_id)
        self.homography: Homography = Homography.from_transform_reference(
            source=source_reference,
            target=target_reference,
        )
        self.t_df = self.homography.transform_dataframe(
            self.df,
            columns=[
                ["ax", "ay"],
                ["vx", "vy"],
            ],
        )
        self.ds = DirectionalStats(self.t_df)