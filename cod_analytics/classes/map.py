import matplotlib.pyplot as plt

from cod_analytics.assets.map_transform_reference import (
    MapSourceOfTruthPoints,
    MapCorrections,
    retrieve_minimap_image,
    MapCalibrationReference,
)
from cod_analytics.classes import TransformReference
from cod_analytics.math.homography import Homography


class MapImage:
    """Map class that handles image loading and homography creation based on
    the map id."""

    def __init__(self, map_id: str, image_path: str = None) -> None:
        """Map class that handles image loading and homography creation based on
        the map id.

        Args:
            map_id (str): Map id.
            image_path (str, optional): Path to the map image. If not provided,
                the image will be loaded from the cod_analytics.assets
                submodule. Defaults to None.
        """
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
        self.homography.add_correction(MapCorrections.get(map_id))

    def use_calibrated_homography(self) -> None:
        """Use the calibrated homography for the map."""
        try:
            self.homography = MapCalibrationReference.get_transform(self.map_id)
        except ValueError:
            pass
