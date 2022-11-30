import numpy as np
from matplotlib import pyplot as plt

from cod_analytics.assets import asset_path
from cod_analytics.assets.map_images import MapPathRemap


def retrieve_minimap_image(map_id: str) -> np.ndarray:
    """Retrieve the minimap image for a given map_id

    Args:
        map_id (str): The map_id to retrieve the minimap for

    Returns:
        plt.Axes: The minimap image
    """

    image_path = asset_path / "map_images" / MapPathRemap.remap(map_id)
    return plt.imread(image_path)
