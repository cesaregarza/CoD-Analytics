from typing import TypedDict


class TransformReference(TypedDict):
    map_left: float
    map_right: float
    map_top: float
    map_bottom: float
    map_rotation: float | None
