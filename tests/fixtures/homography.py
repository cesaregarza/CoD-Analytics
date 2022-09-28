import pytest

from cod_analytics.classes import TransformReference


@pytest.fixture
def transform_reference_base() -> TransformReference:
    """TransformReference fixture."""
    return {
        "map_left": 0.0,
        "map_right": 1.0,
        "map_top": 0.0,
        "map_bottom": 1.0,
        "map_rotation": 0.0,
    }


@pytest.fixture
def transform_reference_rotated() -> TransformReference:
    """TransformReference fixture."""
    return {
        "map_left": 0.0,
        "map_right": 1.0,
        "map_top": 0.0,
        "map_bottom": 1.0,
        "map_rotation": 90.0,
    }


@pytest.fixture
def transform_reference_scaled() -> TransformReference:
    """TransformReference fixture."""
    return {
        "map_left": 0.0,
        "map_right": 2.0,
        "map_top": 0.0,
        "map_bottom": 2.0,
        "map_rotation": 0.0,
    }


@pytest.fixture
def transform_reference_scaled_rotated() -> TransformReference:
    """TransformReference fixture."""
    return {
        "map_left": 0.0,
        "map_right": 2.0,
        "map_top": 0.0,
        "map_bottom": 2.0,
        "map_rotation": 90.0,
    }
