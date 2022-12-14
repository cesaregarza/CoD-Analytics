import typing

import numpy as np
import numpy.typing as npt
import pandas as pd

from cod_analytics.assets.map_images import MapRemap, MapRemapSimple
from cod_analytics.classes import TransformReference
from cod_analytics.constants import ENG_COLUMN_ORDER
from cod_analytics.math.homography import Homography


def parse_map_id(map_id: str) -> str:
    """Parse a map ID and return the map name.

    Attempts to find the map name through various methods. First, it checks if
    the map ID is a simple map name. If not, it checks if the map ID is the
    proper map name. If not, it finally checks if the map ID is a direct map.
    If none of these are true, it raises a ValueError.

    Args:
        map_id (str): The map ID to parse. Can be a simple map name, a proper
            map name, or a direct map ID.

    Raises:
        ValueError: If the map ID is not a simple map name, a proper map name,
            or a direct map ID.

    Returns:
        str: The associated map ID.
    """
    try:
        return MapRemapSimple.remap(map_id)
    except KeyError:
        pass

    try:
        return MapRemap.remap(map_id, reverse=True)
    except KeyError:
        pass

    try:
        MapRemap.remap(map_id)
    except KeyError:
        raise ValueError(f"Invalid map ID: {map_id}")

    return map_id


def parse_match_events(
    json: dict,
    with_transform: bool = False,
    *,
    to_reference: TransformReference | None = None,
    from_points: npt.NDArray[np.float64] | None = None,
    to_points: npt.NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    """Parses the match events from a match json.

    Args:
        json (dict): Match json.
        with_transform (bool): Whether to transform the attacker and
            victim coordinates for visualization. Defaults to False. Requires a
            combination of parameters to work, the default options are not great
            with all maps. Square maps fare better, but non-square maps have
            significant issues.
        to_reference (TransformReference, optional): Reference edges and
            rotation of the new coordinate system. Defaults to None, which is
            a 1024x1024 map with no rotation.
        from_points (npt.NDArray[np.float64], optional): Reference points for
            the map coordinates to use. Assumed to be a higher priority than the
            TransformReference, will override. Must be used concurrently with
            to_points.
        to_points (npt.NDArray[np.float64], optional): Reference points for the
            new coordinate system. Assumed to be a higher priority than the
            TransformReference, will override. Must be used concurrently with
            from_points.

    Returns:
        pd.DataFrame: Dataframe containing the match events.
    """
    if "data" in json:
        json = json["data"]

    match_id = json["matchId"]
    map_name = json["map"]["mapId"]
    teams = json["teams"]
    mode = json["mode"]
    match_start = json["matchStart"]
    match_end = json["matchEnd"]
    engagements = json["engagements"]

    engagements_df = pd.json_normalize(engagements)

    # Create reference for player usernames
    reference_dict: dict[int, str] = {}
    for i, team in enumerate(teams):
        team_id = i * 1000
        for j, player in enumerate(team):
            player_id = team_id + j
            reference_dict[player_id] = player["unoUsername"]

    # Add constant columns
    engagements_df["attacker"] = engagements_df["a"].map(reference_dict)
    engagements_df["victim"] = engagements_df["v"].map(reference_dict)
    engagements_df["match_id"] = match_id
    engagements_df["map"] = map_name
    engagements_df["mode"] = mode
    engagements_df["match_start"] = match_start
    engagements_df["match_end"] = match_end

    engagements_df["team_a"] = engagements_df["a"] // 1000
    engagements_df["team_v"] = engagements_df["v"] // 1000

    engagements_df = engagements_df.drop(columns=["a", "v"])
    desired_order = ENG_COLUMN_ORDER

    if not with_transform:
        engagements["transformed"] = False
        return engagements_df.loc[:, desired_order]

    if (to_points is None) and (from_points is None):
        reference_from = TransformReference(
            map_left=json["map"]["left"],
            map_right=json["map"]["right"],
            map_top=json["map"]["top"],
            map_bottom=json["map"]["bottom"],
            map_rotation=json["map"]["rotation"],
        )
        if to_reference is None:
            to_reference = TransformReference(
                map_left=0,
                map_right=1024,
                map_top=0,
                map_bottom=1024,
                map_rotation=None,
            )
        homography = Homography.from_transform_reference(
            reference_from, to_reference
        )
    elif (to_points is not None) and (from_points is not None):
        homography = Homography()
        homography.fit(from_points, to_points)
    else:
        raise ValueError(
            "to_points and from_points must be used at the same time"
        )
    engagements_df = homography.transform_dataframe(
        engagements_df,
        columns=[
            ["ax", "ay"],
            ["vx", "vy"],
        ],
    )
    engagements_df["transformed"] = True
    return engagements_df.loc[:, desired_order]
