import pandas as pd

from cod_analytics.classes import TransformReference
from cod_analytics.constants import ENG_COLUMN_ORDER
from cod_analytics.math.homography import Homography


def parse_match_events(
    json: dict,
    with_transform: bool = False,
    *,
    to_reference: TransformReference | None = None,
) -> pd.DataFrame:
    """Parses the match events from a match json.

    Args:
        json (dict): Match json.
        with_transform (bool): Whether to transform the attacker and
            victim coordinates for visualization. Defaults to False. Untested.
        to_reference (TransformReference | None, optional): Reference edges and
            rotation of the new coordinate system. Defaults to None, which is
            a 1024x1024 map with no rotation.

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

    engagements_df = engagements_df.drop(columns=["a", "v"])
    desired_order = ENG_COLUMN_ORDER
    if with_transform:
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
        engagements_df = homography.transform_dataframe(
            engagements_df,
            columns=[
                ["ax", "ay"],
                ["vx", "vy"],
            ],
        )
        engagements_df["transformed"] = True
    else:
        engagements_df["transformed"] = False
    return engagements_df.loc[:, desired_order]
