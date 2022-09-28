from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd


class TransformReference(TypedDict):
    map_left: float
    map_right: float
    map_top: float
    map_bottom: float
    map_rotation: float | None


def parse_match_events(
    json: dict,
    with_transform: bool = False,
    *,
    to_reference: TransformReference | None = None,
) -> pd.DataFrame:
    """Parses the match events from a match json.

    Args:
        json (dict): Match json.
        with_transform (bool, optional): Whether to transform the attacker and
            victim coordinates for visualization. Defaults to False. Untested.
        *args: Unused.
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

    # Create reference to the player dict
    reference_dict: dict[int, str] = {}
    for i, team in enumerate(teams):
        team_id = i * 1000
        for j, player in enumerate(team):
            player_id = team_id + j
            reference_dict[player_id] = player["unoUsername"]

    engagements_df["attacker"] = engagements_df["a"].map(reference_dict)
    engagements_df["victim"] = engagements_df["v"].map(reference_dict)
    engagements_df["match_id"] = match_id
    engagements_df["map"] = map_name
    engagements_df["mode"] = mode
    engagements_df["match_start"] = match_start
    engagements_df["match_end"] = match_end

    engagements_df = engagements_df.drop(columns=["a", "v"])
    desired_order = [
        "match_id",
        "map",
        "mode",
        "match_start",
        "match_end",
        "attacker",
        "victim",
        "time",
        "ax",
        "ay",
        "vx",
        "vy",
        "aLoc",
        "vLoc",
        "cause",
    ]
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
        engagements_df.loc[:, ["ax", "ay"]] = transform_coordinates(
            engagements_df[["ax", "ay"]].values,
            reference_from,
            to_reference,
        )
        engagements_df.loc[:, ["vx", "vy"]] = transform_coordinates(
            engagements_df[["vx", "vy"]].values,
            reference_from,
            to_reference,
        )
    return engagements_df.loc[:, desired_order]


def transform_coordinates(
    coordinates: npt.NDArray[np.float64],
    reference_from: TransformReference,
    reference_to: TransformReference,
) -> npt.NDArray[np.float64]:
    """Transforms coordinates to a new coordinate system.

    Given a set of coordinates and the reference edges of the map, this function
    transforms the coordinates to a new coordinate system using homography.

    Args:
        coordinates (npt.NDArray[np.float64]): Array of coordinates to transform
        reference_from (TransformReference): Reference edges and rotation of the
            original coordinate system.
        reference_to (TransformReference): Reference edges and rotation of the
            new coordinate system.

    Returns:
        npt.NDArray[np.float64]: Array of transformed coordinates.
    """
    if reference_from["map_rotation"] is None:
        from_rotation = np.eye(2)
    else:
        from_map_rotation = reference_from["map_rotation"]
        from_map_rotation = np.deg2rad(from_map_rotation)
        from_rotation = np.array(
            [
                [np.cos(from_map_rotation), -np.sin(from_map_rotation)],
                [np.sin(from_map_rotation), np.cos(from_map_rotation)],
            ]
        )

    if reference_to["map_rotation"] is None:
        to_rotation = np.eye(2)
    else:
        to_map_rotation = reference_to["map_rotation"]
        to_map_rotation = np.deg2rad(to_map_rotation)
        to_rotation = np.array(
            [
                [np.cos(to_map_rotation), -np.sin(to_map_rotation)],
                [np.sin(to_map_rotation), np.cos(to_map_rotation)],
            ]
        )

    # Turn the edges into explicit coordinates, then rotate them
    from_points = np.array(
        [
            [reference_from["map_left"], reference_from["map_top"]],
            [reference_from["map_right"], reference_from["map_top"]],
            [reference_from["map_right"], reference_from["map_bottom"]],
            [reference_from["map_left"], reference_from["map_bottom"]],
        ],
        dtype=np.float64,
    )

    to_points = np.array(
        [
            [reference_to["map_left"], reference_to["map_top"]],
            [reference_to["map_right"], reference_to["map_top"]],
            [reference_to["map_right"], reference_to["map_bottom"]],
            [reference_to["map_left"], reference_to["map_bottom"]],
        ],
        dtype=np.float64,
    )

    from_points = from_points @ from_rotation
    to_points = to_points @ to_rotation

    # Use homography to transform the coordinates. With only four points, this
    # method is not very accurate, but it's good enough for visualization.
    target_vector = to_points.reshape((-1, 1))

    # Create the matrix of the source points
    matrix_list: list[npt.NDArray[np.float64]] = []
    for source_point in from_points:
        base_vector = np.zeros(6)
        base_vector[:3] = [*source_point, 1]
        matrix_list += [base_vector.copy()]
        matrix_list += [np.roll(base_vector, 3)]

    matrix = np.array(matrix_list)

    # Use least squares to solve for a1, a2, a3, a4, a5, a6
    solution = np.linalg.lstsq(matrix, target_vector, rcond=None)
    solution_vector = solution[0]
    transform_vector = np.zeros(3)
    transform_vector[-1] = 1
    solution_matrix = np.vstack(
        [solution_vector.reshape(2, -1), transform_vector]
    )

    # Prepare the coordinates for transformation
    ones_vector = np.ones((coordinates.shape[0], 1))
    map_coordinates = np.hstack([coordinates, ones_vector])

    # Transform the coordinates in homogeneous space, then strip the ones.
    initial_solution = solution_matrix @ map_coordinates.T
    return np.delete(initial_solution, 2, axis=0).T #type: ignore
