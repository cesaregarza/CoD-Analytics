from typing import TypedDict


class TransformReference(TypedDict):
    map_left: float
    map_right: float
    map_top: float
    map_bottom: float
    map_rotation: float | None


class RemapClass:
    @classmethod
    def remap(cls, key: str, reverse: bool = False) -> str:
        """Remap a key to a new value based on the remap dictionary

        Args:
            key (str): The key to remap
            reverse (bool, optional): Whether to reverse the remap. Defaults to
                False.

        Returns:
            str: The remapped key
        """

        return_dict = {
            name: getattr(cls, name)
            for name in dir(cls)
            if not name.startswith("__") and not callable(getattr(cls, name))
        }

        if reverse:
            return_dict = {return_dict[key]: key for key in return_dict}

        return return_dict[key]
