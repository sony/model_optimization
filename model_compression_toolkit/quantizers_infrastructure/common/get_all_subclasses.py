from typing import Set


def get_all_subclasses(cls: type) -> Set[type]:
    """
    This function returns a list of all subclasses of the given class,
    including all subclasses of those subclasses, and so on.
    Recursively get all subclasses of the subclass and add them to the list of all subclasses.

    Args:
        cls: A class object.

    Returns: All classes that inherit from cls.

    """

    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)])
