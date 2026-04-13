from .regression import RMSEObjective, MAEObjective
from .classification import LogisticObjective, SoftmaxObjective

__all__ = [
    "RMSEObjective",
    "MAEObjective",
    "LogisticObjective",
    "SoftmaxObjective",
]

OBJECTIVES: dict[str, type] = {
    "reg:squarederror": RMSEObjective,
    "reg:absoluteerror": MAEObjective,
    "binary:logistic": LogisticObjective,
    "multi:softmax": SoftmaxObjective,
}


def get_objective(name: str):
    """
    Return an objective instance by name string.

    Parameters
    ----------
    name : str
        Objective alias.  Must be one of ``"reg:squarederror"``,
        ``"reg:absoluteerror"``, ``"binary:logistic"``, ``"multi:softmax"``.

    Returns
    -------
    BaseObjective
        An instantiated objective object.

    Raises
    ------
    ValueError
        If ``name`` is not a registered objective alias.
    """
    if name not in OBJECTIVES:
        raise ValueError(
            f"Unknown objective '{name}'. Available: {list(OBJECTIVES.keys())}",
        )
    return OBJECTIVES[name]()
