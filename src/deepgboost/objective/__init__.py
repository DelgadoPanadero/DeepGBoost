from .regression import RMSEObjective, MAEObjective
from .classification import LogisticObjective, SoftmaxObjective

__all__ = ["RMSEObjective", "MAEObjective", "LogisticObjective", "SoftmaxObjective"]

OBJECTIVES: dict[str, type] = {
    "reg:squarederror": RMSEObjective,
    "reg:absoluteerror": MAEObjective,
    "binary:logistic": LogisticObjective,
    "multi:softmax": SoftmaxObjective,
}


def get_objective(name: str):
    """Return an objective instance by name string."""
    if name not in OBJECTIVES:
        raise ValueError(
            f"Unknown objective '{name}'. Available: {list(OBJECTIVES.keys())}"
        )
    return OBJECTIVES[name]()
