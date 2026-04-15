from .regression import RMSEMetric, MAEMetric, R2ScoreMetric
from .classification import AccuracyMetric, LogLossMetric, AUCMetric

__all__ = [
    "RMSEMetric",
    "MAEMetric",
    "R2ScoreMetric",
    "AccuracyMetric",
    "LogLossMetric",
    "AUCMetric",
]

METRICS: dict[str, type] = {
    "rmse": RMSEMetric,
    "mae": MAEMetric,
    "r2": R2ScoreMetric,
    "accuracy": AccuracyMetric,
    "logloss": LogLossMetric,
    "auc": AUCMetric,
}


def get_metric(name: str):
    """
    Return a metric instance by name string.

    Parameters
    ----------
    name : str
        Metric alias.  Must be one of ``"rmse"``, ``"mae"``, ``"r2"``,
        ``"accuracy"``, ``"logloss"``, ``"auc"``.

    Returns
    -------
    BaseMetric
        An instantiated metric object.

    Raises
    ------
    ValueError
        If ``name`` is not a registered metric alias.
    """
    if name not in METRICS:
        raise ValueError(
            f"Unknown metric '{name}'. Available: {list(METRICS.keys())}",
        )
    return METRICS[name]()
