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
    """Return a metric instance by name string."""
    if name not in METRICS:
        raise ValueError(
            f"Unknown metric '{name}'. Available: {list(METRICS.keys())}"
        )
    return METRICS[name]()
