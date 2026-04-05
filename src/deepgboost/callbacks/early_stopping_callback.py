import copy
from typing import Any
from .base_callback import TrainingCallback


class EarlyStoppingCallback(TrainingCallback):
    """
    Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of layers with no improvement before stopping.
    metric : str
        Metric key to monitor inside ``evals_log`` values.
        The key is looked up in the *first* eval set in ``evals_log``.
    data : str or None
        Name of the eval dataset to monitor.  If ``None``, uses the first
        dataset found in ``evals_log``.
    restore_best : bool
        If ``True``, restores the model to the best-seen state when stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        metric: str = "train_loss",
        data: str | None = None,
        restore_best: bool = True,
        min_delta: float = 1e-6,
    ):
        self.patience = patience
        self.metric = metric
        self.data = data
        self.restore_best = restore_best
        self.min_delta = min_delta

        self._best_score: float | None = None
        self._best_epoch: int = 0
        self._best_graph: Any = None
        self._best_weights: Any = None
        self._best_linear: Any = None
        self._wait: int = 0

    def before_training(self, model) -> None:
        self._best_score = None
        self._best_epoch = 0
        self._wait = 0

    def after_iteration(
        self,
        model,
        epoch: int,
        evals_log: dict,
    ) -> bool:
        if not evals_log:
            return False

        # Pick dataset to monitor
        dataset = self.data or next(iter(evals_log))
        if dataset not in evals_log:
            return False

        score = evals_log[dataset].get(self.metric)
        if score is None:
            return False

        # Determine if improvement (lower is better for loss metrics)
        improved = (
            self._best_score is None
            or score < self._best_score - self.min_delta
        )

        if improved:
            self._best_score = score
            self._best_epoch = epoch
            self._wait = 0
            if self.restore_best:
                self._best_graph = copy.deepcopy(model.graph_)
                self._best_weights = copy.deepcopy(model.weights_)
                self._best_linear = copy.deepcopy(model.linear_models_)
        else:
            self._wait += 1

        if self._wait >= self.patience:
            if self.restore_best and self._best_graph is not None:
                model.graph_ = self._best_graph
                model.weights_ = self._best_weights
                model.linear_models_ = self._best_linear
            return True  # stop

        return False
