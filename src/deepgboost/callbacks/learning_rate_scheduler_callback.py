from .base_callback import TrainingCallback


class LearningRateSchedulerCallback(TrainingCallback):
    """
    Adjust ``model.learning_rate`` before each boosting layer.

    Parameters
    ----------
    schedule_fn : callable
        A function ``f(epoch: int) -> float`` that returns the new
        learning rate for that layer.

    Example::

        scheduler = LearningRateScheduler(lambda epoch: 0.1 * 0.95**epoch)
    """

    def __init__(self, schedule_fn):
        self.schedule_fn = schedule_fn

    def before_iteration(
        self,
        model,
        epoch: int,
        evals_log: dict,
    ) -> bool:
        model.learning_rate = float(self.schedule_fn(epoch))
        return False
