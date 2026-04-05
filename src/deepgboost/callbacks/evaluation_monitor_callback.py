from .base_callback import TrainingCallback


class EvaluationMonitorCallback(TrainingCallback):
    """
    Print evaluation metrics to stdout after each layer.

    Parameters
    ----------
    period : int
        Print every ``period`` layers (default 1 = every layer).
    """

    def __init__(
        self,
        period: int = 1,
    ):
        self.period = period

    def after_iteration(
        self,
        model,
        epoch: int,
        evals_log: dict,
    ) -> bool:
        if (epoch + 1) % self.period == 0 and evals_log:
            parts = []
            for dataset, metrics in evals_log.items():
                for metric, val in metrics.items():
                    parts.append(f"{dataset}-{metric}: {val:.6f}")
            print(f"[{epoch + 1}]\t" + "\t".join(parts))
        return False
