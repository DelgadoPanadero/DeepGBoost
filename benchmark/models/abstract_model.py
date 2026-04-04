import abc


class AbstractModel(abc.ABC):
    """Common interface for all benchmark models."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def fit(self, X, y):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, X):
        raise NotImplementedError()
