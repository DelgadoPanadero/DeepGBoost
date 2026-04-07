from deepgboost import DeepGBoostClassifier

from .abstract_model import AbstractModel


class DeepGBoostClassifierModel(AbstractModel):
    def __init__(self, n_layers=10, learning_rate=0.3, n_trees=10, **kwargs):
        self._model = DeepGBoostClassifier(
            n_layers=n_layers,
            learning_rate=learning_rate,
            n_trees=n_trees,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "DeepGBoost"

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)
