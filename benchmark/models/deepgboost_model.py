from deepgboost import DeepGBoostRegressor

from .abstract_model import AbstractModel


class DeepGBoostModel(AbstractModel):
    def __init__(self, n_layers=4, learning_rate=0.8, n_trees=25, **kwargs):
        self._model = DeepGBoostRegressor(
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
