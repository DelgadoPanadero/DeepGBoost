from deepgboost import DeepGBoostRegressor

from .abstract_model import AbstractModel


class DeepGBoostRegressorModel(AbstractModel):
    def __init__(self, **kwargs):
        self._model = DeepGBoostRegressor(**kwargs)

    @property
    def name(self) -> str:
        return "DeepGBoost"

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)
