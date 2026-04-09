from sklearn.ensemble import RandomForestRegressor

from .abstract_model import AbstractModel


class RandomForestModel(AbstractModel):
    def __init__(self, **kwargs):
        self._model = RandomForestRegressor(**kwargs)

    @property
    def name(self) -> str:
        return "RandomForest"

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)
