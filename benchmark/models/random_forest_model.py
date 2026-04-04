from sklearn.ensemble import RandomForestRegressor

from .abstract_model import AbstractModel


class RandomForestModel(AbstractModel):

    def __init__(self, n_estimators=100, **kwargs):
        self._model = RandomForestRegressor(n_estimators=n_estimators, **kwargs)

    @property
    def name(self) -> str:
        return "RandomForest"

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)
