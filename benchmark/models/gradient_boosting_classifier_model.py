from sklearn.ensemble import GradientBoostingClassifier

from .abstract_model import AbstractModel


class GradientBoostingClassifierModel(AbstractModel):
    def __init__(self,**kwargs):
        self._model = GradientBoostingClassifier(**kwargs)

    @property
    def name(self) -> str:
        return "GradientBoosting"

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)
