from xgboost import XGBClassifier

from .abstract_model import AbstractModel


class XGBoostClassifierModel(AbstractModel):
    def __init__(self, **kwargs):
        self._model = XGBClassifier(**kwargs)

    @property
    def name(self) -> str:
        return "XGBoost"

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)
