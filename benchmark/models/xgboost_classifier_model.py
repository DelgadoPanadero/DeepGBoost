from xgboost import XGBClassifier

from .abstract_model import AbstractModel


class XGBoostClassifierModel(AbstractModel):

    def __init__(self, n_estimators=100, **kwargs):
        self._n_estimators = n_estimators
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "XGBoost"

    def fit(self, X, y):
        self._model = XGBClassifier(n_estimators=self._n_estimators, **self._kwargs)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)
