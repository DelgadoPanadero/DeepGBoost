import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .abstract_test import AbstractModelTest


class BootstrapModelTest(AbstractModelTest):
    def __init__(
        self,
        models,
        n_bins=100,
        n_runs=100,
        test_size=0.25,
    ):
        self._models = models
        self._n_runs = n_runs
        self._n_bins = n_bins
        self._test_size = test_size

    def score(self, y_test, y_pred):
        return r2_score(y_test, y_pred)

    def create_batch(self, X, y):
        for _ in tqdm(range(self._n_runs)):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self._test_size
            )
            yield X_train, y_train, X_test, y_test

    def run(self, name, X, y):
        scores = {model.name: np.zeros((self._n_runs,)) for model in self._models}

        for i, (X_train, y_train, X_test, y_test) in enumerate(self.create_batch(X, y)):
            for model in self._models:
                y_pred = model.fit(X_train, y_train).predict(X_test)
                scores[model.name][i] = self.score(y_test, y_pred)

        name = f"{name} Bootstrap Test"
        self._save_scores(name, scores)
        self._save_summary(name, scores)
        self._plot_scores(name, scores, self._n_bins)
        self._plot_scores_diff(name, scores, self._n_bins)
