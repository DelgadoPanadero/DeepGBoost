import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

from .abstract_test import AbstractModelTest


class CrossValidationModelTest(AbstractModelTest):
    def __init__(
        self,
        models,
        n_runs=10,
        n_bins=100,
        n_folds=10,
    ):
        self._models = models
        self._n_runs = n_runs
        self._n_bins = n_bins
        self._n_folds = n_folds

    def score(self, y_test, y_pred):
        return r2_score(y_test, y_pred)

    def create_batch(self, X, y):
        n_rows = X.shape[0]
        for i in range(self._n_folds):
            idx = set(range(
                i * n_rows // self._n_folds,
                (i + 1) * n_rows // self._n_folds,
            ))
            train_mask = [j not in idx for j in range(n_rows)]
            test_mask = [j in idx for j in range(n_rows)]
            yield X[train_mask], y[train_mask], X[test_mask], y[test_mask]

    def run(self, name, X, y):
        scores = {model.name: np.zeros((self._n_runs,)) for model in self._models}

        for j in tqdm(range(self._n_runs)):
            ids = np.random.permutation(X.shape[0])
            X_shuffled, y_shuffled = X[ids], y[ids]

            for i, (X_train, y_train, X_test, y_test) in enumerate(
                self.create_batch(X_shuffled, y_shuffled)
            ):
                for model in self._models:
                    y_pred = model.fit(X_train, y_train).predict(X_test)
                    scores[model.name][j] += self.score(y_test, y_pred) / self._n_folds

        name = f"{name} Cross Validation Test"
        self._save_scores(name, scores)
        self._save_summary(name, scores)
        self._plot_scores(name, scores, self._n_bins)
        self._plot_scores_diff(name, scores, self._n_bins)
