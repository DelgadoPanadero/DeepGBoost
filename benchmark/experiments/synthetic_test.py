"""
Abstract base for experiments that generate their own synthetic datasets.

Extends AbstractModelTest for cases where data is produced programmatically
rather than loaded from a fixed dataset file.  Subclasses implement
``generate_conditions()`` to yield (label, X, y) triples, and ``run()``
iterates over them automatically.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .abstract_test import AbstractModelTest


class SyntheticModelTest(AbstractModelTest):
    """
    Base class for experiments that generate data internally.

    Unlike ``BootstrapModelTest`` or ``CrossValidationModelTest``, which
    receive a fixed (X, y) dataset, subclasses of this class generate
    one dataset per experimental condition via ``generate_conditions()``.

    The ``run(name, X, y)`` signature keeps compatibility with
    ``ExperimentRunner`` — X and y are ignored since data is generated
    internally.  Subclasses must still implement ``create_batch``,
    ``score``, and ``generate_conditions``.
    """

    def generate_conditions(self) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
        """
        Yield ``(condition_label, X, y)`` for each experimental condition.

        The condition_label is appended to the experiment name when saving
        results, e.g. ``"Penguins rho=0.6"``.
        """
        raise NotImplementedError()

    def run(self, name: str, X=None, y=None) -> None:
        """
        Run the experiment across all conditions from ``generate_conditions()``.

        Parameters
        ----------
        name : str
            Base name prepended to each condition label in output files.
        X, y : ignored
            Kept for API compatibility with ``ExperimentRunner``.
        """
        for condition_label, X_c, y_c in self.generate_conditions():
            self._run_condition(f"{name} {condition_label}", X_c, y_c)

    def _run_condition(self, name: str, X: np.ndarray, y: np.ndarray) -> None:
        """
        Run ``create_batch`` loop for one condition and save all results.

        Parameters
        ----------
        name : str
            Full experiment + condition name used for output file naming.
        X : (n_samples, n_features)
        y : (n_samples,)
        """
        fold_scores: list[list[float]] = [[] for _ in self._models]

        for X_train, y_train, X_test, y_test in self.create_batch(X, y):
            for idx, model in enumerate(self._models):
                y_pred = model.fit(X_train, y_train).predict(X_test)
                fold_scores[idx].append(self.score(y_test, y_pred))

        scores = {
            model.name: np.array(fold_scores[idx])
            for idx, model in enumerate(self._models)
        }

        self._save_summary(name, scores)
        self._save_scores(name, scores)
        self._plot_scores(name, scores, self._n_bins)
        self._plot_scores_diff(name, scores, self._n_bins)
