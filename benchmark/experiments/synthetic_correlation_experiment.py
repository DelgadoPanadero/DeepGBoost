"""
Synthetic correlation experiment.

Hypothesis 1
    DGBF outperforms XGBoost and RandomForest when input features are highly
    correlated (mean pairwise |correlation| > 0.5).

Hypothesis 2
    DGBF has a small-n advantage over competitors at fixed high correlation.

Usage
-----
    cd /home/thinbaker/Workspace/DeepGBoost
    .venv/bin/python -m benchmark.experiments.synthetic_correlation_experiment
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.linalg import toeplitz
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from .synthetic_test import SyntheticModelTest

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _toeplitz_covariance(p: int, rho: float) -> np.ndarray:
    """Return a p×p Toeplitz correlation matrix with off-diagonal decay rho^|i-j|."""
    return toeplitz(rho ** np.arange(p))


def _target(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Nonlinear target: sin(x0) + x1² + x2·x3 + Gaussian noise."""
    noise = rng.standard_normal(X.shape[0]) * 0.3
    return np.sin(X[:, 0]) + X[:, 1] ** 2 + X[:, 2] * X[:, 3] + noise


def make_correlated_dataset(
    n: int,
    p: int,
    rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw X ~ N(0, Sigma_toeplitz(rho)) and compute nonlinear y."""
    Sigma = _toeplitz_covariance(p, rho)
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n, p))
    X = Z @ L.T
    y = _target(X, rng)
    return X, y


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class SyntheticCorrelationExperiment(SyntheticModelTest):
    """
    Synthetic correlation experiment implementing two hypotheses.

    Hypothesis 1 — ``run_rho_sweep``:
        Varies input correlation ``rho`` at a fixed dataset size ``n``.

    Hypothesis 2 — ``run_n_sweep``:
        Varies dataset size ``n`` at a fixed high correlation ``rho``.

    Calling ``run(name)`` runs both sweeps sequentially.  Each (rho, n)
    condition produces its own result files via the parent infrastructure
    (``_save_summary``, ``_save_scores``, ``_plot_scores``).

    Parameters
    ----------
    models : list of AbstractModel
        Fresh model instances to evaluate.
    rho_values : sequence of float
        Correlation values for Hypothesis 1.
    n_values : sequence of int
        Dataset sizes for Hypothesis 2.
    n : int
        Fixed dataset size for the rho sweep (Hypothesis 1).
    rho : float
        Fixed correlation for the n sweep (Hypothesis 2).
    p : int
        Number of features in the synthetic dataset.
    n_folds : int
        Number of K-fold cross-validation splits per condition.
    seed : int
        Master random seed for dataset generation and CV splits.
    n_bins : int
        Number of histogram bins in score plots.
    """

    def __init__(
        self,
        models,
        rho_values: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9),
        n_values: tuple[int, ...] = (200, 500, 1000, 3000, 10000),
        n: int = 2000,
        rho: float = 0.7,
        p: int = 20,
        n_folds: int = 5,
        seed: int = RANDOM_SEED,
        n_bins: int = 10,
    ):
        self._models = models
        self._rho_values = rho_values
        self._n_values = n_values
        self._n = n
        self._rho = rho
        self._p = p
        self._n_folds = n_folds
        self._seed = seed
        self._n_bins = n_bins

    # ------------------------------------------------------------------
    # AbstractModelTest interface
    # ------------------------------------------------------------------

    @property
    def metric_name(self) -> str:
        return "R2"

    def score(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        return r2_score(y_test, y_pred)

    def create_batch(
        self, X: np.ndarray, y: np.ndarray
    ) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """K-fold cross-validation splits for one synthetic condition."""
        kf = KFold(n_splits=self._n_folds, shuffle=True, random_state=self._seed)
        for train_idx, test_idx in kf.split(X):
            yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    # ------------------------------------------------------------------
    # SyntheticModelTest interface
    # ------------------------------------------------------------------

    def generate_conditions(
        self,
    ) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
        """Default generate_conditions: yields rho sweep conditions."""
        yield from self._rho_conditions()

    # ------------------------------------------------------------------
    # Sweep methods (public entry points)
    # ------------------------------------------------------------------

    def run(self, name: str, X=None, y=None) -> None:
        """Run both hypotheses: rho sweep then n sweep."""
        self.run_rho_sweep(name)
        self.run_n_sweep(name)

    def run_rho_sweep(self, name: str) -> None:
        """
        Hypothesis 1: vary correlation at fixed n.

        Produces one result file per rho value named
        ``{name} Rho Sweep rho={rho:.1f}``.
        """
        for condition_label, X_c, y_c in self._rho_conditions():
            self._run_condition(f"{name} Rho Sweep {condition_label}", X_c, y_c)

    def run_n_sweep(self, name: str) -> None:
        """
        Hypothesis 2: vary dataset size at fixed rho.

        Produces one result file per n value named
        ``{name} N Sweep n={n}``.
        """
        for condition_label, X_c, y_c in self._n_conditions():
            self._run_condition(f"{name} N Sweep {condition_label}", X_c, y_c)

    # ------------------------------------------------------------------
    # Internal condition generators
    # ------------------------------------------------------------------

    def _rho_conditions(
        self,
    ) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(self._seed)
        for rho in self._rho_values:
            X, y = make_correlated_dataset(self._n, self._p, rho, rng)
            yield f"rho={rho:.1f}", X, y

    def _n_conditions(
        self,
    ) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(self._seed)
        for n in self._n_values:
            X, y = make_correlated_dataset(n, self._p, self._rho, rng)
            yield f"n={n}", X, y


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    from benchmark.models import (
        DeepGBoostRegressorModel,
        RandomForestModel,
        XGBoostModel,
    )

    DGBF_PARAMS = {
        "n_layers": 5,
        "n_trees": 20,
        "learning_rate": 0.8,
    }
    XGB_PARAMS = {"n_estimators": 100, "random_state": RANDOM_SEED}
    RF_PARAMS = {"n_estimators": 100, "random_state": RANDOM_SEED, "n_jobs": -1}

    models = [
        RandomForestModel(**RF_PARAMS),
        XGBoostModel(**XGB_PARAMS),
        DeepGBoostRegressorModel(**DGBF_PARAMS),
    ]

    experiment = SyntheticCorrelationExperiment(
        models=models,
        rho_values=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
        n_values=[200, 500, 1000, 3000, 10000],
        n=2000,
        rho=0.7,
        p=20,
        n_folds=5,
        seed=RANDOM_SEED,
    )

    experiment.run("Synthetic Correlation")
