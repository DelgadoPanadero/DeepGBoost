import json
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from benchmark.experiments.crossvalidation_test import CrossValidationModelTest

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACADEMIC_RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results", "academic")


class AcademicCrossValidationModelTest(CrossValidationModelTest):
    """Cross-validation test that writes results to the academic results directory.

    Identical to :class:`~benchmark.experiments.crossvalidation_test.CrossValidationModelTest`
    except that all output artefacts (JSON summaries, score files, PNG plots)
    are written to ``benchmark/results/academic/`` instead of
    ``benchmark/results/``.

    Parameters
    ----------
    models : list
        List of model wrappers implementing ``.fit()`` / ``.predict()``.
    task : str, optional
        Either ``"regression"`` or ``"classification"``.  Default is
        ``"regression"``.
    n_runs : int, optional
        Number of random-permutation repetitions.  Default is ``10``.
    n_bins : int, optional
        Number of histogram bins for score plots.  Default is ``100``.
    n_folds : int, optional
        Number of cross-validation folds.  Default is ``10``.
    results_dir : str, optional
        Directory where output files are written.  Defaults to
        ``benchmark/results/academic/``.

    Notes
    -----
    The ``results_dir`` parameter is the only behavioural difference from the
    parent class; all statistical logic is inherited unchanged.
    """

    def __init__(
        self,
        models,
        task: str = "regression",
        n_runs: int = 10,
        n_bins: int = 100,
        n_folds: int = 10,
        results_dir: str = ACADEMIC_RESULTS_DIR,
    ) -> None:
        super().__init__(
            models=models,
            task=task,
            n_runs=n_runs,
            n_bins=n_bins,
            n_folds=n_folds,
        )
        self._output_dir = results_dir
        os.makedirs(self._output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers — redirect every artefact to self._output_dir
    # ------------------------------------------------------------------

    def _results_exist(self, name: str) -> bool:
        file_name = os.path.join(
            self._output_dir, f"{name.replace(' ', '_').lower()}.json"
        )
        return os.path.exists(file_name)

    def _save_summary(self, name: str, scores_dict: dict) -> None:
        model_names = list(scores_dict.keys())
        reference = model_names[-1]
        ref_scores = scores_dict[reference]

        records = [
            {"model": n, "mean": float(s.mean()), "std": float(s.std())}
            for n, s in scores_dict.items()
        ]
        for model_name, scores in scores_dict.items():
            if model_name != reference:
                diff = ref_scores - scores
                records.append({
                    "diff_reference": reference,
                    "diff_model": model_name,
                    "mean": float(diff.mean()),
                    "std": float(diff.std()),
                })

        file_name = os.path.join(
            self._output_dir, f"{name.replace(' ', '_').lower()}.json"
        )
        with open(file_name, "w") as f:
            json.dump(records, f, indent=2)

    def _save_scores(self, name: str, scores_dict: dict) -> None:
        model_names = list(scores_dict.keys())
        reference = model_names[-1]
        score_arrays = list(scores_dict.values())

        records = []
        for values in zip(*score_arrays):
            record = {n: float(v) for n, v in zip(model_names, values)}
            for n, v in zip(model_names[:-1], values[:-1]):
                record[f"diff_{reference}_vs_{n}"] = float(values[-1]) - float(v)
            records.append(record)

        file_name = os.path.join(
            self._output_dir,
            f"{name.replace(' ', '_').lower()}_scores.json",
        )
        with open(file_name, "w") as f:
            json.dump(records, f, indent=2)

    def _plot_scores(self, name: str, scores_dict: dict, n_bins: int) -> None:
        all_scores = np.concatenate(list(scores_dict.values()))
        bins = np.linspace(all_scores.min(), all_scores.max(), n_bins)

        for model_name, scores in scores_dict.items():
            plt.hist(scores, bins, alpha=0.5, label=model_name)

        plt.title(f"{name} {self.metric_name} Score")
        plt.xlabel(f"{self.metric_name} Score")
        plt.ylabel("Counts")
        plt.legend(loc="upper right")
        file_name = os.path.join(
            self._output_dir,
            f"{name.replace(' ', '_').lower()}_score.png",
        )
        plt.savefig(file_name)
        plt.clf()

    def _plot_scores_diff(self, name: str, scores_dict: dict, n_bins: int) -> None:
        model_names = list(scores_dict.keys())
        reference = model_names[-1]
        ref_scores = scores_dict[reference]

        for model_name in model_names[:-1]:
            differences = ref_scores - scores_dict[model_name]
            bins = np.linspace(differences.min(), differences.max(), n_bins)
            plt.hist(
                differences,
                bins,
                alpha=0.5,
                label=f"{self.metric_name} diff {model_name}",
            )

        plt.title(f"{name} Paired {self.metric_name} Difference")
        plt.xlabel(f"Paired {self.metric_name} Difference")
        plt.ylabel("Counts")
        plt.legend(loc="upper right")
        file_name = os.path.join(
            self._output_dir,
            f"{name.replace(' ', '_').lower()}_score_diff.png",
        )
        plt.savefig(file_name)
        plt.clf()

    def run(self, name: str, X: np.ndarray, y: np.ndarray) -> None:
        """Run cross-validation and save all results to the academic directory.

        Parameters
        ----------
        name : str
            Dataset name used as a prefix for output filenames.
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target array.
        """
        full_name = f"{name} Cross Validation Test"
        if self._results_exist(full_name):
            print(f"Skipping '{full_name}' (already exists)")
            return

        X = np.asarray(X)
        y = np.asarray(y)
        scores: dict[str, list[float]] = {
            model.name: [] for model in self._models
        }

        for _ in tqdm(range(self._n_runs)):
            ids = np.random.permutation(X.shape[0])
            X_shuffled, y_shuffled = X[ids], y[ids]

            for X_train, y_train, X_test, y_test in self.create_batch(
                X_shuffled, y_shuffled
            ):
                for model in self._models:
                    y_pred = model.fit(X_train, y_train).predict(X_test)
                    scores[model.name].append(self.score(y_test, y_pred))

        scores_np = {n: np.array(vals) for n, vals in scores.items()}

        self._save_scores(full_name, scores_np)
        self._save_summary(full_name, scores_np)
        self._plot_scores(full_name, scores_np, self._n_bins)
        self._plot_scores_diff(full_name, scores_np, self._n_bins)

    def run_with_splits(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Run evaluation using predefined OpenML splits and save all results.

        Unlike :py:meth:`run`, this method does not generate random folds.
        Instead it iterates directly over the caller-supplied ``splits``, which
        are typically the train/test index pairs defined by the OpenML task
        (all repeats × folds flattened into a single list).

        Parameters
        ----------
        name : str
            Dataset name used as a prefix for output filenames.
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target array.
        splits : list of (train_idx, test_idx) tuples
            Each element is a pair of integer index arrays.  Indices refer to
            rows of ``X`` and ``y``.
        """
        full_name = f"{name} Cross Validation Test"
        if self._results_exist(full_name):
            print(f"Skipping '{full_name}' (already exists)")
            return

        X = np.asarray(X)
        y = np.asarray(y)
        scores: dict[str, list[float]] = {
            model.name: [] for model in self._models
        }

        for train_idx, test_idx in tqdm(splits):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            for model in self._models:
                y_pred = model.fit(X_train, y_train).predict(X_test)
                scores[model.name].append(self.score(y_test, y_pred))

        scores_np = {n: np.array(vals) for n, vals in scores.items()}

        self._save_scores(full_name, scores_np)
        self._save_summary(full_name, scores_np)
        self._plot_scores(full_name, scores_np, self._n_bins)
        self._plot_scores_diff(full_name, scores_np, self._n_bins)
        self._save_normalized_summary(full_name, scores_np)

    def _save_normalized_summary(self, name: str, scores_dict: dict) -> None:
        """Save normalized scores (score / max_score_on_dataset) to a JSON file.

        For each model the mean score across all splits is computed, then every
        mean is divided by the maximum mean score observed across all models.
        The resulting value lies in ``(0, 1]``, where ``1.0`` indicates the
        best-performing model on this dataset.

        Parameters
        ----------
        name : str
            Dataset name used as a prefix for the output filename.  Spaces are
            replaced with underscores and the string is lower-cased.
        scores_dict : dict
            Mapping from model name to a 1-D ``np.ndarray`` of per-fold scores.

        Notes
        -----
        The output file is written as
        ``<results_dir>/<name>_normalized.json`` and contains a JSON array of
        objects with keys ``"model"``, ``"mean"``, and ``"normalized"``.
        """
        model_names = list(scores_dict.keys())
        mean_scores = {n: float(scores_dict[n].mean()) for n in model_names}
        max_score = max(mean_scores.values())

        records = [
            {
                "model": n,
                "mean": mean_scores[n],
                "normalized": mean_scores[n] / max_score if max_score != 0 else 0.0,
            }
            for n in model_names
        ]

        file_name = os.path.join(
            self._output_dir,
            f"{name.replace(' ', '_').lower()}_normalized.json",
        )
        with open(file_name, "w") as f:
            json.dump(records, f, indent=2)
