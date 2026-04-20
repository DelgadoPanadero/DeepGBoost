import os
from importlib import import_module

import numpy as np

from benchmark.tools.experiment_runner import ExperimentRunner

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACADEMIC_RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results", "academic")


class AcademicExperimentRunner(ExperimentRunner):
    """Experiment runner for the academic benchmark suite.

    Extends :class:`~benchmark.tools.experiment_runner.ExperimentRunner` to
    redirect all result artefacts (JSON summaries, score files, PNG plots) to
    ``benchmark/results/academic/`` instead of ``benchmark/results/``.

    The redirection is achieved by injecting the ``results_dir`` keyword
    argument into each experiment instance constructed by
    :py:meth:`_build_experiments_for_task`.  Experiment classes that do not
    accept ``results_dir`` (e.g. the standard
    :class:`~benchmark.experiments.crossvalidation_test.CrossValidationModelTest`)
    will ignore it, while
    :class:`~benchmark.tools.academic_crossvalidation_test.AcademicCrossValidationModelTest`
    will use it to write artefacts to the correct subdirectory.

    OpenML task datasets in the academic config are handled transparently by
    the parent :py:meth:`_load_datasets` implementation, which dispatches to
    :class:`~benchmark.tools.openml_loader.OpenMLLoader` when
    ``dataset["function"] == "openml_task"``.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary (see
        :class:`~benchmark.tools.academic_config_parser.AcademicConfigParser`).
    results_dir : str, optional
        Directory where result files are written.
        Defaults to ``benchmark/results/academic/``.
    """

    def __init__(self, config: dict, results_dir: str = ACADEMIC_RESULTS_DIR) -> None:
        self._results_dir = results_dir
        os.makedirs(self._results_dir, exist_ok=True)
        super().__init__(config)

    def _load_datasets(self, config: dict) -> dict:
        """Load all datasets, using predefined OpenML splits where available.

        For entries with ``"function": "openml_task"`` the dataset is loaded
        via :py:meth:`~benchmark.tools.openml_loader.OpenMLLoader.load_with_splits`
        so that the predefined train/test index pairs are preserved.  The
        resulting entry in ``self._datasets`` is a 4-tuple
        ``(X, y, task_type, splits)`` instead of the usual 3-tuple.

        All other dataset types are handled by the parent implementation and
        stored as ``(X, y, task_type)`` 3-tuples.

        Parameters
        ----------
        config : dict
            Parsed configuration dictionary containing a ``"Datasets"`` key.

        Returns
        -------
        datasets : dict
            Mapping from dataset name to ``(X, y, task_type)`` or
            ``(X, y, task_type, splits)`` tuples.
        """
        # Delegate the parent implementation first so that non-OpenML datasets
        # are handled without duplicating logic.
        super()._load_datasets(config)

        # Re-load OpenML entries with split information, overwriting the
        # 3-tuple written by the parent.
        from benchmark.tools.openml_loader import OpenMLLoader

        loader = OpenMLLoader()
        for dataset in config["Datasets"]:
            if dataset.get("function") != "openml_task":
                continue
            task_id = int(dataset["task_id"])
            X, y, task_type, splits = loader.load_with_splits(task_id)
            self._datasets[dataset["name"]] = (X, y, task_type, splits)

        return self._datasets

    def run(self) -> None:
        """Run all experiments, dispatching to predefined splits when available.

        For each dataset the method inspects the stored tuple length.  If it is
        a 4-tuple ``(X, y, task_type, splits)`` (i.e. an OpenML dataset loaded
        with :py:meth:`~benchmark.tools.openml_loader.OpenMLLoader.load_with_splits`),
        each experiment is evaluated via
        :py:meth:`~benchmark.tools.academic_crossvalidation_test.AcademicCrossValidationModelTest.run_with_splits`.
        Otherwise the standard :py:meth:`run` path is used for backward
        compatibility.

        Notes
        -----
        Fresh model instances are created per dataset to prevent fitted state
        (e.g. XGBoost ``num_class`` or ``objective``) from leaking across
        datasets with different characteristics.
        """
        for dataset_name, data in self.datasets.items():
            if len(data) == 4:
                X, y, task, splits = data
            else:
                X, y, task = data
                splits = None

            experiments = self._build_experiments_for_task(task)
            for exp_name, experiment in experiments.items():
                if splits is not None and hasattr(experiment, "run_with_splits"):
                    experiment.run_with_splits(dataset_name, X, y, splits)
                else:
                    experiment.run(dataset_name, X, y)

    def _build_experiments_for_task(self, task: str) -> dict:
        """Return freshly instantiated experiments injected with ``results_dir``.

        Parameters
        ----------
        task : str
            Either ``"regression"`` or ``"classification"``.

        Returns
        -------
        experiments : dict
            Mapping from experiment class name to instantiated experiment object.
        """
        experiments = {}
        for experiment in self._experiment_configs:
            module = experiment["module"]
            obj = experiment["object"]
            params = experiment["parameters"]
            fresh_models = [
                getattr(import_module(mc["module"]), mc["object"])(**mc["parameters"])
                for mc in self._model_configs[task].values()
            ]
            task_params = {
                **params,
                "models": fresh_models,
                "task": task,
                "results_dir": self._results_dir,
            }
            experiments[obj] = getattr(import_module(module), obj)(**task_params)
        return experiments
