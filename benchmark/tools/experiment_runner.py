import io
import os
import urllib.request
import zipfile
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ExperimentRunner:
    def __init__(self, config):

        try:
            self._load_models(config)
        except:
            raise Exception("Error loading model modules")

        try:
            self._load_datasets(config)
        except:
            raise Exception("Error loading datatset")

        try:
            self._load_experiments(config)
        except:
            raise Exception("Error loading experiment modules")

    @property
    def models(self):
        return self._models

    @property
    def datasets(self):
        return self._datasets

    @property
    def experiments(self):
        return self._experiments

    def _load_models(self, config):
        # Store raw configs only; instantiation happens per-dataset in run().
        self._model_configs = {}
        self._models = {}

        for task in ["regression", "classification"]:
            self._model_configs[task] = config.get(task, {})
            self._models[task] = {}

        return self._models

    def _load_experiments(self, config):
        # Store experiment configs keyed by task so fresh model instances can be
        # created per-dataset in run() — avoids XGBoost objective/num_class carry-over.
        self._experiment_configs = config["Experiments"]
        self._experiments = {"regression": {}, "classification": {}}
        return self._experiments

    def _build_experiments_for_task(self, task: str):
        """Return a dict of freshly instantiated experiments with new model instances."""
        experiments = {}
        for experiment in self._experiment_configs:
            module = experiment["module"]
            obj = experiment["object"]
            params = experiment["parameters"]
            fresh_models = [
                getattr(import_module(mc["module"]), mc["object"])(**mc["parameters"])
                for mc in self._model_configs[task].values()
            ]
            task_params = {**params, "models": fresh_models, "task": task}
            experiments[obj] = getattr(import_module(module), obj)(**task_params)
        return experiments

    def _load_datasets(self, config):
        self._datasets = {}

        for dataset in config["Datasets"]:
            file_path = os.path.join(BENCHMARK_DIR, dataset["file"])
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                func = getattr(pd, dataset["function"])

                zip_entry = dataset.get("zip_entry")
                encoding = dataset.get("encoding", "utf-8")
                kwargs = {}
                if dataset["function"] == "read_csv":
                    if "sep" in dataset:
                        kwargs["sep"] = dataset["sep"]
                    kwargs["encoding"] = encoding
                    if "column_names" in dataset:
                        kwargs["header"] = None
                        kwargs["names"] = dataset["column_names"]
                    if dataset.get("skipinitialspace"):
                        kwargs["skipinitialspace"] = True

                if zip_entry:
                    with urllib.request.urlopen(dataset["url"]) as response:
                        z = zipfile.ZipFile(io.BytesIO(response.read()))
                    with z.open(zip_entry) as entry:
                        df = func(entry, **kwargs)
                else:
                    df = func(dataset["url"], **kwargs)

                df.to_csv(file_path, index=False)

            data = pd.read_csv(file_path).dropna().reset_index(drop=True)
            task = dataset.get("task", "regression")
            target = dataset["target_column"]

            X = data.drop(target,axis=1)
            if cat_cols := X.select_dtypes(exclude=['number']).columns.tolist():
                encoder = OneHotEncoder()
                new_data = encoder.fit_transform(
                    X[cat_cols].astype(str).apply(lambda col: col.str.strip())
                )
                X = X.drop(cat_cols, axis=1)
                X[encoder.get_feature_names_out()] = new_data.todense()
                X = X.values

            y = data[target]
            if task == "classification":
                encoder = LabelEncoder()
                y = encoder.fit_transform(
                    y.astype(str).apply(lambda col: col.strip())
                )

            self._datasets[dataset["name"]] = (X, y, task)

        return self._datasets

    def run(self):
        for dataset_name, (X, y, task) in self.datasets.items():
            # Fresh model instances per dataset to prevent fitted state carry-over
            # (e.g. XGBoost multi:softmax / num_class leaking into binary tasks).
            experiments = self._build_experiments_for_task(task)
            for name, experiment in experiments.items():
                experiment.run(dataset_name, X, y)
