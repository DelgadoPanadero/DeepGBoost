import io
import os
import urllib.request
import zipfile
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

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
        self._models = {}

        for task_key, task in [("RegressionModels", "regression"), ("ClassificationModels", "classification")]:
            models = {}
            for name, model_config in config.get(task_key, {}).items():
                module = model_config["module"]
                obj = model_config["object"]
                params = model_config["parameters"]
                models[name] = getattr(import_module(module), obj)(**params)
            self._models[task] = models

        return self._models

    def _load_experiments(self, config):
        self._experiments = {"regression": {}, "classification": {}}

        for experiment in config["Experiments"]:
            module = experiment["module"]
            obj = experiment["object"]
            params = experiment["parameters"]
            task = experiment.get("task", "regression")

            params["models"] = list(self._models[task].values())
            self._experiments[task][obj] = getattr(import_module(module), obj)(**params)

        return self._experiments

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

                if "drop_columns" in dataset:
                    df = df.drop(columns=dataset["drop_columns"], errors="ignore")

                df.to_csv(file_path, index=False)

            data = pd.read_csv(file_path)

            if dataset.get("dropna"):
                data = data.dropna().reset_index(drop=True)

            if "target_column" in dataset:
                target_col = dataset["target_column"]
                cols = [c for c in data.columns if c != target_col] + [target_col]
                data = data[cols]

            if "categorical_columns" in dataset:
                cat_cols = dataset["categorical_columns"]
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                data[cat_cols] = encoder.fit_transform(
                    data[cat_cols].astype(str).apply(lambda col: col.str.strip())
                )

            task = dataset.get("task", "regression")
            X = np.array(data.iloc[:, :-1])

            if task == "classification":
                y_raw = data.iloc[:, -1].astype(str).str.strip().str.rstrip(".")
                le = LabelEncoder()
                y = le.fit_transform(y_raw)
            else:
                y = np.array(data.iloc[:, -1])

            self._datasets[dataset["name"]] = (X, y, task)

        return self._datasets

    def run(self):
        for dataset_name, (X, y, task) in self.datasets.items():
            for name, experiment in self._experiments[task].items():
                experiment.run(dataset_name, X, y)
