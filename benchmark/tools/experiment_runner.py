import io
import os
import urllib.request
import zipfile
from importlib import import_module

import numpy as np
import pandas as pd

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

        models = {}
        for name, model_config in config["Models"].items():
            module = model_config["module"]
            object = model_config["object"]
            params = model_config["parameters"]

            models[name] = getattr(import_module(module), object)(**params)

        self._models = models

        return models

    def _load_experiments(self, config):

        experiments = {}
        for experiment in config["Experiments"]:
            module = experiment["module"]
            object = experiment["object"]
            params = experiment["parameters"]

            params["models"] = list(self._models.values())

            experiments[object] = getattr(import_module(module), object)(
                **params
            )

        self._experiments = experiments

        return experiments

    def _load_datasets(self, config):

        datasets = {}
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

                if zip_entry:
                    with urllib.request.urlopen(dataset["url"]) as response:
                        z = zipfile.ZipFile(io.BytesIO(response.read()))
                    with z.open(zip_entry) as entry:
                        df = func(entry, **kwargs)
                else:
                    df = func(dataset["url"], **kwargs)

                df.to_csv(file_path, index=False)

            data = pd.read_csv(file_path)
            X, y = np.array(data.iloc[:, 0:-1]), np.array(data.iloc[:, -1])

            datasets[dataset["name"]] = (X, y)

        self._datasets = datasets

        return datasets

    def run(self):

        for dataset_name, dataset in self.datasets.items():
            X, y = dataset
            for name, experiment in self.experiments.items():
                experiment.run(dataset_name, X, y)
