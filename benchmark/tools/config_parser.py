import json

_MODEL_KEYS = {"module", "object", "parameters"}
_EXPERIMENT_KEYS = {"module", "object", "parameters", "task"}
_DATASET_REQUIRED_KEYS = {"name", "url", "function", "file"}


class ConfigParser:

    def __init__(self, config_file="config.json"):
        try:
            with open(config_file) as file:
                config = json.load(file)
        except Exception:
            raise Exception(f"{config_file} has not a proper json structure")

        try:
            self._validate(config)
        except Exception:
            raise Exception(
                f"{config_file} does not follow the expected json schema"
            )

        self._config = config

    @property
    def config(self):
        return self._config

    def _validate(self, config):
        assert {"Description", "Experiments", "Datasets"} <= config.keys()
        assert "RegressionModels" in config or "ClassificationModels" in config

        for models_key in ("RegressionModels", "ClassificationModels"):
            if models_key in config:
                assert isinstance(config[models_key], dict) and len(config[models_key]) > 0
                for model_cfg in config[models_key].values():
                    assert model_cfg.keys() == _MODEL_KEYS
                    assert isinstance(model_cfg["module"], str)
                    assert isinstance(model_cfg["object"], str)
                    assert isinstance(model_cfg["parameters"], dict)

        assert isinstance(config["Experiments"], list) and len(config["Experiments"]) > 0
        for exp in config["Experiments"]:
            assert _EXPERIMENT_KEYS <= exp.keys()
            assert isinstance(exp["module"], str)
            assert isinstance(exp["object"], str)
            assert isinstance(exp["parameters"], dict)
            assert exp["task"] in ("regression", "classification")

        assert isinstance(config["Datasets"], list) and len(config["Datasets"]) > 0
        for dataset in config["Datasets"]:
            assert _DATASET_REQUIRED_KEYS <= dataset.keys()
            assert isinstance(dataset["name"], str)
            assert isinstance(dataset["url"], str)
            assert isinstance(dataset["function"], str)
            assert isinstance(dataset["file"], str)
            assert dataset.get("task", "regression") in ("regression", "classification")
