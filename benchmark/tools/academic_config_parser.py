import json

_MODEL_KEYS = {"module", "object", "parameters"}
_EXPERIMENT_KEYS = {"module", "object", "parameters"}
_DATASET_REQUIRED_KEYS = {"name", "function"}
_CSV_DATASET_KEYS = {"name", "url", "function", "file"}
_OPENML_DATASET_KEYS = {"name", "function", "task_id"}


class AcademicConfigParser:
    """Config parser for the academic benchmark suite.

    Extends the validation logic of
    :class:`~benchmark.tools.config_parser.ConfigParser` to accept
    ``openml_task`` dataset entries that carry a ``task_id`` instead of a
    ``url`` / ``file`` pair.

    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file.

    Raises
    ------
    Exception
        If the file cannot be read or does not conform to the expected schema.

    Notes
    -----
    Two dataset entry shapes are accepted:

    * Standard CSV/Excel entries — must have ``{"name", "url", "function",
      "file"}``.
    * OpenML task entries — must have ``{"name", "function", "task_id"}``
      with ``"function"`` equal to ``"openml_task"``.
    """

    def __init__(self, config_file: str = "config_academic.json") -> None:
        try:
            with open(config_file) as f:
                config = json.load(f)
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
    def config(self) -> dict:
        """Parsed configuration dictionary."""
        return self._config

    def _validate(self, config: dict) -> None:
        assert {"Description", "Experiments", "Datasets"} <= config.keys()
        assert "regression" in config or "classification" in config

        for models_key in ("regression", "classification"):
            if models_key in config:
                assert (
                    isinstance(config[models_key], dict)
                    and len(config[models_key]) > 0
                )
                for model_cfg in config[models_key].values():
                    assert model_cfg.keys() == _MODEL_KEYS
                    assert isinstance(model_cfg["module"], str)
                    assert isinstance(model_cfg["object"], str)
                    assert isinstance(model_cfg["parameters"], dict)

        assert (
            isinstance(config["Experiments"], list)
            and len(config["Experiments"]) > 0
        )
        for exp in config["Experiments"]:
            assert _EXPERIMENT_KEYS <= exp.keys()
            assert isinstance(exp["module"], str)
            assert isinstance(exp["object"], str)
            assert isinstance(exp["parameters"], dict)

        assert (
            isinstance(config["Datasets"], list) and len(config["Datasets"]) > 0
        )
        for dataset in config["Datasets"]:
            assert _DATASET_REQUIRED_KEYS <= dataset.keys()
            assert isinstance(dataset["name"], str)
            assert isinstance(dataset["function"], str)

            if dataset["function"] == "openml_task":
                assert "task_id" in dataset, (
                    f"Dataset '{dataset['name']}' with function 'openml_task' "
                    "must have a 'task_id' field."
                )
                assert isinstance(dataset["task_id"], int)
            else:
                assert _CSV_DATASET_KEYS <= dataset.keys(), (
                    f"Dataset '{dataset['name']}' is missing required keys for "
                    f"function '{dataset['function']}'."
                )
                assert isinstance(dataset["url"], str)
                assert isinstance(dataset["file"], str)

            assert dataset.get("task", "regression") in (
                "regression",
                "classification",
            )
