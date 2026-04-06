import os
from .tools import ConfigParser
from .tools import ExperimentRunner
from .tools import BenchmarkGenerator


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config_parser = ConfigParser(config_path)
    config = config_parser.config

    #runner = ExperimentRunner(config)
    #runner.run()
    BenchmarkGenerator().save()
