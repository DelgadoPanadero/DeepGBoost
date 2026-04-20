"""Unified benchmark entry point.

Usage::

    # Dev suite (8 UCI datasets, fast iteration)
    .venv/bin/python -m benchmark.run --suite dev

    # Academic suite (18 OpenML datasets, Grinsztajn et al. NeurIPS 2022)
    .venv/bin/python -m benchmark.run --suite academic
"""

import argparse
import os

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))


def run_dev():
    from benchmark.tools.config_parser import ConfigParser
    from benchmark.tools.experiment_runner import ExperimentRunner
    from benchmark.tools.benchmark_generator import BenchmarkGenerator

    config = ConfigParser(os.path.join(BENCHMARK_DIR, "config.json")).config
    runner = ExperimentRunner(config)
    runner.run()
    runner.run_ablations()
    BenchmarkGenerator().save(output_path=os.path.join(BENCHMARK_DIR, "BENCHMARK.md"))


def run_academic():
    from benchmark.tools.academic_config_parser import AcademicConfigParser
    from benchmark.tools.academic_experiment_runner import AcademicExperimentRunner
    from benchmark.tools.benchmark_generator import BenchmarkGenerator

    results_dir = os.path.join(BENCHMARK_DIR, "results")
    config = AcademicConfigParser(os.path.join(BENCHMARK_DIR, "config_academic.json")).config
    runner = AcademicExperimentRunner(config, results_dir=results_dir)
    runner.run()
    BenchmarkGenerator(results_dir=results_dir).save(
        output_path=os.path.join(BENCHMARK_DIR, "BENCHMARK.md")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepGBoost benchmark runner")
    parser.add_argument(
        "--suite",
        choices=["dev", "academic"],
        required=True,
        help="'dev' for the 8 UCI datasets, 'academic' for the 18 OpenML datasets (Grinsztajn et al.)",
    )
    args = parser.parse_args()

    if args.suite == "dev":
        run_dev()
    else:
        run_academic()
