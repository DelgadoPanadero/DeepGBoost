import json
import os
import abc

import numpy as np
from matplotlib import pyplot as plt


BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AbstractModelTest:
    @abc.abstractmethod
    def create_batch(self, X, y):
        raise NotImplementedError()

    @abc.abstractmethod
    def run(self, name, X, y):
        raise NotImplementedError()

    @abc.abstractmethod
    def score(self, y_test, y_pred):
        raise NotImplementedError()

    @property
    def metric_name(self) -> str:
        return "R2"

    def _save_summary(self, name, scores_dict):
        model_names = list(scores_dict.keys())
        reference = model_names[-1]
        ref_scores = scores_dict[reference]

        lines = [
            f"{n} Score: {s.mean():.4f} +- {s.std():.4f}"
            for n, s in scores_dict.items()
        ]
        for model_name, scores in scores_dict.items():
            if model_name != reference:
                diff = ref_scores - scores
                lines.append(
                    f"Difference ({reference} - {model_name}): {diff.mean():.4f} +- {diff.std():.4f}"
                )

        file_name = (
            f"{BENCHMARK_DIR}/results/{name.replace(' ', '_').lower()}.txt"
        )
        with open(file_name, "w") as file:
            file.write("\n".join(lines))

    def _save_scores(self, name, scores_dict):
        model_names = list(scores_dict.keys())
        reference = model_names[-1]
        score_arrays = list(scores_dict.values())

        file_name = f"{BENCHMARK_DIR}/results/{name.replace(' ', '_').lower()}_scores.jsonl"
        with open(file_name, "w") as file:
            for values in zip(*score_arrays):
                record = {n: float(v) for n, v in zip(model_names, values)}
                for n, v in zip(model_names[:-1], values[:-1]):
                    record[f"diff_{reference}_vs_{n}"] = float(values[-1]) - float(v)
                file.write(json.dumps(record) + "\n")

    def _plot_scores(self, name, scores_dict, n_bins):
        all_scores = np.concatenate(list(scores_dict.values()))
        bins = np.linspace(all_scores.min(), all_scores.max(), n_bins)

        for model_name, scores in scores_dict.items():
            plt.hist(scores, bins, alpha=0.5, label=model_name)

        plt.title(f"{name} {self.metric_name} Score")
        plt.xlabel(f"{self.metric_name} Score")
        plt.ylabel("Counts")
        plt.legend(loc="upper right")
        file_name = f"{BENCHMARK_DIR}/results/{name.replace(' ', '_').lower()}_score.png"
        plt.savefig(file_name)
        plt.clf()

    def _plot_scores_diff(self, name, scores_dict, n_bins):
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
        file_name = f"{BENCHMARK_DIR}/results/{name.replace(' ', '_').lower()}_score_diff.png"
        plt.savefig(file_name)
        plt.clf()
