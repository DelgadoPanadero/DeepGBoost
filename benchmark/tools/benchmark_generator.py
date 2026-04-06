import glob
import json
import os
import re

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")


class BenchmarkGenerator:
    def __init__(self, results_dir: str = RESULTS_DIR):
        self._results_dir = results_dir

    def _dataset_name(self, path: str) -> str:
        stem = os.path.basename(path).replace(".json", "")
        name = re.sub(r"_bootstrap_test$", "", stem)
        return name.replace("_", " ").title()

    def _load_results(self) -> dict[str, dict[str, tuple[float, float]]]:
        pattern = os.path.join(self._results_dir, "*_bootstrap_test.json")
        data: dict[str, dict[str, tuple[float, float]]] = {}
        for path in sorted(glob.glob(pattern)):
            dataset = self._dataset_name(path)
            with open(path) as f:
                records = json.load(f)
            data[dataset] = {
                r["model"]: (r["mean"], r["std"])
                for r in records
                if "model" in r
            }
        return data

    def generate_table(self) -> str:
        data = self._load_results()
        if not data:
            return ""

        all_models: list[str] = []
        for scores in data.values():
            for m in scores:
                if m not in all_models:
                    all_models.append(m)

        datasets = list(data.keys())

        header = "| Model | " + " | ".join(datasets) + " |"
        separator = "| :--- | " + " | ".join([":---:"] * len(datasets)) + " |"

        rows = []
        for model in all_models:
            cells = []
            for dataset in datasets:
                if model not in data[dataset]:
                    cells.append("-")
                    continue
                mean, std = data[dataset][model]
                best = max(data[dataset], key=lambda m, d=dataset: data[d][m][0])
                cell = f"{mean:.4f} ± {std:.4f}"
                if model == best:
                    cell = f"**{cell}**"
                cells.append(cell)
            rows.append("| " + model + " | " + " | ".join(cells) + " |")

        return "\n".join([header, separator] + rows)

    def save(self, output_path: str | None = None) -> str:
        if output_path is None:
            output_path = os.path.join(self._results_dir, "benchmark_summary.md")
        with open(output_path, "w") as f:
            f.write("# Benchmark Summary\n\n")
            f.write(self.generate_table())
            f.write("\n")
        return output_path
