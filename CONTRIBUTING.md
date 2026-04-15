# Contributing to DeepGBoost

Thanks for your interest. This document covers everything you need to go from zero to an open pull request.

For significant changes — new algorithms, API modifications, or benchmark methodology — please **open an issue first** so the approach can be discussed before implementation.

---

## 1. Development setup

```bash
git clone https://github.com/PanaderoDelgado/DeepGBoost.git
cd DeepGBoost
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pre-commit install
```

The `[dev]` extra installs pytest, ruff, nbmake, pre-commit, xgboost, tqdm, and matplotlib. The editable install means changes to `src/` are reflected immediately without reinstalling.

---

## 2. Running tests

**Fast unit tests** (134 tests, completes in seconds):

```bash
.venv/bin/pytest tests/
```

**With coverage report:**

```bash
.venv/bin/pytest tests/ --cov=deepgboost --cov-report=term-missing
```

**Notebook tests** (executes every notebook end-to-end):

```bash
.venv/bin/pytest --nbmake examples/ benchmark/
```

Notebook tests are slower — run them before opening a PR if you touched any notebook or the public API. CI runs both suites on every push.

> **Tree budget**: any test or example you add must keep `n_layers × n_trees ≤ 100` to stay fast in CI.

---

## 3. Code style

The project uses [ruff](https://docs.astral.sh/ruff/) with a single active rule set: `COM812` (trailing commas). `ruff-format` enforces 80-character line width.

The pre-commit hook runs both `ruff --fix` and `ruff-format` automatically on every `git commit`. You can also run them manually:

```bash
ruff check --fix src/ tests/
ruff format src/ tests/
```

No other linting conventions are enforced programmatically. Style expectations for new code:

- Concise — no unnecessary inline comments.
- No defensive error handling in internal (non-public) code.
- Follow sklearn naming and interface conventions for anything user-facing.

---

## 4. Project structure

```
src/deepgboost/
    __init__.py                  # Public API — all exports listed here
    deepgboost_regressor.py      # DeepGBoostRegressor (sklearn estimator)
    deepgboost_classifier.py     # DeepGBoostClassifier
    deepgboost_multiclassifier.py# DeepGBoostMultiClassifier
    dgbf/                        # Core DGBF model (DGBFModel, DGBFMultiOutputModel)
    objectives/                  # Loss functions (get_objective)
    tree/                        # Decision tree utilities
    callbacks/                   # TrainingCallback and built-in callbacks
    metric/                      # Evaluation metrics (get_metric)
    linear/                      # Linear algebra helpers (NNLS, condition number)
    predictor/                   # Prediction aggregation
    common/                      # Shared utilities
    plotting.py                  # plot_importance

tests/
    test_regressor.py
    test_classifier.py
    test_multiclassifier.py
    test_dgbf.py
    test_callback.py
    test_adaptive_layer_width.py
    test_metric.py
    test_objective.py

benchmark/
    config.json                  # Models, datasets, and experiment settings
    run_experiments.py           # Entry point — run from benchmark/
    models/                      # Wrapper classes for each model in the benchmark
    experiments/                 # Experiment types (e.g. CrossValidationModelTest)
    data/                        # Downloaded datasets (git-ignored)
    results/                     # Output CSVs and figures (git-ignored)

examples/
    quickstart.ipynb
    regressor.ipynb
    classifier.ipynb
    serialization.ipynb
```

Do NOT modify `dgbf/` unless you are intentionally changing the algorithm. If you are, open an issue first.

---

## 5. Adding a new estimator

Follow this pattern when adding a public estimator (e.g. `DeepGBoostQuantileRegressor`):

1. **New file**: `src/deepgboost/deepgboost_<name>.py`. Subclass `sklearn.base.BaseEstimator` and the appropriate mixin (`RegressorMixin`, `ClassifierMixin`). Implement at minimum `fit`, `predict`, `get_params`, `set_params`.

2. **Export**: add the class to `src/deepgboost/__init__.py` — both the import and the `__all__` list.

3. **Tests**: add `tests/test_<name>.py`. Cover `fit`/`predict`, `sklearn.clone()`, `Pipeline` integration, serialization via `pickle`, and any estimator-specific behaviour.

4. **Docstring**: the class docstring must have `Parameters`, `Attributes`, `Examples` sections. Public methods need at minimum a one-line summary and a `Parameters` / `Returns` block.

5. **Notebook** (if user-facing): add or update a notebook in `examples/` demonstrating realistic use. Keep tree budget within `n_layers × n_trees ≤ 100`.

6. **Benchmark wrapper** (optional): if you want the estimator to appear in benchmark comparisons, add a wrapper class in `benchmark/models/` following the pattern of `deepgboost_regressor_model.py`, then register it in `benchmark/config.json`.

---

## 6. Adding to the benchmark

The benchmark is configured entirely through `benchmark/config.json`. No code changes are required to add a dataset or swap a model.

**Add a dataset:**

```json
{
  "name": "MyDataset",
  "url": "https://example.com/data.csv",
  "function": "read_csv",
  "file": "data/mydataset.csv",
  "sep": ",",
  "target_column": "target",
  "task": "regression"
}
```

Add the object to the `"Datasets"` array. Supported `"task"` values: `"regression"`, `"classification"`.

**Add a model:**

1. Create a wrapper in `benchmark/models/` that inherits from `AbstractModel` (see existing wrappers for the interface).
2. Register it in `config.json` under the appropriate task key (`"regression"` or `"classification"`):

```json
"model_5": {
  "module": "benchmark.models",
  "object": "MyModel",
  "parameters": { "n_estimators": 100 }
}
```

**Run the benchmark** from the `benchmark/` directory:

```bash
cd benchmark
python run_experiments.py
```

Results are written to `benchmark/results/`. The `benchmark/data/` directory is git-ignored; datasets are downloaded on first run.

---

## 7. Pull request checklist

Before opening a PR, confirm:

- [ ] `pytest tests/` passes locally with no failures.
- [ ] `ruff check src/ tests/` reports no errors (pre-commit should handle this automatically).
- [ ] Any new public class or method has a docstring with `Parameters` and `Returns` sections.
- [ ] New or changed behaviour is covered by at least one test.
- [ ] If you touched a notebook, `pytest --nbmake examples/` passes.
- [ ] The PR description explains **why** the change is needed, not just what it does.
- [ ] Branch is up to date with `main` before requesting review.

Branch off `main`, keep commits focused, and use squash merge when landing. Commit messages follow the pattern `type(#issue): description` (e.g. `feat(#42): add quantile regression objective`).
