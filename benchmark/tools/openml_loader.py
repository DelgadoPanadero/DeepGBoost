import os

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENML_CACHE_DIR = os.path.join(BENCHMARK_DIR, "data", "openml")


class OpenMLLoader:
    """Load and cache OpenML tasks as NumPy arrays.

    Downloads the dataset associated with an OpenML task on first access and
    caches it to disk as a compressed ``.npz`` archive.  Subsequent calls with
    the same ``task_id`` read from the cache without hitting the network.

    Feature preprocessing mirrors the pipeline used by
    :class:`~benchmark.tools.experiment_runner.ExperimentRunner`:

    * Numeric columns are imputed with the column *median*.
    * Categorical columns are one-hot encoded (sparse output densified).

    For classification tasks the target is label-encoded to contiguous
    integers starting at 0.

    Parameters
    ----------
    cache_dir : str, optional
        Directory where ``.npz`` cache files are stored.
        Defaults to ``benchmark/data/openml/``.

    Examples
    --------
    >>> loader = OpenMLLoader()
    >>> X, y, task_type = loader.load(361072)
    >>> task_type
    'regression'
    """

    def __init__(self, cache_dir: str = OPENML_CACHE_DIR) -> None:
        self._cache_dir = cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, task_id: int) -> tuple[np.ndarray, np.ndarray, str]:
        """Return preprocessed ``(X, y, task_type)`` for an OpenML task.

        Parameters
        ----------
        task_id : int
            Numeric OpenML task identifier.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix after imputation and encoding.
        y : np.ndarray of shape (n_samples,)
            Target array.  Label-encoded to integers for classification tasks.
        task_type : str
            Either ``"regression"`` or ``"classification"``.

        Raises
        ------
        ValueError
            If the task type is not supervised regression or classification.
        """
        cache_path = self._cache_path(task_id)
        if os.path.exists(cache_path):
            return self._load_cache(cache_path)

        X_raw, y_raw, task_type = self._fetch(task_id)
        X, y = self._preprocess(X_raw, y_raw, task_type)
        self._save_cache(cache_path, X, y, task_type)
        return X, y, task_type

    def load_with_splits(
        self, task_id: int
    ) -> tuple[np.ndarray, np.ndarray, str, list[tuple[np.ndarray, np.ndarray]]]:
        """Return preprocessed data together with the task's predefined CV splits.

        Downloads (or reads from cache) the dataset and the OpenML-defined
        train/test index pairs for every repeat × fold combination.  If an
        existing cache file does not contain split information it is
        regenerated automatically.

        Parameters
        ----------
        task_id : int
            Numeric OpenML task identifier.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix after imputation and encoding.
        y : np.ndarray of shape (n_samples,)
            Target array.  Label-encoded to integers for classification tasks.
        task_type : str
            Either ``"regression"`` or ``"classification"``.
        splits : list of (train_idx, test_idx) tuples
            Each element is a pair of integer index arrays corresponding to one
            repeat/fold combination as defined by the OpenML task.  Indices
            refer to rows of ``X`` and ``y`` after preprocessing (row order is
            preserved by preprocessing).

        Raises
        ------
        ValueError
            If the task type is not supervised regression or classification.
        """
        cache_path = self._cache_path(task_id)

        if os.path.exists(cache_path):
            cached = np.load(cache_path, allow_pickle=False)
            if "n_splits" in cached:
                return self._load_cache_with_splits(cache_path)
            # Stale cache without split information — fall through to re-download.

        X_raw, y_raw, task_type, splits = self._fetch_with_splits(task_id)
        X, y = self._preprocess(X_raw, y_raw, task_type)
        self._save_cache_with_splits(cache_path, X, y, task_type, splits)
        return X, y, task_type, splits

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_path(self, task_id: int) -> str:
        return os.path.join(self._cache_dir, f"task_{task_id}.npz")

    def _load_cache(
        self, cache_path: str
    ) -> tuple[np.ndarray, np.ndarray, str]:
        data = np.load(cache_path, allow_pickle=False)
        task_type: str = str(data["task_type"])
        return data["X"], data["y"], task_type

    def _load_cache_with_splits(
        self, cache_path: str
    ) -> tuple[np.ndarray, np.ndarray, str, list[tuple[np.ndarray, np.ndarray]]]:
        """Load a cache file that includes predefined split indices.

        Parameters
        ----------
        cache_path : str
            Absolute path to the ``.npz`` cache file.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target array.
        task_type : str
            ``"regression"`` or ``"classification"``.
        splits : list of (train_idx, test_idx) tuples
            Predefined CV index pairs.
        """
        data = np.load(cache_path, allow_pickle=False)
        task_type: str = str(data["task_type"])
        n_splits = int(data["n_splits"])
        splits = [(data[f"train_{i}"], data[f"test_{i}"]) for i in range(n_splits)]
        return data["X"], data["y"], task_type, splits

    def _save_cache(
        self,
        cache_path: str,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
    ) -> None:
        np.savez_compressed(cache_path, X=X, y=y, task_type=np.str_(task_type))

    def _save_cache_with_splits(
        self,
        cache_path: str,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Persist the preprocessed dataset together with predefined split indices.

        Parameters
        ----------
        cache_path : str
            Absolute path where the ``.npz`` archive will be written.
        X : np.ndarray of shape (n_samples, n_features)
            Preprocessed feature matrix.
        y : np.ndarray of shape (n_samples,)
            Preprocessed target array.
        task_type : str
            ``"regression"`` or ``"classification"``.
        splits : list of (train_idx, test_idx) tuples
            Each element is a pair of integer index arrays.  Folds may have
            different sizes, so each split is stored as a separate named array
            ``train_i`` / ``test_i`` rather than a 2-D rectangular array.
        """
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            task_type=np.str_(task_type),
            n_splits=np.int64(len(splits)),
            **{f"train_{i}": s[0] for i, s in enumerate(splits)},
            **{f"test_{i}": s[1] for i, s in enumerate(splits)},
        )

    def _fetch(
        self, task_id: int
    ) -> tuple[object, object, str]:
        """Download the task from OpenML and return raw (X, y, task_type).

        Parameters
        ----------
        task_id : int
            OpenML task identifier.

        Returns
        -------
        X : array-like
            Raw feature data as returned by ``task.get_X_and_y()``.
        y : array-like
            Raw target as returned by ``task.get_X_and_y()``.
        task_type : str
            ``"regression"`` or ``"classification"``.
        """
        import openml
        from openml.tasks import TaskType

        task = openml.tasks.get_task(task_id)

        if task.task_type_id == TaskType.SUPERVISED_CLASSIFICATION:
            task_type = "classification"
        elif task.task_type_id == TaskType.SUPERVISED_REGRESSION:
            task_type = "regression"
        else:
            raise ValueError(
                f"Unsupported task type '{task.task_type_id}' for task {task_id}. "
                "Only supervised classification and regression are supported."
            )

        X, y = task.get_X_and_y(dataset_format="dataframe")
        return X, y, task_type

    def _fetch_with_splits(
        self, task_id: int
    ) -> tuple[object, object, str, list[tuple[np.ndarray, np.ndarray]]]:
        """Download the task from OpenML and return raw data with predefined splits.

        Parameters
        ----------
        task_id : int
            OpenML task identifier.

        Returns
        -------
        X : array-like
            Raw feature data as returned by ``task.get_X_and_y()``.
        y : array-like
            Raw target as returned by ``task.get_X_and_y()``.
        task_type : str
            ``"regression"`` or ``"classification"``.
        splits : list of (train_idx, test_idx) tuples
            One entry per repeat × fold combination.  Each element is a pair of
            integer index arrays as returned by
            ``task.get_train_test_split_indices()``.
        """
        import openml
        from openml.tasks import TaskType

        task = openml.tasks.get_task(task_id)

        if task.task_type_id == TaskType.SUPERVISED_CLASSIFICATION:
            task_type = "classification"
        elif task.task_type_id == TaskType.SUPERVISED_REGRESSION:
            task_type = "regression"
        else:
            raise ValueError(
                f"Unsupported task type '{task.task_type_id}' for task {task_id}. "
                "Only supervised classification and regression are supported."
            )

        X, y = task.get_X_and_y(dataset_format="dataframe")

        n_repeats, n_folds, _ = task.get_split_dimensions()
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for repeat in range(n_repeats):
            for fold in range(n_folds):
                train_idx, test_idx = task.get_train_test_split_indices(
                    fold=fold, repeat=repeat
                )
                splits.append((train_idx, test_idx))

        return X, y, task_type, splits

    def _preprocess(
        self,
        X_raw: object,
        y_raw: object,
        task_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Impute, encode, and densify raw feature and target arrays.

        Parameters
        ----------
        X_raw : array-like or pd.DataFrame
            Raw feature data, possibly containing NaN and categorical columns.
        y_raw : array-like or pd.Series
            Raw target data.
        task_type : str
            ``"regression"`` or ``"classification"``.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features_out)
            Preprocessed feature matrix.
        y : np.ndarray of shape (n_samples,)
            Preprocessed target array.
        """
        import pandas as pd

        X_df = pd.DataFrame(X_raw) if not isinstance(X_raw, pd.DataFrame) else X_raw.copy()
        y_arr = np.asarray(y_raw).ravel()

        # --- Feature preprocessing ---
        cat_cols = X_df.select_dtypes(exclude=["number"]).columns.tolist()
        num_cols = X_df.select_dtypes(include=["number"]).columns.tolist()

        parts: list[np.ndarray] = []

        if num_cols:
            X_num = X_df[num_cols].values.astype(float)
            imputer = SimpleImputer(strategy="median")
            parts.append(imputer.fit_transform(X_num))

        if cat_cols:
            X_cat = (
                X_df[cat_cols]
                .astype(str)
                .apply(lambda col: col.str.strip())
            )
            # Impute NaN-derived "nan" strings with the most frequent category
            for col in X_cat.columns:
                mask = X_cat[col] == "nan"
                if mask.any():
                    mode_val = X_cat.loc[~mask, col].mode()
                    fill = mode_val.iloc[0] if len(mode_val) else "unknown"
                    X_cat.loc[mask, col] = fill

            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            parts.append(encoder.fit_transform(X_cat))

        if not parts:
            X = np.empty((len(y_arr), 0), dtype=float)
        elif len(parts) == 1:
            X = parts[0]
        else:
            X = np.hstack(parts)

        # --- Target preprocessing ---
        if task_type == "classification":
            le = LabelEncoder()
            y_arr = le.fit_transform(y_arr.astype(str))
        else:
            y_arr = y_arr.astype(float)

        return X.astype(float), y_arr
