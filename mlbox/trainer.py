"""
mlbox.trainer
-------------
Cross-validated Trainer with first-class support for:

  - sklearn estimators (any BaseEstimator)
  - sklearn Pipelines (make_pipeline / Pipeline) — leakage-safe
  - XGBoost / LightGBM / CatBoost with configurable early stopping
  - pandas AND polars DataFrames / Series
  - binary, multiclass, and regression tasks
  - rich per-fold logging with tqdm

Data Leakage guarantee
-----------------------
For every fold i in the CV:
  • The ENTIRE pipeline (preprocessing + model) is cloned and fitted
    exclusively on the train split of fold i.
  • Validation split and test set are only passed through .predict /
    .predict_proba — never seen during .fit.
  • Early-stopping eval_set is always the FOLD's validation split.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Callable, Optional, Union

import numpy as np

from mlbox.base import BaseTrainer
from mlbox.utils.dataframe import (
    to_numpy,
    iloc_rows,
    validate_X_y,
    ArrayLike,
)
from mlbox.utils.early_stopping import EarlyStoppingConfig, build_fit_params
from mlbox.utils.pipeline import (
    get_last_step_name,
    is_pipeline,
    pipeline_fit,
    pipeline_predict,
)
from mlbox.utils.logging import FoldLogger, fold_range, timer


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer(BaseTrainer):
    """
    Cross-validated trainer for sklearn estimators and Pipelines.

    Parameters
    ----------
    estimator : sklearn estimator or Pipeline
        Any object with a ``.fit`` / ``.predict`` (or ``.predict_proba``)
        interface. Can be a ``make_pipeline(...)`` or ``Pipeline(...)``
        object — preprocessing steps will be fitted per-fold.
    cv : cross-validator
        Any sklearn-compatible splitter (``KFold``, ``StratifiedKFold``,
        ``GroupKFold``, etc.).
    metric : callable
        A scoring function with signature ``metric(y_true, y_pred) -> float``.
    task : {"binary", "multiclass", "regression"}
        Determines which prediction method is called.
    early_stopping : EarlyStoppingConfig, optional
        Early stopping configuration for XGBoost / LightGBM / CatBoost.
        Set to ``None`` to disable (default).
    verbose : bool
        Whether to print fold-level progress and score table.
    n_jobs : int
        Not yet implemented (placeholder for future parallel fold execution).

    Examples
    --------
    Basic usage with sklearn estimator::

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        trainer = Trainer(
            estimator=RandomForestClassifier(n_estimators=100),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            metric=roc_auc_score,
            task="binary",
        )
        trainer.fit(X_train, y_train)
        preds = trainer.predict(X_test)

    With a sklearn Pipeline (leakage-safe preprocessing)::

        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb

        pipe = make_pipeline(StandardScaler(), lgb.LGBMClassifier())
        trainer = Trainer(
            estimator=pipe,
            cv=StratifiedKFold(5),
            metric=roc_auc_score,
            task="binary",
            early_stopping=EarlyStoppingConfig(rounds=50, verbose=False),
        )
        trainer.fit(X_train, y_train)

    With polars DataFrame::

        import polars as pl
        X = pl.DataFrame(...)
        y = pl.Series(...)
        trainer.fit(X, y)
    """

    def __init__(
        self,
        estimator: Any,
        cv: Any,
        metric: Callable[[np.ndarray, np.ndarray], float],
        task: str = "binary",
        early_stopping: Optional[EarlyStoppingConfig] = None,
        verbose: bool = True,
        n_jobs: int = 1,
    ) -> None:
        self.estimator     = estimator
        self.cv            = cv
        self.metric        = metric
        self.task          = task
        self.early_stopping = early_stopping
        self.verbose       = verbose
        self.n_jobs        = n_jobs

        # --- Fitted state (private, exposed via properties) ---
        self._oof_preds:    Optional[np.ndarray] = None
        self._fold_scores:  list[float]          = []
        self._overall_score: Optional[float]     = None
        self._models:        list[Any]           = []

        self._validate_task()

    # ------------------------------------------------------------------
    # Properties (post-fit)
    # ------------------------------------------------------------------

    @property
    def oof_preds_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._oof_preds  # type: ignore[return-value]

    @property
    def fold_scores_(self) -> list[float]:
        self._check_is_fitted()
        return self._fold_scores

    @property
    def overall_score_(self) -> float:
        self._check_is_fitted()
        return self._overall_score  # type: ignore[return-value]

    @property
    def models_(self) -> list[Any]:
        self._check_is_fitted()
        return self._models

    # ------------------------------------------------------------------
    # Public fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: Optional[np.ndarray] = None,
    ) -> "Trainer":
        """
        Fit the estimator using cross-validation.

        Parameters
        ----------
        X      : pandas/polars DataFrame or numpy array — features
        y      : pandas/polars Series or numpy array — target
        groups : array-like, optional — group labels for GroupKFold etc.

        Returns
        -------
        self
        """
        validate_X_y(X, y)

        y_np    = to_numpy(y)
        n       = len(y_np)
        n_folds = self.cv.n_splits if hasattr(self.cv, "n_splits") else "?"

        # Allocate OOF array
        if self.task == "multiclass":
            n_classes = len(np.unique(y_np))
            self._oof_preds = np.zeros((n, n_classes), dtype=np.float64)
        else:
            self._oof_preds = np.zeros(n, dtype=np.float64)

        self._fold_scores = []
        self._models      = []

        logger = FoldLogger(
            n_folds=n_folds if isinstance(n_folds, int) else 5,
            metric_name=self.metric.__name__ if hasattr(self.metric, "__name__") else "Score",
            verbose=self.verbose,
        )
        logger.print_header()

        for fold_idx, (train_idx, val_idx) in enumerate(
            self._split(X, y_np, groups), start=1
        ):
            with timer() as t:
                fold_model = self._fit_fold(X, y_np, train_idx, val_idx)

            X_val = iloc_rows(X, val_idx)
            val_preds = pipeline_predict(fold_model, X_val, self.task)
            self._oof_preds[val_idx] = val_preds

            fold_score = float(self.metric(y_np[val_idx], val_preds))
            self._fold_scores.append(fold_score)
            self._models.append(fold_model)

            logger.log_fold(fold_idx, fold_score, t["elapsed"])

        self._overall_score = float(self.metric(y_np, self._oof_preds))
        logger.print_footer()

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Generate test predictions by averaging across all fold models.

        Parameters
        ----------
        X : pandas/polars DataFrame or numpy array

        Returns
        -------
        np.ndarray
            Averaged predictions across folds.
        """
        self._check_is_fitted()

        all_preds = [
            pipeline_predict(model, X, self.task)
            for model in self._models
        ]
        return np.mean(all_preds, axis=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_fold(
        self,
        X: ArrayLike,
        y_np: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> Any:
        """Clone estimator, build fit_params, and fit on the train split."""
        X_train = iloc_rows(X, train_idx)
        y_train = y_np[train_idx]

        # Deep-clone so each fold gets a fresh model
        model = copy.deepcopy(self.estimator)

        fit_params: dict[str, Any] = {}

        if self.early_stopping is not None:
            # For early stopping we need X_val AFTER pipeline preprocessing.
            # When the estimator IS a Pipeline, we pass raw X_val and let the
            # Pipeline handle transformations internally — zero leakage.
            X_val_raw = iloc_rows(X, val_idx)
            y_val     = y_np[val_idx]

            if is_pipeline(model):
                # The Pipeline will preprocess X_val internally.
                # We need to fit the preprocessing steps on X_train first
                # to produce the eval_set. We clone the preprocessing part,
                # fit it on X_train, transform X_val, then pass to booster.
                X_val_transformed = self._transform_val_for_early_stopping(
                    model, X_train, y_train, X_val_raw
                )
                last_step_name = get_last_step_name(model)
                fit_params = build_fit_params(
                    model,
                    X_val_transformed,
                    y_val,
                    self.early_stopping,
                    pipeline_last_step_name=last_step_name,
                )
            else:
                fit_params = build_fit_params(
                    model,
                    to_numpy(X_val_raw),
                    y_val,
                    self.early_stopping,
                )

        pipeline_fit(model, X_train, y_train, fit_params)
        return model

    def _transform_val_for_early_stopping(
        self,
        pipeline: Any,
        X_train: ArrayLike,
        y_train: np.ndarray,
        X_val: ArrayLike,
    ) -> np.ndarray:
        """
        Fit the preprocessing portion of a Pipeline on X_train, then
        transform X_val.

        This is used ONLY to construct the eval_set for early stopping.
        The main pipeline.fit() call will redo this internally — that is
        intentional and unavoidable given the sklearn Pipeline API.

        No leakage: preprocessing is fitted on X_train only.
        """
        from sklearn.pipeline import Pipeline

        # Build a pipeline with all steps EXCEPT the last (the booster)
        steps = pipeline.steps[:-1]
        if not steps:
            # No preprocessing steps — X_val is already in the right space
            return to_numpy(X_val)

        pre_pipe = Pipeline(steps)
        pre_pipe.fit(X_train, y_train)
        return pre_pipe.transform(X_val)

    def _split(
        self,
        X: ArrayLike,
        y_np: np.ndarray,
        groups: Optional[np.ndarray],
    ):
        """Delegate to cv.split(), handling group-aware splitters."""
        try:
            import inspect
            sig = inspect.signature(self.cv.split)
            if "groups" in sig.parameters:
                return self.cv.split(X, y_np, groups=groups)
        except Exception:
            pass
        return self.cv.split(X, y_np)

    def _validate_task(self) -> None:
        valid = {"binary", "multiclass", "regression"}
        if self.task not in valid:
            raise ValueError(f"task must be one of {valid}, got {self.task!r}")

    def _check_is_fitted(self) -> None:
        if self._oof_preds is None:
            raise RuntimeError(
                "This Trainer instance has not been fitted yet. Call .fit() first."
            )
