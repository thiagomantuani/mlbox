"""
mlbox.utils.pipeline
--------------------
Helpers to handle sklearn Pipelines in a leakage-safe manner inside
cross-validation loops.

The Golden Rule (no data leakage)
----------------------------------
For each fold:
  1. ``pipeline.fit(X_train_fold, y_train_fold)``        ← fit on TRAIN only
  2. ``pipeline.predict / transform(X_val_fold)``        ← apply to VAL
  3. ``pipeline.predict / transform(X_test)``            ← apply to TEST

Fitting the whole pipeline (including preprocessing) on the full dataset
BEFORE the CV split is a data leakage bug. This module makes step 1
explicit and centralised so no caller can accidentally get it wrong.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from mlbox.utils.dataframe import to_numpy, iloc_rows, ArrayLike


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_pipeline(estimator: Any) -> bool:
    """Return True if *estimator* is a sklearn Pipeline."""
    try:
        from sklearn.pipeline import Pipeline
        return isinstance(estimator, Pipeline)
    except ImportError:
        return False


def get_last_step_name(estimator: Any) -> Optional[str]:
    """
    Return the name of the last step in a sklearn Pipeline, or None.

    Used to correctly prefix fit_params for early stopping.
    """
    if is_pipeline(estimator):
        return estimator.steps[-1][0]
    return None


def pipeline_fit(
    estimator: Any,
    X_train: ArrayLike,
    y_train: ArrayLike,
    fit_params: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Fit *estimator* on (X_train, y_train) with optional extra fit params.

    When *estimator* is a Pipeline and early stopping is enabled,
    fit_params must already be prefixed (e.g. ``"xgbclassifier__eval_set"``).
    The Pipeline routes them to the correct step automatically.

    Returns
    -------
    The fitted estimator (same object, mutated in place).
    """
    fit_params = fit_params or {}
    estimator.fit(X_train, y_train, **fit_params)
    return estimator


def pipeline_predict(
    estimator: Any,
    X: ArrayLike,
    task: str,
) -> np.ndarray:
    """
    Generate predictions using the correct method for the task type.

    Parameters
    ----------
    estimator : fitted sklearn estimator or Pipeline
    X         : features
    task      : one of ``"binary"``, ``"multiclass"``, ``"regression"``

    Returns
    -------
    np.ndarray
        1-D array for binary/regression; 2-D array for multiclass.
    """
    if task == "regression":
        return estimator.predict(X)

    if task == "binary":
        proba = estimator.predict_proba(X)
        # Always return P(class=1) for binary
        return proba[:, 1]

    if task == "multiclass":
        return estimator.predict_proba(X)

    raise ValueError(
        f"Unknown task {task!r}. Expected 'binary', 'multiclass', or 'regression'."
    )
