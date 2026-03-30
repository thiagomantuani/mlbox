"""
mlbox.ensemble
--------------
Weighted ensemble that learns optimal per-model weights by minimising
(or maximising) a user-supplied metric on OOF predictions.

Supports regression and binary classification.

Architecture
------------
WeightedEnsembleRegressor   — minimises metric (e.g. RMSE, MAE)
WeightedEnsembleClassifier  — maximises metric (e.g. AUC, F1)

Both inherit from _BaseWeightedEnsemble which holds all logic.
"""

from __future__ import annotations

import warnings
from typing import Callable, Literal

import numpy as np
from scipy.optimize import minimize  # type: ignore[import]

from mlbox.base import BaseEnsemble
from mlbox.utils.dataframe import to_numpy, validate_X_y, ArrayLike


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class _BaseWeightedEnsemble(BaseEnsemble):
    """
    Internal base for weighted ensembles.

    Parameters
    ----------
    objective : {"minimize", "maximize"}
        Whether the metric should be minimised (regression) or maximised
        (classification).
    metric : callable
        Scoring function ``metric(y_true, y_pred) -> float``.
    method : {"nelder-mead", "slsqp"}
        Scipy optimisation method.  "slsqp" enforces the constraint that
        weights sum to 1.
    verbose : bool
    """

    def __init__(
        self,
        objective: Literal["minimize", "maximize"],
        metric: Callable[[np.ndarray, np.ndarray], float],
        method: str = "nelder-mead",
        verbose: bool = True,
    ) -> None:
        self.objective = objective
        self.metric    = metric
        self.method    = method.lower()
        self.verbose   = verbose

        self._weights: np.ndarray | None  = None
        self._oof_matrix: np.ndarray | None = None  # (n_samples, n_models)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._weights  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "_BaseWeightedEnsemble":
        """
        Learn optimal weights from OOF predictions.

        Parameters
        ----------
        X : 2-D array-like of shape (n_samples, n_models)
            Each column is the OOF predictions of one model.
        y : 1-D array-like of shape (n_samples,)
            Ground-truth labels / values.

        Returns
        -------
        self
        """
        validate_X_y(X, y)
        X_np = to_numpy(X).astype(np.float64)
        y_np = to_numpy(y).astype(np.float64)

        if X_np.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples × n_models), got shape {X_np.shape}"
            )

        n_models = X_np.shape[1]
        self._oof_matrix = X_np

        # --- Objective function for scipy ---
        sign = -1.0 if self.objective == "maximize" else 1.0

        def _objective(weights: np.ndarray) -> float:
            w = self._softmax(weights)  # ensure weights in (0,1) and sum to 1
            blended = (X_np * w).sum(axis=1)
            return sign * self.metric(y_np, blended)

        x0 = np.ones(n_models) / n_models
        result = minimize(
            _objective,
            x0,
            method="Nelder-Mead" if self.method == "nelder-mead" else "SLSQP",
            options={"maxiter": 10_000, "xatol": 1e-8, "fatol": 1e-8},
        )

        self._weights = self._softmax(result.x)

        if self.verbose:
            self._print_weights(self._weights, result.fun * sign)

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Blend predictions from multiple models using learned weights.

        Parameters
        ----------
        X : 2-D array-like (n_samples, n_models)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        self._check_is_fitted()
        X_np = to_numpy(X).astype(np.float64)
        if X_np.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples × n_models), got shape {X_np.shape}"
            )
        return (X_np * self._weights).sum(axis=1)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Stable softmax — maps unconstrained weights into a simplex."""
        e = np.exp(x - x.max())
        return e / e.sum()

    def _print_weights(self, weights: np.ndarray, score: float) -> None:
        metric_name = (
            self.metric.__name__ if hasattr(self.metric, "__name__") else "metric"
        )
        print("\n┌─ WeightedEnsemble ─────────────────────────────┐")
        for i, w in enumerate(weights, start=1):
            bar = "█" * int(w * 40)
            print(f"│  Model {i:>2}  {w:.4f}  {bar}")
        print(f"│  {metric_name}: {score:.6f}")
        print("└────────────────────────────────────────────────┘\n")

    def _check_is_fitted(self) -> None:
        if self._weights is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted yet. Call .fit() first."
            )


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

class WeightedEnsembleRegressor(_BaseWeightedEnsemble):
    """
    Weighted ensemble for regression tasks (minimises metric).

    Example
    -------
    ::

        from sklearn.metrics import root_mean_squared_error
        from mlbox import WeightedEnsembleRegressor

        # oof_preds: (n_train, n_models) — OOF predictions from each model
        ensemble = WeightedEnsembleRegressor(metric=root_mean_squared_error)
        ensemble.fit(oof_preds, y_train)

        # test_preds: (n_test, n_models)
        final_preds = ensemble.predict(test_preds)
        print(ensemble.weights_)
    """

    def __init__(
        self,
        metric: Callable[[np.ndarray, np.ndarray], float],
        method: str = "nelder-mead",
        verbose: bool = True,
    ) -> None:
        super().__init__(
            objective="minimize",
            metric=metric,
            method=method,
            verbose=verbose,
        )


class WeightedEnsembleClassifier(_BaseWeightedEnsemble):
    """
    Weighted ensemble for classification tasks (maximises metric).

    Example
    -------
    ::

        from sklearn.metrics import roc_auc_score
        from mlbox import WeightedEnsembleClassifier

        ensemble = WeightedEnsembleClassifier(metric=roc_auc_score)
        ensemble.fit(oof_preds, y_train)
        final_preds = ensemble.predict(test_preds)
    """

    def __init__(
        self,
        metric: Callable[[np.ndarray, np.ndarray], float],
        method: str = "nelder-mead",
        verbose: bool = True,
    ) -> None:
        super().__init__(
            objective="maximize",
            metric=metric,
            method=method,
            verbose=verbose,
        )
