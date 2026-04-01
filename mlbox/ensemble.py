"""
mlbox.ensemble
--------------
Weighted ensemble that learns optimal per-model weights (or a meta-model)
from OOF predictions — without data leakage.

Strategies available via the ``method`` parameter
--------------------------------------------------
"nelder-mead"   Unconstrained scipy optimisation (default).
                Weights mapped through softmax -> always sum to 1, all > 0.

"optuna"        Bayesian search over the weight simplex via Optuna.
                More thorough than Nelder-Mead for many models.
                Requires: pip install optuna

"ridge"         Fits a Ridge regression on OOF predictions (regression only).
                Weights can be negative (allows hedging).
                Uses cross-validated alpha selection via RidgeCV.

"logistic"      Fits a LogisticRegressionCV on OOF predictions (classification).
                One weight per model per class.
                Uses cross-validated C selection.

"meta"          Fits an arbitrary user-supplied sklearn estimator on OOF
                predictions as meta-features. Most flexible strategy.
                Supports both regression and classification.

Architecture
------------
WeightedEnsembleRegressor   -- minimises metric  (regression)
WeightedEnsembleClassifier  -- maximises metric  (classification)

Both share _BaseWeightedEnsemble which holds all strategy logic.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Callable, Literal, Optional

import numpy as np
from scipy.optimize import minimize

from mlbox.base import BaseEnsemble
from mlbox.utils.dataframe import to_numpy, validate_X_y, ArrayLike


Method = Literal["nelder-mead", "optuna", "ridge", "logistic", "meta"]


class _BaseWeightedEnsemble(BaseEnsemble):
    """
    Internal base for weighted ensembles.

    Parameters
    ----------
    objective : {"minimize", "maximize"}
    metric : callable  --  metric(y_true, y_pred) -> float
    method : Method    --  blending strategy (see module docstring)
    meta_estimator : sklearn estimator, optional
        Required when method="meta".
    n_trials : int
        Optuna trials (method="optuna" only). Default 200.
    verbose : bool
    """

    def __init__(
        self,
        objective: Literal["minimize", "maximize"],
        metric: Callable[[np.ndarray, np.ndarray], float],
        method: Method = "nelder-mead",
        meta_estimator: Optional[Any] = None,
        n_trials: int = 200,
        verbose: bool = True,
    ) -> None:
        self.objective      = objective
        self.metric         = metric
        self.method         = method.lower()
        self.meta_estimator = meta_estimator
        self.n_trials       = n_trials
        self.verbose        = verbose

        self._weights: Optional[np.ndarray] = None
        self._meta_model: Optional[Any]     = None
        self._n_models: int                 = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights_(self) -> np.ndarray:
        """Simplex weights (nelder-mead / optuna only)."""
        self._check_is_fitted()
        if self._weights is None:
            raise AttributeError(
                f"method='{self.method}' does not produce scalar weights. "
                "Use .meta_model_ to inspect the fitted meta-model."
            )
        return self._weights

    @property
    def meta_model_(self) -> Any:
        """Fitted meta-model (ridge / logistic / meta)."""
        self._check_is_fitted()
        if self._meta_model is None:
            raise AttributeError(
                f"method='{self.method}' does not use a meta-model. "
                "Use .weights_ instead."
            )
        return self._meta_model

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "_BaseWeightedEnsemble":
        """
        Learn blending weights / meta-model from OOF predictions.

        Parameters
        ----------
        X : 2-D array-like, shape (n_samples, n_models)
            Each column = OOF predictions from one model.
        y : 1-D array-like, shape (n_samples,)
            Ground-truth target.
        """
        validate_X_y(X, y)
        X_np = to_numpy(X).astype(np.float64)
        y_np = to_numpy(y).astype(np.float64)

        if X_np.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples x n_models), got shape {X_np.shape}"
            )

        self._n_models = X_np.shape[1]

        dispatch = {
            "nelder-mead": self._fit_nelder_mead,
            "optuna":      self._fit_optuna,
            "ridge":       self._fit_ridge,
            "logistic":    self._fit_logistic,
            "meta":        self._fit_meta,
        }
        if self.method not in dispatch:
            raise ValueError(
                f"Unknown method '{self.method}'. Choose from: {list(dispatch)}"
            )

        dispatch[self.method](X_np, y_np)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Generate blended predictions.

        Parameters
        ----------
        X : 2-D array-like, shape (n_samples, n_models)
            Each column = test predictions from one model (same order as fit).

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Regression: predicted values.
            Classification (logistic/meta): P(class=1) for binary,
            or predict_proba array for multiclass.
        """
        self._check_is_fitted()
        X_np = to_numpy(X).astype(np.float64)
        if X_np.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples x n_models), got shape {X_np.shape}"
            )

        if self.method in ("nelder-mead", "optuna"):
            return (X_np * self._weights).sum(axis=1)

        if self.method == "ridge":
            return self._meta_model.predict(X_np)

        if self.method == "logistic":
            proba = self._meta_model.predict_proba(X_np)
            return proba[:, 1] if proba.shape[1] == 2 else proba

        if self.method == "meta":
            if hasattr(self._meta_model, "predict_proba"):
                proba = self._meta_model.predict_proba(X_np)
                return proba[:, 1] if proba.shape[1] == 2 else proba
            return self._meta_model.predict(X_np)

        raise RuntimeError(f"Unhandled method '{self.method}' in predict()")

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _fit_nelder_mead(self, X_np: np.ndarray, y_np: np.ndarray) -> None:
        """Unconstrained Nelder-Mead with softmax projection."""
        sign = -1.0 if self.objective == "maximize" else 1.0

        def _obj(w):
            blended = (X_np * self._softmax(w)).sum(axis=1)
            return sign * self.metric(y_np, blended)

        result = minimize(
            _obj,
            np.ones(self._n_models) / self._n_models,
            method="Nelder-Mead",
            options={"maxiter": 10_000, "xatol": 1e-8, "fatol": 1e-8},
        )
        self._weights = self._softmax(result.x)
        if self.verbose:
            self._print_weights(self._weights, result.fun * sign, "Nelder-Mead")

    def _fit_optuna(self, X_np: np.ndarray, y_np: np.ndarray) -> None:
        """Bayesian optimisation over the weight simplex via Optuna."""
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "method='optuna' requires optuna. Install with: pip install optuna"
            ) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sign = -1.0 if self.objective == "maximize" else 1.0

        def _objective(trial):
            raw = np.array([
                trial.suggest_float(f"w{i}", 0.0, 1.0)
                for i in range(self._n_models)
            ])
            w = raw / raw.sum()
            return sign * self.metric(y_np, (X_np * w).sum(axis=1))

        study = optuna.create_study(direction="minimize")
        study.optimize(
            _objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
        )

        best_raw = np.array([study.best_params[f"w{i}"] for i in range(self._n_models)])
        self._weights = best_raw / best_raw.sum()

        if self.verbose:
            self._print_weights(
                self._weights,
                study.best_value * sign,
                f"Optuna ({self.n_trials} trials)",
            )

    def _fit_ridge(self, X_np: np.ndarray, y_np: np.ndarray) -> None:
        """RidgeCV meta-model. Regression only."""
        if self.objective == "maximize":
            raise ValueError(
                "method='ridge' is for regression. "
                "For classification use method='logistic' or method='meta'."
            )
        from sklearn.linear_model import RidgeCV

        model = RidgeCV(alphas=np.logspace(-4, 4, 100), fit_intercept=True)
        model.fit(X_np, y_np)
        self._meta_model = model

        if self.verbose:
            score = self.metric(y_np, model.predict(X_np))
            coefs = model.coef_
            denom = np.abs(coefs).sum()
            norm  = coefs / denom if denom > 0 else coefs
            self._print_meta(norm, score, f"Ridge (alpha={model.alpha_:.4f})")

    def _fit_logistic(self, X_np: np.ndarray, y_np: np.ndarray) -> None:
        """LogisticRegressionCV meta-model. Classification only."""
        if self.objective == "minimize":
            raise ValueError(
                "method='logistic' is for classification. "
                "For regression use method='ridge' or method='meta'."
            )
        from sklearn.linear_model import LogisticRegressionCV

        model = LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 20),
            cv=5,
            max_iter=1_000,
            fit_intercept=True,
            random_state=42,
            penalty="l2",
            solver="lbfgs",
        )
        model.fit(X_np, y_np.astype(int))
        self._meta_model = model

        if self.verbose:
            proba = model.predict_proba(X_np)
            preds = proba[:, 1] if proba.shape[1] == 2 else proba
            score = self.metric(y_np.astype(int), preds)
            coefs = model.coef_.flatten()
            denom = np.abs(coefs).sum()
            norm  = coefs / denom if denom > 0 else coefs
            self._print_meta(norm, score, f"LogisticRegression (C={model.C_[0]:.4f})")

    def _fit_meta(self, X_np: np.ndarray, y_np: np.ndarray) -> None:
        """Arbitrary sklearn estimator as meta-model."""
        if self.meta_estimator is None:
            raise ValueError(
                "method='meta' requires a meta_estimator. "
                "Pass any sklearn-compatible estimator, e.g.: "
                "meta_estimator=Ridge() or meta_estimator=LGBMRegressor()"
            )
        model = copy.deepcopy(self.meta_estimator)
        y_fit = y_np.astype(int) if self.objective == "maximize" else y_np
        model.fit(X_np, y_fit)
        self._meta_model = model

        if self.verbose:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_np)
                preds = proba[:, 1] if proba.shape[1] == 2 else proba
            else:
                preds = model.predict(X_np)
            score = self.metric(y_fit, preds)
            self._print_meta(None, score, f"Meta: {type(model).__name__}")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _metric_name(self) -> str:
        return getattr(self.metric, "__name__", "metric")

    def _print_weights(self, weights: np.ndarray, score: float, label: str) -> None:
        pad = max(0, 32 - len(label))
        print(f"\n+- WeightedEnsemble [{label}] {chr(45) * pad}+")
        for i, w in enumerate(weights, start=1):
            bar = chr(9608) * int(w * 36)
            print(f"|  Model {i:>2}  {w:.4f}  {bar}")
        print(f"|  {self._metric_name()}: {score:.6f}")
        print("+" + "-" * 50 + "+\n")

    def _print_meta(
        self,
        weights: Optional[np.ndarray],
        score: float,
        label: str,
    ) -> None:
        pad = max(0, 32 - len(label))
        print(f"\n+- WeightedEnsemble [{label}] {chr(45) * pad}+")
        if weights is not None:
            for i, w in enumerate(weights, start=1):
                sign_char = "+" if w >= 0 else "-"
                bar = chr(9608) * int(abs(w) * 36)
                print(f"|  Model {i:>2}  {sign_char}{abs(w):.4f}  {bar}")
        print(f"|  {self._metric_name()}: {score:.6f}")
        print("+" + "-" * 50 + "+\n")

    def _check_is_fitted(self) -> None:
        if self._weights is None and self._meta_model is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted yet. "
                "Call .fit() first."
            )


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

class WeightedEnsembleRegressor(_BaseWeightedEnsemble):
    """
    Weighted ensemble for regression tasks (minimises metric).

    Parameters
    ----------
    metric : callable
        metric(y_true, y_pred) -> float to MINIMISE (e.g. MAE, RMSE).
    method : {"nelder-mead", "optuna", "ridge", "meta"}
        - "nelder-mead" : scipy simplex optimisation (default)
        - "optuna"      : Bayesian simplex search (pip install optuna)
        - "ridge"       : RidgeCV (weights can be negative)
        - "meta"        : any sklearn regressor via meta_estimator
    meta_estimator : sklearn regressor, optional
        Required when method="meta".
    n_trials : int
        Optuna trials (method="optuna" only).
    verbose : bool

    Examples
    --------
    ::

        from sklearn.metrics import mean_absolute_error
        from mlbox import WeightedEnsembleRegressor

        # Nelder-Mead (default)
        ens = WeightedEnsembleRegressor(metric=mean_absolute_error)
        ens.fit(oof_matrix, y_train)
        preds = ens.predict(test_matrix)
        print(ens.weights_)

        # Optuna
        ens = WeightedEnsembleRegressor(
            metric=mean_absolute_error, method="optuna", n_trials=500
        )
        ens.fit(oof_matrix, y_train)

        # Ridge
        ens = WeightedEnsembleRegressor(
            metric=mean_absolute_error, method="ridge"
        )
        ens.fit(oof_matrix, y_train)
        print(ens.meta_model_.coef_)

        # Meta model (LightGBM)
        from lightgbm import LGBMRegressor
        ens = WeightedEnsembleRegressor(
            metric=mean_absolute_error,
            method="meta",
            meta_estimator=LGBMRegressor(n_estimators=200),
        )
        ens.fit(oof_matrix, y_train)
    """

    def __init__(
        self,
        metric: Callable[[np.ndarray, np.ndarray], float],
        method: Method = "nelder-mead",
        meta_estimator: Optional[Any] = None,
        n_trials: int = 200,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            objective="minimize",
            metric=metric,
            method=method,
            meta_estimator=meta_estimator,
            n_trials=n_trials,
            verbose=verbose,
        )


class WeightedEnsembleClassifier(_BaseWeightedEnsemble):
    """
    Weighted ensemble for classification tasks (maximises metric).

    Parameters
    ----------
    metric : callable
        metric(y_true, y_pred) -> float to MAXIMISE (e.g. AUC, F1).
    method : {"nelder-mead", "optuna", "logistic", "meta"}
        - "nelder-mead" : scipy simplex optimisation (default)
        - "optuna"      : Bayesian simplex search (pip install optuna)
        - "logistic"    : LogisticRegressionCV on OOF probabilities
        - "meta"        : any sklearn classifier via meta_estimator
    meta_estimator : sklearn classifier, optional
        Required when method="meta".
    n_trials : int
        Optuna trials (method="optuna" only).
    verbose : bool

    Examples
    --------
    ::

        from sklearn.metrics import roc_auc_score
        from mlbox import WeightedEnsembleClassifier

        # Nelder-Mead (default)
        ens = WeightedEnsembleClassifier(metric=roc_auc_score)
        ens.fit(oof_matrix, y_train)
        preds = ens.predict(test_matrix)

        # Logistic
        ens = WeightedEnsembleClassifier(
            metric=roc_auc_score, method="logistic"
        )
        ens.fit(oof_matrix, y_train)
        print(ens.meta_model_.C_)

        # Meta model (LightGBM)
        from lightgbm import LGBMClassifier
        ens = WeightedEnsembleClassifier(
            metric=roc_auc_score,
            method="meta",
            meta_estimator=LGBMClassifier(n_estimators=200),
        )
        ens.fit(oof_matrix, y_train)
    """

    def __init__(
        self,
        metric: Callable[[np.ndarray, np.ndarray], float],
        method: Method = "nelder-mead",
        meta_estimator: Optional[Any] = None,
        n_trials: int = 200,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            objective="maximize",
            metric=metric,
            method=method,
            meta_estimator=meta_estimator,
            n_trials=n_trials,
            verbose=verbose,
        )