"""
mlbox.feature_selection
-----------------------
Cross-validated Sequential Feature Selector.

Direction
---------
  forward  : start empty, add the feature that improves CV metric most.
  backward : start with all features, remove the feature whose removal
             hurts CV metric least.

Data-leakage guarantee
-----------------------
Feature selection uses cross-validation on the TRAINING set only.
The test set (X_test) is never seen during selection — it is merely
transformed by subsetting to the selected feature columns.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Literal, Optional

import numpy as np

from mlbox.utils.dataframe import (
    ArrayLike,
    get_columns,
    iloc_rows,
    is_dataframe,
    to_numpy,
    validate_X_y,
)
from mlbox.utils.logging import FoldLogger, timer
from mlbox.utils.pipeline import pipeline_predict


# ---------------------------------------------------------------------------
# SequentialFeatureSelector
# ---------------------------------------------------------------------------

class SequentialFeatureSelector:
    """
    Greedy sequential feature selector with cross-validation.

    Parameters
    ----------
    estimator : sklearn estimator (NOT a Pipeline — preprocessing must be
                handled externally to keep the interface clean)
    cv        : sklearn cross-validator
    metric    : callable ``metric(y_true, y_pred) -> float``
    objective : {"minimize", "maximize"}
    direction : {"forward", "backward"}
    task      : {"binary", "multiclass", "regression"}
    verbose   : bool

    Attributes (post fit)
    ---------------------
    selected_features_ : list[str | int]
        Names (if DataFrame) or indices of selected features.
    feature_importances_ : dict
        Mapping feature → CV score improvement from including it.
    """

    def __init__(
        self,
        estimator: Any,
        cv: Any,
        metric: Callable[[np.ndarray, np.ndarray], float],
        objective: Literal["minimize", "maximize"] = "maximize",
        direction: Literal["forward", "backward"] = "forward",
        task: str = "binary",
        verbose: bool = True,
    ) -> None:
        self.estimator  = estimator
        self.cv         = cv
        self.metric     = metric
        self.objective  = objective
        self.direction  = direction
        self.task       = task
        self.verbose    = verbose

        self._selected_features: Optional[list] = None
        self._feature_importances: dict         = {}
        self._all_feature_names: Optional[list] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def selected_features_(self) -> list:
        self._check_is_fitted()
        return self._selected_features  # type: ignore[return-value]

    @property
    def feature_importances_(self) -> dict:
        self._check_is_fitted()
        return self._feature_importances

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SequentialFeatureSelector":
        """
        Run sequential feature selection on (X, y).

        Only X is used here. X_test is never touched — call .transform()
        separately on the test set.
        """
        validate_X_y(X, y)
        y_np = to_numpy(y)

        if is_dataframe(X):
            all_features = get_columns(X)
        else:
            X_np = to_numpy(X)
            all_features = list(range(X_np.shape[1]))

        self._all_feature_names = all_features

        if self.direction == "forward":
            self._selected_features = self._forward_selection(X, y_np, all_features)
        else:
            self._selected_features = self._backward_selection(X, y_np, all_features)

        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Subset X to the selected features, preserving the original type."""
        self._check_is_fitted()
        return self._subset_features(X, self._selected_features)  # type: ignore[arg-type]

    def fit_transform(self, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Selection algorithms
    # ------------------------------------------------------------------

    def _forward_selection(
        self, X: ArrayLike, y_np: np.ndarray, all_features: list
    ) -> list:
        selected:   list  = []
        remaining         = list(all_features)
        best_score        = -np.inf if self.objective == "maximize" else np.inf

        if self.verbose:
            print("\n▶ Forward Feature Selection")
            print(f"  Total features: {len(all_features)}")

        while remaining:
            candidate_scores: dict = {}

            for feat in remaining:
                trial = selected + [feat]
                score = self._cv_score(X, y_np, trial)
                candidate_scores[feat] = score

            best_feat = self._best_candidate(candidate_scores)
            best_feat_score = candidate_scores[best_feat]

            # Only add if it improves
            if self._improves(best_feat_score, best_score):
                if self.verbose:
                    direction_str = "↑" if self.objective == "maximize" else "↓"
                    print(
                        f"  + {str(best_feat):<30}  "
                        f"score={best_feat_score:.6f}  {direction_str}"
                    )
                selected.append(best_feat)
                remaining.remove(best_feat)
                best_score = best_feat_score
                self._feature_importances[best_feat] = best_feat_score
            else:
                if self.verbose:
                    print(f"  ✗ No improvement — stopping at {len(selected)} features")
                break

        return selected

    def _backward_selection(
        self, X: ArrayLike, y_np: np.ndarray, all_features: list
    ) -> list:
        selected = list(all_features)
        best_score = self._cv_score(X, y_np, selected)

        if self.verbose:
            print("\n◀ Backward Feature Selection")
            print(f"  Starting features: {len(selected)}  (baseline score: {best_score:.6f})")

        while len(selected) > 1:
            candidate_scores: dict = {}

            for feat in selected:
                trial = [f for f in selected if f != feat]
                score = self._cv_score(X, y_np, trial)
                candidate_scores[feat] = score

            # Feature whose removal hurts the least (or helps most)
            best_removal = self._best_candidate(candidate_scores)
            best_removal_score = candidate_scores[best_removal]

            if self._improves(best_removal_score, best_score) or np.isclose(
                best_removal_score, best_score, atol=1e-8
            ):
                if self.verbose:
                    print(
                        f"  - {str(best_removal):<30}  "
                        f"score={best_removal_score:.6f}"
                    )
                selected.remove(best_removal)
                best_score = best_removal_score
            else:
                if self.verbose:
                    print(f"  ✗ No improvement — stopping at {len(selected)} features")
                break

        if self.verbose:
            print(f"  Selected: {len(selected)} features")

        return selected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cv_score(
        self, X: ArrayLike, y_np: np.ndarray, features: list
    ) -> float:
        """Compute mean CV score for a given feature subset — on TRAIN only."""
        X_sub = self._subset_features(X, features)
        scores: list[float] = []

        for train_idx, val_idx in self.cv.split(X_sub, y_np):
            X_tr = iloc_rows(X_sub, np.array(train_idx))
            X_vl = iloc_rows(X_sub, np.array(val_idx))
            y_tr = y_np[train_idx]
            y_vl = y_np[val_idx]

            model = copy.deepcopy(self.estimator)
            model.fit(X_tr, y_tr)
            preds = pipeline_predict(model, X_vl, self.task)
            scores.append(float(self.metric(y_vl, preds)))

        return float(np.mean(scores))

    def _best_candidate(self, scores: dict) -> Any:
        if self.objective == "maximize":
            return max(scores, key=scores.__getitem__)
        return min(scores, key=scores.__getitem__)

    def _improves(self, new: float, current: float) -> bool:
        if self.objective == "maximize":
            return new > current
        return new < current

    @staticmethod
    def _subset_features(X: ArrayLike, features: list) -> ArrayLike:
        """Select columns by name (DataFrame) or index (ndarray)."""
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                return X[features]
        except ImportError:
            pass
        try:
            import polars as pl
            if isinstance(X, pl.DataFrame):
                str_features = [str(f) for f in features]
                return X.select(str_features)
        except ImportError:
            pass
        # numpy fallback
        X_np = to_numpy(X)
        return X_np[:, features]

    def _check_is_fitted(self) -> None:
        if self._selected_features is None:
            raise RuntimeError(
                "SequentialFeatureSelector has not been fitted yet. "
                "Call .fit() first."
            )
