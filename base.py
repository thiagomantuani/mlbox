"""
mlbox.base
----------
Abstract base classes that define the contracts for all estimators in mlbox.
"""

from __future__ import annotations

import abc
from typing import Any, Optional, Union

import numpy as np


class BaseEstimator(abc.ABC):
    """Root base class for all mlbox estimators."""

    @abc.abstractmethod
    def fit(self, X: Any, y: Any) -> "BaseEstimator":
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({params})"


class BaseTrainer(BaseEstimator):
    """
    Contract for cross-validated trainers.

    Attributes guaranteed after fit():
        oof_preds_     : out-of-fold predictions (same length as training set)
        fold_scores_   : list of per-fold metric values
        overall_score_ : aggregate metric across all OOF predictions
        models_        : list of fitted estimators (one per fold)
    """

    @property
    @abc.abstractmethod
    def oof_preds_(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def fold_scores_(self) -> list[float]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def overall_score_(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def models_(self) -> list[Any]:
        raise NotImplementedError


class BaseEnsemble(BaseEstimator):
    """Contract for ensemble estimators."""

    @property
    @abc.abstractmethod
    def weights_(self) -> np.ndarray:
        raise NotImplementedError
