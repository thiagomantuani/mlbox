"""
mlbox.utils.early_stopping
--------------------------
Centralised early-stopping configuration.

Each boosting framework uses a different API to enable early stopping:

  XGBoost   : fit(..., eval_set=[...], early_stopping_rounds=N)
  LightGBM  : fit(..., eval_set=[...], callbacks=[lgb.early_stopping(N)])
  CatBoost  : fit(..., eval_set=Pool(...), early_stopping_rounds=N)

This module detects the estimator (or the *last step* of a sklearn Pipeline)
and returns the correct `fit_params` dict so the Trainer never has to
branch on framework type.

Design
------
- No leakage: validation data passed here must ALWAYS be the fold's
  validation split, never the full training set.
- If the estimator is not a supported booster, an empty dict is returned
  and training proceeds without early stopping (sklearn behaviour).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class EarlyStoppingConfig:
    """
    User-facing configuration for early stopping.

    Parameters
    ----------
    rounds : int
        Number of rounds without improvement before stopping.
    verbose : bool
        Whether the booster should print its own eval logs.
    metric : str, optional
        Metric name recognised by the booster (e.g. ``"auc"``, ``"rmse"``).
        If None, the booster's default metric is used.
    """
    rounds: int = 50
    verbose: bool = False
    metric: Optional[str] = None

    def __post_init__(self) -> None:
        if self.rounds <= 0:
            raise ValueError(f"rounds must be > 0, got {self.rounds}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unwrap_estimator(estimator: Any) -> Any:
    """
    If *estimator* is a sklearn Pipeline, return its last step.
    Otherwise return the estimator as-is.
    """
    # Avoid hard import of sklearn at module level
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(estimator, Pipeline):
            return estimator.steps[-1][1]
    except ImportError:
        pass
    return estimator


def _is_xgboost(est: Any) -> bool:
    try:
        import xgboost as xgb
        return isinstance(est, xgb.XGBModel)
    except ImportError:
        return False


def _is_lightgbm(est: Any) -> bool:
    try:
        import lightgbm as lgb
        return isinstance(est, lgb.LGBMModel)
    except ImportError:
        return False


def _is_catboost(est: Any) -> bool:
    try:
        from catboost import CatBoost
        return isinstance(est, CatBoost)
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_fit_params(
    estimator: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: EarlyStoppingConfig,
    pipeline_last_step_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build the ``fit_params`` dict for early stopping.

    Parameters
    ----------
    estimator : Any
        The estimator or sklearn Pipeline being fitted.
    X_val : np.ndarray
        Validation features (already preprocessed if *not* inside a Pipeline).
    y_val : np.ndarray
        Validation labels.
    config : EarlyStoppingConfig
        Early stopping settings.
    pipeline_last_step_name : str, optional
        Name of the last step in a sklearn Pipeline. When provided, params
        are prefixed as ``"<name>__<param>"`` to route them correctly.

    Returns
    -------
    dict
        Ready-to-unpack ``**fit_params``.

    Notes
    -----
    When the estimator **is** a Pipeline, X_val and y_val are the RAW
    (unprocessed) validation fold. The Pipeline will internally apply the
    preprocessing steps before passing data to the booster — this is the
    correct, leakage-free behaviour.
    """
    core = _unwrap_estimator(estimator)
    params: dict[str, Any] = {}

    if _is_xgboost(core):
        params = _xgboost_params(X_val, y_val, config)
    elif _is_lightgbm(core):
        params = _lightgbm_params(X_val, y_val, config)
    elif _is_catboost(core):
        params = _catboost_params(X_val, y_val, config)
    else:
        # Not a known booster — early stopping silently disabled
        if config.rounds:
            warnings.warn(
                f"EarlyStoppingConfig was provided but {type(core).__name__!r} "
                "is not a recognised booster (XGBoost / LightGBM / CatBoost). "
                "Early stopping will be ignored.",
                UserWarning,
                stacklevel=3,
            )
        return {}

    # If wrapped in a Pipeline, prefix params with the step name
    if pipeline_last_step_name:
        params = {f"{pipeline_last_step_name}__{k}": v for k, v in params.items()}

    return params


# ---------------------------------------------------------------------------
# Framework-specific helpers
# ---------------------------------------------------------------------------

def _xgboost_params(
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: EarlyStoppingConfig,
) -> dict[str, Any]:
    return {
        "eval_set": [(X_val, y_val)],
        "early_stopping_rounds": config.rounds,
        "verbose": config.verbose,
    }


def _lightgbm_params(
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: EarlyStoppingConfig,
) -> dict[str, Any]:
    import lightgbm as lgb

    callbacks = [lgb.early_stopping(config.rounds, verbose=config.verbose)]
    if not config.verbose:
        callbacks.append(lgb.log_evaluation(period=-1))

    return {
        "eval_set": [(X_val, y_val)],
        "callbacks": callbacks,
    }


def _catboost_params(
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: EarlyStoppingConfig,
) -> dict[str, Any]:
    from catboost import Pool

    return {
        "eval_set": Pool(X_val, y_val),
        "early_stopping_rounds": config.rounds,
        "verbose": config.verbose,
    }
