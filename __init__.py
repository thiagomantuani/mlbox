"""
mlbox
=====
A production-ready ML toolkit for Kaggle competitions and beyond.

Features
--------
- Leakage-safe cross-validated Trainer
- First-class sklearn Pipeline support (make_pipeline / Pipeline)
- Configurable early stopping for XGBoost, LightGBM, CatBoost
- pandas AND polars DataFrames supported everywhere
- WeightedEnsemble[Regressor / Classifier]
- SequentialFeatureSelector (forward / backward)
- Rich tqdm + per-fold score table logging

Quick start
-----------
>>> from mlbox import Trainer, EarlyStoppingConfig
>>> from mlbox import WeightedEnsembleRegressor, WeightedEnsembleClassifier
>>> from mlbox import SequentialFeatureSelector
"""

from mlbox.trainer import Trainer
from mlbox.ensemble import WeightedEnsembleRegressor, WeightedEnsembleClassifier
from mlbox.feature_selection import SequentialFeatureSelector
from mlbox.utils.early_stopping import EarlyStoppingConfig

__all__ = [
    "Trainer",
    "WeightedEnsembleRegressor",
    "WeightedEnsembleClassifier",
    "SequentialFeatureSelector",
    "EarlyStoppingConfig",
]

__version__ = "0.1.0"
__author__  = "mlbox contributors"
