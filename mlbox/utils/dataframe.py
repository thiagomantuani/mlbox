"""
mlbox.utils.dataframe
---------------------
Thin adapter layer that normalises pandas DataFrames / Series and
Polars DataFrames / Series into numpy arrays, keeping column names
available for downstream use.

Design goal: zero copies when possible; fail loudly on unknown types.
"""

from __future__ import annotations

from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
if _PANDAS_AVAILABLE and _POLARS_AVAILABLE:
    DataFrameLike = Union["pd.DataFrame", "pl.DataFrame"]
    SeriesLike = Union["pd.Series", "pl.Series"]
    ArrayLike = Union[DataFrameLike, SeriesLike, np.ndarray]
else:
    DataFrameLike = object
    SeriesLike = object
    ArrayLike = object


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_dataframe(obj: object) -> bool:
    """Return True if *obj* is a pandas or polars DataFrame."""
    if _PANDAS_AVAILABLE and isinstance(obj, pd.DataFrame):
        return True
    if _POLARS_AVAILABLE and isinstance(obj, pl.DataFrame):
        return True
    return False


def is_series(obj: object) -> bool:
    """Return True if *obj* is a pandas or polars Series."""
    if _PANDAS_AVAILABLE and isinstance(obj, pd.Series):
        return True
    if _POLARS_AVAILABLE and isinstance(obj, pl.Series):
        return True
    return False


def get_columns(X: DataFrameLike) -> list[str]:
    """Return column names as a list of strings."""
    if _PANDAS_AVAILABLE and isinstance(X, pd.DataFrame):
        return list(X.columns)
    if _POLARS_AVAILABLE and isinstance(X, pl.DataFrame):
        return X.columns
    raise TypeError(f"Cannot extract columns from type {type(X)!r}")


def to_numpy(obj: ArrayLike) -> np.ndarray:
    """
    Convert pandas / polars DataFrame, Series, or numpy array to ndarray.

    No unnecessary copies are made:
    - pandas  → .to_numpy()
    - polars  → .to_numpy()
    - ndarray → returned as-is
    """
    if isinstance(obj, np.ndarray):
        return obj
    if _PANDAS_AVAILABLE and isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_numpy()
    if _POLARS_AVAILABLE and isinstance(obj, (pl.DataFrame, pl.Series)):
        return obj.to_numpy()
    raise TypeError(
        f"Unsupported type {type(obj)!r}. "
        "Expected np.ndarray, pd.DataFrame/Series, or pl.DataFrame/Series."
    )


def iloc_rows(X: ArrayLike, indices: np.ndarray) -> ArrayLike:
    """
    Subset rows by integer indices while preserving the original type.

    This is the *only* place we perform row subsetting so that the rest
    of the codebase never needs to branch on DataFrame type.
    """
    if _PANDAS_AVAILABLE and isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[indices]
    if _POLARS_AVAILABLE and isinstance(X, pl.DataFrame):
        return X[indices]
    if _POLARS_AVAILABLE and isinstance(X, pl.Series):
        return X[indices]
    if isinstance(X, np.ndarray):
        return X[indices]
    raise TypeError(f"Cannot subset rows of type {type(X)!r}")


def validate_X_y(X: ArrayLike, y: ArrayLike) -> None:
    """
    Minimal sanity checks on X and y.

    Raises
    ------
    TypeError  : unsupported type
    ValueError : length mismatch
    """
    if not (is_dataframe(X) or is_series(X) or isinstance(X, np.ndarray)):
        raise TypeError(
            f"X must be a pandas/polars DataFrame or numpy array, got {type(X)!r}"
        )
    if not (is_series(y) or isinstance(y, np.ndarray)):
        raise TypeError(
            f"y must be a pandas/polars Series or numpy array, got {type(y)!r}"
        )

    n_X = X.shape[0] if hasattr(X, "shape") else len(X)
    n_y = len(y)
    if n_X != n_y:
        raise ValueError(
            f"X and y have inconsistent lengths: X has {n_X} rows, y has {n_y} elements."
        )
