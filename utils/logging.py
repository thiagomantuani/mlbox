"""
mlbox.utils.logging
-------------------
Rich console output helpers used throughout mlbox.

Uses tqdm for progress bars and a lightweight ASCII table renderer
so the output looks great both on Kaggle notebooks and in terminals.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator, Optional

try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colour / style helpers (no external dependency)
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RED    = "\033[91m"
_GREY   = "\033[90m"


def _c(text: str, *codes: str) -> str:
    """Wrap *text* in ANSI escape codes (only when writing to a real TTY)."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _RESET


# ---------------------------------------------------------------------------
# FoldLogger
# ---------------------------------------------------------------------------

class FoldLogger:
    """
    Prints a per-fold summary table and final statistics.

    Example output
    --------------
    ┌──────────┬────────────────┬─────────────┐
    │  Fold    │   Score        │   Duration  │
    ├──────────┼────────────────┼─────────────┤
    │  1 / 5   │   0.85431      │   2.31 s    │
    │  2 / 5   │   0.86012      │   2.28 s    │
    ...
    ├──────────┴────────────────┴─────────────┤
    │  Mean ± Std   0.8547 ± 0.0032           │
    └─────────────────────────────────────────┘
    """

    _COL_FOLD  = 12
    _COL_SCORE = 18
    _COL_DUR   = 13
    _TOTAL_W   = _COL_FOLD + _COL_SCORE + _COL_DUR + 4  # 3 separators + 1 border

    def __init__(self, n_folds: int, metric_name: str = "Score", verbose: bool = True):
        self.n_folds = n_folds
        self.metric_name = metric_name
        self.verbose = verbose
        self._scores: list[float] = []
        self._durations: list[float] = []

    # ------------------------------------------------------------------
    # Internal table helpers
    # ------------------------------------------------------------------

    def _row(self, *cells: tuple[str, int]) -> str:
        """Render a single table row given (content, width) pairs."""
        parts = [f" {c:<{w}} " for c, w in cells]
        return "│" + "│".join(parts) + "│"

    def _divider(self, left: str = "├", mid: str = "┼", right: str = "┤") -> str:
        segs = [
            "─" * (self._COL_FOLD + 2),
            "─" * (self._COL_SCORE + 2),
            "─" * (self._COL_DUR + 2),
        ]
        return left + mid.join(segs) + right

    def _full_line(self, text: str, left: str = "│", right: str = "│") -> str:
        inner = self._TOTAL_W - 2
        return left + f" {text:<{inner - 1}}" + right

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def print_header(self) -> None:
        if not self.verbose:
            return
        top = "┌" + "─" * (self._TOTAL_W - 2) + "┐"
        title = _c(f" mlbox Trainer — {self.n_folds}-fold CV ", _BOLD, _CYAN)
        # Title row (full width)
        inner = self._TOTAL_W - 2
        raw_title = f"mlbox Trainer — {self.n_folds}-fold CV"
        padding = max(0, inner - 1 - len(raw_title))
        title_line = "│ " + _c(raw_title, _BOLD, _CYAN) + " " * padding + "│"

        print(top)
        print(title_line)
        print(self._divider("├", "┬", "┤"))
        print(self._row(
            (_c("Fold", _BOLD), self._COL_FOLD),
            (_c(self.metric_name, _BOLD), self._COL_SCORE),
            (_c("Duration", _BOLD), self._COL_DUR),
        ))
        print(self._divider())

    def log_fold(self, fold: int, score: float, duration: float) -> None:
        self._scores.append(score)
        self._durations.append(duration)
        if not self.verbose:
            return

        fold_str  = f"{fold} / {self.n_folds}"
        score_str = _c(f"{score:.6f}", _GREEN)
        dur_str   = f"{duration:.2f} s"

        print(self._row(
            (fold_str,  self._COL_FOLD),
            (score_str, self._COL_SCORE),
            (dur_str,   self._COL_DUR),
        ))

    def print_footer(self) -> None:
        if not self.verbose:
            return
        import numpy as np
        mean  = float(np.mean(self._scores))
        std   = float(np.std(self._scores))
        total = sum(self._durations)

        print(self._divider("├", "┴", "┤"))

        mean_str  = _c(f"{mean:.6f}", _BOLD, _YELLOW)
        std_str   = _c(f"{std:.6f}", _GREY)
        total_str = _c(f"{total:.2f} s", _GREY)

        inner = self._TOTAL_W - 2
        summary = f"Mean ± Std   {mean:.6f} ± {std:.6f}   Total {total:.2f} s"
        padding = max(0, inner - 1 - len(summary))
        # Coloured version
        col_summary = (
            f"Mean ± Std   "
            + _c(f"{mean:.6f}", _BOLD, _YELLOW)
            + " ± "
            + _c(f"{std:.6f}", _GREY)
            + "   Total "
            + _c(f"{total:.2f} s", _GREY)
        )
        print("│ " + col_summary + " " * padding + "│")
        print("└" + "─" * (self._TOTAL_W - 2) + "┘")


# ---------------------------------------------------------------------------
# tqdm wrapper
# ---------------------------------------------------------------------------

def fold_range(n_folds: int, verbose: bool = True) -> Iterator[int]:
    """
    Yield fold indices 1..n_folds, optionally wrapped in a tqdm bar.
    """
    indices = range(1, n_folds + 1)
    if verbose and _TQDM_AVAILABLE:
        yield from _tqdm(indices, desc="Folds", unit="fold", leave=False)
    else:
        yield from indices


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

@contextmanager
def timer() -> Iterator[dict]:
    """Context manager that records elapsed wall-clock time."""
    info: dict = {}
    start = time.perf_counter()
    try:
        yield info
    finally:
        info["elapsed"] = time.perf_counter() - start
