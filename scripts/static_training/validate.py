"""Validation that answers the question the deployment actually asks.

The existing split is random per pixel. Pixels from one field land on both sides
of it, so the reported score measures how well the model recognises fields it has
already seen, on an image it has already seen. It cannot fall, and it told nobody
that August was going to break.

Two grouped protocols replace it:

  leave-one-date-out   trains on some acquisitions and scores a date never seen.
                       This is the number that predicts behaviour on a new image.
  leave-one-AOI-out    trains on some mills and scores a region never seen. This
                       catches a model that has memorised places rather than crops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class FoldResult:
    held_out: str
    n_train: int
    n_test: int
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    threshold: float = 0.5

    def as_row(self) -> Dict[str, object]:
        return {
            "held_out": self.held_out,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": None if self.roc_auc is None else round(self.roc_auc, 4),
        }


@dataclass
class GroupedReport:
    group_column: str
    folds: List[FoldResult] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([f.as_row() for f in self.folds])

    def summary(self) -> Dict[str, float]:
        """Mean and spread across folds. The spread is the interesting half."""
        table = self.table()
        return {
            "mean_recall": float(table["recall"].mean()),
            "min_recall": float(table["recall"].min()),
            "recall_spread": float(table["recall"].max() - table["recall"].min()),
            "mean_f1": float(table["f1"].mean()),
            "min_f1": float(table["f1"].min()),
        }


def binary_scores(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> tuple[float, float, float]:
    """Precision, recall and F1 for the positive class at a given threshold."""
    predicted = y_prob >= threshold
    true_positive = float(np.sum(predicted & (y_true == 1)))
    false_positive = float(np.sum(predicted & (y_true == 0)))
    false_negative = float(np.sum(~predicted & (y_true == 1)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


#: Kept so existing callers of the private name keep working.
_scores = binary_scores


def leave_one_group_out(
    frame: pd.DataFrame,
    group_column: str,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    label_column: str = "label",
    positive_label: int = 1,
    threshold: float = 0.5,
    groups: Optional[Sequence[str]] = None,
    min_test_rows: int = 1000,
) -> GroupedReport:
    """Hold out each group in turn, fit on the rest, score on the held-out group.

    `fit_predict` takes (train_frame, test_frame) and returns positive-class
    probabilities for the test rows. Keeping the model out of this module means
    the same protocol works for any estimator.
    """
    report = GroupedReport(group_column=group_column)
    candidates = list(groups) if groups is not None else sorted(frame[group_column].unique())

    for group in candidates:
        test_mask = frame[group_column] == group
        test = frame[test_mask]
        train = frame[~test_mask]
        if len(test) < min_test_rows or train.empty:
            continue
        y_true = (test[label_column].to_numpy() == positive_label).astype(int)
        if y_true.min() == y_true.max():
            continue  # a fold with only one class says nothing

        y_prob = fit_predict(train, test)
        precision, recall, f1 = binary_scores(y_true, y_prob, threshold)

        roc_auc = None
        try:
            from sklearn.metrics import roc_auc_score

            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

        report.folds.append(
            FoldResult(str(group), len(train), len(test), precision, recall, f1, roc_auc, threshold)
        )
    return report


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "f1",
    grid: Optional[Sequence[float]] = None,
) -> tuple[float, Dict[str, float]]:
    """Pick a decision threshold, and report what it costs.

    Choose this on a held-out *date*, not a random split, or the threshold is tuned
    to the same acquisition the model already fits well.
    """
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.01), 2)
    best_threshold, best_value, best_scores = 0.5, -1.0, {}
    for threshold in grid:
        precision, recall, f1 = binary_scores(y_true, y_prob, float(threshold))
        value = {"f1": f1, "recall": recall, "precision": precision}[objective]
        if value > best_value:
            best_threshold, best_value = float(threshold), value
            best_scores = {"precision": precision, "recall": recall, "f1": f1}
    return best_threshold, best_scores
