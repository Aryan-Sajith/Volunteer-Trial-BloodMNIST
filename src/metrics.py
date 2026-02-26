from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    weighted_f1: float
    per_class_f1: list[float]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> ClassificationMetrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    wf1 = float(f1_score(y_true, y_pred, average="weighted"))
    per_class = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    return ClassificationMetrics(accuracy=acc, weighted_f1=wf1, per_class_f1=[float(x) for x in per_class])
