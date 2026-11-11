"""Evaluation metrics shared by both models."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Threshold-free metrics. ``scores`` are probabilities of the positive class."""
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
    }


def threshold_for_recall(labels: np.ndarray, scores: np.ndarray, recall: float) -> float:
    """Highest threshold at which at least ``recall`` of the positives are caught."""
    positives = np.sort(scores[labels == 1])
    index = int(np.floor((1 - recall) * len(positives)))
    return float(positives[index])


def operating_point(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Counts and rates when clips scoring >= ``threshold`` are flagged."""
    flagged = scores >= threshold
    tp = int((flagged & (labels == 1)).sum())
    fp = int((flagged & (labels == 0)).sum())
    fn = int((~flagged & (labels == 1)).sum())
    tn = int((~flagged & (labels == 0)).sum())
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_positives_per_1000_clips": 1000 * fp / len(labels),
    }


def recall_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float = 0.01) -> float:
    fprs, tprs, _ = roc_curve(labels, scores)
    return float(np.interp(fpr, fprs, tprs))
