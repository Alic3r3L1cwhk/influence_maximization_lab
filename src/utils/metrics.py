"""Utility functions for metrics and evaluation."""

import numpy as np
from typing import List, Dict
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Accuracy
    metrics['accuracy'] = np.mean(y_true == y_pred_binary)

    # Precision
    true_positives = np.sum((y_true == 1) & (y_pred_binary == 1))
    predicted_positives = np.sum(y_pred_binary == 1)
    metrics['precision'] = true_positives / max(predicted_positives, 1)

    # Recall
    actual_positives = np.sum(y_true == 1)
    metrics['recall'] = true_positives / max(actual_positives, 1)

    # F1 Score
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                       (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0.0

    # AUC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred)
    except ValueError:
        metrics['auc'] = 0.0

    # Average Precision
    try:
        metrics['ap'] = average_precision_score(y_true, y_pred)
    except ValueError:
        metrics['ap'] = 0.0

    return metrics


def compute_influence_statistics(influences: List[float]) -> Dict[str, float]:
    """
    Compute statistics for influence values.

    Args:
        influences: List of influence values from multiple runs

    Returns:
        Dictionary of statistics
    """
    influences = np.array(influences)

    stats = {
        'mean': np.mean(influences),
        'std': np.std(influences),
        'min': np.min(influences),
        'max': np.max(influences),
        'median': np.median(influences),
        'q25': np.percentile(influences, 25),
        'q75': np.percentile(influences, 75)
    }

    return stats


def compare_seed_sets(true_seeds: List[int], learned_seeds: List[int]) -> Dict[str, float]:
    """
    Compare two seed sets.

    Args:
        true_seeds: Seed set selected using true parameters
        learned_seeds: Seed set selected using learned parameters

    Returns:
        Dictionary of comparison metrics
    """
    true_set = set(true_seeds)
    learned_set = set(learned_seeds)

    metrics = {
        'overlap': len(true_set.intersection(learned_set)),
        'overlap_ratio': len(true_set.intersection(learned_set)) / len(true_set),
        'jaccard': len(true_set.intersection(learned_set)) / len(true_set.union(learned_set))
    }

    return metrics
