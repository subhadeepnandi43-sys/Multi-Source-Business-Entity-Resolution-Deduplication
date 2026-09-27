"""
Evaluation and Threshold Optimization Module for Business Entity Resolution.
Calculates Precision, Recall, and F0.5 (precision-weighted metric).
Optimizes decision thresholds for F0.5 and assesses singleton performance.
Generates comprehensive diagnostic reports.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import pandas as pd


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Computes F-beta score.
    For beta=0.5, precision is weighted twice as heavily as recall.
    F0.5 = (1 + 0.25) * (P * R) / (0.25 * P + R)
    """
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    numerator = (1.0 + beta_sq) * (precision * recall)
    denominator = (beta_sq * precision) + recall
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def evaluate_predictions(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Computes binary pair-level classification metrics at a specific threshold.
    """
    y_pred = (y_probs >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f05 = compute_f_beta(precision, recall, beta=0.5)

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f05": float(f05)
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold_range: Optional[np.ndarray] = None
) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
    """
    Grid searches over decision thresholds to find the threshold maximizing F0.5.
    Returns:
        best_threshold: Optimal probability threshold
        best_metrics: Metrics at the optimal threshold
        all_results: History of all tested thresholds for PR curves and reporting
    """
    if threshold_range is None:
        threshold_range = np.linspace(0.30, 0.95, 66)

    best_threshold = 0.50
    best_f05 = -1.0
    best_metrics = {}
    all_results = []

    for th in threshold_range:
        m = evaluate_predictions(y_true, y_probs, th)
        all_results.append(m)
        if m["f05"] > best_f05:
            best_f05 = m["f05"]
            best_threshold = float(th)
            best_metrics = m

    return best_threshold, best_metrics, all_results


def evaluate_entity_level(
    val_s1_ids: Set[str],
    ground_truth_map: Dict[str, Set[str]],
    cand_df: pd.DataFrame,
    y_probs: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Evaluates entity-level matching and singleton detection performance.
    """
    # Group predictions by Source 1 entity
    cand_df = cand_df.copy()
    cand_df["prob"] = y_probs
    cand_df["pred_match"] = cand_df["prob"] >= threshold

    # Predicted matches per S1 entity
    pred_matches_map = {s1: set() for s1 in val_s1_ids}
    for _, row in cand_df[cand_df["pred_match"]].iterrows():
        s1 = row["s1_id"]
        tgt = row["target_id"]
        if s1 in pred_matches_map:
            pred_matches_map[s1].add(tgt)

    # Evaluate entity level
    total_s1 = len(val_s1_ids)
    true_singletons = 0
    correct_singletons = 0
    multi_match_true = 0
    multi_match_pred = 0
    exact_entity_matches = 0

    all_tp = 0
    all_fp = 0
    all_fn = 0

    for s1 in val_s1_ids:
        true_set = ground_truth_map.get(s1, set())
        pred_set = pred_matches_map.get(s1, set())

        # Exact match of set
        if true_set == pred_set:
            exact_entity_matches += 1

        # Singleton evaluation
        if len(true_set) == 0:
            true_singletons += 1
            if len(pred_set) == 0:
                correct_singletons += 1

        # Multi-match evaluation
        if len(true_set) > 1:
            multi_match_true += 1
        if len(pred_set) > 1:
            multi_match_pred += 1

        # Pair counts
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        all_tp += tp
        all_fp += fp
        all_fn += fn

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f05 = compute_f_beta(precision, recall, beta=0.5)
    singleton_accuracy = correct_singletons / true_singletons if true_singletons > 0 else 1.0

    return {
        "total_s1": total_s1,
        "true_singletons": true_singletons,
        "correct_singletons": correct_singletons,
        "singleton_accuracy": singleton_accuracy,
        "exact_entity_accuracy": exact_entity_matches / max(total_s1, 1),
        "multi_match_true": multi_match_true,
        "multi_match_pred": multi_match_pred,
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
        "threshold": threshold
    }
