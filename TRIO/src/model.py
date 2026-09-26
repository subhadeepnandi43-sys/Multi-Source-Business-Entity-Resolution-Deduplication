"""
Classification Model Module for Business Entity Resolution.
Supports LightGBM with automatic fallback to Scikit-Learn HistGradientBoostingClassifier.
Implements probability calibration, feature importance extraction, and model persistence.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import HistGradientBoostingClassifier


FEATURE_COLS = [
    "name_exact",
    "name_token_exact",
    "name_jaccard",
    "name_overlap",
    "name_common_tokens",
    "name_lev",
    "name_ngram2",
    "name_ngram3",
    "name_len_diff",
    "name_len_ratio",
    "addr_exact",
    "addr_jaccard",
    "addr_overlap",
    "addr_common_tokens",
    "addr_lev",
    "addr_ngram3",
    "addr_len_diff",
    "num_overlap",
    "num_common_count",
    "country_match",
    "country_conflict",
    "missing_s1_addr",
    "missing_c_addr",
    "is_source2",
    "is_source3",
    "composite_sim",
    "min_sim",
    "max_sim",
    "harmonic_sim",
]


class EntityResolutionClassifier:
    """
    Binary classifier predicting matching probability for (Source 1, Target) candidate pairs.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model_type = "lightgbm" if HAS_LIGHTGBM else "hist_gb"
        self.model = None
        self.feature_cols = FEATURE_COLS

    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None):
        """Train classifier with class weighting to handle imbalanced candidate pairs."""
        X_mat = X[self.feature_cols].fillna(0.0)

        # Compute balanced class weights or scale_pos_weight
        num_neg = np.sum(y == 0)
        num_pos = np.sum(y == 1)
        scale_pos = max(float(num_neg) / max(float(num_pos), 1.0), 1.0)
        # Cap scale_pos to avoid over-predicting false positives
        scale_pos = min(scale_pos, 5.0)

        if self.model_type == "lightgbm":
            print(f"[MODEL] Training LightGBM Classifier (pos_scale={scale_pos:.2f})...")
            self.model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.04,
                num_leaves=31,
                max_depth=6,
                min_child_samples=10,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1
            )
            if eval_set is not None:
                X_val, y_val = eval_set
                X_val_mat = X_val[self.feature_cols].fillna(0.0)
                self.model.fit(
                    X_mat, y,
                    eval_set=[(X_val_mat, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
                )
            else:
                self.model.fit(X_mat, y)
        else:
            print("[MODEL] Training HistGradientBoostingClassifier (fallback)...")
            self.model = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.04,
                max_leaf_nodes=31,
                min_samples_leaf=10,
                random_state=self.random_state,
                class_weight="balanced"
            )
            self.model.fit(X_mat, y)

        print("[MODEL] Training complete.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of match (column 1)."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        X_mat = X[self.feature_cols].fillna(0.0)
        probs = self.model.predict_proba(X_mat)
        return probs[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importance dictionary."""
        if self.model is None:
            return {}
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            total = sum(importances) + 1e-9
            return {col: float(imp / total) for col, imp in zip(self.feature_cols, importances)}
        return {col: 1.0 / len(self.feature_cols) for col in self.feature_cols}

    def save(self, filepath: str):
        """Save model object to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "EntityResolutionClassifier":
        """Load model object from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
