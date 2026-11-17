"""Training-only feature selection components.

All selectors are scikit-learn transformers placed *inside* model pipelines,
so during cross-validation and final fitting they only ever see the training
portion of the data (no validation/test target is observed while fitting).

Selection chain used by the ``full_selected`` configuration:

1. ``VarianceThreshold``   - removes (near-)constant features.
2. ``CorrelationFilter``   - removes features whose absolute Pearson
   correlation with an already-kept feature exceeds a threshold (redundancy).
3. ``TopKSelector``        - keeps the k features with the highest univariate
   F-statistic (``f_regression``) against the training target; k is tuned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin, VarianceThreshold, f_regression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class CorrelationFilter(SelectorMixin, BaseEstimator):
    """Greedy redundancy filter on training data (keeps first of a correlated pair)."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(X, rowvar=False)) if X.shape[1] > 1 else np.ones((1, 1))
        corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0)
        keep: list[int] = []
        for j in range(X.shape[1]):
            if all(corr[j, k] <= self.threshold for k in keep):
                keep.append(j)
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[keep] = True
        self.support_ = mask
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_


class TopKSelector(SelectorMixin, BaseEstimator):
    """Keep the ``k`` features with the largest f_regression score.

    ``k`` larger than the available number of features (or ``"all"``) keeps all
    features, which makes the selector safe inside a randomized search.
    """

    def __init__(self, k: int | str = 20):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        scores, _ = f_regression(X, np.asarray(y, dtype=float))
        self.scores_ = np.nan_to_num(scores, nan=0.0)
        k = X.shape[1] if self.k == "all" else min(int(self.k), X.shape[1])
        order = np.argsort(-self.scores_, kind="mergesort")
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[order[:k]] = True
        self.support_ = mask
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_


def make_selector(k: int | str = 20, corr_threshold: float = 0.95) -> Pipeline:
    return Pipeline([("variance", VarianceThreshold(1e-8)),
                     ("redundancy", CorrelationFilter(corr_threshold)),
                     ("topk", TopKSelector(k))])


def selected_feature_names(pipeline: Pipeline, input_features: list[str]) -> list[str]:
    """Names of the features that reach the estimator of a fitted pipeline."""
    names = np.asarray(input_features)
    if not hasattr(pipeline, "named_steps") or "select" not in pipeline.named_steps:
        return list(names)
    sel = pipeline.named_steps["select"]
    for _, step in sel.steps:
        names = names[step.get_support()]
    return list(names)


def training_correlation_table(train: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """Pearson/Spearman correlation of each feature with the target (TRAIN ONLY)."""
    rows = []
    for f in features:
        s = train[[f, target]].dropna()
        if len(s) < 3 or s[f].std() == 0:
            rows.append({"feature": f, "pearson": np.nan, "spearman": np.nan, "n": len(s)})
            continue
        rows.append({"feature": f, "pearson": s[f].corr(s[target]),
                     "spearman": s[f].corr(s[target], method="spearman"), "n": len(s)})
    return pd.DataFrame(rows).sort_values("pearson", key=lambda x: -x.abs())
