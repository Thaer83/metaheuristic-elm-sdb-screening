"""Extract a positive-class probability in [0, 1] from each model type.
sklearn -> predict_proba[:, 1]; IntelELM ELMs -> softmax of the raw 2-col output."""
import numpy as np


def softmax_rows(raw):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 1:                      # degenerate; treat as logit for class 1
        raw = np.column_stack([np.zeros_like(raw), raw])
    z = raw - raw.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def positive_probability(model, X, kind):
    """kind in {'sklearn_proba', 'elm'} -> shape (n,) probability of class 1."""
    if kind == "sklearn_proba":
        return np.asarray(model.predict_proba(X))[:, 1]
    if kind == "elm":
        raw = np.asarray(model.predict(X, return_prob=True))
        return softmax_rows(raw)[:, 1]
    raise ValueError(f"unknown scoring kind: {kind}")


def positive_raw_score_elm(model, X):
    """Alternative raw positive score (ELM raw column 1), for verification only.
    NOTE: the original paper computed ELM ROC-AUC from HARD LABELS, not this raw
    score; this experiment uses softmax probabilities (see positive_probability)."""
    raw = np.asarray(model.predict(X, return_prob=True))
    return raw[:, 1] if raw.ndim == 2 else np.asarray(raw, dtype=float).ravel()
