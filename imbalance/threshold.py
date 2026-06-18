"""Decision-threshold optimization by Youden's J, selected without touching test data."""
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


def youden_threshold(y_true, y_score):
    """Threshold maximizing Youden's J = sensitivity + specificity - 1 (= tpr - fpr)."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def apply_threshold(y_score, thr):
    return (np.asarray(y_score) >= thr).astype(int)


def select_threshold_via_split(fit_score_fn, X_train, y_train, seed=0, val_size=0.25):
    """Carve a stratified validation slice from TRAIN, fit on the remainder via
    fit_score_fn(X_fit, y_fit, X_val) -> val scores, return the Youden threshold.
    fit_score_fn encapsulates training so this works for any model type."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=seed)
    val_scores = fit_score_fn(X_fit, y_fit, X_val)
    return youden_threshold(y_val, val_scores)
