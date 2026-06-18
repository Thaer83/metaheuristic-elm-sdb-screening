"""Classification metrics computed from the confusion matrix, each cross-checked
against scikit-learn. Used by the SDB imbalance experiment."""
import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
)


def confusion_counts(y_true, y_pred):
    """Return (tn, fp, fn, tp) for binary labels in {0, 1}."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _safe_div(num, den):
    return float(num) / float(den) if den != 0 else 0.0


def threshold_metrics(y_true, y_pred):
    """All threshold-dependent metrics, computed purely from the confusion matrix."""
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    sens = _safe_div(tp, tp + fn)            # recall / TPR
    spec = _safe_div(tn, tn + fp)            # TNR
    prec = _safe_div(tp, tp + fp)            # PPV (sklearn zero_division=0 convention -> 0.0)
    # NPV is genuinely UNDEFINED (not 0) when the model predicts no negatives (tn+fn==0);
    # report NaN so it is not misread as "all negative predictions are wrong".
    npv = float("nan") if (tn + fn) == 0 else tn / (tn + fn)
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float(((tp * tn) - (fp * fn)) / mcc_den) if mcc_den != 0 else 0.0
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "precision": prec,
        "recall": sens,
        "sensitivity": sens,
        "specificity": spec,
        "npv": npv,
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": 0.5 * (sens + spec),
        "gmean": float(np.sqrt(sens * spec)),
        "mcc": mcc,
    }


def ranking_metrics(y_true, y_score):
    """Rank-based metrics on a continuous positive-class score."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def probability_metrics(y_true, y_prob):
    """Probability-quality metric; y_prob must be in [0, 1]."""
    return {"brier": float(brier_score_loss(y_true, y_prob))}


def verify_threshold_metrics(y_true, y_pred, tol=1e-9):
    """Assert every confusion-matrix metric equals the scikit-learn reference.
    Raises AssertionError on any mismatch. Both classes must appear in y_true."""
    m = threshold_metrics(y_true, y_pred)
    assert abs(m["accuracy"] - accuracy_score(y_true, y_pred)) < tol, "accuracy"
    assert abs(m["precision"] - precision_score(y_true, y_pred, zero_division=0)) < tol, "precision"
    assert abs(m["recall"] - recall_score(y_true, y_pred, zero_division=0)) < tol, "recall"
    assert abs(m["specificity"] - recall_score(y_true, y_pred, pos_label=0, zero_division=0)) < tol, "specificity"
    assert abs(m["f1"] - f1_score(y_true, y_pred, zero_division=0)) < tol, "f1"
    assert abs(m["balanced_accuracy"] - balanced_accuracy_score(y_true, y_pred)) < tol, "balanced_accuracy"
    assert abs(m["mcc"] - matthews_corrcoef(y_true, y_pred)) < tol, "mcc"
    return m
