"""Permutation importance as ROC-AUC drop, uniform across sklearn and ELM models."""
import numpy as np
from sklearn.metrics import roc_auc_score
from imbalance.scoring import positive_probability


def permutation_importance(model, X_test, y_test, kind, n_repeats=10, seed=0):
    """Return (importances, base_auc).

    importances[j] = base_auc - mean ROC-AUC after shuffling feature column j,
    averaged over n_repeats shuffles. Uses positive_probability so the ELM and
    sklearn models are scored the same way.
    """
    X_test = np.asarray(X_test, dtype=float)
    base = roc_auc_score(y_test, positive_probability(model, X_test, kind))
    rng = np.random.default_rng(seed)
    n_features = X_test.shape[1]
    imp = np.zeros(n_features)
    for j in range(n_features):
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            Xp = X_test.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[r] = base - roc_auc_score(y_test, positive_probability(model, Xp, kind))
        imp[j] = drops.mean()
    return imp, base
