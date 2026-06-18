import numpy as np
from sklearn.linear_model import LogisticRegression
from importance.shap_importance import shap_values_for, mean_abs_shap


def test_shap_shapes_and_dead_feature():
    rng = np.random.default_rng(0)
    n = 150
    x0 = rng.normal(size=n)             # predictive
    dead = np.zeros(n)                  # never varies
    noise = rng.normal(size=n)          # irrelevant
    y = (x0 > 0).astype(int)
    X = np.column_stack([x0, dead, noise])
    m = LogisticRegression().fit(X, y)
    sv = shap_values_for(m, X, X[:20], kind="sklearn_proba", seed=0, n_background=20)
    assert sv.shape == (20, 3)
    mabs = mean_abs_shap(sv)
    assert mabs.shape == (3,)
    assert mabs[1] < 1e-6        # dead feature contributes nothing
    assert mabs.argmax() == 0    # predictive feature dominates
