import numpy as np
from sklearn.linear_model import LogisticRegression
from importance.permutation import permutation_importance


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    n = 300
    x0 = rng.normal(size=n)             # predictive
    noise = rng.normal(size=(n, 3))     # irrelevant
    y = (x0 > 0).astype(int)
    X = np.column_stack([x0, noise])
    return X, y


def test_predictive_feature_ranks_top():
    X, y = _toy()
    m = LogisticRegression().fit(X, y)
    imp, base = permutation_importance(m, X, y, kind="sklearn_proba", n_repeats=10, seed=1)
    assert base > 0.95
    assert imp.argmax() == 0
    assert imp[0] > 0.1
    assert np.all(np.abs(imp[1:]) < 0.05)


def test_determinism():
    X, y = _toy()
    m = LogisticRegression().fit(X, y)
    a, _ = permutation_importance(m, X, y, "sklearn_proba", n_repeats=5, seed=7)
    b, _ = permutation_importance(m, X, y, "sklearn_proba", n_repeats=5, seed=7)
    assert np.allclose(a, b)


def test_constant_feature_zero():
    X, y = _toy()
    X = np.column_stack([X, np.ones(len(y))])   # constant column -> shuffling is a no-op
    m = LogisticRegression().fit(X, y)
    imp, _ = permutation_importance(m, X, y, "sklearn_proba", n_repeats=5, seed=3)
    assert abs(imp[-1]) < 1e-9
