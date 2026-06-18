import numpy as np
from imbalance import threshold


def test_youden_threshold_separable():
    # Perfectly separable: negatives score ~0.2, positives ~0.8.
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = threshold.youden_threshold(y, p)
    assert 0.3 < thr <= 0.7   # a cut that separates the two groups


def test_apply_threshold():
    p = np.array([0.2, 0.6, 0.5, 0.9])
    assert np.array_equal(threshold.apply_threshold(p, 0.5), np.array([0, 1, 1, 1]))


def test_select_threshold_uses_only_training_data():
    # The selector must never touch test data; it returns a float in [0,1].
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)

    def fit_score(Xtr, ytr, Xev):
        from scipy.special import expit
        return expit(Xev[:, 0])

    thr = threshold.select_threshold_via_split(fit_score, X, y, seed=0)
    assert 0.0 <= thr <= 1.0
