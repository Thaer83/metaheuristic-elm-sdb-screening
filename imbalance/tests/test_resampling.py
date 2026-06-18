import numpy as np
from imbalance import resampling


def _imbalanced():
    # Overlapping classes (like real imbalanced clinical data) so ADASYN's density
    # rule has majority neighbours to work with; SMOTE works on this too.
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.0, 1.0, (80, 4)), rng.normal(1.2, 1.0, (20, 4))])
    y = np.array([0] * 80 + [1] * 20)
    return X, y


def test_smote_balances_training_set():
    X, y = _imbalanced()
    Xr, yr = resampling.resample_train(X, y, method="smote", seed=42)
    assert (yr == 1).sum() == (yr == 0).sum()       # fully balanced
    assert len(Xr) == len(yr) >= len(X)             # minority oversampled


def test_smote_only_adds_minority():
    X, y = _imbalanced()
    Xr, yr = resampling.resample_train(X, y, method="smote", seed=42)
    assert (yr == 0).sum() == (y == 0).sum()        # majority unchanged
    assert (yr == 1).sum() > (y == 1).sum()         # minority grew


def test_adasyn_oversamples_minority_train_only():
    X, y = _imbalanced()
    Xr, yr = resampling.resample_train(X, y, method="adasyn", seed=42)
    assert (yr == 0).sum() == (y == 0).sum()        # majority unchanged
    assert (yr == 1).sum() > (y == 1).sum()         # minority grew (ADASYN: ~balanced, density-weighted)
    assert len(Xr) == len(yr) > len(X)


def test_none_method_is_identity():
    X, y = _imbalanced()
    Xr, yr = resampling.resample_train(X, y, method="none", seed=42)
    assert np.array_equal(Xr, X) and np.array_equal(yr, y)
