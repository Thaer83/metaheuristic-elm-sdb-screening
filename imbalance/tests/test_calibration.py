import numpy as np
from imbalance import calibration


def test_calibration_points_shape_and_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    frac_pos, mean_pred = calibration.calibration_points(y, p, n_bins=10)
    assert len(frac_pos) == len(mean_pred)
    assert ((frac_pos >= 0) & (frac_pos <= 1)).all()
    assert ((mean_pred >= 0) & (mean_pred <= 1)).all()


def test_brier_matches_definition():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.2, 0.7, 0.6, 0.3])
    assert calibration.brier(y, p) == np.mean((p - y) ** 2)
