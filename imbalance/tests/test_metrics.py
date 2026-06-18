import numpy as np
import pytest
from imbalance import metrics

# Hand-computed example: 4 positives, 6 negatives.
# y_true = 1,1,1,1,0,0,0,0,0,0 ; y_pred = 1,1,1,0,0,0,0,0,1,1
# => TP=3, FN=1, TN=4, FP=2
Y_TRUE = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
Y_PRED = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])


def test_confusion_counts():
    tn, fp, fn, tp = metrics.confusion_counts(Y_TRUE, Y_PRED)
    assert (tn, fp, fn, tp) == (4, 2, 1, 3)


def test_hand_computed_values():
    m = metrics.threshold_metrics(Y_TRUE, Y_PRED)
    assert m["sensitivity"] == pytest.approx(3 / 4)
    assert m["specificity"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(3 / 5)
    assert m["npv"] == pytest.approx(4 / 5)
    assert m["accuracy"] == pytest.approx(7 / 10)
    assert m["f1"] == pytest.approx(2 * 3 / (2 * 3 + 2 + 1))
    assert m["balanced_accuracy"] == pytest.approx((3 / 4 + 4 / 6) / 2)
    assert m["gmean"] == pytest.approx(np.sqrt((3 / 4) * (4 / 6)))
    assert m["mcc"] == pytest.approx((3 * 4 - 2 * 1) / np.sqrt(5 * 4 * 6 * 5))


def test_cross_check_against_sklearn_passes():
    # The verifier asserts every count-based metric equals its sklearn implementation.
    metrics.verify_threshold_metrics(Y_TRUE, Y_PRED)


def test_cross_check_random_cases():
    rng = np.random.default_rng(0)
    for _ in range(50):
        yt = rng.integers(0, 2, size=40)
        yp = rng.integers(0, 2, size=40)
        if len(np.unique(yt)) < 2:   # verifier needs both classes present
            continue
        metrics.verify_threshold_metrics(yt, yp)


def test_safe_division_no_crash():
    # All predicted negative -> tp=fp=0; precision/gmean must be 0.0, not NaN/crash.
    yt = np.array([1, 0, 1, 0])
    yp = np.array([0, 0, 0, 0])
    m = metrics.threshold_metrics(yt, yp)
    assert m["precision"] == 0.0
    assert m["gmean"] == 0.0


def test_npv_is_nan_when_no_negative_predictions():
    # Model predicts ALL positive -> tn+fn = 0 -> NPV is undefined, must be NaN (not 0.0).
    yt = np.array([1, 1, 0, 0])
    yp = np.array([1, 1, 1, 1])
    m = metrics.threshold_metrics(yt, yp)
    assert np.isnan(m["npv"])
    # the defined metrics are unaffected
    assert m["sensitivity"] == pytest.approx(1.0)
    assert m["specificity"] == pytest.approx(0.0)


def test_ranking_and_probability_metrics():
    yt = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.35, 0.8])
    r = metrics.ranking_metrics(yt, score)
    assert 0.0 <= r["roc_auc"] <= 1.0
    assert 0.0 <= r["pr_auc"] <= 1.0
    prob = np.array([0.1, 0.4, 0.35, 0.8])
    b = metrics.probability_metrics(yt, prob)
    assert b["brier"] == pytest.approx(np.mean((prob - yt) ** 2))
