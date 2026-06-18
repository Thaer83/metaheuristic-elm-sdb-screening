import numpy as np
from imbalance import scoring


def test_softmax_rows_are_probabilities():
    raw = np.array([[2.0, 1.0], [-1.0, 3.0], [0.0, 0.0]])
    p = scoring.softmax_rows(raw)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all() and (p <= 1).all()


def test_softmax_positive_is_monotonic_in_difference():
    raw = np.array([[0.0, -1.0], [0.0, 0.0], [0.0, 2.0]])  # s1-s0 increasing
    pos = scoring.softmax_rows(raw)[:, 1]
    assert pos[0] < pos[1] < pos[2]


class _FakeSklearn:
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])


class _FakeElm:
    def predict(self, X, return_prob=False):
        raw = np.tile([1.0, 2.0], (len(X), 1))
        return raw if return_prob else raw.argmax(axis=1)


def test_positive_probability_sklearn():
    p = scoring.positive_probability(_FakeSklearn(), np.zeros((4, 2)), kind="sklearn_proba")
    assert np.allclose(p, 0.7)


def test_positive_probability_elm_is_softmax():
    p = scoring.positive_probability(_FakeElm(), np.zeros((4, 2)), kind="elm")
    expected = np.exp(2.0) / (np.exp(1.0) + np.exp(2.0))
    assert np.allclose(p, expected)
    assert (p >= 0).all() and (p <= 1).all()


def test_positive_raw_score_elm_is_column_one():
    s = scoring.positive_raw_score_elm(_FakeElm(), np.zeros((3, 2)))
    assert np.allclose(s, 2.0)
