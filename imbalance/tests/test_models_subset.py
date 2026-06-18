import numpy as np
from imbalance import models_subset as ms
from imbalance import scoring


def _toy():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (40, 10)), rng.normal(2, 1, (40, 10))])
    y = np.array([0] * 40 + [1] * 40)
    return X, y


def test_nine_models_present():
    assert set(ms.MODEL_NAMES) == {"SVM", "MLP", "XGB", "LR", "ELM",
                                   "MEO", "HIWOA", "HGS", "HHO"}


def test_class_weight_applicability():
    supported = {n for n in ms.MODEL_NAMES if ms.SPECS[n].supports_class_weight}
    assert supported == {"SVM", "LR", "XGB"}


def test_each_model_fits_and_scores_in_unit_interval():
    X, y = _toy()
    for name in ms.MODEL_NAMES:
        model, kind = ms.build(name, arm="none", seed=42)
        model.fit(X, y)
        p = scoring.positive_probability(model, X, kind=kind)
        assert p.shape == (len(y),)
        assert (p >= 0).all() and (p <= 1).all()
