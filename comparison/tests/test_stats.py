import numpy as np
import pandas as pd
from scipy import stats as sps
from imbalance.stats import paired_test, bh_adjust
from comparison import models
import comparison.stats as cs


def test_paired_test_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20)
    b = a + rng.normal(0.3, 0.1, size=20)
    md, lo, hi, p = paired_test(a, b)
    assert np.isclose(md, np.mean(b - a))                  # paired_test(a, b) differences are b - a
    assert np.isclose(p, sps.wilcoxon(b - a).pvalue)       # our p equals scipy's


def test_bh_known_values():
    assert np.allclose(bh_adjust([0.01, 0.02, 0.03, 0.04]), [0.04, 0.04, 0.04, 0.04])
    assert np.allclose(bh_adjust([0.001, 0.5, 0.5, 0.5]), [0.004, 0.5, 0.5, 0.5])


def test_known_difference_and_no_difference():
    a = np.full(20, 0.50)
    b = np.full(20, 0.55)
    md, lo, hi, p = paired_test(a, b)
    assert md > 0 and p < 0.05
    md0, lo0, hi0, p0 = paired_test(a, a)
    assert md0 == 0.0 and p0 == 1.0


def test_compute_comparison_stats_shape(tmp_path, monkeypatch):
    def fake_load(dataset, out_dir):
        rows = []
        for model in models.BASELINES + models.OPTIMIZED[dataset]:
            for run in range(1, 21):
                rows.append({"run": run, "model": model,
                             "accuracy": 0.7, "f1": 0.8, "roc_auc": 0.6})
        return pd.DataFrame(rows)
    monkeypatch.setattr(cs, "_load_runs", fake_load)
    res = cs.compute_comparison_stats(out_dir=tmp_path)
    # 2 datasets x 3 metrics x 4 optimized x 5 baselines = 120 contrasts
    assert len(res) == 120
    assert {"dataset", "metric", "optimized", "baseline", "mean_diff",
            "p_wilcoxon", "p_bh", "significant_bh_0.05"}.issubset(res.columns)
