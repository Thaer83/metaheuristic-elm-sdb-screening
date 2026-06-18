import numpy as np
from scipy import stats as sps
from comparison import models
from comparison.old_data import load_old, METRICS
from comparison.stats_old import ranksum_test, compute_old_stats


def test_load_old_models_and_runs():
    for dataset in ["OSA", "SDB"]:
        df = load_old(dataset)
        expected = set(models.BASELINES) | set(models.OPTIMIZED[dataset])
        assert set(df["model"].unique()) == expected
        assert set(df.groupby("model").size().unique()) == {20}
        assert {"run", "model", *METRICS}.issubset(df.columns)


def test_ranksum_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, size=20)
    b = rng.normal(0.8, 1.0, size=20)
    med_a, med_b, p = ranksum_test(a, b)
    assert np.isclose(med_a, np.median(a)) and np.isclose(med_b, np.median(b))
    assert np.isclose(p, sps.ranksums(b, a).pvalue)


def test_ranksum_known_difference():
    a = np.full(20, 0.65) + np.linspace(-0.01, 0.01, 20)
    b = np.full(20, 0.73) + np.linspace(-0.01, 0.01, 20)
    _, _, p = ranksum_test(a, b)
    assert p < 0.05
    _, _, p_same = ranksum_test(a, a)
    assert p_same > 0.99


def test_compute_old_stats_shape(tmp_path):
    res = compute_old_stats(out_dir=tmp_path)
    # 2 datasets x 5 metrics x 4 optimized x 5 baselines = 200 contrasts
    assert len(res) == 200
    assert {"dataset", "metric", "optimized", "baseline", "mean_diff",
            "p_ranksum", "p_bh", "significant_bh_0.05"}.issubset(res.columns)
