import numpy as np
import pandas as pd
from imbalance import stats


def test_paired_test_known_diff():
    import pytest
    none = np.array([0.5] * 20)
    arm = np.array([0.6] * 20)
    md, lo, hi, p = stats.paired_test(none, arm)
    assert md == pytest.approx(0.1)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)  # ~zero variance
    assert lo <= md <= hi
    assert 0.0 <= p <= 1.0     # a consistent +0.1 difference; p is small, just bound it here


def test_paired_test_identical_is_p1():
    v = np.array([0.7] * 20)
    md, lo, hi, p = stats.paired_test(v, v)
    assert md == 0.0 and p == 1.0


def test_bh_adjust_all_equal_case():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    adj = stats.bh_adjust(p)
    assert np.allclose(adj, 0.05)


def test_bh_adjust_preserves_order_and_bounds():
    p = np.array([0.9, 0.01])
    adj = stats.bh_adjust(p)
    assert adj[1] < adj[0]              # the small p stays the smaller adjusted p
    assert (adj >= 0).all() and (adj <= 1).all()


def test_compute_stats_schema(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for run in range(1, 21):
        for arm, base in [("none", 0.50), ("smote", 0.55), ("threshold", 0.50)]:
            rows.append({"run": run, "model": "LR", "arm": arm,
                         "roc_auc": base + rng.normal(0, 0.02),
                         "pr_auc": 0.78, "balanced_accuracy": base, "gmean": 0.4,
                         "sensitivity": 0.9, "specificity": 0.3, "f1": 0.8,
                         "mcc": 0.05, "accuracy": 0.7})
    csv = tmp_path / "all.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    res = stats.compute_stats(csv_path=csv, out_dir=tmp_path)
    for col in ["model", "arm", "metric", "mean_diff", "ci_low", "ci_high",
                "p_wilcoxon", "p_bh", "significant_bh_0.05"]:
        assert col in res.columns
    # smote roc_auc (+0.05) should come out significant; threshold roc_auc (same dist) should not
    smote_roc = res[(res.arm == "smote") & (res.metric == "roc_auc")].iloc[0]
    assert smote_roc["mean_diff"] > 0.03
