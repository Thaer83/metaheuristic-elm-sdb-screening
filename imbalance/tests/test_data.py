import numpy as np
from imbalance import data


def test_load_sdb_shapes_and_balance():
    X, y = data.load_sdb()
    assert X.shape[0] == y.shape[0] == 500
    assert X.shape[1] == 10
    assert set(np.unique(y)) == {0, 1}
    # SDB: 381 positive / 119 negative (paper Table 1)
    assert int((y == 1).sum()) == 381
    assert int((y == 0).sum()) == 119


def test_run_splits_count_and_seeds():
    X, y = data.load_sdb()
    splits = data.run_splits(X, y)
    assert len(splits) == 20
    assert [s["seed"] for s in splits] == list(range(42, 62))


def test_splits_are_deterministic():
    X, y = data.load_sdb()
    a = data.run_splits(X, y)[0]["test_idx"]
    b = data.run_splits(X, y)[0]["test_idx"]
    assert np.array_equal(a, b)


def test_train_test_disjoint_and_stratified():
    X, y = data.load_sdb()
    s = data.run_splits(X, y)[0]
    tr, te = set(s["train_idx"].tolist()), set(s["test_idx"].tolist())
    assert tr.isdisjoint(te)
    assert len(tr) + len(te) == 500
    assert len(te) == 100                                   # 20% of 500
    assert abs(y[s["test_idx"]].mean() - y.mean()) < 0.03   # stratified
