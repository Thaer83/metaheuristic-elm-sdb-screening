from imbalance import run_sdb_imbalance as drv

EXPECTED_COLS = ["run", "seed", "model", "arm", "accuracy", "sensitivity",
                 "specificity", "balanced_accuracy", "gmean", "mcc",
                 "roc_auc", "pr_auc", "brier", "tn", "fp", "fn", "tp"]


def test_run_one_returns_expected_schema():
    X, y = drv.data.load_sdb()
    split = drv.data.run_splits(X, y)[0]
    row = drv.run_one(X, y, split, model="LR", arm="none")
    for col in EXPECTED_COLS:
        assert col in row, col
    assert row["model"] == "LR" and row["arm"] == "none"
    assert 0.0 <= row["roc_auc"] <= 1.0


def test_threshold_arm_changes_classification():
    X, y = drv.data.load_sdb()
    split = drv.data.run_splits(X, y)[0]
    none = drv.run_one(X, y, split, model="LR", arm="none")
    thr = drv.run_one(X, y, split, model="LR", arm="threshold")
    # same model family, different operating point -> sens or spec should differ
    assert thr["sensitivity"] != none["sensitivity"] or thr["specificity"] != none["specificity"]


def test_threshold_arm_roc_matches_none():
    # M1a: Threshold arm refits on full train, so its rank metrics must equal None's.
    X, y = drv.data.load_sdb()
    split = drv.data.run_splits(X, y)[0]
    none = drv.run_one(X, y, split, model="LR", arm="none")
    thr = drv.run_one(X, y, split, model="LR", arm="threshold")
    assert abs(thr["roc_auc"] - none["roc_auc"]) < 1e-9
    assert abs(thr["pr_auc"] - none["pr_auc"]) < 1e-9
    assert abs(thr["brier"] - none["brier"]) < 1e-9


def test_run_one_works_for_elm_and_smote():
    X, y = drv.data.load_sdb()
    split = drv.data.run_splits(X, y)[0]
    row = drv.run_one(X, y, split, model="ELM", arm="smote")
    assert row["model"] == "ELM" and row["arm"] == "smote"
    assert 0.0 <= row["roc_auc"] <= 1.0
    assert 0.0 <= row["brier"] <= 1.0


def test_run_one_works_for_adasyn():
    X, y = drv.data.load_sdb()
    split = drv.data.run_splits(X, y)[0]
    row = drv.run_one(X, y, split, model="LR", arm="adasyn")
    assert row["model"] == "LR" and row["arm"] == "adasyn"
    assert 0.0 <= row["roc_auc"] <= 1.0
    assert (row["tn"] + row["fp"] + row["fn"] + row["tp"]) == 100   # test set untouched


def test_classweight_not_applicable_returns_none():
    X, y = drv.data.load_sdb()
    split = drv.data.run_splits(X, y)[0]
    assert drv.run_one(X, y, split, model="ELM", arm="classweight") is None


def test_write_summary_flattens_columns_and_writes(tmp_path):
    import pandas as pd
    df = pd.DataFrame([
        {"run": 1, "seed": 42, "model": "LR", "arm": "none",
         "accuracy": 0.7, "recall": 0.9, "specificity": 0.4, "roc_auc": 0.6},
        {"run": 2, "seed": 43, "model": "LR", "arm": "none",
         "accuracy": 0.8, "recall": 0.95, "specificity": 0.5, "roc_auc": 0.65},
    ])
    path = tmp_path / "summary.xlsx"
    summary = drv._write_summary(df, path)
    assert path.exists()
    # flattened single-level columns, no MultiIndex
    assert "accuracy_mean" in summary.columns and "accuracy_std" in summary.columns
    assert "model" in summary.columns and "arm" in summary.columns
    back = pd.read_excel(path, sheet_name="summary")
    assert "accuracy_mean" in back.columns
