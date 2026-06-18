"""Verification: every result row's metrics are consistent with its confusion matrix.

Independently recompute each threshold metric from the stored tn/fp/fn/tp counts
(formulas written from scratch here, NOT reusing imbalance.metrics) and assert it
matches the stored value. This is the academic-correctness check applied to the
actual experiment outputs."""
from pathlib import Path
import math
import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parents[1].parent / "Results_imbalance_SDB" / "sdb_imbalance_all_runs.csv"
TOL = 1e-9


def _safe(num, den):
    return num / den if den != 0 else 0.0


def test_every_row_metrics_match_confusion_matrix():
    assert CSV.exists(), f"run the experiment first; missing {CSV}"
    df = pd.read_csv(CSV)
    assert len(df) > 0
    bad = []
    for i, r in df.iterrows():
        tn, fp, fn, tp = int(r.tn), int(r.fp), int(r.fn), int(r.tp)
        sens = _safe(tp, tp + fn)
        spec = _safe(tn, tn + fp)
        prec = _safe(tp, tp + fp)
        acc = _safe(tp + tn, tp + tn + fp + fn)
        f1 = _safe(2 * tp, 2 * tp + fp + fn)
        bal = 0.5 * (sens + spec)
        gmean = math.sqrt(sens * spec)
        den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / den if den != 0 else 0.0
        checks = {
            "accuracy": acc, "precision": prec, "recall": sens, "sensitivity": sens,
            "specificity": spec, "f1": f1, "balanced_accuracy": bal, "gmean": gmean, "mcc": mcc,
        }
        for col, expected in checks.items():
            if abs(float(r[col]) - expected) > TOL:
                bad.append((i, r["model"], r["arm"], col, float(r[col]), expected))
    assert not bad, f"{len(bad)} metric/confusion-matrix mismatches, e.g. {bad[:3]}"


def test_total_sample_count_is_test_size():
    df = pd.read_csv(CSV)
    totals = df.tn + df.fp + df.fn + df.tp
    assert (totals == 100).all()   # SDB test split = 20% of 500
