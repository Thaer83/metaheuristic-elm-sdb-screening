"""Paired significance tests for arm-vs-None comparisons on the SDB imbalance results.

Each of the 20 runs is a shared stratified split, so arm and None are PAIRED per run.
For every (model, arm, metric) we report the mean difference, a 95% CI on the paired
differences, a Wilcoxon signed-rank p-value, and a Benjamini-Hochberg-adjusted p across
the whole family of contrasts. This backs the honest framing (review H2) and seeds the
later R1-C6 statistical-comparison experiment."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path(__file__).resolve().parents[1] / "Results_imbalance_SDB"
METRICS = ["roc_auc", "pr_auc", "balanced_accuracy", "gmean",
           "sensitivity", "specificity", "f1", "mcc", "accuracy"]
ARMS = ["smote", "adasyn", "classweight", "threshold"]


def paired_test(none_vals, arm_vals):
    """Return (mean_diff, ci_low, ci_high, p) for paired arm-vs-None differences."""
    d = np.asarray(arm_vals, float) - np.asarray(none_vals, float)
    n = len(d)
    mean_diff = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    tcrit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    ci_low, ci_high = mean_diff - tcrit * se, mean_diff + tcrit * se
    if np.allclose(d, 0.0):
        p = 1.0
    else:
        try:
            p = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
        except ValueError:
            p = 1.0
    return mean_diff, ci_low, ci_high, p


def bh_adjust(pvals):
    """Benjamini-Hochberg step-up adjusted p-values (preserves input order)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]   # enforce monotonicity
    adj = np.empty(m)
    adj[order] = np.clip(ranked, 0, 1)
    return adj


def compute_stats(csv_path=None, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    csv_path = Path(csv_path) if csv_path else out_dir / "sdb_imbalance_all_runs.csv"
    df = pd.read_csv(csv_path)
    rows = []
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        none = sub[sub["arm"] == "none"].sort_values("run")
        for arm in ARMS:
            a = sub[sub["arm"] == arm].sort_values("run")
            if len(a) == 0:
                continue
            assert np.array_equal(none["run"].values, a["run"].values), "runs must be paired"
            for met in METRICS:
                nv, av = none[met].values, a[met].values
                if np.isnan(nv).any() or np.isnan(av).any():
                    continue
                md, lo, hi, p = paired_test(nv, av)
                rows.append({"model": model, "arm": arm, "metric": met,
                             "mean_none": float(np.mean(nv)), "mean_arm": float(np.mean(av)),
                             "mean_diff": md, "ci_low": lo, "ci_high": hi, "p_wilcoxon": p})
    res = pd.DataFrame(rows)
    res["p_bh"] = bh_adjust(res["p_wilcoxon"].values)
    res["significant_bh_0.05"] = res["p_bh"] < 0.05
    res.to_csv(out_dir / "sdb_imbalance_stats.csv", index=False)
    return res


if __name__ == "__main__":
    r = compute_stats()
    print(f"wrote {len(r)} contrasts; {int(r['significant_bh_0.05'].sum())} significant after BH")
