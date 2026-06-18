"""Paired Wilcoxon (each optimized variant vs each baseline) + Benjamini-Hochberg,
for both datasets. OSA from the re-run CSV; SDB reused from the imbalance None arm.
Reuses imbalance.stats.paired_test (note: paired_test(a, b) differences are b - a)."""
from pathlib import Path
import numpy as np
import pandas as pd
from imbalance.stats import paired_test, bh_adjust
from comparison import models

OUT_DIR = Path(__file__).resolve().parents[1] / "Results_comparison"
IMBALANCE_CSV = (Path(__file__).resolve().parents[1]
                 / "Results_imbalance_SDB" / "sdb_imbalance_all_runs.csv")
METRICS = ["accuracy", "f1", "roc_auc"]


def _load_runs(dataset, out_dir):
    if dataset == "OSA":
        return pd.read_csv(Path(out_dir) / "osa_comparison_all_runs.csv")
    sdb = pd.read_csv(IMBALANCE_CSV)
    return sdb[sdb["arm"] == "none"][["run", "model", *METRICS]].copy()


def compute_comparison_stats(out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    blocks = []
    for dataset in ["OSA", "SDB"]:
        df = _load_runs(dataset, out_dir)
        for metric in METRICS:
            rows = []
            for opt in models.OPTIMIZED[dataset]:
                ov = df[df.model == opt].sort_values("run")
                for base in models.BASELINES:
                    bv = df[df.model == base].sort_values("run")
                    assert np.array_equal(ov["run"].values, bv["run"].values), "runs must be paired"
                    md, _, _, p = paired_test(bv[metric].values, ov[metric].values)  # opt - base
                    rows.append({"dataset": dataset, "metric": metric,
                                 "optimized": opt, "baseline": base,
                                 "mean_opt": float(ov[metric].mean()),
                                 "mean_base": float(bv[metric].mean()),
                                 "mean_diff": md, "p_wilcoxon": p})
            bdf = pd.DataFrame(rows)
            bdf["p_bh"] = bh_adjust(bdf["p_wilcoxon"].values)
            blocks.append(bdf)
    res = pd.concat(blocks, ignore_index=True)
    res["significant_bh_0.05"] = res["p_bh"] < 0.05
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "comparison_stats.csv", index=False)
    return res
