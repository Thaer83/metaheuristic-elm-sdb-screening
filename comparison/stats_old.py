"""Wilcoxon rank-sum (Mann-Whitney) comparison of each optimized ELM vs each key baseline,
using the ALREADY-GENERATED results (Results/ and Results_MHA/). The samples are treated as
independent (unpaired), which is the appropriate test here: the optimized-ELM runs vary the
optimizer seed on a single fixed split, while the baseline runs vary the data split, so the
runs are not paired. Benjamini-Hochberg correction within each (dataset, metric) family."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from imbalance.stats import bh_adjust
from comparison import models
from comparison.old_data import load_old, METRICS

OUT_DIR = Path(__file__).resolve().parents[1] / "Results_comparison"


def ranksum_test(a, b):
    """Wilcoxon rank-sum test on two independent samples.
    Returns (median_a, median_b, p) with a two-sided p for b versus a."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = float(ranksums(b, a).pvalue)
    return float(np.median(a)), float(np.median(b)), p


def compute_old_stats(out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    blocks = []
    for dataset in ["OSA", "SDB"]:
        df = load_old(dataset)
        for metric in METRICS:
            rows = []
            for opt in models.OPTIMIZED[dataset]:
                ov = df[df.model == opt][metric].values
                for base in models.BASELINES:
                    bv = df[df.model == base][metric].values
                    med_b, med_o, p = ranksum_test(bv, ov)   # median_base, median_opt, p
                    rows.append({"dataset": dataset, "metric": metric,
                                 "optimized": opt, "baseline": base,
                                 "mean_opt": float(np.mean(ov)),
                                 "mean_base": float(np.mean(bv)),
                                 "mean_diff": float(np.mean(ov) - np.mean(bv)),
                                 "median_opt": med_o, "median_base": med_b,
                                 "p_ranksum": p})
            bdf = pd.DataFrame(rows)
            bdf["p_bh"] = bh_adjust(bdf["p_ranksum"].values)
            blocks.append(bdf)
    res = pd.concat(blocks, ignore_index=True)
    res["significant_bh_0.05"] = res["p_bh"] < 0.05
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "comparison_old_stats.csv", index=False)
    return res
