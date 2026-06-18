"""Figures for the SDB imbalance experiment.

No figures are reported for this experiment. The earlier sensitivity-vs-specificity
scatter and balanced-accuracy bar chart were both built on specificity and balanced
accuracy, which are no longer reported in the paper (the reported metric set is
Accuracy, Precision/PPV, Sensitivity/Recall, F1, ROC-AUC, PR-AUC, MCC). The results
section presents Tables A and B only. This module is kept as a stub so the package
imports cleanly; it intentionally produces nothing.

The raw per-run metrics (including the unreported specificity, balanced accuracy and
G-mean) remain in Results_imbalance_SDB/sdb_imbalance_all_runs.csv for the record.
"""
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "Results_imbalance_SDB"


def make_figures(out_dir=OUT_DIR):
    """No-op: the imbalance experiment reports tables only, no figures."""
    print("no figures: the SDB imbalance experiment reports Tables A and B only "
          "(specificity / balanced accuracy are not reported)", flush=True)


if __name__ == "__main__":
    make_figures()
