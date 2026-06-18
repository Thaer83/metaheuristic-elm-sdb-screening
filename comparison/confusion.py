"""Confusion matrices (averaged over the 20 runs) for the four best optimized ELM models
per dataset, for the supplement. Counts come straight from the reported per-run results in
Results_MHA. Because the test set is fixed, accuracy and recall computed from the averaged
matrix reproduce Tables 8 and 9 exactly; precision and F1 match to within rounding."""
from pathlib import Path
import pandas as pd
from comparison import models

_ROOT = Path(__file__).resolve().parents[1]
MHA = {"OSA": _ROOT / "Results_MHA" / "Results OSA" / "sleep_apnea_test_all_runs.csv",
       "SDB": _ROOT / "Results_MHA" / "Results SDB" / "sleep_apnea_test_all_runs.csv"}
OUT_DIR = _ROOT / "Results_comparison"
_PRETTY = {"CLPSO": "CL-PSO", "HIWOA": "HI-WOA"}
_TEST_N = {"OSA": 55, "SDB": 100}


def mean_confusion(dataset):
    """Return a DataFrame [model, TP, FN, FP, TN, accuracy, precision, recall, f1] for the
    best-4 optimized ELMs, with cells averaged over the 20 runs and metrics derived from them."""
    df = pd.read_csv(MHA[dataset]).rename(columns={"algo": "model"})
    rows = []
    for m in models.OPTIMIZED[dataset]:
        s = df[df.model == m]
        tp, fn, fp, tn = s.tp.mean(), s.fn.mean(), s.fp.mean(), s.tn.mean()
        n = tp + fn + fp + tn
        acc = (tp + tn) / n
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        rows.append({"model": m, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                     "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
    return pd.DataFrame(rows)


def make_confusion_supplement(out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    lines = ["# Supplement: Confusion matrices for the four best metaheuristic-based ELM models", "",
             "Each matrix reports the counts averaged over the 20 runs (test set n = 55 for OSA and "
             "n = 100 for SDB). The accuracy, precision, recall, and F1-score computed from these "
             "averaged counts reproduce the values in Tables 8 and 9 (accuracy and recall exactly; "
             "precision and F1 to within rounding)."]
    for ds in ["OSA", "SDB"]:
        lines += ["", f"## {ds} (test n = {_TEST_N[ds]})", ""]
        for _, r in mean_confusion(ds).iterrows():
            name = _PRETTY.get(r.model, r.model)
            lines += [f"**{name}**", "",
                      "| Actual \\\\ Predicted | Positive | Negative |",
                      "|---|---|---|",
                      f"| Positive | TP = {r.TP:.1f} | FN = {r.FN:.1f} |",
                      f"| Negative | FP = {r.FP:.1f} | TN = {r.TN:.1f} |",
                      "",
                      f"Derived: Accuracy = {r.accuracy:.4f}, Precision (PPV) = {r.precision:.4f}, "
                      f"Sensitivity (Recall) = {r.recall:.4f}, F1 = {r.f1:.4f}.", ""]
    (out_dir / "supplement_confusion_matrices.md").write_text("\n".join(lines), encoding="utf-8")
    print("confusion-matrix supplement written", flush=True)


def make_confusion_figures(out_dir=OUT_DIR):
    """One figure per dataset: a 2x2 grid of confusion-matrix heatmaps for the best-4 models,
    cells annotated with the counts averaged over the 20 runs."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    for ds in ["OSA", "SDB"]:
        cm = mean_confusion(ds)
        fig, axes = plt.subplots(2, 2, figsize=(8, 7))
        for ax, (_, r) in zip(axes.ravel(), cm.iterrows()):
            mat = np.array([[r.TP, r.FN], [r.FP, r.TN]])
            ax.imshow(mat, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Positive", "Negative"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["Positive", "Negative"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            thr = mat.max() / 2.0
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                            color="white" if mat[i, j] > thr else "black", fontsize=12)
            ax.set_title(_PRETTY.get(r.model, r.model))
        fig.suptitle(f"Confusion matrices on {ds} (mean over 20 runs, test n = {_TEST_N[ds]})")
        plt.tight_layout()
        plt.savefig(out_dir / f"confusion_{ds}.png", dpi=300)
        plt.close()
    print("confusion-matrix figures written", flush=True)


if __name__ == "__main__":
    make_confusion_supplement()
    make_confusion_figures()
