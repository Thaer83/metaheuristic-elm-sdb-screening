"""Appendix table from the old-data Wilcoxon rank-sum comparison (R1-C6)."""
from pathlib import Path
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "Results_comparison"
METRIC_KEYS = ["accuracy", "f1", "roc_auc"]


def _cell(diff, p):
    star = "*" if p < 0.05 else ""
    return f"{diff:+.3f}{star} (p={p:.3f})"


def _table(stats, dataset):
    sub = stats[stats.dataset == dataset]
    lines = [f"### {dataset}", "",
             "| Optimized | Baseline | dAccuracy | dF1 | dROC-AUC |",
             "|---|---|---|---|---|"]
    pivot = {}
    for _, r in sub.iterrows():
        pivot.setdefault((r.optimized, r.baseline), {})[r.metric] = (r.mean_diff, r["p_bh"])
    for (opt, base), m in pivot.items():
        cells = [_cell(*m[k]) for k in METRIC_KEYS]
        lines.append(f"| {opt} | {base} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")
    return "\n".join(lines)


def make_old_tables(out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    stats = pd.read_csv(out_dir / "comparison_old_stats.csv")
    md = ["# Statistical comparison (reported results): optimized ELM vs baselines (R1-C6)", "",
          "Wilcoxon rank-sum (Mann-Whitney) test on the reported per-run results "
          "(baselines from Results/, optimized ELM from Results_MHA/, 20 runs each). "
          "d = optimized mean minus baseline mean (positive favours the optimized ELM). "
          "p is Benjamini-Hochberg adjusted within each dataset and metric; * marks p < 0.05.", ""]
    for dataset in ["OSA", "SDB"]:
        md.append(_table(stats, dataset))
    (out_dir / "comparison_old_tables.md").write_text("\n".join(md), encoding="utf-8")
    print("old-data tables written", flush=True)


# --- compact matrix layout: one file per dataset, a 4x5 matrix per metric ---
from comparison import models   # noqa: E402

_METRIC_LABEL = {"roc_auc": "ROC-AUC", "f1": "F1-score",
                 "recall": "Sensitivity (Recall)", "precision": "Precision (PPV)",
                 "accuracy": "Accuracy"}
_MATRIX_ORDER = ["roc_auc", "f1", "recall", "precision", "accuracy"]


def _matrix(stats, dataset, metric):
    sub = stats[(stats.dataset == dataset) & (stats.metric == metric)]
    lookup = {(r.optimized, r.baseline): (r.mean_diff, r["p_bh"]) for _, r in sub.iterrows()}
    bases = models.BASELINES
    lines = [f"**{_METRIC_LABEL[metric]}**", "",
             "| Optimized | " + " | ".join(bases) + " |",
             "|---|" + "---|" * len(bases)]
    for opt in models.OPTIMIZED[dataset]:
        cells = []
        for base in bases:
            d, p = lookup[(opt, base)]
            cells.append(f"{d:+.3f}{'*' if p < 0.05 else ''}")
        lines.append(f"| **{opt}** | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def make_matrix_tables(out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    stats = pd.read_csv(out_dir / "comparison_old_stats.csv")
    for dataset in ["OSA", "SDB"]:
        md = [f"# Statistical comparison ({dataset}): optimized ELM vs key baselines (R1-C6)", "",
              "Wilcoxon rank-sum test on the per-run results, Benjamini-Hochberg corrected. "
              "Rows are the four best optimized ELMs, columns the five key baselines. Each cell is "
              "the mean-metric difference (optimized minus baseline); a positive value favours the "
              "optimized ELM and * marks adjusted p < 0.05.", ""]
        for metric in _MATRIX_ORDER:
            md.append(_matrix(stats, dataset, metric))
        (out_dir / f"comparison_old_{dataset}.md").write_text("\n".join(md), encoding="utf-8")
    print("per-dataset matrix tables written", flush=True)


_PANEL_LETTER = "abcde"
_OPT_PRETTY = {"CLPSO": "CL-PSO", "HIWOA": "HI-WOA"}


def make_appendix_b(out_dir=OUT_DIR):
    """Assemble Appendix B: Table B1 (OSA) and Table B2 (SDB), one multi-panel table per
    dataset, from the verified comparison_old_stats.csv."""
    out_dir = Path(out_dir)
    stats = pd.read_csv(out_dir / "comparison_old_stats.csv")
    lines = ["# Appendix B. Statistical Comparison of Optimized ELM Variants and Baselines", "",
             "We compared the four best optimized ELM variants with the five key baselines (SVM, "
             "MLP, LR, XGBoost, and the standard ELM) using a Wilcoxon rank-sum test over the 20 "
             "runs, with Benjamini-Hochberg correction within each dataset and metric. In every "
             "panel, a cell is the difference in the mean metric between an optimized ELM (row) and "
             "a baseline (column), computed as optimized minus baseline, so a positive value favours "
             "the optimized ELM and * marks an adjusted p-value below 0.05.", ""]
    caps = {
        "OSA": ("Table B1.", "MGO, RUN, CL-PSO, and GA", "OSA"),
        "SDB": ("Table B2.", "MEO, HI-WOA, HGS, and HHO", "SDB"),
    }
    for dataset in ["OSA", "SDB"]:
        tag, opt_list, ds = caps[dataset]
        lines += ["", f"**{tag}** Statistical comparison of the four best optimized ELM variants "
                  f"({opt_list}) against the five key baselines on the {ds} dataset "
                  f"(Wilcoxon rank-sum test, Benjamini-Hochberg corrected; * = adjusted p < 0.05).", ""]
        for i, metric in enumerate(_MATRIX_ORDER):
            panel = _matrix(stats, dataset, metric)
            panel = panel.replace(f"**{_METRIC_LABEL[metric]}**",
                                  f"*({_PANEL_LETTER[i]}) {_METRIC_LABEL[metric]}*", 1)
            for raw, pretty in _OPT_PRETTY.items():
                panel = panel.replace(f"**{raw}**", f"**{pretty}**")
            lines.append(panel)
    (out_dir / "appendix_B.md").write_text("\n".join(lines), encoding="utf-8")
    print("appendix_B.md written", flush=True)


if __name__ == "__main__":
    make_old_tables()
    make_matrix_tables()
    make_appendix_b()
