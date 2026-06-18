"""Appendix comparison table per dataset (optimized vs baseline, three metrics)."""
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


def make_tables(out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    stats = pd.read_csv(out_dir / "comparison_stats.csv")
    md = ["# Statistical comparison: optimized ELM vs baselines (R1-C6)", "",
          "Paired Wilcoxon signed-rank test over 20 shared stratified splits. "
          "d = optimized minus baseline (positive favours the optimized ELM). "
          "p is Benjamini-Hochberg adjusted within each dataset and metric; * marks p < 0.05.", ""]
    for dataset in ["OSA", "SDB"]:
        md.append(_table(stats, dataset))
    (out_dir / "comparison_tables.md").write_text("\n".join(md), encoding="utf-8")
    print("tables written", flush=True)


if __name__ == "__main__":
    make_tables()
