"""Ranking tables (permutation vs SHAP) with PSG-derived features flagged."""
from pathlib import Path
import pandas as pd
from importance.data import SDB_PSG_FEATURES, OSA_PSG_FEATURES

OUT_DIR = Path(__file__).resolve().parents[1] / "Results_importance"
PSG = {"SDB": set(SDB_PSG_FEATURES), "OSA": set(OSA_PSG_FEATURES)}


def _rank_table(dataset, out_dir):
    perm = pd.read_csv(out_dir / f"{dataset.lower()}_permutation_importance.csv")
    shp = pd.read_csv(out_dir / f"{dataset.lower()}_shap_importance.csv")
    lines = [f"## {dataset}", ""]
    for m in perm["model"].unique():
        p = perm[perm.model == m].copy()
        p["perm_rank"] = p["mean_importance"].rank(ascending=False, method="min").astype(int)
        s = shp[shp.model == m][["feature", "mean_abs_shap"]].copy()
        s["shap_rank"] = s["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
        merged = p.merge(s, on="feature", how="left").sort_values("perm_rank")
        base = p["mean_base_auc"].iloc[0]
        lines += [f"### {dataset} - {m} (base ROC-AUC {base:.3f})", "",
                  "| Perm rank | Feature | Perm importance | mean|SHAP| | SHAP rank | Type |",
                  "|---|---|---|---|---|---|"]
        for _, r in merged.iterrows():
            ptype = "PSG" if r["feature"] in PSG[dataset] else ""
            shap_rank = "" if pd.isna(r["shap_rank"]) else int(r["shap_rank"])
            shap_val = "" if pd.isna(r["mean_abs_shap"]) else f"{r['mean_abs_shap']:.4f}"
            lines.append(f"| {r['perm_rank']} | {r['feature']} | {r['mean_importance']:.4f} | "
                         f"{shap_val} | {shap_rank} | {ptype} |")
        lines.append("")
    return "\n".join(lines)


def make_tables(out_dir=OUT_DIR):
    md = ["# Feature-importance rankings (R4-C5)", "",
          "Permutation importance = mean ROC-AUC drop over 20 runs on the test set. "
          "mean|SHAP| = mean absolute SHAP value on the seed-42 split. "
          "Type 'PSG' marks PSG / diagnostic-derived predictors (outcome-leakage set; see R4-C1).", ""]
    for dataset in ["SDB", "OSA"]:
        md.append(_rank_table(dataset, out_dir))
    (out_dir / "importance_results_tables.md").write_text("\n".join(md), encoding="utf-8")
    print("tables written", flush=True)


if __name__ == "__main__":
    make_tables()
