"""Importance figures from the CSVs: permutation bars (with error bars) and
mean|SHAP| bars. SHAP beeswarms are produced by the driver from live values."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[1] / "Results_importance"


def _bars(df, value_col, err_col, xlabel, title_prefix, out_path):
    model_names = list(df["model"].unique())
    fig, axes = plt.subplots(1, len(model_names),
                             figsize=(6 * len(model_names), 6), squeeze=False)
    for ax, m in zip(axes[0], model_names):
        sub = df[df.model == m].sort_values(value_col)
        xerr = sub[err_col] if err_col else None
        ax.barh(sub["feature"], sub[value_col], xerr=xerr)
        ax.set_title(f"{title_prefix} - {m}")
        ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_figures(out_dir=OUT_DIR):
    for dataset in ["SDB", "OSA"]:
        perm = pd.read_csv(out_dir / f"{dataset.lower()}_permutation_importance.csv")
        _bars(perm, "mean_importance", "std_importance",
              "Permutation importance (ROC-AUC drop)", dataset,
              out_dir / f"{dataset.lower()}_permutation_importance.png")
        shp = pd.read_csv(out_dir / f"{dataset.lower()}_shap_importance.csv")
        _bars(shp, "mean_abs_shap", None, "mean |SHAP|", dataset,
              out_dir / f"{dataset.lower()}_shap_bar.png")
    print(f"figures written to {out_dir}", flush=True)


if __name__ == "__main__":
    make_figures()
