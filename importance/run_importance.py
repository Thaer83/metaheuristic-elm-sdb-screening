"""Driver: 20-run permutation importance + single-split SHAP for both datasets.
Writes per-dataset CSVs and SHAP beeswarm PNGs to Results_importance/."""
from pathlib import Path
import numpy as np
import pandas as pd
from importance import data, models
from importance.permutation import permutation_importance
from importance.shap_importance import shap_values_for, mean_abs_shap, save_beeswarm

OUT_DIR = Path(__file__).resolve().parents[1] / "Results_importance"
N_REPEATS = 10
SHAP_SPLIT = 0          # split index used for SHAP (seed 42)
EXPLAIN_CAP = 100       # cap explained rows for SHAP cost

LOADERS = {"SDB": data.load_sdb, "OSA": data.load_osa}


def _log_error(out_dir, dataset, model_name, err):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "importance_errors_log.csv"
    pd.DataFrame([{"dataset": dataset, "model": model_name, "error": repr(err)}]).to_csv(
        p, mode="a", header=not p.exists(), index=False)


def run_permutation(dataset, n_runs=20, out_dir=OUT_DIR):
    X, y, names = LOADERS[dataset]()
    splits = data.run_splits(X, y, n_runs=n_runs)
    rows = []
    for model_name in models.REPRESENTATIVE[dataset]:
        try:
            per_run, bases = [], []
            for s in splits:
                tr, te = s["train_idx"], s["test_idx"]
                Xtr, Xte = data.scale(X[tr], X[te])
                model, kind = models.build(model_name, s["seed"])
                model.fit(Xtr, y[tr])
                imp, base = permutation_importance(
                    model, Xte, y[te], kind, n_repeats=N_REPEATS, seed=s["seed"])
                per_run.append(imp)
                bases.append(base)
            arr = np.vstack(per_run)
            for j, feat in enumerate(names):
                rows.append({
                    "model": model_name, "feature": feat,
                    "mean_importance": float(arr[:, j].mean()),
                    "std_importance": float(arr[:, j].std(ddof=1)),
                    "mean_base_auc": float(np.mean(bases)),
                })
        except Exception as e:   # noqa: BLE001 - log and keep going
            _log_error(out_dir, dataset, model_name, e)
    return pd.DataFrame(rows)


def run_shap(dataset, out_dir=OUT_DIR):
    X, y, names = LOADERS[dataset]()
    s = data.run_splits(X, y)[SHAP_SPLIT]
    tr, te = s["train_idx"], s["test_idx"]
    Xtr, Xte = data.scale(X[tr], X[te])
    Xte = Xte[:EXPLAIN_CAP]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in models.REPRESENTATIVE[dataset]:
        try:
            model, kind = models.build(model_name, s["seed"])
            model.fit(Xtr, y[tr])
            sv = shap_values_for(model, Xtr, Xte, kind, seed=s["seed"])
            mabs = mean_abs_shap(sv)
            save_beeswarm(sv, Xte, names,
                          out_dir / f"{dataset.lower()}_shap_beeswarm_{model_name}.png")
            for j, feat in enumerate(names):
                rows.append({"model": model_name, "feature": feat,
                             "mean_abs_shap": float(mabs[j])})
        except Exception as e:   # noqa: BLE001
            _log_error(out_dir, dataset, model_name + "_shap", e)
    return pd.DataFrame(rows)


def run_all(out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in ["SDB", "OSA"]:
        run_permutation(dataset, out_dir=out_dir).to_csv(
            out_dir / f"{dataset.lower()}_permutation_importance.csv", index=False)
        run_shap(dataset, out_dir=out_dir).to_csv(
            out_dir / f"{dataset.lower()}_shap_importance.csv", index=False)
        print(f"{dataset} done", flush=True)


if __name__ == "__main__":
    run_all()
