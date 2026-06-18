"""Re-run the 9 OSA models (5 baselines + best-4 optimized) over the shared 20 stratified
splits, recording per-run accuracy, F1, and ROC-AUC for the R1-C6 paired comparison.
Hard predictions via model.predict (matches the imbalance None-arm computation); ROC-AUC
from the softmax/predict_proba positive-class score."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imbalance.data import run_splits, scale
from imbalance.scoring import positive_probability
from importance.data import load_osa
from comparison import models

OUT_DIR = Path(__file__).resolve().parents[1] / "Results_comparison"
OSA_MODELS = models.BASELINES + models.OPTIMIZED["OSA"]


def _log_error(out_dir, model_name, run_id, err):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "comparison_errors_log.csv"
    pd.DataFrame([{"model": model_name, "run": run_id, "error": repr(err)}]).to_csv(
        p, mode="a", header=not p.exists(), index=False)


def run_osa(out_dir=OUT_DIR, n_runs=20):
    out_dir = Path(out_dir)
    X, y, _ = load_osa()
    splits = run_splits(X, y, n_runs=n_runs)
    rows = []
    for i, s in enumerate(splits, start=1):
        tr, te = s["train_idx"], s["test_idx"]
        Xtr, Xte = scale(X[tr], X[te])
        ytr, yte = y[tr], y[te]
        for name in OSA_MODELS:
            try:
                model, kind = models.build(name, s["seed"])
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)
                proba = positive_probability(model, Xte, kind)
                rows.append({
                    "run": i, "seed": s["seed"], "model": name,
                    "accuracy": float(accuracy_score(yte, pred)),
                    "f1": float(f1_score(yte, pred)),
                    "roc_auc": float(roc_auc_score(yte, proba)),
                })
            except Exception as e:   # noqa: BLE001 - log and keep going
                _log_error(out_dir, name, i, e)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "osa_comparison_all_runs.csv", index=False)
    return df


if __name__ == "__main__":
    run_osa()
