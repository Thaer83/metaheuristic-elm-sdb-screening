"""Driver for the SDB imbalance experiment. Writes Results_imbalance_SDB/.

Per (run, model, arm) it trains on the run's training split (optionally SMOTE'd),
gets test-set positive probabilities, classifies at 0.5 (None/SMOTE/ClassWeight)
or at the Youden-J threshold (Threshold arm, selected on an internal validation
slice of train), then records confusion-matrix metrics + ranking + Brier."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from imbalance import data, scoring, resampling, threshold, models_subset as ms
from imbalance import metrics as M

ARMS = ["none", "smote", "adasyn", "classweight", "threshold"]
OUT_DIR = Path(__file__).resolve().parents[1] / "Results_imbalance_SDB"


def run_one(X, y, split, model, arm):
    """Return one metrics row dict, or None if the arm does not apply to the model."""
    seed = split["seed"]
    if arm == "classweight" and not ms.SPECS[model].supports_class_weight:
        return None
    tr, te = split["train_idx"], split["test_idx"]
    X_tr, X_te = data.scale(X[tr], X[te])
    y_tr, y_te = y[tr], y[te]

    if arm == "threshold":
        # Pick the Youden threshold on an inner 25% validation slice (model fit on the
        # 75% core), then REFIT on the FULL training split for test scoring. The
        # Threshold arm thus differs from None only in the decision cutoff, so its
        # ROC-AUC/PR-AUC/Brier are identical to None (no training-size confound).
        # Leak-free: the test split is never used in threshold selection or fitting.
        X_core, X_val, y_core, y_val = train_test_split(
            X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=seed)
        sel, kind = ms.build(model, arm="none", seed=seed)
        sel.fit(X_core, y_core)
        thr = threshold.youden_threshold(
            y_val, scoring.positive_probability(sel, X_val, kind=kind))
        est, _ = ms.build(model, arm="none", seed=seed)
        est.fit(X_tr, y_tr)
        p_te = scoring.positive_probability(est, X_te, kind=kind)
        y_pred = threshold.apply_threshold(p_te, thr)
    else:
        if arm in ("smote", "adasyn"):
            X_tr, y_tr = resampling.resample_train(X_tr, y_tr, method=arm, seed=seed)
        model_obj, kind = ms.build(model, arm=arm, seed=seed)
        if arm == "classweight" and model == "XGB":
            n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
            model_obj.set_params(scale_pos_weight=(n_neg / n_pos))
        model_obj.fit(X_tr, y_tr)
        p_te = scoring.positive_probability(model_obj, X_te, kind=kind)
        y_pred = (p_te >= 0.5).astype(int)

    row = {"run": seed - data.BASE_SEED + 1, "seed": seed, "model": model, "arm": arm}
    row.update(M.threshold_metrics(y_te, y_pred))
    row.update(M.ranking_metrics(y_te, p_te))
    row.update(M.probability_metrics(y_te, p_te))
    return row


def run_all(out_dir=OUT_DIR):
    X, y = data.load_sdb()
    splits = data.run_splits(X, y)
    rows = []
    for split in splits:
        for model in ms.MODEL_NAMES:
            for arm in ARMS:
                r = run_one(X, y, split, model, arm)
                if r is not None:
                    rows.append(r)
        print(f"run {split['seed'] - data.BASE_SEED + 1}/20 done", flush=True)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sdb_imbalance_all_runs.csv", index=False)
    _write_summary(df, out_dir / "sdb_imbalance_summary.xlsx")
    print(f"wrote {len(df)} rows to {out_dir}", flush=True)
    return df


def run_single_arm(arm, out_dir=OUT_DIR):
    """Run ONE arm for all models and APPEND to the existing all_runs.csv, then
    regenerate the summary. Existing rows are never modified (only appended to).
    Refuses to run if the arm is already present."""
    out_dir = Path(out_dir)
    csv = out_dir / "sdb_imbalance_all_runs.csv"
    old = pd.read_csv(csv)
    if arm in set(old["arm"].unique()):
        raise ValueError(f"arm '{arm}' already present in {csv}")
    X, y = data.load_sdb()
    rows = []
    for split in data.run_splits(X, y):
        for model in ms.MODEL_NAMES:
            r = run_one(X, y, split, model, arm)
            if r is not None:
                rows.append(r)
        print(f"{arm} run {split['seed'] - data.BASE_SEED + 1}/20 done", flush=True)
    combined = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(csv, index=False)
    _write_summary(combined, out_dir / "sdb_imbalance_summary.xlsx")
    print(f"appended {len(rows)} '{arm}' rows; total now {len(combined)}", flush=True)
    return combined


def _write_summary(df, path):
    metric_cols = [c for c in df.columns if c not in {"run", "seed", "model", "arm"}]
    agg = df.groupby(["model", "arm"])[metric_cols].agg(["mean", "std"])
    # Flatten the (metric, stat) MultiIndex columns -> 'accuracy_mean', 'accuracy_std', ...
    # (pandas cannot write MultiIndex columns to Excel with index=False.)
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    summary = agg.reset_index()
    with pd.ExcelWriter(path) as w:
        df.to_excel(w, sheet_name="all_runs", index=False)
        summary.to_excel(w, sheet_name="summary", index=False)
    return summary


if __name__ == "__main__":
    run_all()
