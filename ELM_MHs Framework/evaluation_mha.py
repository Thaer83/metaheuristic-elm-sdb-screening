# framework/evaluation_mha.py

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from config_mha import POS_LABEL


def _safe_binary_scores(y_true, y_proba):
    """
    Extract scores for binary ROC/AUC.

    Supports:
      - vector of scores
      - (n_samples, n_classes) with positive class at index 1
    """
    if y_proba is None:
        return None

    y_proba = np.asarray(y_proba)

    if y_proba.ndim == 1:
        return y_proba

    if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
        # assume positive class at index 1 (y encoded as 0/1)
        return y_proba[:, 1]

    return None


def compute_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Compute classification metrics + confusion matrix + ROC data.

    Returns:
        metrics_dict, conf_mat (2x2), roc_data (fpr, tpr, thresholds) or (None, None, None)
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    conf_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])

    y_scores = _safe_binary_scores(y_true, y_proba)
    if y_scores is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
            fpr, tpr, thresholds = roc_curve(
                y_true, y_scores, pos_label=POS_LABEL
            )
        except ValueError:
            metrics["roc_auc"] = np.nan
            fpr = tpr = thresholds = None
    else:
        metrics["roc_auc"] = np.nan
        fpr = tpr = thresholds = None

    return metrics, conf_mat, (fpr, tpr, thresholds)


def conf_mat_to_row(conf_mat):
    """
    Flatten 2x2 confusion matrix into a dict: tn, fp, fn, tp
    """
    tn, fp, fn, tp = conf_mat.ravel()
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _export_split_results(df: pd.DataFrame, dataset_name: str, split_label: str, out_dir: Path):
    """
    Save all runs for a split (train or test) into:
      - one CSV file (all algorithms, all runs)
      - one Excel file with:
          * 'all_runs' sheet
          * one sheet per algorithm with all runs + mean/std rows
    """
    if df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) CSV with everything
    csv_path = out_dir / f"{dataset_name}_{split_label}_all_runs.csv"
    df.to_csv(csv_path, index=False)

    # 2) Excel with all runs + per-algorithm tabs
    excel_path = out_dir / f"{dataset_name}_{split_label}_results.xlsx"

    # numeric columns for mean/std (exclude id columns)
    id_cols = {"algo", "run_id", "seed"}
    numeric_cols = [c for c in df.columns if c not in id_cols]

    with pd.ExcelWriter(excel_path) as writer:
        # all runs
        df.to_excel(writer, sheet_name="all_runs", index=False)

        # per-algorithm sheets
        for algo in sorted(df["algo"].unique()):
            sub = df[df["algo"] == algo].copy()

            mean_vals = sub[numeric_cols].mean(numeric_only=True)
            std_vals = sub[numeric_cols].std(numeric_only=True)

            mean_row = {col: mean_vals.get(col, np.nan) for col in numeric_cols}
            std_row = {col: std_vals.get(col, np.nan) for col in numeric_cols}

            mean_row.update({"algo": algo, "run_id": "mean", "seed": ""})
            std_row.update({"algo": algo, "run_id": "std", "seed": ""})

            summary_df = pd.concat(
                [sub, pd.DataFrame([mean_row, std_row])],
                ignore_index=True,
            )

            summary_df.to_excel(writer, sheet_name=algo, index=False)


def save_global_results(
    train_rows: list[dict],
    test_rows: list[dict],
    dataset_name: str,
    out_dir: Path,
):
    """
    Save global TRAIN and TEST metrics into CSV + Excel, as requested.
    """
    if test_rows:
        test_df = pd.DataFrame(test_rows)
        _export_split_results(test_df, dataset_name, "test", out_dir)

    if train_rows:
        train_df = pd.DataFrame(train_rows)
        _export_split_results(train_df, dataset_name, "train", out_dir)
