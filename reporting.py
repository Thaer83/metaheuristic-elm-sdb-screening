# reporting.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    TEST_CSV_OUTPUT,
    TRAIN_CSV_OUTPUT,
    TEST_EXCEL_OUTPUT,
    TRAIN_EXCEL_OUTPUT,
    ROC_FIG_PATH,
)


def export_to_excel(df, excel_path):
    """Save all runs + per-model sheets with mean/std rows."""
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_runs", index=False)

        numeric_cols = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "tn",
            "fp",
            "fn",
            "tp",
            "train_time_sec",
            "test_time_sec",
        ]

        # Per-model sheets (with mean/std appended)
        for model_name in sorted(df["model"].unique()):
            model_df = df[df["model"] == model_name].copy()

            mean_vals = model_df[numeric_cols].mean(numeric_only=True)
            std_vals = model_df[numeric_cols].std(numeric_only=True)

            mean_row = {col: mean_vals[col] for col in numeric_cols}
            std_row = {col: std_vals[col] for col in numeric_cols}

            mean_row.update({"run": "mean", "seed": "", "model": model_name})
            std_row.update({"run": "std", "seed": "", "model": model_name})

            summary_df = pd.concat(
                [model_df, pd.DataFrame([mean_row, std_row])],
                ignore_index=True,
            )

            summary_df.to_excel(writer, sheet_name=model_name, index=False)


def save_results(train_df, test_df):
    """Save CSV and Excel files for train and test splits."""
    # CSV
    test_df.to_csv(TEST_CSV_OUTPUT, index=False)
    train_df.to_csv(TRAIN_CSV_OUTPUT, index=False)

    print(f"Saved TEST run-level results to: {TEST_CSV_OUTPUT}")
    print(f"Saved TRAIN run-level results to: {TRAIN_CSV_OUTPUT}")

    # Excel
    export_to_excel(test_df, TEST_EXCEL_OUTPUT)
    export_to_excel(train_df, TRAIN_EXCEL_OUTPUT)

    print(f"Saved detailed TEST sheets to: {TEST_EXCEL_OUTPUT}")
    print(f"Saved detailed TRAIN sheets to: {TRAIN_EXCEL_OUTPUT}")


def plot_roc_curves(test_df, roc_data):
    """Plot mean ROC curves over all runs and save to file."""
    plt.figure(figsize=(7, 6))

    # Mean AUC per model across runs (for legend)
    mean_auc = test_df.groupby("model")["roc_auc"].mean(numeric_only=True)

    # Common FPR grid for interpolation
    fpr_grid = np.linspace(0.0, 1.0, 101)

    for model_name, curves in roc_data.items():
        if not curves:
            continue

        tprs_interp = []
        for fpr, tpr in curves:
            # Interpolate TPR over the common FPR grid
            tprs_interp.append(np.interp(fpr_grid, fpr, tpr))

        tprs_interp = np.array(tprs_interp)
        mean_tpr = tprs_interp.mean(axis=0)

        auc_val = float(mean_auc.get(model_name, np.nan))

        label = (
            f"{model_name} (mean AUC={auc_val:.3f})"
            if not np.isnan(auc_val)
            else f"{model_name} (mean AUC=NaN)"
        )

        # Only plot the mean curve (no std band)
        plt.plot(fpr_grid, mean_tpr, linewidth=1.5, label=label)

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curves – Baseline Models on Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_FIG_PATH, dpi=300)
    plt.close()

    print(f"Saved mean ROC curves figure (TEST set, labeled axes) to: {ROC_FIG_PATH}")

