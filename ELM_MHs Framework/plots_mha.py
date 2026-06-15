from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config_mha import RESULTS_ROOT


def plot_roc_curve(fpr, tpr, algo_label, run_id, split_label, out_dir: Path):
    """
    Plot ROC curve and save as PNG.

    Axes are explicitly labeled as requested.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {algo_label} - {split_label} - Run {run_id}")
    plt.grid(True)
    png_path = out_dir / f"{algo_label}_{split_label}_run{run_id:02d}_roc.png"
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()


def save_roc_data(fpr, tpr, thresholds, algo_label, run_id, split_label, out_dir: Path):
    """
    Save ROC curve points to CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )
    csv_path = out_dir / f"{algo_label}_{split_label}_run{run_id:02d}_roc.csv"
    df.to_csv(csv_path, index=False)


def aggregate_and_plot_mean_roc(roc_runs, algo_label, split_label, out_dir: Path):
    """
    Compute and plot mean ROC across runs using interpolation over a common FPR grid.
    Returns a dict with curve data for combined plots.
    """
    if not roc_runs:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    fpr_grid = np.linspace(0.0, 1.0, 101)
    tpr_matrix = []

    for run_data in roc_runs:
        fpr, tpr, _ = run_data["roc"]
        if fpr is None or tpr is None:
            continue
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_matrix.append(tpr_interp)

    if not tpr_matrix:
        return None

    tpr_matrix = np.vstack(tpr_matrix)
    tpr_mean = tpr_matrix.mean(axis=0)
    tpr_std = tpr_matrix.std(axis=0)

    # Save CSV for this algorithm
    df_mean = pd.DataFrame(
        {
            "fpr": fpr_grid,
            "tpr_mean": tpr_mean,
            "tpr_std": tpr_std,
        }
    )
    csv_path = out_dir / f"{algo_label}_{split_label}_roc_mean.csv"
    df_mean.to_csv(csv_path, index=False)

    # Plot mean ROC for this algorithm
    plt.figure()
    plt.plot(fpr_grid, tpr_mean, label=f"{algo_label} mean ROC")
    plt.fill_between(
        fpr_grid,
        np.maximum(tpr_mean - tpr_std, 0.0),
        np.minimum(tpr_mean + tpr_std, 1.0),
        alpha=0.2,
        label="±1 std",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Mean ROC - {algo_label} - {split_label}")
    plt.legend()
    plt.grid(True)
    png_path = out_dir / f"{algo_label}_{split_label}_roc_mean.png"
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()

    return {
        "algo": algo_label,
        "split": split_label,
        "fpr": fpr_grid,
        "tpr_mean": tpr_mean,
        "tpr_std": tpr_std,
    }

def plot_all_mean_rocs(mean_roc_list, split_label: str, dataset_name: str, out_dir: Path):
    """
    Plot all algorithms' mean ROC curves together and save PNG + CSV.
    """
    if not mean_roc_list:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    base_fpr = mean_roc_list[0]["fpr"]
    df = pd.DataFrame({"fpr": base_fpr})

    plt.figure()
    for curve in mean_roc_list:
        algo = curve["algo"]
        tpr_mean = curve["tpr_mean"]
        tpr_std = curve["tpr_std"]

        df[f"tpr_mean_{algo}"] = tpr_mean
        df[f"tpr_std_{algo}"] = tpr_std

        plt.plot(base_fpr, tpr_mean, label=f"{algo} mean ROC")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Mean ROC Curves – all algorithms – {split_label}")
    plt.legend()
    plt.grid(True)

    png_path = out_dir / f"{dataset_name}_{split_label}_roc_all_algos.png"
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()

    csv_path = out_dir / f"{dataset_name}_{split_label}_roc_all_algos.csv"
    df.to_csv(csv_path, index=False)


def aggregate_and_plot_convergence(algo_label, algo_dir: Path):
    """
    Read 'loss_train.csv' from each run, compute mean convergence curve,
    save CSV + PNG, and return the curve DataFrame.
    """
    run_dirs = sorted([d for d in algo_dir.glob("run_*") if d.is_dir()])
    if not run_dirs:
        return None

    loss_list = []

    for run_dir in run_dirs:
        csv_path = run_dir / "loss_train.csv"
        if not csv_path.exists():
            continue
        df_loss = pd.read_csv(csv_path)
        numeric_cols = df_loss.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            continue
        loss_values = df_loss[numeric_cols[-1]].values
        loss_list.append(loss_values)

    if not loss_list:
        return None

    min_len = min(len(arr) for arr in loss_list)
    loss_array = np.vstack([arr[:min_len] for arr in loss_list])

    epochs = np.arange(1, min_len + 1)
    loss_mean = loss_array.mean(axis=0)
    loss_std = loss_array.std(axis=0)

    df_conv = pd.DataFrame(
        {"epoch": epochs, "loss_mean": loss_mean, "loss_std": loss_std}
    )
    csv_path = algo_dir / f"{algo_label}_convergence_mean.csv"
    df_conv.to_csv(csv_path, index=False)

    # Plot convergence for this algorithm
    plt.figure()
    plt.plot(epochs, loss_mean, label="Mean Fitness/Loss")
    plt.fill_between(
        epochs,
        loss_mean - loss_std,
        loss_mean + loss_std,
        alpha=0.2,
        label="±1 std",
    )
    plt.xlabel("Iterations / Epochs")
    plt.ylabel("Fitness / Objective value")
    plt.title(f"Convergence Curve - {algo_label}")
    plt.grid(True)
    plt.legend()
    png_path = algo_dir / f"{algo_label}_convergence_mean.png"
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    return df_conv

def plot_all_convergence_curves(conv_list, dataset_name: str, out_dir: Path):
    """
    Plot all algorithms' mean convergence curves together in one figure.
    Uses each algorithm's own epoch axis.
    """
    if not conv_list:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for item in conv_list:
        algo = item["algo"]
        df_conv = item["df"]
        plt.plot(df_conv["epoch"], df_conv["loss_mean"], label=algo)

    plt.xlabel("Iterations / Epochs")
    plt.ylabel("Fitness / Objective value")
    plt.title("Convergence Curves – all algorithms")
    plt.grid(True)
    plt.legend()

    png_path = out_dir / f"{dataset_name}_convergence_all_algos.png"
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
