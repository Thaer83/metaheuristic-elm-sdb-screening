"""Reliability-curve data + Brier score + a saved calibration figure."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_points(y_true, y_prob, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return frac_pos, mean_pred


def brier(y_true, y_prob):
    return float(brier_score_loss(y_true, y_prob))


def plot_reliability(curves, out_path, title="SDB calibration"):
    """curves: list of (label, y_true, y_prob). Saves one reliability plot."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for label, y_true, y_prob in curves:
        fp, mp = calibration_points(y_true, y_prob)
        plt.plot(mp, fp, marker="o", label=label)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
