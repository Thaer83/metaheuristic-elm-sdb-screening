"""Load the ALREADY-GENERATED per-run results for the R1-C6 rank-sum analysis:
baselines from Results/, optimized ELMs from Results_MHA/. These are the data behind
the paper's Tables 4/7 (baselines) and 8/9 (optimized ELM). Model names are normalized
to the common keys used by comparison.models (SVM, MLP, LR, XGB, ELM + the best-4)."""
from pathlib import Path
import pandas as pd
from comparison import models

_ROOT = Path(__file__).resolve().parents[1]
_BASE_CSV = {
    "OSA": _ROOT / "Results" / "Baseline models on OSA dataset" / "baseline_results_test_all_runs.csv",
    "SDB": _ROOT / "Results" / "Baseline models on SDB dataset" / "baseline_results_test_all_runs.csv",
}
_MHA_CSV = {
    "OSA": _ROOT / "Results_MHA" / "Results OSA" / "sleep_apnea_test_all_runs.csv",
    "SDB": _ROOT / "Results_MHA" / "Results SDB" / "sleep_apnea_test_all_runs.csv",
}
# common key -> baseline name as stored in the Results CSVs
_BASE_NAME = {"SVM": "SVC_RBF", "MLP": "MLP", "LR": "LogReg", "XGB": "XGB", "ELM": "ELM"}
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def load_old(dataset):
    """Return a tidy frame [run, model, accuracy, f1, roc_auc] for the 5 key baselines
    and the dataset's best-4 optimized ELMs, drawn from the reported result files."""
    inv = {v: k for k, v in _BASE_NAME.items()}
    base = pd.read_csv(_BASE_CSV[dataset])
    base = base[base["model"].isin(_BASE_NAME.values())].copy()
    base["model"] = base["model"].map(inv)
    base = base[["run", "model", *METRICS]]

    mha = pd.read_csv(_MHA_CSV[dataset]).rename(columns={"run_id": "run", "algo": "model"})
    mha = mha[mha["model"].isin(models.OPTIMIZED[dataset])][["run", "model", *METRICS]]

    return pd.concat([base, mha], ignore_index=True)
