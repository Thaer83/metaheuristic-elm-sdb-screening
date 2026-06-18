"""Dataset loading with feature names for the importance experiment.
Reuses the imbalance package's 20-run split protocol and train-only scaler."""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imbalance.data import run_splits, scale  # re-exported for callers

_ROOT = Path(__file__).resolve().parents[1]
SDB_PATH = _ROOT / "Dataset" / "sdb_dataset.csv"
OSA_PATH = _ROOT / "Dataset" / "OSA-data.csv"

# PSG / diagnostic-derived predictors (outcome-leakage set; feeds R4-C1).
SDB_PSG_FEATURES = [
    "Oxygen_Saturation", "AHI", "ECG_Heart_Rate", "SpO2",
    "Nasal_Airflow", "Chest_Movement",
]
OSA_PSG_FEATURES = [
    "RDI", "TST", "Sleep Effic", "REM AHI", "NREM AHI", "Supine AHI",
    "Apnea Index", "Hypopnea Index", "Arousal index", "Awakening Index",
    "PLM Index", "Mins.SaO2", "Mins.SaO2Desats", "Lowest Sa02",
]


def load_sdb():
    """SDB: last column is the binary target; 10 numeric predictors."""
    df = pd.read_csv(SDB_PATH)
    names = list(df.columns[:-1])
    X = df.iloc[:, :-1].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y, names


def load_osa():
    """OSA: target column 'class'; drop NA rows; 31 predictors."""
    df = pd.read_csv(OSA_PATH).dropna(axis=0)
    names = [c for c in df.columns if c != "class"]
    X = df[names].values.astype(float)
    y = LabelEncoder().fit_transform(df["class"].values)
    return X, y, names
