"""SDB dataset loading and the shared 20-run stratified split protocol."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

SDB_PATH = Path(__file__).resolve().parents[1] / "Dataset" / "sdb_dataset.csv"
N_RUNS = 20
BASE_SEED = 42
TEST_SIZE = 0.2


def load_sdb():
    """Load the model-ready SDB file (last column = binary Diagnosis_of_SDB)."""
    df = pd.read_csv(SDB_PATH)
    X = df.iloc[:, :-1].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y


def run_splits(X, y, n_runs=N_RUNS, base_seed=BASE_SEED, test_size=TEST_SIZE):
    """Return list of dicts: {seed, train_idx, test_idx}, stratified, one per run."""
    idx = np.arange(len(y))
    splits = []
    for i in range(n_runs):
        seed = base_seed + i
        tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
        splits.append({"seed": seed, "train_idx": np.sort(tr), "test_idx": np.sort(te)})
    return splits


def scale(X_train, X_test):
    """StandardScaler fit on train only."""
    sc = StandardScaler().fit(X_train)
    return sc.transform(X_train), sc.transform(X_test)
