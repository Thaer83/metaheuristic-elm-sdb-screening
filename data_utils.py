# data_utils.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import ADASYN
from config import DATA_PATH, TARGET_COL, TEST_SIZE, USE_ADASYN, ADASYN_RATIO

# ==========================
# Data Loading & Preprocessing
# ==========================
def load_dataset():
    """Load CSV, drop NAs, encode target as 0/1, return X, y, label_encoder."""
    df = pd.read_csv(DATA_PATH).dropna(axis=0)
    # dropna(axis=0): Simple handling of missing values (you can replace with a more advanced imputer)

    y_raw = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL]).values

    # Encode labels to 0/1 if needed
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Sanity check: ensure binary classification
    if len(np.unique(y)) != 2:
        raise ValueError(
            f"Expected a binary target, but found classes {np.unique(y)}. "
            f"Please adapt the code for multi-class or remap labels to binary."
        )

    return X, y, le


def prepare_data_for_run(X, y, run_seed):
    """
    Split into train/test and apply StandardScaler (fit on train only).
    Returns: X_train_scaled, X_test_scaled, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=run_seed,
    )

    # Optional ADASYN oversampling on the training split only
    if USE_ADASYN:
        adasyn = ADASYN(sampling_strategy=ADASYN_RATIO, random_state=run_seed)
        X_train, y_train = adasyn.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
