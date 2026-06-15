import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config_mha import DATA_PATH, TEST_SIZE, RANDOM_STATE


def load_sleep_apnea_data():
    """
    Load and split the sleep apnea dataset.

    Assumes last column is the target label.
    Adjust if your structure is different.
    """
    df = pd.read_csv(DATA_PATH)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Encode labels
    le_y = LabelEncoder()
    y_train_enc = le_y.fit_transform(y_train)
    y_test_enc = le_y.transform(y_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_enc,
        y_test_enc,
        scaler_X,
        le_y,
    )
