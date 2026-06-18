"""Training-set resampling for the SDB imbalance experiment. Apply to train only."""
from imblearn.over_sampling import SMOTE, ADASYN


def resample_train(X_train, y_train, method="smote", seed=42):
    """Return (X_res, y_res). method in {'none', 'smote', 'adasyn'}.
    Must only ever be called on the training split."""
    if method == "none":
        return X_train, y_train
    if method == "smote":
        sampler = SMOTE(random_state=seed)
    elif method == "adasyn":
        sampler = ADASYN(random_state=seed)
    else:
        raise ValueError(f"unknown resampling method: {method}")
    return sampler.fit_resample(X_train, y_train)
