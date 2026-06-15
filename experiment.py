# experiment.py

import time
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

from config import N_RUNS, BASE_RANDOM_SEED
from data_utils import load_dataset, prepare_data_for_run
from models import make_models

# ==========================
# Helper: get probability for ROC/AUC
# ==========================
def get_positive_proba(model, X):
    """
    Get continuous score for positive class for ROC/AUC.
    Priority:
        1) predict_proba
        2) decision_function (logistic transform)
        3) predicted labels as fallback
    """
    # Case 1: predict_proba available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()

    # Case 2: decision_function available
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores).ravel()
        # map arbitrary scores to (0, 1) via logistic function
        return 1.0 / (1.0 + np.exp(-scores))  # logistic

    # Case 3: fallback – use predicted labels as "probabilities"
    preds = model.predict(X)
    return np.asarray(preds, dtype=float)


def run_experiments():
    """
    Run all models for N_RUNS with different seeds.
    Returns:
        train_df, test_df, roc_info_last_run
    """
    X, y, _ = load_dataset()

    train_results = []
    test_results = []
    #roc_info_last_run = {}  # model_name -> (fpr, tpr, auc) on TEST set
    roc_data = {}  # model_name -> list of (fpr, tpr) on TEST set   <-- CHANGED
    for run_idx in range(N_RUNS):
        run_seed = BASE_RANDOM_SEED + run_idx
        print(f"\n=== Run {run_idx + 1}/{N_RUNS} (seed={run_seed}) ===")

        # Stratified split for class balance
        X_train_scaled, X_test_scaled, y_train, y_test = prepare_data_for_run(
            X, y, run_seed
        )

        models = make_models(run_seed)

        for model_name, model in models.items():
            print(f"  Training {model_name}...")

            # Measure training and testing time per model per run
            # Training time: fit + train evaluation
            # Testing time: test prediction + evaluation
            t0 = time.perf_counter()

            # ----- TRAINING -----
            model.fit(X_train_scaled, y_train)
            # TRAIN predictions and scores
            y_train_pred = model.predict(X_train_scaled)
            y_train_score = get_positive_proba(model, X_train_scaled)

            t1 = time.perf_counter()  # end of training + train eval

            # TEST
            y_test_pred = model.predict(X_test_scaled)
            y_test_score = get_positive_proba(model, X_test_scaled)

            t2 = time.perf_counter()   # end of test eval

            train_time_sec = t1 - t0
            test_time_sec = t2 - t1

            # ---- TRAIN metrics ----
            train_acc = accuracy_score(y_train, y_train_pred)
            train_prec = precision_score(y_train, y_train_pred, zero_division=0)
            train_rec = recall_score(y_train, y_train_pred, zero_division=0)
            train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
            try:
                train_auc = roc_auc_score(y_train, y_train_score)
            except ValueError:
                train_auc = np.nan

            cm_train = confusion_matrix(y_train, y_train_pred, labels=[0, 1])
            tn_tr, fp_tr, fn_tr, tp_tr = cm_train.ravel()

            train_results.append(
                {
                    "run": run_idx + 1,
                    "seed": run_seed,
                    "model": model_name,
                    "accuracy": train_acc,
                    "precision": train_prec,
                    "recall": train_rec,
                    "f1": train_f1,
                    "roc_auc": train_auc,
                    "tn": tn_tr,
                    "fp": fp_tr,
                    "fn": fn_tr,
                    "tp": tp_tr,
                    "train_time_sec": train_time_sec,
                    "test_time_sec": test_time_sec,
                }
            )

            # ---- TEST metrics ----
            test_acc = accuracy_score(y_test, y_test_pred)
            test_prec = precision_score(y_test, y_test_pred, zero_division=0)
            test_rec = recall_score(y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            try:
                test_auc = roc_auc_score(y_test, y_test_score)
            except ValueError:
                test_auc = np.nan

            cm_test = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
            tn_te, fp_te, fn_te, tp_te = cm_test.ravel()

            test_results.append(
                {
                    "run": run_idx + 1,
                    "seed": run_seed,
                    "model": model_name,
                    "accuracy": test_acc,
                    "precision": test_prec,
                    "recall": test_rec,
                    "f1": test_f1,
                    "roc_auc": test_auc,
                    "tn": tn_te,
                    "fp": fp_te,
                    "fn": fn_te,
                    "tp": tp_te,
                    "train_time_sec": train_time_sec,
                    "test_time_sec": test_time_sec,
                }
            )

            # Store ROC info for the LAST run only (TEST set, for plotting)
            # Collect ROC info (TEST set) for ALL runs
            
            try:
                fpr, tpr, _ = roc_curve(y_test, y_test_score)
                roc_data.setdefault(model_name, []).append((fpr, tpr))
            except ValueError:
                pass

    # Convert to DataFrames
    train_df = pd.DataFrame(train_results)
    test_df = pd.DataFrame(test_results)

    return train_df, test_df, roc_data
