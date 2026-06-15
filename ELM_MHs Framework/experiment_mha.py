import time
import traceback
from pathlib import Path
import pandas as pd
from intelelm import MhaElmClassifier

from config_mha import (
    RESULTS_ROOT,
    ELM_LAYER_SIZES,
    ELM_ACTIVATION,
    ELM_OBJ_NAME,
    LB,
    UB,
    N_RUNS_PER_ALGO,
    RANDOM_STATE,
)
from optimizers_config import get_selected_optimizers
from evaluation_mha import (
    compute_classification_metrics,
    conf_mat_to_row,
    save_global_results,
)
from plots_mha import (
    plot_roc_curve,
    save_roc_data,
    aggregate_and_plot_mean_roc,
    aggregate_and_plot_convergence,
    plot_all_mean_rocs,
    plot_all_convergence_curves,
)


def _get_probs_safe(model, X):
    """
    Try to obtain probabilities for ROC calculation.
    IntelELM's MhaElmClassifier supports 'return_prob' in predict().
    """
    try:
        return model.predict(X, return_prob=True)
    except TypeError:
        # older API or fallback
        try:
            return model.predict_proba(X)
        except Exception:
            return None


def run_mha_elm_experiments(X_train, X_test, y_train, y_test, dataset_name: str):
    """
    Main experiment loop: for each optimizer, run N_RUNS_PER_ALGO times, record metrics,
    save ROC + convergence curves, and log errors without stopping the whole batch.
    Produces:
      - global TRAIN/TEST CSV + Excel (all algos, all runs)
      - per-algorithm ROC & convergence plots/CSVs
      - combined ROC and convergence plots across algorithms.
    """
    errors = []

    # global collectors (all algorithms)
    all_test_rows = []
    all_train_rows = []
    mean_roc_curves = []   # for combined ROC
    all_conv_curves = []   # for combined convergence

    optimizers = get_selected_optimizers()

    for algo_label, cfg in optimizers.items():
        print(f"\n=== Optimizer: {algo_label} ===")
        algo_dir = RESULTS_ROOT / f"{dataset_name}_{algo_label}"
        algo_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results directory: {algo_dir}")

        test_roc_runs = []   # for this algo

        for run_id in range(1, N_RUNS_PER_ALGO + 1):
            print(f"  -> Run {run_id}/{N_RUNS_PER_ALGO} for {algo_label}...")
            run_dir = algo_dir / f"run_{run_id:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            try:
                seed = cfg.get("seed", RANDOM_STATE + run_id)

                model = MhaElmClassifier(
                    layer_sizes=ELM_LAYER_SIZES,
                    act_name=ELM_ACTIVATION,
                    obj_name=ELM_OBJ_NAME,
                    optim=cfg["optim"],
                    optim_params=cfg["optim_params"],
                    verbose=False,
                    seed=seed,
                    lb=LB,
                    ub=UB,
                    mode=cfg.get("mode", "single"),
                    n_workers=cfg.get("n_workers", None),
                    termination=cfg.get("termination", None),
                )

                # ---- Training ----
                t0_train = time.perf_counter()
                model.fit(X_train, y_train)
                t1_train = time.perf_counter()
                train_time = t1_train - t0_train
                print(f"     Training done in {train_time:.3f} sec")

                # Save fitness history for convergence curves
                try:
                    model.save_loss_train(
                        save_path=str(run_dir),
                        filename="loss_train.csv",
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "algo": algo_label,
                            "run_id": run_id,
                            "stage": "save_loss_train",
                            "exception": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

                # ---- Predictions ----
                y_pred_train = model.predict(X_train)
                y_proba_train = _get_probs_safe(model, X_train)

                t0_test = time.perf_counter()
                y_pred_test = model.predict(X_test)
                t1_test = time.perf_counter()
                test_time = t1_test - t0_test

                y_proba_test = _get_probs_safe(model, X_test)

                # ---- Metrics: TEST ----
                test_metrics, test_conf, test_roc = compute_classification_metrics(
                    y_test, y_pred_test, y_proba_test
                )
                test_row = {
                    "algo": algo_label,
                    "run_id": run_id,
                    "seed": seed,
                    "train_time_sec": train_time,
                    "test_time_sec": test_time,
                }
                test_row.update(test_metrics)
                test_row.update(conf_mat_to_row(test_conf))
                all_test_rows.append(test_row)

                test_roc_runs.append({"run_id": run_id, "roc": test_roc})

                # per-run ROC + CSV for TEST
                fpr_test, tpr_test, thr_test = test_roc
                if fpr_test is not None and tpr_test is not None:
                    plot_roc_curve(
                        fpr_test,
                        tpr_test,
                        algo_label=algo_label,
                        run_id=run_id,
                        split_label="test",
                        out_dir=run_dir,
                    )
                    save_roc_data(
                        fpr_test,
                        tpr_test,
                        thr_test,
                        algo_label=algo_label,
                        run_id=run_id,
                        split_label="test",
                        out_dir=run_dir,
                    )

                # ---- Metrics: TRAIN ----
                train_metrics, train_conf, _ = compute_classification_metrics(
                    y_train, y_pred_train, y_proba_train
                )
                train_row = {
                    "algo": algo_label,
                    "run_id": run_id,
                    "seed": seed,
                    "train_time_sec": train_time,
                    "test_time_sec": test_time,
                }
                train_row.update(train_metrics)
                train_row.update(conf_mat_to_row(train_conf))
                all_train_rows.append(train_row)

            except Exception as exc:
                print(f"     [ERROR] {algo_label} run {run_id}: {exc}")
                errors.append(
                    {
                        "algo": algo_label,
                        "run_id": run_id,
                        "stage": "fit/predict",
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                continue

        # ---- Mean ROC & Convergence for this algorithm ----
        if test_roc_runs:
            mean_roc = aggregate_and_plot_mean_roc(
                roc_runs=test_roc_runs,
                algo_label=algo_label,
                split_label="test",
                out_dir=algo_dir,
            )
            if mean_roc is not None:
                mean_roc_curves.append(mean_roc)

        df_conv = aggregate_and_plot_convergence(
            algo_label=algo_label,
            algo_dir=algo_dir,
        )
        if df_conv is not None:
            all_conv_curves.append({"algo": algo_label, "df": df_conv})

        print(f"=== Finished {algo_label} ===")

    # ---- Save global TRAIN/TEST metrics ----
    save_global_results(
        train_rows=all_train_rows,
        test_rows=all_test_rows,
        dataset_name=dataset_name,
        out_dir=RESULTS_ROOT,
    )

    # ---- Combined ROC & convergence figures ----
    if mean_roc_curves:
        plot_all_mean_rocs(
            mean_roc_list=mean_roc_curves,
            split_label="test",
            dataset_name=dataset_name,
            out_dir=RESULTS_ROOT,
        )

    if all_conv_curves:
        plot_all_convergence_curves(
            conv_list=all_conv_curves,
            dataset_name=dataset_name,
            out_dir=RESULTS_ROOT,
        )

    # ---- Error log ----
    if errors:
        errors_df = pd.DataFrame(errors)
        err_path = RESULTS_ROOT / f"{dataset_name}_mha_errors_log.csv"
        errors_df.to_csv(err_path, index=False)
        print(f"\nSome runs failed. Error log saved to: {err_path}")
    else:
        print("\nNo errors encountered during MHA experiments.")
