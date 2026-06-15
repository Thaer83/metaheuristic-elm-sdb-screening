# config.py
import os

# Folder to store all result files
RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

#DATA_PATH = "Dataset/OSA-data.csv"   # change to dataset path
#TARGET_COL = "class"                 # change to target column name

DATA_PATH = "Dataset/sdb_dataset.csv"   # change to dataset path
TARGET_COL = "Diagnosis_of_SDB"                 # change to target column name

# Oversampling options (used only for imbalanced dataset )
USE_ADASYN = False          # set to True when you run the second dataset
ADASYN_RATIO = 0.9         # sampling_strategy for ADASYN (n_minority / n_majority)

TEST_SIZE = 0.2
N_RUNS = 20
BASE_RANDOM_SEED = 42

# Test-set outputs
TEST_EXCEL_OUTPUT = os.path.join(RESULTS_DIR, "baseline_results_test.xlsx") #"baseline_results_test.xlsx"
TEST_CSV_OUTPUT = os.path.join(RESULTS_DIR, "baseline_results_test_all_runs.csv") #"baseline_results_test_all_runs.csv"

# Train-set outputs
TRAIN_EXCEL_OUTPUT = os.path.join(RESULTS_DIR, "baseline_results_train.xlsx") #"baseline_results_train.xlsx"
TRAIN_CSV_OUTPUT = os.path.join(RESULTS_DIR, "baseline_results_train_all_runs.csv") #"baseline_results_train_all_runs.csv"

# ROC figure
ROC_FIG_PATH = os.path.join(RESULTS_DIR, "baseline_roc_curves.png")# "baseline_roc_curves.png"
