from pathlib import Path

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "Results_MHA"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# Path to your processed dataset (adjust to match Phase 1)
#DATA_PATH = PROJECT_ROOT / "Dataset" / "OSA-data.csv"
DATA_PATH = PROJECT_ROOT / "Dataset" / "sdb_dataset.csv"

# ---- Data / splitting ----
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---- ELM / MHA config ----
ELM_LAYER_SIZES = (50,)          # hidden-layer sizes (tune later)
ELM_ACTIVATION = "relu"
# Use a supported classification objective from IntelELM (e.g., 'F1S', 'BSL', 'KLDL', etc.)
ELM_OBJ_NAME = "CEL"

# Lower/upper bounds for MH search space (IntelELM paper suggests wider range)
LB = -1.0
UB = 1.0

# How many independent runs per optimizer
N_RUNS_PER_ALGO = 20

# Positive class label for ROC / confusion matrix interpretation
POS_LABEL = 1

# Columns for metric tables
METRIC_COLUMNS = [
    "algo", "run_id",
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "train_time_sec", "test_time_sec"
]
