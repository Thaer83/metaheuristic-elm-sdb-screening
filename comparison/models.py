"""Five baselines + the per-dataset best-4 optimized ELMs, for the R1-C6 comparison.
Baseline hyperparameters match the paper's Table 3; optimized configs match optimizers_config.py."""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from intelelm import ElmClassifier, MhaElmClassifier

BASELINES = ["SVM", "MLP", "LR", "XGB", "ELM"]
OPTIMIZED = {"OSA": ["MGO", "RUN", "CLPSO", "GA"],   # paper Fig. 8 best-4
             "SDB": ["MEO", "HIWOA", "HGS", "HHO"]}  # paper Fig. 9 best-4 (reused from imbalance)

# optimizer key -> (MEALPY optim string, label)
_OPTIM = {
    "MGO":   ("OriginalMGO", "MGO"),
    "RUN":   ("OriginalRUN", "RUN"),
    "CLPSO": ("CL_PSO", "CLPSO"),
    "GA":    ("BaseGA", "GA"),
}


def _opt_elm(key, seed):
    optim, label = _OPTIM[key]
    return MhaElmClassifier(
        layer_sizes=(50,), act_name="relu", obj_name="CEL",
        optim=optim, optim_params={"name": label, "epoch": 100, "pop_size": 30},
        seed=seed, lb=-1.0, ub=1.0, mode="single", verbose=False,
    )


def build(name, seed):
    """Return (model, kind). kind in {'sklearn_proba', 'elm'}."""
    if name == "ELM":
        return ElmClassifier(layer_sizes=(50,), act_name="relu", seed=seed), "elm"
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, random_state=seed), "sklearn_proba"
    if name == "LR":
        return LogisticRegression(max_iter=200, random_state=seed), "sklearn_proba"
    if name == "XGB":
        return XGBClassifier(
            n_estimators=20, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=seed,
        ), "sklearn_proba"
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam",
                             max_iter=200, random_state=seed), "sklearn_proba"
    if name in _OPTIM:
        return _opt_elm(name, seed), "elm"
    raise ValueError(f"unknown model: {name}")
