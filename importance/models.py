"""Representative model subset per dataset for the importance experiment.
Hyperparameters match the paper's Table 3 / optimized-ELM config (see imbalance)."""
from sklearn.svm import SVC
from xgboost import XGBClassifier
from intelelm import ElmClassifier, MhaElmClassifier

REPRESENTATIVE = {
    "OSA": ["ELM", "CLPSO", "SVM"],   # plain ELM, best opt-ELM (Table 8), best baseline (Table 4)
    "SDB": ["ELM", "HGS", "XGB"],     # plain ELM, best opt-ELM (Table 9), best baseline (Table 7)
}

# optimized-ELM key -> (MEALPY optim class string, optim label)
_OPTIM = {
    "HGS":   ("OriginalHGS", "HGS"),
    "CLPSO": ("CL_PSO", "CLPSO"),
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
    if name == "XGB":
        return XGBClassifier(
            n_estimators=20, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=seed,
        ), "sklearn_proba"
    if name in _OPTIM:
        return _opt_elm(name, seed), "elm"
    raise ValueError(f"unknown model: {name}")
