"""The 9-model SDB subset, built per imbalance arm with a uniform fit/score kind.
Reuses the paper's hyperparameters (Table 3) and the optimized-ELM config."""
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from intelelm import ElmClassifier, MhaElmClassifier

MODEL_NAMES = ["SVM", "MLP", "XGB", "LR", "ELM", "MEO", "HIWOA", "HGS", "HHO"]

# optimized-ELM optimizer keys -> MEALPY class names (paper's SDB best-4)
_OPTIM = {"MEO": "ModifiedEO", "HIWOA": "HI_WOA", "HGS": "OriginalHGS", "HHO": "OriginalHHO"}


@dataclass
class Spec:
    kind: str                 # 'sklearn_proba' or 'elm'
    supports_class_weight: bool


SPECS = {
    "SVM": Spec("sklearn_proba", True),
    "LR":  Spec("sklearn_proba", True),
    "XGB": Spec("sklearn_proba", True),
    "MLP": Spec("sklearn_proba", False),
    "ELM": Spec("elm", False),
    **{k: Spec("elm", False) for k in _OPTIM},
}


def _elm_optimized(optim_key, seed):
    return MhaElmClassifier(
        layer_sizes=(50,), act_name="relu", obj_name="CEL",
        optim=_OPTIM[optim_key],
        optim_params={"name": optim_key, "epoch": 100, "pop_size": 30},
        seed=seed, lb=-1.0, ub=1.0, mode="single", verbose=False,
    )


def build(name, arm, seed):
    """Return (model, kind). arm in {'none','smote','classweight','threshold'}.
    Resampling/threshold are handled by the driver, not here; this only varies
    class_weight when arm == 'classweight' and the model supports it.
    For XGB classweight the driver sets scale_pos_weight = n_neg/n_pos."""
    cw = (arm == "classweight")
    if name == "LR":
        return LogisticRegression(max_iter=200, random_state=seed,
                                  class_weight=("balanced" if cw else None)), "sklearn_proba"
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, random_state=seed,
                   class_weight=("balanced" if cw else None)), "sklearn_proba"
    if name == "XGB":
        kwargs = dict(n_estimators=20, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, eval_metric="logloss", random_state=seed)
        return XGBClassifier(**kwargs), "sklearn_proba"
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam",
                             max_iter=200, random_state=seed), "sklearn_proba"
    if name == "ELM":
        return ElmClassifier(layer_sizes=(50,), act_name="relu", seed=seed), "elm"
    if name in _OPTIM:
        return _elm_optimized(name, seed), "elm"
    raise ValueError(f"unknown model: {name}")
