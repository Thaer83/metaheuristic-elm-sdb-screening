# models.py

from intelelm import ElmClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier          
#from lightgbm import LGBMClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def make_models(random_state: int):
    """
    Create and return a dict of models for a given random_state.
    ELM uses IntelELM's ElmClassifier.
    """
    models = {
        "ELM": ElmClassifier(
            layer_sizes=(50,),   # number of hidden neurons (can be tuned later)
            act_name="relu",
            seed=random_state,
        ),
        "LogReg": LogisticRegression(
            max_iter=200,
            random_state=random_state,
        ),
        "RF": RandomForestClassifier(
            n_estimators=20,
            random_state=random_state,
        ),
        "SVC_RBF": SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
        ),
        "XGB": XGBClassifier(                 
            n_estimators=20,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            #use_label_encoder=False, not used in newer version
        ),
        "MLP": MLPClassifier(          # NEW: Multilayer Perceptron
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=random_state,
        ),
         "KNN": KNeighborsClassifier(         # NEW: k-Nearest Neighbors
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
        ),
        "DT": DecisionTreeClassifier(        # NEW: Decision Tree
            max_depth=None,
            random_state=random_state,
        ),
    }
    return models
