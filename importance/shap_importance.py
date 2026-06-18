"""SHAP (KernelExplainer) importance for one fitted model, via the uniform scorer.
The model is wrapped as a black-box probability function, so IntelELM ELMs work
without touching the estimator API."""
import numpy as np
import shap
from imbalance.scoring import positive_probability


def shap_values_for(model, X_background, X_explain, kind, seed=0, n_background=50):
    """Return SHAP values array (n_explain, n_features) for P(class 1)."""
    np.random.seed(seed)
    X_background = np.asarray(X_background, dtype=float)
    X_explain = np.asarray(X_explain, dtype=float)
    k = min(n_background, len(X_background))
    background = shap.kmeans(X_background, k)

    def f(X):
        return positive_probability(model, np.asarray(X, dtype=float), kind)

    explainer = shap.KernelExplainer(f, background)
    sv = explainer.shap_values(X_explain, silent=True)
    return np.asarray(sv)


def mean_abs_shap(shap_values):
    """Mean |SHAP| per feature -> array (n_features,)."""
    return np.abs(np.asarray(shap_values)).mean(axis=0)


def save_beeswarm(shap_values, X, feature_names, out_path, max_display=20):
    """Render a SHAP beeswarm summary plot to a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
