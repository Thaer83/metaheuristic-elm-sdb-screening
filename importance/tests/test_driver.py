from importance import models
from importance.run_importance import run_permutation


def test_run_permutation_smoke(monkeypatch):
    # Use a single fast model so the smoke test stays quick (no optimized-ELM fit).
    monkeypatch.setitem(models.REPRESENTATIVE, "SDB", ["XGB"])
    df = run_permutation("SDB", n_runs=2)
    assert set(df.columns) == {"model", "feature", "mean_importance",
                               "std_importance", "mean_base_auc"}
    assert len(df) == 10            # 10 SDB features x 1 model
    assert set(df["model"]) == {"XGB"}
