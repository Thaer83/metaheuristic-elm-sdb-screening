import comparison.run_osa as r


def test_run_osa_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "OSA_MODELS", ["LR"])   # one fast model, no optimizer fit
    df = r.run_osa(out_dir=tmp_path, n_runs=2)
    assert set(df.columns) == {"run", "seed", "model", "accuracy", "f1", "roc_auc"}
    assert len(df) == 2
    assert set(df["model"]) == {"LR"}
