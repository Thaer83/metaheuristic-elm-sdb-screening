from intelelm import MhaElmClassifier, ElmClassifier
from comparison import models


def test_sets():
    assert models.BASELINES == ["SVM", "MLP", "LR", "XGB", "ELM"]
    assert models.OPTIMIZED["OSA"] == ["MGO", "RUN", "CLPSO", "GA"]
    assert models.OPTIMIZED["SDB"] == ["MEO", "HIWOA", "HGS", "HHO"]


def test_optim_strings():
    assert models._OPTIM["MGO"][0] == "OriginalMGO"
    assert models._OPTIM["RUN"][0] == "OriginalRUN"
    assert models._OPTIM["CLPSO"][0] == "CL_PSO"
    assert models._OPTIM["GA"][0] == "BaseGA"


def test_build_kinds_types():
    for name in ["SVM", "MLP", "LR", "XGB"]:
        m, k = models.build(name, 42)
        assert k == "sklearn_proba"
    m, k = models.build("ELM", 42)
    assert k == "elm" and isinstance(m, ElmClassifier)
    for name in ["MGO", "RUN", "CLPSO", "GA"]:
        m, k = models.build(name, 42)
        assert k == "elm" and isinstance(m, MhaElmClassifier)


def test_unknown():
    import pytest
    with pytest.raises(ValueError):
        models.build("ZZZ", 42)
