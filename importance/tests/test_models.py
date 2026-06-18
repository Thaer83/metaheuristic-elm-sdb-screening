from intelelm import ElmClassifier, MhaElmClassifier
from importance import models


def test_representative_sets():
    assert models.REPRESENTATIVE["OSA"] == ["ELM", "CLPSO", "SVM"]
    assert models.REPRESENTATIVE["SDB"] == ["ELM", "HGS", "XGB"]


def test_optim_strings():
    assert models._OPTIM["HGS"][0] == "OriginalHGS"
    assert models._OPTIM["CLPSO"][0] == "CL_PSO"


def test_build_kinds():
    for name in ["ELM", "HGS", "CLPSO"]:
        m, kind = models.build(name, seed=42)
        assert kind == "elm"
    for name in ["SVM", "XGB"]:
        m, kind = models.build(name, seed=42)
        assert kind == "sklearn_proba"


def test_build_types():
    assert isinstance(models.build("ELM", 42)[0], ElmClassifier)
    assert isinstance(models.build("CLPSO", 42)[0], MhaElmClassifier)
    assert isinstance(models.build("HGS", 42)[0], MhaElmClassifier)


def test_build_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        models.build("NOPE", 42)
