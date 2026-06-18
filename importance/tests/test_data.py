import numpy as np
from importance import data


def test_load_sdb_shapes():
    X, y, names = data.load_sdb()
    assert X.shape[1] == 10
    assert len(names) == 10
    assert names[0] == "Age" and names[-1] == "Chest_Movement"
    assert set(np.unique(y)) <= {0, 1}


def test_load_osa_shapes():
    X, y, names = data.load_osa()
    assert X.shape[1] == 31
    assert len(names) == 31
    assert "class" not in names
    assert "RDI" in names and "AHI" not in names  # OSA has 'REM AHI' etc., not bare 'AHI'
    assert set(np.unique(y)) <= {0, 1}


def test_psg_feature_constants():
    # leakage sets are subsets of the actual feature names
    _, _, sdb = data.load_sdb()
    _, _, osa = data.load_osa()
    assert set(data.SDB_PSG_FEATURES) <= set(sdb)
    assert set(data.OSA_PSG_FEATURES) <= set(osa)
    assert len(data.SDB_PSG_FEATURES) == 6
    assert len(data.OSA_PSG_FEATURES) == 14
