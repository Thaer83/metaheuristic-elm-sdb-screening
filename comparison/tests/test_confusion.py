import numpy as np
import pandas as pd
from comparison import models
from comparison.confusion import mean_confusion, MHA


def test_confusion_matches_reported():
    for ds in ["OSA", "SDB"]:
        df = pd.read_csv(MHA[ds]).rename(columns={"algo": "model"})
        cm = mean_confusion(ds)
        assert list(cm.model) == models.OPTIMIZED[ds]
        for _, r in cm.iterrows():
            s = df[df.model == r.model]
            # averaged cells equal the raw per-run column means
            assert np.isclose(r.TP, s.tp.mean()) and np.isclose(r.TN, s.tn.mean())
            assert np.isclose(r.FP, s.fp.mean()) and np.isclose(r.FN, s.fn.mean())
            # the test set is fixed, so accuracy and recall from the averaged matrix
            # equal the reported per-run means exactly
            assert np.isclose(r.accuracy, s.accuracy.mean(), atol=1e-9)
            assert np.isclose(r.recall, s.recall.mean(), atol=1e-9)
            # precision and F1 are pooled estimates, very close to the reported means
            assert abs(r.precision - s.precision.mean()) < 0.005
            assert abs(r.f1 - s.f1.mean()) < 0.005
