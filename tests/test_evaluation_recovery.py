import hashlib
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest
from scipy.io import savemat

from ocular_mi.evaluation import cached_labels


def test_recovery_binds_labels_to_original_source_and_detects_changes(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "recover_openbmi.py"
    extract = run_path(str(script))["extract"]
    source = tmp_path / "source.mat"
    y = np.tile([1, 2], 50)
    savemat(source, {"EEG_MI_train": {"y_dec": y}})
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = tmp_path / "labels"
    with pytest.raises(ValueError, match="Source differs"):
        extract(source, cache, 2, "wrong", "EEG_MI_train")
    extract(source, cache, 2, original, "EEG_MI_train")
    np.testing.assert_array_equal(cached_labels(cache, 2, original), y)
    with pytest.raises(ValueError, match="another source"):
        cached_labels(cache, 2, "wrong")
    np.savez(cache / "labels_02.npz", y=y[::-1])
    with pytest.raises(ValueError, match="changed"):
        cached_labels(cache, 2, original)
