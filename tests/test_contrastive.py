import numpy as np
import pytest

pytest.importorskip("torch")
from ocular_mi.contrastive import negative_indices


def test_native_negatives_are_distinct_and_closer_than_uniform():
    d = np.arange(48, dtype=float)[:, None]
    results = {}
    for mode in ("native", "uniform"):
        ids = negative_indices(d, mode, 8, 0.2, np.random.default_rng(7))
        assert all(len(set(row)) == 8 and i not in row for i, row in enumerate(ids))
        results[mode] = np.square(d[:, None] - d[ids]).mean()
    assert results["native"] < results["uniform"]
