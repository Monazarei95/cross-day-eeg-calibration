import numpy as np
import pytest

pytest.importorskip("torch")
from ocular_mi.openbmi import balanced_draws, labels, pca_matrix


def test_calibration_removes_three_directions_and_ignores_non_eye_tasks():
    rng = np.random.default_rng(195)
    names = np.array([f"ch{i:02d}" for i in reversed(range(62))])
    epochs = rng.normal(size=(10000, 5, 62)).astype(np.float32)
    rec = dict(smt=epochs, chan=names, fs=1000, y_dec=np.arange(71, 76))
    cfg = dict(eye_codes=[71, 72, 73], rank_removed=3)
    project, channels = pca_matrix(rec, cfg)
    assert np.array_equal(channels, np.sort(names))
    assert np.linalg.matrix_rank(project, tol=1e-6) == 59
    np.testing.assert_allclose(project @ project, project, atol=1e-10)
    epochs[:, 3:, :] *= 1000
    np.testing.assert_allclose(pca_matrix(rec, cfg)[0], project)


def test_draws_are_paired_balanced_and_repeatable():
    y = labels({"y_dec": np.tile([1, 2], 50)})
    config = dict(draw_seed=81573, classes=[1, 2], draws=5, per_class=10)
    draws = balanced_draws(y, 1, config)
    np.testing.assert_array_equal(draws, balanced_draws(y, 1, config))
    for draw in draws:
        assert len(np.unique(draw)) == 20
        np.testing.assert_array_equal(np.unique(y[draw], return_counts=True)[1], [10, 10])
