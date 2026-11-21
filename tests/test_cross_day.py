import numpy as np
import pytest

from ocular_mi.score_cross_day import class_recalls, participant_scores


def test_locked_scoring_averages_draws_and_random_controls():
    y = np.tile(np.arange(1, 5), 72)
    correct = np.stack([y, y])
    partly = np.stack([y, (y % 4) + 1])
    data = {"cue_samples": np.arange(288)}
    for method in ("classical", "frozen_ssl"):
        for name in ("raw", "ocular", "pca", "random_01", "random_02"):
            data[f"{method}.{name}"] = correct if name in ("ocular", "random_01") else partly
    scores, rows = participant_scores(data, y, 1, 2, 2)
    assert len(scores) == 20
    assert rows[0]["random"] == 0.75 and rows[1]["raw"] == 0.5 and rows[0]["ocular"] == 1.0
    with pytest.raises(ValueError):
        class_recalls(np.ones(288), correct)
    data["cue_samples"] = np.zeros(288)
    with pytest.raises(ValueError):
        participant_scores(data, y, 1, 2, 2)
