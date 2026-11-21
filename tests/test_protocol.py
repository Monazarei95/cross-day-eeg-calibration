"""Tests for split boundaries and the ocular-sensitivity controls."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ocular_mi.config import PilotConfig
from ocular_mi.data import (
    calibration_valid_samples,
    cue_windows,
    development_path,
    load_development_session,
)
from ocular_mi.ocular import bounded_exchange, fit_eog_map, ocular_projection, rms, select_donors
from ocular_mi.pilot import calibration_draw

ROOT = Path(__file__).resolve().parents[1]


def test_missing_marker_must_match_all_eeg_and_eog_channels() -> None:
    eeg = np.zeros((22, 4))
    eog = np.zeros((3, 4))
    eeg[:, :2] = -1600
    eog[:, 0] = -16000
    eog[1, 3] = np.nan
    np.testing.assert_array_equal(calibration_valid_samples(eeg, eog), [False, True, True, False])


def test_late_filter_context_excludes_cue() -> None:
    config = PilotConfig.from_toml(ROOT / "configs/bci2a_pilot.toml")
    with pytest.raises(ValueError, match="follow cue offset"):
        replace(config, windows={"cue": [0, 2], "late": [1.5, 3.5]}).validate()


def test_heldout_trial_never_enters_calibration_draw() -> None:
    labels = np.tile([1, 2, 3, 4], 12)
    training = np.arange(24, 48)
    sample = calibration_draw(labels, training, 3, np.random.default_rng(4))
    np.testing.assert_array_equal(np.bincount(labels[sample]), [0, 3, 3, 3, 3])
    assert set(sample) <= set(training)
    np.testing.assert_array_equal(
        sample, calibration_draw(labels, training, 3, np.random.default_rng(4))
    )


def test_paired_sham_preserves_energy_and_donor_choices() -> None:
    rng = np.random.default_rng(2)
    labels = np.tile([1, 2, 3, 4], 3)
    donors = select_donors(labels, 2, rng)
    assert not (donors == np.arange(12)[:, None]).any()
    np.testing.assert_array_equal(labels[donors], np.tile(np.repeat([1, 2, 3, 4], 2), (12, 1)))
    projected = rng.standard_normal((12, 5, 40))
    delta, sham = bounded_exchange(projected, donors, 0.3, np.roll(np.arange(5), 2))
    np.testing.assert_allclose(rms(sham), rms(delta), atol=1e-12)
    assert rms(delta).max() <= 0.3 + 1e-12


def test_regression_and_ssp_remove_known_ocular_subspace() -> None:
    rng = np.random.default_rng(5)
    ocular = rng.standard_normal((3, 400))
    expected = rng.standard_normal((8, 3))
    mapping = fit_eog_map(expected @ ocular, ocular, ridge=0)
    np.testing.assert_allclose(mapping, expected)
    projection = ocular_projection(mapping)
    np.testing.assert_allclose(projection @ mapping, 0, atol=1e-12)
    np.testing.assert_allclose(projection @ projection, projection, atol=1e-12)


def test_windows_convert_cue_relative_bounds_to_samples() -> None:
    signals = np.arange(30).reshape(1, -1)
    epochs = cue_windows(signals, np.array([10, 20]), fs=10, window=(-0.2, 0.3))
    np.testing.assert_array_equal(epochs[:, 0], [range(8, 13), range(18, 23)])


@pytest.mark.parametrize("participant", [1, 4])
def test_original_development_files(participant: int) -> None:
    root = ROOT / "BCICIV_2a"
    if not development_path(root, participant).exists():
        pytest.skip("Original BCI IV 2a recordings unavailable")
    session = load_development_session(root, participant)
    assert session.eeg_uv.shape[0] == 22
    assert session.eog_uv.shape[0] == 3
    np.testing.assert_array_equal(np.bincount(session.runs), [48] * 6)
    np.testing.assert_array_equal(np.bincount(session.labels), [0, 72, 72, 72, 72])
    assert session.calibration.stop - session.calibration.start >= 10 * session.sampling_rate
