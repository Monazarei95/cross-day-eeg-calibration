"""Read and validate the original Graz BCI IV 2a development recordings."""

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from numpy.typing import NDArray

EEG_CHANNELS = tuple(
    "EEG-Fz EEG-0 EEG-1 EEG-2 EEG-3 EEG-4 EEG-5 EEG-C3 EEG-6 EEG-Cz EEG-7 "
    "EEG-C4 EEG-8 EEG-9 EEG-10 EEG-11 EEG-12 EEG-13 EEG-14 EEG-Pz EEG-15 EEG-16".split()
)
EOG_CHANNELS = ("EOG-left", "EOG-central", "EOG-right")


def calibration_valid_samples(eeg_uv: NDArray, eog_uv: NDArray) -> NDArray:
    """Detect the minimum-int16 missing code after Graz 2a scaling to microvolts."""
    marker = (eeg_uv == -1600).all(axis=0) & (eog_uv == -16000).all(axis=0)
    finite = np.isfinite(eeg_uv).all(axis=0) & np.isfinite(eog_uv).all(axis=0)
    return finite & ~marker


@dataclass(frozen=True)
class Session:
    participant: int
    sampling_rate: int
    eeg_uv: NDArray
    eog_uv: NDArray
    calibration: slice
    cue_samples: NDArray
    labels: NDArray
    runs: NDArray
    flagged: NDArray


def development_path(root: Path, participant: int) -> Path:
    if participant not in range(1, 10):
        raise ValueError("BCI IV 2a participant must be between 1 and 9")
    return root / f"A{participant:02d}T.gdf"


def load_development_session(root: Path, participant: int) -> Session:
    path = development_path(root, participant)
    raw = mne.io.read_raw_gdf(
        path, eog=list(EOG_CHANNELS), stim_channel=None, preload=False, verbose="ERROR"
    )
    try:
        if tuple(raw.ch_names) != EEG_CHANNELS + EOG_CHANNELS or raw.info["sfreq"] != 250:
            raise ValueError(f"{path.name}: unexpected channel order, names or sample rate")
        fs = int(raw.info["sfreq"])
        marks = raw.annotations
        samples = np.rint(marks.onset * fs).astype(int)
        codes = marks.description.astype(int)
        starts = samples[codes == 768]
        cues = (codes >= 769) & (codes <= 772)
        cue_samples = samples[cues]
        labels = codes[cues] - 768
        if len(starts) != 288 or len(labels) != 288 or not np.all(cue_samples - starts == 2 * fs):
            raise ValueError(f"{path.name}: invalid trial starts or cue sequence")
        run_starts = samples[codes == 32766]
        runs = np.searchsorted(run_starts, starts, side="right") - 1
        unique_runs, runs = np.unique(runs, return_inverse=True)
        if len(unique_runs) != 6 or unique_runs[0] < 0:
            raise ValueError(f"{path.name}: expected six imagery runs")
        for run in range(6):
            counts = np.bincount(labels[runs == run], minlength=5)
            if not np.array_equal(counts, [0, 12, 12, 12, 12]):
                raise ValueError(f"{path.name}: unexpected number of trials per run and class")

        artifact_samples = samples[codes == 1023]
        artifact_trials = np.searchsorted(starts, artifact_samples, side="right") - 1
        if np.any(artifact_trials < 0) or np.any(
            artifact_samples >= starts[artifact_trials] + 6 * fs
        ):
            raise ValueError(f"{path.name}: artifact marker outside trial")
        flagged = np.zeros(len(starts), dtype=bool)
        flagged[artifact_trials] = True

        eye = np.flatnonzero(codes == 1072)
        if eye.size != 1:
            raise ValueError(f"{path.name}: expected one eye-movement calibration block")
        start = samples[eye[0]]
        stop = start + round(marks.duration[eye[0]] * fs)
        if not 0 <= start < stop <= starts[0]:
            raise ValueError(f"{path.name}: invalid eye-movement calibration interval")
        signals = (raw.get_data() * 1e6).astype(np.float32)
        valid = calibration_valid_samples(signals[:22], signals[22:])
        signals[:, ~valid] = np.nan
    finally:
        raw.close()
    return Session(
        participant=participant,
        sampling_rate=fs,
        eeg_uv=signals[: len(EEG_CHANNELS)],
        eog_uv=signals[len(EEG_CHANNELS) :],
        calibration=slice(int(start), int(stop)),
        cue_samples=cue_samples,
        labels=labels,
        runs=runs,
        flagged=flagged,
    )


def cue_windows(signal: NDArray, cues: NDArray, fs: int, window: tuple[float, float]) -> NDArray:
    start, stop = (round(t * fs) for t in window)
    if np.min(cues) + start < 0 or np.max(cues) + stop > signal.shape[-1] or start >= stop:
        raise ValueError("Requested epoch crosses the recording boundary or has no samples")
    epochs = np.stack([signal[:, cue + start : cue + stop] for cue in cues])
    if not np.isfinite(epochs).all():
        raise ValueError("Non-finite samples inside requested trial windows")
    return epochs
