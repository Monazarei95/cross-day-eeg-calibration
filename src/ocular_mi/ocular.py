"""Calibration-only EOG regression, spatial projection, and paired perturbations."""

import numpy as np
from numpy.typing import NDArray


def fit_eog_map(eeg: NDArray, eog: NDArray, ridge: float) -> NDArray:
    eeg_centered = eeg.astype(np.float64) - eeg.mean(axis=-1, keepdims=True)
    eog_centered = eog.astype(np.float64) - eog.mean(axis=-1, keepdims=True)
    gram = eog_centered @ eog_centered.T
    penalty = ridge * np.trace(gram) / len(gram)
    if np.trace(gram) <= 0:
        raise ValueError("EOG calibration has no variance")
    return np.linalg.solve(gram + penalty * np.eye(len(gram)), eog_centered @ eeg_centered.T).T


def cross_validate_map(
    eeg: NDArray,
    eog: NDArray,
    fs: int,
    folds: int,
    gap_seconds: float,
    ridge: float,
    on_fold=None,
) -> tuple[float, float]:
    """Compute pooled held-out R² for real and time-shifted calibration predictors."""
    n = eeg.shape[-1]
    boundaries = np.linspace(0, n, folds + 1, dtype=int)
    gap = round(gap_seconds * fs)
    null_error = 0.0
    real_error = 0.0
    shifted_error = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        samples = np.arange(n)
        train = (samples < left - gap) | (samples >= right + gap)
        y_train, e_train = eeg[:, train], eog[:, train]
        y_test, e_test = eeg[:, left:right], eog[:, left:right]
        if min(y_train.shape[-1], y_test.shape[-1]) < 2 * fs:
            raise ValueError("Calibration split must supply at least two seconds per partition")
        y_train_mean = y_train.mean(axis=-1, keepdims=True)
        truth = y_test - y_train_mean
        null_error += np.square(truth).sum()
        real = fit_eog_map(y_train, e_train, ridge)
        real_error += np.square(truth - real @ (e_test - e_train.mean(1, keepdims=True))).sum()

        # Shift inside each contiguous block to avoid linking held-out and training samples.
        shifted = eog.copy()
        for start, stop in ((0, max(0, left - gap)), (left, right), (min(n, right + gap), n)):
            part = eog[:, start:stop]
            shifted[:, start:stop] = np.roll(part, part.shape[-1] // 2, axis=-1)
        control_train = shifted[:, train]
        control = fit_eog_map(y_train, control_train, ridge)
        control_test = shifted[:, left:right] - control_train.mean(1, keepdims=True)
        shifted_error += np.square(truth - control @ control_test).sum()
        if on_fold is not None:
            baseline_sse = float(np.square(truth).sum())
            prediction = real @ (e_test - e_train.mean(1, keepdims=True))
            on_fold(
                start_sample=int(left),
                stop_sample=int(right),
                train_samples=int(train.sum()),
                baseline_sse=baseline_sse,
                regression_sse=float(np.square(truth - prediction).sum()),
                shifted_sse=float(np.square(truth - control @ control_test).sum()),
                eeg_peak_uv=float(np.abs(y_test).max()),
                eog_peak_uv=float(np.abs(e_test).max()),
            )
    return float(1 - real_error / null_error), float(1 - shifted_error / null_error)


def ocular_projection(mapping: NDArray) -> NDArray:
    basis, singular_values, _ = np.linalg.svd(mapping, full_matrices=False)
    rank = np.sum(singular_values > singular_values[0] * 1e-10)
    ocular_basis = basis[:, :rank]
    return np.eye(mapping.shape[0]) - ocular_basis @ ocular_basis.T


def select_donors(labels: NDArray, per_class: int, rng: np.random.Generator) -> NDArray:
    """Select balanced donors within a held-out run without reusing the recipient."""
    indices = np.arange(len(labels))
    donors = []
    for recipient in indices:
        pool = indices[indices != recipient]
        donors.append(
            np.concatenate(
                [
                    rng.choice(pool[labels[pool] == category], per_class, replace=False)
                    for category in np.unique(labels)
                ]
            )
        )
    return np.stack(donors)


def centered(signal: NDArray) -> NDArray:
    return signal - signal.mean(axis=-1, keepdims=True)


def rms(signal: NDArray) -> NDArray:
    return np.sqrt(np.mean(np.square(signal), axis=(-2, -1)))


def bounded_exchange(
    projected_eog: NDArray, donors: NDArray, max_rms: float, permutation: NDArray
) -> tuple[NDArray, NDArray]:
    delta = projected_eog[donors] - projected_eog[:, None]
    magnitude = rms(delta)
    safe_magnitude = np.maximum(magnitude, np.finfo(delta.dtype).tiny)
    delta *= np.minimum(1, max_rms / safe_magnitude)[..., None, None]
    # A channel permutation preserves energy and commutes with channelwise filtering.
    return delta, delta[..., permutation, :]
