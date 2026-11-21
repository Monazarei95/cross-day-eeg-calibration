"""Windowed filters, EOG descriptors and training-fold covariance features."""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin

from .ocular import centered


def bandpass(epochs: NDArray, fs: int, band: list[float]) -> NDArray:
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, epochs, axis=-1).astype(np.float32)


def eog_features(epochs: NDArray, baseline: NDArray) -> NDArray:
    horizontal = epochs[:, 0] - epochs[:, 2]
    vertical = epochs[:, 1] - (epochs[:, 0] + epochs[:, 2]) / 2
    baseline_horizontal = baseline[:, 0] - baseline[:, 2]
    baseline_vertical = baseline[:, 1] - (baseline[:, 0] + baseline[:, 2]) / 2
    contrasts = np.stack((horizontal - baseline_horizontal, vertical - baseline_vertical), axis=1)
    pieces = np.array_split(contrasts, 4, axis=-1)
    features = [
        statistic
        for piece in pieces
        for statistic in (
            piece.mean(axis=-1),
            piece.std(axis=-1),
            np.abs(np.diff(piece, axis=-1)).mean(axis=-1),
        )
    ]
    return np.concatenate(features, axis=-1)


def regularized_covariances(epochs: NDArray, shrinkage: float) -> NDArray:
    epochs = centered(epochs.astype(np.float64))
    cov = epochs @ np.swapaxes(epochs, -1, -2) / (epochs.shape[-1] - 1)
    size = epochs.shape[-2]
    average_power = np.maximum(np.trace(cov, axis1=-2, axis2=-1) / size, 1e-8)
    return (1 - shrinkage) * cov + shrinkage * average_power[..., None, None] * np.eye(size)


def spectral_function(matrices: NDArray, function) -> NDArray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    return (eigenvectors * function(eigenvalues)[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)


class TangentSpace(TransformerMixin, BaseEstimator):
    """Tangent-space features with a log-Euclidean reference fit on training trials."""

    def fit(self, covariances: NDArray, y=None):
        mean_log = spectral_function(covariances, np.log).mean(axis=0)
        self.whitener_ = spectral_function(mean_log, lambda values: np.exp(-0.5 * values))
        self.rows_, self.cols_ = np.triu_indices(mean_log.shape[0])
        self.scales_ = np.where(self.rows_ == self.cols_, 1, np.sqrt(2))
        return self

    def transform(self, covariances: NDArray) -> NDArray:
        centered_covariances = self.whitener_ @ covariances @ self.whitener_
        tangent = spectral_function(centered_covariances, np.log)
        return tangent[..., self.rows_, self.cols_] * self.scales_
