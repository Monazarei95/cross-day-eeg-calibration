"""Within-day, held-out-run development evaluation for BCI IV 2a."""

from collections.abc import Iterator
from functools import partial

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import PilotConfig
from .data import Session, calibration_valid_samples, cue_windows
from .features import TangentSpace, bandpass, eog_features, regularized_covariances
from .ocular import (
    bounded_exchange,
    centered,
    cross_validate_map,
    fit_eog_map,
    ocular_projection,
    rms,
    select_donors,
)


def calibration_draw(
    labels: NDArray, training: NDArray, per_class: int, rng: np.random.Generator
) -> NDArray:
    indices = [
        rng.choice(training[labels[training] == label], per_class, replace=False)
        for label in (1, 2, 3, 4)
    ]
    return np.sort(np.concatenate(indices))


def classifier(tangent_space: bool):
    preprocessing = [TangentSpace()] if tangent_space else []
    return make_pipeline(*preprocessing, StandardScaler(), LogisticRegression(max_iter=1000))


def _record(
    session: Session,
    draw: int,
    indices: NDArray,
    high_load: NDArray,
    predictions: NDArray,
    window: str,
    model: str,
    band: str,
    ocular_flips: NDArray | None = None,
    sham_flips: NDArray | None = None,
) -> Iterator[dict]:
    for i, index in enumerate(indices):
        yield {
            "participant": session.participant,
            "trial": int(index) + 1,
            "run": int(session.runs[index]) + 1,
            "draw": draw,
            "window": window,
            "model": model,
            "band": band,
            "label": int(session.labels[index]),
            "prediction": int(predictions[i]),
            "high_eog": bool(high_load[i]),
            "expert_flagged": bool(session.flagged[index]),
            "ocular_flip_rate": float(ocular_flips[i]) if ocular_flips is not None else None,
            "sham_flip_rate": float(sham_flips[i]) if sham_flips is not None else None,
        }


def audit_session(session: Session, config: PilotConfig, emit=None) -> tuple[dict, list[dict]]:
    fs = session.sampling_rate
    eye_eeg = session.eeg_uv[:, session.calibration]
    eye_eog = session.eog_uv[:, session.calibration]
    valid = calibration_valid_samples(eye_eeg, eye_eog)
    if emit is not None:
        emit(
            "calibration_input",
            participant=session.participant,
            raw_start_sample=session.calibration.start,
            raw_stop_sample=session.calibration.stop,
            excluded_offsets=np.flatnonzero(~valid).tolist(),
        )
    eye_eeg, eye_eog = eye_eeg[:, valid], eye_eog[:, valid]
    real, shifted = cross_validate_map(
        eye_eeg,
        eye_eog,
        fs,
        config.calibration_folds,
        config.calibration_gap_seconds,
        config.regression_ridge,
        on_fold=partial(emit, "calibration_fold", participant=session.participant)
        if emit
        else None,
    )
    calibration = {
        "participant": session.participant,
        "samples": eye_eeg.shape[-1],
        "excluded_samples": int((~valid).sum()),
        "heldout_r2": real,
        "shifted_eog_r2": shifted,
    }
    mapping = fit_eog_map(eye_eeg, eye_eog, config.regression_ridge).astype(np.float32)
    projection = ocular_projection(mapping).astype(np.float32)

    split_plan = []
    for run in np.unique(session.runs):
        seed = np.random.SeedSequence([config.seed, session.participant, int(run)])
        draw_rng, donor_rng, sham_rng = [np.random.default_rng(s) for s in seed.spawn(3)]
        train = np.flatnonzero(session.runs != run)
        test = np.flatnonzero(session.runs == run)
        draws = [
            calibration_draw(session.labels, train, config.labels_per_class, draw_rng)
            for _ in range(config.calibration_draws)
        ]
        donors = select_donors(session.labels[test], config.donors_per_class, donor_rng)
        indices = np.arange(session.eeg_uv.shape[0])
        # A nonzero shift moves every sensor and preserves the injected EEG energy.
        sham = np.roll(indices, int(sham_rng.integers(1, len(indices))))
        split_plan.append((train, test, draws, donors, sham))
        if emit is not None:
            emit(
                "fold_planned",
                participant=session.participant,
                heldout_run=int(run) + 1,
                test_trials=(test + 1).tolist(),
                calibration_draws=[(draw + 1).tolist() for draw in draws],
                donor_trials=(test[donors] + 1).tolist(),
                sham_channel_indices=sham.tolist(),
            )

    fixation = cue_windows(session.eog_uv, session.cue_samples, fs, (-1.0, -0.25)).mean(
        axis=-1, keepdims=True
    )
    rows = []
    shrinkage = config.covariance_shrinkage
    pad = config.filter_context_seconds
    for window_name, (start, stop) in config.windows.items():
        context = (start - pad, stop + pad)
        analysis = slice(
            round(start * fs) - round(context[0] * fs),
            round(stop * fs) - round(context[0] * fs),
        )
        eeg_epochs = cue_windows(session.eeg_uv, session.cue_samples, fs, context)
        eog_epochs = cue_windows(session.eog_uv, session.cue_samples, fs, context)
        eog_only = eog_features(eog_epochs[..., analysis], fixation)
        ocular_component = mapping @ centered(eog_epochs)
        load = rms(centered(ocular_component[..., analysis]))
        for train, test, draws, _, _ in split_plan:
            high_load = load[test] >= np.quantile(load[train], config.ocular_load_quantile)
            for draw_number, draw in enumerate(draws, start=1):
                model = classifier(tangent_space=False).fit(eog_only[draw], session.labels[draw])
                pred = model.predict(eog_only[test])
                rows.extend(
                    _record(
                        session,
                        draw_number,
                        test,
                        high_load,
                        pred,
                        window_name,
                        "eog_only_diagnostic",
                        "eog",
                    )
                )

        for band_name, band in config.bands.items():
            eeg = bandpass(eeg_epochs, fs, band)[..., analysis]
            ocular = bandpass(ocular_component, fs, band)[..., analysis]
            covariances = {
                "eeg_only": regularized_covariances(eeg, shrinkage),
                "eog_regression": regularized_covariances(eeg - ocular, shrinkage),
                "ocular_ssp": regularized_covariances(projection @ eeg, shrinkage),
            }
            for train, test, draws, donors, sham in split_plan:
                high_load = load[test] >= np.quantile(load[train], config.ocular_load_quantile)
                ceiling = float(np.quantile(rms(ocular[train]), config.perturbation_quantile))
                if emit is not None:
                    emit(
                        "fold_condition",
                        participant=session.participant,
                        heldout_run=int(session.runs[test[0]]) + 1,
                        window=window_name,
                        band=band_name,
                        high_load_threshold=float(
                            np.quantile(load[train], config.ocular_load_quantile)
                        ),
                        exchange_rms_cap=ceiling,
                    )
                exchange, control = bounded_exchange(ocular[test], donors, ceiling, sham)
                reference = eeg[test, None]
                perturbed = [
                    regularized_covariances(reference + delta, shrinkage)
                    for delta in (exchange, control)
                ]
                for draw_number, draw in enumerate(draws, start=1):
                    for model_name, cov in covariances.items():
                        model = classifier(tangent_space=True).fit(cov[draw], session.labels[draw])
                        pred = model.predict(cov[test])
                        flips = [None, None]
                        if model_name == "eeg_only":
                            flips = [
                                (
                                    model.predict(changed.reshape(-1, *cov.shape[-2:])).reshape(
                                        donors.shape
                                    )
                                    != pred[:, None]
                                ).mean(axis=1)
                                for changed in perturbed
                            ]
                        rows.extend(
                            _record(
                                session,
                                draw_number,
                                test,
                                high_load,
                                pred,
                                window_name,
                                model_name,
                                band_name,
                                *flips,
                            )
                        )
    return calibration, rows
