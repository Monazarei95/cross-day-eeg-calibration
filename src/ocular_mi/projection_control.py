"""Paired rank-matched development control for the eye-calibrated projection."""

import argparse
import shutil
import tomllib
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from .config import PilotConfig
from .contrastive import Encoder
from .data import (
    calibration_valid_samples,
    cue_windows,
    development_path,
    load_development_session,
)
from .features import bandpass, regularized_covariances
from .ocular import fit_eog_map, ocular_projection
from .pilot import calibration_draw, classifier
from .results import file_sha256, manifest, write_csv, write_json
from .runlog import record_run
from .ssl_screen import encode


def projection_matrices(session, pilot, config):
    cal = session.calibration
    eeg, eog = session.eeg_uv[:, cal], session.eog_uv[:, cal]
    valid = calibration_valid_samples(eeg, eog)
    mapping = fit_eog_map(eeg[:, valid], eog[:, valid], pilot.regression_ridge).astype(np.float32)
    eye = ocular_projection(mapping)
    n = eye.shape[0]
    rank = round(n - np.trace(eye))
    matrices = {"raw": np.eye(n), "ocular": eye}
    rng = np.random.default_rng([config["projection_seed"], session.participant])
    for rep in range(config["random_projections"]):
        basis, _ = np.linalg.qr(rng.standard_normal((n, rank)))
        matrices[f"random_{rep + 1:02d}"] = np.eye(n) - basis @ basis.T
    return matrices


def motor_trials(eeg_uv, cue_samples, fs, pilot):
    start, stop = pilot.windows["late"]
    pad = pilot.filter_context_seconds
    bounds = (start - pad, stop + pad)
    epochs = cue_windows(eeg_uv, cue_samples, fs, bounds)
    offset = round(bounds[0] * fs)
    return bandpass(epochs, fs, pilot.bands["motor"])[
        ..., round(start * fs) - offset : round(stop * fs) - offset
    ]


def evaluate(session, pilot, config, net, out, emit):
    p = session.participant
    x = motor_trials(session.eeg_uv, session.cue_samples, session.sampling_rate, pilot)
    folds = []
    for run in np.unique(session.runs):
        seed = np.random.SeedSequence([pilot.seed, p, int(run)]).spawn(1)[0]
        rng = np.random.default_rng(seed)
        train = np.flatnonzero(session.runs != run)
        test = np.flatnonzero(session.runs == run)
        draws = [
            calibration_draw(session.labels, train, pilot.labels_per_class, rng)
            for _ in range(pilot.calibration_draws)
        ]
        emit(
            "fold",
            participant=p,
            run=int(run) + 1,
            test=(test + 1).tolist(),
            labels=session.labels[test].tolist(),
            draws=[(d + 1).tolist() for d in draws],
        )
        folds.append((int(run) + 1, test, draws))
    matrices = projection_matrices(session, pilot, config)
    np.savez(out / f"projections_{p:02d}.npz", **matrices)
    scores = []
    for condition, matrix in matrices.items():
        transformed = x if condition == "raw" else matrix.astype(np.float32) @ x
        features = {"classical": regularized_covariances(transformed, pilot.covariance_shrinkage)}
        if net is not None:
            features["frozen_ssl"] = encode(net, transformed[..., ::2] / 10)
        for method, values in features.items():
            results = []
            for run, test, draws in folds:
                for i, draw in enumerate(draws, 1):
                    probe = classifier(method == "classical").fit(
                        values[draw], session.labels[draw]
                    )
                    pred = probe.predict(values[test])
                    results.append(balanced_accuracy_score(session.labels[test], pred))
                    emit(
                        "predicted",
                        participant=p,
                        run=run,
                        draw=i,
                        method=method,
                        projection=condition,
                        predictions=pred.tolist(),
                    )
            scores.append(
                dict(
                    participant=p,
                    method=method,
                    projection=condition,
                    accuracy=float(np.mean(results)),
                )
            )
    emit("participant_completed", participant=p)
    return scores


def summarize(rows):
    result = {}
    for method in sorted({r["method"] for r in rows}):
        paired = []
        for p in sorted({r["participant"] for r in rows if r["method"] == method}):
            scores = {
                r["projection"]: r["accuracy"]
                for r in rows
                if r["participant"] == p and r["method"] == method
            }
            random_mean = float(np.mean([v for k, v in scores.items() if k.startswith("random_")]))
            paired.append(
                dict(
                    participant=p,
                    raw=scores["raw"],
                    ocular=scores["ocular"],
                    random_mean=random_mean,
                    ocular_minus_random=scores["ocular"] - random_mean,
                    ocular_minus_raw=scores["ocular"] - scores["raw"],
                )
            )
        result[method] = {
            "participants": paired,
            "mean": {
                k: float(np.mean([r[k] for r in paired])) for k in paired[0] if k != "participant"
            },
            "beat_both": sum(r["ocular"] > max(r["raw"], r["random_mean"]) for r in paired),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("config", "data", "encoder-run", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    pilot_path = args.config.parent / config["pilot_config"]
    pilot = PilotConfig.from_toml(pilot_path)
    if config["random_projections"] < 1:
        raise ValueError("At least one random projection is required")
    parent_config = args.encoder_run / "config.toml"
    parent = tomllib.loads(parent_config.read_text())
    checkpoint = args.encoder_run / "uniform.pt"
    if file_sha256(checkpoint) != config["uniform_checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the configured encoder")
    if (
        set(parent["targets"]) & set(parent["sources"])
        or parent["window_s"] != pilot.windows["late"]
        or parent["band_hz"] != pilot.bands["motor"]
    ):
        raise ValueError("Encoder participant or input configuration mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    net = Encoder(22).eval()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    out = args.output
    with (
        threadpool_limits(limits=1),
        record_run(out, {k: str(v) for k, v in vars(args).items()}) as emit,
    ):
        for path, name in [
            (args.config, "config.toml"),
            (pilot_path, "pilot.toml"),
            (parent_config, "encoder.toml"),
        ]:
            shutil.copyfile(path, out / name)
        files = [development_path(args.data, p) for p in pilot.participants]
        metadata = manifest(args.config, files + [checkpoint, parent_config, pilot_path])
        metadata["packages"]["torch"] = torch.__version__
        write_json(out / "manifest.json", metadata)
        rows = []
        for p in pilot.participants:
            session = load_development_session(args.data, p)
            rows.extend(
                evaluate(
                    session, pilot, config, net if p in parent["targets"] else None, out, emit
                )
            )
        write_csv(out / "projection_scores.csv", rows)
        write_json(out / "summary.json", summarize(rows))
        write_json(
            out / "hashes.json",
            {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.name != "events.jsonl"},
        )


if __name__ == "__main__":
    main()
