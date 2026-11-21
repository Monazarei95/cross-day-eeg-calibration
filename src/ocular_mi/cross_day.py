"""Generate frozen day-two predictions without opening evaluation label files."""

import argparse
import shutil
import tomllib
from functools import partial
from pathlib import Path

import mne
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .config import PilotConfig
from .contrastive import train
from .data import EEG_CHANNELS, EOG_CHANNELS, development_path, load_development_session
from .features import regularized_covariances
from .pilot import calibration_draw, classifier
from .projection_control import motor_trials, projection_matrices
from .results import file_sha256, manifest, write_json
from .runlog import record_run
from .ssl_screen import encode, source_data


def evaluation_trials(path, pilot):
    raw = mne.io.read_raw_gdf(path, eog=list(EOG_CHANNELS), stim_channel=None, verbose="ERROR")
    try:
        if tuple(raw.ch_names) != EEG_CHANNELS + EOG_CHANNELS or raw.info["sfreq"] != 250:
            raise ValueError(f"Invalid channel names or sample rate in {path.name}")
        codes = raw.annotations.description.astype(int)
        samples = np.rint(raw.annotations.onset * 250).astype(int)
        cues, starts = samples[codes == 783], samples[codes == 768]
        if (
            len(cues) != 288
            or len(starts) != 288
            or not np.all(np.diff(cues) > 0)
            or not np.all(cues - starts == 500)
        ):
            raise ValueError(f"Invalid chronological trial markers in {path.name}")
        eeg = (raw.get_data(picks=list(EEG_CHANNELS)) * 1e6).astype(np.float32)
        eeg[:, (eeg == -1600).all(axis=0)] = np.nan
        return motor_trials(eeg, cues, 250, pilot), cues
    finally:
        raw.close()


def calibration_pca(session, rank):
    cal = session.eeg_uv[:, session.calibration]
    samples = cal[:, np.isfinite(cal).all(axis=0)].astype(np.float64)
    values, basis = np.linalg.eigh(np.cov(samples))
    directions = basis[:, np.argsort(-values)[:rank]]
    return np.eye(len(cal)) - directions @ directions.T


def predict(session, evaluation, net, pilot, config, projection_config, out, emit):
    p = session.participant
    x = motor_trials(session.eeg_uv, session.cue_samples, session.sampling_rate, pilot)
    matrices = projection_matrices(session, pilot, projection_config)
    rank = round(x.shape[1] - np.trace(matrices["ocular"]))
    matrices["pca"] = calibration_pca(session, rank)
    np.savez(out / f"projections_{p:02d}.npz", **matrices)
    rng = np.random.default_rng([config["draw_seed"], p])
    draws = [
        calibration_draw(session.labels, np.arange(len(x)), pilot.labels_per_class, rng)
        for _ in range(pilot.calibration_draws)
    ]
    emit("draws", participant=p, trial_ids=[(d + 1).tolist() for d in draws], removed_rank=rank)
    # The transformations below are per trial. All fitted statistics use day-one draws.
    both = np.concatenate([x, evaluation])
    predictions = {}
    for name, matrix in matrices.items():
        transformed = both if name == "raw" else matrix.astype(np.float32) @ both
        features = {
            "classical": regularized_covariances(transformed, pilot.covariance_shrinkage),
            "frozen_ssl": encode(net, transformed[..., ::2] / 10),
        }
        for method, values in features.items():
            predictions[f"{method}.{name}"] = np.stack(
                [
                    classifier(method == "classical")
                    .fit(values[draw], session.labels[draw])
                    .predict(values[len(x) :])
                    for draw in draws
                ]
            ).astype(np.int8)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    paths = {
        key: args.config.parent / config[key + "_config"]
        for key in ("pilot", "encoder", "projection")
    }
    pilot = PilotConfig.from_toml(paths["pilot"])
    encoder = tomllib.loads(paths["encoder"].read_text())
    projection = tomllib.loads(paths["projection"].read_text())
    if encoder["window_s"] != pilot.windows["late"] or encoder["band_hz"] != pilot.bands["motor"]:
        raise ValueError("Pretraining and probe windows and bands must agree")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = args.output
    with (
        threadpool_limits(limits=1),
        record_run(out, {k: str(v) for k, v in vars(args).items()}) as emit,
    ):
        for key, path in (paths | {"config": args.config}).items():
            shutil.copyfile(path, out / f"{key}.toml")
        files = [development_path(args.data, p) for p in pilot.participants]
        eval_files = {p: args.data / f"A{p:02d}E.gdf" for p in pilot.participants}
        metadata = manifest(args.config, files + list(eval_files.values()) + list(paths.values()))
        metadata["packages"]["torch"] = torch.__version__
        write_json(out / "manifest.json", metadata)
        sources = {
            p: source_data(load_development_session(args.data, p), encoder)
            for p in pilot.participants
        }
        prediction_hashes = {}
        for p in pilot.participants:
            training = {key: value for key, value in sources.items() if key != p}
            emit("outer_fold", participant=p, pretraining_participants=list(training))
            net = train(training, encoder, "uniform", partial(emit, participant=p))
            torch.save(net.state_dict(), out / f"encoder_{p:02d}.pt")
            evaluation, cues = evaluation_trials(eval_files[p], pilot)
            session = load_development_session(args.data, p)
            predictions = predict(session, evaluation, net, pilot, config, projection, out, emit)
            path = out / f"predictions_{p:02d}.npz"
            np.savez(path, cue_samples=cues, **predictions)
            prediction_hashes[path.name] = file_sha256(path)
            emit("predictions_saved", participant=p, sha256=prediction_hashes[path.name])
        write_json(out / "prediction_hashes.json", prediction_hashes)


if __name__ == "__main__":
    main()
