"""Development-only EEG contrastive pretraining and paired frozen-probe evaluation."""

import argparse
import shutil
import tomllib
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from .contrastive import train
from .data import cue_windows, development_path, load_development_session
from .features import bandpass, eog_features, regularized_covariances
from .ocular import bounded_exchange, centered, fit_eog_map, ocular_projection, rms, select_donors
from .pilot import calibration_draw, classifier
from .results import file_sha256, manifest, write_csv, write_json
from .runlog import record_run


def filtered_trials(session, signal, config):
    fs = session.sampling_rate
    start, stop = config["window_s"]
    bounds = (start - 0.25, stop + 0.25)
    raw = cue_windows(signal, session.cue_samples, fs, bounds)
    offset = round(start * fs) - round(bounds[0] * fs)
    size = round(stop * fs) - round(start * fs)
    filtered = bandpass(raw, fs, config["band_hz"])
    return np.ascontiguousarray(filtered[..., offset : offset + size : 2] / 10)


def source_data(session, config):
    def eye(window):
        return cue_windows(session.eog_uv, session.cue_samples, session.sampling_rate, window)

    desc = eog_features(eye(config["window_s"]), eye((-1, -0.25)).mean(axis=-1, keepdims=True))
    lower, upper = np.percentile(desc, [25, 75], axis=0)
    scaled = (desc - np.median(desc, axis=0)) / np.maximum(upper - lower, 1e-6)
    return {
        "x": filtered_trials(session, session.eeg_uv, config),
        "descriptor": scaled.clip(-5, 5),
    }


@torch.inference_mode()
def encode(net, x):
    flat = x.reshape(-1, *x.shape[-2:])
    features = np.concatenate(
        [
            net(torch.from_numpy(np.ascontiguousarray(flat[i : i + 64]))).numpy()
            for i in range(0, len(flat), 64)
        ]
    )
    return features.reshape(*x.shape[:-2], -1)


def evaluate(session, models, config, emit):
    x = filtered_trials(session, session.eeg_uv, config)
    eye = filtered_trials(session, session.eog_uv, config)
    cal = session.calibration
    eeg_cal, eye_cal = session.eeg_uv[:, cal], session.eog_uv[:, cal]
    valid = np.isfinite(eeg_cal).all(0) & np.isfinite(eye_cal).all(0)
    mapping = fit_eog_map(eeg_cal[:, valid], eye_cal[:, valid], config["ridge"])
    ocular = mapping.astype(np.float32) @ centered(eye)
    ssp_cov = regularized_covariances(ocular_projection(mapping) @ x, config["shrinkage"])
    train_ids = np.flatnonzero(~np.isin(session.runs + 1, config["heldout_runs"]))
    cap = float(np.quantile(rms(ocular[train_ids]), config["perturbation_quantile"]))
    seeds = np.random.SeedSequence([config["seed"], session.participant]).spawn(3)
    draw_rng, donor_rng, sham_rng = map(np.random.default_rng, seeds)
    labels = session.labels
    draws = [
        calibration_draw(labels, train_ids, config["labels_per_class"], draw_rng)
        for _ in range(config["draws"])
    ]
    emit(
        "calibration",
        participant=session.participant,
        excluded_samples=int((~valid).sum()),
        training_draws=[(draw + 1).tolist() for draw in draws],
        rms_cap_in_10uv_units=cap,
    )
    features = {arm: encode(net, x) for arm, net in models.items()}
    features["ssp"] = ssp_cov
    probes = {
        arm: [classifier(arm == "ssp").fit(values[draw], labels[draw]) for draw in draws]
        for arm, values in features.items()
    }
    rows = []
    for run in config["heldout_runs"]:
        ids = np.flatnonzero(session.runs + 1 == run)
        donors = select_donors(labels[ids], config["donors_per_class"], donor_rng)
        perm = np.roll(np.arange(x.shape[1]), int(sham_rng.integers(1, x.shape[1])))
        delta, sham = bounded_exchange(ocular[ids], donors, cap, perm)
        emit(
            "evaluation_run",
            participant=session.participant,
            run=run,
            trials=(ids + 1).tolist(),
            donors=(ids[donors] + 1).tolist(),
            sham_order=perm.tolist(),
        )
        for arm, values in features.items():
            if arm != "ssp":
                real_feat, sham_feat = [
                    encode(models[arm], x[ids, None] + d) for d in (delta, sham)
                ]
            for draw, probe in enumerate(probes[arm], 1):
                base = probe.predict(values[ids])
                changed = {}
                if arm != "ssp":
                    for name, feat in (("real", real_feat), ("sham", sham_feat)):
                        changed[name] = probe.predict(feat.reshape(-1, feat.shape[-1])).reshape(
                            len(ids), -1
                        )
                for i, idx in enumerate(ids):
                    entry = dict(
                        participant=session.participant,
                        run=run,
                        draw=draw,
                        arm=arm,
                        trial=int(idx) + 1,
                        label=int(labels[idx]),
                        prediction=int(base[i]),
                    )
                    for name in ("real", "sham"):
                        predictions = changed.get(name)
                        entry[name + "_predictions"] = (
                            " ".join(map(str, predictions[i])) if predictions is not None else ""
                        )
                        entry[name + "_flip"] = (
                            float(np.mean(predictions[i] != base[i]))
                            if predictions is not None
                            else None
                        )
                    rows.append(entry)
    return rows


def summarize(rows, config):
    participants = []
    for participant in config["targets"]:
        for arm in ("uniform", "native", "shuffled", "ssp"):
            block = [r for r in rows if r["participant"] == participant and r["arm"] == arm]
            metrics = {
                "accuracy": float(
                    balanced_accuracy_score(
                        [r["label"] for r in block], [r["prediction"] for r in block]
                    )
                )
            }
            if arm != "ssp":
                for key in ("real_flip", "sham_flip"):
                    metrics[key] = float(np.mean([r[key] for r in block]))
                metrics["excess_flip"] = metrics["real_flip"] - metrics["sham_flip"]
            participants.append(dict(participant=participant, arm=arm, **metrics))
    means = {}
    for arm in ("uniform", "native", "shuffled", "ssp"):
        block = [r for r in participants if r["arm"] == arm]
        means[arm] = {
            k: float(np.mean([r[k] for r in block]))
            for k in block[0]
            if k not in ("arm", "participant")
        }
    diff = {k: v - means["uniform"][k] for k, v in means["native"].items()}
    advance = (
        diff["accuracy"] >= -config["accuracy_margin"]
        and diff["real_flip"] < 0
        and diff["excess_flip"] < 0
    )
    return dict(participants=participants, means=means, native_minus_uniform=diff, advance=advance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    ids = config["sources"] + config["targets"]
    if len(ids) != len(set(ids)):
        raise ValueError("Source and target participants must be distinct")
    out = args.output
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with (
        threadpool_limits(limits=1),
        record_run(
            out,
            vars(args) | {"config": str(args.config), "data": str(args.data), "output": str(out)},
        ) as emit,
    ):
        shutil.copyfile(args.config, out / "config.toml")
        metadata = manifest(args.config, [development_path(args.data, p) for p in ids])
        metadata["packages"]["torch"] = torch.__version__
        write_json(out / "manifest.json", metadata)
        sources = {
            p: source_data(load_development_session(args.data, p), config)
            for p in config["sources"]
        }
        models = {}
        for arm in ("uniform", "native", "shuffled"):
            models[arm] = train(sources, config, arm, emit)
            torch.save(models[arm].state_dict(), out / f"{arm}.pt")
        rows = []
        for p in config["targets"]:
            rows.extend(evaluate(load_development_session(args.data, p), models, config, emit))
        write_csv(out / "predictions.csv", rows)
        write_json(out / "summary.json", summarize(rows, config))
        write_json(
            out / "hashes.json",
            {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.name != "events.jsonl"},
        )


if __name__ == "__main__":
    main()
