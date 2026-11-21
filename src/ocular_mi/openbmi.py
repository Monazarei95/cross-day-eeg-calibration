"""Locked independent OpenBMI comparison. Stage, predict, then score once."""

import argparse
import json
import shutil
import tomllib
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import t
from threadpoolctl import threadpool_limits

from .contrastive import train
from .evaluation import cached_labels
from .features import bandpass
from .pilot import classifier
from .results import file_sha256, manifest, write_csv, write_json
from .runlog import record_run
from .ssl_screen import encode


def save_arrays(path, **values):
    tmp = path.with_suffix(".partial.npz")
    np.savez(tmp, **values)
    tmp.replace(path)


def labels(record):
    y = np.asarray(record["y_dec"]).ravel()
    if y.shape != (100,) or not np.array_equal(
        np.unique(y, return_counts=True), [[1, 2], [50, 50]]
    ):
        raise ValueError("Expected 50 right and 50 left offline trials")
    return y.astype(np.int8)


def pca_matrix(record, config):
    codes = np.asarray(record["y_dec"]).ravel()
    indices = [np.flatnonzero(codes == k).item() for k in config["eye_codes"]]
    samples = record["smt"][:, indices, :].reshape(-1, len(record["chan"]))
    if samples.shape != (30000, 62) or int(record["fs"]) != 1000 or not np.isfinite(samples).all():
        raise ValueError("Invalid eye calibration")
    channels = np.asarray(record["chan"], dtype=str)
    ordered = np.argsort(channels)
    _, vectors = np.linalg.eigh(np.cov(samples[:, ordered].T))
    basis = vectors[:, -config["rank_removed"] :]
    return np.eye(len(channels)) - basis @ basis.T, channels[ordered]


def imagery(record, config, channels):
    fs = int(record["fs"])
    names = np.asarray(record["chan"], dtype=str)
    order = np.argsort(names)
    if len(set(names)) != 62 or not np.array_equal(names[order], channels) or fs != 1000:
        raise ValueError("EEG channels or sampling rates differ across files")
    start, stop = config["window"]
    pad = round(config["padding"] * fs)
    # OpenBMI provides sample times in milliseconds relative to imagery onset for smt.
    times = np.asarray(record["ival"]).ravel().astype(float)
    ep = record["smt"]
    if ep.shape != (len(times), 100, 62) or not np.all(np.diff(times) == 1):
        raise ValueError("Unexpected offline epoch axes or timing")
    # Some releases use 1..4000 for the first four seconds.
    if times[0] == 1:
        times = times - 1
    i = np.flatnonzero(times == round(start * fs) - pad).item()
    j = i + round((stop - start) * fs) + 2 * pad
    if j > len(times) or fs % config["rate"]:
        raise ValueError("Epoch does not cover analysis window")
    # Limit peak working memory while filtering at the original sampling rate.
    chunks = []
    for k in range(0, 100, 8):
        batch = ep[i:j, k : k + 8, :].transpose(1, 2, 0)[:, order, :]
        filtered = bandpass(batch, fs, config["band"])
        chunks.append(filtered[..., pad : -pad : fs // config["rate"]] / 10)
    x = np.ascontiguousarray(np.concatenate(chunks))
    if not np.isfinite(x).all():
        raise ValueError("Missing imagery samples")
    return x


def input_path(raw, day, p, kind):
    return raw / f"sess{day:02d}_subj{p:02d}_EEG_{kind}.mat"


def stage(p, config, raw, cache, signature):
    out = cache / f"subject_{p:02d}.npz"
    if out.exists():
        with np.load(out) as previous:
            if str(previous["signature"]) != signature:
                raise ValueError("Cache revision mismatch")
        return
    artifact = input_path(raw, 1, p, "Artifact")
    first = input_path(raw, 1, p, "MI")
    second = input_path(raw, 2, p, "MI")
    receipts = {}
    for path in (artifact, first, second):
        receipt = json.loads(path.with_suffix(".mat.json").read_text())
        if path.stat().st_size != receipt["size"]:
            raise ValueError(f"Invalid staged input {path}")
        receipts[path.name] = receipt["sha256"]
    mat = loadmat(artifact, simplify_cells=True, variable_names=["EEG_Artifact"])
    projection, channels = pca_matrix(mat["EEG_Artifact"], config)
    arrays = {}
    for key, path in (("first", first), ("second", second)):
        var = config["mat_variable"]
        record = loadmat(path, simplify_cells=True, variable_names=[var])[var]
        arrays[key] = imagery(record, config, channels)
        if key == "first":
            arrays["labels"] = labels(record)
        del record
    save_arrays(
        out,
        **arrays,
        projection=projection,
        channels=channels,
        signature=signature,
        receipts=json.dumps(receipts, sort_keys=True),
    )
    print(f"Staged participant {p:02d}", flush=True)


def balanced_draws(y, p, config):
    rng = np.random.default_rng([config["draw_seed"], p])
    return np.stack(
        [
            np.sort(
                np.concatenate(
                    [
                        rng.choice(np.flatnonzero(y == label), config["per_class"], replace=False)
                        for label in config["classes"]
                    ]
                )
            )
            for _ in range(config["draws"])
        ]
    )


def predictions(net, data, p, config):
    matrices = {"raw": None, "pca": data["projection"]}
    n = len(data["channels"])
    rng = np.random.default_rng([config["projection_seed"], p])
    for i in range(config["random_projections"]):
        basis, _ = np.linalg.qr(rng.normal(size=(n, config["rank_removed"])))
        matrices[f"random_{i + 1:02d}"] = np.eye(n) - basis @ basis.T
    x = np.concatenate([data["first"], data["second"]])
    y = data["labels"]
    draws = balanced_draws(y, p, config)
    result = {"draws": draws}
    for name, m in matrices.items():
        h = encode(net, x if m is None else m.astype(np.float32) @ x)
        result[name] = np.stack(
            [classifier(False).fit(h[draw], y[draw]).predict(h[len(y) :]) for draw in draws]
        )
    return result


def predict_all(args, config, encoder_path):
    encoder = tomllib.loads(encoder_path.read_text())
    people = list(range(1, config["participants"] + 1))
    files = [args.cache / f"subject_{p:02d}.npz" for p in people]
    with record_run(args.output, {"stage": "OpenBMI prediction"}) as emit:
        shutil.copyfile(args.config, args.output / "config.toml")
        shutil.copyfile(encoder_path, args.output / "encoder.toml")
        metadata = manifest(args.config, files + [encoder_path])
        metadata["packages"]["torch"] = torch.__version__
        write_json(args.output / "manifest.json", metadata)
        sources = {}
        lineage = {}
        for p, path in zip(people, files, strict=True):
            with np.load(path) as data:
                x = data["first"]
                # Uniform sampling ignores the descriptor. No eye descriptor enters training.
                sources[p] = {"x": x, "descriptor": np.zeros((len(x), 1))}
                lineage[p] = json.loads(str(data["receipts"]))
        write_json(args.output / "input_hashes.json", lineage)
        hashes = {}
        size = config["group_size"]
        for start in range(0, len(people), size):
            targets = people[start : start + size]
            training = {p: sources[p] for p in people if p not in targets}
            emit("fold", targets=targets, sources=list(training))
            net = train(training, encoder, "uniform", partial(emit, fold=start // size + 1))
            torch.save(net.state_dict(), args.output / f"encoder_{start // size + 1}.pt")
            for p in targets:
                with np.load(files[p - 1]) as data:
                    values = predictions(net, data, p, config)
                path = args.output / f"predictions_{p:02d}.npz"
                save_arrays(path, **values)
                hashes[path.name] = file_sha256(path)
                emit("predicted", participant=p, sha256=hashes[path.name])
        write_json(args.output / "prediction_hashes.json", hashes)


def score_all(args, config):
    root = args.predictions
    if file_sha256(root / "config.toml") != file_sha256(args.config):
        raise ValueError("Score configuration differs from prediction configuration")
    hashes = json.loads((root / "prediction_hashes.json").read_text())
    people = list(range(1, config["participants"] + 1))
    paths = [root / f"predictions_{p:02d}.npz" for p in people]
    if {p.name: file_sha256(p) for p in paths} != hashes:
        raise ValueError("Frozen prediction inventory mismatch")
    lineage = json.loads((root / "input_hashes.json").read_text())
    rows = []
    per_draw = []
    with record_run(args.output, {"stage": "OpenBMI scoring", "parent": str(root)}) as emit:
        for p, path in zip(people, paths, strict=True):
            raw = input_path(args.raw, 2, p, "MI")
            if getattr(args, "labels_cache", None):
                y = cached_labels(args.labels_cache, p, lineage[str(p)][raw.name])
            else:
                if file_sha256(raw) != lineage[str(p)][raw.name]:
                    raise ValueError(f"Evaluation file changed for {p}")
                var = config["mat_variable"]
                y = labels(loadmat(raw, simplify_cells=True, variable_names=[var])[var])
            scores = {}
            with np.load(path) as archive:
                for k in archive.files:
                    if k == "draws":
                        continue
                    pred = archive[k]
                    if (
                        pred.shape != (config["draws"], len(y))
                        or not np.isin(pred, config["classes"]).all()
                    ):
                        raise ValueError("Invalid predictions")
                    acc = np.mean(
                        [
                            (pred[:, y == label] == label).mean(axis=1)
                            for label in config["classes"]
                        ],
                        axis=0,
                    )
                    scores[k] = float(acc.mean())
                    per_draw.extend(
                        dict(participant=p, condition=k, draw=i, accuracy=float(v))
                        for i, v in enumerate(acc, 1)
                    )
            random = [scores[f"random_{i + 1:02d}"] for i in range(config["random_projections"])]
            rows.append(
                dict(
                    participant=p,
                    pca=scores["pca"],
                    raw=scores["raw"],
                    random=float(np.mean(random)),
                )
            )
            emit("scored", participant=p)
        summary = {
            "participants": len(rows),
            "means": {k: float(np.mean([r[k] for r in rows])) for k in ("pca", "raw", "random")},
        }
        for k in ("raw", "random"):
            diff = np.asarray([r["pca"] - r[k] for r in rows])
            margin = float(t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff)))
            mean = float(diff.mean())
            summary["pca_minus_" + k] = dict(
                mean=mean, ci95=[mean - margin, mean + margin], improved=int((diff > 1e-12).sum())
            )
        write_csv(args.output / "participants.csv", rows)
        write_csv(args.output / "draw_scores.csv", per_draw)
        write_json(args.output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "predict", "score"])
    for name in ("config", "raw", "cache", "output", "predictions"):
        parser.add_argument("--" + name, type=Path, required=name == "config")
    parser.add_argument("--participant", type=int)
    parser.add_argument("--labels-cache", type=Path)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    encoder_path = args.config.parent / config["encoder_config"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        if args.action == "prepare":
            args.cache.mkdir(parents=True, exist_ok=True)
            signature = file_sha256(args.config) + file_sha256(Path(__file__))
            ids = [args.participant] if args.participant else range(1, config["participants"] + 1)
            for p in ids:
                stage(p, config, args.raw, args.cache, signature)
        elif args.action == "predict":
            predict_all(args, config, encoder_path)
        else:
            score_all(args, config)


if __name__ == "__main__":
    main()
