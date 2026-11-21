"""Score completed next-day predictions against chronological official class labels."""

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import t

from .config import PilotConfig
from .results import file_sha256, manifest, write_csv, write_json
from .runlog import record_run


def class_recalls(y, predictions):
    if (
        y.shape != (288,)
        or not np.array_equal(np.unique(y, return_counts=True)[1], [72] * 4)
        or not np.array_equal(np.unique(y), [1, 2, 3, 4])
    ):
        raise ValueError("Expected 72 official labels from each of four classes")
    if (
        predictions.ndim != 2
        or predictions.shape[1] != len(y)
        or not np.isin(predictions, [1, 2, 3, 4]).all()
    ):
        raise ValueError("Invalid prediction shape or class IDs")
    return np.stack([(predictions[:, y == k] == k).mean(axis=1) for k in (1, 2, 3, 4)], axis=1)


def participant_scores(predictions, y, pid, draws, controls):
    scores = []
    summary = []
    random_names = [f"random_{i:02d}" for i in range(1, controls + 1)]
    if predictions["cue_samples"].shape != (288,) or not np.all(
        np.diff(predictions["cue_samples"]) > 0
    ):
        raise ValueError("Missing or unordered chronological trial IDs")
    for method in ("classical", "frozen_ssl"):
        means = {}
        for condition in ["raw", "ocular", "pca", *random_names]:
            pred = predictions[f"{method}.{condition}"]
            if len(pred) != draws:
                raise ValueError("Incorrect draw count")
            recalls = class_recalls(y, pred)
            means[condition] = float(recalls.mean())
            for i, recall in enumerate(recalls, 1):
                scores.append(
                    dict(
                        participant=pid,
                        method=method,
                        projection=condition,
                        draw=i,
                        balanced_accuracy=float(recall.mean()),
                        **{f"recall_{j}": float(v) for j, v in enumerate(recall, 1)},
                    )
                )
        row = {k: means[k] for k in ("raw", "ocular", "pca")}
        row["random"] = float(np.mean([means[k] for k in random_names]))
        summary.append(dict(participant=pid, method=method, **row))
    return scores, summary


def group_summary(participants):
    results = {}
    for method in ("frozen_ssl", "classical"):
        rows = [r for r in participants if r["method"] == method]
        values = {k: np.array([r[k] for r in rows]) for k in ("raw", "ocular", "random", "pca")}
        diffs = {}
        for condition in ("raw", "random", "pca"):
            delta = values["ocular"] - values[condition]
            margin = float(t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta)))
            mean = float(delta.mean())
            diffs["ocular_minus_" + condition] = dict(
                mean=mean, ci95=[mean - margin, mean + margin], improved=int((delta > 1e-12).sum())
            )
        results[method] = dict(
            mean_accuracy={k: float(v.mean()) for k, v in values.items()}, paired_differences=diffs
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("predictions", "labels", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    root = args.predictions
    pilot = PilotConfig.from_toml(root / "pilot.toml")
    config = tomllib.loads((root / "projection.toml").read_text())
    hashes = json.loads((root / "prediction_hashes.json").read_text())
    paths = {p: root / f"predictions_{p:02d}.npz" for p in pilot.participants}
    if {path.name: file_sha256(path) for path in paths.values()} != hashes:
        raise ValueError("Frozen prediction inventory or checksum mismatch")
    labels = {p: args.labels / f"A{p:02d}E.mat" for p in paths}
    with record_run(args.output, {k: str(v) for k, v in vars(args).items()}) as emit:
        metadata = manifest(root / "config.toml", list(paths.values()) + list(labels.values()))
        write_json(args.output / "manifest.json", metadata)
        rows, participants = [], []
        for pid, path in paths.items():
            y = np.asarray(loadmat(labels[pid])["classlabel"]).ravel()
            with np.load(path, allow_pickle=False) as predictions:
                scores, paired = participant_scores(
                    predictions, y, pid, pilot.calibration_draws, config["random_projections"]
                )
            rows.extend(scores)
            participants.extend(paired)
            emit("participant_scored", participant=pid)
        write_csv(args.output / "draw_scores.csv", rows)
        write_csv(args.output / "participants.csv", participants)
        write_json(args.output / "summary.json", group_summary(participants))


if __name__ == "__main__":
    main()
