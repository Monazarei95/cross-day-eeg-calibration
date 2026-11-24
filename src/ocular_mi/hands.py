"""Exploratory next-day left/right analysis with frozen, previously trained encoders."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import t
from threadpoolctl import threadpool_limits

from .config import PilotConfig
from .contrastive import Encoder
from .cross_day import evaluation_trials
from .data import EEG_CHANNELS, load_development_session
from .features import bandpass, regularized_covariances
from .pilot import classifier
from .projection_control import motor_trials
from .results import file_sha256, write_csv, write_json
from .ssl_screen import encode

CONDITIONS = ("raw", "ocular", "pca")
METHODS = ("frozen_ssl", "classical")
BUDGETS = (10, 72)


def hand_draws(labels, participant, budget, count=5):
    """Sample only first-day left/right trials with a fixed participant seed."""
    rng = np.random.default_rng([81573, 2401, participant, budget])
    pools = [np.flatnonzero(labels == k) for k in (1, 2)]
    if any(len(pool) != 72 for pool in pools):
        raise ValueError("Expected 72 first-day trials per hand class")
    if budget == 72:
        return [np.sort(np.concatenate(pools))]
    return [np.sort(np.concatenate([rng.choice(pool, budget, replace=False) for pool in pools]))
            for _ in range(count)]


def sensor_asymmetry(epochs, fs=250):
    """Return hand-sensitive C3 minus C4 log power in mu and beta bands."""
    sensors = [EEG_CHANNELS.index("EEG-C3"), EEG_CHANNELS.index("EEG-C4")]
    pair = epochs[:, sensors]
    values = []
    for band in ((8, 13), (13, 30)):
        filtered = bandpass(pair, fs, band)
        power = np.mean(filtered.astype(np.float64) ** 2, axis=-1)
        values.append(np.log(np.maximum(power[:, 0], 1e-12)) -
                      np.log(np.maximum(power[:, 1], 1e-12)))
    return np.stack(values, axis=1)


def make_predictions(data, previous, output):
    pilot = PilotConfig.from_toml(previous / "pilot.toml")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    hashes = {}
    provenance = {}
    with threadpool_limits(limits=1):
        for p in pilot.participants:
            session = load_development_session(data, p)
            evaluation, cues = evaluation_trials(data / f"A{p:02d}E.gdf", pilot)
            train_trials = motor_trials(session.eeg_uv, session.cue_samples, 250, pilot)
            both = np.concatenate((train_trials, evaluation))
            checkpoint = previous / f"encoder_{p:02d}.pt"
            projection_path = previous / f"projections_{p:02d}.npz"
            with np.load(projection_path, allow_pickle=False) as archive:
                matrices = {name: archive[name] for name in CONDITIONS}
            for name, matrix in matrices.items():
                if matrix.shape != (22, 22) or not np.allclose(matrix, matrix.T):
                    raise ValueError(f"Invalid {name} projection for participant {p}")
            net = Encoder(22).eval()
            net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            predictions = {"cue_samples": cues}
            diagnostics = {}
            for name, matrix in matrices.items():
                transformed = both if name == "raw" else matrix.astype(np.float32) @ both
                features = {
                    "frozen_ssl": encode(net, transformed[..., ::2] / 10),
                    "classical": regularized_covariances(transformed, pilot.covariance_shrinkage),
                }
                diagnostics[f"asymmetry.{name}"] = sensor_asymmetry(
                    transformed[len(train_trials):]
                )
                if name != "raw":
                    original = evaluation.astype(np.float64)
                    removed = original - transformed[len(train_trials):]
                    diagnostics[f"energy_retained.{name}"] = np.array(
                        1 - np.sum(removed ** 2) / np.sum(original ** 2)
                    )
                for budget in BUDGETS:
                    draws = hand_draws(session.labels, p, budget)
                    for method in METHODS:
                        values = features[method]
                        predictions[f"b{budget}.{method}.{name}"] = np.stack([
                            classifier(method == "classical")
                            .fit(values[draw], session.labels[draw])
                            .predict(values[len(train_trials):])
                            for draw in draws
                        ]).astype(np.int8)
            prediction_path = output / f"predictions_{p:02d}.npz"
            np.savez(prediction_path, **predictions)
            np.savez(output / f"diagnostics_{p:02d}.npz", **diagnostics)
            hashes[prediction_path.name] = file_sha256(prediction_path)
            provenance[f"{p:02d}"] = {
                "checkpoint_sha256": file_sha256(checkpoint),
                "projection_sha256": file_sha256(projection_path),
                "training_session_sha256": file_sha256(data / f"A{p:02d}T.gdf"),
                "evaluation_session_sha256": file_sha256(data / f"A{p:02d}E.gdf"),
                "draw_indices": {str(b): [d.tolist() for d in hand_draws(session.labels, p, b)]
                                 for b in BUDGETS},
            }
            print(f"Predictions frozen for participant {p}/9", flush=True)
    write_json(output / "prediction_hashes.json", hashes)
    write_json(output / "provenance.json", provenance)
    write_json(output / "protocol.json", {
        "status": "exploratory_after_four_class_label_access",
        "classes": {"left_hand": 1, "right_hand": 2},
        "training_day": 1, "evaluation_day": 2,
        "labels_per_class": list(BUDGETS), "draws": {"10": 5, "72": 1},
        "encoder": "unchanged_leave_one_participant_out_day_one_checkpoint",
        "projections": list(CONDITIONS), "methods": list(METHODS),
        "test_predictions_saved_before_hand_labels_opened_in_this_run": True,
    })


def _interval(values):
    values = np.asarray(values, dtype=float)
    center = float(values.mean())
    margin = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": center, "ci95": [center - margin, center + margin],
            "improved": int(np.sum(values > 0))}


def score_predictions(prediction_dir, labels_dir, output):
    hashes = json.loads((prediction_dir / "prediction_hashes.json").read_text())
    if len(hashes) != 9 or any(file_sha256(prediction_dir / name) != digest
                               for name, digest in hashes.items()):
        raise ValueError("Frozen prediction inventory is incomplete or changed")
    output.mkdir(parents=True, exist_ok=False)
    rows, diagnostics = [], []
    for p in range(1, 10):
        labels = np.asarray(loadmat(labels_dir / f"A{p:02d}E.mat")["classlabel"]).ravel()
        if labels.shape != (288,) or not np.array_equal(np.bincount(labels)[1:], [72] * 4):
            raise ValueError(f"Invalid official labels for participant {p}")
        hand = np.isin(labels, (1, 2))
        with (np.load(prediction_dir / f"predictions_{p:02d}.npz", allow_pickle=False) as pred,
              np.load(prediction_dir / f"diagnostics_{p:02d}.npz", allow_pickle=False) as diag):
            if len(np.unique(pred["cue_samples"])) != 288:
                raise ValueError("Unordered or duplicate trial markers")
            for budget in BUDGETS:
                for method in METHODS:
                    for name in CONDITIONS:
                        values = pred[f"b{budget}.{method}.{name}"]
                        if values.shape != (5 if budget == 10 else 1, 288):
                            raise ValueError("Unexpected prediction shape")
                        if not np.isin(values, (1, 2)).all():
                            raise ValueError("Hand decoder predicted a non-hand class")
                        recalls = np.stack([(values[:, labels == k] == k).mean(axis=1)
                                            for k in (1, 2)], axis=1)
                        rows.append({"participant": p, "budget": budget, "method": method,
                                     "projection": name,
                                     "balanced_accuracy": float(recalls.mean()),
                                     "left_recall": float(recalls[:, 0].mean()),
                                     "right_recall": float(recalls[:, 1].mean())})
            asym_raw = diag["asymmetry.raw"][hand]
            for name in ("ocular", "pca"):
                asym = diag[f"asymmetry.{name}"][hand]
                for index, band in enumerate(("mu", "beta")):
                    raw_contrast = (asym_raw[labels[hand] == 1, index].mean() -
                                    asym_raw[labels[hand] == 2, index].mean())
                    contrast = (asym[labels[hand] == 1, index].mean() -
                                asym[labels[hand] == 2, index].mean())
                    diagnostics.append({"participant": p, "projection": name, "band": band,
                                        "energy_retained": float(diag[f"energy_retained.{name}"]),
                                        "raw_hand_contrast": float(raw_contrast),
                                        "projected_hand_contrast": float(contrast),
                                        "same_contrast_direction": bool(
                                            np.sign(raw_contrast) == np.sign(contrast)
                                        )})
    write_csv(output / "participant_scores.csv", rows)
    write_csv(output / "signal_diagnostics.csv", diagnostics)
    summary = {"status": "exploratory_after_prior_four_class_scoring", "participants": 9,
               "hand_test_trials_per_participant": 144, "results": {}}
    for budget in BUDGETS:
        summary["results"][str(budget)] = {}
        for method in METHODS:
            selected = [r for r in rows if r["budget"] == budget and r["method"] == method]
            by_condition = {name: np.array([r["balanced_accuracy"] for r in selected
                                            if r["projection"] == name]) for name in CONDITIONS}
            summary["results"][str(budget)][method] = {
                "mean_accuracy": {name: float(v.mean()) for name, v in by_condition.items()},
                "paired_differences": {f"ocular_minus_{name}": _interval(
                    by_condition["ocular"] - v
                )
                                       for name, v in by_condition.items() if name != "ocular"},
            }
    summary["signal_diagnostics"] = {name: {
        "energy_retained_mean": float(np.mean([r["energy_retained"] for r in diagnostics
                                                if r["projection"] == name])),
        "hand_contrast_direction_preserved": {band: int(sum(r["same_contrast_direction"]
            for r in diagnostics if r["projection"] == name and r["band"] == band))
            for band in ("mu", "beta")},
    } for name in ("ocular", "pca")}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--previous", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "predict":
        make_predictions(args.data, args.previous, args.output)
    else:
        score_predictions(args.predictions, args.labels, args.output)


if __name__ == "__main__":
    main()
