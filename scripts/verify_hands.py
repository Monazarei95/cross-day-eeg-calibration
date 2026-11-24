"""Recompute every left/right score directly from frozen trial predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score

from ocular_mi.results import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads((args.predictions / "prediction_hashes.json").read_text())
    with (args.scores / "participant_scores.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 9 * 2 * 2 * 3 or len(inventory) != 9:
        raise ValueError("Expected 108 participant-condition score rows and nine prediction files")
    for p in range(1, 10):
        path = args.predictions / f"predictions_{p:02d}.npz"
        if file_sha256(path) != inventory[path.name]:
            raise ValueError(f"Prediction hash mismatch for participant {p}")
        labels = np.asarray(loadmat(args.labels / f"A{p:02d}E.mat")["classlabel"]).ravel()
        hand = np.isin(labels, (1, 2))
        if np.sum(hand) != 144:
            raise ValueError("Expected 144 left/right evaluation trials")
        with np.load(path, allow_pickle=False) as data:
            for row in (r for r in rows if int(r["participant"]) == p):
                key = f"b{row['budget']}.{row['method']}.{row['projection']}"
                pred = data[key][:, hand]
                verified = float(np.mean([balanced_accuracy_score(labels[hand], draw)
                                          for draw in pred]))
                if not np.isclose(verified, float(row["balanced_accuracy"]), atol=1e-12):
                    raise ValueError(f"Independent accuracy disagrees for {p}, {key}")
    print("Verified all 108 scores against frozen predictions and official hand labels")


if __name__ == "__main__":
    main()
