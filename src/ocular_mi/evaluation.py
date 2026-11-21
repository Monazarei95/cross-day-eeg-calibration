"""Verify compact evaluation labels recovered from the original source files."""

import json

import numpy as np

from .results import file_sha256


def cached_labels(cache, participant, source_sha256):
    path = cache / f"labels_{participant:02d}.npz"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["source_sha256"] != source_sha256 or receipt["participant"] != participant:
        raise ValueError("Recovered labels belong to another source")
    if file_sha256(path) != receipt["labels_sha256"]:
        raise ValueError("Recovered labels changed after extraction")
    with np.load(path, allow_pickle=False) as data:
        y = data["y"]
    if y.shape != (100,) or not np.array_equal(
        np.unique(y, return_counts=True), [[1, 2], [50, 50]]
    ):
        raise ValueError("Expected balanced offline evaluation labels")
    return y
