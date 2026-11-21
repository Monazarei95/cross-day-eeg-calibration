"""Participant-first summaries and reproducibility metadata."""

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.stats import t


def _accuracy(rows: list[dict]) -> float | None:
    recalls = [
        [row["label"] == row["prediction"] for row in rows if row["label"] == category]
        for category in (1, 2, 3, 4)
    ]
    return float(np.mean([np.mean(scores) for scores in recalls])) if all(recalls) else None


def participant_metrics(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (row["participant"], row["window"], row["band"], row["model"])
        groups[key].append(row)
    records = []
    for (participant, window, band, model), trials in sorted(groups.items()):
        high = [r for r in trials if r["high_eog"]]
        low = [r for r in trials if not r["high_eog"]]
        flagged = [r for r in trials if r["expert_flagged"]]
        records.append(
            {
                "participant": participant,
                "window": window,
                "band": band,
                "model": model,
                "balanced_accuracy": _accuracy(trials),
                "high_eog_accuracy": _accuracy(high),
                "low_eog_accuracy": _accuracy(low),
                "flagged_accuracy": _accuracy(flagged),
                "high_eog_trials": len({r["trial"] for r in high}),
                "flagged_trials": len({r["trial"] for r in flagged}),
                "ocular_flip_rate": _mean(r["ocular_flip_rate"] for r in trials),
                "sham_flip_rate": _mean(r["sham_flip_rate"] for r in trials),
            }
        )
    return records


def _mean(values) -> float | None:
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def _estimate(values) -> dict:
    valid = np.array([value for value in values if value is not None])
    n = valid.size
    mean = float(valid.mean()) if n else None
    ci = None
    if n > 1:
        half_width = float(t.ppf(0.975, n - 1) * valid.std(ddof=1) / np.sqrt(n))
        ci = [mean - half_width, mean + half_width]
    return {"participants": n, "mean": mean, "ci95": ci}


def summarize(per_participant: list[dict], calibration: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in per_participant:
        groups[row["window"], row["band"], row["model"]].append(row)
    accuracy = [
        {
            "window": window,
            "band": band,
            "model": model,
            **_estimate(r["balanced_accuracy"] for r in rows),
        }
        for (window, band, model), rows in sorted(groups.items())
    ]
    paired = []
    for (window, band, model), rows in sorted(groups.items()):
        if model != "eeg_only":
            continue
        regress = groups[window, band, "eog_regression"]
        ssp = groups[window, band, "ocular_ssp"]
        paired.append(
            {
                "window": window,
                "band": band,
                "ocular_minus_sham_flips": _estimate(
                    r["ocular_flip_rate"] - r["sham_flip_rate"] for r in rows
                ),
                "high_minus_low_eog_accuracy": _estimate(
                    r["high_eog_accuracy"] - r["low_eog_accuracy"]
                    for r in rows
                    if r["high_eog_accuracy"] is not None and r["low_eog_accuracy"] is not None
                ),
                "regression_minus_eeg_accuracy": _estimate(
                    other["balanced_accuracy"] - base["balanced_accuracy"]
                    for base, other in zip(rows, regress, strict=True)
                ),
                "ssp_minus_eeg_accuracy": _estimate(
                    other["balanced_accuracy"] - base["balanced_accuracy"]
                    for base, other in zip(rows, ssp, strict=True)
                ),
            }
        )
    return {
        "protocol": "BCI IV 2a T sessions, leave-one-run-out development",
        "calibration_r2": _estimate(r["heldout_r2"] for r in calibration),
        "calibration_r2_minus_shifted": _estimate(
            r["heldout_r2"] - r["shifted_eog_r2"] for r in calibration
        ),
        "balanced_accuracy": accuracy,
        "paired_differences": paired,
    }


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git(*args) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    for attempt in range(3):
        try:
            with path.open("rb") as file:
                return hashlib.file_digest(file, "sha256").hexdigest()
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(2)


def manifest(config_path: Path, files: list[Path]) -> dict:
    status = _git("status", "--porcelain", "--untracked-files=normal")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("mne", "numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "config_sha256": file_sha256(config_path),
        "compute": {"device": "cpu", "numerical_threads": 1},
        "inputs": [{"file": path.name, "sha256": file_sha256(path)} for path in files],
    }
