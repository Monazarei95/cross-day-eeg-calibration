"""Download, stage, predict, score and plot the independent OpenBMI protocol."""

import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

from ocular_mi.openbmi import input_path, stage
from ocular_mi.results import file_sha256


def completed(path):
    log = path / "events.jsonl"
    return log.is_file() and json.loads(log.read_text().splitlines()[-1])["event"] == "completed"


def prepare(args, config):
    from threadpoolctl import threadpool_limits

    from ocular_mi import openbmi

    signature = file_sha256(args.config) + file_sha256(Path(openbmi.__file__))
    raw = args.storage / "raw"
    cache = args.storage / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    waiting = set(range(1, config["participants"] + 1))
    # An initial standalone fetch may already be writing range-resumable partials.
    active = None
    attempts = 0
    if args.existing_downloader:
        import psutil

        active = psutil.Process(args.existing_downloader)
        if not any("fetch_openbmi.py" in arg for arg in active.cmdline()):
            raise ValueError("Specified process is not the project's downloader")
    child = None
    with threadpool_limits(limits=1):
        while waiting:
            for p in sorted(waiting):
                ready = all(
                    input_path(raw, d, p, k).with_suffix(".mat.json").is_file()
                    for d, k in [(1, "Artifact"), (1, "MI"), (2, "MI")]
                )
                if ready:
                    stage(p, config, raw, cache, signature)
                    waiting.remove(p)
            if not waiting:
                break
            downloading = (active is not None and active.is_running()) or (
                child is not None and child.poll() is None
            )
            if not downloading:
                if child is not None and child.returncode == 75:
                    print(
                        f"Network interrupted. Retrying {sorted(waiting)} in 60 seconds",
                        flush=True,
                    )
                    time.sleep(60)
                    attempts = 0
                if attempts == 3:
                    raise RuntimeError(f"Download retries exhausted for {sorted(waiting)}")
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "scripts/fetch_openbmi.py",
                        str(raw),
                        "--workers",
                        str(args.workers),
                    ]
                )
                attempts += 1
            time.sleep(30)
    return raw, cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/openbmi.toml"))
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--existing-downloader", type=int)
    parser.add_argument("--labels-cache", type=Path)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    raw, cache = args.storage / "raw", args.storage / "cache"
    if not args.labels_cache:
        raw, cache = prepare(args, config)
    run = args.runs / "openbmi_v1"
    scores = args.runs / "openbmi_v1_scores"
    base = [sys.executable, "-m", "ocular_mi.openbmi"]
    label_args = ["--labels-cache", str(args.labels_cache)] if args.labels_cache else []
    if args.labels_cache and not completed(run):
        raise ValueError("Label recovery requires completed frozen predictions")
    if not completed(run):
        subprocess.run(
            [
                *base,
                "predict",
                "--config",
                str(args.config),
                "--cache",
                str(cache),
                "--output",
                str(run),
            ],
            check=True,
        )
    if not completed(scores):
        subprocess.run(
            [
                *base,
                "score",
                "--config",
                str(args.config),
                "--raw",
                str(raw),
                "--predictions",
                str(run),
                "--output",
                str(scores),
                *label_args,
            ],
            check=True,
        )
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (scores / "participants.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    delta = 100 * np.array([float(r["pca"]) - float(r["raw"]) for r in rows])
    results = json.loads((scores / "summary.json").read_text())
    effect = results["pca_minus_raw"]
    np.testing.assert_allclose(delta.mean(), 100 * effect["mean"], atol=1e-10)
    fig, ax = plt.subplots(figsize=(7.2, 4), layout="constrained")
    ax.axhline(0, color="0.6", linewidth=0.9)
    ax.scatter([int(r["participant"]) for r in rows], delta, color="#116487", s=20)
    m = delta.mean()
    lo, hi = np.array(effect["ci95"]) * 100
    ax.errorbar(len(rows) + 4, m, yerr=[[m - lo], [hi - m]], fmt="D", capsize=3, color="#182f3a")
    ax.set(
        xticks=[1, 10, 20, 30, 40, 50, len(rows) + 4],
        xticklabels=["1", "10", "20", "30", "40", "50", "Mean"],
        xlabel="OpenBMI participant",
        ylabel="Day-two gain (percentage points)",
        title="Day-one calibration EEG PCA versus raw EEG",
    )
    fig.savefig(scores / "openbmi_gain.png", dpi=220)
    plt.close(fig)
    subprocess.run(
        [
            sys.executable,
            "scripts/report_openbmi.py",
            "--raw",
            str(raw),
            "--predictions",
            str(run),
            "--scores",
            str(scores),
            *label_args,
        ],
        check=True,
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
