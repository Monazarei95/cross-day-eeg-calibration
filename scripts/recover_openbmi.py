"""Recover verified official labels and finish a frozen OpenBMI evaluation."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import tomllib
from io import BytesIO
from pathlib import Path

from scipy.io import loadmat

from ocular_mi.evaluation import cached_labels
from ocular_mi.openbmi import labels, save_arrays
from ocular_mi.results import file_sha256, write_json


def extract(source, cache, participant, expected, variable):
    contents = source.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError("Source differs from the frozen input inventory")
    record = loadmat(BytesIO(contents), simplify_cells=True, variable_names=[variable])[variable]
    y = labels(record)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"labels_{participant:02d}.npz"
    save_arrays(path, y=y)
    receipt = dict(
        participant=participant, source_sha256=expected, labels_sha256=file_sha256(path)
    )
    write_json(path.with_suffix(".json"), receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("runs/openbmi_v1"))
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--labels-cache", type=Path, default=Path("runs/openbmi_official_labels"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--participant", type=int)
    args = parser.parse_args()
    config = tomllib.loads((args.predictions / "config.toml").read_text())
    lineage = json.loads((args.predictions / "input_hashes.json").read_text())
    if args.source:
        name = f"sess02_subj{args.participant:02d}_EEG_MI.mat"
        extract(
            args.source,
            args.labels_cache,
            args.participant,
            lineage[str(args.participant)][name],
            config["mat_variable"],
        )
        return
    hashes = json.loads((args.predictions / "prediction_hashes.json").read_text())
    if {name: file_sha256(args.predictions / name) for name in hashes} != hashes:
        raise ValueError("Frozen predictions have changed")
    failed = []
    for p in range(1, config["participants"] + 1):
        name = f"sess02_subj{p:02d}_EEG_MI.mat"
        if (args.labels_cache / f"labels_{p:02d}.json").exists():
            cached_labels(args.labels_cache, p, lineage[str(p)][name])
            continue
        if p == 2:
            failed.append(p)
            continue
        command = [
            sys.executable,
            __file__,
            "--storage",
            str(args.storage),
            "--labels-cache",
            str(args.labels_cache),
            "--predictions",
            str(args.predictions),
            "--participant",
            str(p),
            "--source",
            str(args.storage / "raw" / name),
        ]
        child = subprocess.Popen(command)
        try:
            if child.wait(timeout=100) != 0:
                raise subprocess.CalledProcessError(child.returncode, command)
            print(f"Extracted verified labels for participant {p:02d}", flush=True)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True
            )
            child.wait()
            failed.append(p)
            print(f"Original read timed out for participant {p:02d}", flush=True)
        except subprocess.CalledProcessError:
            failed.append(p)
            print(f"Original file needs recovery for participant {p:02d}", flush=True)
    for p in failed:
        name = f"sess02_subj{p:02d}_EEG_MI.mat"
        part = args.labels_cache.parent / f"evaluation_{p:02d}.part"
        if p == 2:
            part = args.labels_cache.parent / "openbmi_evaluation_recovery.part"
            while part.stat().st_size < 598511386:
                time.sleep(30)
        else:
            base = (
                "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/"
                "10.5524/100001_101000/100542"
            )
            url = f"{base}/session2/s{p}/{name}"
            while True:
                if shutil.disk_usage(part.parent).free < 750_000_000 and not part.exists():
                    raise OSError("Insufficient space for an evaluation recovery file")
                result = subprocess.run(
                    [
                        "curl.exe",
                        "-fsSL",
                        "--connect-timeout",
                        "15",
                        "--max-time",
                        "120",
                        "-C",
                        "-",
                        "-o",
                        str(part),
                        url,
                    ]
                )
                print(f"Recovery participant {p:02d}, exit {result.returncode}", flush=True)
                if result.returncode == 0:
                    break
                if result.returncode not in {5, 6, 7, 18, 28, 52, 55, 56}:
                    raise RuntimeError("Recovery requires intervention")
                time.sleep(15)
        extract(part, args.labels_cache, p, lineage[str(p)][name], config["mat_variable"])
        print(f"Recovered and verified participant {p:02d}", flush=True)
        part.unlink()
    scores = args.predictions.parent / "openbmi_v1_scores"
    if scores.exists():
        events = json.loads((scores / "events.jsonl").read_text().splitlines()[-1])
        if events["event"] == "failed":
            archive = scores.with_name("openbmi_v1_scores_failed_01")
            if archive.exists():
                raise FileExistsError("Previous failed-run archive already exists")
            scores.rename(archive)
    subprocess.run(
        [
            sys.executable,
            "scripts/run_openbmi.py",
            "--storage",
            str(args.storage),
            "--labels-cache",
            str(args.labels_cache),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
