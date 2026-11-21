"""Verify two completed pilot runs and show configuration and output differences."""

import argparse
import hashlib
import json
from pathlib import Path

RESULT_FILES = (
    "trial_predictions.csv",
    "calibration_metrics.csv",
    "participant_metrics.csv",
    "summary.json",
)


def read_run(path: Path):
    with (path / "events.jsonl").open(encoding="utf-8") as stream:
        last = None
        for line in stream:
            last = json.loads(line)
    if last is None or last["event"] != "completed":
        raise ValueError(f"Incomplete run: {path}")
    hashes = json.loads((path / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        if Path(name).name != name:
            raise ValueError(f"Invalid artifact name in {path}: {name}")
        with (path / name).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                raise ValueError(f"Artifact checksum mismatch: {path / name}")
    config = json.loads((path / "resolved_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return hashes, config, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--require-identical", action="store_true")
    args = parser.parse_args()
    first, config_a, meta_a = read_run(args.first)
    second, config_b, meta_b = read_run(args.second)
    changes = {
        key: [config_a.get(key), config_b.get(key)]
        for key in sorted(config_a.keys() | config_b.keys())
        if config_a.get(key) != config_b.get(key)
    }
    matched = {name: first[name] == second[name] for name in RESULT_FILES}
    same_inputs = meta_a["inputs"] == meta_b["inputs"]
    print(
        json.dumps(
            {
                "commits": [meta_a["git_commit"], meta_b["git_commit"]],
                "config_changes": changes,
                "same_inputs": same_inputs,
                "identical_scientific_outputs": matched,
            },
            indent=2,
        )
    )
    if args.require_identical and (changes or not same_inputs or not all(matched.values())):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
