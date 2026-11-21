"""Run the development audit from a versioned TOML configuration."""

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

from threadpoolctl import threadpool_limits

from .config import PilotConfig
from .data import development_path, load_development_session
from .pilot import audit_session
from .results import file_sha256, manifest, participant_metrics, summarize, write_csv, write_json
from .runlog import record_run


def run_pilot(config_path: Path, data_root: Path, output: Path) -> None:
    invocation = {
        "config": str(config_path.resolve()),
        "data_root": str(data_root.resolve()),
        "output": str(output.resolve()),
    }
    with record_run(output, invocation) as emit:
        shutil.copyfile(config_path, output / "config.toml")
        config = PilotConfig.from_toml(output / "config.toml")
        write_json(output / "resolved_config.json", asdict(config))
        files = [development_path(data_root, participant) for participant in config.participants]
        metadata = manifest(output / "config.toml", files)
        write_json(output / "manifest.json", metadata)
        rows = []
        calibration = []
        with threadpool_limits(limits=1):
            for participant in config.participants:
                emit("participant_started", participant=participant)
                session = load_development_session(data_root, participant)
                diagnostics, trials = audit_session(session, config, emit)
                calibration.append(diagnostics)
                rows.extend(trials)
                emit("participant_completed", predictions=len(trials), **diagnostics)
                print(f"A{participant:02d}T: {len(session.labels)} trials evaluated", flush=True)
        metrics = participant_metrics(rows)
        write_csv(output / "trial_predictions.csv", rows)
        write_csv(output / "participant_metrics.csv", metrics)
        write_csv(output / "calibration_metrics.csv", calibration)
        write_json(output / "summary.json", summarize(metrics, calibration))
        products = sorted(p for p in output.iterdir() if p.name != "events.jsonl")
        write_json(output / "checksums.json", {p.name: file_sha256(p) for p in products})
    print(f"Development audit saved to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Pilot TOML config")
    parser.add_argument("--data-root", required=True, type=Path, help="Original 2a recordings")
    parser.add_argument("--output", required=True, type=Path, help="New directory for run outputs")
    args = parser.parse_args()
    run_pilot(args.config, args.data_root, args.output)


if __name__ == "__main__":
    main()
