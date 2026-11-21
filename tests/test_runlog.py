import json
from zipfile import ZipFile

import pytest

from ocular_mi.cli import run_pilot
from ocular_mi.runlog import record_run


def test_run_history_survives_invalid_config_and_retry(tmp_path):
    config = tmp_path / "bad.toml"
    config.write_text("seed = ", encoding="utf-8")
    output = tmp_path / "experiment"
    with pytest.raises(ValueError):
        run_pilot(config, tmp_path, output)
    log = (output / "events.jsonl").read_bytes()
    events = [json.loads(line) for line in log.splitlines()]
    assert [entry["event"] for entry in events] == ["started", "failed"]
    assert events[-1]["error"] == "TOMLDecodeError"
    assert (output / "config.toml").read_bytes() == config.read_bytes()
    with pytest.raises(FileExistsError):
        run_pilot(config, tmp_path, output)
    assert (output / "events.jsonl").read_bytes() == log


def test_success_records_source_and_terminal_event(tmp_path):
    output = tmp_path / "experiment"
    with record_run(output, {}) as emit:
        emit("stage", detail="verification")
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert [entry["event"] for entry in events] == ["started", "stage", "completed"]
    with ZipFile(output / "source.zip") as archive:
        assert "ocular_mi/pilot.py" in archive.namelist()
