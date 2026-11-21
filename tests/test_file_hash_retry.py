from hashlib import sha256

import pytest

from ocular_mi import results


def test_hash_restarts_after_interrupted_read(tmp_path, monkeypatch):
    path = tmp_path / "source"
    path.write_bytes(b"complete file contents")
    original = type(path).open
    attempts = []

    def open_file(self, *args, **kwargs):
        attempts.append(self)
        if len(attempts) == 1:
            raise PermissionError("transient read failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(path), "open", open_file)
    monkeypatch.setattr(results.time, "sleep", lambda _: None)
    assert results.file_sha256(path) == sha256(b"complete file contents").hexdigest()
    assert len(attempts) == 2


def test_hash_reports_persistent_failure(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise PermissionError("persistent read failure")

    monkeypatch.setattr(type(tmp_path), "open", fail)
    monkeypatch.setattr(results.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        results.file_sha256(tmp_path / "unreadable")
