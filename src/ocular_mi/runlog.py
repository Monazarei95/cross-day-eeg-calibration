"""Persistent, per-run event history and executable source snapshots."""

import json
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


@contextmanager
def record_run(output: Path, invocation: dict):
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with (output / "events.jsonl").open("x", encoding="utf-8", buffering=1) as stream:

        def emit(event, **details):
            entry = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - start, 4),
                "event": event,
                **details,
            }
            print(json.dumps(entry, allow_nan=False), file=stream, flush=True)

        emit("started", python=sys.executable, cwd=str(Path.cwd()), **invocation)
        try:
            package = Path(__file__).resolve().parent
            with ZipFile(output / "source.zip", "x", compression=ZIP_DEFLATED) as archive:
                for source in sorted(package.glob("*.py")):
                    archive.write(source, f"ocular_mi/{source.name}")
            yield emit
        except BaseException as error:
            emit("failed", error=type(error).__name__, traceback=traceback.format_exc())
            raise
        emit("completed")
