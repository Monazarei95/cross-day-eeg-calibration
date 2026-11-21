"""Resumable download of the first-day calibration and both days of MI for OpenBMI."""

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542"
)


def fetch(root, day, person, kind):
    name = f"sess{day:02d}_subj{person:02d}_EEG_{kind}.mat"
    path = root / name
    receipt = root / (name + ".json")
    if (
        receipt.exists()
        and path.exists()
        and path.stat().st_size == json.loads(receipt.read_text())["size"]
    ):
        return
    url = f"{BASE}/session{day}/s{person}/{name}"
    part = path.with_suffix(".part")
    existing = Path("runs/openbmi_sess01_subj01_EEG_Artifact.mat")
    if day == person == 1 and kind == "Artifact" and existing.exists() and not part.exists():
        shutil.copyfile(existing, part)
    head = subprocess.check_output(
        [
            "curl.exe",
            "-fsSIL",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-max-time",
            "240",
            "--max-time",
            "60",
            url,
        ],
        text=True,
    )
    fields = {
        key.lower(): value
        for key, value in (line.split(": ", 1) for line in head.splitlines() if ": " in line)
    }
    size = int(fields["content-length"])
    if not part.exists() or part.stat().st_size != size:
        subprocess.run(
            [
                "curl.exe",
                "-fsSL",
                "--retry",
                "8",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                "--speed-limit",
                "512",
                "--speed-time",
                "90",
                "-C",
                "-",
                "-o",
                str(part),
                url,
            ],
            check=True,
        )
    if part.stat().st_size != size:
        raise ValueError(f"Truncated {name}")
    etag = fields["etag"].strip('"')
    with part.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
        published = fields.get("x-amz-meta-md5chksum")
        expected = base64.b64decode(published, validate=True).hex() if published else etag
        if len(expected) == 32:
            f.seek(0)
            if hashlib.file_digest(f, "md5").hexdigest() != expected:
                raise ValueError(f"Checksum mismatch {name}")
    part.replace(path)
    receipt.write_text(json.dumps(dict(url=url, size=size, sha256=sha, etag=etag), indent=2))
    print("verified", name, size, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--participants", type=int, default=54)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (args.root, d, p, k)
        for p in range(1, args.participants + 1)
        for d, k in [(1, "Artifact"), (1, "MI"), (2, "MI")]
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, *task) for task in tasks]
        errors = []
        network_errors_only = True
        for task, future in zip(tasks, futures, strict=True):
            try:
                future.result()
            except Exception as error:
                errors.append(task[1:])
                network_errors_only &= isinstance(error, subprocess.CalledProcessError) and (
                    error.returncode in {5, 6, 7, 18, 28, 52, 55, 56}
                )
                print("failed", task[1:], str(error), flush=True)
        if errors:
            if network_errors_only:
                print(
                    "Temporary network failure; partial files are available for resume", flush=True
                )
                raise SystemExit(75)
            raise RuntimeError(f"Files requiring retry: {errors}")


if __name__ == "__main__":
    main()
