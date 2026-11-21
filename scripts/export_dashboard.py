"""Package the reviewed BCI IV 2a scores for the static results explorer."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/scores.js"))
    args = parser.parse_args()
    with (args.scores / "participants.csv").open(newline="", encoding="utf-8") as stream:
        participants = [
            {
                "participant": int(row["participant"]),
                "method": row["method"],
                **{key: float(row[key]) for key in ("raw", "ocular", "random", "pca")},
            }
            for row in csv.DictReader(stream)
        ]
    summary = json.loads((args.scores / "summary.json").read_text())
    for method in ("classical", "frozen_ssl"):
        ids = sorted(row["participant"] for row in participants if row["method"] == method)
        if ids != list(range(1, 10)):
            raise ValueError(f"Expected one row for each participant for {method}")
    payload = {"participants": participants, "summary": summary}
    args.output.write_text(
        "const studyScores = " + json.dumps(payload, indent=2, allow_nan=False) + ";\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
