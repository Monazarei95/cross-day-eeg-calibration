"""Independently verify the locked OpenBMI scores and render the manuscript supplement."""

import argparse
import csv
import json
import shutil
import tomllib
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import ttest_rel
from sklearn.metrics import balanced_accuracy_score

from ocular_mi.evaluation import cached_labels
from ocular_mi.openbmi import input_path
from ocular_mi.results import file_sha256


def replace_result(path, result):
    start = "<!-- openbmi-result:start -->"
    end = "<!-- openbmi-result:end -->"
    content = path.read_text(encoding="utf-8")
    if content.count(start) != 1 or content.count(end) != 1:
        raise ValueError(f"Expected one OpenBMI result section in {path}")
    before, middle = content.split(start)
    _, after = middle.split(end)
    return before + start + "\n" + result + "\n" + end + after


def update_local_project(root, scores, report, summary):
    effect = summary["pca_minus_raw"]
    mean = 100 * effect["mean"]
    lo, hi = [100 * endpoint for endpoint in effect["ci95"]]
    statement = (
        f"Across {summary['participants']} people, scalp-only calibration changed next-day "
        f"balanced accuracy by {mean:+.2f} percentage points "
        f"(95% confidence interval {lo:+.2f} to {hi:+.2f}). "
        f"{effect['improved']} people improved."
    )
    paper = root / "cross_day_paper.md"
    dashboard = root / "docs" / "index.html"
    paragraph = statement + " [Independent OpenBMI evaluation](openbmi_results.md)"
    card = (
        f'<p class="status">{statement}</p>\n'
        '<a href="./openbmi_gain.png">View the participant results figure</a>'
    )
    updates = {
        paper: replace_result(paper, paragraph),
        dashboard: replace_result(dashboard, card),
        root / "openbmi_results.md": report.replace(
            "(openbmi_gain.png)", "(docs/openbmi_gain.png)"
        ),
    }
    shutil.copyfile(scores / "openbmi_gain.png", root / "docs" / "openbmi_gain.png")
    for path, content in updates.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--labels-cache", type=Path)
    args = parser.parse_args()
    config = tomllib.loads((args.predictions / "config.toml").read_text())
    hashes = json.loads((args.predictions / "prediction_hashes.json").read_text())
    summary = json.loads((args.scores / "summary.json").read_text())
    with (args.scores / "participants.csv").open(newline="") as stream:
        stored = {int(r["participant"]): r for r in csv.DictReader(stream)}
    ids = list(range(1, config["participants"] + 1))
    lineage = json.loads((args.predictions / "input_hashes.json").read_text())
    if set(ids) != set(stored):
        raise ValueError("Incomplete participant results")
    for p in ids:
        var = config["mat_variable"]
        raw = input_path(args.raw, 2, p, "MI")
        if args.labels_cache:
            y = cached_labels(args.labels_cache, p, lineage[str(p)][raw.name])
        else:
            y = np.ravel(loadmat(raw, simplify_cells=True, variable_names=[var])[var]["y_dec"])
        path = args.predictions / f"predictions_{p:02d}.npz"
        if file_sha256(path) != hashes[path.name]:
            raise ValueError(f"Prediction bundle changed for participant {p}")
        with np.load(path) as data:
            for condition in ("raw", "pca", "random"):
                keys = [k for k in data.files if k == condition or k.startswith(condition + "_")]
                expected = config["random_projections"] if condition == "random" else 1
                if len(keys) != expected:
                    raise ValueError("Invalid projection count")
                values = [balanced_accuracy_score(y, pred) for k in keys for pred in data[k]]
                np.testing.assert_allclose(
                    np.mean(values), float(stored[p][condition]), atol=1e-12
                )
        print(f"Independently verified participant {p:02d}", flush=True)
    v = {k: np.array([float(stored[p][k]) for p in ids]) for k in ("pca", "raw", "random")}
    if summary["participants"] != len(ids):
        raise ValueError("Participant count differs from the protocol")
    for k in v:
        np.testing.assert_allclose(v[k].mean(), summary["means"][k], atol=1e-12)
    for k in ("raw", "random"):
        d = v["pca"] - v[k]
        effect = summary["pca_minus_" + k]
        np.testing.assert_allclose(d.mean(), effect["mean"], atol=1e-12)
        np.testing.assert_allclose(
            ttest_rel(v["pca"], v[k]).confidence_interval(), effect["ci95"], atol=1e-12
        )
        assert np.sum(d > 1e-12) == effect["improved"]
    primary = summary["pca_minus_raw"]
    success = primary["mean"] >= 0.03 and primary["ci95"][0] > 0
    estimate = 100 * primary["mean"]
    lo, hi = 100 * np.array(primary["ci95"])
    versus_random = summary["pca_minus_random"]
    rlo, rhi = 100 * np.array(versus_random["ci95"])
    lines = [
        "# Independent OpenBMI evaluation",
        "",
        f"{len(ids)} people performed left- and right-hand imagery on two days. "
        "Each target contributed ten first-day labels per class for each of five paired draws. "
        "We fit PCA during 30 seconds of first-day eye tasks and removed three EEG directions. "
        "The day-two decoder used EEG alone. Each encoder excluded its evaluation participants.",
        "",
        "| Input to the frozen encoder | Day-two balanced accuracy |",
        "|---|---:|",
        *[
            f"| {label} | {100 * v[key].mean():.2f}% |"
            for key, label in [
                ("raw", "Raw EEG"),
                ("pca", "Day-one calibration EEG PCA"),
                ("random", "Mean of 16 random projections"),
            ]
        ],
        "",
        f"The primary calibrated-minus-raw difference was {estimate:+.2f} percentage points "
        f"(paired 95% CI {lo:+.2f} to {hi:+.2f}). "
        f"{primary['improved']} of {len(ids)} people improved. "
        f"This {'met' if success else 'fell short of'} the criterion of a gain of at least "
        "three points with a positive lower confidence bound.",
        "",
        f"Against random projections the difference was {100 * versus_random['mean']:+.2f} points "
        f"(95% CI {rlo:+.2f} to {rhi:+.2f}). OpenBMI tests the EEG-only PCA procedure. "
        "The dedicated eye-electrode procedure remains evaluated in BCI IV 2a. "
        "Binary OpenBMI and four-class 2a scores are reported separately.",
        "",
        "![OpenBMI participant effects](openbmi_gain.png)",
        "",
        "An independent pass recomputed balanced accuracy from frozen predictions and labels "
        "and checked participant-level summaries and both paired intervals.",
    ]
    report = "\n".join(lines) + "\n"
    (args.scores / "openbmi_results.md").write_text(report, encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    if args.scores.resolve() == root / "runs" / "openbmi_v1_scores":
        update_local_project(root, args.scores, report, summary)
    print(f"Verified all {len(ids)} participants and saved openbmi_results.md", flush=True)


if __name__ == "__main__":
    main()
