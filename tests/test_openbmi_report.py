from pathlib import Path
from runpy import run_path

import pytest


def test_release_updates_only_result_sections_and_can_be_replayed(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "report_openbmi.py"
    update = run_path(str(script))["update_local_project"]
    docs = tmp_path / "docs"
    scores = tmp_path / "scores"
    docs.mkdir()
    scores.mkdir()
    paper = tmp_path / "cross_day_paper.md"
    dashboard = docs / "index.html"
    for path in (paper, dashboard):
        path.write_text("a<!-- openbmi-result:start -->pending<!-- openbmi-result:end -->b")
    (scores / "openbmi_gain.png").write_bytes(b"test figure")
    summary = {
        "participants": 54,
        "pca_minus_raw": {"mean": -0.01, "ci95": [-0.03, 0.01], "improved": 19},
    }
    report = "![Figure](openbmi_gain.png)"
    update(tmp_path, scores, report, summary)
    first = paper.read_bytes()
    update(tmp_path, scores, report, summary)
    assert paper.read_bytes() == first
    assert first.startswith(b"a<!--") and first.endswith(b"-->b")
    assert "-1.00 percentage points" in dashboard.read_text()
    assert (docs / "openbmi_gain.png").read_bytes() == b"test figure"
    assert "(docs/openbmi_gain.png)" in (tmp_path / "openbmi_results.md").read_text()
    dashboard.write_text("edited without markers")
    with pytest.raises(ValueError):
        update(tmp_path, scores, report, summary)
    assert dashboard.read_text() == "edited without markers"
