"use strict";

const byId = (id) => document.getElementById(id);
const percent = (value) => (100 * value).toFixed(2) + "%";
const points = (value) =>
  (value < 0 ? "−" : "+") + Math.abs(100 * value).toFixed(2);
const person = (id) => "A" + String(id).padStart(2, "0");
const activeLabel = (id) => byId(id).selectedOptions[0].text;
const definitions = {
  raw: "Uncorrected EEG uses the original scalp recording.",
  pca: "Scalp-only calibration removes the strongest patterns found in scalp EEG during the eye movements.",
  random:
    "Random correction removes the same number of directions, chosen at random. Scores average 16 such corrections.",
};

function selection() {
  return studyScores.participants.filter(
    (row) => row.method === byId("model").value,
  );
}

function element(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
}

function render() {
  const model = byId("model").value;
  const base = byId("baseline").value;
  const rows = selection();
  const group = studyScores.summary[model];
  const effect = group.paired_differences["ocular_minus_" + base];
  const prefix =
    model === "classical"
      ? "The classical model compares how signals vary together across scalp sensors. "
      : "The self-supervised model learned patterns from unlabeled recordings. ";
  byId("definition").textContent = prefix + definitions[base];
  byId("difference").textContent = points(effect.mean);
  byId("precision").textContent =
    "95% confidence interval across people: " +
    effect.ci95.map(points).join(" to ") +
    " points.";
  byId("improved").textContent =
    effect.improved + " of " + rows.length + " people improved.";
  byId("comparison").textContent =
    activeLabel("baseline") +
    " averaged " +
    percent(group.mean_accuracy[base]) +
    ". Eye calibration averaged " +
    percent(group.mean_accuracy.ocular) +
    ".";
  byId("base-heading").textContent = activeLabel("baseline");
  byId("caption").textContent =
    activeLabel("model") + " · next-day balanced accuracy";
  byId("bars").replaceChildren();
  byId("rows").replaceChildren();
  for (const row of rows) {
    const change = row.ocular - row[base];
    const bar = element("div", undefined, "bar-row");
    const track = element("div", undefined, "track");
    const fill = element("i", undefined, change < 0 ? "fill fall" : "fill");
    const width = (Math.abs(change) / 0.24) * 100;
    fill.style.width = width + "%";
    fill.style.left = (change < 0 ? 50 - width : 50) + "%";
    track.append(fill);
    bar.append(
      element("span", person(row.participant)),
      track,
      element("strong", points(change)),
    );
    byId("bars").append(bar);
    const tr = element("tr");
    const heading = element("th", person(row.participant));
    heading.scope = "row";
    tr.append(
      heading,
      element("td", percent(row[base])),
      element("td", percent(row.ocular)),
      element("td", points(change)),
    );
    byId("rows").append(tr);
  }
}

function downloadSelection() {
  const base = byId("baseline").value;
  const model = byId("model").value;
  const records = [
    [
      "participant",
      "model",
      "comparison",
      "comparison_percent",
      "eye_calibration_percent",
      "difference_points",
    ],
  ];
  for (const row of selection()) {
    records.push([
      person(row.participant),
      model,
      base,
      100 * row[base],
      100 * row.ocular,
      100 * (row.ocular - row[base]),
    ]);
  }
  const csv = records.map((record) => record.join(",")).join("\r\n") + "\r\n";
  const link = element("a");
  link.href = URL.createObjectURL(
    new Blob([csv], { type: "text/csv;charset=utf-8" }),
  );
  link.download = "bci2a-" + model + "-versus-" + base + ".csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

for (const id of ["baseline", "model"])
  byId(id).addEventListener("change", render);
byId("download").addEventListener("click", downloadSelection);
render();
byId("explorer").hidden = false;

// Project Pages and account Pages use different repository paths.
if (location.hostname.endsWith(".github.io")) {
  const owner = location.hostname.split(".")[0];
  const project = location.pathname.split("/").filter(Boolean)[0];
  const repo =
    !project || project === "index.html" ? location.hostname : project;
  for (const [id, file] of [
    ["paper", "cross_day_paper.md"],
    ["history", "EXPERIMENT_LOG.md"],
  ]) {
    const link = byId(id);
    link.href =
      "https://github.com/" + owner + "/" + repo + "/blob/main/" + file;
    link.hidden = false;
  }
}
