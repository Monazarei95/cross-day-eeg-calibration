"""Validated experiment settings loaded from TOML."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PilotConfig:
    seed: int
    participants: list[int]
    labels_per_class: int
    calibration_draws: int
    covariance_shrinkage: float
    regression_ridge: float
    calibration_folds: int
    calibration_gap_seconds: float
    filter_context_seconds: float
    donors_per_class: int
    ocular_load_quantile: float
    perturbation_quantile: float
    windows: dict[str, list[float]]
    bands: dict[str, list[float]]

    @classmethod
    def from_toml(cls, path: Path) -> "PilotConfig":
        with path.open("rb") as file:
            settings = cls(**tomllib.load(file))
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.participants or len(self.participants) != len(set(self.participants)):
            raise ValueError("Participants must be a nonempty list of unique IDs")
        if any(p not in range(1, 10) for p in self.participants):
            raise ValueError("Participant IDs must be in 1–9")
        if not 1 <= self.labels_per_class <= 60 or self.calibration_draws < 1:
            raise ValueError("Need 1–60 labels per class and at least one calibration draw")
        if self.seed < 0 or not 0 < self.covariance_shrinkage < 1:
            raise ValueError("Seed must be nonnegative and covariance shrinkage must be in (0, 1)")
        if self.regression_ridge < 0 or self.calibration_folds < 2:
            raise ValueError("Ridge must be nonnegative and calibration needs at least two folds")
        if self.calibration_gap_seconds < 0 or not 0 < self.filter_context_seconds <= 0.5:
            raise ValueError("Invalid calibration gap or filter context duration")
        if not 1 <= self.donors_per_class <= 11:
            raise ValueError("Need 1–11 same-run donors per class")
        if not all(0 < q < 1 for q in (self.ocular_load_quantile, self.perturbation_quantile)):
            raise ValueError("Quantiles must be in (0, 1)")
        if set(self.windows) != {"cue", "late"} or set(self.bands) != {"broad", "motor"}:
            raise ValueError("Expected cue/late windows and broad/motor bands")
        pad = self.filter_context_seconds
        cue_start, cue_stop = self.windows["cue"]
        late_start, late_stop = self.windows["late"]
        if not -2 <= cue_start - pad <= 0 < cue_stop <= 4 - pad:
            raise ValueError("Cue window and context must fit the trial and include cue onset")
        if not 1.25 < late_start - pad < late_stop <= 4 - pad:
            raise ValueError("Late window and context must follow cue offset and end by 4 s")
        if abs((cue_stop - cue_start) - (late_stop - late_start)) > 1e-9:
            raise ValueError("Cue and late analysis windows must have equal duration")
        for low, high in self.bands.values():
            if not 0 < low < high < 125:
                raise ValueError("Band cutoffs must strictly increase and lie below Nyquist")
