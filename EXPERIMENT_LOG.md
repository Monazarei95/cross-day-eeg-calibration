# Experiment and decision log

## 2026-11-19 · First BCI IV 2a development audit

**Question.** Is there enough evidence of ocular sensitivity in a small-label EEG
decoder to justify training the proposed contrastive model?

**Design before seeing the results.** Commit `f14c17e` contains the original
implementation and protocol. We used all nine T recordings, held out each complete
run, trained each probe with ten trials per class, and paired five calibration
draws across conditions. This within-day pilot keeps the next-day evaluation
available for a future locked study. The reference settings are in
[the pilot configuration](configs/bci2a_pilot.toml).

**First run, superseded after input validation.** Output directory `runs/bci2a_cpu_pilot`. All nine recordings
completed. Mean late-window motor-band balanced accuracy was 44.45% for the
EEG covariance decoder and 48.45% after ocular-subspace projection. The paired
difference was +4.00 percentage points (descriptive 95% interval +1.44 to
+6.55). EOG regression changed mean accuracy by +0.04 points. Measured exchanges
produced fewer prediction flips than their spatially scrambled shams in this
condition (difference −2.42 points). These are classical within-day results.

**Open issue.** Five of nine eye-calibration regressions had negative pooled
held-out R². A02T and A03T were particularly poor. Investigate calibration-block
boundaries, acquisition discontinuities, signal amplitudes and fold-specific
prediction errors before interpreting the ocular exchanges. This is a reason
to defer the full SSL study. The spatial projection result alone does not
establish that the removed subspace is specifically ocular.

**Instrumentation decision.** Preserve the original run. Subsequent runs record
timestamps, trial assignments, donor choices, sham permutations, fitted thresholds,
calibration-fold errors, results hashes and the executed package source. Use a
new run directory for each attempt. Failed runs retain their error and progress.

**Correction before repeating the analysis.** Direct inspection of the calibration
arrays found minimum-int16 separator values scaled to −1600 µV on all EEG channels
and −16000 µV on all EOG channels. Eight calibration blocks begin with a
100-sample separator. These are missing samples in the Graz format. The initial
fit treated them as signal, so its ocular results are superseded. The corrected
pipeline records the excluded offsets before fitting and preserves the measured
values elsewhere. Calibration-fold offsets in events refer to the retained array.
The rerun uses the same seed, windows, splits and models.

## 2026-11-19 · Corrected run and independent replay

Commit `a00535c` produced [the corrected summary](results/bci2a_cpu_v2/summary.json)
and [its replay](results/bci2a_cpu_v2_repeat/summary.json). Both Python processes
completed all nine T sessions, 54 held-out runs and five paired draws per fold.
The comparator verified all artifact checksums in each run. Input hashes,
configuration and all four scientific output files matched byte for byte.
The SHA-256 of the corrected summary is
`e2c94630964308ca3efac626ae09926af51bd8709bff701d26bb86ddb2af5783`.

Mean held-out calibration R² was 0.587. Late-window motor-band balanced accuracy
was 44.45% for EEG alone, 48.15% for SSP and 44.34% for EOG regression.
The mean paired SSP gain was +3.70 points (descriptive 95% interval +1.35 to
+6.05). The measured exchange produced 4.30 percentage points fewer prediction
flips than the energy-matched sham. This audit supports SSP as a competitive
baseline and leaves the proposed excess-flip mechanism unresolved for the
classical decoder. An SSL benefit and next-day performance remain unmeasured.

## First paired SSL development screen · design before training

Parent run `bci2a_cpu_v2`. Compare native EOG-matched negatives against
uniform negatives and within-participant shuffled EOG descriptors. Fix the
encoder, initialization, augmentations, seed, labeled draws and test runs across
arms. Pretrain on T sessions 01–06 and probe T sessions 07–09 with ten labels
per class from runs 1–4. Score runs 5–6. Use the late motor-band window and
include the SSP covariance decoder on the same splits. Average draws within
participant, then weight participants equally. Advance to a larger development
replication only if native matching lowers both measured flips and measured-minus-sham
flips versus uniform while retaining accuracy within two percentage points.
The shuffled control tests whether any gain tracks the correct EOG assignments.
One seed and three targets support an engineering decision only.

## First paired SSL screen · results and decision

Revision `a9f956d` produced [the development results](results/ssl_screen_v1/summary.json).
Native matching shortened the actual eye-descriptor distance in every epoch.
Its balanced accuracy was 40.00%, compared with 38.82% for uniform negatives,
39.93% for shuffled descriptors and 46.46% for classical SSP on the same splits.
Native-minus-uniform accuracy was +1.25, 0.00 and +2.29 points for participants
07, 08 and 09. Real flips rose 0.66 points and sham flips rose 1.05 points.
The resulting 0.39-point decrease in excess flips does not establish reduced
ocular sensitivity. The predeclared decision was to stop scaling this configuration.
This screen used random fixed-length crops without testing within-crop descriptor stability,
one training seed, three target participants, and within-day target probes.
Its scope is narrower than the proposed cross-day study.

A fresh Python process reproduced the predictions and summary byte for byte in
[`ssl_screen_v1_repeat`](results/ssl_screen_v1_repeat/summary.json). Both runs
also recorded identical SHA-256 hashes for all three model checkpoints.

The next hypothesis follows the nine-person classical result. Test the
calibrated ocular projection against unprojected EEG and repeated equal-rank
random projections before interpreting the gain as specific to the ocular
subspace. Share projection draws across classical and frozen uniform-SSL
comparisons. Keep the next-day sessions reserved until the method is fixed.

## Recording a future change

**Rank-matched control, design before execution.** Reuse the original nine-person
classical pilot's six held-out runs, five paired draws and late motor band. For
each participant compare raw EEG, the calibrated projection, and 16 isotropic
random subspaces of identical removed rank, generated before scoring and
shared across decoders. Use the average of all random projections as the
control. Score the frozen uniform encoder only on the three participants
excluded from its pretraining. Report participant-weighted means and count
participants beating both controls. These exploratory within-day comparisons
use only T recordings.

**Observed at revision `581b30a`.** The [paired summary](results/projection_control_v1/summary.json)
shows 48.15% balanced accuracy with eye calibration, 44.69% averaged over the
16 random projections and 44.45% on raw EEG for the classical decoder. Every
participant exceeded both personal control values. All 25,920 classical raw
and calibrated trial predictions matched the parent pilot exactly. The saved
matrices passed symmetry, idempotence and equal-rank checks. For the frozen SSL
encoder the corresponding accuracies were 44.24%, 40.01% and 40.72%. The
calibrated projection beat both controls for participants 07 and 08, while
participant 09 lost 1.39 points versus raw input. Different denominators
(nine classical, three SSL) matter for interpretation. The next validation
question is cross-day transfer with first-day calibration. An equal-rank random
control establishes a comparison with arbitrary directions, while structured
high-variance EEG directions remain another possible explanation.

Before examining new results, append the question, motivation, parent run,
intended single change, and expected comparison here. Commit the configuration
and code with Git. After execution, record the run path, commit, observed
participant-level effects and any decision. Keep the original run available.
The run manifest identifies the revision to retrieve with `git show` or
`git worktree add --detach <directory> <commit>`.

## Cross-day projection validation · locked design before scoring

The revised question concerns next-day four-class balanced accuracy. The earlier
ocular-matched-negative robustness hypothesis did not pass development screening.
Use all nine participants, one uniform-negative SSL pretraining seed per outer
fold and the eight other people's T sessions. Freeze the encoder. For each person,
fit projections on the separate day-one eye calibration and train the probe
on ten day-one trials per class, repeated with five fixed paired draws. Predict all
288 chronological day-two trials using EEG alone. Use the established late
8–30 Hz window, covariance classifier, encoder and 16 equal-rank random controls.
Include a structured specificity control removing the same number of largest EEG
principal components from the same day-one eye calibration, fitted without EOG.

Primary contrast is calibrated minus raw balanced accuracy for frozen SSL.
Report all participant values and a two-sided paired 95% t interval across the
nine participant differences. A positive lower bound supports a mean gain under
that analysis. An observed gain of at least three points meeting this criterion
also meets our practical target. Average draws and then random controls within
participant. The random and PCA contrasts and the classical decoder are secondary.
They describe specificity and practical alternatives. Save all predictions and
their hashes before the scoring program reads official labels. This accuracy
test does not measure the original perturbation endpoint. A later change using
these scores constitutes development and requires a new independent test.

**Predictions frozen.** Code and scoring rules were committed at `e155601`.
All 12 automated tests passed. The CPU run finished in about four minutes.
All nine prediction files passed checksum, shape, class-range and ordering
checks. The 2,592 distinct day-two trials have 492,480 prediction entries across
the two decoders, five draws and 19 projection conditions. Fold rosters excluded
each target from pretraining. The calibrated and random projection matrices
matched the preceding development run exactly, and the PCA control removed the
same number of directions. The prediction inventory SHA-256 is
`bff8a935b4a3633da9ec370f55d6eb423011bc4158ff31e524d55251a191b7af`.
No official E label file was opened in the prediction stage.

## Official day-two scoring

Scored the frozen inventory against the nine official E label files under
revision `0afc55b`. Mean SSL balanced accuracy increased from 37.04% to 41.29%.
The primary paired difference was +4.24 points (95% CI +1.22 to +7.27), with
eight of nine people improving. This met the locked criterion. Secondary
SSL contrasts were +4.04 points against random and +1.21 points against
calibration EEG PCA. The latter interval spanned zero (−0.24 to +2.66).
Classical balanced accuracy was 43.80% raw, 46.74% ocular, 44.14% random
and 47.26% PCA. Independent scoring from prediction files with scikit-learn
and SciPy matched all 72 participant/decoder/condition means and the six
group contrasts and intervals. Summary SHA-256:
`5d4e3f0bb8ba45b966f17d0e88ad26e2c7b5aecc7dfa5c79bc3285cb1ff027e6`.
See the [paper draft](cross_day_paper.md) and [scored results](results/cross_day_v1_scores/summary.json).

## Participant influence and independent follow-up feasibility

Using the frozen SSL ocular-minus-raw differences in `participants.csv`, omit
one scored participant at a time and average the remaining eight. The smallest
gain was +3.40 points (A01 omitted), and the largest was +4.87 (A04 omitted).
This was a post-scoring influence analysis. The models were held fixed.

OpenBMI provides two recording days and a binary hand-imagery task in 54
participants. The authors recorded separate instructed blink, horizontal and
vertical eye-movement epochs. We inspected day one for subject 01. Codes 71–73
provided 30,000 samples across 62 scalp channels at 1 kHz. The file listed four
auxiliary EMG channels and no dedicated EOG channels. A three-component PCA fit
to those epochs produced a rank-59 retained projection. File SHA-256
`678057b652462774497f72d3068d0cd5c96d37ad5caf05326f7b0ed08d56a7ed`.
[Study](https://academic.oup.com/gigascience/article/8/5/giz002/5304369)
and [data release](https://gigadb.org/dataset/100542).

Recommended independent extension: compare calibration EEG PCA against raw EEG
and the mean of 16 equal-rank random projections in the frozen SSL decoder.
Use first-day eye-task EEG to fit PCA, ten first-day imagery labels per class
for the probe, and day-two EEG for scoring. Fix the timing, training cohorts,
participant eligibility, and primary paired contrast before evaluating the new
cohort. Report binary accuracy separately from the four-class 2a experiment.
Only one day-one artifact file was examined. External performance remains
unknown. The local drive had about 1.45 GB free at inspection; a full run
would need a separate storage volume or staged preprocessing.

## OpenBMI independent extension · design fixed before scoring

Include all 54 participants. Use the offline 100-trial left/right imagery block
(`EEG_MI_train`) on each day. Hold out six consecutive groups of nine people,
pretraining a single uniform-negative encoder on all 45 other people's day-one
offline trials per fold. Use the spatial/temporal encoder and eight-epoch
contrastive recipe from 2a, adapted to the 62 scalp channels. Use the 8–30 Hz
band, 1.75–3.75 s after cue onset, and 125 Hz for the encoder. Match five fixed
day-one probe draws of ten trials per class and all day-two trials across arms.
Fit the three leading spatial directions by PCA on the 30 s of broadband day-one
eye-task EEG (blink, horizontal and vertical). Project those directions out of
both days' target EEG. Compare with raw EEG and 16 fixed isotropic rank-matched
projections. No eye electrodes are required. The primary endpoint is the mean
within-participant day-two balanced accuracy difference for PCA versus raw from
the frozen encoder, with a paired two-sided 95% t interval across participants.
Preserve the 2a criterion of a positive lower bound and an observed gain of
at least three percentage points for practical success. The random contrast is
secondary. Freeze all prediction files before evaluating labels. Report separately
from 2a because class counts, cueing and montages differ. Log missing or malformed
data as failures; do not silently exclude participants. Use the F: drive as the
input/cache volume. Download only first-day eye calibrations and both days' MI.

**20 November transfer recovery.** The first 35 participant caches were complete
when the mirror began returning a TLS certificate name mismatch. After on-disk
retries exhausted, both the download and driver stopped before prediction or
scoring. A fresh verified HTTPS request to the same mirror succeeded for subject
36. The driver resumed from the original receipts and partial files. The
scientific protocol remains at `409b519`. Additional transfer retries and a
post-score independent calculation have been added for reproducibility.

**21 November transfer recovery.** At the morning check, 131 of the 162 source
files had verification receipts and 41 of 54 participants had staging caches.
The previous driver and downloader had exited. Their log ended during transfer
without recording the reason. We restarted the same driver on the existing F:
storage and confirmed that it launched download workers. Its log is
`runs/openbmi_nov21.log`. No external predictions or scores existed at restart.

The second resume subsequently staged participants 48 and 49 before repeated
connection resets and connection failures exhausted the download retries. With
152 of 162 files verified, we resumed the remaining ten using six workers. The
driver now waits and retries after transient transport errors while retaining
the existing failure handling for integrity errors. The new transfer log is
`runs/openbmi_network_resume.log`.

**OpenBMI prediction complete.** All 162 source files received verification
receipts and all 54 participant caches were prepared. At revision `8868736`,
the six-fold prediction run completed in 281.079 seconds. The inventory is
`runs/openbmi_v1/prediction_hashes.json`.

**Scoring interrupted.** The scorer processed participant 01, then failed while
hashing participant 02's original day-two recording with a PermissionError.
Repeated read attempts on participants 02 and 03 failed. The failed run and
traceback remain in `runs/openbmi_v1_scores/events.jsonl`. Windows System events
reported bad blocks on physical disk 0 and filesystem corruption on E:.
A volume-extent query established that F: also resides on physical disk 0,
while C: resides on physical disk 1. Recovery will use original published
files and verify their hashes against the frozen prediction lineage.

**Evaluation recovery.** A separate recovery runner extracts compact official
label arrays only after matching each original recording to its saved source
SHA-256. Unreadable recordings are fetched again from the original release,
checked against those same digests, and processed on the separate C: volume.
The scorer and independent verification pass accept this verified label cache.
The original failed scoring record is preserved. Frozen model outputs and
the prespecified scoring comparisons are reused. Recovery tests validate
source identity and reject altered label arrays.


## Verified OpenBMI evaluation

Independent OpenBMI evaluation completed for all 54 participants. The frozen encoder reached 54.20% on raw EEG and 55.08% after scalp-only calibration. The paired gain was +0.89 points (95% CI -0.33 to +2.10), with 28 people improving. Official labels were recovered from files matching the frozen source hashes. An independent pass verified participant accuracies and both paired intervals.

## 2026-11-24 · Exploratory hand-only extension and signal check

After the four-class and OpenBMI results were known, the analysis question changed to left versus right hand on BCI IV 2a, with an additional concern about losing useful EEG information. This is explicitly exploratory. Reuse the nine leave-one-person-out encoder checkpoints and target first-day projections. Fit new binary probes with either 10 or all 72 first-day labels per hand class. Test on only the 144 second-day hand trials for each person. Five paired draws are used at the 10-label budget; the 72-label budget has one full-data fit. The conditions are raw, eye projection, and scalp-only PCA, each for frozen SSL and classical decoders. Save predictions and hashes before the scoring command reads labels. Record provenance of the reused checkpoints, projections and data.

The binary prediction inventory is `runs/hands_v1/prediction_hashes.json`, and scores are in `runs/hands_v1_scores`. With 10 labels per class, the frozen encoder reached 57.13% raw, 61.39% eye-projected and 60.93% scalp PCA. Eye minus raw was +4.26 points with descriptive 95% interval +1.43 to +7.09; eye minus PCA was +0.46 with interval −2.38 to +3.31. Classical scalp PCA reached 64.88%. With all 72 labels per hand class, the best mean was classical scalp PCA at 69.37%; frozen SSL scalp PCA reached 67.28%. An independent scikit-learn scoring pass matched all 108 participant-condition scores. No condition reached the desired 80% nine-person average.

The hard eye projection removed three of 22 spatial directions. Its mean retained energy on the second-day 8–30 Hz task windows was 11.26%; scalp PCA retained 6.80%. The raw-to-projected C3/C4 hand-contrast sign was unchanged for seven of nine participants in each of mu and beta. These observations do not separate neural signal from artifact and cannot establish neural preservation. The present paper must state that uncertainty. [Detailed analysis](hands_results.md)
