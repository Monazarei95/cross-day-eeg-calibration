# Exploratory next-day left-versus-right-hand analysis

This analysis uses all nine participants in BCI Competition IV 2a. Only left- and right-hand trials enter supervised classifier training and scoring. The target person's first day supplies either 10 or 72 labeled trials per hand class. The second day supplies 72 test trials per class, or 144 per person. A separate binary classifier is fitted for each condition. The existing leave-one-participant-out self-supervised encoder checkpoints and first-day calibration projections are reused without retraining. Those encoders were originally pretrained on unlabeled first-day trials from all four imagery classes in the other eight participants.

The four-class evaluation labels had already been opened for the earlier study. This analysis was proposed after seeing those results, so its intervals are descriptive and it is not an independent confirmatory test. In this run, binary predictions were saved and hashed before the official hand labels were read by the scoring command. [Prediction protocol](results/hands_v1/protocol.json) · [Frozen hashes](results/hands_v1/prediction_hashes.json)

## Next-day balanced accuracy

| First-day labels per hand class | Decoder | Uncorrected EEG | Eye projection | Scalp PCA projection |
|---|---|---:|---:|---:|
| 10, five matched draws | Frozen self-supervised encoder | 57.13% | 61.39% | 60.93% |
| 10, five matched draws | Classical covariance decoder | 63.01% | 64.58% | 64.88% |
| 72, all first-day hand trials | Frozen self-supervised encoder | 64.81% | 66.05% | 67.28% |
| 72, all first-day hand trials | Classical covariance decoder | 66.36% | 67.82% | **69.37%** |

Each value averages matched draws within a participant, then averages the nine participants equally. Chance balanced accuracy is 50%. The 10-label self-supervised eye-minus-raw difference is **+4.26 percentage points** with a descriptive paired 95% t interval of **+1.43 to +7.09**; seven of nine participants improve. The difference from scalp PCA is **+0.46 points**, interval **−2.38 to +3.31**. With 72 labels per class, the self-supervised eye-minus-raw difference shrinks to **+1.23 points**, interval **−2.54 to +5.01**. The classical decoder exceeds the self-supervised decoder in every mean comparison. The 80% mean-accuracy target is unmet in either label setting.

Individual people can reach 80%: with 72 labels, the best classical condition exceeds 80% for three of nine participants. This does not support reporting 80% as the study average or selecting only those participants. [Participant scores](results/hands_v1_scores/participant_scores.csv) · [Summary](results/hands_v1_scores/summary.json)

## Does the projection preserve EEG information?

The eye projection removes a three-dimensional spatial subspace from 22 EEG channels. A nonzero projection necessarily discards some measured signal. On the second-day 8–30 Hz imagery windows, the projection retained **11.26% of measured squared signal energy on average** across participants; scalp PCA retained **6.80%**. These values include both neural activity and artifacts. They cannot identify what fraction of *neural* information was lost. They do show that this hard projection changes the measured EEG substantially.

As one task-specific check, the sign of the left-versus-right C3/C4 log-power contrast was unchanged by eye projection for **7 of 9 participants** in the mu band and **7 of 9** in the beta band. Two participants changed sign in each band. Some raw contrasts are close to zero, so this is a limited diagnostic rather than proof of neural preservation. Improved decoding in the 10-label setting shows some useful information remains; it does not show that all useful brain information remains. [Signal diagnostics](results/hands_v1_scores/signal_diagnostics.csv)

A preservation claim would require a prospectively defined neural criterion, such as contralateral mu/beta event-related desynchronization and its topography, alongside held-out decoding. A protected or softly attenuated projection should be selected using first-day data alone and evaluated on untouched data.

## Interpretation

The hand-only task is easier than the four-class task, yet its best nine-person mean is 69.37%. The current self-supervised encoder remains below a classical baseline, and the eye-specific advantage over scalp-only PCA is unresolved. These findings do not justify claiming an 80% mean, superiority of self-supervised learning, or preservation of all neural information. [Original dataset description](https://www.bbci.de/competition/iv/desc_2a.pdf) · [Related 2026 C3/C4 preservation analysis](https://www.mdpi.com/2076-3425/16/10/1124)

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m ocular_mi.hands predict --data BCICIV_2a --previous results/cross_day_v1 --output runs/hands_v1
.\.venv\Scripts\python.exe -m ocular_mi.hands score --predictions runs/hands_v1 --labels BCICIV_2a/labels --output runs/hands_v1_scores
.\.venv\Scripts\python.exe scripts/verify_hands.py --predictions runs/hands_v1 --labels BCICIV_2a/labels --scores runs/hands_v1_scores
```

The public repository bundles the earlier encoder checkpoints and projection matrices under `results/cross_day_v1`, along with frozen binary predictions, participant scores, signal diagnostics, hashes, and checkpoint/projection SHA-256 references. The verification script recomputes all 108 participant-condition scores using scikit-learn.
