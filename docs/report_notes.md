# Report notes

Points to address when writing the report. Numbers are test-set mean ± sample std over seeds 0/1/2 unless stated.

## Part A — status (verified 2026-11-22)

All required Part A outputs exist and were recomputed from `results/A/informative/*/predictions_*.csv`; every reported accuracy and macro-F1 matches.

| Requirement | Location |
|---|---|
| Majority, text-only, image-only, fusion (TextCNN and BiLSTM swap) | `results/A/informative/<model>/` |
| Macro-F1, accuracy, informative precision/recall, params, training time | `results/A/informative/summary.csv`, `runs.csv` |
| Confusion matrices (median-validation run) | `figures/A_informative_*_cm_test_median.png` |
| Six fusion help/fail cases | `results/A/informative/fusion_error_analysis.md` |
| Selected checkpoints | Filenames in `README.md`; verify availability in Team Drive `checkpoints/` |

| Model | Macro-F1 | Accuracy | Params | Published acc. |
|---|---|---|---|---|
| Majority | 0.402 | 67.1% | 0 | – |
| TextCNN | 0.790 ± 0.007 | 81.8% | 994k | 80.8% |
| BiLSTM | 0.792 ± 0.007 | 82.0% | 1.18M | – |
| Image-only | 0.807 ± 0.007 | 82.9% | 132k | 83.3% |
| Fusion TextCNN | 0.857 ± 0.008 | 87.6% | 1.20M | 84.4% |
| Fusion BiLSTM | 0.860 ± 0.006 | 87.5% | 1.38M | 84.4% |

Median-validation seeds: TextCNN 2; all other learned models 0.

## Points to explain

1. **Training time was measured on three machines.** Image-only: Apple M4 Pro CPU. Text-only: RTX 5060 Laptop GPU. Fusion: RTX 4060 Laptop GPU. State the hardware next to every time, or re-time all runs on one machine (this adds tracker rows).
2. **Fusion projects image features before concatenation.** 512-d ResNet18 features pass through the shared `ImageEncoder` (Linear 512→256, ReLU, Dropout) and are concatenated with the 300-d text vector (556-d), then Linear 128 → classes. The brief says "concatenate image and text features, then use a small MLP"; describe the projection as a design choice.
3. **Fusion exceeds the published result** (87.5% vs 84.4%) despite smaller encoders (ResNet18 vs VGG16, random-init embeddings vs pretrained word vectors). Discuss differences in preprocessing, training budget and model selection (validation macro-F1, patience 3) rather than claiming a better architecture. Compare literature values by accuracy only.
4. **Image-only beats text-only** (0.807 vs 0.790/0.792), consistent with Ofli et al. (83.3% image vs 80.8% text).
5. **Modalities are complementary.** Image-only seed 0 vs TextCNN seed 2 on 1,534 test tweets: both correct 1,058; only image 225; only text 194; both wrong 57.
6. **TextCNN vs BiLSTM inside fusion** differ by 0.003 macro-F1, within one standard deviation.
7. **Fusion is not a vote.** Fusion BiLSTM s0 is correct when both single models are wrong in 9 test cases, and wrong when both are correct in 27; it still wins overall on cases where the branches disagree.
8. **Image-only confusion matrix (seed 0):** informative 913/1,030 correct (P 0.872, R 0.886); not_informative 370/504 (P 0.760, R 0.734).
9. **Feature cache:** `features/resnet18_imagenet1k_v1.pt`, 18,082 images, frozen ResNet18 IMAGENET1K_V1 in eval mode, built once in 274 s (0.0038 kWh). Every image-based run reuses it.

## Part B

Class counts (train / dev / test): affected_individuals 71 / 9 / 9; infrastructure_and_utility_damage 612 / 80 / 81; rescue_volunteering_or_donation_effort 912 / 149 / 126; other_relevant_information 1,279 / 239 / 235; not_humanitarian 3,252 / 521 / 504. The largest class is 46× the smallest in training.

### Image-only, ordinary cross-entropy (complete)

Runs `B-hum-image-s{0,1,2}-final-01`, CPU (Apple M4 Pro), 132,613 params, ~0.4 s training. Median-validation seed 2 (checkpoint `B-hum-image-s2-final-01.pt`, figure `figures/B_humanitarian_image_only_cm_test_median.png`).

| Metric | Test |
|---|---|
| Macro-F1 | 0.581 ± 0.005 |
| Weighted-F1 | 0.761 ± 0.002 |
| Accuracy | 77.1% ± 0.1 (published VGG16: 76.8%) |

| Class | F1 | Correct per seed (s0, s1, s2) / support |
|---|---|---|
| affected_individuals | 0.000 ± 0.000 | 0, 0, 0 / 9 |
| infrastructure_and_utility_damage | 0.754 ± 0.021 | 65, 70, 63 / 81 |
| rescue_volunteering_or_donation_effort | 0.547 ± 0.024 | 59, 57, 55 / 126 |
| other_relevant_information | 0.783 ± 0.006 | 182, 175, 175 / 235 |
| not_humanitarian | 0.818 ± 0.002 | 431, 434, 442 / 504 |

Points to explain:

1. **The rare class is never predicted.** affected_individuals is 0/9 in every seed; seed 2 sends 7 of them to rescue_volunteering and 2 to not_humanitarian. The model never outputs this class, so the gap between weighted-F1 (0.76) and macro-F1 (0.58) is mostly this class at F1 = 0.
2. **Errors collapse toward not_humanitarian.** In seed 2, 60/126 rescue_volunteering and 56/235 other_relevant images are predicted not_humanitarian. Rescue/donation is often a graphic, poster or group photo with little visual disaster content.
3. **Infrastructure damage is the easiest humanitarian class from pixels** (F1 0.75 from 612 training examples), consistent with damage being visually distinctive.
4. **The rare class has signal, but the class prior wins.** Using the seed-2 checkpoint, true affected_individuals images get a mean P(affected) of 0.077 on test, against 0.005 for other images (about 15×). Yet it is never the top class: on training data the model predicts it for only 1 of 71 images. This motivates the imbalance intervention instead of a different architecture.
5. **Accuracy matches the published image model** (77.1% vs 76.8%) while macro-F1 shows the imbalance problem that accuracy hides.

### Image-only, class-weighted cross-entropy (complete)

Team decision: one imbalance intervention for every Part B model, class-weighted cross-entropy with balanced weights `n / (C * n_c)` from training labels (`src/train.py: class_weights`). Part B training weights: affected 17.26, infrastructure 2.00, rescue 1.34, other_relevant 0.96, not_humanitarian 0.38. Everything else is unchanged from the CE runs (architecture, optimiser, budget, seeds, selection rule).

Runs `B-hum-image-w-s{0,1,2}-final-01`; median-validation seed 2 (checkpoint `B-hum-image-w-s2-final-01.pt`, figure `figures/B_humanitarian_image_only_weighted_cm_test_median.png`).

| Metric | CE | Weighted |
|---|---|---|
| Macro-F1 | 0.581 ± 0.005 | **0.617 ± 0.002** |
| Weighted-F1 | **0.761 ± 0.002** | 0.731 ± 0.003 |
| Accuracy | **77.1%** | 71.9% |
| Validation macro-F1 | 0.577 ± 0.010 | 0.581 ± 0.005 |

| Class (test support) | CE F1 / correct per seed | Weighted F1 / correct per seed |
|---|---|---|
| affected_individuals (9) | 0.000 / 0, 0, 0 | 0.232 / 8, 7, 5 |
| infrastructure_and_utility_damage (81) | 0.754 / 65, 70, 63 | 0.744 / 71, 72, 65 |
| rescue_volunteering_or_donation_effort (126) | 0.547 / 59, 57, 55 | 0.574 / 70, 75, 82 |
| other_relevant_information (235) | 0.783 / 182, 175, 175 | 0.776 / 175, 182, 189 |
| not_humanitarian (504) | 0.818 / 431, 434, 442 | 0.757 / 361, 348, 350 |

Points to explain (common vs rare trade-off):

1. **Weighting trades accuracy for macro-F1.** Macro-F1 +0.036, accuracy −5.2 points, weighted-F1 −0.03. The gain comes from the rare and mid-sized classes; the cost is mostly not_humanitarian recall (0.86 → 0.70).
2. **The rare class is now found but over-predicted.** affected_individuals recall 0.74 (5–8 of 9 correct) but precision 0.14: in seed 2, 33 images are predicted affected, 5 of them correctly; 18 of the false positives are not_humanitarian. F1 0.23 is still low.
3. **Validation and test disagree on the rare class.** Only 2–4 of 9 affected images are correct on validation but 5–8 of 9 on test. With 9 examples per split, one image moves recall by 11 points. Report counts, not just F1, and treat the rare-class numbers as high variance.
4. **Validation macro-F1 barely moves** (0.577 → 0.581) while test macro-F1 improves by 0.036. Do not over-claim; the improvement is consistent across seeds (std 0.002) but partly driven by 9 test images.

### Fusion, ordinary versus class-weighted cross-entropy (complete 2026-11-23)

All 12 final runs use the official 6,126 / 998 / 955 agreed-label splits, the same shared ResNet18 cache, training-only vocabulary (5,266 tokens), Adam at 1e-3, batch size 64, at most 20 epochs and patience 3. Weighted loss is the only change within each text-encoder pair; its five weights are 17.2563, 2.0020, 1.3434, 0.9579 and 0.3768, calculated from training labels only. The image branch and fusion classifier remain fixed between ordinary and weighted runs. All runs used a local NVIDIA GeForce RTX 5070 Laptop GPU. Training time cannot be directly compared with the teammates' different machines.

| Fusion model | Test macro-F1 | Test weighted-F1 | Test accuracy | affected correct by seed (/9) | Median-validation seed; best epoch |
|---|---:|---:|---:|---|---|
| TextCNN, ordinary | 0.6678 ± 0.0511 | 0.8192 ± 0.0016 | 0.8234 ± 0.0022 | 3, 0, 0 | 0; epoch 6 |
| TextCNN, weighted | **0.7185 ± 0.0048** | 0.8164 ± 0.0092 | 0.8157 ± 0.0100 | 4, 3, 4 | 0; epoch 8 |
| BiLSTM, ordinary | 0.6587 ± 0.0331 | 0.7985 ± 0.0116 | 0.8010 ± 0.0121 | 2, 3, 0 | 1; epoch 6 |
| BiLSTM, weighted | 0.6881 ± 0.0304 | 0.8109 ± 0.0082 | 0.8101 ± 0.0100 | 2, 3, 3 | 2; epoch 5 |

Mean test per-class F1 over three seeds:

| Model | affected | infrastructure | rescue/donation | other relevant | not humanitarian |
|---|---:|---:|---:|---:|---:|
| Fusion TextCNN | 0.133 | 0.805 | 0.732 | 0.808 | 0.861 |
| Fusion TextCNN, weighted | 0.391 | 0.793 | 0.754 | 0.807 | 0.848 |
| Fusion BiLSTM | 0.204 | 0.741 | 0.723 | 0.781 | 0.846 |
| Fusion BiLSTM, weighted | 0.269 | 0.791 | 0.729 | 0.804 | 0.847 |

Weighting improves TextCNN fusion macro-F1 by 0.0508 and reduces its seed standard deviation from 0.0511 to 0.0048. Average accuracy drops by 0.77 percentage points, and the common `not_humanitarian` F1 drops by 0.013. Weighted BiLSTM fusion improves both macro-F1 (+0.0294) and accuracy (+0.91 points); do not describe every intervention as an accuracy cost. The nine affected test examples make rare-class conclusions sensitive to single cases. Fusion TextCNN's 82.3% ordinary accuracy and 81.6% weighted accuracy can be compared to the published 78.4% fusion accuracy, with different encoders and preprocessing stated; macro-F1 must remain separate from published accuracy.

Four concrete errors from the median-validation **weighted TextCNN fusion seed 0**. The ordinary-loss fusion comparison and the separately trained weighted text-only and image-only models below all use seed 0 on the *same official test pairs*. The image descriptions come from a teammate's manual inspection of the four source JPGs; those JPGs were unavailable in this workspace for independent visual verification. Predictions and probabilities were checked against the official TSV and saved CSVs. Each proposed cause is an interpretation of the evidence, not a measured attention weight or proof of which modality dominated.

1. **Harvey flood, `904568770982764545_0` (true: `affected_individuals`).** The tweet says “flood victims”; the teammate observed people wading through floodwater and travelling by boat, visual evidence compatible with people affected by the disaster. Weighted text-only predicts `rescue_volunteering_or_donation_effort`, weighted image-only correctly predicts `affected_individuals`, but weighted fusion predicts `rescue_volunteering_or_donation_effort` with probability 0.990; ordinary fusion makes the same error. The group and boat could resemble rescue activity, but the single-model contrast only establishes that a useful image-only signal did not produce the correct fused decision.
2. **Puerto Rico children, `921867940882014208_0` (true: `affected_individuals`).** The tweet describes children unable to attend school, and the teammate observed visibly damaged surroundings in the image. Weighted text-only predicts `rescue_volunteering_or_donation_effort`; weighted image-only and weighted fusion predict `not_humanitarian` (fusion probability 0.779). Ordinary fusion also predicts `not_humanitarian`. Both separately trained single-modality models miss the rare label here; this may reflect difficulty connecting indirect effects on people and background damage to `affected_individuals`.
3. **Family and dog after wildfire, `919600201538658310_0` (true: `not_humanitarian`).** The teammate described the dog photo as visually ordinary. Ordinary fusion correctly predicts `not_humanitarian`, as does weighted image-only, but weighted text-only predicts `affected_individuals` and weighted fusion switches to that wrong label (probability 0.604). Words such as “family” and “presumed dead” are a plausible rare-class cue; the prediction pattern is consistent with the weighted fusion reacting to those words despite the unremarkable image, without establishing a causal attribution.
4. **Puerto Rico food delivery, `921565262238887937_0` (true: `rescue_volunteering_or_donation_effort`).** The tweet mentions getting food to citizens; the teammate observed a vehicle carrying boxes, compatible with aid or supply delivery. Both weighted text-only and weighted image-only correctly predict `rescue_volunteering_or_donation_effort`, and ordinary fusion is also correct. Weighted fusion instead predicts `not_humanitarian` (probability 0.760). This is a clear fused-decision failure even though both separately trained modalities classify the pair correctly; it does not identify the internal mechanism.

The four median-validation test confusion matrices are in `figures/B_humanitarian_fusion_*_cm_test_median.png`, with per-class precision, recall, F1 and correct counts in each `metrics.json`. The source images remain outside Git under the dataset's `data_image/` paths.

Each selected checkpoint is the best epoch of its model's median-validation seed: `B-hum-fusion-textcnn-s0-final-01.pt`, `B-hum-fusion-textcnn-w-s0-final-01.pt`, `B-hum-fusion-bilstm-s1-final-01.pt` and `B-hum-fusion-bilstm-w-s2-final-01.pt`. Copies for Drive upload are in `checkpoints/drive_handoff/PartB_fusion/`; these binaries stay out of Git.

**Environmental reporting caveat.** These 12 runs have unique rows in `env_tracker.csv`; `EnvRun` requested CodeCarbon 3.3.2 in `tracking_mode="process"`, and the ledger uses 207.05 gCO₂e/kWh. The first ordinary TextCNN fusion run (`B-hum-fusion-textcnn-s0-final-01`) records 0.00204856 kWh in 9.6 s, equivalent to **768 W** averaged over the logged runtime. The next two ordinary TextCNN runs imply 114 W and 107 W, respectively. The first value is an outlier, not evidence that this model or seed inherently requires seven times the power.

**Why the exact cause is unknown.** [CodeCarbon's configuration](https://docs.codecarbon.io/latest/how-to/configuration/) gives a default 15 s measurement interval, longer than this 9.6 s run; its [accuracy guidance](https://docs.codecarbon.io/latest/explanation/accuracy/) warns that short or spiky runs can be sensitive to sampling and startup. In process mode, [GPU power remains device-wide](https://docs.codecarbon.io/3.3/explanation/methodology/), so unrelated GPU activity is another possible contributor. `EnvRun` saves only the total energy, not CPU/GPU/RAM components, backend, sampling trace or whether its own backup estimate was used. Its runtime clock starts after `tracker.start()` and stops before `tracker.stop()`, so the energy and logged-runtime windows may differ. With the *current default* backup settings (`GPU_POWER_W=0`), a 9.6 s estimate would be only about 0.000094 kWh, suggesting that the logged outlier came from CodeCarbon if those settings were unchanged; historical settings and component readings were not retained, so this cannot be proven. Neither startup, GPU sharing nor a tracker/counter error can be singled out as the cause.

Keep the original row. For a **sensitivity illustration only**, applying the mean implied power of the next two ordinary TextCNN runs to 9.6 s gives about 0.000295 kWh, versus the recorded 0.002049 kWh. The 12 fusion runs total 0.005892 kWh as logged, or 0.004138 kWh in this illustrative replacement scenario (about 30% lower). Do not substitute the scenario value into `env_tracker.csv` or present it as a measurement. State the uncertainty before using this run in group totals or efficiency ratios; the backup's default 0 W GPU assumption would also underestimate any GPU run that fell back to that method.
