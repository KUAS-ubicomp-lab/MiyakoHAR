# Results

## Final standing

The competition scores top-1 accuracy on 405 test clips from four subjects that do
not appear in training. The public leaderboard scores 201 of those clips and was
visible during the competition; the private leaderboard scores the other 204, was
revealed after the leaderboard froze on 16 September 2026, and decides the ranking.
Each team selects two final submissions before the freeze; the better private score
of the two counts.

| Final submission | Private score | Public score | Role in this repository |
|---|---|---|---|
| **026** `026_hr320_person_pixels.csv` | **0.78921** (161 of 204), **39th of 326 teams** | 0.78606 (158 of 201) | the default model, `model-026.pth` |
| 022 `022_r152_fence.csv` | 0.76960 (157 of 204) | 0.76616 (154 of 201) | the reference training build, `model-022.pth` |

One clip is worth 0.4975 points on the public board and 0.4902 points on the
private board. The standard error of a score at this accuracy is about six clips,
so two files that differ by a few clips are not distinguished by either board.

The final private leaderboard, read after the competition closed, lists 326 teams.
The winning score was 0.98529 (201 of 204 clips) and the fifteenth place, the cut
for the finalist selection stage, was 0.87745 (179 of 204), eighteen clips above
the team's score. The team rose four places from its public standing to 39th, and
shares its private score with the team ranked 38th, which submitted 41 entries to
the team's 27. The team's other final, 022, would have placed 48th on its own.

## Every submission

Twenty-seven files were scored between 12 August and 14 September 2026, each spent to
answer a question rather than to search. Private scores were read from the submissions
page after the reveal; one private clip is 0.4902 points.

| # | Date | Public | Private | What it was and what it tested |
|---|---|---|---|---|
| 001 | 12 Aug | 0.10945 | 0.10294 | Constant class 36 for every clip, emitted by `inference.sh` before any model existed. Proved the submission path; corroborated the public split at 201 clips. |
| 002 | 15 Aug | 0.41791 | 0.42156 | First real model: ResNet-18 with a temporal shift module at 120 x 160, depth + infrared and thermal branches fused, flip test-time augmentation, one fold. |
| 003 | 16 Aug | 0.42786 | 0.44607 | 002 with its bookkeeping fixed (it had shipped one fold while quoting a 3-fold mean). Gave the first validation-to-leaderboard constant. |
| 004 | 17 Aug | 0.61691 | 0.65686 | The backbone pivot: S3D pretrained on Kinetics-400 at 168 x 224, both branches retrained. The largest single move of the project, +38 clips. |
| 005 | 17 Aug | 0.61194 | 0.65196 | The fold axis priced: five members across folds. One clip worse; the axis buys nothing measurable. |
| 006 | 17 Aug | 0.61194 | 0.64215 | Architectural diversity: four S3D members plus one MC3-18. Same score as 005. |
| 007 | 17 Aug | 0.61194 | 0.64215 | The seed-2 twin of 004, predicted in writing at 122 of 201 clips; scored 123. |
| 008 | 18 Aug | 0.62189 | 0.62745 | Every member trained on all 18 subjects instead of a fold's 12. New best by one clip. |
| 009 | 18 Aug | 0.61691 | 0.63235 | 008's weights with the fusion temperature at 16. Lost a clip: a validation gain of +1.147 points that did not transfer. |
| 010 | 18 Aug | 0.63681 | 0.66666 | Self-training: 142 test clips at confidence 0.7 or more join training, temperature 16. |
| 011 | 18 Aug | 0.64179 | 0.65196 | 010's weights with the temperature back at 1. New best. |
| 012 | 18 Aug | 0.64179 | 0.64705 | Self-training round two (168 clips). Identical score across a 51-clip churn: the second round nets nothing. |
| 013 | 19 Aug | 0.61691 | 0.69117 | Pseudo-label threshold 0.5 (204 clips): five clips below 011. Purity dominates volume. |
| 014 | | | | Reserved for the threshold-0 arm; never submitted. |
| 015 | 19 Aug | 0.59701 | 0.64215 | Threshold 0.85 (84 clips): nine clips below 011, the worst of the family. |
| 016 | 20 Aug | 0.68159 | 0.72058 | The first ir-CSN-R50 members (IG-65M to Kinetics-400), +8 clips over 011. |
| 017 | 21 Aug | 0.68656 | 0.69117 | Self-training removed on the new backbone. New best by one clip: removing it cost nothing. |
| 018 | 21 Aug | 0.71144 | 0.71078 | The operating point re-derived for the new family: learning rate 5 x 10^-4, T = 16, 224 x 224, batch 12; two depth + infrared seeds and one thermal member, all on 18 subjects. |
| 019 | 26 Aug | 0.71144 | 0.72549 | 018 with the thermal member retrained under the temporal crop augmentation. |
| 020 | 27 Aug | 0.71144 | 0.75000 | 018 with the thermal member retrained under SWAD and the temporal crop. |
| 021 | 28 Aug | 0.71641 | 0.73529 | 020 with the thermal member at T = 32. |
| 022 | 29 Aug | 0.76616 | 0.76960 | One depth + infrared ir-CSN-R50 member and one ir-CSN-152 thermal member at T = 32 with SWAD and the temporal crop. Public best for two weeks; a final. |
| probe A | 7 Sep | 0.63681 | 0.64705 | Calibration only, never selectable: 022's depth + infrared member alone. |
| 023 | 8 Sep | 0.74129 | 0.77450 | 022's two members retrained with a class-balanced set of 127 pseudo-labelled test clips. |
| 024 | 12 Sep | 0.75621 | 0.77450 | 022's weights unchanged; only the rule combining the branches differs (a log-linear pool with a logit adjustment by the training prior). |
| 025 | 12 Sep | 0.71641 | 0.77941 | The thermal member replaced by a VideoMAE V2 ViT-B/16, stored as an int8 container. |
| 026 | 13 Sep | 0.78606 | 0.78921 | 022 with the thermal member retrained on frames decoded at the native 320 x 240 and a 320 x 320 input. A final. |
| 027 | 14 Sep | 0.76119 | 0.78431 | 026 with the depth + infrared member also retrained at native resolution. |

Three observations from the private column. First, the ordering of the two finals
held: 026 scored above 022 on both halves, and the three backbone families kept
their order on both halves (the best file of each, 003, 013 and 026, scored 91, 141 and
161 private clips). Second, the two halves disagreed
strongly for some candidates: the ViT-B container 025 scored 144 of 201 clips
publicly and 159 of 204 privately, and the pseudo-labelled 023 and the fusion-rule
twin 024 both scored 158 privately against 149 and 152 publicly. On 204 clips those
differences are one to three standard errors. Third, the same pattern held earlier in
the project: the pseudo-label arms 013 and 016 scored 141 and 147 private clips against
124 and 137 public ones, so the public half under-read that family throughout. The two
finals were selected before any private score existed, by the rules in
[experiments-summary.md](experiments-summary.md).

## Candidates against the ranges written before their scores

From submission 023 onward, every candidate's expected public range was written
into the team's record before it was submitted: its fold result converted at 3.01
public clips per fold percentage point (a conversion measured on the earlier
submissions), plus or minus the seven-clip noise floor of a 201-clip score.

| Candidate | Range written first | Public clips scored | Held? |
|---|---|---|---|
| 023 | 154 to 168 | 149 | no |
| 024 | 149 to 163 | 152 | yes |
| 025 | 150 to 164 | 144 | no |
| 026 | 152 to 166 | 158 | yes |
| 027 | 154 to 168 | 153 | no, by one clip |

Two of the five ranges held (024 and 026) and three were missed (023, 025 and 027).
Submission 026 is the only candidate that held its range above 022, and it carried
the largest fold gain measured without any test input. None reached the 161 clips
that the selection rules required for a challenger to replace 022 as the first pick.

## The validation instrument

Every recipe decision was made on a frozen leave-6-subjects-out split with two
seeds ([training.md](training.md) gives the folds and the gate). It scores 3,036
clips and resolves about 1.0 to 1.5 points on a paired comparison, against the
public board's 201 clips and roughly 5 points of noise; the board was used as the
independent check, never as the objective.

**Held-out accuracy of the 022 recipe per training subject**, in percent, seed
mean of two members trained on the other 12 subjects of the fold:

| Fold | Subject | Clips | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|---|
| 0 | 4 | 143 | 72.73 | 72.73 | 72.73 |
| 0 | 5 | 160 | 78.12 | 81.25 | 79.69 |
| 0 | 9 | 187 | 83.42 | 83.96 | 83.69 |
| 0 | 21 | 133 | 62.41 | 62.41 | 62.41 |
| 0 | 22 | 194 | 70.10 | 70.62 | 70.36 |
| 0 | 24 | 159 | 74.21 | 75.47 | 74.84 |
| 1 | 2 | 168 | 75.60 | 77.38 | 76.49 |
| 1 | 6 | 203 | 64.04 | 63.55 | 63.79 |
| 1 | 8 | 169 | 66.86 | 64.50 | 65.68 |
| 1 | 17 | 166 | 78.92 | 80.12 | 79.52 |
| 1 | 18 | 179 | 70.95 | 65.92 | 68.44 |
| 1 | 19 | 188 | 80.85 | 82.98 | 81.91 |
| 2 | 1 | 153 | 67.97 | 70.59 | 69.28 |
| 2 | 3 | 163 | 69.94 | 65.03 | 67.48 |
| 2 | 7 | 185 | 74.59 | 72.43 | 73.51 |
| 2 | 16 | 186 | 74.19 | 76.88 | 75.54 |
| 2 | 20 | 159 | 74.84 | 72.33 | 73.58 |
| 2 | 23 | 141 | 68.09 | 65.25 | 66.67 |

Clip-weighted mean 72.7%; range 62.41% (subject 21) to 83.69% (subject 9);
between-subject standard deviation 6.2 points. The 026 recipe scores 74.44% on the
same cells, +1.715 points (t = +3.68, 6 of 6 cells). Because every released member
was then retrained on all 18 subjects, clips of those subjects are in-sample for
the released models and are not comparable to these held-out figures.

The per-clip predictions behind these figures, for both recipes and both seeds,
are in `evaluation/fold_predictions_022_026.csv`, with the accuracy per activity
and per subject and the confusion matrices derived from them; see
[held-out-predictions.md](held-out-predictions.md).

## What the leaderboard attributed, on the S3D family

From the decomposed submissions 008 to 012, with one change at a time:
self-training +4 clips; training on all 18 subjects instead of 12, +1 clip; a
fusion temperature of 16 instead of 1, one clip lost, measured twice on two weight
sets. On the ir-CSN family the self-training gain vanished (016 to 017: removing it
gained a clip).

## Gains that did and did not transfer across backbones

Seven levers measured on the first backbone (ResNet-18) were re-measured on the
second (S3D). Three grew or held, four inverted or vanished:

| Lever | On ResNet-18 | On S3D | |
|---|---|---|---|
| Branch fusion (two streams against the better single stream) | +2.84 points | +4.19 points [+3.18, +5.20], 6 of 6 | grew |
| Depth + infrared over thermal alone | +3.96 points | +0.597 points, not significant | collapsed |
| Input resolution | a main effect | an interaction: pays only with Kinetics pretraining | changed kind |
| Training every BatchNorm layer (no partial freezing) | +8.1 and +16.6 points on a frozen probe | -1.68 points, 0 of 4 | inverted |
| Flip test-time augmentation | +0.84 points | +0.15 points, 3 of 6, p = 1.000 | vanished |
| Person crop | +0.57 points, not significant | -2.11 points, 0 of 6 | inverted |
| Fold ensembling | about +2 points | one clip lost on the board | vanished |

The two levers the released ensemble rests on were re-measured a third time on
ir-CSN-R50, on three subject-disjoint folds and two branches:

| Lever | On S3D | On ir-CSN-R50 | |
|---|---|---|---|
| Branch fusion | +4.19 points, 6 of 6 | +3.084 points [+0.164, +6.005], 3 of 3, t = 4.54 on 2 df | held |
| Flip test-time augmentation | +0.15 points, a null | +0.809 points [+0.043, +1.574], 5 of 6, t = 2.72 on 5 df | came back |

The direction of failure was not predictable, which is the finding: results from a
previous backbone were not too high, they were uninformative. The working rule
that followed, and that shaped the rest of the project, was that no result
measured on the previous backbone enters a plan, a projection or a claim until it
has been re-measured on the backbone that ships.
