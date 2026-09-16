# Experiments summary

Five weeks of work, 12 August to 14 September 2026, produced the two released
models. This page tells that story in the order it happened, with the numbers that
decided each turn. [results.md](results.md) has the full tables; the 16-page
technical description in this folder has the finals in depth.

## How decisions were made

Every experiment was pre-registered before it ran: its hypothesis, its arms, the
comparison it would be read by, and the bar it had to clear were written into the
team's record first, and the result was written beside them afterwards. The
instrument was a frozen leave-6-subjects-out split of the 18 training subjects, 3
folds x 2 seeds ([training.md](training.md)): a change had to gain at least +0.5
points on all three folds at a first screen, then pass t > 2.571 on 5 degrees of
freedom with at least 5 of 6 cells in its direction. The number of such gated reads
that could change the shipped model was capped at fourteen in advance, and the
family-wise error rate implied by the count was printed with every read. The
public leaderboard was never the objective: it scores 201 clips, one clip is 0.4975
points, and its noise floor is about seven clips, so it served as an independent
check on whether validation gains transferred, and each submission was spent to
answer one question.

## The corpus first

The first week measured the data rather than fitting to it ([data.md](data.md)):
the (activity, subject) design is 537 of 720 cells, class frequency is 30.4 to 1 and
structured by subject, radar presence is a subject-group label, thermal cannot be
aligned frame by frame with the other streams and sees 59% of their field of view,
the person is a median 5.9% of the frame, and the test set has no frames in common
with training but duplicates itself. Those measurements fixed the shape of the
system before the first model was trained: two branches fused at the probability
level, no radar, no missing-modality flags, no resampling, and a person detector
built early and held in reserve.

## Three backbones

**ResNet-18 with a temporal shift module** (submissions 002 and 003, public
0.42786) established the pipeline and the first validation-to-leaderboard
constant. **S3D pretrained on Kinetics-400** (004, 0.61691) was the largest single
move of the project, +38 clips, and the first sign that pretraining was the lever
that mattered. On S3D the team priced the ensemble axes with paired submissions:
members across folds bought nothing (005), architectural diversity bought nothing
(006), a seed-2 twin (007) gave the family its second validation-to-board point, training every member on all 18
subjects instead of a fold's 12 gained a clip (008), and a fusion temperature of 16
lost one (009). Self-training on confident test clips gained four clips (010 to
012), the threshold arms showed that label purity dominates volume (013, 015), and
a second round netted nothing (012).

**ir-CSN-R50 pretrained on IG-65M and fine-tuned on Kinetics-400** (016, 0.68159)
moved the fold instrument by +5.075 points over S3D, the largest effect measured
anywhere in the project. On this family the self-training gain vanished: removing
it gained a clip (017, 0.68656). Re-deriving the operating point for the family,
learning rate 5 x 10^-4 instead of the inherited 5 x 10^-3 (which destabilised
training on 6 of 6 cells), batch 12 instead of 16 (+2.116 points), 224 x 224 input,
gave 018 (0.71144), the base of everything after it.

Seven levers measured on ResNet-18 were re-measured on S3D and four of them
inverted or vanished ([results.md](results.md)); the branch-fusion lever survived
all three backbones (+2.84, +4.19, +3.08 points). From then on no result from a
previous backbone entered a plan until it had been re-measured on the one that
shipped. Most of what the project got right came from enforcing that rule.

## The thermal member

From 018 on, the depth + infrared member was fixed and the work moved to the
thermal branch, where the smaller network was capacity-limited. Three changes were
measured on the fold instrument and then carried to the board one at a time: a
temporal crop augmentation (019), a dense weight average at a constant learning
rate over epochs 5 to 39, SWAD (020), and 32 sampled frames instead of 16 (021,
0.71641, +1 clip). The step that mattered was the deeper backbone: an
**ir-CSN-152** thermal member with SWAD, the temporal crop and T = 32, fused with
one depth + infrared ir-CSN-R50 member, gave submission **022** (0.76616, +10 clips
over 021) on 29 August. It held the team's public lead for two weeks and became the
first of the two finals.

## Five candidates after 022

Each later candidate was submitted with a public range written into the record
first, from its fold result at 3.01 clips per fold point, plus or minus seven clips.

- **023, class-balanced pseudo-labels** (127 test clips labelled by 022): fold gain
  +2.080 points on the subjects whose test clips it had seen but +0.506 on
  held-out subjects; public 149 against a range of 154 to 168. Missed. The gain was
  transductive: it lived on the subjects the pseudo-labels came from.
- **024, the fusion-rule twin**: 022's weights with a log-linear pool and a logit
  adjustment by the training prior in place of the arithmetic mean; fold +0.809
  points; public 152 against 149 to 163. Held, but two clips below 022.
- **025, a ViT-B thermal member** (VideoMAE V2, int8-stored beside the fp16
  depth + infrared member, 99.7 MB): fold gain +1.347 points, short of the
  significance bar (t = 2.41 against the 2.571 required); public 144 against 150
  to 164. Missed by six. Its int8 storage cost about 0.2 to 0.5 points per fold and
  its per-subject reading had failed a subject-level check before the score.
- **026, thermal at native resolution**: the person is small, so the thermal
  network was retrained on frames decoded at the sensor's 320 x 240 with a 320 x 320
  input at batch 6; fold +1.715 points (t = +3.68, 6 of 6), the first change to pass
  the promotion gate on an instrument with no exposure to test data; public 158
  against 152 to 166. Held, four clips above 022. The second final.
- **027, both members at native resolution**: fold +1.127 points over the 026 pair,
  but its per-subject shape failed the subject-level check; public 153 against 154
  to 168. Missed by one.

Of the five candidates, only 024 and 026 held their written ranges, and only 026
held it above 022. The finals were selected by rules fixed before any of these scores existed:
022 stays the first pick unless a challenger scores at least 161 of 201 (none did);
the second pick is the candidate with the highest expected private accuracy judged
first by the fold instrument and then by the public reading (026 on both counts,
and the only one whose range held). The two disagree on 42 of 405 clips, so
selecting both covered the case in which the thermal change did not carry.

After the reveal, the private board scored 026 at 161 of 204 (0.78921, 39th place)
and 022 at 157 (0.76960); the ordering held. The private half also scored 025 at 159,
and 023 and 024 at 158, well above their public readings ([results.md](results.md)),
which is a reminder of how noisy a 201-clip board is and why no candidate was
selected by its public score alone.

## What was measured and not used

- **A person detector.** A zero-dependency depth-threshold and connected-component
  box was built and beat a pretrained SSDLite320 detector head to head on 293 clips
  (median intersection over union 0.612), at zero bytes and zero licence exposure.
  At the shipped operating point cropping cost 2.11 points over six cells, so no
  crop ships. The observation behind it, that the person is small in the frame, led
  to 026 instead.
- **Radar, IMU and skeleton.** Radar is a subject-block indicator; IMU carries a
  measured train-to-test shift in absolute orientation and, in battery level, a
  session identifier; the skeleton and IMU loaders exist in `src/` under strict
  hygiene rules, and no released model reads them.
- **Fold ensembling and architectural diversity** bought nothing on the board on
  two independent attempts; the ensemble's diversity axis became the branch and, on
  the S3D family, the seed.
- **Self-training** gained four clips on S3D and nothing on ir-CSN; the
  pseudo-labelled final candidate (023) missed its range, and neither released model
  uses any test input.
- **Fusion temperature and fusion rule.** A temperature of 16 lost a clip twice; the
  log-linear pool (024) held its range but did not beat the arithmetic mean.
- **A large pretrained vision transformer** as the thermal member (025) did not
  transfer from the fold instrument to the public board and is not a final. Large
  pretrained models were otherwise used only as probes and as distillation teachers
  during development and are not part of either release.
- **Int8 storage** of the packaged file was measured on the shipped pair
  (a loss of 0.2 to 0.3 points per fold) and rejected; both released files are fp16.

## Lessons

1. Measure the corpus before the model. Most of the system's shape came from
   properties of the data that a validation split would have hidden.
2. Re-measure every lever on the backbone that ships. Four of seven inverted or
   vanished across one backbone change, in directions that were not predictable.
3. Treat a small public board as an instrument with a seven-clip noise floor, and
   write the expected range before the score. Three of four fold-passing candidates
   missed their ranges; the written ranges are what made that visible.
4. Read gains per subject, not only per fold. The changes that carried to the board
   were the ones that improved most held-out subjects; the ones that did not carry
   were driven by a few.
5. Reproducibility is a deliverable. Every submission was rehearsed from a fresh
   clone with a cold environment and one file, on CPU, before the picks were made.
