# Held-out predictions

This page describes the per-clip held-out predictions of the two released recipes
and the tables derived from them, all under `evaluation/`. They exist so that the
models can be examined at a finer grain than the fold means in
[results.md](results.md): accuracy per activity, accuracy per subject, and which
activities are confused with which.

## The files

| File | Content |
|---|---|
| `evaluation/fold_predictions_022_026.csv` | One row per training clip (3,036 rows): its label and subject, the fold in which it was held out, which streams it carries, and the predicted class of each network and of the fused model, for both recipes and both training seeds. |
| `evaluation/per_activity_accuracy.csv` | Accuracy per activity, for each recipe fused and for each network alone. |
| `evaluation/per_subject_accuracy.csv` | The same per training subject. |
| `evaluation/confusion_fused_026.csv`, `evaluation/confusion_fused_022.csv` | The 40 x 40 confusion matrix of each recipe (rows true, columns predicted), both seeds pooled. |
| `evaluation/clips_by_activity_and_subject.csv` | The training split's clip count for every (activity, subject) pair. |

The columns of the predictions file:

| Column | Meaning |
|---|---|
| `clip_id` | The clip's directory path in the training corpus, `<action>/<user>/<take>`, which is also where its label comes from ([data.md](data.md)). |
| `label`, `activity` | The true class index and name; the 40 classes are listed in [data.md](data.md). |
| `subject` | The training subject, 1 to 9 or 16 to 24. |
| `fold` | The fold of `splits/folds.yaml` in which this subject was held out. |
| `has_depth_ir`, `has_thermal` | 1 if the clip carries depth and infrared frames, and thermal frames, respectively. A clip that lacks a stream is predicted by the other network alone, as at inference. |
| `pred_depth_ir_seed1`, `pred_depth_ir_seed2` | The predicted class of the depth + infrared network (ir-CSN-R50), which both recipes share; empty when the clip has no depth and infrared frames. |
| `pred_thermal_022_seed1`, `pred_thermal_022_seed2`, `pred_thermal_026_seed1`, `pred_thermal_026_seed2` | The predicted class of each recipe's thermal network (ir-CSN-152; the 026 network reads thermal decoded at 320 x 240); empty when the clip has no thermal frames. |
| `pred_fused_022_seed1`, `pred_fused_022_seed2`, `pred_fused_026_seed1`, `pred_fused_026_seed2` | The predicted class of each recipe's fused model, the mean of the probabilities of the networks present. Never empty. |

## Where the predictions come from

They are the predictions of the recipe-selection instrument described in
[training.md](training.md): the frozen 3-fold subject-grouped split, each fold
training on 12 of the 18 training subjects and validating on the other 6, at two
training seeds. Every training subject is held out exactly once, so the file is an
out-of-fold prediction of the entire training split. The predictions are those of
the fold members, not of the released weights: the released models were retrained
on all 18 subjects, so every training clip is in-sample for them and no held-out
prediction of theirs exists.

Each network's prediction is the mean of its softmax on the upright clip and on
the horizontally flipped clip; the fused prediction is the mean over the networks
present, exactly as `src/predict.py` fuses at inference. `tools/fold_predictions.py`
performs this derivation from the fold runs' cached logits and refuses to write
the file unless the fused accuracy of every one of the twelve (recipe, fold,
seed) cells reproduces the value published in [training.md](training.md) to
0.005 points; the committed file passed that check on all twelve. The cached
logits are development outputs and are not part of this repository, so the
derivation is documented by the tool rather than re-runnable from a clone.
`tools/fold_tables.py` regenerates every aggregate table and every Markdown
table on this page from the committed predictions file alone.

Accuracy in the tables below is the share of correct predictions over clips and
seeds: each clip is predicted once per seed, so it counts twice. A network alone
is scored on the clips that carry its stream (2,931 clips with depth and infrared,
2,891 with thermal); the fused model is scored on all 3,036.

## Accuracy per activity

| Id | Activity | Clips | Subjects | Fused 026 | Fused 022 | Depth + infrared alone | Thermal alone (026) | Thermal alone (022) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Wash face | 44 | 11 | 71.6 | 72.7 | 68.2 | 72.0 | 78.0 |
| 1 | Brush teeth | 51 | 13 | 70.6 | 72.5 | 80.2 | 59.6 | 56.4 |
| 2 | Comb hair | 57 | 16 | 78.9 | 78.1 | 70.4 | 81.5 | 82.4 |
| 3 | Take off clothes | 41 | 9 | 78.0 | 78.0 | 78.0 | 78.0 | 76.8 |
| 4 | Wipe hands | 75 | 16 | 77.3 | 75.3 | 76.4 | 74.6 | 73.9 |
| 5 | Put on clothes | 57 | 12 | 95.6 | 95.6 | 91.2 | 84.3 | 84.3 |
| 6 | Drink water | 122 | 17 | 83.6 | 83.2 | 72.1 | 79.5 | 77.0 |
| 7 | Eat food | 155 | 18 | 71.9 | 64.8 | 56.8 | 70.5 | 64.6 |
| 8 | Take and use tableware | 97 | 12 | 34.5 | 36.6 | 30.9 | 34.6 | 34.6 |
| 9 | Pour drinks | 133 | 17 | 84.6 | 81.2 | 67.6 | 82.0 | 79.7 |
| 10 | Stir drinks | 120 | 16 | 74.2 | 70.8 | 57.7 | 75.4 | 77.1 |
| 11 | Peel fruits | 106 | 16 | 73.1 | 68.9 | 62.3 | 68.3 | 66.3 |
| 12 | Sweep the floor | 63 | 16 | 82.5 | 81.0 | 84.1 | 73.7 | 70.2 |
| 13 | Mop the floor | 56 | 14 | 64.3 | 63.4 | 47.3 | 54.7 | 55.7 |
| 14 | Wipe bowls | 35 | 11 | 68.6 | 67.1 | 62.9 | 65.2 | 62.1 |
| 15 | Wipe windows and tables | 54 | 12 | 60.2 | 60.2 | 56.7 | 59.4 | 56.6 |
| 16 | Fold clothes | 24 | 8 | 58.3 | 58.3 | 58.3 | 60.4 | 56.2 |
| 17 | Tap the keyboard | 87 | 16 | 87.4 | 83.3 | 76.2 | 83.5 | 82.4 |
| 18 | Write | 39 | 11 | 19.2 | 20.5 | 19.7 | 21.8 | 15.4 |
| 19 | Make a phone call | 43 | 12 | 61.6 | 58.1 | 29.1 | 70.2 | 69.0 |
| 20 | Check the time | 96 | 15 | 60.4 | 57.8 | 51.6 | 58.6 | 56.5 |
| 21 | Read documents | 82 | 17 | 57.9 | 52.4 | 47.9 | 57.5 | 53.8 |
| 22 | Turn pages | 59 | 13 | 35.6 | 35.6 | 33.1 | 33.1 | 30.5 |
| 23 | Listen to music with headphones | 78 | 17 | 73.7 | 74.4 | 63.3 | 70.8 | 69.5 |
| 24 | Use a mobile phone | 48 | 11 | 20.8 | 7.3 | 8.3 | 31.1 | 14.4 |
| 25 | Watch TV | 12 | 3 | 4.2 | 0.0 | 8.3 | 0.0 | 0.0 |
| 26 | Play games | 40 | 6 | 32.5 | 36.2 | 17.5 | 38.8 | 42.5 |
| 27 | Take a selfie | 41 | 13 | 76.8 | 76.8 | 76.8 | 70.7 | 72.0 |
| 28 | Jog in place | 37 | 12 | 94.6 | 94.6 | 87.8 | 100.0 | 100.0 |
| 29 | Do squats | 77 | 15 | 90.9 | 92.9 | 88.3 | 90.4 | 91.8 |
| 30 | Do jumping jacks | 44 | 14 | 97.7 | 97.7 | 89.8 | 97.7 | 96.6 |
| 31 | Do stretching exercises | 88 | 15 | 82.4 | 80.7 | 82.9 | 80.1 | 78.8 |
| 32 | Stand up | 94 | 18 | 92.6 | 92.0 | 83.3 | 92.9 | 91.2 |
| 33 | Lie down | 39 | 9 | 92.3 | 92.3 | 87.9 | 85.9 | 88.5 |
| 34 | Sit down | 147 | 16 | 94.6 | 91.8 | 91.5 | 88.0 | 84.6 |
| 35 | Do lunges | 27 | 8 | 53.7 | 55.6 | 51.9 | 53.8 | 57.7 |
| 36 | Walk | 365 | 18 | 98.1 | 97.9 | 97.8 | 97.1 | 97.1 |
| 37 | Take medicine | 72 | 17 | 57.6 | 52.1 | 39.7 | 56.5 | 51.4 |
| 38 | Massage oneself | 74 | 14 | 54.7 | 51.4 | 45.2 | 47.1 | 43.4 |
| 39 | Take body temperature | 57 | 13 | 54.4 | 57.0 | 31.8 | 57.4 | 57.4 |
| | All clips, clip-weighted | 3036 | 18 | **74.4** | **72.7** | **66.6** | **72.1** | **70.5** |

Clip-weighted, the 026 recipe reaches 74.4% and the 022 recipe 72.7%. The
locomotion and posture classes are close to the ceiling: Walk 98.1%, Do jumping
jacks 97.7%, Put on clothes 95.6%, and Jog in place, Sit down, Stand up and Lie
down above 92%. The weak classes are activities that differ in the hands and in a
held object rather than in the body: Watch TV 4.2% (12 clips from 3 subjects, so
the instrument has almost no support for it), Write 19.2%, Use a mobile phone
20.8%, Play games 32.5%, Take and use tableware 34.5% and Turn pages 35.6%. The
026 change, thermal decoded at its native resolution, gained most on such classes
(Use a mobile phone +13.5 points, Eat food +7.1, Read documents and Take medicine
+5.5 each, Peel fruits +4.2, Tap the keyboard +4.1) and lost ground on a few
(Play games -3.7, Take body temperature -2.6, Take and use tableware -2.1). The
two networks err differently: thermal alone is the stronger network on most
classes, while depth + infrared alone is clearly better on Brush teeth (80.2%
against 59.6%) and Sweep the floor (84.1% against 73.7%), which is the
complementarity the fusion draws on. Per-activity figures rest on 12 to 365
clips each and should be read with that support in mind; within one class the
two seeds often differ by several points.

## Accuracy per subject

| Subject | Fold | Clips | Fused 026 | Fused 022 | Depth + infrared alone | Thermal alone (026) | Thermal alone (022) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 153 | 70.3 | 69.3 | 60.4 | 64.9 | 63.9 |
| 2 | 1 | 168 | 77.1 | 76.5 | 75.1 | 61.6 | 59.5 |
| 3 | 2 | 163 | 72.1 | 67.5 | 59.6 | 72.4 | 71.4 |
| 4 | 0 | 143 | 75.2 | 72.7 | 66.8 | 73.9 | 71.5 |
| 5 | 0 | 160 | 82.8 | 79.7 | 70.5 | 78.7 | 76.5 |
| 6 | 1 | 203 | 62.3 | 63.8 | 56.5 | 57.4 | 60.3 |
| 7 | 2 | 185 | 77.3 | 73.5 | 68.8 | 78.6 | 76.1 |
| 8 | 1 | 169 | 65.7 | 65.7 | 60.3 | 69.6 | 69.3 |
| 9 | 0 | 187 | 83.7 | 83.7 | 76.2 | 81.7 | 82.0 |
| 16 | 2 | 186 | 78.8 | 75.5 | 69.1 | 76.5 | 76.8 |
| 17 | 1 | 166 | 81.0 | 79.5 | 73.2 | 74.8 | 74.5 |
| 18 | 1 | 179 | 71.8 | 68.4 | 64.3 | 68.1 | 62.4 |
| 19 | 1 | 188 | 81.1 | 81.9 | 78.5 | 80.3 | 81.7 |
| 20 | 2 | 159 | 76.1 | 73.6 | 62.9 | 77.0 | 74.5 |
| 21 | 0 | 133 | 64.7 | 62.4 | 58.6 | 65.2 | 58.7 |
| 22 | 0 | 194 | 71.6 | 70.4 | 65.2 | 68.1 | 66.2 |
| 23 | 2 | 141 | 68.8 | 66.7 | 60.7 | 66.7 | 63.3 |
| 24 | 0 | 159 | 76.7 | 74.8 | 69.5 | 75.8 | 70.6 |

The 026 recipe ranges from 62.3% (subject 6) to 83.7% (subject 9), a
between-subject standard deviation of about 6 points, as for the 022 recipe in
[results.md](results.md). The subject remains the dominant factor: the same
model spans 21 points between its easiest and its hardest subject.

## Most confused pairs

Of the 026 recipe's 1555 errors over 6072 clip-seed predictions, the ten most frequent pairs of
confused activities, counting both directions:

| Pair | Confusions, both directions | Share of all errors |
|---|---:|---:|
| Read documents and Turn pages | 95 | 6.1% |
| Take and use tableware and Pour drinks | 57 | 3.7% |
| Sweep the floor and Mop the floor | 56 | 3.6% |
| Drink water and Eat food | 54 | 3.5% |
| Pour drinks and Stir drinks | 38 | 2.4% |
| Use a mobile phone and Play games | 36 | 2.3% |
| Drink water and Take medicine | 33 | 2.1% |
| Take and use tableware and Stir drinks | 23 | 1.5% |
| Eat food and Take and use tableware | 23 | 1.5% |
| Do squats and Do lunges | 21 | 1.4% |

Eight of the ten pairs are activities performed with the hands at a table or a
surface, and differ in the object held or in the motion of the hands. The two
pairs of activities that are temporal reversals of each other are not among them:
Stand up and Sit down are confused with each other 3 times and Take
off clothes and Put on clothes 14 times, out of 1555 errors.

## Regenerating the tables

```
.venv/bin/python tools/fold_tables.py --markdown
```

rewrites the five aggregate CSVs from `evaluation/fold_predictions_022_026.csv`
and prints the Markdown tables of this page and of [data.md](data.md). Rebuilding
the predictions file itself needs the fold runs' cached logits:

```
.venv/bin/python tools/fold_predictions.py --runs-dir <runs> --out evaluation/fold_predictions_022_026.csv
```

## Limits

- **No test labels.** The released models' predictions on the 405 test clips are
  in `submissions/`, but the organisers have not released the test labels, so no
  per-activity or per-subject accuracy exists on the test subjects; only the two
  leaderboard totals do ([results.md](results.md)).
- **Support.** Class 25 (Watch TV) has 12 training clips from 3 subjects and no
  clip in fold 1's validation set; class 16 (Fold clothes) has 24 and class 35
  (Do lunges) 27. Their accuracies are estimates from a handful of clips.
- **Fold members, not released weights.** The predictions come from the twelve
  networks trained on 12 subjects each. The released models, trained on all 18
  subjects, are expected to be at least as accurate on new subjects, but that
  cannot be measured on the training clips.
- **Attribution.** The `clip_id` column names each training clip by the directory
  path that carries its label, which is the dataset owner's annotation; see
  [compliance-and-licensing.md](compliance-and-licensing.md) for the terms under
  which it is reproduced here.
