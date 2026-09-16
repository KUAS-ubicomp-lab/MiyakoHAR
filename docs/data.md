# Data

The models were trained and evaluated on the CUHK-X Multimodal Human Activity
dataset as released for the 2026 challenge's Small Model Track. The dataset is not
part of this repository: its licence permits non-commercial research use only and
forbids redistribution in any form. This page describes what the code expects to
find on disk, what the corpus contains, and what we measured about it that shaped
the model.

## Obtaining the data

The dataset is distributed by its owner, the AIoT Lab, Department of Information
Engineering, The Chinese University of Hong Kong, through the challenge
(https://openaiotlab.github.io/CUHK-X-Challenge/ and the Kaggle competition page
for the Small Model Track), under the CUHK-X License Version 2.0. Any publication,
presentation, model or other work that uses the data must cite the CUHK-X paper and
acknowledge the AIoT Lab as the creator; the citation is in
[compliance-and-licensing.md](compliance-and-licensing.md).

## Streams

| Stream | Format | Resolution | Files per clip | Used by the models |
|---|---|---|---|---|
| Depth colormap | PNG, 3 channels | 640 x 480 | one per frame, `Depth_<date>_<time>_<index>_Color.png` | yes |
| Infrared | PNG, 1 channel | 640 x 480 | one per frame, `IR_<date>_<time>_<index>.png` | yes |
| Thermal | JPEG, 3 channels | 320 x 240 | one per frame, `frame_<counter>.jpg` | yes |
| IMU | CSV | two devices | `up(LA+RA+C).csv`, `down(LL+RL).csv` | no |
| Radar | CSV | | one file | no |
| Skeleton | JSON key points | | `predictions/*.json` | no |

Depth and infrared frames share a global frame index in their file names and are
paired on that index, never on file order; the two streams are pixel-registered.
Thermal frames are named by a bare counter with no timestamp, so at inference they
are aligned to the clip as a whole, not frame by frame.

## Directory layout

**Training corpus.** Streams are the top level, then the activity, the subject
and the take:

```
HAR/data/
  Depth_Color/<action>/<user>/<take>/Depth_*_Color.png
  IR/<action>/<user>/<take>/IR_*.png
  Thermal/<action>/<user>/<take>/frame_*.jpg
  IMU/…  Radar/…  Skeleton/…
```

where `<action>` is one of the 40 class directories (`0_Wash_face` to
`39_Take_body_temperature`), `<user>` is `user1` to `user9` or `user16` to `user24` (the 18 training subjects), and `<take>` is a
recording identifier such as `1-1-3`. The label of every training clip comes from
its directory path and from nothing else. `train.sh` takes the `HAR/data`
directory and the organisers' `class_mapping.csv` (columns `action_id,
action_name`) as its two arguments.

**Test clips.** One directory per clip, the streams inside it:

```
small_model_track_test/
  SM_test_0001/
    Depth_Color/  IR/  Thermal/  IMU/  Radar/  Skeleton/
  SM_test_0002/
  …
test_file/
  test.csv                 path,prediction   (the row list; prediction empty)
  sample_submission.csv
```

`inference.sh` takes the directory that contains `small_model_track_test/` (or that
directory itself) and writes one row per entry of the row list; see
[inference-and-deployment.md](inference-and-deployment.md) for the row-list rules
and for running on clips with other names.

## The 40 classes

Class indices and names as defined by the dataset's `class_mapping.csv`.

| Id | Activity | Id | Activity | Id | Activity | Id | Activity |
|---|---|---|---|---|---|---|---|
| 0 | Wash face | 10 | Stir drinks | 20 | Check the time | 30 | Do jumping jacks |
| 1 | Brush teeth | 11 | Peel fruits | 21 | Read documents | 31 | Do stretching exercises |
| 2 | Comb hair | 12 | Sweep the floor | 22 | Turn pages | 32 | Stand up |
| 3 | Take off clothes | 13 | Mop the floor | 23 | Listen to music with headphones | 33 | Lie down |
| 4 | Wipe hands | 14 | Wipe bowls | 24 | Use a mobile phone | 34 | Sit down |
| 5 | Put on clothes | 15 | Wipe windows and tables | 25 | Watch TV | 35 | Do lunges |
| 6 | Drink water | 16 | Fold clothes | 26 | Play games | 36 | Walk |
| 7 | Eat food | 17 | Tap the keyboard | 27 | Take a selfie | 37 | Take medicine |
| 8 | Take and use tableware | 18 | Write | 28 | Jog in place | 38 | Massage oneself |
| 9 | Pour drinks | 19 | Make a phone call | 29 | Do squats | 39 | Take body temperature |

The prediction CSV carries the integer id. Class 36, Walk, is the most frequent
training class (365 of 3,036 clips) and is the per-clip fallback when a clip
cannot be decoded at all.

## Size of the splits

| Split | Subjects | Clips | Notes |
|---|---|---|---|
| Training | 18 (users 1 to 9 and 16 to 24) | 2,931 with depth and infrared; 2,891 with thermal | 3,036 clips indexed in all |
| Test | 4 subjects not in training | 405 | scored as 201 public and 204 private clips; 4 clips have no infrared, 10 no thermal, none lacks both |

### Clips per activity and subject

The 3,036 indexed training clips by activity and subject; `.` marks an empty
cell. 537 of the 720 (activity, subject) pairs are occupied. The same counts are
in `evaluation/clips_by_activity_and_subject.csv`, and `tools/fold_tables.py`
regenerates both from the held-out predictions file
([held-out-predictions.md](held-out-predictions.md)).

| Id | Activity | u1 | u2 | u3 | u4 | u5 | u6 | u7 | u8 | u9 | u16 | u17 | u18 | u19 | u20 | u21 | u22 | u23 | u24 | Clips | Subj. |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Wash face | . | . | . | 3 | 3 | 3 | . | 6 | . | 3 | . | 3 | . | 3 | 6 | 5 | 3 | 6 | **44** | 11 |
| 1 | Brush teeth | . | . | . | . | . | 3 | 3 | 6 | 1 | 3 | 3 | 6 | 3 | 3 | 3 | 5 | 6 | 6 | **51** | 13 |
| 2 | Comb hair | 3 | . | 3 | . | 6 | 2 | 3 | 4 | 3 | 3 | 3 | 3 | 3 | 6 | 3 | 6 | 3 | 3 | **57** | 16 |
| 3 | Take off clothes | . | . | . | 2 | . | . | 6 | 6 | 3 | . | 3 | 6 | 3 | . | . | 6 | 6 | . | **41** | 9 |
| 4 | Wipe hands | 3 | 4 | . | 3 | . | 3 | 6 | 6 | 6 | 3 | 6 | 6 | 8 | 3 | 6 | 6 | 3 | 3 | **75** | 16 |
| 5 | Put on clothes | 3 | . | 3 | 3 | . | . | 6 | 6 | 3 | . | 6 | 6 | 3 | . | 3 | 9 | 6 | . | **57** | 12 |
| 6 | Drink water | 6 | 12 | 9 | 6 | 15 | 12 | 3 | 3 | 6 | 12 | . | 2 | 6 | 6 | 6 | 9 | 3 | 6 | **122** | 17 |
| 7 | Eat food | 11 | 15 | 12 | 11 | 12 | 11 | 14 | 3 | 3 | 12 | 15 | 3 | 3 | 9 | 3 | 12 | 3 | 3 | **155** | 18 |
| 8 | Take and use tableware | 10 | 12 | 12 | 5 | . | 15 | 12 | . | . | 3 | 9 | . | 3 | 3 | 7 | 6 | . | . | **97** | 12 |
| 9 | Pour drinks | 6 | 5 | 6 | 6 | 12 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 15 | 3 | 6 | 2 | . | **133** | 17 |
| 10 | Stir drinks | 9 | 2 | 9 | 6 | 5 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 12 | . | 3 | 2 | . | **120** | 16 |
| 11 | Peel fruits | 8 | 6 | 5 | 12 | 12 | 9 | 3 | 9 | 3 | 9 | 3 | 9 | 3 | . | 3 | 9 | . | 3 | **106** | 16 |
| 12 | Sweep the floor | 3 | 3 | 3 | 3 | 3 | 6 | 3 | 3 | 6 | 6 | 3 | 3 | 6 | 6 | 3 | . | 3 | . | **63** | 16 |
| 13 | Mop the floor | 3 | 3 | . | 3 | 6 | 3 | 3 | 2 | 6 | 3 | 3 | 3 | 6 | 9 | 3 | . | . | . | **56** | 14 |
| 14 | Wipe bowls | 5 | . | 3 | 3 | . | 3 | 3 | 3 | . | 3 | 3 | 3 | . | 3 | . | 3 | . | . | **35** | 11 |
| 15 | Wipe windows and tables | . | . | . | . | 3 | 7 | 5 | 6 | 3 | 6 | 6 | 6 | 3 | 3 | . | 3 | 3 | . | **54** | 12 |
| 16 | Fold clothes | . | 3 | . | . | . | . | 3 | 3 | . | . | 3 | 3 | . | . | . | 3 | 3 | 3 | **24** | 8 |
| 17 | Tap the keyboard | 6 | . | 6 | 6 | 3 | 3 | 4 | 3 | 11 | 3 | 3 | 3 | 12 | 9 | . | 6 | 3 | 6 | **87** | 16 |
| 18 | Write | . | 5 | 3 | 3 | 2 | 3 | . | 3 | . | . | . | 6 | . | 3 | . | 3 | 5 | 3 | **39** | 11 |
| 19 | Make a phone call | . | 3 | 6 | 2 | 3 | 3 | . | 5 | . | 3 | . | 6 | . | . | 3 | 3 | 3 | 3 | **43** | 12 |
| 20 | Check the time | 6 | . | . | 3 | 3 | 12 | 3 | 9 | 6 | 9 | 6 | 9 | 6 | 12 | 6 | 3 | . | 3 | **96** | 15 |
| 21 | Read documents | 3 | 2 | 8 | 3 | 9 | 6 | 3 | 6 | 6 | 6 | 3 | 6 | 3 | 3 | . | 6 | 3 | 6 | **82** | 17 |
| 22 | Turn pages | 3 | 2 | 6 | 3 | 3 | 6 | . | 6 | . | 6 | . | 6 | . | 6 | 3 | 3 | . | 6 | **59** | 13 |
| 23 | Listen to music with headphones | 3 | 4 | 5 | 3 | . | 3 | 6 | 4 | 6 | 6 | 6 | 3 | 6 | 3 | 6 | 3 | 5 | 6 | **78** | 17 |
| 24 | Use a mobile phone | . | 5 | . | 6 | . | 6 | 3 | . | . | 3 | 1 | . | . | 3 | 9 | 3 | 3 | 6 | **48** | 11 |
| 25 | Watch TV | 3 | . | . | . | . | . | 3 | . | . | . | . | . | . | . | . | . | . | 6 | **12** | 3 |
| 26 | Play games | . | . | . | . | . | 2 | . | . | 12 | 3 | . | . | 11 | . | . | . | 3 | 9 | **40** | 6 |
| 27 | Take a selfie | 3 | . | . | 3 | . | . | 3 | 3 | 3 | . | 3 | 3 | 3 | 3 | 3 | 3 | 5 | 3 | **41** | 13 |
| 28 | Jog in place | . | 5 | 3 | 3 | . | 3 | 3 | 3 | . | 3 | 3 | 3 | . | . | . | 3 | 2 | 3 | **37** | 12 |
| 29 | Do squats | 6 | 6 | 3 | . | . | 3 | 3 | . | 11 | 3 | 3 | 3 | 12 | 6 | 3 | 3 | 9 | 3 | **77** | 15 |
| 30 | Do jumping jacks | 3 | 5 | . | 3 | . | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | . | . | **44** | 14 |
| 31 | Do stretching exercises | 3 | 11 | 6 | 3 | 3 | . | 6 | 9 | 5 | . | 6 | 9 | 6 | 3 | . | 9 | 6 | 3 | **88** | 15 |
| 32 | Stand up | 9 | 7 | 6 | 9 | 9 | 3 | 3 | 2 | 3 | 6 | 3 | 6 | 3 | 3 | 6 | 4 | 6 | 6 | **94** | 18 |
| 33 | Lie down | . | . | 3 | . | 6 | 6 | 3 | . | 6 | 6 | 3 | . | 3 | . | . | . | . | 3 | **39** | 9 |
| 34 | Sit down | 12 | 11 | 7 | 3 | 6 | 13 | 17 | 6 | 9 | 9 | 12 | 3 | 9 | . | . | 14 | 4 | 12 | **147** | 16 |
| 35 | Do lunges | 3 | . | . | . | 3 | 3 | . | 3 | . | 3 | . | 3 | . | 3 | . | . | 6 | . | **27** | 8 |
| 36 | Walk | 17 | 25 | 21 | 15 | 30 | 21 | 22 | 14 | 27 | 21 | 18 | 15 | 26 | 9 | 24 | 22 | 17 | 21 | **365** | 18 |
| 37 | Take medicine | . | 3 | 6 | 3 | 3 | 3 | 3 | 6 | 6 | 3 | 3 | 6 | 6 | 3 | 3 | 3 | 6 | 6 | **72** | 17 |
| 38 | Massage oneself | 3 | 6 | 6 | 6 | . | 3 | . | . | 6 | 3 | . | 3 | 5 | 6 | 9 | 6 | 6 | 6 | **74** | 14 |
| 39 | Take body temperature | . | 3 | 3 | . | . | 3 | 6 | . | 6 | 3 | 6 | 3 | 6 | . | 6 | 3 | 3 | 6 | **57** | 13 |
| | **All activities** | **153** | **168** | **163** | **143** | **160** | **203** | **185** | **169** | **187** | **186** | **166** | **179** | **188** | **159** | **133** | **194** | **141** | **159** | **3036** | |
| | Activities with clips | 27 | 26 | 26 | 30 | 23 | 34 | 33 | 32 | 29 | 33 | 31 | 35 | 30 | 29 | 25 | 34 | 31 | 29 | | |

## What we measured about the corpus, and what it forced

Each item below is a measurement over the real data made during development. They
are recorded here because several of them are the reason the model looks the way it
does.

- **The corpus is not the factorial design it appears to be.** 537 of the 720
  possible (activity, subject) pairs exist. Class frequency is imbalanced 30.4 to 1
  (Walk 365 clips, Watch TV 12), and the imbalance is structured by subject: class
  25 exists only for three subjects. On a cross-subject benchmark, correcting class
  frequency by resampling or re-weighting therefore amplifies subject identity, so
  the models do neither.
- **The radar stream is a perfect subject-block indicator, so it was dropped.**
  Radar recorded for users 1 to 9 and was off for every one of users 16 to 24.
  Its presence is a subject-group label, not an activity signal: feeding radar, or
  even a missing-modality flag, shows a gain in any subject-mixed validation and
  cannot generalise to new subjects.
- **Thermal cannot be channel-stacked with the vision streams.** Thermal frames are
  named by a counter with no timestamp; counters reset per take, so only per-clip
  alignment exists at test time, with more than 0.5 s of error on 20 to 23% of
  clips. Thermal also sees a different field of view (59% of the infrared frame,
  linearly), so a naive box transfer between the two gives an intersection over
  union of 0.35. Hence two branches, fused at the probability level.
- **The person is small in the frame.** The person occupies a median 5.9% of the
  frame by silhouette (a tenth of clips under 3.3%). A zero-dependency
  depth-threshold person detector was built and beat a pretrained SSDLite detector
  in a head-to-head on 293 clips, but at the shipped operating point cropping did
  not help (a loss of 2.11 points over six cells), so no crop is used. The same
  observation motivated model 026: decoding thermal at its native 320 x 240 gives
  the network every pixel the sensor recorded of a small person.
- **The test set contains no frames from training, and has duplicates of its own.**
  Zero of 9,190 test frames are byte-identical to any training frame. But 66 of the
  405 test clips share at least one frame with another test clip and 16 are at
  least 90% duplicated by another, so about 4 points of the public score may be
  irreducible in a direction that cannot be determined.
- **Not every sensor channel is usable.** Of the IMU's 18 numeric channels, 8
  (accelerometer, gyroscope, pitch, roll) are free of a measured train-to-test shift
  in absolute orientation (yaw shifts by +34.7 degrees); the magnetometer, yaw and
  quaternion are not, and battery level identifies the recording session. The IMU
  loader in `src/imu_data.py` implements these rules; no released model reads the
  IMU.

## What is in this repository, and what is not

No frame, no sensor file and no derived cache from the dataset is tracked, and
`.gitignore` excludes every corpus path and image extension. Three kinds of
derived file of our own are tracked. `splits/folds.yaml` is our assignment of the
18 training subjects to three validation folds (subject numbers only). The two
submitted CSVs under `submissions/` are our model's predictions per test clip,
which `verify.sh` compares against. The tables under `evaluation/` are our
held-out predictions for the training clips and the accuracy and count tables
derived from them ([held-out-predictions.md](held-out-predictions.md)); to be
usable they list each training clip by the directory path that carries its label,
which is the dataset owner's annotation. Those annotations are licensed under
CC BY-NC 4.0 and are reproduced with attribution for non-commercial research, as
that licence permits; the clips themselves are not. The `class_mapping.csv` table
is the dataset owner's and is read from the data directory, not shipped.
