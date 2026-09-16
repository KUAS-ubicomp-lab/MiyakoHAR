# Model card

## Model details

- **Name.** MiyakoHAR, models 026 (default) and 022.
- **Developed by.** Team MiyakoHAR, Ubiquitous and Personal Computing Lab, Kyoto
  University of Advanced Science: Joseph Arthur Koo, Ken Argani Toendan, Nhung
  Huyen Hoang; faculty advisor Prof. Zilu Liang.
- **Date.** Model 022 trained and submitted on 29 August 2026; model 026 on
  13 September 2026. Released 16 September 2026 as version 1.0.0.
- **Type.** Two 3D convolutional video classifiers (ir-CSN-R50 on depth + infrared,
  ir-CSN-152 on thermal) fused by averaging class probabilities; 41,368,848
  parameters; one fp16 file of 83.6 MB. See [architecture.md](architecture.md).
- **Initialisation.** mmaction2's IG-65M to Kinetics-400 ir-CSN checkpoints
  (Apache-2.0), fine-tuned on the competition's training split.
- **Licence.** Apache License 2.0 for our code and the fine-tuned weights; see
  [compliance-and-licensing.md](compliance-and-licensing.md) for the pretrained
  weights and the dataset terms.
- **Contact.** The corresponding author is named in
  `technical-description-022-026.pdf`.

## Intended use

The models classify a short clip of one person performing one of the 40 CUHK-X
activities of daily living (washing the face, eating, sweeping the floor,
taking medicine, and so on; the list is in [data.md](data.md)), recorded by the
CUHK-X sensor rig: a depth camera and an infrared camera at 640 x 480 that are
pixel-registered, and a thermal camera at 320 x 240. Input is a directory of frames
per stream; output is one class index per clip.

The primary use is research on cross-subject activity recognition and as the
reference implementation of the team's competition entry. A deployment that keeps
the same sensors, the same frame formats and a similar home setting is within the
models' training distribution; anything else is not, see the limitations below.

**Out of scope.** RGB video (the models never saw colour images); other sensor
rigs, resolutions or mounting geometries without re-validation; activities outside
the 40 classes (the model always answers with one of them); multiple people in the
frame; any use as a safety-critical or medical decision system; any use on people
who have not consented to being recorded.

## Training data

The CUHK-X Small Model Track training split: 2,931 clips with depth and infrared
frames and 2,891 clips with thermal frames, from 18 subjects (users 1 to 9 and 16
to 24), labelled with one of 40 activities. No other data was used. No test clip,
test label or pseudo-label enters either released model. The corpus is described
in [data.md](data.md); it is licensed for non-commercial research and is not
redistributable, so it is not part of this repository.

Every released member was trained on all 18 subjects. The recipe was selected
beforehand on a frozen leave-6-subjects-out split, so no validation clip selects
the shipped weights; see [training.md](training.md).

## Evaluation

**Competition test set.** 405 clips from four subjects that do not appear in
training, scored as 201 public and 204 private clips by the organisers.

| Model | Private score | Public score |
|---|---|---|
| 026 (default) | 0.78921 (161 of 204), 39th of 326 teams | 0.78606 (158 of 201) |
| 022 | 0.76960 (157 of 204) | 0.76616 (154 of 201) |

**Held-out subjects, our own instrument.** On the frozen 3-fold split, each fold
trains on 12 training subjects and validates on the other 6, with two seeds, so
every training subject is scored once by members that never saw it. The 022
recipe scores 72.7% clip-weighted over the 18 subjects; the 026 recipe scores
74.4%, an improvement of +1.715 percentage points (t = +3.68 over the six
subject-disjoint cells, 6 of 6 positive). Per-subject accuracy of the 022 recipe
ranges from 62.41% (subject 21) to 83.69% (subject 9), a between-subject standard
deviation of 6.2 points. The full tables are in [results.md](results.md).

**Factors.** The dominant factor is the subject. Two of the 18 training subjects
sit more than 7.3 points below the mean on their own, and the same model spans 21
points between the easiest and the hardest subject. A sample of a few unseen
subjects therefore carries a sampling term of about 4.2 points of standard
deviation for two subjects before any property of the model is measured. Class
support is the second factor: the training split is imbalanced 30.4 to 1 between
its most and least frequent class (Walk, 365 clips; Watch TV, 12 clips from 3
subjects). Both models nevertheless predict all 40 classes on the test set.

## Behaviour on unusual input

- A clip that lacks one stream is classified from the other. A clip with no usable
  stream, or one whose decoding fails, receives the majority training class (36,
  Walk) and is reported on the error stream; the run continues.
- If the model file itself cannot be loaded, the entry point refuses to write any
  output and exits with status 1 rather than emitting a constant prediction.
- Inference is per clip: a random subset of 40 test clips reproduces the full run's
  predictions for those clips exactly. The result for a clip does not depend on
  which other clips are processed with it.

## Limitations

1. **Evidence for the 026 change.** On the fold instrument the thermal change is
   +1.715 points; on the public leaderboard it is four clips of 201, inside one
   standard error, and on the private leaderboard four clips of 204. Of the five
   candidates scored publicly after 022, three missed the range written for them
   before their score existed; 026 held its range, and is the only one that did
   so above 022. We describe
   026 as consistent with its fold result and not as a measured gain.
2. **Spread between subjects.** The same recipe scores between 62.41% and 83.69%
   across the 18 training subjects when each is held out. Accuracy on a new person
   should be expected anywhere in that range.
3. **Geometry.** Every frame is stretched 1.33x vertically to a square; the models
   learned on this geometry and expect it. Frames of a different aspect ratio or
   resolution, or from a camera mounted differently, are outside the training
   distribution.
4. **Class support.** The least frequent class has 12 training clips from 3
   subjects, and the fold instrument has no validation clip of that class in one
   fold.
5. **Cost.** The thermal network of 026 needs 1.8 times the CPU time of 022's per
   clip (3.45 s against 1.91 s at 8 threads) and 1.4 times the GPU time.
6. **Test-set structure.** In the competition's test set, 66 of 405 clips share at
   least one frame with another test clip and 16 are at least 90% duplicated by
   another, so about 4 points of the public score may be irreducible in a
   direction that cannot be determined. This is a property of that test set, not
   of the model, but it bounds how precisely the reported scores measure anything.

## Ethical considerations

The models recognise what a person is doing at home from depth, infrared and
thermal imagery. They contain no face-recognition or identity component and never
saw RGB frames, but the depth and thermal streams still show people in private
settings. Any deployment must rest on the informed consent of the people recorded,
must keep the recordings under their control, and must not be used for
surveillance of people who have not agreed to it. The training data were collected
and released by the CUHK AIoT Lab under their own consent procedures; the dataset
licence restricts use to non-commercial research and forbids redistribution.
