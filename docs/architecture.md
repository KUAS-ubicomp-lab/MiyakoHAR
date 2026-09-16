# Architecture

This page describes the recognition system that both released models share, the
preprocessing that runs inside `inference.sh`, and the one component in which the
two models differ. The numbers are those of the released files; the code that
implements each step is named where it applies.

## Task

CUHK-X 2026 is a cross-subject activity-recognition benchmark: 40 classes of
activities of daily living, recorded at home in several sensing modalities, scored
by top-1 accuracy against one label per clip. Eighteen subjects appear in the
training split; the test split holds four other subjects, so a model has to
recognise activities performed by people it has never seen. The Small Model Track
limits everything loaded at inference to one checkpoint file of at most
100,000,000 bytes and requires inference to run from the original files without
network access.

## Streams

| Stream | Released as | Used | Read by |
|---|---|---|---|
| Depth colormap | 640 x 480, 3 channels, PNG | yes | the depth + infrared network, channels 1 to 3 |
| Infrared | 640 x 480, 1 channel, PNG | yes | the depth + infrared network, channel 4 |
| Thermal | 320 x 240, 3 channels, JPEG | yes | the thermal network |
| IMU, radar, skeleton | time series and key points | no | not read by either model |

Depth and infrared are pixel-registered at a common origin, so one network reads
them as a four-channel stack. Thermal comes from a different sensor: it covers 59%
of the infrared field of view, and its frames carry a counter rather than a
wall-clock time, so aligning it frame by frame with the other two streams would
carry more than 0.5 s of error on 20 to 23% of clips. Thermal therefore has a
network of its own, and the two networks are combined at the probability level.
The radar stream was dropped for a different reason: it was recorded for some
subjects and switched off for others, so its mere presence identifies a group of
subjects, and a model that reads it learns subject identity rather than activity
(see [data.md](data.md)).

## Two branches, one probability vector

```
depth colormap (640x480, 3ch) + infrared (640x480, 1ch)      thermal (320x240, 3ch)
            |  pixel-registered, stacked to 4 channels                |
            |  box-reduce 4x to 160x120                               |  022: box-reduce 2x to 160x120
            |  sample T = 16 frames, resize to 224x224                |  026: decode at native 320x240
            v                                                         |  sample T = 32 frames
      ir-CSN-R50, 4-channel stem                                      |  resize to 224x224 (022) or 320x320 (026)
      12,402,920 parameters, 40-way head                              v
            |                                               ir-CSN-152, 28,965,928 parameters, 40-way head
            |  scored upright and mirrored,                           |  scored upright and mirrored,
            |  the two softmax vectors averaged                       |  the two softmax vectors averaged
            v                                                         v
       40 probabilities  ------------- arithmetic mean over the branches present -------------  40 probabilities
                                                        |
                                              one label, the largest entry
```

1. The depth colormap (3 channels) and the infrared frame (1 channel) are stacked
   into a four-channel input and read by an **ir-CSN-R50**, a 3D convolutional
   network with channel-separated convolutions (Tran et al., ICCV 2019). The stem
   is widened from 3 to 4 input channels with a response-preserving x0.75
   rescaling of the pretrained filters.
2. The thermal frame is read by an **ir-CSN-152**, the deeper network of the same
   family. The deeper network ships only on the thermal side, where the smaller
   one was capacity-limited; on the depth and infrared side it did not pay.
3. Each network produces a vector of 40 softmax probabilities. Each frame stack is
   scored twice, upright and horizontally flipped, and the two vectors are
   averaged. The branch vectors are then averaged over the branches present in the
   clip, and the class with the largest probability is the prediction.

A clip that lacks one modality is classified from the other branch rather than
dropped. Of the 405 competition test clips, 4 carry no infrared frames and 10 no
thermal frames; none is missing both.

Both networks start from published weights, mmaction2's ir-CSN checkpoints
pretrained on IG-65M and fine-tuned on Kinetics-400 (Apache-2.0), and are then
fine-tuned on the competition's training split with 40-way classification heads.
The pretraining corpus proved to be the largest single effect we measured:
replacing the earlier Kinetics-only backbone with ir-CSN-R50 moved our fold
instrument by +5.075 percentage points. During fine-tuning every BatchNorm layer
except the stem's is held at its pretrained statistics (52 of the 53 in the R50),
because those statistics are what the pretraining exists to supply. The two
networks together hold 41,368,848 parameters, counted on the packaged file with
the 40-way heads; stored in half precision they make one file of 83.5 MB (022) or
83.6 MB (026).

The ir-CSN architecture is reimplemented in `src/csn.py` from mmaction2's
definition so that the published checkpoint loads with strict key matching and no
remapping; a mismatch raises an error rather than loading silently. The model
registry and the stem adaptation are in `src/model.py`.

## Preprocessing at inference

Everything below is executed by `inference.sh` from the original frames. There is
no separate preprocessing step and no preprocessing artefact; the same functions
that built the training cache (`src/preprocess.py`, `src/dataset.py`) are called
per clip by `src/predict.py`.

| Step | Depth + infrared | Thermal, model 022 | Thermal, model 026 |
|---|---|---|---|
| Decode | 640 x 480 PNG, box-reduced 4x to 160 x 120 | 320 x 240 JPEG, box-reduced 2x to 160 x 120 | 320 x 240 JPEG at native size |
| Sample | T = 16 segments, one frame each | T = 32 segments, one frame each | T = 32 segments, one frame each |
| Resize | bilinear to 224 x 224 | bilinear to 224 x 224 | bilinear to 320 x 320 |
| Network | ir-CSN-R50 | ir-CSN-152 | ir-CSN-152 retrained for this input |

- **Sampling.** Frames are sampled by temporal segment sampling (Wang et al.,
  ECCV 2016): the clip is divided into T equal segments and one frame is taken
  from each, at a fixed position at inference so the result is deterministic.
  Each member stores its own T in the checkpoint and the loader builds the clip
  accordingly.
- **Resizing.** The aspect ratio is not preserved: a 4:3 frame is stretched 1.33x
  vertically to a square, and every trained member learned on exactly this
  geometry.
- **The decode size is a property of the checkpoint.** The 026 thermal member
  carries a stamped key in its metadata, `source_wh = [320, 240]`, which the loader
  reads and applies to that member's stream. A member without the key is decoded
  exactly as before, bit for bit, which is why the depth and infrared member
  behaves identically in both models and why the same code reproduces either
  submission from its own file.
- **No crop, no detector, no learned preprocessing.** The shipped path decodes the
  full frame of every stream and passes it to the networks. A zero-dependency
  person detector exists in `src/person_crop.py` and was measured; at the shipped
  operating point the crop did not help, so it is not used.

## What differs between the two released models

| | Model 022 | Model 026 |
|---|---|---|
| Submitted CSV | `022_r152_fence.csv` | `026_hr320_person_pixels.csv` |
| Depth + infrared network | ir-CSN-R50, T = 16, 224 x 224 | the same member, same bytes |
| Thermal network | ir-CSN-152, T = 32, 224 x 224, trained on 160 x 120 frames | ir-CSN-152, T = 32, 320 x 320, retrained on 320 x 240 frames |
| Thermal training batch | 12 | 6 |
| Parameters (both networks) | 41,368,848 | 41,368,848 |
| File | 83,543,089 bytes, fp16 | 83,559,635 bytes, fp16 |
| SHA-256 (first 16 hex) | `218266cc3230e87f` | `cd27bc72c09f6e41` |
| Public / private score | 0.76616 / 0.76960 | 0.78606 / 0.78921 |

The two files disagree on 42 of the 405 test clips; both predict all 40 classes.
The motivation for the change: a person occupies about 5.9% of a thermal frame,
so at 160 x 120 the classes that depend on the hands and on hand-held objects are
decided on a few dozen source pixels. Decoding at 320 x 240 gives the thermal
network every pixel the sensor recorded. The batch went from 12 to 6 because the
ir-CSN-152 at a 320 x 320 input needs 18.3 GB of a 24 GB card at batch 6.

## What is not in the system

No language model and no external service is called, no adaptation happens at test
time, and no state is carried from one clip to the next. The repository also
carries code paths from earlier experiments: a pseudo-label generator, a VideoMAE
V2 ViT branch (`src/vmae.py`), IMU and skeleton loaders. None of them is part of
either released model; [experiments-summary.md](experiments-summary.md) says what
each was for and why it did not ship.
