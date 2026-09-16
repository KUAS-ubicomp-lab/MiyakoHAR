# MiyakoHAR

Cross-subject human activity recognition from depth, infrared and thermal video.
This is team MiyakoHAR's solution to the CUHK-X 2026 Multimodal Human Activity
Challenge, Small Model Track (UbiComp / ISWC 2026), built at the Ubiquitous and
Personal Computing Lab, Kyoto University of Advanced Science.

The model reads a clip of a person performing one of 40 activities of daily living,
recorded at home by a depth camera, an infrared camera and a thermal camera, and
returns the activity. It is two 3D convolutional networks packaged in one file of
83.6 MB. Inference is per clip, stateless, needs no network access and no GPU.

| CUHK-X 2026 Small Model Track | Score | Clips correct |
|---|---|---|
| Private leaderboard (the scored one), **39th place** | **0.78921** | 161 of 204 |
| Public leaderboard | 0.78606 | 158 of 201 |

Both scores belong to submission 026, the default model of this repository. The
team's other final, submission 022, is released beside it (private 0.76960, public
0.76616). The private score decides the competition; it was revealed after the
leaderboard froze on 16 September 2026. Details in [docs/results.md](docs/results.md).

## Quick start

Requirements: Python 3.12, about 3 GB of RAM for inference on CPU, a GPU optional.
The repository is private, so downloading the release asset needs the GitHub CLI
(`gh`) signed in to an account with access, or a browser session on the
[releases page](https://github.com/KUAS-ubicomp-lab/MiyakoHAR/releases).

```bash
git clone git@github.com:KUAS-ubicomp-lab/MiyakoHAR.git
cd MiyakoHAR
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# The trained model: one file, at the path inference.sh loads.
mkdir -p checkpoints
gh release download v1.0.0 --repo KUAS-ubicomp-lab/MiyakoHAR \
   --pattern model-026.pth --output checkpoints/model.pth
sha256sum checkpoints/model.pth        # must start with cd27bc72c09f6e41

# Predict every clip under <data_dir>; writes rows of path,prediction.
./inference.sh <data_dir> predictions.csv
```

`<data_dir>` holds one folder per clip, each with the frames of its streams
(`Depth_Color/`, `IR/`, `Thermal/`). The exact layout, the row-list options and
the Python API are in [docs/inference-and-deployment.md](docs/inference-and-deployment.md).
The 405 competition test clips take about 90 seconds on an RTX 3090 and about
23 minutes on a CPU with 8 threads.

To confirm that an installation reproduces the competition submission bit for
bit (this needs the competition's test clips):

```bash
./verify.sh <data_dir> submissions/026_hr320_person_pixels.csv
```

The script prints the checksum and size of the model file, re-runs `inference.sh`
on the test clips and exits 0 only when the produced CSV is byte-identical to the
submitted one. Both released models were reproduced this way from a fresh clone
with a freshly built environment, 405 of 405 rows identical.

## Repository layout

| Path | Purpose |
|---|---|
| `inference.sh` | Entry point: `<data_dir>` in, prediction CSV out. Runs the complete preprocessing from the original frames. |
| `train.sh` | Rebuilds the two trained networks from the training corpus. |
| `verify.sh` | Reproduction check: re-runs inference and diffs against a submitted CSV. |
| `src/` | The model (`model.py`, `csn.py`), decoding and sampling (`preprocess.py`, `dataset.py`), the predictor (`predict.py`), the trainer (`train.py`, `swad.py`, `pk_sampler.py`), the corpus index (`manifest.py`). `imu_data.py`, `skeleton_data.py`, `person_crop.py` and `vmae.py` are development paths that no released model uses. |
| `tools/` | `package.py` packs trained members into the single inference file and weighs it; `package_int8.py` is its int8 variant; `freshclone_rehearsal.sh` runs the committee's procedure end to end; `derive_index_from_cache.py` and `p0d_efficiency.py` support training and benchmarking. |
| `tests/` | Eleven checks on the model code, the packaged file and the entry point (offline, robustness, statelessness). See [docs/inference-and-deployment.md](docs/inference-and-deployment.md#tests). |
| `splits/folds.yaml` | The frozen leave-6-subjects-out split behind every model-selection decision. |
| `submissions/` | The two final CSVs that `verify.sh` compares against. |
| `requirements.txt` | The pinned inference environment. `environment-labpc.txt` and `environment-alien.txt` are the full package lists of the two training machines. |
| `NOTICE` | Every third-party component and pretrained weight, with its licence. |
| `docs/` | The documentation listed below. |

The trained weights are not in the repository. They are attached to the GitHub
release as `model-026.pth`, `model-022.pth` and `SHA256SUMS`.

## Documentation

| Document | Read it for |
|---|---|
| [docs/architecture.md](docs/architecture.md) | The recognition system: streams, the two networks, preprocessing, fusion, and what differs between the two released models. |
| [docs/model-card.md](docs/model-card.md) | Intended use, training data, evaluation, per-subject variability, limitations. |
| [docs/data.md](docs/data.md) | The CUHK-X dataset as the code expects it: streams, directory layout, the 40 classes, licence terms, and what we measured about the corpus. |
| [docs/training.md](docs/training.md) | The recipe and hyper-parameters, compute, how to retrain and repackage, and the validation instrument. |
| [docs/inference-and-deployment.md](docs/inference-and-deployment.md) | Input contract, command line, Python API, timings and memory, the model file format, failure behaviour, tests. |
| [docs/results.md](docs/results.md) | Leaderboard results, every submission, the validation instrument, and how measured gains transferred. |
| [docs/experiments-summary.md](docs/experiments-summary.md) | What was tried over five weeks, what worked, what did not, and why. |
| [docs/compliance-and-licensing.md](docs/compliance-and-licensing.md) | Licences of code, weights and data; the competition's constraints and how each is met. |
| [docs/technical-description-022-026.pdf](docs/technical-description-022-026.pdf) | The 16-page technical description of both finals, written for the competition's verification committee. |

## The model in one paragraph

Depth and infrared frames are pixel-registered, so they are stacked into a
four-channel input and read by an ir-CSN-R50, a 3D convolutional network with
channel-separated convolutions (12.4 million parameters). Thermal frames come from
a separate sensor with its own field of view and clock, so they are read by their
own network, an ir-CSN-152 (29.0 million parameters). Both start from published
IG-65M to Kinetics-400 weights and are fine-tuned on the 18 training subjects with
40-way heads, every BatchNorm layer except the stem's held at its pretrained
statistics. Each network scores a clip upright and mirrored and averages the two
probability vectors; the branch vectors are then averaged over the streams present,
and the largest entry is the label. The two released models differ in the thermal
network only: model 026 reads thermal frames at the sensor's native 320 x 240 into
a 320 x 320 input, model 022 reduces them to 160 x 120 and reads 224 x 224.

## Team

Joseph Arthur Koo (lead), Ken Argani Toendan and Nhung Huyen Hoang, with
Prof. Zilu Liang as faculty advisor. Ubiquitous and Personal Computing Lab,
Faculty of Engineering, Kyoto University of Advanced Science.

## Licence and attribution

Our code is licensed under the Apache License 2.0 (`LICENSE`). Third-party
components and the pretrained weights the models start from are identified in
`NOTICE`. The CUHK-X dataset is not part of this repository and may not be
redistributed; any work that uses it must cite the CUHK-X paper and acknowledge
the AIoT Lab, Department of Information Engineering, The Chinese University of
Hong Kong, as its creator (see [docs/compliance-and-licensing.md](docs/compliance-and-licensing.md)).
To cite this repository, use `CITATION.cff`.

## Reading the code

Comments in `src/`, `tests/` and the shell scripts cite the team's internal
decision ledger by row: a letter followed by a number, or a section sign with a
number. Those rows record why a design choice was made and what was measured when
it was made. The ledger is not part of this repository; the conclusions that
matter are carried by the documents in `docs/`.
