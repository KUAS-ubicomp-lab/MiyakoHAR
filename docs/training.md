# Training

This page gives the recipe of the two released models, how to rebuild them from
the training corpus, how the recipe was selected, and what it cost. The numbers
come from the training logs and from the checkpoints themselves; every member
stores its architecture, input size, frames per clip and normalisation in its
metadata, and `tests/check_ckpt_meta.py` asserts that `train.sh` and the loader
agree on them.

## Requirements

- The CUHK-X Small Model Track training corpus (see [data.md](data.md)) and its
  `class_mapping.csv`. The data are not part of this repository.
- Python 3.12 with the pinned packages plus PyYAML, which the training path needs
  for `splits/folds.yaml`:

  ```bash
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt PyYAML==6.0.3
  ```

- One GPU with 24 GB of memory for the recipes as shipped (the ir-CSN-152 at
  batch 12 and 224 x 224 input, or at batch 6 and 320 x 320, both fit an RTX 3090
  with 24 GB; the 320 x 320 fit check measured 18.3 GB). About 20 GB of disk for
  the decoded cache at 160 x 120, and a further 46 GB (43 GiB) if the 320 x 240
  thermal cache of model 026 is built.

## Rebuilding model 022 with `train.sh`

```bash
./train.sh <HAR/data directory> <class_mapping.csv>
.venv/bin/python tools/package.py --shipping --out checkpoints/model.pth
```

`train.sh` runs four steps:

1. **Index the corpus.** `src/manifest.py` walks `HAR/data` and writes one row per
   clip with its label, its subject and the presence of each stream. (It also
   looks for a test root and warns if none is found; the test clips are not needed
   for training.)
2. **Decode into memmap shards.** `src/preprocess.py` decodes every training clip
   once into uint8 shards under `cache/`: the thermal stream and the four-channel
   depth + infrared stack, both at 160 x 120. This is the expensive step, about 35
   minutes of CPU over 468,033 files, and it is done once; training reads the
   shards, never the raw frames.
3. **Derive the training index from the cache** with
   `tools/derive_index_from_cache.py`. Training reads this index.
4. **Train the two members** with `python -m src.train`, one run each, both on all
   18 training subjects (`--all-subjects`). The specs are in the `STAGE2` array of
   `train.sh`; each one states the run name, the architecture, the input size, the
   seed, the learning rate, the batch size, the frames per clip and the extra
   flags, so that a checkpoint's name cannot disagree with the recipe that made it.

A finished run leaves `runs/<run name>/swa.pt`. `tools/package.py --shipping` reads
the two run names from `src/predict.py`'s `SHIPPING` list, stores every tensor in
half precision, stamps each member with its metadata, writes the single file and
weighs it against the packaging budget of 95 MB; `verify.sh` later weighs the same
file against the competition's 100,000,000-byte limit.

Training is seeded (seed 20260813) and cuDNN autotuning is disabled, but GPU
training is not bit-reproducible across hardware and library versions; expect a
retrained member to differ from the released one at the level of a few clips.
Inference from the released file is bit-reproducible, which is what `verify.sh`
checks.

## The recipe

| | Depth + infrared member | Thermal member (022) | Thermal member (026) |
|---|---|---|---|
| Run name | `depthir_all18_alien_q16_ircsn224_s1` | `thermal_all18_labpc_r152stack_s1` | `labpc_hr320_thermal_all18_r152stack_s1` |
| Architecture | ir-CSN-R50, 4-channel stem | ir-CSN-152 | ir-CSN-152 |
| Initialisation | IG-65M to Kinetics-400 (mmaction2) | same | same |
| Source frames decoded at | 160 x 120 | 160 x 120 | 320 x 240 |
| Input size | 224 x 224 | 224 x 224 | 320 x 320 |
| Frames per clip T | 16 | 32 | 32 |
| Batch size | 12 | 12 | 6 |
| Optimiser | SGD, momentum 0.9, Nesterov | same | same |
| Learning rate | 5 x 10^-4 | 5 x 10^-4 | 5 x 10^-4 |
| Schedule | 3-epoch linear warmup, cosine to 0 | 3-epoch warmup, then constant | 3-epoch warmup, then constant |
| Weight averaging | SWA over the last quarter of the iterations | SWAD, dense average over epochs 5 to 39 | SWAD, epochs 5 to 39 |
| Epochs | 40 | 40 | 40 |
| Weight decay | 5 x 10^-4 | 5 x 10^-4 | 5 x 10^-4 |
| Label smoothing | 0.1 | 0.1 | 0.1 |
| Gradient-norm clip | 20 | 20 | 20 |
| Temporal crop augmentation | no | yes | yes |
| BatchNorm | held at pretrained statistics except the stem | same | same |
| Training subjects | all 18 | all 18 | all 18 |
| Seed | 20260813 | 20260813 | 20260813 |
| Machine | Alienware workstation (RTX 5060 Laptop GPU, 8 GB) | RTX 3090 workstation | RTX 3090 workstation |

The learning rate 5 x 10^-4 is the measured optimum for this backbone family; the
inherited 5 x 10^-3 destabilised it on 6 of 6 validation cells. Batch 12 was
measured as worth +2.116 points over the inherited default of 16 on this family.
Because no subject is held out in the final runs, there is no validation set and
the checkpoint is a weight average over the run rather than a best epoch: a
stochastic weight average (Izmailov et al., UAI 2018) over the last quarter of the
iterations for the depth + infrared member, and SWAD (Cha et al., NeurIPS 2021), a
dense average at a constant learning rate over epochs 5 to 39, for the thermal
members. The thermal members also train under a temporal crop augmentation.
The depth + infrared member is the same trained file in both released models.

`train.py` accepts a `--machine` tag that is written into the run name and the
experiment log; the released names carry `alien` and `labpc`, the team's two
training machines, and `train.sh` passes those tags explicitly so that the names
match `SHIPPING` on any host.

## Model 026: the thermal member at native resolution

Model 026 differs from 022 in the thermal member only. Its member was trained from a
second thermal cache decoded at the sensor's native 320 x 240 (46.4 GB, 43.2 GiB), with a
320 x 320 input at batch 6; nothing else in the recipe changed. `train.sh` builds
model 022; to rebuild the 026 member, run steps 1 to 3 of `train.sh` and then:

```bash
# the high-resolution thermal cache (train split only, about 46 GB)
.venv/bin/python -m src.preprocess --manifest /tmp/m_fresh.csv --out cache_hr \
    --streams thermal --splits train --size-wh 320x240

# the thermal member: the 022 recipe with the 320x240 source and a 320x320 input at batch 6
.venv/bin/python -m src.train --branch thermal --fold 0 --machine LABPC --all-subjects \
    --manifest /tmp/m_derived.csv --cache cache_hr \
    --no-cudnn-benchmark --workers 3 \
    --T 32 --lr 0.0005 --batch-size 6 --arch ircsn_r152 --input-size 320x320 \
    --swad --swad-window 5,39 --aug-tcrop \
    --run-name labpc_hr320_thermal_all18_r152stack_s1 --seed 20260813

# package it beside the released depth + infrared member, stamping the decode size
.venv/bin/python tools/package.py \
    --checkpoints runs/depthir_all18_alien_q16_ircsn224_s1 runs/labpc_hr320_thermal_all18_r152stack_s1 \
    --source-wh thermal=320x240 --out checkpoints/model.pth
```

The `--source-wh` stamp is what tells the loader to decode the thermal stream at
320 x 240 for this member; a member without the stamp is decoded at 160 x 120
exactly as before ([architecture.md](architecture.md)).

## How the recipe was selected

Every change to the recipe was judged on a frozen 3-fold subject-grouped split of
the 18 training subjects (`splits/folds.yaml`), generated once from a recorded
seed and never changed. Each fold trains on 12 subjects and validates on the
other 6, so every subject is validated exactly once by members that never saw it:

| Fold | Validation subjects | Training subjects |
|---|---|---|
| 0 | 4, 5, 9, 21, 22, 24 | 1, 2, 3, 6, 7, 8, 16, 17, 18, 19, 20, 23 |
| 1 | 2, 6, 8, 17, 18, 19 | 1, 3, 4, 5, 7, 9, 16, 20, 21, 22, 23, 24 |
| 2 | 1, 3, 7, 16, 20, 23 | 2, 4, 5, 6, 8, 9, 17, 18, 19, 21, 22, 24 |

With two training seeds a recipe change is read as a paired difference over six
subject-disjoint cells. A change was adopted only if the fused two-branch model
improved by at least +0.5 points on all three folds at a first screen and then,
over all six cells, passed t > 2.571 on 5 degrees of freedom with at least 5 of the
6 cells in the direction of the effect. The number of such gated reads was capped in
advance at fourteen, and the cap was printed with every read. The 026 change was the
thirteenth and the first whose fused result cleared the gate on an instrument with
no exposure to test data:

| Recipe | Seed 1: fold 0 | fold 1 | fold 2 | Seed 2: fold 0 | fold 1 | fold 2 | Mean |
|---|---|---|---|---|---|---|---|
| 022 recipe | 73.975 | 72.693 | 71.834 | 74.898 | 72.227 | 70.719 | 72.724 |
| 026 recipe | 76.025 | 73.253 | 75.076 | 76.332 | 72.600 | 73.354 | 74.440 |
| Difference (points) | +2.049 | +0.559 | +3.242 | +1.434 | +0.373 | +2.634 | +1.715 |

Standard deviation of the difference 1.141, t = +3.68, 6 of 6 cells positive. The
instrument decides which recipe to ship; the members that ship are then retrained
on all 18 subjects (Section "The recipe"). The per-clip predictions of all twelve
cells are in `evaluation/fold_predictions_022_026.csv`
([held-out-predictions.md](held-out-predictions.md)).

## Compute

| Item | Cost |
|---|---|
| Decoding the training split at 160 x 120 | about 35 minutes of CPU, once |
| Six fold cells of the 026 thermal recipe (3 folds x 2 seeds) | 15.2 GPU-hours on one RTX 3090 |
| The 026 thermal member on all 18 subjects | 3.0 GPU-hours on one RTX 3090 |
| Peak GPU memory, ir-CSN-152 at 320 x 320, batch 6 | 18.3 GB |

The depth + infrared member was trained on a separate workstation (its package
list is `environment-alien.txt`); the thermal members on the RTX 3090 workstation
(`environment-labpc.txt`).

## Training-side checks

`tests/check_manifest.py <manifest.csv>` cross-checks the corpus index against
independent counts; `tests/check_cache.py --cache cache/ --manifest <manifest.csv>`
verifies the memmap shards against the corpus they came from (a truncated memmap
does not raise by itself); `tests/check_dataset.py --cache cache/ --manifest
<index.csv>` verifies the loader against the cache and the frozen folds, including
that no validation subject leaks into training. All three need the data and the
cache. `tests/check_model.py` and `tests/check_swa.py` test the model builder and
the weight averaging without data.
