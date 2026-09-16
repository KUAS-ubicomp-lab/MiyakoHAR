# Compliance and licensing

## Our code and weights

Everything the team wrote in this repository, the training code, the inference
code, the packaging tools and the documentation, is licensed under the
**Apache License 2.0** (`LICENSE`). The fine-tuned weights attached to the release
are ours under the same licence; the pretrained weights they start from keep the
licences of their publishers, identified below and in `NOTICE`.

## Third-party components

`NOTICE` is the authoritative list: every dependency and every pretrained weight is
recorded there with its licence, its source and the date it was verified, and the
entries were written when each component was first used. In summary:

| Component | Where it enters | Licence |
|---|---|---|
| PyTorch 2.13.0 | runtime | BSD-3-Clause |
| TorchVision 0.28.0 | runtime | BSD-3-Clause |
| NumPy 2.4.4 | runtime | BSD-3-Clause |
| Pillow 12.2.0 | runtime | MIT-CMU |
| ir-CSN-R50, IG-65M pretrained, Kinetics-400 fine-tuned (`ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-44395bae.pth`) | initialisation of the depth + infrared member | Apache-2.0 (mmaction2 model zoo, download.openmmlab.com) |
| ir-CSN-152, IG-65M pretrained, Kinetics-400 fine-tuned (`ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-7d1dacde.pth`) | initialisation of the thermal member | Apache-2.0 (mmaction2 model zoo) |
| The ir-CSN architecture definition | reimplemented in `src/csn.py` from mmaction2's source, so the published state dict loads with strict key matching; no mmaction2 package is imported | Apache-2.0 retained, attribution in the file header |
| PyYAML 6.0.3 | training only (`splits/folds.yaml`); the inference path is asserted YAML-free | MIT |
| SciPy | the person-crop development path only; not used by any released model | BSD-3-Clause |
| VideoMAE V2 ViT structure and weights | a development-only probe and teacher path (`src/vmae.py`); not part of either released model | Apache-2.0 (mmaction2 re-host); upstream MIT |

Components with GPL or AGPL licences were excluded by rule (the public baseline
notebook's YOLO detector among them), because they could not be relicensed under
the Apache-2.0 grant the competition requires of winning entries.

## The dataset

The CUHK-X dataset is the property of the AIoT Lab, Department of Information
Engineering, The Chinese University of Hong Kong, and is licensed under the
**CUHK-X License Version 2.0**: non-commercial research use only, no redistribution
in any form, in whole or in part. The owner's derived annotations, splits and
metadata are under CC BY-NC 4.0. Nothing from the dataset is in this repository
([data.md](data.md)); `.gitignore` excludes every corpus path and image extension.
The two submitted CSVs and the fold split under version control are our own
derivations (our predictions per test clip, and our assignment of subject numbers
to folds) and contain no dataset content.

The licence's attribution clause requires that any publication, presentation,
model or other work that uses the data cite the CUHK-X paper and acknowledge the
AIoT Lab, Department of Information Engineering, CUHK, as the creator of the
dataset. We do so here and ask that downstream work do the same:

> This work uses the CUHK-X dataset, created by the AIoT Lab, Department of
> Information Engineering, The Chinese University of Hong Kong.

```bibtex
@inproceedings{cuhkx2026,
  author    = {Siyang Jiang and Mu Yuan and Xiang Ji and Bufang Yang and Zeyu Liu and Lilin Xu and
               Yang Li and Yuting He and Liran Dong and Wenrui Lu and Zhenyu Yan and Xiaofan Jiang and
               Wei Gao and Hongkai Chen and Guoliang Xing},
  title     = {A Large-Scale Multimodal Dataset and Benchmarks for Human Activity Scene Understanding and Reasoning},
  booktitle = {Proceedings of the 24th Annual International Conference on Mobile Systems, Applications, and Services (MobiSys '26)},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  year      = {2026},
  pages     = {352--370},
  doi       = {10.1145/3745756.3809209},
  url       = {https://doi.org/10.1145/3745756.3809209}
}
```

## The competition's constraints, and how each is met

The statements below apply to both released models. Each is tested by code in
this repository where a test is possible.

| Constraint | How it is met | Evidence |
|---|---|---|
| Everything loaded at inference is one checkpoint of at most 100,000,000 bytes (the organisers confirmed the limit covers every model used at inference, preprocessing included) | One fp16 file holding both networks: 83,543,089 B (022) or 83,559,635 B (026). No preprocessing model is loaded. | `verify.sh` weighs the file on disk before every reproduction; the loader enforces a 95 MB budget of its own |
| Inference runs from the original test files with the complete preprocessing inside the entry script | `inference.sh` decodes, samples, resizes and scores from the frames; no precomputed artefact is read | `tools/freshclone_rehearsal.sh`: a fresh clone plus one file reproduces the submitted CSV, 405 of 405 rows |
| No network access at inference | Nothing is downloaded; the pretrained weights are inside the packaged file | `tests/test_inference_offline.sh` runs the entry point in a network namespace with the loopback interface down |
| No test labels; no test data beyond the competition's own distribution; no pseudo-labels in the finals | Both released members were trained on the 18 training subjects only. One earlier submission (023) used pseudo-labels and is not a final | `train.sh` has no pseudo-label stage; `tests/check_ckpt_meta.py` asserts it |
| Nothing derived from file names, timestamps or metadata enters a model | Labels come from the directory path of training clips only; at inference the frame index in a file name is used only to pair depth and infrared frames and to order them | code review; the technical description states the scope |
| No large language model; no closed-source API | The models are convolutional networks; no external call exists in the code | `tests/test_inference_offline.sh` |
| No large pretrained backbone (Small Model Track) | The backbones are small pretrained video CNNs (12.4 M and 29.0 M parameters), which the organisers confirmed in writing as permitted pretrained CNNs rather than prohibited large backbones. An 86-million-parameter ViT-B backbone was tried (025) and is not a final | `NOTICE` |
| Reproducibility from the submitted code | A fresh clone at the released commit plus the one file reproduces each submitted CSV byte for byte on CPU (405 of 405 rows) and a random 40-clip subset reproduces the full run's rows | `verify.sh`, `tests/test_inference_subset.sh`, `tools/freshclone_rehearsal.sh` |
| Third-party components clearly identified; open-source-compatible licences | `NOTICE`, maintained from the first commit; BSD, MIT and Apache-2.0 only in the inference path | `NOTICE`; `tests/check_model.py` asserts no mmaction2 or pytorchvideo import is reachable |
| Development tools | AI coding assistants were used to help write code during development, which the organisers permit for development; no such tool is part of the solution or called at inference | the technical description, Section 8 |

## Status of this repository

The repository is private to the team's laboratory organisation. Under the
competition's terms, public release of a solution is an obligation of finalist
teams (top 6 per track) and is not provided for other teams; the team retains
ownership of its code and weights and licenses them under Apache-2.0 within this
repository. The dataset terms above apply to anyone who works with the data
regardless of how the code is shared.
