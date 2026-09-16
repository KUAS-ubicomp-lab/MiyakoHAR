# Inference and deployment

This page is the contract of the released models: what goes in, what comes out,
how to call them from the shell and from Python, what they cost, how the model file
is structured, and how the code behaves when something is wrong. Every number was
measured on the machine named at the end of the page; the reproduction checks that
produced them are described under "Tests".

## Contract

- **Input.** A directory of clip folders. Each clip folder holds the frames of its
  streams in sub-directories named `Depth_Color/`, `IR/` and `Thermal/`
  (PNG, PNG and JPEG respectively; see [data.md](data.md) for the file names). A
  clip may lack a stream; it is then classified from the streams present.
- **Output.** A CSV with the header `path,prediction` and one row per clip: the
  clip's path as given in the row list, and an integer class id from 0 to 39
  ([data.md](data.md) lists the classes).
- **Model file.** One file at `checkpoints/model.pth` (the legacy name
  `checkpoints/model.pt` is also accepted; if both exist, `model.pth` is loaded and
  the other is ignored with a message). The file holds both networks in half
  precision with their metadata; nothing else is loaded.
- **Environment.** Python 3.12 with `torch==2.13.0`, `torchvision==0.28.0`,
  `numpy==2.4.4`, `Pillow==12.2.0` (`requirements.txt`). The inference path imports
  nothing else, which a test asserts by running the entry point with a poisoned
  `yaml` module on the path. A GPU is optional and changes no prediction.
- **No network, no state.** Inference needs no network access (tested in a network
  namespace with the loopback interface down and empty model caches) and carries no
  state between clips (a random subset of clips reproduces the full run's
  predictions for those clips exactly).

## Getting the model file

The two released models are attached to the GitHub release `v1.0.0` of this
repository. The repository is private, so the download needs an account with
access: either the GitHub CLI signed in (`gh auth login`) or a browser.

| Asset | Model | Bytes | SHA-256 |
|---|---|---|---|
| `model-026.pth` | 026, the default: private 0.78921, public 0.78606 | 83,559,635 | `cd27bc72c09f6e4173cd8cf1026e2f966eea1bb29ba7288d72e9ea4c3c255940` |
| `model-022.pth` | 022: private 0.76960, public 0.76616 | 83,543,089 | `218266cc3230e87f441d01abe657e4b2cf3c87c4294005082114c96c6ef071a9` |
| `SHA256SUMS` | the two checksums above, in `sha256sum -c` format | | |

```bash
mkdir -p checkpoints
gh release download v1.0.0 --repo KUAS-ubicomp-lab/MiyakoHAR --pattern model-026.pth --output checkpoints/model.pth
sha256sum checkpoints/model.pth
```

To use model 022 instead, download `model-022.pth` to the same path. The code
reads every setting it needs from the file (architecture, input size, frames per
clip, normalisation, thermal decode size), so switching models is only a file
swap; `verify.sh` then compares against that model's CSV under `submissions/`.

## Command line

```bash
./inference.sh <data_dir> [output.csv]        # default output: ./submission.csv
```

`inference.sh` selects the interpreter (the repository's `.venv` if present, else
`python3`), refuses to run if `torch` cannot be imported, and calls
`src/predict.py`, which does everything else. The predictor can also be called
directly:

```
python src/predict.py <data_dir> [--out OUT.csv] [--test-csv LIST.csv] [--device auto|cpu]
                      [--checkpoints PATH ...] [--max-mb 95] [--allow-oversize] [--allow-constant-fallback]
```

| Option | Meaning |
|---|---|
| `--out` | Output CSV path. Default `submission.csv`. |
| `--test-csv` | An explicit row list (see below). |
| `--device` | `auto` uses CUDA when visible and falls back to CPU; `cpu` forces the CPU. `CUDA_VISIBLE_DEVICES=""` has the same effect. |
| `--checkpoints` | Load these files or run directories instead of `checkpoints/model.pth`. For development. |
| `--max-mb`, `--allow-oversize` | The packaged-size budget the loader enforces (95 MB by default, under the competition's 100,000,000-byte limit) and the switch to measure a file over it. |
| `--allow-constant-fallback` | Only for measuring the constant-prediction floor. Never for real use; see "Failure behaviour". |

**The row list.** The output has one row per entry of a row list, in its order:

1. If `--test-csv` names a file, its `path` column is the list. Paths are clip
   directories relative to `<data_dir>` or to its parent; a trailing slash and a
   byte-order mark are tolerated.
2. Otherwise the predictor looks for `test.csv` or `sample_submission.csv` in
   `<data_dir>`, in `<data_dir>/test_file/` and in the corresponding places one
   level up, and uses the first it finds.
3. With no row list at all, it walks `<data_dir>` (and one level down) for
   directories named `SM_test_<digits>` and lists them in natural order. If there
   are none, it stops with a message rather than guessing.

For clips with arbitrary names, therefore, either pass `--test-csv` with a `path`
column or place a `test.csv` next to the clips. Both routes were rehearsed on
renamed copies of test clips and reproduce the shipped predictions clip for clip.

## Python API

`src/predict.py` imports its neighbours by bare module name, so `src/` goes on the
path. The ensemble is built once and used per clip:

```python
import sys
from pathlib import Path
sys.path.insert(0, "src")                       # from the repository root
from predict import Ensemble, SHIPPED_CONTAINER, predict_clip

ensemble = Ensemble([SHIPPED_CONTAINER], "auto")  # loads checkpoints/model.pth; "cpu" forces the CPU
label = predict_clip(Path("/data/small_model_track_test/SM_test_0001"), ensemble)
print(label)                                      # an int, 0..39
```

`Ensemble.predict(clip_dir)` is the same call without the fallback: it returns the
class id, or `None` when no stream of the clip could be decoded. `predict_clip`
maps that `None` to the fallback class 36. Building the ensemble takes a few
seconds; each subsequent clip costs the per-clip time in the table below. The
model objects are ordinary `torch.nn.Module`s in evaluation mode; a long-lived
process (a service, a batch worker) should build one `Ensemble` and reuse it. The
code was not designed for concurrent calls on one `Ensemble` from several threads;
use one ensemble per worker process.

## Cost

Measured on the verification machine (below) for the 405 competition test clips.

| | Model 022 | Model 026 |
|---|---|---|
| CPU time per clip, 8 threads | 1.91 s | 3.45 s |
| CPU wall time, 405 clips, through `verify.sh` | 823 s | 1,348 s |
| CPU wall time from a fresh clone, environment build included | 855 s | 1,379 s |
| GPU time per clip, RTX 3090 | 0.159 s | 0.221 s |
| GPU wall time, 405 clips | 65 s | about 90 s |
| Peak GPU memory | 908 MiB allocated | 1,679 MiB allocated |
| Peak CPU memory | 1.66 GB | 2.43 GB |
| Model file | 83,543,089 B | 83,559,635 B |

The time per clip is dominated by decoding the frames and running the two
networks on the sampled 16 + 32 frames, upright and mirrored. Model 026's thermal
network reads a 320 x 320 input, which is why it costs 1.8 times 022's CPU time.

## The model file

`checkpoints/model.pth` is a `torch.save` of a Python list with one dictionary per
member. Each member carries:

| Key | Content |
|---|---|
| `model` | the state dict, every tensor in fp16 |
| `branch` | `depthir` or `thermal`: the stream the member reads |
| `arch` | the architecture name (`ircsn_r50`, `ircsn_r152`) |
| `norm` | the normalisation key applied to the input |
| `T` | frames per clip sampled for this member (16 or 32) |
| `input_size` | the network input, for example `224x224` or `320x320` |
| `source_wh` | the decode size of the member's stream; absent means the default 160 x 120 |
| `run` | the training run the member came from |
| `crops`, `fusion`, `quant`, `depth_lut` | optional stamps for development variants; absent in both released models |

The loader (`src/predict.py`, class `Ensemble`) builds each member from these keys
and refuses inconsistent files: members of one branch that disagree on the decode
size, a malformed size, a missing input size on a packed member. It never guesses.
`tools/package.py` writes this format from training runs and weighs the result on
disk against the limit; `tests/check_ckpt_meta.py` asserts that every stamp is
read back the way it was written.

## Failure behaviour

The two failure modes are treated differently on purpose, because they cost
differently.

- **A clip fails.** A missing stream, a truncated or corrupt frame, a short clip, an
  odd resolution: the clip is classified from whatever decodes, and if nothing
  does, it receives the majority training class (36, Walk) and a line on the error
  stream names it. The run continues and the exit status stays 0 as long as a
  complete CSV was written. `tests/test_inference_robustness.sh` runs the real
  entry point on six such cases and requires a prediction for each.
- **The model cannot be loaded.** A missing or misplaced model file, an
  interpreter without `torch`, a file that is not a packed container: the entry
  point prints the cause and exits 1 without writing a CSV. It never falls back to
  a different model or to a constant prediction, because a full run of fallbacks
  would produce a file of the right length and header that scores the
  constant-class floor (0.10945) while looking like a working submission.

## Swapping or updating the model

Place the new file at `checkpoints/model.pth`, check its SHA-256 against
`SHA256SUMS`, and run `verify.sh` against the CSV that belongs to it:

```bash
./verify.sh <data_dir> submissions/026_hr320_person_pixels.csv     # for model-026.pth
./verify.sh <data_dir> submissions/022_r152_fence.csv              # for model-022.pth
```

`verify.sh` prints the file's checksum and size, weighs it against the limit,
re-runs `inference.sh` on the test clips and exits 0 only on a byte-identical CSV.
A model retrained with `train.sh` and packaged with `tools/package.py` is loaded the
same way; it will not reproduce the released CSVs bit for bit, because GPU training
is not bit-reproducible, and `verify.sh` will say so.

## Tests

| Test | What it proves | Needs |
|---|---|---|
| `tests/check_model.py` | the model builder: architectures load, the pretrained state dict matches strictly, the temporal module is not a silent no-op | nothing |
| `tests/check_swa.py` | the weight averaging is an average and preserves the frozen BatchNorm statistics | nothing |
| `tests/check_ckpt_meta.py` | every checkpoint stamp is written and read back consistently; `train.sh`'s specs equal the loader's shipping list | nothing (one section needs training runs and is skipped otherwise) |
| `tests/test_inference_robustness.sh` | the entry point survives six degenerate clips and imports no YAML | the model file |
| `tests/test_inference_offline.sh` | the entry point loads the model and writes a submission with no network route | the model file, `unshare -rn` |
| `tests/test_inference_subset.sh <data_dir> <csv> [n] [seed]` | a random subset of clips reproduces the full run's rows (statelessness) | the model file and the test clips |
| `tests/check_source_wh.py`, `tests/check_crop_tta.py` | the decode-size and crop stamps reach the decode bit for bit, and are inert when absent | model 022 in the slot and the test clips (`CUHKX_TEST_ROOT`) |
| `tests/check_manifest.py`, `tests/check_cache.py`, `tests/check_dataset.py` | the corpus index, the decoded cache and the data loader | the training data and cache ([training.md](training.md)) |

Run the data-free tests from the repository root with the CPU forced:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python tests/check_model.py
CUDA_VISIBLE_DEVICES="" .venv/bin/python tests/check_swa.py
CUDA_VISIBLE_DEVICES="" .venv/bin/python tests/check_ckpt_meta.py
bash tests/test_inference_robustness.sh
bash tests/test_inference_offline.sh
```

`tools/freshclone_rehearsal.sh <scratch_dir> <reference_csv>` runs the whole
committee procedure end to end (clone, environment from the pin, the one file,
`verify.sh` on CPU) and prints the wall-clock time; set `CUHKX_TEST_ROOT` to the
directory that contains `small_model_track_test/`. Both released models were
reproduced this way, 405 of 405 rows identical.

## The verification machine

Ubuntu 24.04, one NVIDIA RTX 3090 with 24 GB (driver 595.84), CPython 3.12.3,
torch 2.13.0 and torchvision 0.28.0 from the stock PyPI wheels (CUDA 13 build).
`environment-labpc.txt` is its full package list. CPU-only wheels reproduce every
prediction; only the timings differ. On a machine whose driver is CUDA 12.x,
install the same versions from the matching index
(`pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.13.0 torchvision==0.28.0`).
If `python3 -m venv` stops because `ensurepip` is missing, `apt install
python3-venv`, `virtualenv .venv` or `uv venv .venv` all work; `inference.sh`
prints the same three options when it refuses to run without `torch`.
