#!/usr/bin/env bash
#
# MiyakoHAR: training entry point.
#
#   ./train.sh <train_root> <class_mapping.csv>
#
# train_root         the HAR/data directory of the CUHK-X Small Model Track training
#                    corpus, holding <modality>/<action>/<user>/<take>/ (see docs/data.md)
# class_mapping.csv  the organisers' action_id,action_name table, shipped with the data
#
# Rebuilds, from the original corpus, the two trained members that inference.sh
# loads: src/predict.py's SHIPPING list, and nothing else. tests/check_ckpt_meta.py
# asserts that the specs below and that list agree, so the two cannot drift apart.
#
# THE CACHE IS BUILT FIRST. Steps 1 to 3 index the corpus and decode it into uint8
# memmap shards under cache/. Training reads the shards, never the raw frames. The
# decode is a one-off cost of about 35 minutes of CPU over 468,033 files; skip it
# only if cache/ is already populated from the same corpus.
#
# Then one training run per member, 40 epochs each, on one GPU: about 3 GPU-hours
# for the ir-CSN-152 thermal member on an RTX 3090 with 24 GB. Training is seeded
# but, like all GPU training, not bit-reproducible across hardware and library
# versions; inference from the packaged weights is (see verify.sh). CPU-only wheels
# reproduce every accuracy number; they do not reproduce timings.
#
# --workers 3 rather than train.py's default of 6: at T=16 each worker holds two
# clips' worth of frames, and six workers exhausted a 10 GB machine. With more RAM
# the count may be raised; the trained weights do not depend on it.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USAGE="usage: ./train.sh <train_root (the HAR/data directory)> <class_mapping.csv>"
DATA_DIR="${1:?${USAGE}}"
CLASS_MAP="${2:?${USAGE}}"

# Same interpreter policy as inference.sh: prefer the pinned venv, warn on a
# bare python3, and refuse outright without torch. A training run that cannot
# import torch has nothing to fall back to.
if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
    PYTHON="${REPO_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
    echo "[train] WARNING: no ${REPO_DIR}/.venv -- using ${PYTHON}" >&2
else
    echo "[train] error: no python3 found" >&2
    exit 1
fi

if ! "${PYTHON}" -c "import torch, yaml" >/dev/null 2>&1; then
    {
        echo "[train] FATAL: ${PYTHON} cannot import torch and PyYAML."
        echo "[train]    The training path reads splits/folds.yaml, so it needs both."
        echo "[train]    Fix: python3 -m venv ${REPO_DIR}/.venv"
        echo "[train]      ${REPO_DIR}/.venv/bin/pip install -r ${REPO_DIR}/requirements.txt PyYAML==6.0.3"
        echo "[train]    If the first line fails with 'ensurepip is not available', use"
        echo "[train]    'apt install python3-venv', or virtualenv, or 'uv venv'. See"
        echo "[train]    inference.sh for the same note. Built on CPython 3.12."
    } >&2
    exit 1
fi

[[ -d "${DATA_DIR}" ]] || { echo "[train] error: no such train_root: ${DATA_DIR}" >&2; exit 2; }
[[ -f "${CLASS_MAP}" ]] || { echo "[train] error: no such class map: ${CLASS_MAP}" >&2; exit 2; }
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
CLASS_MAP="$(cd "$(dirname "${CLASS_MAP}")" && pwd)/$(basename "${CLASS_MAP}")"

cd "${REPO_DIR}"
echo "[train] python:     ${PYTHON}" >&2
echo "[train] train_root: ${DATA_DIR}" >&2
echo "[train] class map:  ${CLASS_MAP}" >&2

# 1. Index the corpus: one row per clip with its label, its subject and the
#    presence of each stream. (A warning that no test root was found is expected
#    here; the test clips are not needed for training.)
"${PYTHON}" src/manifest.py --train-root "${DATA_DIR}" --class-map "${CLASS_MAP}" --out /tmp/m_fresh.csv

# 2. Decode the training clips into memmap shards: the thermal stream and the
#    four-channel depth + infrared stack, both at 160 x 120, the sizes the two
#    shipped members were trained on.
"${PYTHON}" -m src.preprocess --manifest /tmp/m_fresh.csv --out cache/ --splits train

# 3. Derive the training index from the cache. Training reads this file, not the
#    manifest of step 1.
"${PYTHON}" tools/derive_index_from_cache.py --cache cache/ --class-map "${CLASS_MAP}" --out /tmp/m_derived.csv

# 4. The two shipped members. Each spec states the full run name, the
#    architecture, the input size, the seed, the learning rate, the batch size,
#    the machine tag carried in the run name, the frames per clip T and any extra
#    flags, so that the name written to disk cannot disagree with the recipe that
#    produced it. Every member trains on all 18 training subjects
#    (--all-subjects). With no held-out subjects there is no validation set, so
#    the checkpoint is a weight average over the run rather than a best epoch:
#    SWA over the last quarter of the iterations for the depth + infrared member;
#    SWAD, a dense average over epochs 5 to 39 at a constant learning rate, for
#    the thermal member, which also trains under the temporal-crop augmentation.
#    The learning rate 5e-4 and the batch size 12 are the measured optimum for
#    this backbone family and are part of the recipe: a different batch size does
#    not reproduce the shipped weights.
#    Fields: branch fold suffix arch res seed lr bs [machine=ALIEN] [T=16] [extra...]
STAGE2=("depthir all18 q16_ircsn224_s1 ircsn_r50 224x224 20260813 0.0005 12 ALIEN 16"
        "thermal all18 r152stack_s1 ircsn_r152 224x224 20260813 0.0005 12 LABPC 32 --swad --swad-window 5,39 --aug-tcrop")

for spec in "${STAGE2[@]}"; do
    read -r branch fold suffix arch res seed lr bs machine T extra <<<"${spec}"
    machine="${machine:-ALIEN}"; T="${T:-16}"; read -ra EXTRA <<<"${extra:-}"
    run="${branch}_${fold}_${machine,,}_${suffix}"
    if [[ -f "runs/${run}/swa.pt" ]]; then
        echo "[train] skip ${run} (already complete)" >&2
        continue
    fi
    echo "[train] === ${run}  (${arch} @ ${res}, lr ${lr}, batch ${bs}, T ${T}; all 18 subjects, no pseudo-labels) ===" >&2
    "${PYTHON}" -m src.train \
        --branch "${branch}" --fold 0 --machine "${machine}" --all-subjects \
        --manifest /tmp/m_derived.csv \
        --no-cudnn-benchmark --workers 3 \
        --T "${T}" --lr "${lr}" --batch-size "${bs}" --arch "${arch}" --input-size "${res}" \
        "${EXTRA[@]}" \
        --run-name "${run}" --seed "${seed}"
done

echo "[train] done: ${#STAGE2[@]} trained members under runs/." >&2
echo "[train] Package them into the single inference file with:" >&2
echo "[train]   ${PYTHON} tools/package.py --shipping --out checkpoints/model.pth" >&2
