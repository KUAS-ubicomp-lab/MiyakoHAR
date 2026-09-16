#!/usr/bin/env bash
# Fresh-clone rehearsal: does a clean clone of this repository, a virtual
# environment built from the pinned requirements.txt and the ONE packaged
# checkpoint file reproduce a submitted CSV byte for byte on the CPU, and how
# long does that take? This is the procedure a verification committee follows,
# run end to end on purpose, with the GPU hidden so the CPU time is measured.
#
#   CUHKX_TEST_ROOT=<dir containing small_model_track_test/> \
#   tools/freshclone_rehearsal.sh <scratch_dir> <reference_csv>
#
# CUHKX_SRC (optional) names the repository to clone; it defaults to the
# repository this script lives in. The checkpoint is copied from
# <CUHKX_SRC>/checkpoints/model.pth.

set -uo pipefail
SC="${1:?usage: freshclone_rehearsal.sh <scratch_dir> <reference_csv>}"
REF="${2:?usage: freshclone_rehearsal.sh <scratch_dir> <reference_csv>}"
# ABSOLUTISE BEFORE ANY cd. Step 4 runs verify.sh from INSIDE the clone, so a
# relative reference path silently re-resolves there — where the CSV does not
# exist — and the run reports "no such CSV / exit 2 / 0 min", which reads exactly
# like the artefact failing. It is not; it is the harness losing the file.
REF="$(cd "$(dirname "${REF}")" && pwd)/$(basename "${REF}")"
SRC="${CUHKX_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CLONE="${SC}/rehearsal"
DATA="${CUHKX_TEST_ROOT:?set CUHKX_TEST_ROOT to the directory that contains small_model_track_test/}"

rm -r "${CLONE}" 2>/dev/null
echo "=== 1 · clone (no runs/, no cache/, no .venv — .gitignore is a licence constraint) ==="
git clone -q "${SRC}" "${CLONE}" || exit 1

echo "=== 2 · build the interpreter from the PINNED requirements.txt ==="
# The obvious `python3 -m venv` FAILS on this box: the system CPython 3.13.5
# has no ensurepip, and there is no python3.12 and no virtualenv on PATH. That
# is a real finding about the guidance we ship (inference.sh told the reader to
# run exactly that command) and it is fixed there. For the rehearsal itself the
# question is whether requirements.txt reconstructs the DEPENDENCIES, so it
# bootstraps from the interpreter the numbers were measured on -- CPython
# 3.12.13 -- which is stated in requirements.txt rather than assumed here.
BOOTSTRAP=python3
python3 -c "import ensurepip" 2>/dev/null || BOOTSTRAP="${SRC}/.venv/bin/python"
echo "  bootstrapping with: ${BOOTSTRAP} ($(${BOOTSTRAP} --version 2>&1))"
"${BOOTSTRAP}" -m venv "${CLONE}/.venv" || exit 1
"${CLONE}/.venv/bin/pip" install -q --upgrade pip
# PyYAML is training-only and requirements.txt says so; inference must not need it.
"${CLONE}/.venv/bin/pip" install -r "${CLONE}/requirements.txt" 2>&1 | tail -5
"${CLONE}/.venv/bin/python" -c "import torch, torchvision, numpy, PIL; print(f'  torch {torch.__version__} · torchvision {torchvision.__version__} · numpy {numpy.__version__}')" || exit 1
echo -n "  PyYAML present in the fresh venv? "
"${CLONE}/.venv/bin/python" -c "import yaml" 2>/dev/null && echo "YES (unexpected)" || echo "no — correct, inference must not need it"

echo "=== 3 · place ONLY the R4 artefact — ONE file, which is what a committee gets ==="
# CHANGED 2026-08-20 (B1). This step used to place predict.py's SHIPPING run
# checkpoints, i.e. it rehearsed the FALLBACK path rather than the artefact.
# R4 requires everything loaded at inference to ship as ONE checkpoint file, and
# since B1 that file is `checkpoints/model.pt` — so rehearsing the run dirs would
# prove a path the committee never takes. Placing the container instead makes the
# claim strictly stronger: a clean clone of HEAD plus ONE file reproduces the
# submission, with no runs/ directory in existence.
cd "${CLONE}" || exit 1
# P0 (a), 2026-08-31: the committee is told to look for `checkpoints/model.pth`, so
# that is the name placed in the clone WHATEVER the source slot is called (`.pth`
# preferred, the legacy `model.pt` accepted) — the rehearsal exercises the deliverable name.
SRC_CONTAINER="checkpoints/model.pth"
[[ -f "${SRC}/${SRC_CONTAINER}" ]] || { [[ -f "${SRC}/checkpoints/model.pt" ]] && SRC_CONTAINER="checkpoints/model.pt"; }
CONTAINER="checkpoints/model.pth"
if [[ ! -f "${SRC}/${SRC_CONTAINER}" ]]; then
    echo "  no ${CONTAINER} (nor checkpoints/model.pt) in ${SRC} — build it first:"
    echo "     tools/package.py --shipping --out ${CONTAINER}"
    exit 1
fi
mkdir -p "${CLONE}/checkpoints"
cp "${SRC}/${SRC_CONTAINER}" "${CLONE}/${CONTAINER}" || exit 1
echo "  placed ${CONTAINER} from ${SRC_CONTAINER}  ($(stat -c%s "${CLONE}/${CONTAINER}") B, sha $(sha256sum "${CLONE}/${CONTAINER}" | cut -c1-16))"
# (2026-09-09, F2 / m362's finding: a clone carries seven force-tracked audit files under runs/ but NO run directory and NO checkpoint —
#  the question is whether any RUN CHECKPOINT exists, which is what the fallback path would load; the tracked logs are not that.)
echo "  run checkpoints are deliberately ABSENT (only the R4 container exists): $( ls -d "${CLONE}"/runs/*/swa.pt >/dev/null 2>&1 && echo "PRESENT" || echo "confirmed absent$( [[ -d "${CLONE}/runs" ]] && echo " (runs/ holds tracked audit logs only)" )" )"

echo "=== 4 · verify.sh ON CPU — the number finding 18 says does not exist ==="
start=$(date +%s)
CUDA_VISIBLE_DEVICES="" ./verify.sh "${DATA}" "${REF}"
rc=$?
mins=$(( ($(date +%s) - start) / 60 ))
echo
echo "=============================================================="
echo "  CPU inference wall clock: ${mins} min  (organisers' ceiling: 120 min)"
echo "  verify.sh exit: ${rc}   (0 = byte-identical to ${REF})"
echo "=============================================================="
exit ${rc}
