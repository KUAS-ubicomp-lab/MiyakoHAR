#!/usr/bin/env bash
#
# CUHK-X Small Model Track — required inference entry point.
#
#   ./inference.sh <data_dir> [output_csv]
#
# data_dir    directory holding the test clips. Either the directory that
#             contains small_model_track_test/, or that directory itself.
# output_csv  defaults to ./submission.csv
#
# Per organiser ruling R8, the COMPLETE preprocessing pipeline must live inside
# this script's call path, so that inference runs directly from the original
# test data with no manual step in between. Nothing may be precomputed by hand.
#
# This script must never crash on a missing, short, corrupt or odd-resolution
# clip. Per-clip failures fall back to a default class and are reported on
# stderr; the exit status stays 0 as long as a complete CSV was written.
#
# PER-CLIP FALLBACK IS CORRECT. WHOLE-RUN FALLBACK IS A SILENT ZERO.
# The two are not the same policy and the difference is disqualification.
# A per-clip fallback costs one clip (0.4975 pp of the public LB) and keeps 404
# real predictions. A whole-run fallback emits 405 copies of the majority class,
# scores the 0.10945 sanity floor, exits 0, and writes a CSV that is the right
# length with the right header -- so nothing downstream can tell it apart from a
# working submission. Verification is pure reproduction and the gate is a >10%
# accuracy gap, so that file is not a degraded submission: it is a disqualified
# one that looks fine. Everything below the per-clip level therefore HARD-FAILS.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <data_dir> [output_csv]" >&2
    exit 2
fi

DATA_DIR="$1"
OUT_CSV="${2:-submission.csv}"

# Prefer the project virtualenv so the committee gets the pinned interpreter.
# A bare python3 is still allowed -- the committee may install the requirements
# any way they like -- but it is a WARNING, not a silent substitution, and the
# torch gate below is what actually decides whether the run may proceed.
#
# The comment this replaced read "the current stage needs only the standard
# library, so a bare python3 is a legitimate fallback." That was true on day 1
# and stopped being true the moment predict.py imported torch (2026-08-15).
# A comment that licenses a fallback has to be re-read every time the thing it
# describes changes; this one was not, and it is defect (a) of §J-DQ.
if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
    PYTHON="${REPO_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
    echo "[inference] WARNING: no ${REPO_DIR}/.venv — using ${PYTHON}" >&2
else
    echo "[inference] error: no python3 found" >&2
    exit 1
fi

# THE GATE. An interpreter that cannot import torch cannot run the model, and
# the only thing it can produce is the constant. Fail here, loudly, while the
# cause is still legible -- not 405 rows later inside a valid-looking CSV.
if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
    {
        echo "[inference] FATAL: ${PYTHON} cannot import torch."
        echo "[inference]    Refusing to run: without torch this script can only emit a"
        echo "[inference]    constant class, which scores the 0.10945 sanity floor and fails"
        echo "[inference]    reproduction by >10%. That is a disqualification, not a fallback."
        echo "[inference]    Fix: create the venv and install the pinned requirements —"
        echo "[inference]      python3 -m venv ${REPO_DIR}/.venv"
        echo "[inference]      ${REPO_DIR}/.venv/bin/pip install -r ${REPO_DIR}/requirements.txt"
        echo "[inference]"
        echo "[inference]    If that first line fails with 'ensurepip is not available', the"
        echo "[inference]    interpreter has no venv support and NONE of this is our code's"
        echo "[inference]    fault -- it is a packaging split. Any ONE of these works:"
        echo "[inference]      apt install python3-venv     # Debian/Ubuntu, needs root"
        echo "[inference]      pip install virtualenv && virtualenv ${REPO_DIR}/.venv"
        echo "[inference]      uv venv ${REPO_DIR}/.venv"
        echo "[inference]    Numbers in RESULTS.md were produced on CPython 3.12.13."
    } >&2
    exit 1
fi

echo "[inference] python:   ${PYTHON}" >&2
echo "[inference] data_dir: ${DATA_DIR}" >&2
echo "[inference] output:   ${OUT_CSV}" >&2

# predict.py hard-fails on an unloadable ensemble by default (same policy, one
# level down). Per-clip failures inside it stay non-fatal.
"${PYTHON}" "${REPO_DIR}/src/predict.py" "${DATA_DIR}" --out "${OUT_CSV}"

echo "[inference] done" >&2
