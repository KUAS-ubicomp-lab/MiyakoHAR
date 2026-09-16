#!/usr/bin/env bash
#
# inference.sh must run with NO NETWORK and a COLD weight cache.
#
# §J-DQ (b), 2026-08-15. src/model.py defaulted `pretrained=True` until this
# landed, so any caller that did not say otherwise reached
# download.pytorch.org for 45 MB of ImageNet weights -- inside the inference
# path, on a machine the committee controls and we do not. Locally this is
# invisible: ~/.cache/torch is warm on every box that has ever trained, so the
# download is skipped and the bug cannot be observed. DECISIONS §3 also records
# that the HF_HUB_OFFLINE=1 guard was deliberately deleted.
#
# The failure mode is not a slow run. It is a run that raises inside Ensemble()
# on a machine with no route, which -- before §J-DQ (a) -- degraded the whole
# submission to a constant and exited 0.
#
# THIS IS THE TEST TECHNICAL.md'S HONOUR DECLARATION ALREADY CLAIMED EXISTED.
# That file states "it runs with networking disabled". That was an assertion
# about the code; this is an execution of it.
#
# METHOD: a real network namespace, not an env var. `unshare -rn` gives the
# process a fresh netns whose only interface is a DOWN loopback, so a socket to
# download.pytorch.org fails at connect() regardless of what any library does
# with proxy settings. TORCH_HOME and XDG_CACHE_HOME point at empty directories,
# so nothing can be served from cache either -- the two together are what make
# "no download happened" provable rather than merely likely.
#
# Usage: tests/test_inference_offline.sh
#
# REQUIRES runs/ (the trained checkpoints, which are gitignored) and
# `unshare -rn`. Both are checked below and reported as SKIP, not PASS -- a test
# that cannot run must never be counted as a test that passed.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

fail() { echo "FAIL: $1"; [[ -f ${WORK}/stderr.log ]] && { echo "--- stderr ---"; cat "${WORK}/stderr.log"; }; exit 1; }
skip() { echo "SKIP: $1"; exit 0; }

# ── preconditions, reported honestly ────────────────────────────────────────
unshare -rn true 2>/dev/null || skip "unshare -rn unavailable — cannot build a network namespace"
# B1 / LABPC (2026-08-26): the object under test is the R4 container when it exists — a box that
# holds ONLY `checkpoints/model.pth` (or the legacy `model.pt`; P0 a) (the committee's configuration, and this box's, whose run
# names are labpc_*) must not SKIP here. SKIP is not a pass.
{ [[ -f "${REPO_DIR}/checkpoints/model.pth" ]] || [[ -f "${REPO_DIR}/checkpoints/model.pt" ]] || ls -d "${REPO_DIR}"/runs/depthir_f0_*/swa.pt >/dev/null 2>&1; } \
    || skip "no checkpoints/model.pth (nor model.pt) and no trained checkpoints under runs/ — nothing to prove offline loading with"

# ── a self-contained fixture: no corpus, no cache ───────────────────────────
# The clips are degenerate on purpose. This test asserts that the MODEL LOADS
# with no network; per-clip accuracy is tests/test_inference_robustness.sh's job
# and the two must not be conflated.
CLIPS="${WORK}/small_model_track_test"
mkdir -p "${CLIPS}"/SM_test_0001/IR
printf '\x89PNG\r\n\x1a\n' > "${CLIPS}/SM_test_0001/IR/IR_x_0.png"
printf 'path,prediction\nsmall_model_track_test/SM_test_0001/,\n' > "${WORK}/test.csv"

# ── the cold cache ──────────────────────────────────────────────────────────
COLD="${WORK}/torch_home"
XDG="${WORK}/xdg_cache"
mkdir -p "${COLD}" "${XDG}"

OUT="${WORK}/submission.csv"

# The whole point: no route to anywhere. `unshare -rn` drops us into a netns
# with loopback DOWN, so even a resolver lookup fails.
unshare -rn env \
    TORCH_HOME="${COLD}" \
    XDG_CACHE_HOME="${XDG}" \
    HOME="${WORK}" \
    "${REPO_DIR}/inference.sh" "${WORK}" "${OUT}" \
    >/dev/null 2>"${WORK}/stderr.log"
rc=$?

# ── assertions ──────────────────────────────────────────────────────────────
[[ ${rc} -eq 0 ]] || fail "exit code ${rc} with no network — inference.sh must not need one"
[[ -f ${OUT} ]]   || fail "no submission written"

# The load actually happened. Without this the test passes on a run that fell
# back to the constant, which is precisely the outcome §J-DQ (a) is about.
grep -q "\[predict\] ensemble: " "${WORK}/stderr.log" \
    || fail "no ensemble line on stderr — the model did not load offline"
grep -qi "FATAL" "${WORK}/stderr.log" \
    && fail "predict.py reported FATAL with no network"

# Nothing was fetched. An empty TORCH_HOME that is still empty afterwards is the
# direct evidence; a non-empty one means a download path is still live.
downloaded=$(find "${COLD}" "${XDG}" -type f 2>/dev/null | wc -l)
[[ ${downloaded} -eq 0 ]] \
    || fail "${downloaded} file(s) appeared in the cold weight cache — something downloaded:
$(find "${COLD}" "${XDG}" -type f)"

n=$(grep -c "\[predict\]   " "${WORK}/stderr.log")
echo "PASS: inference.sh loaded ${n} checkpoint(s) and wrote a submission with no network route and an empty TORCH_HOME"
