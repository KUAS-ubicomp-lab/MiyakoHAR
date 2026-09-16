#!/usr/bin/env bash
#
# T-L7 — SUBSET IDENTITY: the artefact is non-transductive.
#
# Run the real entry point on a random N-clip subset of the test set and assert
# that every one of the N predictions is IDENTICAL, clip for clip, to the
# prediction the same artefact gave that clip inside the full 405-clip run
# (the submitted CSV that verify.sh reproduces). A pipeline whose prediction
# for a clip depends on WHICH OTHER clips are in the run — a per-split scaler,
# a test-time BN statistic, self-training, a batch-composition effect — fails
# here; a pipeline that treats every clip alone passes. This is the property
# the organisers' ">10 % reproduction gap" rule tests, and it is a Stage-2
# rehearsal in itself (a committee may verify on a subset).
#
# Pre-registration: GOAL-SPRINT-3 §4 P0 (b) / §7 (test T-L7); round1-C §3.2.
# It must be able to fail: a reference CSV with ONE prediction altered must turn
# it red (mutation-tested on landing, recorded in the ledger row).
#
# Usage:
#   tests/test_inference_subset.sh <test_root> <reference_csv> [n=40] [seed=20260831]
#     test_root      the directory that contains small_model_track_test/
#                    (this box: ~/cuhk-x/test_extracted)
#     reference_csv  the CSV the FULL run of this artefact produced — the object
#                    verify.sh reproduces byte-identically (e.g. submissions/022_r152_fence.csv)
#   CUDA_VISIBLE_DEVICES="" tests/test_inference_subset.sh ...   # the committee's CPU path

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${1:?usage: test_inference_subset.sh <test_root> <reference_csv> [n] [seed]}"
REF="${2:?usage: test_inference_subset.sh <test_root> <reference_csv> [n] [seed]}"
N="${3:-40}"
SEED="${4:-20260831}"

if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then PYTHON="${REPO_DIR}/.venv/bin/python"; else PYTHON="$(command -v python3)"; fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
fail() { echo "FAIL: $1"; [[ -f ${WORK}/stderr.log ]] && { echo "--- stderr (tail) ---"; tail -20 "${WORK}/stderr.log"; }; exit 1; }

CLIPS_SRC="${ROOT}/small_model_track_test"
[[ -d "${CLIPS_SRC}" ]] || fail "no ${CLIPS_SRC}"
[[ -f "${REF}" ]]       || fail "no reference CSV ${REF}"

# ── 1 · the subset: N row keys drawn from the REFERENCE's own key set, seeded ──
# Drawn from the reference (not from the directory) so the subset is defined on
# the graded key set, and each chosen key's reference prediction travels with it.
CLIPS="${WORK}/small_model_track_test"
mkdir -p "${CLIPS}"
"${PYTHON}" - "${REF}" "${N}" "${SEED}" "${WORK}" <<'PY'
import csv, random, sys
from pathlib import Path
ref, n, seed, work = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
with ref.open(newline="", encoding="utf-8-sig") as fh:
    rows = [(r["path"], r["prediction"]) for r in csv.DictReader(fh) if r.get("path")]
if len(rows) < n:
    raise SystemExit(f"reference has {len(rows)} rows < n={n}")
chosen = sorted(random.Random(seed).sample(range(len(rows)), n))
sub = [rows[i] for i in chosen]
with (work / "test.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["path", "prediction"])
    w.writerows((k, "") for k, _ in sub)
with (work / "expected.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["path", "prediction"])
    w.writerows(sub)
print(f"[subset] {n} of {len(rows)} reference rows, seed {seed}: " + " ".join(k.rstrip('/').rsplit('/', 1)[-1] for k, _ in sub[:5]) + " …")
PY
[[ $? -eq 0 ]] || fail "could not draw the subset"

# Symlink ONLY the chosen clip directories into a fresh tree — the other clips
# do not exist in this run, which is the whole point.
while IFS=, read -r key _; do
    leaf="$(basename "${key}")"
    [[ -d "${CLIPS_SRC}/${leaf}" ]] || fail "reference key ${key} has no directory under ${CLIPS_SRC}"
    ln -s "${CLIPS_SRC}/${leaf}" "${CLIPS}/${leaf}"
done < <(tail -n +2 "${WORK}/expected.csv")
placed=$(find "${CLIPS}" -maxdepth 1 -type l | wc -l)
[[ ${placed} -eq ${N} ]] || fail "placed ${placed} clip links, expected ${N}"

# ── 2 · the real entry point on the subset ────────────────────────────────────
OUT="${WORK}/submission.csv"
start=$(date +%s)
"${REPO_DIR}/inference.sh" "${WORK}" "${OUT}" >/dev/null 2>"${WORK}/stderr.log"
rc=$?
secs=$(( $(date +%s) - start ))
[[ ${rc} -eq 0 ]] || fail "inference.sh exit ${rc} on the subset"
[[ -f ${OUT} ]]   || fail "no submission written"
grep -q "\[predict\] ensemble: " "${WORK}/stderr.log" || fail "no ensemble line — the model did not load; the comparison would be vacuous"
grep -qi "FATAL" "${WORK}/stderr.log" && fail "predict.py reported FATAL"
grep -q "row keys from ${WORK}/test.csv" "${WORK}/stderr.log" || fail "the run did not take its row keys from the subset test.csv"
rows=$(( $(wc -l < "${OUT}") - 1 ))
[[ ${rows} -eq ${N} ]] || fail "wrote ${rows} rows, expected ${N}"

# ── 3 · clip-for-clip identity against the full-run reference ─────────────────
mismatch=$("${PYTHON}" - "${WORK}/expected.csv" "${OUT}" <<'PY'
import csv, sys
def load(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return {r["path"]: r["prediction"] for r in csv.DictReader(fh)}
exp, got = load(sys.argv[1]), load(sys.argv[2])
missing = [k for k in exp if k not in got]
diff = [(k, exp[k], got[k]) for k in exp if k in got and exp[k] != got[k]]
for k in missing: print(f"MISSING {k}")
for k, e, g in diff: print(f"DIFF {k} full-run={e} subset-run={g}")
print(f"N_BAD={len(missing) + len(diff)}")
PY
)
n_bad=$(echo "${mismatch}" | sed -n 's/^N_BAD=//p')
if [[ "${n_bad}" != "0" ]]; then
    echo "${mismatch}" | grep -v '^N_BAD='
    fail "${n_bad} of ${N} subset predictions differ from the full-run reference — the pipeline is NOT clip-independent"
fi

dev=$(sed -n 's/.*checkpoints over .* on \([a-z]*\)[, ].*/\1/p' "${WORK}/stderr.log" | head -1)
echo "PASS: T-L7 subset identity — ${N}/${N} predictions on a ${N}-clip subset (seed ${SEED}) are identical to the full-run reference's, clip for clip; device ${dev:-?}, ${secs} s"
