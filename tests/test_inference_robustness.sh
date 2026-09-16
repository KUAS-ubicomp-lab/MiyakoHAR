#!/usr/bin/env bash
#
# inference.sh must never crash on a degenerate clip.
#
# This is a hard competition requirement, not a nicety: the script is run by the
# committee on a hidden set we cannot inspect, and a traceback there costs the
# whole submission. Every case below is drawn from a measured property of the
# real corpus, not invented:
#
#   1-frame clip        42 training clips have exactly one frame
#   empty clip          modality directories are not guaranteed
#   corrupt image       122 unreadable IR files, confined to 4 test clips
#   missing modality    10 named test clips ship no Thermal/ directory at all
#   CJK + parens name   61 non-canonical IMU filenames in train, 14 in test
#   row with no dir     defends the test.csv row list against a partial unzip
#
# Usage: tests/test_inference_robustness.sh

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

CLIPS="${WORK}/small_model_track_test"
mkdir -p "${CLIPS}"/SM_test_0001/IR \
         "${CLIPS}"/SM_test_0002 \
         "${CLIPS}"/SM_test_0003/IR \
         "${CLIPS}"/SM_test_0004/Thermal \
         "${CLIPS}"/SM_test_0005/IR

printf '\x89PNG\r\n\x1a\n'  > "${CLIPS}/SM_test_0001/IR/IR_x_0.png"          # 1 frame, header only
head -c 17 /dev/urandom     > "${CLIPS}/SM_test_0003/IR/IR_x_0.png"          # truncated / corrupt
printf 'not a jpeg'         > "${CLIPS}/SM_test_0004/Thermal/frame_000001.jpg"
printf '\x89PNG\r\n\x1a\n'  > "${CLIPS}/SM_test_0005/IR/IR_上(LA+C+RA)_0.png"

# Row 0006 is listed by the organisers but absent from disk.
printf 'path,prediction\n' > "${WORK}/test.csv"
for i in 0001 0002 0003 0004 0005 0006; do
    printf 'small_model_track_test/SM_test_%s/,\n' "${i}" >> "${WORK}/test.csv"
done

OUT="${WORK}/submission.csv"
"${REPO_DIR}/inference.sh" "${WORK}" "${OUT}" >/dev/null 2>"${WORK}/stderr.log"
rc=$?

fail() { echo "FAIL: $1"; echo "--- stderr ---"; cat "${WORK}/stderr.log"; exit 1; }

[[ ${rc} -eq 0 ]] || fail "exit code ${rc}, expected 0"
[[ -f ${OUT} ]]   || fail "no submission written"

rows=$(($(wc -l < "${OUT}") - 1))
[[ ${rows} -eq 6 ]] || fail "wrote ${rows} rows, expected 6"

header=$(head -1 "${OUT}")
[[ ${header} == "path,prediction" ]] || fail "bad header: ${header}"

# Every prediction must be an integer in [0, 39].
bad=$(tail -n +2 "${OUT}" | awk -F, '{ if ($2 !~ /^[0-9]+$/ || $2 < 0 || $2 > 39) print }')
[[ -z ${bad} ]] || fail "prediction out of range: ${bad}"

grep -q "no directory on disk" "${WORK}/stderr.log" \
    || fail "missing clip was not reported on stderr"

# ── the submission path's dependency surface ────────────────────────────────
# PyYAML was installed 2026-08-13 for TRAINING ONLY (dataset.py reads the frozen
# folds). The organisers run inference.sh in an environment we do not control,
# so every import it makes is a way for the submission to die on their machine
# rather than ours. This asserts the boundary instead of trusting a NOTICE line.
#
# Enforced by execution against the REAL entry point, not by grep. A poisoned
# yaml module is placed on PYTHONPATH, which precedes site-packages, so any
# `import yaml` anywhere inference.sh reaches raises instead of resolving.
mkdir -p "${WORK}/blockyaml"
cat > "${WORK}/blockyaml/yaml.py" <<'PYEOF'
raise ImportError("PyYAML must not be reachable from the inference path (see NOTICE)")
PYEOF

OUT2="${WORK}/sub_noyaml.csv"
PYTHONPATH="${WORK}/blockyaml" "${REPO_DIR}/inference.sh" "${WORK}" "${OUT2}" \
    >/dev/null 2>"${WORK}/noyaml.log" \
    || { echo "FAIL: the inference path needs PyYAML — it must stay stdlib-only (see NOTICE)";
         echo "--- stderr ---"; cat "${WORK}/noyaml.log"; exit 1; }

noyaml_rows=$(($(wc -l < "${OUT2}") - 1))
[[ ${noyaml_rows} -eq 6 ]] || fail "yaml-free run wrote ${noyaml_rows} rows, expected 6"

# ROW COUNT ALONE IS NOT THE ASSERTION, AND FOR MONTHS IT WAS.
# This block passed from the day it was written while PyYAML was imported at
# dataset.py module level -- i.e. while the property it names was FALSE. The
# ImportError fired inside predict.py's per-clip try/except, all six clips fell
# back to the majority class, a well-formed 6-row CSV was written, the script
# exited 0, and every assertion above was satisfied. The test could not tell a
# yaml-free pipeline from a pipeline whose model never loaded.
#
# So assert the MODEL, not the file. If the ensemble line is absent, the run
# degraded and this check means nothing. (Found 2026-08-15 when §J-DQ (a) turned
# the whole-run fallback into a hard failure and this test went red -- the fix to
# one defect is what made the other one visible.)
grep -q "\[predict\] ensemble: " "${WORK}/noyaml.log" \
    || fail "yaml-free run wrote its rows WITHOUT loading a model — this check was vacuous"

echo "PASS: 6/6 degenerate clips survived, exit 0, all predictions in range, inference path is YAML-free"
