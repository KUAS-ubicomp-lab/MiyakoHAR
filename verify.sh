#!/usr/bin/env bash
#
# CUHK-X Small Model Track — reproduction check.
#
#   ./verify.sh <data_dir> <submitted_csv>
#
# Re-runs the real entry point on the real data and diffs the result against a
# CSV that was actually submitted. Answers the one question verification asks:
# does this repository, as shipped, reproduce that file?
#
# IT IS NOT A TEST OF THE MODEL. It is a test of REPRODUCIBILITY, and the two
# fail differently. A model that is merely bad still verifies; a pipeline that is
# non-deterministic, that silently picks up a different checkpoint, or that
# depends on something not in this repository, does not — and that is a
# disqualification rather than a low score. The gate on the organisers' side is
# reproduction against a >10% accuracy gap, so a run that quietly changes 5% of
# its predictions between invocations is the failure mode worth catching here.
#
# Exit 0 ONLY on a byte-identical CSV. Anything else is a non-zero exit with the
# count and the first few differing rows printed — never a warning that scrolls
# past.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <data_dir> <submitted_csv>" >&2
    exit 2
fi

DATA_DIR="$1"
SUBMITTED="$2"
[[ -d "${DATA_DIR}"  ]] || { echo "[verify] error: no such data_dir: ${DATA_DIR}" >&2; exit 2; }
[[ -f "${SUBMITTED}" ]] || { echo "[verify] error: no such CSV: ${SUBMITTED}" >&2; exit 2; }

if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
    PYTHON="${REPO_DIR}/.venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi

echo "=============================================================="
echo "  1 · THE WEIGHTS THAT WILL VOTE — sha256 of each, and the set"
echo "=============================================================="
# Read the shipping list from predict.py itself rather than repeating it here.
# A second copy of that list is exactly the drift m72 caught between predict.py
# and package.py; tests/check_ckpt_meta.py asserts train.sh agrees with it too.
mapfile -t CKPTS < <("${PYTHON}" - <<'PY'
import sys
sys.path.insert(0, "src")
from predict import SHIPPING
print("\n".join(f"{s}/swa.pt" for s in SHIPPING))
PY
)
# B1, 2026-08-20. R4 requires everything loaded at inference to ship as ONE
# file, and since B1 that file is `checkpoints/model.pt`. A committee therefore
# receives the CONTAINER and NO runs/ directory at all — so demanding the run
# checkpoints here hard-failed the very configuration we are supposed to be
# verifying. The container is now the primary object; the run checkpoints remain
# the dev-path fallback, exactly as in predict.py, and the two are reported
# distinctly so nobody can mistake which one was actually verified.
# P0 (a), 2026-08-31: the site names the deliverable `checkpoints/model.pth`; this
# repo's slot was `model.pt` since B1. Both are accepted, `.pth` preferred — the same
# two names in the same order as predict.py's shipped_container(); the one weighed and
# loaded is printed by name below.
CONTAINER="checkpoints/model.pth"
[[ -f "${REPO_DIR}/${CONTAINER}" ]] || { [[ -f "${REPO_DIR}/checkpoints/model.pt" ]] && CONTAINER="checkpoints/model.pt"; }
if [[ -f "${REPO_DIR}/${CONTAINER}" ]]; then
    echo "  R4 SINGLE-FILE ARTEFACT (this is what ships):"
    echo "  $(sha256sum "${REPO_DIR}/${CONTAINER}" | cut -c1-16)  ${CONTAINER}  ($(stat -c%s "${REPO_DIR}/${CONTAINER}") B)"
    "${PYTHON}" - "${REPO_DIR}/${CONTAINER}" <<'PY'
import sys, torch
mem = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if not isinstance(mem, list):
    print("  FATAL: that file is not a packed container"); raise SystemExit(1)
for m in mem:
    print(f"    member {m['run']:<44} arch={m['arch']:<11} norm={m['norm']:<9} "
          f"T={m['T']} in_ch={m['in_channels']} size={m.get('input_size')}")
print(f"  {len(mem)} member(s) in ONE file")
PY
    [[ $? -eq 0 ]] || { echo "[verify] FATAL: container unreadable." >&2; exit 1; }
else
    echo "  no ${CONTAINER} — falling back to the run checkpoints. THIS IS NOT"
    echo "    THE R4 ARTEFACT; build it with tools/package.py --shipping --out ${CONTAINER}"
    missing=0
    for c in "${CKPTS[@]}"; do
        if [[ -f "${REPO_DIR}/${c}" ]]; then
            echo "  $(sha256sum "${REPO_DIR}/${c}" | cut -c1-16)  ${c}"
        else
            echo "  MISSING                ${c}"; missing=1
        fi
    done
    if [[ ${missing} -ne 0 ]]; then
        echo "[verify] FATAL: no container AND a shipping checkpoint is absent." >&2
        exit 1
    fi
fi

echo
echo "=============================================================="
echo "  2 · THE PACKAGED ARTEFACT — weighed against D6's 95 MB cap"
echo "=============================================================="
# B1 again: `package.py --shipping` REPACKS from runs/, which do not exist in
# the configuration a committee receives. When the container is present, weigh
# the container itself — that is the object the cap actually applies to, and
# weighing it directly is stricter than re-deriving something equal to it.
if [[ -f "${REPO_DIR}/${CONTAINER}" ]]; then
    bytes=$(stat -c%s "${REPO_DIR}/${CONTAINER}")
    mb=$(awk -v b="${bytes}" 'BEGIN{printf "%.2f", b/1e6}')
    echo "  MEASURED ${mb} MB  (${bytes} B, weighed on disk — never params x 2)"
    awk -v m="${mb}" 'BEGIN{ if (m+0 <= 95) print "  D6 ceiling 95 MB packaged → LEGAL";
                             else { print "  D6 ceiling 95 MB packaged → OVER"; exit 1 } }' || exit 1
    # 2026-09-09 (D49, the F1 audit): the ORGANISERS' line, in decimal bytes as D28 reads it and as the host defines it
    # (discussion 735601: "both your pre-processing model and your recognition model combined must be under 100 MB") —
    # ADDITIVE to our own 95 MB gate above, never a replacement. This container is the only model loaded at inference
    # (src/predict.py loads ONE file; the depth-region pre-processing has no learned parameters).
    if [[ "${bytes}" -lt 100000000 ]]; then
        echo "  organisers' limit 100,000,000 B (every model loaded at inference, combined) → ${bytes} B UNDER"
    else
        echo "  organisers' limit 100,000,000 B (every model loaded at inference, combined) → ${bytes} B OVER"; exit 1
    fi
else
    "${PYTHON}" "${REPO_DIR}/tools/package.py" --shipping | tail -6
fi

echo
echo "=============================================================="
echo "  3 · RE-RUN THE REAL ENTRY POINT"
echo "=============================================================="
TMP_CSV="$(mktemp -t verify_XXXXXX.csv)"
trap 'rm -f "${TMP_CSV}"' EXIT
"${REPO_DIR}/inference.sh" "${DATA_DIR}" "${TMP_CSV}"

echo
echo "=============================================================="
echo "  4 · DIFF AGAINST THE SUBMITTED FILE"
echo "=============================================================="
echo "  submitted: $(sha256sum "${SUBMITTED}" | cut -c1-16)  ${SUBMITTED}"
echo "  reproduced: $(sha256sum "${TMP_CSV}" | cut -c1-16)"

if diff -q "${SUBMITTED}" "${TMP_CSV}" >/dev/null; then
    echo
    echo "  IDENTICAL — every one of $(( $(wc -l < "${SUBMITTED}") - 1 )) rows reproduces."
    exit 0
fi

n=$(diff <(tail -n +2 "${SUBMITTED}") <(tail -n +2 "${TMP_CSV}") | grep -c '^<' || true)
total=$(( $(wc -l < "${SUBMITTED}") - 1 ))
echo
echo "  DIFFERS on ${n} of ${total} rows. First 5:" >&2
diff <(tail -n +2 "${SUBMITTED}") <(tail -n +2 "${TMP_CSV}") | head -12 >&2
echo "  This is a reproducibility failure, not a scoring one. Do not ship until" >&2
echo "  it is explained: a non-deterministic pipeline fails verification even" >&2
echo "  when the model is fine." >&2
exit 1
