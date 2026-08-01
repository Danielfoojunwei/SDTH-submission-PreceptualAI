#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "run_hil_validation.sh must run as root on the target Jetson" >&2
  exit 1
fi

kval_bin=${KVAL_BIN:-/opt/kval/venv/bin/kval-node}
config=${KVAL_CONFIG:-/etc/kval/node.json}
evidence_dir=${KVAL_EVIDENCE_DIR:-/var/lib/kval/validation}
run_id=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="${evidence_dir}/${run_id}"
install -d -m 0750 "${run_dir}"

"${kval_bin}" preflight | tee "${run_dir}/01-platform.json"
"${kval_bin}" verify-catalog --config "${config}" | tee "${run_dir}/02-catalog.json"
"${kval_bin}" once --config "${config}" | tee "${run_dir}/03-first-cycle.json"
sleep 3
"${kval_bin}" once --config "${config}" | tee "${run_dir}/04-second-cycle.json"
"${kval_bin}" benchmark --config "${config}" --iterations 100 | tee "${run_dir}/05-benchmark.json"
"${kval_bin}" verify-audit --config "${config}" | tee "${run_dir}/06-audit.json"

jq -e '.p95_ms <= 100.0' "${run_dir}/05-benchmark.json" >/dev/null
jq -e '.valid == true' "${run_dir}/06-audit.json" >/dev/null
jq -e '[.telemetry.observations[] | select(.confidence >= 0.75)] | length >= 1' \
  "${run_dir}/04-second-cycle.json" >/dev/null

if [[ -n ${KVAL_HIL_APPLY_PROFILE:-} ]]; then
  KVAL_ENABLE_APPLY=YES "${kval_bin}" hil-activate \
    --config "${config}" \
    --profile-id "${KVAL_HIL_APPLY_PROFILE}" \
    | tee "${run_dir}/07-activation.json"
  jq -e '(.activation.success == true) or (.decision.reason_codes | index("CURRENT_PROFILE_ALREADY_SELECTED") != null)' \
    "${run_dir}/07-activation.json" >/dev/null
fi

find "${run_dir}" -maxdepth 1 -type f -name '*.json' -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${run_dir}/MANIFEST.sha256"

echo "Physical validation passed. Evidence: ${run_dir}"

