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

# Optional capability gates. Each records evidence when the capability is
# configured, and asserts the capability is genuinely available before treating
# its result as a pass. An unconfigured capability is skipped, never faked.
"${kval_bin}" power-plan --config "${config}" >"${run_dir}/07-power.json" 2>&1 || true
if jq -e '.available == true' "${run_dir}/07-power.json" >/dev/null 2>&1; then
  echo "power governance: available"
  jq -e '.recommendation.target_nvpmodel_id != null' "${run_dir}/07-power.json" >/dev/null
else
  echo "power governance: not configured or no reading (skipped)"
fi

"${kval_bin}" spectrum-scan --config "${config}" >"${run_dir}/08-spectrum.json" 2>&1 || true
if jq -e '.available == true' "${run_dir}/08-spectrum.json" >/dev/null 2>&1; then
  echo "spectrum sensing: available"
  jq -e '[.bands[] | select(.status != "INSUFFICIENT_EVIDENCE")] | length >= 1' \
    "${run_dir}/08-spectrum.json" >/dev/null
  # Receive-only invariant: every observation the adapter returned must be passive.
  jq -e 'all(.observations[]; .passive == true)' "${run_dir}/08-spectrum.json" >/dev/null
else
  echo "spectrum sensing: not configured or adapter absent (skipped)"
fi

"${kval_bin}" transport-preflight --config "${config}" >"${run_dir}/09-transport.json" 2>&1 || true
if jq -e '.available == true' "${run_dir}/09-transport.json" >/dev/null 2>&1; then
  echo "secure transport: configured"
  if [[ ${KVAL_HIL_REQUIRE_TRANSPORT:-} == YES ]]; then
    jq -e '.verified == true' "${run_dir}/09-transport.json" >/dev/null
  fi
else
  echo "secure transport: not configured (skipped)"
fi

if [[ -n ${KVAL_HIL_APPLY_PROFILE:-} ]]; then
  KVAL_ENABLE_APPLY=YES "${kval_bin}" hil-activate \
    --config "${config}" \
    --profile-id "${KVAL_HIL_APPLY_PROFILE}" \
    | tee "${run_dir}/10-activation.json"
  jq -e '(.activation.success == true) or (.decision.reason_codes | index("CURRENT_PROFILE_ALREADY_SELECTED") != null)' \
    "${run_dir}/10-activation.json" >/dev/null
fi

find "${run_dir}" -maxdepth 1 -type f -name '*.json' -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${run_dir}/MANIFEST.sha256"

echo "Physical validation passed. Evidence: ${run_dir}"

