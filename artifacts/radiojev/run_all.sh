#!/usr/bin/env bash
# Resume full RadioJEV E1 -> E5. Completed reports and checkpoints are retained.
# Usage: bash artifacts/radiojev/run_all.sh [--dry-run] [other-job-pgrep-pattern]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
LOG=artifacts/radiojev/run_all.log
PATTERN="antivenom/.venv/bin/python experiments"
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
if [[ $# -gt 1 ]]; then echo "Usage: $0 [--dry-run] [other-job-pgrep-pattern]" >&2; exit 2; fi
if [[ $# -eq 1 ]]; then PATTERN=$1; fi
PYTHON="${WJ_PYTHON:-.venv/bin/python}"
UV="${WJ_UV_BIN:-uv}"
unset WJ_SMOKE
# A pattern supplied on the command line also appears in this runner's own
# command and its launching shells. Those are not competing experiments.
EXCLUDE_PIDS=" $$ "
ancestor=$PPID
while [[ -n "$ancestor" && $ancestor -gt 1 ]]; do
  EXCLUDE_PIDS+="$ancestor "
  ancestor=$(ps -o ppid= -p "$ancestor" | tr -d ' ') || break
done

# E1 radio_lora is already complete. Do not rerun it or the smoke chain.
STAGES=(
  "wirelessjev.radiojev.e1 --arm radio_frozen"
  "wirelessjev.radiojev.e1 --arm intent"
  "wirelessjev.radiojev.e2 --arm radio_q"
  "wirelessjev.radiojev.e2 --arm e1_zero"
  "wirelessjev.radiojev.e2 --arm intent"
  "wirelessjev.radiojev.e3"
  "wirelessjev.radiojev.e45"
)
RESULTS=(e1_radio_frozen.json e1_intent.json e2_radio_q.json e2_e1_zero.json e2_intent.json e3.json e45.json)
# Only demand checkpoints consumed by later stages. E1 frozen can legitimately
# finish without saving an improved checkpoint, and no later arm loads it.
CHECKPOINTS=("" "" ckpt_radio_q.pt "" "" ckpt_e3.pt "")

wait_quiet() {
  local quiet=0 announced=0
  while [[ $quiet -lt ${WJ_QUIET_POLLS:-4} ]]; do
    if pgrep -f "$PATTERN" | awk -v excluded="$EXCLUDE_PIDS" '
      index(excluded, " " $1 " ") == 0 { busy=1 }
      END { exit !busy }'; then
      quiet=0
      if [[ $announced -eq 0 ]]; then
        echo "[$(date +%T)] waiting: '$PATTERN' is running" >> "$LOG"
        announced=1
      fi
    else
      quiet=$((quiet + 1))
    fi
    sleep "${WJ_QUIET_INTERVAL:-30}"
  done
}

if [[ $DRY_RUN -eq 0 && ! -s artifacts/radiojev/ckpt_radio_lora.pt ]]; then
  echo "Missing artifacts/radiojev/ckpt_radio_lora.pt, required by E2. Restore the completed E1 checkpoint from the original machine; the JSON report is not a checkpoint. No stages started." >&2
  exit 2
fi
for i in "${!STAGES[@]}"; do
  read -r -a args <<< "${STAGES[$i]}"
  result="artifacts/radiojev/${RESULTS[$i]}"
  checkpoint=${CHECKPOINTS[$i]}
  if [[ -f "$result" ]]; then
    # Reports are written only after full runs; smoke runs do not write them.
    "$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert isinstance(r,dict) and r' "$result"
    if [[ -n "$checkpoint" && ! -s artifacts/radiojev/$checkpoint ]]; then
      echo "Completed report $result lacks $checkpoint. Restore its checkpoint before resuming; existing evidence was preserved." >&2
      exit 2
    fi
    echo "SKIP completed: ${STAGES[$i]}"
    continue
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "$UV run --frozen python -m ${STAGES[$i]}"
    continue
  fi
  wait_quiet
  name=${RESULTS[$i]%.json}
  echo "[$(date +%T)] START full ${STAGES[$i]}" >> "$LOG"
  if PYTHONUNBUFFERED=1 "$UV" run --frozen python -m "${args[@]}" >> "artifacts/radiojev/full_${name}.log" 2>&1; then
    echo "[$(date +%T)] END full ${STAGES[$i]} (exit 0)" >> "$LOG"
  else
    rc=$?
    echo "[$(date +%T)] FAILED ${STAGES[$i]} (exit $rc); stopping" >> "$LOG"
    exit "$rc"
  fi
done
if [[ $DRY_RUN -eq 0 ]]; then echo "[$(date +%T)] ALL DONE" >> "$LOG"; fi
