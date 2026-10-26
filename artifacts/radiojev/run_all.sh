#!/bin/zsh
# RadioJEV E1 -> E5. Waits for the given PIDs (other GPU users) to exit,
# runs every stage in smoke mode first (stops on any failure), then the
# full chain. Progress: artifacts/radiojev/run_all.log
cd "/Users/danielfoojunwei/Preceptual.ai WirelessJEV"
LOG=artifacts/radiojev/run_all.log
STAGES=(
  "wirelessjev.radiojev.e1 --arm radio_lora"
  "wirelessjev.radiojev.e1 --arm radio_frozen"
  "wirelessjev.radiojev.e1 --arm intent"
  "wirelessjev.radiojev.e2 --arm radio_q"
  "wirelessjev.radiojev.e2 --arm e1_zero"
  "wirelessjev.radiojev.e2 --arm intent"
  "wirelessjev.radiojev.e3"
  "wirelessjev.radiojev.e45"
)
# Wait until no process matching the given pattern (another project's GPU
# jobs) has run for two consecutive minutes.
PATTERN="${1:-Swarm Alignment Hackathon}"
echo "[$(date +%T)] waiting for the GPU (no '$PATTERN' processes for 2 min)" >> $LOG
quiet=0
while [ $quiet -lt 4 ]; do
  if pgrep -f "$PATTERN" >/dev/null; then quiet=0; else quiet=$((quiet + 1)); fi
  sleep 30
done
echo "[$(date +%T)] GPU free" >> $LOG
run() {  # $1 = mode label, rest = module + args
  local mode=$1; shift
  local name=$(echo "$*" | tr ' /' '__')
  echo "[$(date +%T)] START $mode $*" >> $LOG
  PYTHONUNBUFFERED=1 ~/.local/bin/uv run python -m ${=*} > artifacts/radiojev/${mode}_${name}.log 2>&1
  local rc=$?
  echo "[$(date +%T)] END $mode $* (exit $rc)" >> $LOG
  return $rc
}
for st in "${STAGES[@]}"; do
  WJ_SMOKE=1 run smoke $st || { echo "[$(date +%T)] SMOKE FAILED; stopping" >> $LOG; exit 1; }
done
rm -f artifacts/radiojev/ckpt_*.pt
echo "[$(date +%T)] smoke passed; full run" >> $LOG
for st in "${STAGES[@]}"; do
  run full $st
done
echo "[$(date +%T)] ALL DONE" >> $LOG
