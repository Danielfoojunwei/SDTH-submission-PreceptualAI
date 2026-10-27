#!/bin/zsh
# RadioJEV E1 -> E5. Before every stage, waits until the other project's GPU
# experiments (PATTERN) have been quiet for two minutes. Runs every stage in
# smoke mode first (stops on any failure), then the full chain.
# Progress: artifacts/radiojev/run_all.log
cd "/Users/danielfoojunwei/Preceptual.ai WirelessJEV"
LOG=artifacts/radiojev/run_all.log
PATTERN="${1:-antivenom/.venv/bin/python experiments}"
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
wait_quiet() {
  local quiet=0 announced=0
  while [ $quiet -lt 4 ]; do
    if pgrep -f "$PATTERN" >/dev/null; then
      quiet=0
      [ $announced -eq 0 ] && echo "[$(date +%T)] waiting: '$PATTERN' is running" >> $LOG && announced=1
    else
      quiet=$((quiet + 1))
    fi
    sleep 30
  done
}
run() {  # $1 = mode label, rest = module + args
  local mode=$1; shift
  local name=$(echo "$*" | tr ' /' '__')
  wait_quiet
  echo "[$(date +%T)] START $mode $*" >> $LOG
  PYTHONUNBUFFERED=1 ~/.local/bin/uv run python -m ${=*} > artifacts/radiojev/${mode}_${name}.log 2>&1
  local rc=$?
  echo "[$(date +%T)] END $mode $* (exit $rc)" >> $LOG
  return $rc
}
echo "[$(date +%T)] runner started" >> $LOG
for st in "${STAGES[@]}"; do
  WJ_SMOKE=1 run smoke $st || { echo "[$(date +%T)] SMOKE FAILED; stopping" >> $LOG; exit 1; }
done
rm -f artifacts/radiojev/ckpt_*.pt
echo "[$(date +%T)] smoke passed; full run" >> $LOG
for st in "${STAGES[@]}"; do
  run full $st
done
echo "[$(date +%T)] ALL DONE" >> $LOG
