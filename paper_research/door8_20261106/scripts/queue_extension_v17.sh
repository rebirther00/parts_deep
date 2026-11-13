#!/usr/bin/env bash
# Background queue for config/extension_v17_protocol.json: the 16 fold-1 unseen-condition CNNs trained for 40 epochs
# without early stopping (RULE and LAST checkpoints), their evaluation, then the tabulation. Five lanes, as in v16.
cd "$(dirname "$0")/.."
PY=../../venv/parts_deep/bin/python
LOG=reports/extension_v17/logs; mkdir -p "$LOG"
export PYTORCH_ALLOC_CONF=expandable_segments:True
mapfile -t JOBS < <($PY scripts/run_extension_v17.py list-jobs 2>/dev/null | grep -E '^(rgb|rgbe) ')

step() {  # name, command...
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$LOG/queue.log"
  if "$@" > "$LOG/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" >> "$LOG/queue.log"
  else echo "$(date '+%F %T') FAIL  $name" >> "$LOG/queue.log"; fi
}

lane() {  # lane index
  local i=$1
  for ((j=i; j<${#JOBS[@]}; j+=5)); do
    read -r kind c <<< "${JOBS[$j]}"
    step ${kind}_${c}_train     $PY scripts/run_extension_v17.py train --kind $kind --exclude $c
    step ${kind}_${c}_eval_last $PY scripts/run_extension_v17.py eval --kind $kind --exclude $c --ckpt last
    step ${kind}_${c}_eval_rule $PY scripts/run_extension_v17.py eval --kind $kind --exclude $c --ckpt rule
  done
}

echo "$(date '+%F %T') QUEUE ${#JOBS[@]} jobs" >> "$LOG/queue.log"
step prepare $PY scripts/run_extension_v17.py prepare
lane 0 & lane 1 & lane 2 & lane 3 & lane 4 &
wait
step analyze $PY scripts/run_extension_v17.py analyze
echo "$(date '+%F %T') ALL DONE" >> "$LOG/queue.log"
