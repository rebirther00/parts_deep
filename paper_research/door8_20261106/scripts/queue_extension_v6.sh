#!/usr/bin/env bash
# Background queue for config/extension_v6_protocol.json: 8 strict leave-one-class-out hole detectors (fold 0) in three lanes.
cd "$(dirname "$0")/.."
PY=../../venv/parts_deep/bin/python
LOG=reports/extension_v6/logs; mkdir -p "$LOG"
export PYTORCH_ALLOC_CONF=expandable_segments:True
CLASSES=(E25_door_LH_FRT E30_door_LH_FRT E38_door_LH_FRT E25_door_RH E25_door_LH_RR E30_E38_door_RH E30_door_LH_RR E38_door_LH_RR)

step() {  # name, command...
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$LOG/queue.log"
  if "$@" > "$LOG/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" >> "$LOG/queue.log"
  else echo "$(date '+%F %T') FAIL  $name" >> "$LOG/queue.log"; fi
}

lane() {  # lane index
  local i=$1
  for ((j=i; j<${#CLASSES[@]}; j+=3)); do
    local c=${CLASSES[$j]}
    step strict_${c}_train $PY scripts/run_extension_v6.py train-strict --exclude $c
    step strict_${c}_eval  $PY scripts/run_extension_v6.py eval-strict --exclude $c
  done
}

lane 0 & lane 1 & lane 2 &
wait
step collect $PY scripts/run_extension_v6.py collect
echo "$(date '+%F %T') ALL DONE" >> "$LOG/queue.log"
