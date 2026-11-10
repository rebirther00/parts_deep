#!/usr/bin/env bash
# Background queue for config/extension_v5_protocol.json: 16 leave-one-class-out CNN runs (fold 0) in three lanes.
cd "$(dirname "$0")/.."
PY=../../venv/parts_deep/bin/python
LOG=reports/extension_v5/logs; mkdir -p "$LOG"
export PYTORCH_ALLOC_CONF=expandable_segments:True
CLASSES=(E25_door_LH_FRT E30_door_LH_FRT E38_door_LH_FRT E25_door_RH E25_door_LH_RR E30_E38_door_RH E30_door_LH_RR E38_door_LH_RR)
JOBS=()
for c in "${CLASSES[@]}"; do JOBS+=("rgb:$c" "rgbe:$c"); done

step() {  # name, command...
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$LOG/queue.log"
  if "$@" > "$LOG/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" >> "$LOG/queue.log"
  else echo "$(date '+%F %T') FAIL  $name" >> "$LOG/queue.log"; fi
}

lane() {  # lane index
  local i=$1
  for ((j=i; j<${#JOBS[@]}; j+=3)); do
    local kind=${JOBS[$j]%%:*} c=${JOBS[$j]#*:}
    local run=${kind}_loco_${c}_f0_s42
    step ${run}_train $PY scripts/run_extension_v5.py train-cnn-loco --kind $kind --exclude $c --run $run --workers 4
    step ${run}_eval  $PY scripts/run_extension_v5.py eval-cnn-loco --run $run
  done
}

lane 0 & lane 1 & lane 2 &
wait
step collect $PY scripts/run_extension_v5.py collect
echo "$(date '+%F %T') ALL DONE" >> "$LOG/queue.log"
