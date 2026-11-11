#!/usr/bin/env bash
# Background queue for config/extension_v12_protocol.json (no training): CPU analyses first, then CNN forward passes and
# sub-pixel feature extraction in three lanes, then the analyses that need them.
cd "$(dirname "$0")/.."
PY=../../venv/parts_deep/bin/python
LOG=reports/extension_v12/logs; mkdir -p "$LOG"
export PYTORCH_ALLOC_CONF=expandable_segments:True

step() {  # name, command...
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$LOG/queue.log"
  if "$@" > "$LOG/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" >> "$LOG/queue.log"
  else echo "$(date '+%F %T') FAIL  $name" >> "$LOG/queue.log"; fi
}

step hole_register $PY scripts/run_extension_v12.py hole-register
step hole_gate     $PY scripts/run_extension_v12.py hole-gate
step frames        $PY scripts/run_extension_v12.py frames

mapfile -t JOBS < <($PY scripts/run_extension_v12.py list-jobs 2>/dev/null)
lane() {  # lane index
  local i=$1
  for ((j=i; j<${#JOBS[@]}; j+=3)); do
    local job=${JOBS[$j]}; local name=$(echo "$job" | sed 's/ --run / /; s/ --fold / f/; s/ /_/g')
    step "$name" $PY scripts/run_extension_v12.py $job
  done
}
lane 0 & lane 1 & lane 2 &
wait
step cnn_analyze   $PY scripts/run_extension_v12.py cnn-analyze
step subpixel_eval $PY scripts/run_extension_v12.py subpixel-eval
step timing        $PY scripts/run_extension_v12.py timing
step collect       $PY scripts/run_extension_v12.py collect
echo "$(date '+%F %T') ALL DONE" >> "$LOG/queue.log"
