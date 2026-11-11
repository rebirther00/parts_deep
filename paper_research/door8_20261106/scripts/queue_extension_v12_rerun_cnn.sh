#!/usr/bin/env bash
# Re-run of the CNN forward passes of config/extension_v12_protocol.json after a bookkeeping failure
# (TypeError while writing the run metadata, after the forward pass; no score had been analysed). Analysis code unchanged.
cd "$(dirname "$0")/.."
PY=../../venv/parts_deep/bin/python
LOG=reports/extension_v12/logs; mkdir -p "$LOG"
export PYTORCH_ALLOC_CONF=expandable_segments:True
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$LOG/queue.log"
  if "$@" > "$LOG/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" >> "$LOG/queue.log"
  else echo "$(date '+%F %T') FAIL  $name" >> "$LOG/queue.log"; fi; }
echo "$(date '+%F %T') RERUN cnn-infer (attempt 2)" >> "$LOG/queue.log"
mapfile -t JOBS < <($PY scripts/run_extension_v12.py list-jobs 2>/dev/null | grep cnn-infer)
lane() { local i=$1
  for ((j=i; j<${#JOBS[@]}; j+=3)); do
    local job=${JOBS[$j]}; local name=$(echo "$job" | sed 's/ --run / /; s/ /_/g')
    step "$name" $PY scripts/run_extension_v12.py $job
  done; }
lane 0 & lane 1 & lane 2 &
wait
step cnn_analyze $PY scripts/run_extension_v12.py cnn-analyze
step collect     $PY scripts/run_extension_v12.py collect
echo "$(date '+%F %T') ALL DONE (attempt 2)" >> "$LOG/queue.log"
