#!/usr/bin/env bash
# Background queue for config/extension_v3_protocol.json (seed 42). Three lanes share the GPU.
cd "$(dirname "$0")/.."
PY=../../venv/parts_deep/bin/python
LOG=reports/extension_v3/logs; mkdir -p "$LOG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

step() {  # name, command...
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$LOG/queue.log"
  if "$@" > "$LOG/$name.log" 2>&1; then
    echo "$(date '+%F %T') OK    $name" >> "$LOG/queue.log"
  else
    echo "$(date '+%F %T') FAIL  $name" >> "$LOG/queue.log"
  fi
}

lane_holes() {
  step holes_temporal_train $PY scripts/run_extension_v3.py train-holes --fold temporal --run holes_field_temporal_s42 --select all_train
  step holes_temporal_eval  $PY scripts/run_extension_v3.py eval-holes --fold temporal --run holes_field_temporal_s42
  for k in 1 2 3 4; do
    step holes_f${k}_train $PY scripts/run_extension_v3.py train-holes --fold $k --run holes_field8_f${k}_s42_v3 --select nested8
    step holes_f${k}_eval  $PY scripts/run_extension_v3.py eval-holes --fold $k --run holes_field8_f${k}_s42_v3
  done
}

lane_cnn() {  # kind
  local kind=$1
  for k in temporal 1 2 3 4; do
    step ${kind}_${k}_train $PY run.py train-cnn --kind $kind --fold $k --seed 42 --run ${kind}_${k/#[0-9]/f$k}_s42 --workers 4
    step ${kind}_${k}_eval  $PY run.py eval-cnn --run ${kind}_${k/#[0-9]/f$k}_s42 --report ${kind}_${k/#[0-9]/f$k}_s42_test --workers 4
  done
}

lane_holes &
lane_cnn rgb &
lane_cnn rgbe &
wait
step collect $PY scripts/run_extension_v3.py collect
echo "$(date '+%F %T') ALL DONE" >> "$LOG/queue.log"
