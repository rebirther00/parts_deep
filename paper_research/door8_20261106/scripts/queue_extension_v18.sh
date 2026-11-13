#!/usr/bin/env bash
# Background queue for config/extension_v18_protocol.json: T0 (same-hole share of the stored v16e detections), W0 / W1
# (plane distance, outer width with the TRAIN-only constant), then T1 (gap) and T2 (perkey) in two lanes. CPU only.
cd "$(dirname "$0")/.."
PY=../../venv/parts_deep/bin/python
LOG=reports/extension_v18/logs; mkdir -p "$LOG"

step() {  # name, command...
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$LOG/queue.log"
  if "$@" > "$LOG/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" >> "$LOG/queue.log"
  else echo "$(date '+%F %T') FAIL  $name" >> "$LOG/queue.log"; fi
}

lane() {  # variant
  local v=$1
  for s in 0 1 2 3 4 temporal; do
    step ${v}_${s}_select $PY scripts/run_extension_v18.py select --variant $v --split $s
    step ${v}_${s}_detect $PY scripts/run_extension_v18.py detect --variant $v --split $s --procs 10
    step ${v}_${s}_eval   $PY scripts/run_extension_v18.py eval --variant $v --split $s
  done
  step ${v}_collect $PY scripts/run_extension_v18.py collect --variant $v
}

echo "$(date '+%F %T') QUEUE start" >> "$LOG/queue.log"
step same_hole $PY scripts/run_extension_v18.py same-hole
step plane     $PY scripts/run_extension_v18.py plane
step width     $PY scripts/run_extension_v18.py width
lane gap & lane perkey &
wait
echo "$(date '+%F %T') ALL DONE" >> "$LOG/queue.log"
