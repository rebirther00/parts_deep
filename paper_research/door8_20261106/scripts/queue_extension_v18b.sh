#!/usr/bin/env bash
# Addendum queue (config/extension_v18b_protocol.json): T1b = the 500 px rule with the order-free validation score, and the
# order-free test-role localisation of T1 and T1b.
cd "$(dirname "$0")/.."
PY=../../venv/parts_deep/bin/python
LOG=reports/extension_v18/logs
step() {  # name, command...
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$LOG/queue.log"
  if "$@" > "$LOG/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" >> "$LOG/queue.log"
  else echo "$(date '+%F %T') FAIL  $name" >> "$LOG/queue.log"; fi
}
echo "$(date '+%F %T') QUEUE addendum start" >> "$LOG/queue.log"
step gap_locfree $PY scripts/run_extension_v18.py loc-free --variant gap
for s in 0 1 2 3 4 temporal; do
  step gapfree_${s}_select $PY scripts/run_extension_v18.py select --variant gapfree --split $s
  step gapfree_${s}_detect $PY scripts/run_extension_v18.py detect --variant gapfree --split $s --procs 20
  step gapfree_${s}_eval   $PY scripts/run_extension_v18.py eval --variant gapfree --split $s
done
step gapfree_collect $PY scripts/run_extension_v18.py collect --variant gapfree
step gapfree_locfree $PY scripts/run_extension_v18.py loc-free --variant gapfree
echo "$(date '+%F %T') ADDENDUM DONE" >> "$LOG/queue.log"
