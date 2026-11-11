#!/usr/bin/env bash
# Background queue for config/extension_v12s_protocol.json: unseen-class condition with seeds 123 and 456
# (16 strict hole detectors + 32 CNNs, fold 0) in five lanes, then the per-seed analysis and the three-seed summary.
cd "$(dirname "$0")/.."
PY=../../venv/parts_deep/bin/python
LOG=reports/extension_v12s/logs; mkdir -p "$LOG"
export PYTORCH_ALLOC_CONF=expandable_segments:True
mapfile -t JOBS < <($PY scripts/run_extension_v12s.py list-jobs 2>/dev/null | grep -E '^(hole|rgb|rgbe) ')

step() {  # name, command...
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$LOG/queue.log"
  if "$@" > "$LOG/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" >> "$LOG/queue.log"
  else echo "$(date '+%F %T') FAIL  $name" >> "$LOG/queue.log"; fi
}

lane() {  # lane index
  local i=$1
  for ((j=i; j<${#JOBS[@]}; j+=5)); do
    read -r kind c seed <<< "${JOBS[$j]}"
    if [ "$kind" = hole ]; then
      step hole_${c}_s${seed}_train $PY scripts/run_extension_v12s.py train-hole --exclude $c --seed $seed
      step hole_${c}_s${seed}_eval  $PY scripts/run_extension_v12s.py eval-hole --exclude $c --seed $seed
    else
      step ${kind}_${c}_s${seed}_train $PY scripts/run_extension_v12s.py train-cnn --kind $kind --exclude $c --seed $seed
      step ${kind}_${c}_s${seed}_eval  $PY scripts/run_extension_v12s.py eval-cnn --kind $kind --exclude $c --seed $seed
    fi
  done
}

echo "$(date '+%F %T') QUEUE ${#JOBS[@]} jobs" >> "$LOG/queue.log"
lane 0 & lane 1 & lane 2 & lane 3 & lane 4 &
wait
step analyze_s123 $PY scripts/run_extension_v12s.py analyze --seed 123
step analyze_s456 $PY scripts/run_extension_v12s.py analyze --seed 456
step collect      $PY scripts/run_extension_v12s.py collect
echo "$(date '+%F %T') ALL DONE" >> "$LOG/queue.log"
