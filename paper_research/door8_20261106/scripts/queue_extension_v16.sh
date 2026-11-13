#!/usr/bin/env bash
# Background queue for config/extension_v16_protocol.json (bundle D): unseen-class condition on folds 1-4, seed 42
# (32 strict hole detectors, then 64 CNNs) in five lanes; validation features for C2 (new runs and the 24 fold-0 runs);
# then the per-fold analysis. Hole jobs come first because they carry the main result.
cd "$(dirname "$0")/.."
PY=../../venv/parts_deep/bin/python
LOG=reports/extension_v16/logs; mkdir -p "$LOG"
export PYTORCH_ALLOC_CONF=expandable_segments:True
mapfile -t JOBS < <($PY scripts/run_extension_v16.py list-jobs 2>/dev/null | grep -E '^(hole|rgb|rgbe) ')
CLASSES=$(printf '%s\n' "${JOBS[@]}" | awk '$1=="hole" && $3=="1"{print $2}')
for seed in 42 123 456; do for c in $CLASSES; do JOBS+=("val0 $c $seed"); done; done

step() {  # name, command...
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$LOG/queue.log"
  if "$@" > "$LOG/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" >> "$LOG/queue.log"
  else echo "$(date '+%F %T') FAIL  $name" >> "$LOG/queue.log"; fi
}

lane() {  # lane index
  local i=$1
  for ((j=i; j<${#JOBS[@]}; j+=5)); do
    read -r kind c k <<< "${JOBS[$j]}"
    if [ "$kind" = hole ]; then
      step hole_${c}_f${k}_train $PY scripts/run_extension_v16.py train-hole --exclude $c --fold $k
      step hole_${c}_f${k}_eval  $PY scripts/run_extension_v16.py eval-hole --exclude $c --fold $k
      step hole_${c}_f${k}_val   $PY scripts/run_extension_v16.py val-hole --exclude $c --fold $k
    elif [ "$kind" = val0 ]; then
      step hole_${c}_f0_s${k}_val $PY scripts/run_extension_v16.py val-hole --exclude $c --fold 0 --seed $k
    else
      step ${kind}_${c}_f${k}_train $PY scripts/run_extension_v16.py train-cnn --kind $kind --exclude $c --fold $k
      step ${kind}_${c}_f${k}_eval  $PY scripts/run_extension_v16.py eval-cnn --kind $kind --exclude $c --fold $k
    fi
  done
}

echo "$(date '+%F %T') QUEUE ${#JOBS[@]} jobs" >> "$LOG/queue.log"
lane 0 & lane 1 & lane 2 & lane 3 & lane 4 &
wait
for k in 1 2 3 4; do step analyze_f$k $PY scripts/run_extension_v16.py analyze --fold $k; done
echo "$(date '+%F %T') ALL DONE" >> "$LOG/queue.log"
