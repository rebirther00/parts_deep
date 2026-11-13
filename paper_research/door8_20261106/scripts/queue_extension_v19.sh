#!/usr/bin/env bash
# Background queue for config/extension_v19_protocol.json: E1 (E23), E2 (40 unseen-class runs, four lanes), E3 (six fixed
# NCC settings x six splits, three lanes). CPU only.
cd "$(dirname "$0")/.."
PY=../../venv/parts_deep/bin/python
LOG=reports/extension_v19/logs; mkdir -p "$LOG"
CLASSES=(E25_door_LH_FRT E30_door_LH_FRT E38_door_LH_FRT E25_door_RH E25_door_LH_RR E30_E38_door_RH E30_door_LH_RR E38_door_LH_RR)

step() {  # name, command...
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$LOG/queue.log"
  if "$@" > "$LOG/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" >> "$LOG/queue.log"
  else echo "$(date '+%F %T') FAIL  $name" >> "$LOG/queue.log"; fi
}

LOCO=(); for s in 0 1 2 3 4; do for c in "${CLASSES[@]}"; do LOCO+=("$s $c"); done; done
FIXED=(); for i in 0 1 2 3 4 5; do for s in temporal 0 1 2 3 4; do FIXED+=("$s $i"); done; done

loco_lane() { local i=$1; for ((j=i; j<${#LOCO[@]}; j+=4)); do read -r s c <<< "${LOCO[$j]}"; step loco_${s}_${c} $PY scripts/run_extension_v19.py loco --split $s --cls $c --procs 5; done; }
fixed_lane() { local i=$1; for ((j=i; j<${#FIXED[@]}; j+=3)); do read -r s k <<< "${FIXED[$j]}"; step fixed_${s}_c${k} $PY scripts/run_extension_v19.py fixed --split $s --cand $k --procs 7; done; }

echo "$(date '+%F %T') QUEUE start" >> "$LOG/queue.log"
step e23 $PY scripts/run_extension_v19.py e23 --procs 8
loco_lane 0 & loco_lane 1 & loco_lane 2 & loco_lane 3 &
wait
step loco_collect $PY scripts/run_extension_v19.py loco-collect
fixed_lane 0 & fixed_lane 1 & fixed_lane 2 &
wait
step fixed_collect $PY scripts/run_extension_v19.py fixed-collect
echo "$(date '+%F %T') ALL DONE" >> "$LOG/queue.log"
