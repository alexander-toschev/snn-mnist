#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/d/projects/snn-mnist
OUTDIR="$BASE/runs_csnn"
EXP_MD="$BASE/EXPERIMENTS-CSNN-CIFAR10020.md"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER_LOG="$OUTDIR/night_sweep_cifar10020_${STAMP}.log"

cd "$BASE"

# Activate conda
source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bindsnet

echo "[night_sweep] start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$MASTER_LOG"

BASE_STATUS="$OUTDIR/20260501T163648Z_358915b06a96/status.json"
if [ ! -f "$BASE_STATUS" ]; then
  echo "Missing base status: $BASE_STATUS" | tee -a "$MASTER_LOG"
  exit 2
fi

# Parameter grid (kept small for overnight throughput)
C2_OUT_LIST=(64 128)
RATE_LIST=(0.006 0.008 0.010)
MODE_LIST=("wta" "wta_ei")
EI_MULT_LIST=(0.005 0.010)

best_acc="-1"
best_run=""

append_result () {
  local run_id="$1"
  local status_json="$OUTDIR/$run_id/status.json"
  if [ ! -f "$status_json" ]; then
    return
  fi
  python - <<PY
import json
from pathlib import Path
p=Path("$status_json")
o=json.loads(p.read_text(encoding='utf-8'))
cfg=o.get('cfg',{})
acc=o.get('best_readout_acc')
assigned=(o.get('label_map_summary') or {}).get('assigned_neurons')
total=(o.get('label_map_summary') or {}).get('total_neurons')
err=o.get('error')
line = (
  f"- {o.get('created_at','')} {o.get('run_id')} | "
  f"c2_out={cfg.get('c2_out')} rate={cfg.get('poisson_rate_scale')} "
  f"wta={int(bool(cfg.get('wta_enable')))} ei={int(bool(cfg.get('ei_enable')))} "
  f"ei_mult2={cfg.get('ei_inh_mult_2layer')} N={cfg.get('N')} | "
  f"{o.get('status')} best_acc={acc} assigned={assigned}/{total} err={err}\n"
)
Path("$EXP_MD").write_text(Path("$EXP_MD").read_text(encoding='utf-8') + "\n" + line, encoding='utf-8')
PY
}

run_one () {
  local c2_out="$1"
  local rate="$2"
  local mode="$3"
  local ei_mult="$4"

  local cfg_json="$BASE/tmp_night_cfg_${c2_out}_${rate}_${mode}_${ei_mult}.json"

  python - <<PY
import json
from pathlib import Path
base = json.loads(Path("$BASE_STATUS").read_text(encoding='utf-8'))['cfg']
# base is 2-layer greedy + WTA
cfg=dict(base)
cfg['c2_out']=int($c2_out)
cfg['poisson_rate_scale']=float($rate)
# keep training budget constant
cfg['N']=5000
cfg['greedy_enable']=True
cfg['greedy_n1']=2500
# activity guard
cfg['activity_check_after']=1000
cfg['activity_min_spikes_win_mean']=1000.0

if "$mode"=="wta":
  cfg['wta_enable']=True
  cfg['ei_enable']=False
else:
  cfg['wta_enable']=True
  cfg['ei_enable']=True
  cfg.setdefault('ei_inh', 120.0)
  cfg.setdefault('ei_exc', 22.5)
  cfg['ei_inh_mult_2layer']=float($ei_mult)

Path("$cfg_json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
PY

  local log="$OUTDIR/night_${STAMP}_c2${c2_out}_r${rate}_${mode}_m${ei_mult}.log"
  echo "[night_sweep] run c2_out=$c2_out rate=$rate mode=$mode ei_mult=$ei_mult" | tee -a "$MASTER_LOG"

  PYTHONUNBUFFERED=1 python -u agent_mvp/experiment_runner.py \
    --outdir runs_csnn \
    --config-json "$cfg_json" \
    >"$log" 2>&1 || true

  # Extract run_id from the last JSON printed by experiment_runner (if any)
  local run_id
  run_id=$(grep -E '"run_id"' -n "$log" | tail -n 1 | sed -E 's/.*"run_id"\s*:\s*"([^"]+)".*/\1/' || true)
  if [ -n "$run_id" ]; then
    append_result "$run_id"

    # Track best (only if ok)
    local acc
    acc=$(python - <<PY
import json
from pathlib import Path
p=Path("$OUTDIR/$run_id/status.json")
o=json.loads(p.read_text(encoding='utf-8'))
print(o.get('best_readout_acc') or -1)
PY
)
    if python - <<PY
b=float("$best_acc")
a=float("$acc")
print(int(a>b))
PY
    | grep -q '^1$'; then
      best_acc="$acc"
      best_run="$run_id"
    fi
  fi
}

for c2_out in "${C2_OUT_LIST[@]}"; do
  for rate in "${RATE_LIST[@]}"; do
    for mode in "${MODE_LIST[@]}"; do
      if [ "$mode" = "wta" ]; then
        run_one "$c2_out" "$rate" "$mode" "0.0"
      else
        for m in "${EI_MULT_LIST[@]}"; do
          run_one "$c2_out" "$rate" "$mode" "$m"
        done
      fi
    done
  done
done

echo "[night_sweep] done $(date -u +%Y-%m-%dT%H:%M:%SZ) best_run=$best_run best_acc=$best_acc" | tee -a "$MASTER_LOG"

# Print a final one-liner for external caller
printf '{"best_run":"%s","best_acc":%s,"master_log":"%s"}\n' "$best_run" "$best_acc" "$MASTER_LOG"
