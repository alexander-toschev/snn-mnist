#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/projects/snn-mnist

source /home/alex/anaconda3/etc/profile.d/conda.sh
conda activate bindsnet

OUTDIR="runs_csnn"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUTDIR"

EXPERIMENTS_MD="/mnt/d/projects/snn-mnist/EXPERIMENTS-CSNN-CIFAR10020.md"

echo "Sweep WTA-only: rate_scale in 0.007 0.008 0.009 0.010 (started $(date -Is))"

rates=(0.007 0.008 0.009 0.010)

for rate in "${rates[@]}"; do
  LOG="$OUTDIR/sweep_wta_only_rate_scale_${rate}_${STAMP}.log"
  echo "\n=== rate_scale=${rate} started at $(date -Is) ===" | tee -a "$LOG"

  # Run (kept same budgets for comparability)
  python -u agent_mvp/experiment_runner.py \
    --outdir "$OUTDIR" \
    --n-calib 1000 \
    --n-train-counts 5000 \
    --n-test-counts 2000 \
    --activity-log-every 1000 \
    --set arch=csnn --set device=cuda \
    --set dataset=cifar100:20 \
    --set input_channels=3 --set input_h=32 --set input_w=32 \
    --set c1_out=32 --set c1_kernel=5 --set c1_pad=2 --set c1_stride=1 \
    --set c2_out=64 --set c2_kernel=3 --set c2_pad=1 --set c2_stride=2 \
    --set greedy_enable=true --set greedy_n1=2500 \
    --set encoder=poisson --set poisson_deterministic=true --set poisson_rate_scale=${rate} --set encoder_rate_boost=1.0 \
    --set time=100 --set N=5000 \
    --set wta_enable=true \
    --set ei_enable=false \
    --set adapt_thresh_enable=true --set theta_plus=0.05 --set tau_theta=10000 \
    --set w_norm_enable=true --set w_norm_target=12.5 \
    --set homeo_enable=false \
    --set activity_check_after=1000 --set activity_min_spikes_win_mean=1000.0 \
    2>&1 | tee -a "$LOG"

  echo "=== rate_scale=${rate} finished at $(date -Is) ===" | tee -a "$LOG"

  # Extract run_id from the runner's final JSON block
  run_id=$(python3 - <<PY
import re, pathlib
text = pathlib.Path("$LOG").read_text(encoding="utf-8", errors="ignore")
m = re.findall(r'"run_id"\s*:\s*"([^"]+)"', text)
print(m[-1] if m else "")
PY
)

  if [[ -z "$run_id" ]]; then
    echo "WARN: couldn't parse run_id from $LOG" | tee -a "$LOG"
    continue
  fi

  status_path="$OUTDIR/$run_id/status.json"
  if [[ ! -f "$status_path" ]]; then
    echo "WARN: missing status.json at $status_path" | tee -a "$LOG"
    continue
  fi

  # Parse metrics
  metrics=$(python3 - <<PY
import json
p = "$status_path"
with open(p, "r", encoding="utf-8") as f:
    s = json.load(f)
status = s.get("status")
stage = s.get("stage")
acc = s.get("best_readout_acc")
sp = None
try:
    sp = s.get("train_summary", {}).get("spikes_per_sample")
except Exception:
    sp = None
lm = s.get("label_map_summary", {})
assigned = lm.get("assigned_neurons")
total = lm.get("total_neurons")
print(f"status={status} stage={stage} acc={acc} spikes_per_sample={sp} assigned={assigned}/{total}")
PY
)

  # Append to experiments log
  {
    echo ""
    echo "### 2026-05-04 — sweep WTA-only (2-layer greedy, no EI, no homeo)"
    echo "- run_id: ${run_id}"
    echo "- rate_scale: ${rate} | budgets: n_calib=1000 n_train_counts=5000 n_test_counts=2000"
    echo "- ${metrics}"
    echo "- log: ${LOG}"
  } >> "$EXPERIMENTS_MD"

done

echo "Sweep done at $(date -Is)"