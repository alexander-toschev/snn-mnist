#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/projects/snn-mnist

source /home/alex/anaconda3/etc/profile.d/conda.sh
conda activate bindsnet

OUTDIR="runs_csnn"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUTDIR/night_homeostasis_fashion_${STAMP}.log"
SUMMARY_JSONL="$OUTDIR/night_homeostasis_fashion_${STAMP}.summary.jsonl"

mkdir -p "$OUTDIR"

echo "Night plan started at $(date -Is)" | tee -a "$LOG"

dataset=fashion
common=(
  --outdir "$OUTDIR"
  --n-calib 1000
  --n-train-counts 5000
  --n-test-counts 5000
  --activity-log-every 1000
  --set arch=csnn --set device=cuda
  --set dataset=$dataset
  --set encoder=poisson --set poisson_deterministic=true --set poisson_rate_scale=0.006
  --set time=100 --set N=5000
  --set ei_enable=true --set ei_exc=22.5 --set ei_inh=120
  --set adapt_thresh_enable=true --set theta_plus=0.05 --set tau_theta=10000
  --set w_norm_enable=true --set w_norm_target=12.5
)

latest_run_dir() {
  ls -1t "$OUTDIR" | grep -E '^20[0-9]{6}T[0-9]{6}Z_' | head -n 1 || true
}

run_one() {
  local name="$1"; shift
  echo "\n=== $name ===" | tee -a "$LOG"
  local before after
  before=$(latest_run_dir)
  echo "before=$before" | tee -a "$LOG"

  # Run
  python agent_mvp/experiment_runner.py "${common[@]}" "$@" | tee -a "$LOG"

  after=$(latest_run_dir)
  echo "after=$after" | tee -a "$LOG"
  if [[ -n "$after" && -f "$OUTDIR/$after/summary.json" ]]; then
    python - <<PY >> "$SUMMARY_JSONL"
import json
p="$OUTDIR/$after/summary.json"
d=json.load(open(p))
d["night_name"]="$name"
print(json.dumps(d, ensure_ascii=False))
PY
    echo "summary: $OUTDIR/$after/summary.json" | tee -a "$LOG"
  else
    echo "WARN: no summary.json found for $after" | tee -a "$LOG"
  fi
}

# 1) Homeostasis tuned near baseline
run_one "homeo_target7000_gain0.15" \
  --set homeo_enable=true \
  --set homeo_spikes_lo=5500 --set homeo_spikes_hi=8000 --set homeo_spikes_target=7000 \
  --set homeo_update_every=50 --set homeo_warmup=500 --set homeo_gain=0.15 \
  --set homeo_rate_mul_min=0.5 --set homeo_rate_mul_max=3.0 \
  --set homeo_ema_alpha=0.005

run_one "homeo_target7000_gain0.10" \
  --set homeo_enable=true \
  --set homeo_spikes_lo=5500 --set homeo_spikes_hi=8000 --set homeo_spikes_target=7000 \
  --set homeo_update_every=50 --set homeo_warmup=500 --set homeo_gain=0.10 \
  --set homeo_rate_mul_min=0.5 --set homeo_rate_mul_max=3.0 \
  --set homeo_ema_alpha=0.005

run_one "homeo_target7500_gain0.10" \
  --set homeo_enable=true \
  --set homeo_spikes_lo=6000 --set homeo_spikes_hi=8500 --set homeo_spikes_target=7500 \
  --set homeo_update_every=50 --set homeo_warmup=500 --set homeo_gain=0.10 \
  --set homeo_rate_mul_min=0.5 --set homeo_rate_mul_max=3.0 \
  --set homeo_ema_alpha=0.005

run_one "homeo_target8000_gain0.10" \
  --set homeo_enable=true \
  --set homeo_spikes_lo=6500 --set homeo_spikes_hi=9500 --set homeo_spikes_target=8000 \
  --set homeo_update_every=50 --set homeo_warmup=500 --set homeo_gain=0.10 \
  --set homeo_rate_mul_min=0.5 --set homeo_rate_mul_max=3.0 \
  --set homeo_ema_alpha=0.005

# 2) Control (no homeostasis)
run_one "baseline_no_homeo" \
  --set homeo_enable=false

echo "\nNight plan finished at $(date -Is)" | tee -a "$LOG"
echo "Summary JSONL: $SUMMARY_JSONL" | tee -a "$LOG"
