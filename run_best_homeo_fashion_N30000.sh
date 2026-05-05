#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/projects/snn-mnist

source /home/alex/anaconda3/etc/profile.d/conda.sh
conda activate bindsnet

OUTDIR="runs_csnn"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUTDIR/run_best_homeo_fashion_N30000_${STAMP}.log"
SUMMARY_JSON="$OUTDIR/run_best_homeo_fashion_N30000_${STAMP}.summary.json"

mkdir -p "$OUTDIR"

echo "Run best-config (homeo fixed) N=30000 started at $(date -Is)" | tee -a "$LOG"

python agent_mvp/experiment_runner.py \
  --outdir "$OUTDIR" \
  --n-calib 1000 \
  --n-train-counts 5000 \
  --n-test-counts 5000 \
  --activity-log-every 1000 \
  --set arch=csnn --set device=cuda \
  --set dataset=fashion \
  --set encoder=poisson --set poisson_deterministic=true --set poisson_rate_scale=0.006 \
  --set time=100 --set N=30000 \
  --set ei_enable=true --set ei_exc=22.5 --set ei_inh=120 \
  --set adapt_thresh_enable=true --set theta_plus=0.05 --set tau_theta=10000 \
  --set w_norm_enable=true --set w_norm_target=12.5 \
  --set homeo_enable=true \
  --set homeo_spikes_lo=6000 --set homeo_spikes_hi=8500 --set homeo_spikes_target=7500 \
  --set homeo_update_every=50 --set homeo_warmup=500 --set homeo_gain=0.10 \
  --set homeo_rate_mul_min=0.5 --set homeo_rate_mul_max=3.0 \
  --set homeo_ema_alpha=0.005 \
  | tee -a "$LOG"

echo "Finished at $(date -Is)" | tee -a "$LOG"

# Try to capture last run summary.json (in run dir)
LATEST=$(ls -1t "$OUTDIR" | grep -E '^20[0-9]{6}T[0-9]{6}Z_' | head -n 1 || true)
if [[ -n "$LATEST" && -f "$OUTDIR/$LATEST/summary.json" ]]; then
  cp "$OUTDIR/$LATEST/summary.json" "$SUMMARY_JSON"
  echo "Summary: $OUTDIR/$LATEST/summary.json" | tee -a "$LOG"
  echo "Copied: $SUMMARY_JSON" | tee -a "$LOG"
fi
