#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/projects/snn-mnist

source /home/alex/anaconda3/etc/profile.d/conda.sh
conda activate bindsnet

OUTDIR="runs_csnn"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUTDIR/sweep_cifar10020_rgb32_N5000_pad2_ei_inh_${STAMP}.log"
mkdir -p "$OUTDIR"

echo "Sweep CIFAR100:20 RGB32 N=5000 pad2 EI inh=[80,120,160] started at $(date -Is)" | tee -a "$LOG"

for INH in 80 120 160; do
  echo "=== pad2 + EI (ei_inh=${INH}) ===" | tee -a "$LOG"
  python agent_mvp/experiment_runner.py \
    --outdir "$OUTDIR" \
    --n-calib 1000 \
    --n-train-counts 5000 \
    --n-test-counts 2000 \
    --activity-log-every 1000 \
    --set arch=csnn --set device=cuda \
    --set dataset=cifar100:20 \
    --set input_channels=3 --set input_h=32 --set input_w=32 \
    --set c1_out=32 --set c1_kernel=5 --set c1_pad=2 --set c1_stride=1 \
    --set encoder=poisson --set poisson_deterministic=true --set poisson_rate_scale=0.008 --set encoder_rate_boost=1.0 \
    --set time=100 --set N=5000 \
    --set ei_enable=true --set ei_exc=22.5 --set ei_inh=${INH} \
    --set wta_enable=false \
    --set adapt_thresh_enable=true --set theta_plus=0.05 --set tau_theta=10000 \
    --set w_norm_enable=true --set w_norm_target=12.5 \
    --set homeo_enable=true \
    --set homeo_spikes_lo=6000 --set homeo_spikes_hi=8500 --set homeo_spikes_target=7500 \
    --set homeo_update_every=50 --set homeo_warmup=500 --set homeo_gain=0.10 \
    --set homeo_rate_mul_min=0.5 --set homeo_rate_mul_max=3.0 \
    --set homeo_ema_alpha=0.005 \
    | tee -a "$LOG"
done

echo "Sweep finished at $(date -Is)" | tee -a "$LOG"
