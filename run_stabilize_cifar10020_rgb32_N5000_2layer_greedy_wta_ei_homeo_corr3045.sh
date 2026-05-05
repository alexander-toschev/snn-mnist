#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/projects/snn-mnist

source /home/alex/anaconda3/etc/profile.d/conda.sh
conda activate bindsnet

OUTDIR="runs_csnn"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUTDIR/stabilize_cifar10020_rgb32_N5000_2layer_greedy_wta_ei_homeo_corr3045_${STAMP}.log"
mkdir -p "$OUTDIR"

echo "Run STABILIZE CIFAR100:20 RGB32 N=5000 2layer greedy+WTA+EI+HOMEOSTASIS corridor(30k-45k) started at $(date -Is)" | tee -a "$LOG"

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
  --set c2_out=64 --set c2_kernel=3 --set c2_pad=1 --set c2_stride=2 \
  --set greedy_enable=true --set greedy_n1=2500 \
  --set encoder=poisson --set poisson_deterministic=true --set poisson_rate_scale=0.006 --set encoder_rate_boost=1.0 \
  --set time=100 --set N=5000 \
  --set wta_enable=true \
  --set ei_enable=true --set ei_exc=22.5 --set ei_inh=120.0 --set ei_inh_mult_2layer=0.01 \
  --set adapt_thresh_enable=true --set theta_plus=0.05 --set tau_theta=10000 \
  --set w_norm_enable=true --set w_norm_target=12.5 \
  --set homeo_enable=true \
  --set homeo_spikes_lo=30000 --set homeo_spikes_hi=45000 --set homeo_spikes_target=37500 \
  --set homeo_gain=0.5 --set homeo_ema_alpha=0.02 --set homeo_update_every=25 --set homeo_warmup=200 \
  --set homeo_rate_mul_min=0.05 --set homeo_rate_mul_max=3.0 \
  --set activity_check_after=1000 --set activity_min_spikes_win_mean=1000.0 \
  | tee -a "$LOG"

echo "Finished at $(date -Is)" | tee -a "$LOG"
