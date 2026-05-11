#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/projects/snn-mnist

source /home/alex/anaconda3/etc/profile.d/conda.sh
conda activate bindsnet

DATASET="cifar100:0,1,2,3,5,8,13,14,17,19"

OUTDIR="runs_csnn"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUTDIR/${DATASET//[: ,]/_}_rgb32_T200_N20000_2layer_WIDEc1_64_c2_128_gn1_2500_localinhib_s085_rate0035_${STAMP}.log"
mkdir -p "$OUTDIR"

echo "Run ${DATASET} RGB32 (diverse10) 2-layer WIDE(c1=64,c2=128) T=200 N=20000 greedy_n1=2500 local_inhib=0.85 rate_scale=0.0035 started at $(date -Is)" | tee -a "$LOG"

python agent_mvp/experiment_runner.py \
  --outdir "$OUTDIR" \
  --n-calib 1000 \
  --n-train-counts 20000 \
  --n-test-counts 1000 \
  --activity-log-every 1000 \
  --set arch=csnn --set device=cuda \
  --set dataset="$DATASET" \
  --set input_mode=rgb --set input_channels=3 --set input_h=32 --set input_w=32 \
  --set c1_out=64 --set c1_kernel=5 --set c1_pad=2 --set c1_stride=1 \
  --set c2_out=128 --set c2_kernel=3 --set c2_pad=1 --set c2_stride=2 \
  --set greedy_enable=true --set greedy_n1=2500 \
  --set encoder=poisson --set poisson_deterministic=true --set poisson_rate_scale=0.0035 --set encoder_rate_boost=1.0 \
  --set time=200 --set N=20000 \
  --set local_inhib_enable=true --set local_inhib_strength=0.85 \
  --set wta_enable=false \
  --set ei_enable=false \
  --set adapt_thresh_enable=true --set theta_plus=0.05 --set tau_theta=10000 \
  --set w_norm_enable=true --set w_norm_target=12.5 \
  --set homeo_enable=false \
  --set activity_check_after=1000 --set activity_min_spikes_win_mean=1000.0 \
  | tee -a "$LOG"

echo "Finished at $(date -Is)" | tee -a "$LOG"
