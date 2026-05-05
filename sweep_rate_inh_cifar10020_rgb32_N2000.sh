#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/projects/snn-mnist

source /home/alex/anaconda3/etc/profile.d/conda.sh
conda activate bindsnet

OUTDIR="runs_csnn"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUTDIR/sweep_rate_inh_cifar10020_rgb32_N2000_${STAMP}.log"
SUMMARY_JSONL="$OUTDIR/sweep_rate_inh_cifar10020_rgb32_N2000_${STAMP}.summary.jsonl"
mkdir -p "$OUTDIR"

echo "Sweep CIFAR100:20 RGB32 (N=2000) started at $(date -Is)" | tee -a "$LOG"

dataset=cifar100:20
common=(
  --outdir "$OUTDIR"
  --n-calib 200
  --n-train-counts 1000
  --n-test-counts 1000
  --activity-log-every 500
  --set arch=csnn --set device=cuda
  --set dataset=$dataset
  --set input_channels=3 --set input_h=32 --set input_w=32
  --set c1_out=32 --set c1_kernel=5 --set c1_pad=0 --set c1_stride=1
  --set encoder=poisson --set poisson_deterministic=true
  --set time=100 --set N=2000
  --set ei_enable=true --set ei_exc=22.5
  --set adapt_thresh_enable=true --set theta_plus=0.05 --set tau_theta=10000
  --set w_norm_enable=true --set w_norm_target=12.5
  --set homeo_enable=true
  --set homeo_spikes_lo=6000 --set homeo_spikes_hi=8500 --set homeo_spikes_target=7500
  --set homeo_update_every=50 --set homeo_warmup=200 --set homeo_gain=0.10
  --set homeo_rate_mul_min=0.5 --set homeo_rate_mul_max=3.0
  --set homeo_ema_alpha=0.005
)

latest_run_dir() {
  ls -1t "$OUTDIR" | grep -E '^20[0-9]{6}T[0-9]{6}Z_' | head -n 1 || true
}

run_one() {
  local name="$1"; shift
  echo -e "\n=== $name ===" | tee -a "$LOG"
  local before after
  before=$(latest_run_dir)
  echo "before=$before" | tee -a "$LOG"

  python agent_mvp/experiment_runner.py "${common[@]}" "$@" | tee -a "$LOG"

  after=$(latest_run_dir)
  echo "after=$after" | tee -a "$LOG"
  if [[ -n "$after" && -f "$OUTDIR/$after/summary.json" ]]; then
    python - <<PY >> "$SUMMARY_JSONL"
import json
p="$OUTDIR/$after/summary.json"
d=json.load(open(p))
d["sweep_name"]="$name"
print(json.dumps(d, ensure_ascii=False))
PY
    echo "summary: $OUTDIR/$after/summary.json" | tee -a "$LOG"
  else
    echo "WARN: no summary.json found for $after" | tee -a "$LOG"
  fi
}

# Grid
for rate in 0.006 0.008 0.010 0.012; do
  for inh in 90 120 150; do
    run_one "rate_${rate}_inh_${inh}" --set poisson_rate_scale=${rate} --set ei_inh=${inh}
  done
done

echo -e "\nSweep finished at $(date -Is)" | tee -a "$LOG"
echo "Summary JSONL: $SUMMARY_JSONL" | tee -a "$LOG"
