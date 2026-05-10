#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/projects/snn-mnist

# Diverse10 accuracy push plan (2026-05-10)
# Stage A: capacity sweep @ N=5000, T=200, rate_scale=0.0035

runs=(
  ./run_cifar100_diverse10_rgb32_T200_N5000_2layer_greedy_localinhib_s085_rate0035.sh
  ./run_cifar100_diverse10_rgb32_T200_N5000_2layer_wide_c1_64_c2_128_greedy_gn1_2500_localinhib_s085_rate0035.sh
  ./run_cifar100_diverse10_rgb32_T200_N5000_2layer_greedy_gn1_5000_localinhib_s085_rate0035.sh
  ./run_cifar100_diverse10_rgb32_T200_N5000_2layer_wide_c1_64_c2_128_greedy_gn1_5000_localinhib_s085_rate0035.sh
)

for r in "${runs[@]}"; do
  echo "==> $(date -Is) running: $r"
  bash "$r"
  echo "==> $(date -Is) done: $r"
  echo
done

echo "All Stage A runs finished at $(date -Is)"
