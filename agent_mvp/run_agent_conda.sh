#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$PWD}"
POLICY_PATH="${2:-agent_mvp/agent_policy_stage1_quick.json}"
BUDGET="${3:-4}"
OUTDIR="${4:-runs_agent}"
MODE="${5:-run}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "conda.sh not found under ~/miniconda3 or ~/anaconda3" >&2
  exit 1
fi

conda activate bindsnet
cd "$PROJECT_DIR"

# Avoid BrokenPipeError from tqdm when running detached.
# Force unbuffered output and redirect stdout/stderr to a log file inside OUTDIR.
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/agent_loop_$(date -u +%Y%m%dT%H%M%SZ).log"

PYTHONUNBUFFERED=1 python -u agent_mvp/agent_loop.py \
  --policy "$POLICY_PATH" \
  --outdir "$OUTDIR" \
  --budget "$BUDGET" \
  --mode "$MODE" \
  >"$LOGFILE" 2>&1

echo "[run_agent_conda] logs: $LOGFILE"
