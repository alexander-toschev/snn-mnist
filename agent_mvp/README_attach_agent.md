# Agent attach bundle for SNN-MNIST

## What to copy into your repo

Copy these files into `agent_mvp/`:
- `agent_loop.py`
- `agent_policy_stage1_quick.json`
- `agent_policy_stage2_full.json`
- `print_agent_leaderboard.py`

Copy `run_agent_conda.sh` into repo root.

Use your fixed `snn_mnist_net.py` with CUDA-safe device handling before running the agent.

## Recommended workflow

### 1. Quick search

```bash
bash run_agent_conda.sh /mnt/d/projects/snn-mnist agent_mvp/agent_policy_stage1_quick.json 4 runs_agent_quick run
```

### 2. View leaderboard

```bash
conda activate bindsnet
cd /mnt/d/projects/snn-mnist
python agent_mvp/print_agent_leaderboard.py --registry runs_agent_quick/registry.jsonl --top 10
```

### 3. Full search around promising configs

```bash
bash run_agent_conda.sh /mnt/d/projects/snn-mnist agent_mvp/agent_policy_stage2_full.json 3 runs_agent_full run
```

## Good starting practice

- Stage 1 first: quick screening
- Then inspect best runs
- Then Stage 2 only around good regions
- Do not start with huge budget while debugging

## Suggested first budgets

- quick: 3-4 runs
- full: 2-3 runs

