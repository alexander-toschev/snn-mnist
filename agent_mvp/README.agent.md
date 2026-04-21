# SNN-MNIST agent MVP

Это минимальная обвязка для автономных экспериментов поверх вашего проекта.

## Что делает

- берёт `Cfg`-параметры;
- запускает `run_experiment(cfg)`;
- при желании строит `label_map`;
- гоняет `eval_readouts_from_net(...)`;
- сохраняет `model.pt`, `summary.json`, `status.json`;
- ведёт общий `registry.jsonl`;
- умеет предлагать и запускать серию конфигов по policy-файлу.

## Файлы

- `experiment_runner.py` — один запуск train/eval с сохранением артефактов.
- `agent_loop.py` — простой агент, который предлагает или запускает серию экспериментов.
- `agent_policy.example.json` — пример policy под текущий SNN-MNIST.
- `agent_common.py` — общие утилиты.

## Быстрый старт

Из корня репозитория:

```bash
python agent_mvp/experiment_runner.py \
  --outdir runs_agent \
  --set device=cuda \
  --set N=60000 \
  --set time=200 \
  --set n_hidden=100 \
  --set encoder=poisson \
  --set poisson_rate_scale=0.011 \
  --set inhib_strength=0.705 \
  --set thresh_init=0.38
```

С policy-файлом:

```bash
python agent_mvp/agent_loop.py \
  --policy agent_mvp/agent_policy.example.json \
  --outdir runs_agent \
  --budget 3 \
  --mode suggest
```

```bash
python agent_mvp/agent_loop.py \
  --policy agent_mvp/agent_policy.example.json \
  --outdir runs_agent \
  --budget 3 \
  --mode run
```

## Что появится на диске

```text
runs_agent/
  registry.jsonl
  last_agent_session.json
  _queue/
    candidate_001_....json
  2026-..._<hash>/
    model.pt
    label_map.pt
    summary.json
    status.json
```

## Как подключить к OpenClaw

Самый простой режим — не управлять Jupyter UI напрямую.
OpenClaw должен запускать обычные команды внутри WSL, например:

```bash
python /path/to/snn-mnist/agent_mvp/agent_loop.py \
  --policy /path/to/snn-mnist/agent_mvp/agent_policy.example.json \
  --outdir /path/to/snn-mnist/runs_agent \
  --budget 2 \
  --mode run
```

## Что я бы делал дальше

1. Добавил отдельные policy-файлы: `policy_quick_smoke`, `policy_energy_pareto`, `policy_latency_only`.
2. Ввел stop-guards не только по ошибкам запуска, но и по качеству: например, останавливать серию, если `best_readout_acc < 0.15` после нескольких попыток.
3. Добавил экспорт краткого markdown-отчёта по лучшим запускам из `registry.jsonl`.
4. Для рискованных изменений архитектуры вынес шаблоны в отдельный слой `templates`, а прямую правку кода — только в отдельную ветку.
