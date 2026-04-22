from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def score(row: dict) -> float:
    acc = float(row.get("best_readout_acc") or 0.0)
    summary = row.get("summary") or {}
    energy = float(summary.get("energy_proxy_per_sample") or 0.0)
    synops = float(summary.get("synops_per_sample") or 0.0)
    spikes = float(summary.get("spikes_per_sample") or 0.0)
    return acc - 0.00025 * energy - 0.00002 * synops + 0.0005 * spikes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="runs_agent/registry.jsonl")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rows = [r for r in load_jsonl(Path(args.registry)) if r.get("status") == "ok"]
    rows.sort(key=score, reverse=True)

    print(f"successful runs: {len(rows)}")
    for row in rows[: args.top]:
        summary = row.get("summary") or {}
        print(json.dumps({
            "run_id": row.get("run_id"),
            "best_readout_name": row.get("best_readout_name"),
            "best_readout_acc": row.get("best_readout_acc"),
            "spikes_per_sample": summary.get("spikes_per_sample"),
            "synops_per_sample": summary.get("synops_per_sample"),
            "energy_proxy_per_sample": summary.get("energy_proxy_per_sample"),
            "cfg": row.get("cfg"),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
