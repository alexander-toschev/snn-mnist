from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import selectors
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from agent_common import config_hash, load_jsonl, read_json, utc_now_iso, write_json

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_RUNNER = Path(__file__).resolve().parent / "experiment_runner.py"


def linear_score(result: Dict[str, Any], weights: Dict[str, float]) -> float | None:
    total = 0.0
    seen = False
    for metric, weight in weights.items():
        value = result.get(metric)
        if isinstance(value, (int, float)):
            total += float(value) * float(weight)
            seen = True
    return total if seen else None


def flatten_result(row: Dict[str, Any]) -> Dict[str, Any]:
    flat = dict(row)
    for group in ("summary", "train_summary", "eval_summary"):
        payload = row.get(group)
        if isinstance(payload, dict):
            for k, v in payload.items():
                flat[k] = v
    return flat


def load_policy(path: str | Path) -> Dict[str, Any]:
    policy = read_json(path)
    policy.setdefault("base_overrides", {})
    policy.setdefault("seed_queue", [])
    policy.setdefault("search_space", {})
    policy.setdefault("score_weights", {})
    policy.setdefault("guards", {})
    policy.setdefault("runner", {})
    return policy


def config_from_template(policy: Dict[str, Any], template_name: str) -> Dict[str, Any]:
    templates = policy.get("templates", {})
    if template_name not in templates:
        raise KeyError(f"Unknown template: {template_name}")
    return deepcopy(templates[template_name])


def iter_seed_configs(policy: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for item in policy.get("seed_queue", []):
        if isinstance(item, str):
            yield config_from_template(policy, item)
        else:
            yield deepcopy(item)


def pick_random_config(policy: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    cfg = deepcopy(policy.get("base_overrides", {}))
    for key, choices in policy.get("search_space", {}).items():
        if choices:
            cfg[key] = rng.choice(list(choices))
    return cfg


def mutate_config(base_cfg: Dict[str, Any], policy: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    mutated = deepcopy(base_cfg)
    search_space = policy.get("search_space", {})
    mutable_keys = [k for k, v in search_space.items() if v]
    if not mutable_keys:
        return mutated
    n_changes = min(max(1, policy.get("mutation_changes", 2)), len(mutable_keys))
    for key in rng.sample(mutable_keys, k=n_changes):
        choices = list(search_space[key])
        if len(choices) == 1:
            mutated[key] = choices[0]
            continue
        current = mutated.get(key)
        alternatives = [c for c in choices if c != current]
        mutated[key] = rng.choice(alternatives or choices)
    return mutated


def existing_hashes(registry_rows: List[Dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for row in registry_rows:
        cfg = row.get("cfg")
        if isinstance(cfg, dict):
            hashes.add(config_hash(cfg))
        cfg_hash = row.get("config_hash")
        if isinstance(cfg_hash, str):
            hashes.add(cfg_hash)
    return hashes


def propose_configs(policy: Dict[str, Any], registry_rows: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    rng = random.Random(int(policy.get("random_seed", 42)))
    seen = existing_hashes(registry_rows)
    proposals: List[Dict[str, Any]] = []

    def _accept(cfg: Dict[str, Any]) -> bool:
        digest = config_hash(cfg)
        if digest in seen:
            return False
        seen.add(digest)
        proposals.append(cfg)
        return True

    for cfg in iter_seed_configs(policy):
        cfg = {**policy.get("base_overrides", {}), **cfg}
        if _accept(cfg) and len(proposals) >= budget:
            return proposals

    successful = [flatten_result(r) for r in registry_rows if r.get("status") == "ok"]
    score_weights = policy.get("score_weights", {})
    best_cfg = None
    if successful and score_weights:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in successful:
            score = linear_score(row, score_weights)
            if score is not None:
                scored.append((score, row))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best_cfg = deepcopy(scored[0][1].get("cfg", {}))

    while len(proposals) < budget:
        if best_cfg and rng.random() < float(policy.get("mutate_best_probability", 0.7)):
            cfg = mutate_config(best_cfg, policy, rng)
        else:
            cfg = pick_random_config(policy, rng)
        cfg = {**policy.get("base_overrides", {}), **cfg}
        _accept(cfg)
        if len(seen) > 100000:
            break

    return proposals


def _compact_cfg(cfg: Dict[str, Any]) -> str:
    preferred = [
        "device", "encoder", "time", "n_hidden", "N", "warmup_N",
        "poisson_rate_scale", "encoder_rate_boost", "latency_x_min",
        "inhib_strength", "top_k", "thresh_init", "seed",
    ]
    ordered = []
    seen = set()
    for key in preferred:
        if key in cfg:
            ordered.append((key, cfg[key]))
            seen.add(key)
    for key in sorted(cfg):
        if key not in seen:
            ordered.append((key, cfg[key]))
    return ", ".join(f"{k}={v}" for k, v in ordered)


def _stream_subprocess(cmd: List[str], cwd: str) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    selector = selectors.DefaultSelector()
    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []

    assert proc.stdout is not None
    assert proc.stderr is not None
    selector.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, data="stderr")

    while selector.get_map():
        for key, _ in selector.select(timeout=0.2):
            stream = key.fileobj
            channel = key.data
            line = stream.readline()
            if line == "":
                selector.unregister(stream)
                continue
            if channel == "stdout":
                stdout_chunks.append(line)
                print(line, end="", flush=True)
            else:
                stderr_chunks.append(line)
                print(line, end="", file=sys.stderr, flush=True)

        if proc.poll() is not None:
            # Drain remaining buffered lines.
            for stream, bucket in ((proc.stdout, stdout_chunks), (proc.stderr, stderr_chunks)):
                if stream is None:
                    continue
                rest = stream.read()
                if rest:
                    bucket.append(rest)
                    target = sys.stdout if bucket is stdout_chunks else sys.stderr
                    print(rest, end="", file=target, flush=True)
            break

    return proc.wait(), "".join(stdout_chunks), "".join(stderr_chunks)


def run_budget(
    policy: Dict[str, Any],
    proposals: List[Dict[str, Any]],
    outdir: str | Path,
    *,
    live_output: bool = True,
) -> List[Dict[str, Any]]:
    outdir = Path(outdir)
    session_rows: List[Dict[str, Any]] = []
    failures = 0
    max_failures = int(policy.get("guards", {}).get("max_failures", 3))

    total = len(proposals)
    session_started = time.monotonic()

    for idx, cfg in enumerate(proposals, start=1):
        cfg_path = outdir / "_queue" / f"candidate_{idx:03d}_{config_hash(cfg)}.json"
        write_json(cfg_path, cfg)
        cmd = [
            sys.executable,
            str(EXPERIMENT_RUNNER),
            "--outdir",
            str(outdir),
            "--config-json",
            str(cfg_path),
        ]
        if policy.get("runner", {}).get("quiet", False):
            cmd.append("--quiet")
        if policy.get("runner", {}).get("no_progress", False):
            cmd.append("--no-progress")
        if "n_calib" in policy.get("runner", {}):
            cmd += ["--n-calib", str(policy["runner"]["n_calib"])]
        if "n_train_counts" in policy.get("runner", {}):
            cmd += ["--n-train-counts", str(policy["runner"]["n_train_counts"])]
        if "n_test_counts" in policy.get("runner", {}):
            cmd += ["--n-test-counts", str(policy["runner"]["n_test_counts"])]
        if policy.get("runner", {}).get("skip_label_map", False):
            cmd.append("--skip-label-map")
        if policy.get("runner", {}).get("skip_eval", False):
            cmd.append("--skip-eval")

        run_started_iso = utc_now_iso()
        run_started = time.monotonic()
        cfg_digest = config_hash(cfg)

        print("\n" + "=" * 100, flush=True)
        print(f"[agent] run {idx}/{total} started at {run_started_iso}", flush=True)
        print(f"[agent] config_hash={cfg_digest}", flush=True)
        print(f"[agent] cfg: {_compact_cfg(cfg)}", flush=True)
        print(f"[agent] cmd: {' '.join(cmd)}", flush=True)
        print("-" * 100, flush=True)

        if live_output:
            returncode, stdout_text, stderr_text = _stream_subprocess(cmd, cwd=str(ROOT))
        else:
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            returncode, stdout_text, stderr_text = proc.returncode, proc.stdout, proc.stderr

        elapsed = time.monotonic() - run_started
        row = {
            "queued_at": run_started_iso,
            "cfg": cfg,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "returncode": returncode,
            "elapsed_sec": round(elapsed, 3),
        }
        session_rows.append(row)

        if returncode == 0:
            failures = 0
            print("-" * 100, flush=True)
            print(f"[agent] run {idx}/{total} finished OK in {elapsed:.1f}s", flush=True)
        else:
            failures += 1
            print("-" * 100, flush=True)
            print(
                f"[agent] run {idx}/{total} FAILED in {elapsed:.1f}s | returncode={returncode} | consecutive_failures={failures}",
                flush=True,
            )
            if failures >= max_failures:
                print(f"[agent] stopping early: reached max_failures={max_failures}", flush=True)
                break

    total_elapsed = time.monotonic() - session_started
    ok_count = sum(1 for r in session_rows if r.get("returncode") == 0)
    print("\n" + "=" * 100, flush=True)
    print(
        f"[agent] session finished | runs={len(session_rows)} ok={ok_count} failed={len(session_rows)-ok_count} elapsed={total_elapsed:.1f}s",
        flush=True,
    )
    print("=" * 100, flush=True)
    return session_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple autonomous loop for the SNN-MNIST experiments.")
    parser.add_argument("--policy", required=True, help="Path to JSON policy file")
    parser.add_argument("--outdir", default="runs_agent")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--mode", choices=["suggest", "run"], default="suggest")
    parser.add_argument("--live-output", dest="live_output", action="store_true", default=True,
                        help="Stream child experiment output in real time (default: on)")
    parser.add_argument("--no-live-output", dest="live_output", action="store_false",
                        help="Do not stream child experiment output")
    args = parser.parse_args()

    policy = load_policy(args.policy)
    outdir = Path(args.outdir)
    registry_rows = load_jsonl(outdir / "registry.jsonl")
    proposals = propose_configs(policy, registry_rows, budget=args.budget)

    if args.mode == "suggest":
        print(json.dumps({
            "policy": str(args.policy),
            "budget": args.budget,
            "proposals": proposals,
        }, ensure_ascii=False, indent=2))
        return

    session_info = run_budget(policy, proposals, outdir=outdir, live_output=args.live_output)
    write_json(outdir / "last_agent_session.json", {
        "started_at": utc_now_iso(),
        "policy": str(args.policy),
        "budget": args.budget,
        "proposals": proposals,
        "results": session_info,
    })
    print(json.dumps({
        "policy": str(args.policy),
        "budget": args.budget,
        "executed": len(session_info),
        "outdir": str(outdir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
