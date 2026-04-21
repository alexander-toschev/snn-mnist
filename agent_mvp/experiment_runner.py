from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from agent_common import (
    append_jsonl,
    best_readout,
    config_hash,
    ensure_dir,
    merge_dicts,
    parse_overrides,
    read_json,
    slugify,
    utc_now_iso,
    write_json,
)

# project imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn_mnist_net import Cfg, run_experiment, save_snn  # noqa: E402
from label_map import build_label_map, save_label_map  # noqa: E402
from evaluation import eval_readouts_from_net  # noqa: E402


SAFE_SUMMARY_METRICS = [
    "spikes_per_sample",
    "synops_per_sample",
    "v_updates_per_sample",
    "energy_proxy_per_sample",
    "winners_unique",
    "winner_HHI",
]


def build_cfg(overrides: Dict[str, Any]) -> Cfg:
    valid = {name for name in Cfg.__dataclass_fields__.keys()}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise ValueError(f"Unknown Cfg fields: {unknown}")
    return Cfg(**overrides)



def run_single_experiment(
    overrides: Dict[str, Any],
    outdir: str | Path,
    *,
    n_calib: int = 2000,
    n_train_counts: int = 60000,
    n_test_counts: int = 10000,
    skip_label_map: bool = False,
    skip_eval: bool = False,
    verbose: bool = True,
    progress: bool = True,
) -> Dict[str, Any]:
    cfg = build_cfg(overrides)
    outdir = ensure_dir(outdir)
    cfg_dict = dict(overrides)
    cfg_digest = config_hash(cfg_dict)
    run_id = f"{utc_now_iso().replace(':', '').replace('-', '')}_{cfg_digest}"
    run_dir = ensure_dir(outdir / run_id)

    status_path = run_dir / "status.json"
    registry_path = outdir / "registry.jsonl"

    base_payload: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "status": "running",
        "config_hash": cfg_digest,
        "cfg": cfg_dict,
        "run_dir": str(run_dir),
    }
    write_json(status_path, base_payload)

    try:
        train_summary, connection, lif_layer, net, encoder = run_experiment(
            cfg, verbose=verbose, progress=progress
        )

        label_map_path = None
        label_map_summary = None
        label_map = None
        if not skip_label_map:
            label_map = build_label_map(
                net,
                None,
                lif_layer,
                encoder,
                n_calib=n_calib,
                T=cfg.time,
                top_k=max(1, int(cfg.top_k) if int(cfg.top_k) > 0 else 3),
                seed=cfg.seed,
            )
            label_map_path = run_dir / "label_map.pt"
            save_label_map(
                str(label_map_path),
                label_map,
                meta={
                    "run_id": run_id,
                    "n_calib": n_calib,
                    "time": cfg.time,
                },
            )
            label_map_summary = {
                "assigned_neurons": int((label_map >= 0).sum().item()),
                "total_neurons": int(label_map.numel()),
            }

        eval_summary = None
        if not skip_eval:
            eval_summary = eval_readouts_from_net(
                net,
                lif_layer,
                encoder,
                cfg,
                label_map=label_map,
                n_train_counts=n_train_counts,
                n_test_counts=n_test_counts,
            )

        best_name, best_acc = best_readout(eval_summary)
        ckpt_path = run_dir / "model.pt"
        save_snn(
            str(ckpt_path),
            cfg,
            connection,
            lif_layer,
            train_summary=train_summary,
            eval_summary=eval_summary,
            notes={
                "run_id": run_id,
                "config_hash": cfg_digest,
                "label_map_summary": label_map_summary,
            },
        )

        payload: Dict[str, Any] = {
            **base_payload,
            "status": "ok",
            "finished_at": utc_now_iso(),
            "checkpoint_path": str(ckpt_path),
            "label_map_path": str(label_map_path) if label_map_path else None,
            "label_map_summary": label_map_summary,
            "best_readout_name": best_name,
            "best_readout_acc": best_acc,
            "train_summary": train_summary,
            "eval_summary": eval_summary,
            "summary": {
                k: train_summary.get(k)
                for k in SAFE_SUMMARY_METRICS
                if isinstance(train_summary, dict) and k in train_summary
            },
        }
        write_json(run_dir / "summary.json", payload)
        write_json(status_path, payload)
        append_jsonl(registry_path, payload)
        return payload
    except Exception as exc:  # pragma: no cover - defensive runtime wrapper
        err_payload = {
            **base_payload,
            "status": "failed",
            "finished_at": utc_now_iso(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "summary.json", err_payload)
        write_json(status_path, err_payload)
        append_jsonl(registry_path, err_payload)
        raise



def main() -> None:
    parser = argparse.ArgumentParser(description="Run one SNN experiment and persist artifacts.")
    parser.add_argument("--outdir", default="runs_agent", help="Root directory for run folders and registry.jsonl")
    parser.add_argument("--config-json", help="Path to JSON file with Cfg overrides")
    parser.add_argument("--set", action="append", default=[], help="Cfg override, e.g. --set time=300")
    parser.add_argument("--n-calib", type=int, default=2000)
    parser.add_argument("--n-train-counts", type=int, default=60000)
    parser.add_argument("--n-test-counts", type=int, default=10000)
    parser.add_argument("--skip-label-map", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    file_overrides = read_json(args.config_json) if args.config_json else {}
    cli_overrides = parse_overrides(args.set)
    overrides = merge_dicts(file_overrides, cli_overrides)
    if not overrides:
        raise SystemExit("No config overrides provided. Pass --config-json or --set key=value.")

    payload = run_single_experiment(
        overrides,
        outdir=args.outdir,
        n_calib=args.n_calib,
        n_train_counts=args.n_train_counts,
        n_test_counts=args.n_test_counts,
        skip_label_map=args.skip_label_map,
        skip_eval=args.skip_eval,
        verbose=not args.quiet,
        progress=not args.no_progress,
    )
    print(json.dumps({
        "run_id": payload["run_id"],
        "status": payload["status"],
        "best_readout_name": payload.get("best_readout_name"),
        "best_readout_acc": payload.get("best_readout_acc"),
        "run_dir": payload["run_dir"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
