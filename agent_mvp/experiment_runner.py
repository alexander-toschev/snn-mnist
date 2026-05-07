from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict
import time

import torch

# project imports
ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = Path(__file__).resolve().parent
for p in (ROOT, AGENT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

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

from snn_mnist_net import Cfg, run_experiment, save_snn  # noqa: E402
from csnn_mnist_net import (
    CSCfg,
    build_csnn,
    build_encoder_from_cfg as build_csnn_encoder,
    _spikes_flat_to_hw,
    load_csnn_weights_into,
)  # noqa: E402
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


def build_cs_cfg(overrides: Dict[str, Any]) -> CSCfg:
    valid = {name for name in CSCfg.__dataclass_fields__.keys()}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise ValueError(f"Unknown CSCfg fields: {unknown}")
    return CSCfg(**overrides)



def run_single_experiment(
    overrides: Dict[str, Any],
    outdir: str | Path,
    *,
    n_calib: int = 2000,
    n_train_counts: int = 60000,
    n_test_counts: int = 10000,
    activity_log_every: int = 1000,
    skip_label_map: bool = False,
    skip_eval: bool = False,
    verbose: bool = True,
    progress: bool = True,
) -> Dict[str, Any]:
    arch = str(overrides.get("arch", "fc")).lower()
    if arch == "csnn":
        cfg = build_cs_cfg({k: v for k, v in overrides.items() if k != "arch"})
    else:
        cfg = build_cfg({k: v for k, v in overrides.items() if k != "arch"})
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

    live_log_path = run_dir / "live.log"
    def _status_update(**fields):
        payload = {**read_json(status_path), **fields}
        payload["updated_at"] = utc_now_iso()
        write_json(status_path, payload)
        try:
            with open(live_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({k: payload.get(k) for k in ("status","stage","i","n","pct","eta_seconds") if k in payload}, ensure_ascii=False) + "\n")
        except BrokenPipeError:
            pass

    try:
        if arch == "csnn":
            # Minimal CSNN training loop (v0): 1 conv layer + STDP.
            from torchvision import transforms
            from bindsnet.network.monitors import Monitor
            from datasets_vision import make_vision_datasets

            net, input_layer, lif_layer, connection = build_csnn(cfg)
            encoder = build_csnn_encoder(cfg)
            device = cfg.torch_device()
            # Online homeostasis controller (optional): adjusts Poisson rate_scale to keep spikes/sample in band.
            homeo = None
            try:
                from activity_homeostasis import HomeostasisCfg, SpikesHomeostasis
                homeo_cfg = HomeostasisCfg(
                    enable=bool(getattr(cfg, "homeo_enable", False)),
                    spikes_lo=float(getattr(cfg, "homeo_spikes_lo", 3000.0)),
                    spikes_hi=float(getattr(cfg, "homeo_spikes_hi", 5000.0)),
                    spikes_target=float(getattr(cfg, "homeo_spikes_target", 4000.0)),
                    ema_alpha=float(getattr(cfg, "homeo_ema_alpha", 0.01)),
                    update_every=int(getattr(cfg, "homeo_update_every", 25)),
                    warmup=int(getattr(cfg, "homeo_warmup", 200)),
                    gain=float(getattr(cfg, "homeo_gain", 0.25)),
                    rate_mul_min=float(getattr(cfg, "homeo_rate_mul_min", 0.3)),
                    rate_mul_max=float(getattr(cfg, "homeo_rate_mul_max", 3.0)),
                )
                # Only makes sense for Poisson encoder which exposes rate_scale.
                if hasattr(encoder, "rate_scale"):
                    homeo = SpikesHomeostasis(homeo_cfg, base_rate_scale=float(getattr(cfg, "poisson_rate_scale", 0.0)))
            except Exception:
                homeo = None
            # Ensure the conv input adapter is available even if label_map is skipped.
            from csnn_mnist_net import _spikes_flat_to_hw

            # Input transform
            # CIFAR returns PIL images; transforms are applied before ToTensor.
            mode = str(getattr(cfg, "input_mode", "rgb") or "rgb").lower()
            if mode in {"gray1", "grayscale1", "mono"}:
                transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ])
            elif mode in {"gray3", "grayscale3"}:
                # Grayscale but keep 3 channels (replicated) to preserve Conv weight shapes.
                transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([transforms.ToTensor()])
            ds_train, _ds_test = make_vision_datasets(dataset=str(getattr(cfg, "dataset", "mnist")), root="./data", transform=transform)
            ds = ds_train
            n_train = min(int(getattr(cfg, "N", 12000)), len(ds))
            T = int(cfg.time)

            # Shuffle training order for better class mixing (important for CIFAR etc.).
            # Controlled by cfg.shuffle (default True).
            shuffle = bool(getattr(cfg, "shuffle", True))
            if shuffle:
                g = torch.Generator(device="cpu")
                try:
                    g.manual_seed(int(getattr(cfg, "seed", 42)))
                except Exception:
                    g.manual_seed(42)
                order = torch.randperm(len(ds), generator=g).tolist()
            else:
                order = list(range(len(ds)))

            mon_X = Monitor(input_layer, state_vars=("s",), time=T)
            mon_H = Monitor(lif_layer, state_vars=("s",), time=T)
            net.add_monitor(mon_X, name="mon_X")
            net.add_monitor(mon_H, name="mon_H")

            # Optional per-layer monitors for multi-layer CSNN (Conv1/2/3), for debugging dynamics.
            mon_layers = {}
            last_layer_name = None
            try:
                csnn_lifs = list(getattr(net, "_csnn_lifs", []) or [])
                if len(csnn_lifs) >= 2:
                    for li, lif in enumerate(csnn_lifs):
                        lname = f"Conv{li+1}"
                        mon = Monitor(lif, state_vars=("s",), time=T)
                        net.add_monitor(mon, name=f"mon_{lname}")
                        mon_layers[lname] = mon
                    if mon_layers:
                        last_layer_name = list(mon_layers.keys())[-1]
            except Exception:
                mon_layers = {}
                last_layer_name = None

            # Multi-layer CSNN support: build_csnn attaches net._csnn_conns in forward order.
            csnn_conns = list(getattr(net, "_csnn_conns", [])) or [connection]
            csnn_nu_orig = []
            csnn_nu_zero = []
            for c in csnn_conns:
                ur = getattr(c, "update_rule", None)
                nu = getattr(ur, "nu", None)
                if isinstance(nu, (tuple, list)) and len(nu) == 2:
                    csnn_nu_orig.append((nu[0].clone(), nu[1].clone()))
                    z0 = nu[0].detach().clone().fill_(0.0)
                    z1 = nu[1].detach().clone().fill_(0.0)
                    csnn_nu_zero.append((z0, z1))
                else:
                    csnn_nu_orig.append(None)
                    csnn_nu_zero.append(None)

            S_out = 0
            last_report = time.time()
            # For stable activity logging (windowed stats)
            act_win_sum = 0
            act_win_n = 0
            act_win_min = None
            act_win_max = None

            # Winner stats for last layer (argmax over per-neuron spike counts per sample)
            winner_counts_total = None
            winner_counts_win = None
            try:
                winner_counts_total = torch.zeros((int(lif_layer.n),), dtype=torch.long)
                winner_counts_win = torch.zeros_like(winner_counts_total)
            except Exception:
                winner_counts_total = None
                winner_counts_win = None

            # Per-layer windowed stats (only when mon_layers enabled)
            act_layers_sum = {k: 0 for k in mon_layers}
            act_layers_min = {k: None for k in mon_layers}
            act_layers_max = {k: None for k in mon_layers}
            # Resume: skip training if resume_checkpoint provided.
            if getattr(cfg, "resume_checkpoint", None):
                _status_update(stage="resume_checkpoint")
                load_csnn_weights_into(net, connection, lif_layer, str(cfg.resume_checkpoint))
                n_train = 0
            else:
                _status_update(stage="train", i=0, n=int(n_train), pct=0.0)

            activity_log_path = run_dir / "activity.jsonl"

            for i in range(n_train):
                idx = order[i]
                x = ds[idx]["image"].to(device)
                spikes = encoder(x)
                # spikes should already be [T,1,1,28,28] when encoder_out_format=TBNCHW
                if hasattr(spikes, "dim") and spikes.dim() == 3:
                    spikes_hw = _spikes_flat_to_hw(
                        spikes,
                        device,
                        C=int(getattr(cfg, "input_channels", 1)),
                        H=int(getattr(cfg, "input_h", 28)),
                        W=int(getattr(cfg, "input_w", 28)),
                    )
                else:
                    spikes_hw = spikes
                # Greedy layer-wise STDP when multiple conv connections exist.
                # - For 2 conv layers: train Conv1 for greedy_n1 samples, then Conv2.
                # - For 3 conv layers: train Conv1 for greedy_n1, then Conv2 for greedy_n2, then Conv3 for the rest.
                if bool(getattr(cfg, "greedy_enable", False)) and (len(csnn_conns) >= 2):
                    # Only apply greedy when nu tensors are available.
                    if all(x is not None for x in csnn_nu_orig[: min(3, len(csnn_conns))]):
                        n1 = int(getattr(cfg, "greedy_n1", 2500))
                        if len(csnn_conns) >= 3:
                            n2 = int(getattr(cfg, "greedy_n2", n1))
                            if i < n1:
                                # Train Conv1 only.
                                csnn_conns[0].update_rule.nu = csnn_nu_orig[0]
                                csnn_conns[1].update_rule.nu = csnn_nu_zero[1]
                                csnn_conns[2].update_rule.nu = csnn_nu_zero[2]
                            elif i < (n1 + n2):
                                # Train Conv2 only.
                                csnn_conns[0].update_rule.nu = csnn_nu_zero[0]
                                csnn_conns[1].update_rule.nu = csnn_nu_orig[1]
                                csnn_conns[2].update_rule.nu = csnn_nu_zero[2]
                            else:
                                # Train Conv3 only.
                                csnn_conns[0].update_rule.nu = csnn_nu_zero[0]
                                csnn_conns[1].update_rule.nu = csnn_nu_zero[1]
                                csnn_conns[2].update_rule.nu = csnn_nu_orig[2]
                        else:
                            if i < n1:
                                csnn_conns[0].update_rule.nu = csnn_nu_orig[0]
                                csnn_conns[1].update_rule.nu = csnn_nu_zero[1]
                            else:
                                csnn_conns[0].update_rule.nu = csnn_nu_zero[0]
                                csnn_conns[1].update_rule.nu = csnn_nu_orig[1]

                net.run(inputs={"Input": spikes_hw}, time=T)

                # Per-layer spike counts (sum over time) if per-layer monitors are available.
                layer_spikes = {}
                if mon_layers:
                    for lname, mon in mon_layers.items():
                        try:
                            sL = mon.get("s")  # [T,B,C,H,W]
                            layer_spikes[lname] = int(sL.sum().item()) if torch.is_tensor(sL) else None
                            mon.reset_state_variables()
                        except Exception:
                            layer_spikes[lname] = None

                # Per-sample spike count: sum over time from monitor, then reset monitor buffers.
                ssum = None
                try:
                    sH = mon_H.get("s")  # expected [T,B,C,H,W]
                    if torch.is_tensor(sH):
                        ssum = int(sH.sum().item())

                        # Track per-sample winner (top-1 neuron by total spikes over time).
                        try:
                            if winner_counts_total is not None:
                                if sH.dim() == 5:
                                    counts = sH[:, 0, :, :, :].reshape(sH.shape[0], -1).sum(0)
                                else:
                                    counts = sH[:, 0, :].sum(0)
                                if torch.is_tensor(counts) and int(counts.sum().item()) > 0:
                                    widx = int(counts.argmax().item())
                                    winner_counts_total[widx] += 1
                                    if winner_counts_win is not None:
                                        winner_counts_win[widx] += 1
                        except Exception:
                            pass

                        mon_H.reset_state_variables()
                except Exception:
                    ssum = None

                # Prefer last-layer per-layer monitor for ssum (more explicit in multi-layer mode).
                if (ssum is None) and layer_spikes and last_layer_name in layer_spikes:
                    try:
                        if layer_spikes[last_layer_name] is not None:
                            ssum = int(layer_spikes[last_layer_name])
                    except Exception:
                        pass

                if ssum is None:
                    # Fallback: last-step spikes only.
                    s_layer = getattr(lif_layer, "s", None)
                    ssum = int(s_layer.sum().item()) if torch.is_tensor(s_layer) else 0
                S_out += ssum

                # Update windowed activity stats (more stable than single-sample snapshots)
                act_win_sum += int(ssum)
                act_win_n += 1
                act_win_min = int(ssum) if act_win_min is None else min(act_win_min, int(ssum))
                act_win_max = int(ssum) if act_win_max is None else max(act_win_max, int(ssum))

                # Per-layer windowed stats
                if layer_spikes and act_layers_sum:
                    for lname, v in layer_spikes.items():
                        if v is None:
                            continue
                        act_layers_sum[lname] += int(v)
                        act_layers_min[lname] = int(v) if act_layers_min[lname] is None else min(act_layers_min[lname], int(v))
                        act_layers_max[lname] = int(v) if act_layers_max[lname] is None else max(act_layers_max[lname], int(v))

                # Homeostasis: update encoder rate scale for next samples.
                homeo_info = None
                if homeo is not None:
                    try:
                        homeo_info = homeo.observe(float(ssum), int(i + 1))
                        # Apply for subsequent samples.
                        if homeo_info and hasattr(encoder, "rate_scale"):
                            encoder.rate_scale = float(homeo_info.get("homeo_rate_scale", encoder.rate_scale))
                    except Exception:
                        homeo_info = None

                if activity_log_every and ((i + 1) % int(activity_log_every) == 0):
                    # --- Activity guard: abort early if the network is effectively silent ---
                    try:
                        win_mean_guard = float(act_win_sum) / max(1, act_win_n)
                        in_spikes_guard = None
                        if torch.is_tensor(spikes_hw):
                            in_spikes_guard = int(spikes_hw.sum().item())
                        check_after = int(getattr(cfg, "activity_check_after", 1000))
                        min_win_mean = float(getattr(cfg, "activity_min_spikes_win_mean", 1000.0))
                        if (int(i + 1) >= check_after) and (in_spikes_guard is None or int(in_spikes_guard) > 0):
                            if win_mean_guard < min_win_mean:
                                msg = (
                                    f"low activity: spikes_win_mean={win_mean_guard:.3f} < {min_win_mean:.3f} "
                                    f"at i={i+1} (in_spikes={in_spikes_guard})"
                                )
                                _status_update(stage="train_failed", error=msg, error_type="RuntimeError")
                                raise RuntimeError(msg)
                    except RuntimeError:
                        raise
                    except Exception:
                        pass

                    try:
                        in_spikes = None
                        if torch.is_tensor(spikes_hw):
                            in_spikes = int(spikes_hw.sum().item())

                        # monitor cross-check (same as ssum); keep None to avoid confusion
                        ssum_mon = None

                        v_mean = None
                        v_max = None
                        try:
                            v = getattr(lif_layer, "v", None)
                            if torch.is_tensor(v) and v.numel() > 0:
                                v_mean = float(v.mean().item())
                                v_max = float(v.max().item())
                        except Exception:
                            pass

                        wsum = None
                        wmin = None
                        wmax = None
                        if hasattr(connection, "w") and torch.is_tensor(connection.w):
                            wsum = float(connection.w.detach().abs().sum().item())
                            wmin = float(connection.w.detach().min().item())
                            wmax = float(connection.w.detach().max().item())

                        rec_extra = {
                            "in_spikes": in_spikes,
                            "w_abs_sum": wsum,
                            "w_min": wmin,
                            "w_max": wmax,
                            "v_mean": v_mean,
                            "v_max": v_max,
                            "spikes_mon": ssum_mon,
                        }
                    except Exception:
                        rec_extra = {}
                    try:
                        theta_mean = None
                        theta = getattr(lif_layer, "theta", None)
                        if theta is not None and hasattr(theta, "mean"):
                            theta_mean = float(theta.mean().item())
                        win_mean = (float(act_win_sum) / max(1, act_win_n))

                        # Windowed winner stats (top-1 winner per sample)
                        win_unique = None
                        win_hhi = None
                        try:
                            if winner_counts_win is not None:
                                tot = int(winner_counts_win.sum().item())
                                win_unique = int((winner_counts_win > 0).sum().item())
                                if tot > 0:
                                    p = winner_counts_win.float() / float(tot)
                                    win_hhi = float((p * p).sum().item())
                                else:
                                    win_hhi = 0.0
                        except Exception:
                            win_unique = None
                            win_hhi = None

                        layers_rec = None
                        if act_layers_sum and act_win_n > 0:
                            try:
                                layers_rec = {}
                                for lname in act_layers_sum:
                                    layers_rec[lname] = {
                                        "spikes_win_mean": float(act_layers_sum[lname]) / float(max(1, act_win_n)),
                                        "spikes_win_min": int(act_layers_min[lname]) if act_layers_min[lname] is not None else None,
                                        "spikes_win_max": int(act_layers_max[lname]) if act_layers_max[lname] is not None else None,
                                    }
                            except Exception:
                                layers_rec = None

                        rec = {
                            "i": int(i + 1),
                            "spikes": int(ssum),
                            "spikes_per_sample": float(ssum),
                            "spikes_win_mean": float(win_mean),
                            "spikes_win_min": int(act_win_min) if act_win_min is not None else None,
                            "spikes_win_max": int(act_win_max) if act_win_max is not None else None,
                            "winners_unique_win": win_unique,
                            "winner_HHI_win": win_hhi,
                            "theta_mean": theta_mean,
                            "t": utc_now_iso(),
                            **rec_extra,
                        }
                        if layers_rec is not None:
                            rec["layers"] = layers_rec
                        if homeo_info:
                            rec.update(homeo_info)
                        with open(activity_log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        _status_update(
                            stage="train",
                            activity_spikes=float(ssum),
                            activity_spikes_win_mean=float(win_mean),
                            activity_spikes_win_min=(int(act_win_min) if act_win_min is not None else None),
                            activity_spikes_win_max=(int(act_win_max) if act_win_max is not None else None),
                            activity_theta_mean=theta_mean,
                            activity_winners_unique=(int(win_unique) if win_unique is not None else None),
                            activity_winner_HHI=(float(win_hhi) if win_hhi is not None else None),
                            activity_homeo_rate_scale=(float(homeo_info.get("homeo_rate_scale")) if homeo_info else None),
                            activity_homeo_ema_spikes=(float(homeo_info.get("homeo_ema_spikes")) if homeo_info else None),
                            activity_homeo_rate_mul=(float(homeo_info.get("homeo_rate_mul")) if homeo_info else None),
                            activity_homeo_action=(homeo_info.get("homeo_action") if homeo_info else None),
                        )

                        # Reset window at each activity log tick so stats reflect the last window.
                        act_win_sum = 0
                        act_win_n = 0
                        act_win_min = None
                        act_win_max = None

                        if winner_counts_win is not None:
                            winner_counts_win.zero_()

                        # Reset per-layer windows.
                        if act_layers_sum:
                            for lname in act_layers_sum:
                                act_layers_sum[lname] = 0
                                act_layers_min[lname] = None
                                act_layers_max[lname] = None
                    except Exception:
                        pass

                # --- Optional online voting check (proportional vote on a small calibration set) ---
                # Enabled via env var to avoid slowing down default runs.
                # Uses counts_readout.collect_counts_plus_fast and the proportional vote implementation in evaluation.
                try:
                    import os
                    vote_every = int(os.environ.get("SNN_VOTE_EVERY", "0") or "0")
                    vote_calib = int(os.environ.get("SNN_VOTE_CALIB", "0") or "0")
                except Exception:
                    vote_every, vote_calib = 0, 0

                if vote_every > 0 and vote_calib > 0 and ((i + 1) % vote_every == 0):
                    try:
                        from counts_readout import make_vision_datasets, collect_counts_plus_fast

                        _status_update(stage="vote_check", i=int(i + 1), n=int(n_train), pct=float(100.0 * (i + 1) / max(1, n_train)))
                        ds_train_chk, ds_test_chk = make_vision_datasets(dataset=str(getattr(cfg, "dataset", "mnist")))

                        # We compute proportional vote accuracy by reusing eval_readouts_from_net with env flag.
                        os.environ["SNN_PROPORTIONAL_VOTE"] = "1"
                        os.environ["SNN_DISABLE_READOUT_PROBE"] = "1"

                        # NOTE: do not import eval_readouts_from_net inside this function.
                        # Python would treat it as a local variable and break later calls.
                        accs = eval_readouts_from_net(
                            net, lif_layer, encoder, cfg,
                            label_map=[-1] * int(lif_layer.n),
                            n_train_counts=vote_calib,
                            n_test_counts=min(vote_calib, 2000),
                            status_cb=_status_update,
                        )
                        vote_acc = None
                        if isinstance(accs, dict):
                            vote_acc = accs.get("proportional_vote")
                        _status_update(stage="vote_check_done", vote_acc=vote_acc)
                    except Exception as e:
                        _status_update(stage="vote_check_failed", error=str(e), error_type=type(e).__name__)

                # --- Zalipe detection: detect obviously broken dynamics and fail fast ---
                if (i + 1) % max(1, int(activity_log_every or 1000)) == 0:
                    try:
                        # If weights look constant AND the layer is effectively silent, abort early.
                        # Note: v can legitimately be 0 even when spiking (reset-on-spike), so use spikes.
                        if (wmin is not None and wmax is not None and abs(wmax - wmin) < 1e-8):
                            if (ssum is not None) and int(ssum) == 0 and (in_spikes is None or int(in_spikes) > 0):
                                raise RuntimeError(
                                    f"zalipe detected: w_min==w_max=={wmin} and spikes==0 at i={i+1}"
                                )
                    except Exception as e:
                        _status_update(stage="train_failed", error=str(e), error_type=type(e).__name__)
                        raise

                net.reset_state_variables()

                now = time.time()
                if now - last_report >= 30:
                    pct = 100.0 * (i + 1) / max(1, n_train)
                    it_s = (i + 1) / max(1e-6, (now - (last_report - 30)))  # rough
                    eta = int((n_train - (i + 1)) / max(1e-6, it_s))
                    _status_update(stage="train", i=int(i + 1), n=int(n_train), pct=float(pct), eta_seconds=int(eta))
                    last_report = now

            spikes_per_sample = float(S_out) / max(1, n_train)
            # propagate arch tag for evaluation pipeline
            cfg.arch = "csnn"

            train_summary = {
                "arch": "csnn",
                "N": int(n_train),
                "device": str(device),
                "time": int(cfg.time),
                "encoder": str(cfg.encoder),
                "spikes_per_sample": spikes_per_sample,
                "energy_proxy_per_sample": spikes_per_sample,
                "winners_unique": (int((winner_counts_total > 0).sum().item()) if winner_counts_total is not None else None),
                "winner_HHI": None,
            }
            try:
                if winner_counts_total is not None:
                    tot = int(winner_counts_total.sum().item())
                    if tot > 0:
                        p = winner_counts_total.float() / float(tot)
                        train_summary["winner_HHI"] = float((p * p).sum().item())
                    else:
                        train_summary["winner_HHI"] = 0.0
            except Exception:
                pass
        else:
            train_summary, connection, lif_layer, net, encoder = run_experiment(
                cfg, verbose=verbose, progress=progress
            )

        # Save a checkpoint right after training so we can resume eval/readouts later.
        if arch == "csnn" and not getattr(cfg, "resume_checkpoint", None):
            _status_update(stage="checkpoint_after_train")
            early_ckpt = run_dir / "model_after_train.pt"
            save_snn(
                str(early_ckpt),
                cfg,
                connection,
                lif_layer,
                train_summary=train_summary,
                eval_summary=None,
                notes={"run_id": run_id, "config_hash": cfg_digest, "stage": "after_train"},
            )

        label_map_path = None
        label_map_summary = None
        label_map = None
        if not skip_label_map:
            _status_update(stage="label_map")
            if arch == "csnn":
                from csnn_mnist_net import _spikes_flat_to_hw

                # Note: encoder_rate_boost is used in counts collection; for label_map we also apply
                # it by temporarily scaling encoder.rate_scale. This keeps PoissonEncoder's
                # deterministic mode behaviour consistent with the rest of the pipeline.
                _lm_dev = cfg.torch_device()
                _lm_C = int(getattr(cfg, "input_channels", 1))
                _lm_H = int(getattr(cfg, "input_h", 28))
                _lm_W = int(getattr(cfg, "input_w", 28))
                _lm_boost = float(getattr(cfg, "encoder_rate_boost", 1.0))
                _lm_use_boost = (
                    (str(getattr(cfg, "encoder", "poisson")).lower() == "poisson")
                    and (_lm_boost != 1.0)
                    and hasattr(encoder, "rate_scale")
                )

                def _enc_hw(x):
                    if _lm_use_boost:
                        old = float(getattr(encoder, "rate_scale"))
                        try:
                            encoder.rate_scale = old * _lm_boost
                            sp = encoder(x)
                        finally:
                            encoder.rate_scale = old
                        return _spikes_flat_to_hw(sp, _lm_dev, C=_lm_C, H=_lm_H, W=_lm_W)

                    return _spikes_flat_to_hw(
                        encoder(x),
                        _lm_dev,
                        C=_lm_C,
                        H=_lm_H,
                        W=_lm_W,
                    )

                label_map = build_label_map(
                    net,
                    None,
                    lif_layer,
                    _enc_hw,
                    n_calib=n_calib,
                    T=cfg.time,
                    top_k=max(1, int(getattr(cfg, "top_k", 0)) if int(getattr(cfg, "top_k", 0)) > 0 else 3),
                    seed=cfg.seed,
                    dataset=str(getattr(cfg, "dataset", "mnist")),
                )
            else:
                label_map = build_label_map(
                    net,
                    None,
                    lif_layer,
                    encoder,
                    n_calib=n_calib,
                    T=cfg.time,
                    top_k=max(1, int(cfg.top_k) if int(cfg.top_k) > 0 else 3),
                    seed=cfg.seed,
                    dataset=str(getattr(cfg, "dataset", "mnist")),
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
            # Extra metric: neurons-per-class distribution (assigned neurons only).
            try:
                lm = torch.as_tensor(label_map, dtype=torch.long)
                assigned = lm[lm >= 0]
                n_classes = int(assigned.max().item()) + 1 if assigned.numel() > 0 else 0
                if n_classes > 0:
                    counts = torch.bincount(assigned.cpu(), minlength=n_classes).tolist()
                    label_map_summary["neurons_per_class"] = {str(i): int(c) for i, c in enumerate(counts)}
                    # Basic dispersion stats for quick scanning.
                    mean = float(sum(counts)) / float(max(1, len(counts)))
                    var = float(sum((c - mean) ** 2 for c in counts)) / float(max(1, len(counts)))
                    cv = (var ** 0.5) / mean if mean > 0 else None
                    label_map_summary["neurons_per_class_mean"] = mean
                    label_map_summary["neurons_per_class_min"] = int(min(counts)) if counts else None
                    label_map_summary["neurons_per_class_max"] = int(max(counts)) if counts else None
                    label_map_summary["neurons_per_class_cv"] = float(cv) if cv is not None else None
            except Exception:
                pass

        eval_summary = None
        if not skip_eval:
            _status_update(stage="eval")
            def _eval_status_cb(stage: str, **kw):
                # kw: step, write_pos, batch_size, n_samples
                i = int(kw.get("write_pos", 0) + kw.get("batch_size", 0))
                n = int(kw.get("n_samples", 0))
                pct = 100.0 * i / max(1, n) if n else None
                _status_update(stage=stage, i=i, n=n, pct=pct)

            eval_summary = eval_readouts_from_net(
                net,
                lif_layer,
                encoder,
                cfg,
                label_map=label_map,
                n_train_counts=n_train_counts,
                n_test_counts=n_test_counts,
                status_cb=_eval_status_cb,
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
    parser.add_argument(
        "--resume-from",
        help=(
            "Path to an existing run directory to resume from. "
            "Loads <run_dir>/model_after_train.pt and skips training (N is forced to 0)."
        ),
    )
    parser.add_argument("--n-calib", type=int, default=2000)
    parser.add_argument("--n-train-counts", type=int, default=60000)
    parser.add_argument("--n-test-counts", type=int, default=10000)
    parser.add_argument("--activity-log-every", type=int, default=1000)
    parser.add_argument("--skip-label-map", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    file_overrides = read_json(args.config_json) if args.config_json else {}
    cli_overrides = parse_overrides(args.set)
    overrides = merge_dicts(file_overrides, cli_overrides)

    if args.resume_from:
        resume_dir = Path(args.resume_from)
        ckpt = resume_dir / "model_after_train.pt"
        if not ckpt.exists():
            raise SystemExit(f"--resume-from: missing checkpoint: {ckpt}")
        # Do not force N=0: label_map/eval use cfg.N for dataset sizing.
        # Training will be skipped inside run_single_experiment when resume_checkpoint is provided.
        overrides = merge_dicts(overrides, {
            "resume_checkpoint": str(ckpt),
        })

    if not overrides:
        raise SystemExit("No config overrides provided. Pass --config-json or --set key=value.")

    payload = run_single_experiment(
        overrides,
        outdir=args.outdir,
        n_calib=args.n_calib,
        n_train_counts=args.n_train_counts,
        n_test_counts=args.n_test_counts,
        activity_log_every=args.activity_log_every,
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
