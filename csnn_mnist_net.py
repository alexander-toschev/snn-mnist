# csnn_mnist_net.py
"""Convolutional Spiking Neural Network (CSNN) for MNIST.

Goal: provide a biologically-inspired upgrade path from the current FC SNN.

Design (v0):
- Input spikes: (T, 1, 1, 28, 28) or (T, 1, 784) depending on encoder output
- Conv1 (shared weights) -> LIF -> local competition/inhibition
- (Optional) pooling
- Flatten (counts over time) -> readout (reuse existing readout_models.py)

This file is intentionally minimal and mirrors the existing snn_mnist_net.py structure
so experiment_runner can swap nets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Conv2dConnection, Connection
from bindsnet.learning import PostPre, WeightDependentPostPre

from encoders import LatencyEncoder, PoissonEncoder


@dataclass
class CSCfg:
    time: int = 200
    device: str = "cpu"
    seed: int = 42
    shuffle: bool = True
    N: int = 12000
    dataset: str = "mnist"  # mnist | fashion | kmnist | emnist[:split] | cifar100[:K]
    resume_checkpoint: str | None = None

    # input shape (for non-MNIST datasets)
    input_channels: int = 1
    input_h: int = 28
    input_w: int = 28

    # input encoding
    encoder: str = "poisson"
    poisson_rate_scale: float = 0.011
    poisson_base_seed: int = 123
    poisson_deterministic: bool = False
    encoder_out_format: str = "TBNCHW"
    encoder_rate_boost: float = 3.0
    latency_x_min: float = 0.05

    # Online homeostasis (keep spikes/sample within a corridor by adjusting Poisson rate scale)
    homeo_enable: bool = False
    homeo_spikes_lo: float = 3000.0
    homeo_spikes_hi: float = 5000.0
    homeo_spikes_target: float = 4000.0
    homeo_ema_alpha: float = 0.01
    homeo_update_every: int = 25
    homeo_warmup: int = 200
    homeo_gain: float = 0.25
    homeo_rate_mul_min: float = 0.3
    homeo_rate_mul_max: float = 3.0

    # conv params
    c1_out: int = 32
    c1_kernel: int = 5
    c1_stride: int = 1
    c1_pad: int = 0

    # Optional 2nd conv layer (Stage-2: deeper STDP-CNN)
    c2_out: int = 0
    c2_kernel: int = 3
    c2_stride: int = 2
    c2_pad: int = 1

    # Optional 3rd conv layer (Stage-3)
    # Enabled only when c2_out>0 and c3_out>0.
    c3_out: int = 0
    c3_kernel: int = 3
    c3_stride: int = 2
    c3_pad: int = 1

    # Greedy layer-wise STDP (when c2_out>0): train Conv1 first, then Conv2.
    greedy_enable: bool = False
    greedy_n1: int = 2500
    # When Conv3 is enabled, train Conv2 for greedy_n2 samples next, then Conv3 for the rest.
    greedy_n2: int = 2500

    # weight normalization (Diehl&Cook uses norm=78.4 for FC). For conv we normalize per out-channel.
    w_norm_enable: bool = False
    w_norm_target: float = 78.4

    # neuron params (shared)
    tau_val: float = 150.0
    refrac_val: float = 2.0
    reset_val: float = 0.0
    rest_val: float = 0.0
    thresh_init: float = 0.38

    # STDP
    nu_plus: float = 1e-4
    nu_minus: float = -1e-3

    # inhibition/competition
    inhib_strength: float = 0.7
    top_k: int = 0  # if >0, apply top-k competition on counts per sample
    wta_enable: bool = False  # if True, apply per-position channel WTA on Conv1 spikes

    # Diehl & Cook-style E/I competition (conv-friendly): add explicit inhibitory population.
    # We implement local (per-(h,w)) E->I and I->E without NxN weights.
    ei_enable: bool = False
    ei_exc: float = 22.5
    ei_inh: float = 120.0
    # In multi-layer (Conv2 enabled), the same absolute inhibition often over-suppresses activity.
    # We scale I->E strength by this multiplier when c2_out>0.
    ei_inh_mult_2layer: float = 0.01

    # Legacy local competition hook (kept for ablations)
    local_inhib_enable: bool = False
    local_inhib_strength: float = 0.7

    # Adaptive threshold (Diehl & Cook): v_thresh = thresh_base + theta
    adapt_thresh_enable: bool = False
    theta_plus: float = 0.05
    tau_theta: float = 1e4

    # Temporal sparsity: allow each neuron to spike at most once per sample
    # (approximate first-spike / time-to-first-spike behavior for rate encoders).
    first_spike_only: bool = False

    # misc (keep parity with FC pipeline overrides)
    log_every: int = 50
    warmup_N: int = 0

    # activity guard (fail-fast for obviously silent networks)
    # Checked at activity_log_every ticks in experiment_runner.
    activity_check_after: int = 1000
    activity_min_spikes_win_mean: float = 1000.0

    def torch_device(self) -> torch.device:
        dev = torch.device(self.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return dev


def build_encoder_from_cfg(cfg: CSCfg):
    enc = (cfg.encoder or "poisson").lower()
    out_fmt = getattr(cfg, "encoder_out_format", "TBNCHW")
    if enc == "poisson":
        return PoissonEncoder(
            T=cfg.time,
            rate_scale=cfg.poisson_rate_scale,
            base_seed=cfg.poisson_base_seed,
            deterministic=bool(getattr(cfg, "poisson_deterministic", False)),
            out_format=out_fmt,
        )
    if enc == "latency":
        return LatencyEncoder(time=cfg.time, out_format=out_fmt, x_min=float(cfg.latency_x_min))
    raise ValueError(f"Unknown encoder: {cfg.encoder}")


def _spikes_flat_to_hw(spikes: Tensor, device: torch.device, *, C: int = 1, H: int = 28, W: int = 28) -> Tensor:
    """Convert encoder output to [T,B,C,H,W] for Conv2dConnection.

    Accepts encoder outputs:
    - [T,B,C,H,W] (preferred)
    - [T, B, N]
    - [T, B, 1, N]

    Where N == C*H*W.
    """
    if not torch.is_tensor(spikes):
        spikes = torch.as_tensor(spikes)
    spikes = spikes.to(device=device, dtype=torch.float32, non_blocking=True)

    if spikes.ndim == 5:
        return spikes

    exp = int(C) * int(H) * int(W)
    if spikes.ndim == 3:
        T, B, N = spikes.shape
        if N != exp:
            raise ValueError(f"Expected last dim {exp}, got {N}")
        spikes_tbn = spikes
    elif spikes.ndim == 4:
        T, B, C1, N = spikes.shape
        if C1 != 1 or N != exp:
            raise ValueError(f"Expected [T,B,1,{exp}], got {tuple(spikes.shape)}")
        spikes_tbn = spikes[:, :, 0, :]
    else:
        raise ValueError(f"Unexpected spikes ndim={spikes.ndim}")

    return spikes_tbn.view(T, B, int(C), int(H), int(W))


def load_csnn_weights_into(net: Network, conn: Connection, lif_layer: LIFNodes, ckpt_path: str) -> None:
    """Load weights (and optional thresholds) from a save_snn() checkpoint.

    Expects keys: W, v_thresh (optional), cfg.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "W" not in ckpt:
        raise ValueError(f"Checkpoint missing 'W': {ckpt_path}")

    with torch.no_grad():
        conn.w.copy_(ckpt["W"].to(conn.w.device))
        # Threshold tensor name depends on BindsNet version.
        vt = getattr(lif_layer, "v_thresh", getattr(lif_layer, "thresh", None))
        if vt is not None and "v_thresh" in ckpt and torch.is_tensor(vt):
            ckpt_vt = ckpt["v_thresh"]
            if torch.is_tensor(ckpt_vt):
                ckpt_vt = ckpt_vt.to(vt.device)
                # Some BindsNet versions store threshold as a scalar tensor ([]) even for shaped layers.
                # In that case, loading a per-neuron threshold tensor would crash.
                if tuple(vt.shape) == tuple(ckpt_vt.shape):
                    vt.copy_(ckpt_vt)
                elif vt.numel() == 1:
                    vt.fill_(float(ckpt_vt.float().mean().item()))
                    print(
                        f"[csnn] WARN: vt shape {tuple(vt.shape)} != ckpt_vt {tuple(ckpt_vt.shape)}; "
                        "loaded mean(v_thresh) into scalar threshold"
                    )
                elif vt.numel() == ckpt_vt.numel():
                    vt.copy_(ckpt_vt.reshape_as(vt))
                    print(f"[csnn] WARN: reshaped ckpt v_thresh {tuple(ckpt_vt.shape)} -> {tuple(vt.shape)}")
                else:
                    print(f"[csnn] WARN: skip v_thresh load: vt shape {tuple(vt.shape)} vs ckpt {tuple(ckpt_vt.shape)}")
    print(f"[csnn] Loaded checkpoint: {ckpt_path}")


def build_csnn(cfg: CSCfg) -> Tuple[Network, Input, LIFNodes, Connection]:
    """Build a CSNN.

    - Base: Conv1 STDP
    - Optional: Conv2 STDP when cfg.c2_out > 0
    - Optional: Conv3 STDP when cfg.c2_out > 0 and cfg.c3_out > 0

    Returns (net, input_layer, lif_layer, connection) where lif_layer/connection refer to
    the *last* conv layer (Conv2 if enabled, else Conv1) for backward compatibility.
    Additional references are attached to the returned net:
      - net._csnn_lifs: list[LIFNodes] in forward order
      - net._csnn_conns: list[Connection] in forward order
    """

    device = cfg.torch_device()

    net = Network()

    # Conv2dConnection requires explicit 3D shapes (C,H,W) on source/target layers.
    C_in = int(getattr(cfg, "input_channels", 1))
    H_in = int(getattr(cfg, "input_h", 28))
    W_in = int(getattr(cfg, "input_w", 28))
    input_layer = Input(shape=(C_in, H_in, W_in), traces=True)

    # Conv output spatial size: (H + 2*pad - kernel)/stride + 1
    h = (H_in + 2 * cfg.c1_pad - cfg.c1_kernel) // cfg.c1_stride + 1
    w = (W_in + 2 * cfg.c1_pad - cfg.c1_kernel) // cfg.c1_stride + 1
    conv_lif = LIFNodes(
        shape=(cfg.c1_out, h, w),
        traces=True,
        thresh=float(cfg.thresh_init),
        rest=float(cfg.rest_val),
        reset=float(cfg.reset_val),
        refrac=int(cfg.refrac_val),
        tc_decay=float(cfg.tau_val),
    )

    # Inhibitory population (Ai) like Diehl&Cook, same shape.
    inhib_lif = None
    if bool(getattr(cfg, "ei_enable", False)):
        inhib_lif = LIFNodes(
            shape=(cfg.c1_out, h, w),
            traces=True,
            thresh=float(cfg.thresh_init),
            rest=float(cfg.rest_val),
            reset=float(cfg.reset_val),
            refrac=int(cfg.refrac_val),
            tc_decay=float(cfg.tau_val),
        )

    net.add_layer(input_layer, name="Input")
    net.add_layer(conv_lif, name="Conv1")
    if inhib_lif is not None:
        net.add_layer(inhib_lif, name="Inhib1")

    # Optional local inhibition (soft WTA) is applied via a net.run hook below.

    # Optional local competition + adaptive threshold hooks.
    if (
        bool(getattr(cfg, "wta_enable", False))
        or bool(getattr(cfg, "local_inhib_enable", False))
        or bool(getattr(cfg, "ei_enable", False))
        or bool(getattr(cfg, "adapt_thresh_enable", False))
    ):
        import types

        _orig_run = net.run

        # theta state for adaptive threshold
        #
        # IMPORTANT: This must survive net.reset_state_variables() (called after each sample/batch).
        # BindsNet resets v/s/traces, but won't touch our custom attribute `theta`.
        #
        # We implement adaptive threshold by changing conv_lif.thresh (scalar or [C,H,W]).
        # Do NOT make it batch-shaped.
        if bool(getattr(cfg, "adapt_thresh_enable", False)):
            conv_lif.theta = None  # lazily initialized on first run
            conv_lif.thresh_base = torch.as_tensor(float(cfg.thresh_init), device=device)
            conv_lif.thresh = conv_lif.thresh_base

        def _winner_mask(s_t: torch.Tensor) -> torch.Tensor:
            win = s_t.argmax(dim=1, keepdim=True)  # [B,1,H,W]
            mask = torch.zeros_like(s_t, dtype=torch.bool)
            mask.scatter_(1, win, True)
            return mask

        def _after_step_hooks():
            """Apply competition/homeostasis after a single simulation step.

            NOTE: These act *after* BindsNet updates for the current step.
            Running the network step-by-step ensures the effect influences subsequent timesteps
            (and therefore shapes STDP over time), which is closer to Diehl&Cook dynamics.
            """
            try:
                # Optional weight norm (after STDP update inside the step).
                # IMPORTANT for multi-layer CSNN: normalize *all* conv connections, not only Conv1.
                if bool(getattr(cfg, "w_norm_enable", False)):
                    tgt = float(getattr(cfg, "w_norm_target", 78.4))
                    eps = 1e-8
                    conns = list(getattr(net, "_csnn_conns", [])) or [conn]
                    for _c in conns:
                        w = getattr(_c, "w", None)
                        if torch.is_tensor(w) and w.ndim == 4 and w.numel() > 0:
                            sabs = w.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
                            scale = (tgt / sabs).clamp_max(1.0)
                            _c.w.data = w * scale

                # Multi-layer CSNN: apply adaptive threshold + competition to *each* conv LIF layer.
                lifs = list(getattr(net, "_csnn_lifs", [])) or [conv_lif]
                conns = list(getattr(net, "_csnn_conns", [])) or [conn]
                if not lifs:
                    return

                for li, lif in enumerate(lifs):
                    s = getattr(lif, "s", None)
                    if not (torch.is_tensor(s) and s.ndim == 4 and s.numel() > 0):
                        continue

                    # adaptive threshold update (per-timestep) — per-layer
                    if bool(getattr(cfg, "adapt_thresh_enable", False)):
                        # lazy per-layer init
                        if not hasattr(lif, "theta"):
                            lif.theta = None
                        if not hasattr(lif, "thresh_base"):
                            lif.thresh_base = torch.as_tensor(float(cfg.thresh_init), device=device)
                        if not hasattr(lif, "thresh"):
                            lif.thresh = lif.thresh_base

                        c = conns[li] if li < len(conns) else conns[-1]
                        frozen_stdp = False
                        try:
                            ur = getattr(c, "update_rule", None)
                            nu = getattr(ur, "nu", None)
                            if nu is not None and isinstance(nu, (tuple, list)) and len(nu) == 2:
                                n0 = float(torch.as_tensor(nu[0]).abs().sum().item())
                                n1 = float(torch.as_tensor(nu[1]).abs().sum().item())
                                frozen_stdp = (n0 + n1) <= 0.0
                        except Exception:
                            frozen_stdp = False

                        theta = getattr(lif, "theta", None)
                        if (theta is None) or (not torch.is_tensor(theta)) or (theta.ndim != 3):
                            theta = torch.zeros_like(s[0])  # [C,H,W]

                        if not frozen_stdp:
                            tau = float(getattr(cfg, "tau_theta", 1e4))
                            theta = theta * (1.0 - 1.0 / max(1.0, tau))
                            theta = theta + float(getattr(cfg, "theta_plus", 0.05)) * s.detach().sum(dim=0)
                            theta = theta.clamp_(min=0.0, max=10.0)
                            lif.theta = theta

                        lif.thresh = lif.thresh_base + theta

                    # competition
                    if li == 0 and bool(getattr(cfg, "ei_enable", False)) and inhib_lif is not None:
                        # EI currently implemented only for Conv1.
                        exc = float(getattr(cfg, "ei_exc", 22.5))
                        inh = float(getattr(cfg, "ei_inh", 120.0))
                        try:
                            if int(getattr(cfg, "c2_out", 0) or 0) > 0:
                                inh *= float(getattr(cfg, "ei_inh_mult_2layer", 0.01))
                        except Exception:
                            pass
                        inhib_lif.forward(exc * s.detach())
                        i_s = inhib_lif.s.detach().to(s.dtype)
                        lif.v = lif.v - inh * i_s

                        # Optional WTA after EI: keep only per-position winners.
                        if bool(getattr(cfg, "wta_enable", False)):
                            s2 = getattr(lif, "s", None)
                            if torch.is_tensor(s2) and s2.ndim == 4:
                                m = _winner_mask(s2)
                                lif.s = s2.to(s.dtype) * m.to(s.dtype)
                    elif bool(getattr(cfg, "wta_enable", False)) or bool(getattr(cfg, "local_inhib_enable", False)):
                        m = _winner_mask(s)
                        if bool(getattr(cfg, "wta_enable", False)):
                            lif.s = s * m.to(s.dtype)
                        elif bool(getattr(cfg, "local_inhib_enable", False)):
                            g = float(getattr(cfg, "local_inhib_strength", 0.7))
                            lif.s = s * (m.to(s.dtype) + (1.0 - g) * (~m).to(s.dtype))
            except Exception:
                pass

        def _run_with_hooks(self, *args, **kwargs):
            # Parse signature: net.run(inputs=..., time=...)
            inputs = kwargs.get("inputs", None)
            time = kwargs.get("time", None)
            if inputs is None and len(args) >= 1:
                inputs = args[0]
            if time is None and len(args) >= 2:
                time = args[1]
            time = int(time) if time is not None else 0

            # Run step-by-step when inputs are time-major, so hooks affect subsequent timesteps.
            if (
                time > 1
                and isinstance(inputs, dict)
                and "Input" in inputs
                and torch.is_tensor(inputs["Input"])
                and inputs["Input"].ndim >= 1
                and int(inputs["Input"].shape[0]) == time
            ):
                rest = {k: v for k, v in kwargs.items() if k not in ("inputs", "time")}
                out = None
                x = inputs["Input"]

                # Optional first-spike-only mode (per sample): once a neuron has spiked, keep it
                # refractory for the rest of the sample. Helps reduce spike avalanches in deep layers.
                fs_only = bool(getattr(cfg, "first_spike_only", False))
                fired = None
                if fs_only:
                    try:
                        lifs0 = list(getattr(net, "_csnn_lifs", [])) or [conv_lif]
                        fired = {}
                        for lif in lifs0:
                            s0 = getattr(lif, "s", None)
                            if torch.is_tensor(s0) and s0.ndim == 4:
                                fired[id(lif)] = torch.zeros_like(s0, dtype=torch.bool)
                    except Exception:
                        fired = None

                for t in range(time):
                    step_inputs = dict(inputs)
                    step_inputs["Input"] = x[t : t + 1]
                    out = _orig_run(inputs=step_inputs, time=1, **rest)
                    _after_step_hooks()

                    if fs_only and fired:
                        try:
                            lifs1 = list(getattr(net, "_csnn_lifs", [])) or [conv_lif]
                            for lif in lifs1:
                                s = getattr(lif, "s", None)
                                if not (torch.is_tensor(s) and s.ndim == 4):
                                    continue
                                key = id(lif)
                                if key not in fired:
                                    fired[key] = torch.zeros_like(s, dtype=torch.bool)
                                prev = fired[key]
                                now = s.detach() > 0
                                fired[key] = prev | now

                                # Keep already-fired neurons silent for the rest of the sample.
                                rc = getattr(lif, "refrac_count", None)
                                if torch.is_tensor(rc) and rc.shape == s.shape:
                                    # Set refractory to remaining timesteps.
                                    rem = int(max(1, time - (t + 1)))
                                    rc = torch.where(fired[key], torch.as_tensor(rem, device=rc.device, dtype=rc.dtype), rc)
                                    lif.refrac_count = rc
                                v = getattr(lif, "v", None)
                                if torch.is_tensor(v) and v.shape == s.shape:
                                    lif.v = torch.where(fired[key], torch.zeros_like(v), v)
                        except Exception:
                            pass
                return out

            # Fallback: single call
            out = _orig_run(*args, **kwargs)
            _after_step_hooks()
            return out

        net.run = types.MethodType(_run_with_hooks, net)

    conn = Conv2dConnection(
        source=input_layer,
        target=conv_lif,
        kernel_size=cfg.c1_kernel,
        stride=cfg.c1_stride,
        padding=cfg.c1_pad,
        wmin=0.0,
        wmax=1.0,
    )

    # Initialize conv weights to a reasonable scale and optionally normalize per out-channel.
    try:
        # Start with uniform weights in [0, 0.3] (empirically avoids silent networks).
        conn.w.data.uniform_(0.0, 0.3)
        if bool(getattr(cfg, "w_norm_enable", False)):
            tgt = float(getattr(cfg, "w_norm_target", 78.4))
            eps = 1e-8
            sabs = conn.w.data.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
            # Same conv-safe normalization at init: only scale DOWN.
            conn.w.data.mul_((tgt / sabs).clamp_max(1.0))
    except Exception:
        pass
    # Attach STDP (PostPre) like in the FC setup.
    # IMPORTANT: In BindsNET PostPre, nu is (pre_rate, post_rate) and is expected non-negative.
    # The rule itself applies negative sign to the pre-synaptic term and positive to post.
    nu_pre = float(getattr(cfg, "nu_minus", 1e-3))
    nu_post = float(getattr(cfg, "nu_plus", 1e-4))
    conn.update_rule = WeightDependentPostPre(
        connection=conn,
        nu=(torch.tensor(nu_pre), torch.tensor(nu_post)),
        weight_decay=1.0,
    )

    net.add_connection(conn, source="Input", target="Conv1")

    # Optional Conv2
    conv2_lif = None
    conn2 = None
    if int(getattr(cfg, "c2_out", 0)) > 0:
        c2_out = int(getattr(cfg, "c2_out", 0))
        c2_k = int(getattr(cfg, "c2_kernel", 3))
        c2_s = int(getattr(cfg, "c2_stride", 2))
        c2_p = int(getattr(cfg, "c2_pad", 1))

        # Conv2 output spatial size from Conv1 (h,w)
        h2 = (h + 2 * c2_p - c2_k) // c2_s + 1
        w2 = (w + 2 * c2_p - c2_k) // c2_s + 1
        conv2_lif = LIFNodes(
            shape=(c2_out, h2, w2),
            traces=True,
            thresh=float(cfg.thresh_init),
            rest=float(cfg.rest_val),
            reset=float(cfg.reset_val),
            refrac=int(cfg.refrac_val),
            tc_decay=float(cfg.tau_val),
        )
        net.add_layer(conv2_lif, name="Conv2")

        conn2 = Conv2dConnection(
            source=conv_lif,
            target=conv2_lif,
            kernel_size=c2_k,
            stride=c2_s,
            padding=c2_p,
            wmin=0.0,
            wmax=1.0,
        )
        try:
            conn2.w.data.uniform_(0.0, 0.3)
            if bool(getattr(cfg, "w_norm_enable", False)):
                tgt = float(getattr(cfg, "w_norm_target", 78.4))
                eps = 1e-8
                sabs = conn2.w.data.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
                conn2.w.data.mul_((tgt / sabs).clamp_max(1.0))
        except Exception:
            pass

        conn2.update_rule = WeightDependentPostPre(
            connection=conn2,
            nu=(torch.tensor(nu_pre), torch.tensor(nu_post)),
            weight_decay=1.0,
        )
        net.add_connection(conn2, source="Conv1", target="Conv2")

    # Optional Conv3 (only if Conv2 exists)
    conv3_lif = None
    conn3 = None
    if (conv2_lif is not None) and (conn2 is not None) and (int(getattr(cfg, "c3_out", 0)) > 0):
        c3_out = int(getattr(cfg, "c3_out", 0))
        c3_k = int(getattr(cfg, "c3_kernel", 3))
        c3_s = int(getattr(cfg, "c3_stride", 2))
        c3_p = int(getattr(cfg, "c3_pad", 1))

        # Conv3 output spatial size from Conv2 (h2,w2)
        h3 = (h2 + 2 * c3_p - c3_k) // c3_s + 1
        w3 = (w2 + 2 * c3_p - c3_k) // c3_s + 1
        conv3_lif = LIFNodes(
            shape=(c3_out, h3, w3),
            traces=True,
            thresh=float(cfg.thresh_init),
            rest=float(cfg.rest_val),
            reset=float(cfg.reset_val),
            refrac=int(cfg.refrac_val),
            tc_decay=float(cfg.tau_val),
        )
        net.add_layer(conv3_lif, name="Conv3")

        conn3 = Conv2dConnection(
            source=conv2_lif,
            target=conv3_lif,
            kernel_size=c3_k,
            stride=c3_s,
            padding=c3_p,
            wmin=0.0,
            wmax=1.0,
        )
        try:
            conn3.w.data.uniform_(0.0, 0.3)
            if bool(getattr(cfg, "w_norm_enable", False)):
                tgt = float(getattr(cfg, "w_norm_target", 78.4))
                eps = 1e-8
                sabs = conn3.w.data.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
                conn3.w.data.mul_((tgt / sabs).clamp_max(1.0))
        except Exception:
            pass

        conn3.update_rule = WeightDependentPostPre(
            connection=conn3,
            nu=(torch.tensor(nu_pre), torch.tensor(nu_post)),
            weight_decay=1.0,
        )
        net.add_connection(conn3, source="Conv2", target="Conv3")


    if hasattr(net, "to"):
        net.to(device)

    # Proactively move layer/connection runtime tensors to the target device.
    def _move_layer_state_(layer):
        for name in (
            "s", "x", "v", "refrac_count", "trace", "summed",
            "rest", "v_rest", "reset", "v_reset", "refrac",
            "thresh", "v_thresh", "tc_decay", "tau", "dt",
            "trace_scale",
        ):
            t = getattr(layer, name, None)
            if torch.is_tensor(t):
                setattr(layer, name, t.to(device))
            elif name in ("trace_scale",):
                # Some BindsNet versions store scalars as floats; keep them as-is.
                pass

    def _move_conn_state_(c):
        for name in ("w","b","wmin","wmax"):
            t = getattr(c, name, None)
            if torch.is_tensor(t):
                setattr(c, name, t.to(device))
        if hasattr(c, "update_rule") and c.update_rule is not None:
            nu = getattr(c.update_rule, "nu", None)
            if isinstance(nu, tuple):
                c.update_rule.nu = tuple(v.to(device) if torch.is_tensor(v) else v for v in nu)

    _move_layer_state_(input_layer)
    _move_layer_state_(conv_lif)
    _move_conn_state_(conn)

    if conv2_lif is not None and conn2 is not None:
        _move_layer_state_(conv2_lif)
        _move_conn_state_(conn2)

    if conv3_lif is not None and conn3 is not None:
        _move_layer_state_(conv3_lif)
        _move_conn_state_(conn3)

    # Attach references for multi-layer training control.
    lifs = [conv_lif] + ([conv2_lif] if conv2_lif is not None else []) + ([conv3_lif] if conv3_lif is not None else [])
    conns = [conn] + ([conn2] if conn2 is not None else []) + ([conn3] if conn3 is not None else [])
    net._csnn_lifs = lifs
    net._csnn_conns = conns

    # Backward-compat return: last layer/connection.
    last_lif = conv3_lif if conv3_lif is not None else (conv2_lif if conv2_lif is not None else conv_lif)
    last_conn = conn3 if conn3 is not None else (conn2 if conn2 is not None else conn)
    return net, input_layer, last_lif, last_conn
