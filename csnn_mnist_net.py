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
    N: int = 12000
    resume_checkpoint: str | None = None

    # input encoding
    encoder: str = "poisson"
    poisson_rate_scale: float = 0.011
    poisson_base_seed: int = 123
    poisson_deterministic: bool = False
    encoder_out_format: str = "auto"
    encoder_rate_boost: float = 3.0
    latency_x_min: float = 0.05

    # conv params
    c1_out: int = 32
    c1_kernel: int = 5
    c1_stride: int = 1
    c1_pad: int = 0

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

    # Legacy local competition hook (kept for ablations)
    local_inhib_enable: bool = False
    local_inhib_strength: float = 0.7

    # Adaptive threshold (Diehl & Cook): v_thresh = thresh_base + theta
    adapt_thresh_enable: bool = False
    theta_plus: float = 0.05
    tau_theta: float = 1e4

    # misc (keep parity with FC pipeline overrides)
    log_every: int = 50
    warmup_N: int = 0

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


def _spikes_flat_to_hw(spikes: Tensor, device: torch.device) -> Tensor:
    """Convert encoder output to [T,B,1,28,28] for Conv2dConnection.

    Accepts encoder outputs:
    - [T,B,1,28,28] (preferred)
    - [T, 1, 784] (legacy single)
    - [T, B, 784]
    - [T, B, 1, 784]

    Returns:
    - [T, B, 1, 28, 28]
    """
    if not torch.is_tensor(spikes):
        spikes = torch.as_tensor(spikes)
    spikes = spikes.to(device=device, dtype=torch.float32, non_blocking=True)

    if spikes.ndim == 5:
        # [T,B,1,28,28]
        T, B, C, H, W = spikes.shape
        if (C, H, W) != (1, 28, 28):
            raise ValueError(f"Expected [T,B,1,28,28], got {tuple(spikes.shape)}")
        return spikes

    if spikes.ndim == 3:
        T, B, N = spikes.shape
        if N != 784:
            raise ValueError(f"Expected last dim 784, got {N}")
        spikes_tbn = spikes
    elif spikes.ndim == 4:
        T, B, C, N = spikes.shape
        if C != 1 or N != 784:
            raise ValueError(f"Expected [T,B,1,784], got {tuple(spikes.shape)}")
        spikes_tbn = spikes[:, :, 0, :]
    else:
        raise ValueError(f"Unexpected spikes ndim={spikes.ndim}")

    return spikes_tbn.view(T, B, 1, 28, 28)


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
    """Build a 1-conv-layer CSNN.

    Notes:
    - We keep a single conv layer to stay close to bio STDP setups.
    - Optional competition: simple per-position winner-take-all (WTA) inhibition.
      When enabled, after each simulation step we keep only the max-spiking
      channel at each (h,w) and zero-out the rest.
    """
    device = cfg.torch_device()

    net = Network()

    # Conv2dConnection requires explicit 3D shapes (C,H,W) on source/target layers.
    input_layer = Input(shape=(1, 28, 28), traces=True)

    # Conv output spatial size: (28 + 2*pad - kernel)/stride + 1
    h = (28 + 2 * cfg.c1_pad - cfg.c1_kernel) // cfg.c1_stride + 1
    w = (28 + 2 * cfg.c1_pad - cfg.c1_kernel) // cfg.c1_stride + 1
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
                if bool(getattr(cfg, "w_norm_enable", False)):
                    w = conn.w  # [out_ch, in_ch, kH, kW]
                    if torch.is_tensor(w) and w.ndim == 4:
                        tgt = float(getattr(cfg, "w_norm_target", 78.4))
                        eps = 1e-8
                        sabs = w.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
                        scale = (tgt / sabs).clamp_max(1.0)
                        conn.w.data = w * scale

                s = conv_lif.s
                if not (torch.is_tensor(s) and s.ndim == 4 and s.numel() > 0):
                    return

                # adaptive threshold update (per-timestep)
                if bool(getattr(cfg, "adapt_thresh_enable", False)):
                    frozen_stdp = False
                    try:
                        ur = getattr(conn, "update_rule", None)
                        nu = getattr(ur, "nu", None)
                        if nu is not None and isinstance(nu, (tuple, list)) and len(nu) == 2:
                            n0 = float(torch.as_tensor(nu[0]).abs().sum().item())
                            n1 = float(torch.as_tensor(nu[1]).abs().sum().item())
                            frozen_stdp = (n0 + n1) <= 0.0
                    except Exception:
                        frozen_stdp = False

                    theta = conv_lif.theta
                    if (theta is None) or (not torch.is_tensor(theta)) or (theta.ndim != 3):
                        # [C,H,W]
                        theta = torch.zeros_like(s[0])

                    if not frozen_stdp:
                        tau = float(getattr(cfg, "tau_theta", 1e4))
                        theta = theta * (1.0 - 1.0 / max(1.0, tau))
                        # s is [B,C,H,W]; update theta using batch-sum spikes at this timestep
                        theta = theta + float(getattr(cfg, "theta_plus", 0.05)) * s.detach().sum(dim=0)
                        theta = theta.clamp_(min=0.0, max=10.0)
                        conv_lif.theta = theta

                    conv_lif.thresh = conv_lif.thresh_base + theta

                # competition
                if bool(getattr(cfg, "ei_enable", False)) and inhib_lif is not None:
                    exc = float(getattr(cfg, "ei_exc", 22.5))
                    inh = float(getattr(cfg, "ei_inh", 120.0))
                    inhib_lif.forward(exc * s.detach())
                    i_s = inhib_lif.s.detach().to(s.dtype)
                    conv_lif.v = conv_lif.v - inh * i_s
                    conv_lif.s = conv_lif.v >= conv_lif.thresh
                elif bool(getattr(cfg, "wta_enable", False)) or bool(getattr(cfg, "local_inhib_enable", False)):
                    m = _winner_mask(s)
                    if bool(getattr(cfg, "wta_enable", False)):
                        conv_lif.s = s * m.to(s.dtype)
                    elif bool(getattr(cfg, "local_inhib_enable", False)):
                        g = float(getattr(cfg, "local_inhib_strength", 0.7))
                        conv_lif.s = s * (m.to(s.dtype) + (1.0 - g) * (~m).to(s.dtype))
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
                for t in range(time):
                    step_inputs = dict(inputs)
                    step_inputs["Input"] = x[t : t + 1]
                    out = _orig_run(inputs=step_inputs, time=1, **rest)
                    _after_step_hooks()
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

    return net, input_layer, conv_lif, conn
