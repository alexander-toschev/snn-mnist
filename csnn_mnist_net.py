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
from bindsnet.learning import PostPre

from encoders import LatencyEncoder, PoissonEncoder


@dataclass
class CSCfg:
    time: int = 200
    device: str = "cpu"
    seed: int = 42
    N: int = 12000

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


def build_csnn(cfg: CSCfg) -> Tuple[Network, Input, LIFNodes, Connection]:
    """Build a 1-conv-layer CSNN.

    Notes:
    - We keep a single conv layer to stay close to bio STDP setups.
    - Later we can add Conv2 + pooling + E/I competition.
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

    net.add_layer(input_layer, name="Input")
    net.add_layer(conv_lif, name="Conv1")

    conn = Conv2dConnection(
        source=input_layer,
        target=conv_lif,
        kernel_size=cfg.c1_kernel,
        stride=cfg.c1_stride,
        padding=cfg.c1_pad,
        wmin=0.0,
        wmax=1.0,
    )
    # Attach STDP (PostPre) like in the FC setup.
    conn.update_rule = PostPre(
        connection=conn,
        nu=(
            torch.tensor(float(getattr(cfg, "nu_plus", 1e-4))),
            torch.tensor(float(getattr(cfg, "nu_minus", -1e-3))),
        ),
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
