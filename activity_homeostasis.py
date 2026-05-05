"""Online homeostasis for spiking activity.

Goal: keep network activity (spikes/sample) within a target corridor during training
by adjusting the Poisson input rate scale smoothly.

This is intentionally lightweight and model-agnostic. It can be used for both FC
and CSNN training loops.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class HomeostasisCfg:
    enable: bool = False

    # Corridor in spikes/sample (per image over T timesteps)
    spikes_lo: float = 3000.0
    spikes_hi: float = 5000.0
    spikes_target: float = 4000.0

    # EMA over observed spikes/sample
    ema_alpha: float = 0.01

    # Update cadence (in samples)
    update_every: int = 25
    warmup: int = 200

    # Multiplicative controller strength
    gain: float = 0.25

    # Clamp multiplier to avoid extremes
    rate_mul_min: float = 0.3
    rate_mul_max: float = 3.0


class SpikesHomeostasis:
    """Track activity and adjust a multiplicative rate scale factor."""

    def __init__(self, cfg: HomeostasisCfg, *, base_rate_scale: float):
        self.cfg = cfg
        self.base_rate_scale = float(base_rate_scale)
        self.rate_mul = 1.0
        self.ema_spikes = None  # type: float | None
        self.last_action = "init"

    def _clamp_mul(self) -> None:
        self.rate_mul = float(max(self.cfg.rate_mul_min, min(self.cfg.rate_mul_max, self.rate_mul)))

    def current_rate_scale(self) -> float:
        return float(self.base_rate_scale * self.rate_mul)

    def observe(self, spikes_per_sample: float, step: int) -> dict:
        """Observe one sample and (optionally) update the rate multiplier.

        Returns a small debug dict suitable for logs.
        """
        s = float(spikes_per_sample)
        if self.ema_spikes is None:
            self.ema_spikes = s
        else:
            a = float(self.cfg.ema_alpha)
            self.ema_spikes = (1.0 - a) * float(self.ema_spikes) + a * s

        if not self.cfg.enable:
            self.last_action = "disabled"
            return {
                "homeo_enable": False,
                "homeo_ema_spikes": float(self.ema_spikes),
                "homeo_rate_mul": float(self.rate_mul),
                "homeo_rate_scale": self.current_rate_scale(),
                "homeo_action": self.last_action,
            }

        if step < int(self.cfg.warmup) or (int(self.cfg.update_every) > 0 and (step % int(self.cfg.update_every) != 0)):
            self.last_action = "hold"
            return {
                "homeo_enable": True,
                "homeo_ema_spikes": float(self.ema_spikes),
                "homeo_rate_mul": float(self.rate_mul),
                "homeo_rate_scale": self.current_rate_scale(),
                "homeo_action": self.last_action,
            }

        lo = float(self.cfg.spikes_lo)
        hi = float(self.cfg.spikes_hi)
        tgt = float(self.cfg.spikes_target)
        ema = float(self.ema_spikes)

        # Only act when outside corridor.
        if ema > hi:
            # Too many spikes -> reduce input rate.
            err = (ema - tgt) / max(1e-6, tgt)
            self.rate_mul *= math.exp(-float(self.cfg.gain) * err)
            self.last_action = "down"
        elif ema < lo:
            # Too few spikes -> increase input rate.
            err = (tgt - ema) / max(1e-6, tgt)
            self.rate_mul *= math.exp(+float(self.cfg.gain) * err)
            self.last_action = "up"
        else:
            self.last_action = "in_band"

        self._clamp_mul()

        return {
            "homeo_enable": True,
            "homeo_ema_spikes": float(self.ema_spikes),
            "homeo_rate_mul": float(self.rate_mul),
            "homeo_rate_scale": self.current_rate_scale(),
            "homeo_action": self.last_action,
        }

