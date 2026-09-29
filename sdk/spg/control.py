"""GeoControl — soft regularizer toward a geometry target or band.

Modes:
  - ``point`` — ``λ · (cur − target)²``
  - ``band``  — hinge outside ``[lo, hi]`` (0 inside)

Also: :class:`FeatureRelativeControl` to shrink related-pair Δ and push
unrelated-pair Δ apart (structure the tangle, not just scalars).

Opt-in via train YAML ``control.enabled: true``. Watch metrics stay in
``spg.geometry`` (no-grad); use :mod:`spg.grad_geometry` for backprop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Union

import torch

Number = Union[float, int]
GeomMap = Mapping[str, Number]


@dataclass
class ControlStatus:
    """Snapshot for live logging / CSV extras."""

    enabled: bool
    layer: int
    metric: str
    target: float
    weight: float
    current: Optional[float]
    error: Optional[float]
    penalty: float
    warned: Optional[str] = None
    mode: str = "point"
    lo: Optional[float] = None
    hi: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GeoControl:
    """Push one geometry metric toward a point target or into a band.

    Control is **opt-in**: construct with ``enabled=True`` (or call ``enable()``).
    Trainers should default config ``control.enabled: false``.

    Geometry keys accepted (first hit wins)::

        L{layer}_{metric}
        geometry_L{layer}_{metric}
        {metric}                     # only if single-layer / plain dict
    """

    def __init__(
        self,
        layer: int = 8,
        metric: str = "interference_mean",
        target: float = 0.35,
        weight: float = 1e-3,
        *,
        mode: str = "point",
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        enabled: bool = False,
        flat_eps: float = 1e-4,
        sat_hi: float = 0.98,
        sat_lo: float = 0.02,
    ) -> None:
        self.layer = int(layer)
        self.metric = str(metric)
        self.target = float(target)
        self.weight = float(weight)
        self.mode = str(mode or "point").lower()
        if self.mode not in ("point", "band"):
            raise ValueError(f"control mode must be 'point' or 'band', got {mode!r}")
        self.lo = None if lo is None else float(lo)
        self.hi = None if hi is None else float(hi)
        if self.mode == "band":
            if self.lo is None or self.hi is None:
                # default band around target if only target given
                width = 0.05
                self.lo = self.target - width if self.lo is None else self.lo
                self.hi = self.target + width if self.hi is None else self.hi
            if self.lo > self.hi:
                raise ValueError(f"band requires lo <= hi (got lo={self.lo}, hi={self.hi})")
        self.enabled = bool(enabled)
        self.flat_eps = float(flat_eps)
        self.sat_hi = float(sat_hi)
        self.sat_lo = float(sat_lo)
        self._last: Optional[ControlStatus] = None
        self._history: list[float] = []

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def resolve(self, geometry: GeomMap) -> Optional[float]:
        """Read the controlled metric from a flat geometry dict."""
        keys = [
            f"L{self.layer}_{self.metric}",
            f"geometry_L{self.layer}_{self.metric}",
            self.metric,
            f"geometry_{self.metric}",
        ]
        for k in keys:
            if k in geometry:
                return float(geometry[k])
        return None

    def check_safety(self, current: Optional[float]) -> Optional[str]:
        """Return a warning string if geo looks flat/saturated (caller may abort)."""
        if current is None:
            return "missing_metric"
        self._history.append(float(current))
        if len(self._history) >= 5:
            window = self._history[-5:]
            if max(window) - min(window) < self.flat_eps:
                return "flat_geometry"
        if current >= self.sat_hi or current <= self.sat_lo:
            return "saturated_geometry"
        return None

    def gate_enable(self, geometry: GeomMap, *, min_history: int = 3) -> Optional[str]:
        """Pre-flight before turning the regularizer on after watch warmup."""
        cur = self.resolve(geometry)
        if cur is None:
            return "missing_metric"
        if cur >= self.sat_hi or cur <= self.sat_lo:
            return "saturated_geometry"
        if len(self._history) < min_history:
            return "warmup_too_short"
        window = self._history[-min(len(self._history), 5) :]
        if max(window) - min(window) < self.flat_eps:
            return "flat_geometry"
        return None

    def _error_tensor(self, current: torch.Tensor) -> torch.Tensor:
        if self.mode == "band":
            assert self.lo is not None and self.hi is not None
            lo = torch.as_tensor(self.lo, device=current.device, dtype=current.dtype)
            hi = torch.as_tensor(self.hi, device=current.device, dtype=current.dtype)
            below = torch.relu(lo - current)
            above = torch.relu(current - hi)
            return below + above
        tgt = torch.as_tensor(self.target, device=current.device, dtype=current.dtype)
        return current - tgt

    def penalty_on_value(self, current: torch.Tensor) -> torch.Tensor:
        """``λ · err²`` where err is (cur−target) or hinge distance outside band."""
        if not self.enabled:
            return current * 0
        err = self._error_tensor(current)
        return self.weight * err**2

    def penalty(
        self,
        geometry: GeomMap,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Penalty from a float geometry dict (no grad through the model)."""
        if not self.enabled:
            return torch.zeros((), device=device, dtype=dtype or torch.float32)

        cur = self.resolve(geometry)
        if cur is None:
            return torch.zeros((), device=device, dtype=dtype or torch.float32)

        cur_t = torch.tensor(cur, device=device, dtype=dtype or torch.float32)
        return self.penalty_on_value(cur_t)

    def status(self, geometry: GeomMap) -> ControlStatus:
        cur = self.resolve(geometry)
        warned = self.check_safety(cur) if self.enabled else None
        if cur is None:
            err = None
            pen = 0.0
        elif self.mode == "band" and self.lo is not None and self.hi is not None:
            if cur < self.lo:
                err = cur - self.lo
            elif cur > self.hi:
                err = cur - self.hi
            else:
                err = 0.0
            pen = float(self.weight * err**2) if self.enabled else 0.0
        else:
            err = float(cur) - self.target
            pen = float(self.weight * err**2) if self.enabled else 0.0
        st = ControlStatus(
            enabled=self.enabled,
            layer=self.layer,
            metric=self.metric,
            target=self.target,
            weight=self.weight,
            current=cur,
            error=err,
            penalty=pen,
            warned=warned,
            mode=self.mode,
            lo=self.lo,
            hi=self.hi,
        )
        self._last = st
        return st

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]]) -> "GeoControl":
        """Build from YAML ``control:`` block; disabled if missing/false."""
        if not cfg or not cfg.get("enabled", False):
            return cls(enabled=False)
        mode = str(cfg.get("mode", "point")).lower()
        lo = cfg.get("lo", cfg.get("band_lo"))
        hi = cfg.get("hi", cfg.get("band_hi"))
        return cls(
            layer=int(cfg.get("layer", cfg.get("layers", [8])[0] if cfg.get("layers") else 8)),
            metric=str(cfg.get("metric", "interference_mean")),
            target=float(cfg.get("target", 0.35)),
            weight=float(cfg.get("weight", cfg.get("lambda", 1e-3))),
            mode=mode,
            lo=None if lo is None else float(lo),
            hi=None if hi is None else float(hi),
            enabled=True,
            flat_eps=float(cfg.get("flat_eps", 1e-4)),
            sat_hi=float(cfg.get("sat_hi", 0.98)),
            sat_lo=float(cfg.get("sat_lo", 0.02)),
        )


@dataclass
class FeatureRelativeStatus:
    enabled: bool
    related_score: Optional[float]
    unrelated_score: Optional[float]
    margin: float
    weight_related: float
    weight_unrelated: float
    penalty: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureRelativeControl:
    """Structure the tangle: shrink related-pair Δ, keep unrelated Δ above a margin.

    ``related_score`` / ``unrelated_score`` are Diff summary scores (lower = closer).
    Penalty::

        λ_r · related²  +  λ_u · relu(margin − unrelated)²
    """

    def __init__(
        self,
        *,
        weight_related: float = 1e-2,
        weight_unrelated: float = 1e-2,
        margin: float = 0.15,
        enabled: bool = False,
    ) -> None:
        self.weight_related = float(weight_related)
        self.weight_unrelated = float(weight_unrelated)
        self.margin = float(margin)
        self.enabled = bool(enabled)

    def penalty_on_scores(
        self,
        related: torch.Tensor,
        unrelated: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            return related * 0
        marg = torch.as_tensor(self.margin, device=related.device, dtype=related.dtype)
        pen_r = self.weight_related * related**2
        pen_u = self.weight_unrelated * torch.relu(marg - unrelated) ** 2
        return pen_r + pen_u

    def penalty(self, related: float, unrelated: float) -> torch.Tensor:
        r = torch.tensor(float(related))
        u = torch.tensor(float(unrelated))
        return self.penalty_on_scores(r, u)

    def status(self, related: float, unrelated: float) -> FeatureRelativeStatus:
        pen = float(self.penalty(related, unrelated).item()) if self.enabled else 0.0
        return FeatureRelativeStatus(
            enabled=self.enabled,
            related_score=float(related),
            unrelated_score=float(unrelated),
            margin=self.margin,
            weight_related=self.weight_related,
            weight_unrelated=self.weight_unrelated,
            penalty=pen,
        )

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]]) -> "FeatureRelativeControl":
        if not cfg or not cfg.get("enabled", False):
            return cls(enabled=False)
        return cls(
            weight_related=float(cfg.get("weight_related", cfg.get("lambda_related", 1e-2))),
            weight_unrelated=float(
                cfg.get("weight_unrelated", cfg.get("lambda_unrelated", 1e-2))
            ),
            margin=float(cfg.get("margin", 0.15)),
            enabled=True,
        )
