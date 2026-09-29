"""Live stdout line with optional geo_target."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from spg.control import ControlStatus
from spg.signals import format_live_line

Number = Union[float, int]


def format_control_line(
    step: int,
    loss: Optional[float],
    geometry: Optional[Mapping[str, Number]] = None,
    control: Optional[Union[ControlStatus, Mapping[str, Any]]] = None,
    *,
    color: Optional[bool] = None,
) -> str:
    """Watch line + ``| geo_target: …`` when control is enabled.

    Color groups:
      geo_target: L#  ·  metric→goal  ·  (cur λ pen)
    """
    from spg import tui

    if color is None:
        color = tui.color_enabled()

    base = format_live_line(step, loss, dict(geometry or {}), color=color)
    if control is None:
        return base

    if isinstance(control, ControlStatus):
        st = control
    else:
        st = ControlStatus(
            enabled=bool(control.get("enabled", False)),
            layer=int(control.get("layer", -1)),
            metric=str(control.get("metric", "?")),
            target=float(control.get("target", 0.0)),
            weight=float(control.get("weight", 0.0)),
            current=control.get("current"),  # type: ignore[arg-type]
            error=control.get("error"),  # type: ignore[arg-type]
            penalty=float(control.get("penalty", 0.0)),
            warned=control.get("warned"),  # type: ignore[arg-type]
            mode=str(control.get("mode", "point")),
            lo=control.get("lo"),  # type: ignore[arg-type]
            hi=control.get("hi"),  # type: ignore[arg-type]
        )

    if not st.enabled:
        return base

    short = {
        "interference_mean": "interference",
        "spectral_participation": "spectral",
        "coactivation_overlap": "coact",
    }.get(st.metric, st.metric)
    cur_s = "nan" if st.current is None else f"{st.current:.4f}"
    if st.mode == "band" and st.lo is not None and st.hi is not None:
        goal = f"[{st.lo:.4f},{st.hi:.4f}]"
    else:
        goal = f"{st.target:.4f}"

    # geo_target: L8   |   interference→[0.30,0.40]   |   (cur=… λ=… pen=…)
    head = tui.target_label(f"geo_target: L{st.layer}", enabled=color)
    arrow = tui.goal(f"{short}→{goal}", enabled=color)
    paren = tui.status(
        f"(cur={cur_s} λ={st.weight:g} pen={st.penalty:.4g})",
        enabled=color,
    )
    bit = f"{head} {arrow} {paren}"
    if st.warned:
        bit += " " + tui.warn(f"WARN={st.warned}", enabled=color)
    return base + tui.pipe() + bit
