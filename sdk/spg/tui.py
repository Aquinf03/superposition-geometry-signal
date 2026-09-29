"""Minimal ANSI color for spg live lines and CLI.

Color groups (not every token):
  geometry block · geo_target: L# · metric→goal · (cur λ pen) · penalty.item()
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional, TextIO


TEAL = "\033[38;2;15;118;110m"    # whole geometry metrics block
ROSE = "\033[38;2;190;18;60m"     # geo_target: L#
CYAN = "\033[36m"                 # step/loss values, metric→goal
AMBER = "\033[38;2;180;83;9m"     # (cur λ pen) status group
GREEN = "\033[32m"                # penalty.item() / ok
BOLD = "\033[1m"
RESET = "\033[0m"


def color_enabled(stream: Optional[TextIO] = None) -> bool:
    env = os.environ.get("SPG_COLOR", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    if os.environ.get("NO_COLOR", ""):
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    stream = stream if stream is not None else sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def paint(text: str, *codes: str, enabled: Optional[bool] = None) -> str:
    if enabled is None:
        enabled = color_enabled()
    if not enabled or not codes:
        return text
    return f"{''.join(codes)}{text}{RESET}"


def geo_block(text: str, *, enabled: Optional[bool] = None) -> str:
    """Whole geometry metrics chunk."""
    return paint(text, TEAL, enabled=enabled)


def target_label(text: str, *, enabled: Optional[bool] = None) -> str:
    """``geo_target: L8``."""
    return paint(text, ROSE, BOLD, enabled=enabled)


def goal(text: str, *, enabled: Optional[bool] = None) -> str:
    """``interference→[0.30,0.40]`` or ``interference→0.35``."""
    return paint(text, CYAN, BOLD, enabled=enabled)


def status(text: str, *, enabled: Optional[bool] = None) -> str:
    """``(cur=… λ=… pen=…)``."""
    return paint(text, AMBER, enabled=enabled)


def penalty_line(text: str, *, enabled: Optional[bool] = None) -> str:
    """``penalty.item()=…``."""
    return paint(text, GREEN, enabled=enabled)


def val(text: str, *, enabled: Optional[bool] = None) -> str:
    return paint(text, CYAN, BOLD, enabled=enabled)


def warn(text: str, *, enabled: Optional[bool] = None) -> str:
    return paint(text, AMBER, BOLD, enabled=enabled)


def ok(text: str, *, enabled: Optional[bool] = None) -> str:
    return paint(text, GREEN, enabled=enabled)


def header(text: str, *, enabled: Optional[bool] = None) -> str:
    return paint(text, BOLD, enabled=enabled)


def dim(text: str, *, enabled: Optional[bool] = None) -> str:
    return text


def key(text: str, *, enabled: Optional[bool] = None) -> str:
    return text


def geo(text: str, *, enabled: Optional[bool] = None) -> str:
    return geo_block(text, enabled=enabled)


def ctrl(text: str, *, enabled: Optional[bool] = None) -> str:
    return target_label(text, enabled=enabled)


def kv(k: str, v: Any, *, enabled: Optional[bool] = None) -> str:
    return f"{k}{val(str(v), enabled=enabled)}"


def pipe(*, enabled: Optional[bool] = None) -> str:
    return "  |  "
