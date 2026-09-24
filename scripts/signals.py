"""Log loss and geometry side by side — the training-signal tracker.

Track verb: append (step, loss, geometry_*) every N steps.
Optionally print a live line:
  step=12  loss=0.41  |  geometry: interference=0.22  spectral=0.31  coact=0.09
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


# Short names for live stdout (still full names in CSV).
_LIVE_GEO_ALIAS = {
    "interference_mean": "interference",
    "spectral_participation": "spectral",
    "coactivation_overlap": "coact",
    "geometry_interference_mean": "interference",
    "geometry_spectral_participation": "spectral",
    "geometry_coactivation_overlap": "coact",
}


@dataclass
class StepRecord:
    step: int
    loss: Optional[float] = None
    geometry: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


def format_live_line(step: int, loss: Optional[float], geometry: Optional[Dict[str, float]] = None) -> str:
    """Human live metric line: loss beside named geometry."""
    loss_s = f"{loss:.4f}" if loss is not None else "nan"
    geometry = geometry or {}
    if not geometry:
        geo_s = "(none)"
    else:
        bits = []
        for key, value in geometry.items():
            short = _LIVE_GEO_ALIAS.get(key, key.removeprefix("geometry_"))
            bits.append(f"{short}={float(value):.4f}")
        geo_s = "  ".join(bits)
    return f"step={step}  loss={loss_s}  |  geometry: {geo_s}"


class SignalLogger:
    """Append-only logger: one row per step with loss + geometry_* columns."""

    def __init__(
        self,
        results_dir: str | Path,
        run_name: Optional[str] = None,
        live_print: bool = True,
        live_stream: Any = None,
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_name = run_name or f"run_{stamp}"
        self.root = Path(results_dir) / self.run_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.root / "signals.csv"
        self.meta_path = self.root / "meta.json"
        self.live_print = bool(live_print)
        self.live_stream = live_stream if live_stream is not None else sys.stdout
        self._fields: list[str] = ["step", "loss"]
        self._rows: list[Dict[str, Any]] = []

    def log(
        self,
        step: int,
        loss: Optional[float] = None,
        geometry: Optional[Dict[str, float]] = None,
        **extra: Any,
    ) -> StepRecord:
        geometry = geometry or {}
        rec = StepRecord(step=step, loss=loss, geometry=dict(geometry), extra=dict(extra))
        row: Dict[str, Any] = {"step": step, "loss": loss}
        for key, value in geometry.items():
            col = key if key.startswith("geometry_") else f"geometry_{key}"
            row[col] = value
            if col not in self._fields:
                self._fields.append(col)
        for key, value in extra.items():
            row[key] = value
            if key not in self._fields:
                self._fields.append(key)
        self._rows.append(row)
        self._flush_csv()
        if self.live_print:
            line = format_live_line(step, loss, geometry)
            print(line, file=self.live_stream, flush=True)
        return rec

    def write_meta(self, **meta: Any) -> None:
        payload = {"run_name": self.run_name, **meta}
        self.meta_path.write_text(json.dumps(payload, indent=2) + "\n")

    def _flush_csv(self) -> None:
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)


def records_to_plot_arrays(rows: Iterable[Dict[str, Any]]) -> Dict[str, list]:
    """Helper shape for a loss+geometry figure."""
    rows = list(rows)
    out: Dict[str, list] = {"step": [r["step"] for r in rows], "loss": [r.get("loss") for r in rows]}
    geo_keys = sorted({k for r in rows for k in r if k.startswith("geometry_")})
    for key in geo_keys:
        out[key] = [r.get(key) for r in rows]
    return out
