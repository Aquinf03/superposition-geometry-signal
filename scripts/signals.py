"""Log loss and geometry side by side — the training-signal dashboard."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class StepRecord:
    step: int
    loss: Optional[float] = None
    geometry: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class SignalLogger:
    """Append-only logger: one row per step with loss + geometry_* columns."""

    def __init__(self, results_dir: str | Path, run_name: Optional[str] = None) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_name = run_name or f"run_{stamp}"
        self.root = Path(results_dir) / self.run_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.root / "signals.csv"
        self.meta_path = self.root / "meta.json"
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
