"""Compare a watch-only baseline run to a controlled fine-tune (same seed family).

Expects two run dirs with ``signals.csv`` (+ optional ``meta.json``).

Example::

  python experiments/compare_control_vs_baseline.py \\
    --baseline experiments/results/train_gpt2_small_geometry \\
    --control experiments/results/train_gpt2_small_geometry_control
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _series(rows: List[Dict[str, str]], key: str) -> List[Optional[float]]:
    return [_f(r, key) for r in rows]


def _last(xs: List[Optional[float]]) -> Optional[float]:
    for v in reversed(xs):
        if v is not None:
            return v
    return None


def _first(xs: List[Optional[float]]) -> Optional[float]:
    for v in xs:
        if v is not None:
            return v
    return None


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    rows = _read_csv(run_dir / "signals.csv")
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    loss = _series(rows, "loss")
    L8 = _series(rows, "geometry_L8_interference_mean")
    L4 = _series(rows, "geometry_L4_interference_mean")
    L11 = _series(rows, "geometry_L11_interference_mean")
    ctrl_cur = _series(rows, "control_current")
    ctrl_pen = _series(rows, "control_penalty")
    phases = [r.get("control_phase", "") for r in rows]
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "seed": meta.get("seed"),
        "model": meta.get("model"),
        "control_cfg": meta.get("control"),
        "n_steps": len(rows),
        "loss_first": _first(loss),
        "loss_last": _last(loss),
        "L8_interference_first": _first(L8),
        "L8_interference_last": _last(L8),
        "L8_interference_min": min((v for v in L8 if v is not None), default=None),
        "L4_interference_last": _last(L4),
        "L11_interference_last": _last(L11),
        "control_current_last": _last(ctrl_cur),
        "control_penalty_last": _last(ctrl_pen),
        "control_phases": sorted({p for p in phases if p}),
        "armed_steps": sum(1 for p in phases if p == "active"),
        "warmup_steps": sum(1 for p in phases if p == "warmup"),
    }


def compare(baseline: Dict[str, Any], control: Dict[str, Any]) -> Dict[str, Any]:
    def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return b - a

    target = None
    cfg = control.get("control_cfg") or {}
    if isinstance(cfg, dict):
        target = cfg.get("target")

    return {
        "matched_seed": baseline.get("seed") == control.get("seed"),
        "matched_n_steps": baseline.get("n_steps") == control.get("n_steps"),
        "delta_loss_last": delta(baseline.get("loss_last"), control.get("loss_last")),
        "delta_L8_interference_last": delta(
            baseline.get("L8_interference_last"), control.get("L8_interference_last")
        ),
        "delta_L8_vs_target": (
            None
            if target is None or control.get("L8_interference_last") is None
            else float(control["L8_interference_last"]) - float(target)
        ),
        "control_closer_to_target": (
            None
            if target is None
            or baseline.get("L8_interference_last") is None
            or control.get("L8_interference_last") is None
            else abs(float(control["L8_interference_last"]) - float(target))
            < abs(float(baseline["L8_interference_last"]) - float(target))
        ),
        "abs_err_baseline_to_target": (
            None
            if target is None or baseline.get("L8_interference_last") is None
            else abs(float(baseline["L8_interference_last"]) - float(target))
        ),
        "abs_err_control_to_target": (
            None
            if target is None or control.get("L8_interference_last") is None
            else abs(float(control["L8_interference_last"]) - float(target))
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "experiments/results/train_gpt2_small_geometry",
    )
    p.add_argument(
        "--control",
        type=Path,
        default=ROOT / "experiments/results/train_gpt2_small_geometry_control",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Write comparison JSON (default: <control>/../control_vs_baseline_seed0/compare.json)",
    )
    args = p.parse_args()

    base = summarize_run(args.baseline)
    ctrl = summarize_run(args.control)
    diff = compare(base, ctrl)
    payload = {
        "baseline": base,
        "control": ctrl,
        "compare": diff,
        "note": (
            "Same seed family / hero corpus fine-tunes. "
            "control_closer_to_target: |L8_ctrl − target| < |L8_base − target|."
        ),
    }

    out = args.out
    if out is None:
        out = args.control.parent / "control_vs_baseline_seed0" / "compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    c = diff
    print(
        f"matched_seed={c['matched_seed']}  "
        f"Δloss_last={c['delta_loss_last']}  "
        f"ΔL8_last={c['delta_L8_interference_last']}  "
        f"control_closer_to_target={c['control_closer_to_target']}"
    )


if __name__ == "__main__":
    main()
