"""Compare a watch-only baseline run to a controlled fine-tune (same seed family).

Expects two run dirs with ``signals.csv`` (+ optional ``meta.json``).

Example::

  python experiments/compare_control_vs_baseline.py \\
    --baseline experiments/results/modal_pull/train_qwen257b_geometry_mstar_s0 \\
    --control experiments/results/modal_pull/train_qwen257b_geometry_control_mstar_s0
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


def _control_layer(meta: Dict[str, Any], rows: List[Dict[str, str]]) -> int:
    cfg = meta.get("control") or {}
    if isinstance(cfg, dict) and cfg.get("layer") is not None:
        return int(cfg["layer"])
    # Fall back: prefer L14 / L8 columns if present, else first geometry layer.
    for cand in (14, 8, 7, 4):
        if any(f"geometry_L{cand}_interference_mean" in r for r in rows[:1] or rows):
            return cand
    for key in (rows[0] if rows else {}):
        if key.startswith("geometry_L") and key.endswith("_interference_mean"):
            return int(key.split("_")[1][1:])
    return 8


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    rows = _read_csv(run_dir / "signals.csv")
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    layer = _control_layer(meta, rows)
    loss = _series(rows, "loss")
    target_col = f"geometry_L{layer}_interference_mean"
    target_inter = _series(rows, target_col)
    ctrl_cur = _series(rows, "control_current")
    ctrl_pen = _series(rows, "control_penalty")
    phases = [r.get("control_phase", "") for r in rows]

    # Prefer frozen final geometry sidecar when present (aligned end-of-run).
    final_geo: Dict[str, Any] = {}
    for cand in (
        run_dir / "model_final.geometry.json",
        run_dir / "train_result.json",
    ):
        if not cand.is_file():
            continue
        payload = json.loads(cand.read_text())
        if "final_geometry" in payload and isinstance(payload["final_geometry"], dict):
            final_geo = payload["final_geometry"]
            break
        if any(k.startswith("L") and "_interference_mean" in k for k in payload):
            final_geo = payload
            break
    final_key = f"L{layer}_interference_mean"
    target_last = (
        float(final_geo[final_key])
        if final_key in final_geo
        else _last(target_inter)
    )

    # Also surface common watch layers when present.
    layer_last: Dict[str, Optional[float]] = {}
    for L in sorted(
        {
            int(k.split("_")[1][1:])
            for k in (rows[0] if rows else {})
            if k.startswith("geometry_L") and k.endswith("_interference_mean")
        }
    ):
        key = f"L{L}_interference_mean"
        layer_last[f"L{L}_interference_last"] = (
            float(final_geo[key]) if key in final_geo else _last(_series(rows, f"geometry_L{L}_interference_mean"))
        )

    out = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "seed": meta.get("seed"),
        "model": meta.get("model"),
        "control_cfg": meta.get("control"),
        "control_layer": layer,
        "n_steps": len(rows),
        "loss_first": _first(loss),
        "loss_last": _last(loss),
        "target_interference_first": _first(target_inter),
        "target_interference_last": target_last,
        "target_interference_min": min(
            (v for v in target_inter if v is not None), default=None
        ),
        # Backward-compatible aliases used by paper_figures (historically L8).
        "L8_interference_first": _first(target_inter),
        "L8_interference_last": target_last,
        "L8_interference_min": min(
            (v for v in target_inter if v is not None), default=None
        ),
        "control_current_last": _last(ctrl_cur),
        "control_penalty_last": _last(ctrl_pen),
        "control_phases": sorted({p for p in phases if p}),
        "armed_steps": sum(1 for p in phases if p == "active"),
        "warmup_steps": sum(1 for p in phases if p == "warmup"),
    }
    out.update(layer_last)
    return out


def compare(baseline: Dict[str, Any], control: Dict[str, Any]) -> Dict[str, Any]:
    def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return b - a

    target = None
    cfg = control.get("control_cfg") or {}
    if isinstance(cfg, dict):
        target = cfg.get("target")
    layer = int(control.get("control_layer") or baseline.get("control_layer") or 8)
    b_last = baseline.get("target_interference_last")
    c_last = control.get("target_interference_last")

    return {
        "matched_seed": baseline.get("seed") == control.get("seed"),
        "matched_n_steps": baseline.get("n_steps") == control.get("n_steps"),
        "control_layer": layer,
        "delta_loss_last": delta(baseline.get("loss_last"), control.get("loss_last")),
        "delta_target_interference_last": delta(b_last, c_last),
        "delta_L8_interference_last": delta(b_last, c_last),  # alias
        "delta_L8_vs_target": (
            None if target is None or c_last is None else float(c_last) - float(target)
        ),
        "control_closer_to_target": (
            None
            if target is None or b_last is None or c_last is None
            else abs(float(c_last) - float(target)) < abs(float(b_last) - float(target))
        ),
        "abs_err_baseline_to_target": (
            None if target is None or b_last is None else abs(float(b_last) - float(target))
        ),
        "abs_err_control_to_target": (
            None if target is None or c_last is None else abs(float(c_last) - float(target))
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
    layer = diff.get("control_layer", 8)
    payload = {
        "baseline": base,
        "control": ctrl,
        "compare": diff,
        "note": (
            "Same seed family / matched fine-tunes. "
            f"control_closer_to_target: |L{layer}_ctrl − target| < |L{layer}_base − target|."
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
        f"closer={c['control_closer_to_target']}  "
        f"abs_err_base={c['abs_err_baseline_to_target']}  "
        f"abs_err_ctrl={c['abs_err_control_to_target']}"
    )


if __name__ == "__main__":
    main()
