"""Checkpoint-aligned geometry freezes.

When training saves weights at step t, also write a geometry JSON for that exact
step (same probes / layers / metrics) so later ckpt A vs B diffs stay paired.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def geometry_sidecar_path(weight_path: Path) -> Path:
    """``step_00019.pt`` → ``step_00019.geometry.json``."""
    return Path(weight_path).with_suffix(".geometry.json")


def write_aligned_geometry(
    weight_path: Path,
    *,
    step: int,
    loss: Optional[float],
    geometry: Dict[str, float],
    layers: List[int],
    metrics: List[str],
    bank_positions: str,
    top_k_neighbors: int,
    probe_prompts: List[str],
    seed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write geometry JSON next to a weight checkpoint. Returns sidecar path."""
    weight_path = Path(weight_path)
    sidecar = geometry_sidecar_path(weight_path)
    payload: Dict[str, Any] = {
        "aligned": True,
        "step": int(step),
        "loss": None if loss is None else float(loss),
        "seed": int(seed),
        "layers": [int(L) for L in layers],
        "metrics": list(metrics),
        "bank_positions": bank_positions,
        "top_k_neighbors": int(top_k_neighbors),
        "probe_prompts": list(probe_prompts),
        "geometry": {k: float(v) for k, v in geometry.items()},
        "weight_file": weight_path.name,
        "weight_path": str(weight_path),
    }
    if extra:
        payload["extra"] = extra
    sidecar.write_text(json.dumps(payload, indent=2) + "\n")
    return sidecar


def load_aligned_geometry(path: Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if not data.get("aligned"):
        raise ValueError(f"not an aligned geometry freeze: {path}")
    return data


def list_aligned_checkpoints(ckpt_dir: Path) -> List[Dict[str, Any]]:
    """List ``*.pt`` files that have a matching ``*.geometry.json`` sidecar."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for weight in sorted(ckpt_dir.glob("*.pt")):
        geo = geometry_sidecar_path(weight)
        if not geo.is_file():
            continue
        snap = load_aligned_geometry(geo)
        rows.append(
            {
                "step": snap["step"],
                "loss": snap.get("loss"),
                "weight_path": str(weight),
                "geometry_path": str(geo),
                "geometry": snap.get("geometry", {}),
            }
        )
    rows.sort(key=lambda r: r["step"])
    return rows


def write_checkpoint_manifest(ckpt_dir: Path) -> Path:
    """Refresh ``checkpoints/manifest.json`` from on-disk aligned pairs."""
    ckpt_dir = Path(ckpt_dir)
    rows = list_aligned_checkpoints(ckpt_dir)
    out = ckpt_dir / "manifest.json"
    out.write_text(
        json.dumps({"n": len(rows), "checkpoints": rows}, indent=2) + "\n"
    )
    return out


def save_aligned_checkpoint(
    ckpt_dir: Path,
    *,
    step: int,
    model_state: Dict[str, Any],
    seed: int,
    loss: Optional[float],
    geometry: Dict[str, float],
    layers: List[int],
    metrics: List[str],
    bank_positions: str,
    top_k_neighbors: int,
    probe_prompts: List[str],
    stem: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Save ``stem.pt`` + ``stem.geometry.json`` and refresh the manifest."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"step_{step:05d}"
    weight_path = ckpt_dir / f"{stem}.pt"
    torch_save = _torch_save
    torch_save({"step": step, "model_state": model_state, "seed": seed}, weight_path)
    geo_path = write_aligned_geometry(
        weight_path,
        step=step,
        loss=loss,
        geometry=geometry,
        layers=layers,
        metrics=metrics,
        bank_positions=bank_positions,
        top_k_neighbors=top_k_neighbors,
        probe_prompts=probe_prompts,
        seed=seed,
        extra=extra,
    )
    manifest = write_checkpoint_manifest(ckpt_dir)
    return {
        "weight_path": str(weight_path),
        "geometry_path": str(geo_path),
        "manifest_path": str(manifest),
    }


def _torch_save(obj: Any, path: Path) -> None:
    import torch

    torch.save(obj, path)
