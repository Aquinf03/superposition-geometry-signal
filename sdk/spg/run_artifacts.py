"""Freeze config + seed artifacts into a run directory."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def save_run_artifacts(
    run_dir: Path,
    cfg: Dict[str, Any],
    seed: int,
    *,
    source_config: Optional[Path] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Write config.frozen.yaml, seed.txt, and update/create run_manifest.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    frozen = dict(cfg)
    frozen["seed"] = int(seed)
    frozen_path = run_dir / "config.frozen.yaml"
    frozen_path.write_text(yaml.safe_dump(frozen, sort_keys=False))

    seed_path = run_dir / "seed.txt"
    seed_path.write_text(f"{int(seed)}\n")

    source_path = None
    if source_config is not None and Path(source_config).is_file():
        source_path = run_dir / "config.source.yaml"
        shutil.copy2(source_config, source_path)

    manifest = {
        "seed": int(seed),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_frozen": str(frozen_path.name),
        "config_source": source_path.name if source_path else None,
        "source_config_path": str(source_config) if source_config else None,
    }
    if extra_meta:
        manifest.update(extra_meta)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "config_frozen": str(frozen_path),
        "seed_txt": str(seed_path),
        "config_source": str(source_path) if source_path else "",
        "run_manifest": str(manifest_path),
    }
