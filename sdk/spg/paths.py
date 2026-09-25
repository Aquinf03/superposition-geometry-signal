"""Repo / experiments path helpers."""

from __future__ import annotations

from pathlib import Path

# scripts/helpers/paths.py → repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"
CONFIGS = EXPERIMENTS / "configs"
DATA = EXPERIMENTS / "data"
RESULTS = EXPERIMENTS / "results"


def resolve_under_experiments(path: str | Path) -> Path:
    """Resolve a path relative to ``experiments/`` (absolute paths pass through)."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (EXPERIMENTS / p).resolve()
