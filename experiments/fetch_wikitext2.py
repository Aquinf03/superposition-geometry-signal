"""Fetch WikiText-2 raw into ``experiments/data/wikitext2/``.

Uses HuggingFace ``Salesforce/wikitext`` (wikitext-2-raw-v1). Large train file
is gitignored — re-run this script on a fresh clone.

  python experiments/fetch_wikitext2.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.helpers.paths import resolve_under_experiments


def fetch(out_dir: Path) -> dict:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    stats = {}
    for split, name in [
        ("train", "wiki.train.txt"),
        ("validation", "wiki.valid.txt"),
        ("test", "wiki.test.txt"),
    ]:
        lines = []
        for row in ds[split]:
            t = (row.get("text") or "").strip()
            if not t:
                continue
            if t.startswith("=") and t.endswith("="):
                continue
            lines.append(t)
        path = out_dir / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats[name] = {"lines": len(lines), "chars": sum(map(len, lines)), "path": str(path)}
        print(f"wrote {path}  lines={len(lines)}  chars={stats[name]['chars']}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=resolve_under_experiments("data/wikitext2"),
    )
    args = p.parse_args()
    fetch(args.out)


if __name__ == "__main__":
    main()
