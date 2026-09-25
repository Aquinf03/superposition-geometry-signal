"""Command-line interface: ``spg track | diff | plot``.

Examples::

    spg plot experiments/results/train_gpt2_small_geometry/signals.csv
    spg track --config experiments/configs/train_gpt2_small_geometry.yaml
    spg diff --config experiments/configs/train_gpt2_small_geometry.yaml
    spg diff --features --config experiments/configs/train_gpt2_small_geometry.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def _ensure_repo_on_path() -> Path:
    """Editable installs + experiments/ live at the repo root."""
    # sdk/spg/cli.py → parents[2] = repo root
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _cmd_plot(args: argparse.Namespace) -> int:
    from spg.plot import plot

    out = plot(args.csv, out=args.out)
    print(f"wrote {out}")
    return 0


def _cmd_track(args: argparse.Namespace) -> int:
    _ensure_repo_on_path()
    from experiments.train_with_geometry import load_config, train

    cfg = load_config(Path(args.config))
    result = train(cfg, source_config=Path(args.config))
    print(f"final_loss={result['final_loss']:.4f}")
    print(f"signals={result['signals_csv']}")
    print(f"wrote {result['result_json']}")
    print("Plot with:")
    print(f"  spg plot {result['signals_csv']}")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    _ensure_repo_on_path()
    if args.features:
        # Cross-feature A vs B at one checkpoint
        from experiments.diff_features import main as features_main

        argv = ["--config", str(args.config)]
        if args.run_dir is not None:
            argv += ["--run-dir", str(args.run_dir)]
        if args.step is not None:
            argv += ["--step", str(args.step)]
        if args.device is not None:
            argv += ["--device", args.device]
        if args.no_plots:
            argv.append("--no-plots")
        if args.pair:
            for a, b in args.pair:
                argv += ["--pair", a, b]
        old = sys.argv
        try:
            sys.argv = ["spg-diff-features", *argv]
            features_main()
        finally:
            sys.argv = old
        return 0

    # Checkpoint A vs B (default)
    from experiments.diff_checkpoints import main as ckpt_main

    argv = ["--config", str(args.config)]
    if args.run_dir is not None:
        argv += ["--run-dir", str(args.run_dir)]
    if args.step_a is not None:
        argv += ["--step-a", str(args.step_a)]
    if args.step_b is not None:
        argv += ["--step-b", str(args.step_b)]
    if args.device is not None:
        argv += ["--device", args.device]
    if args.no_plots:
        argv.append("--no-plots")
    old = sys.argv
    try:
        sys.argv = ["spg-diff-checkpoints", *argv]
        ckpt_main()
    finally:
        sys.argv = old
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = _ensure_repo_on_path()
    try:
        from spg.paths import CONFIGS, RESULTS

        default_cfg = CONFIGS / "train_gpt2_small_geometry.yaml"
        default_csv = RESULTS / "train_gpt2_small_geometry" / "signals.csv"
    except Exception:
        default_cfg = root / "experiments" / "configs" / "train_gpt2_small_geometry.yaml"
        default_csv = (
            root / "experiments" / "results" / "train_gpt2_small_geometry" / "signals.csv"
        )

    parser = argparse.ArgumentParser(
        prog="spg",
        description="Superposition Geometry — track / diff / plot beside loss",
    )
    parser.add_argument("--version", action="version", version="spg 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plot = sub.add_parser("plot", help="Plot loss + geometry from signals.csv")
    p_plot.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=default_csv,
        help="Path to signals.csv",
    )
    p_plot.add_argument("-o", "--out", type=Path, default=None, help="Output PNG")
    p_plot.set_defaults(func=_cmd_plot)

    p_track = sub.add_parser(
        "track",
        help="Fine-tune with live loss | geometry (hero training loop)",
    )
    p_track.add_argument(
        "-c",
        "--config",
        type=Path,
        default=default_cfg,
        help="Train YAML under experiments/configs/",
    )
    p_track.set_defaults(func=_cmd_track)

    p_diff = sub.add_parser(
        "diff",
        help="Diff geometry: checkpoints (default) or --features",
    )
    p_diff.add_argument(
        "-c",
        "--config",
        type=Path,
        default=default_cfg,
        help="Train/diff YAML",
    )
    p_diff.add_argument("--run-dir", type=Path, default=None)
    p_diff.add_argument(
        "--features",
        action="store_true",
        help="Cross-feature A vs B at one ckpt (instead of ckpt A vs B)",
    )
    p_diff.add_argument("--step-a", type=int, default=None, help="Earlier ckpt step")
    p_diff.add_argument("--step-b", type=int, default=None, help="Later ckpt step")
    p_diff.add_argument("--step", type=int, default=None, help="Ckpt step for --features")
    p_diff.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("A", "B"),
        default=None,
        help="Feature pair for --features (repeatable)",
    )
    p_diff.add_argument("--device", type=str, default=None)
    p_diff.add_argument("--no-plots", action="store_true")
    p_diff.set_defaults(func=_cmd_diff)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
