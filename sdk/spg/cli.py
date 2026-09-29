"""Command-line interface: ``spg track | diff | plot | view | demo | edit-demo``.

Examples::

    spg plot experiments/results/train_gpt2_small_geometry/signals.csv
    spg track --config experiments/configs/train_gpt2_small_geometry.yaml
    spg track --config experiments/configs/train_gpt2_small_geometry_control.yaml
    spg track --config experiments/configs/train_gpt2_small_geometry_control_band.yaml
    spg diff --config experiments/configs/train_gpt2_small_geometry.yaml
    spg diff --features --config experiments/configs/train_gpt2_small_geometry.yaml
    spg view --run-dir experiments/results/train_gpt2_small_geometry --open
    spg demo
    spg edit-demo
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
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
    from spg import tui
    from spg.plot import plot

    out = plot(args.csv, out=args.out)
    print(tui.ok("wrote") + " " + tui.dim(str(out)))
    return 0


def _cmd_track(args: argparse.Namespace) -> int:
    from spg import tui

    _ensure_repo_on_path()
    from experiments.train_with_geometry import load_config, train

    cfg = load_config(Path(args.config))
    result = train(cfg, source_config=Path(args.config))
    print(tui.kv("final_loss=", f"{result['final_loss']:.4f}"))
    print(tui.kv("signals=", result["signals_csv"]))
    print(tui.ok("wrote") + " " + tui.dim(str(result["result_json"])))
    print(tui.dim("Plot with:"))
    print(tui.geo(f"  spg plot {result['signals_csv']}"))
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    from spg import tui
    from spg.view import write_view_html

    out = write_view_html(Path(args.run_dir), out=args.out)
    print(tui.ok("wrote") + " " + tui.dim(str(out)))
    if args.open:
        webbrowser.open(out.resolve().as_uri())
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


def _cmd_demo(_args: argparse.Namespace) -> int:
    """Toy sanity: GeoControl point + band + FeatureRelativeControl (no model)."""
    from spg import FeatureRelativeControl, GeoControl, format_control_line, tui

    print(tui.header("spg demo") + tui.dim("  point target → interference=0.35 @ L8\n"))
    control = GeoControl(
        layer=8,
        metric="interference_mean",
        target=0.35,
        weight=1e-2,
        enabled=True,
    )
    for step, cur in enumerate([0.55, 0.48, 0.40, 0.36, 0.35, 0.35]):
        geo = {
            "L8_interference_mean": cur,
            "L8_spectral_participation": 0.2,
            "L4_interference_mean": 0.4,
        }
        pen = control.penalty(geo)
        st = control.status(geo)
        loss = 1.0 + float(pen)
        print(format_control_line(step=step, loss=loss, geometry=geo, control=st))
        print(tui.penalty_line(f"         penalty.item()={float(pen):.6f}"))

    print(tui.header("\nspg demo") + tui.dim("  band [0.30, 0.40] (hinge outside)\n"))
    band = GeoControl(
        layer=8,
        metric="interference_mean",
        target=0.35,
        weight=1e-2,
        mode="band",
        lo=0.30,
        hi=0.40,
        enabled=True,
    )
    for step, cur in enumerate([0.22, 0.28, 0.35, 0.42, 0.50]):
        geo = {"L8_interference_mean": cur}
        pen = band.penalty(geo)
        st = band.status(geo)
        print(format_control_line(step=step, loss=float(pen), geometry=geo, control=st))
        print(tui.penalty_line(f"         penalty.item()={float(pen):.6f}"))

    print(tui.header("\nspg demo") + tui.dim("  feature-relative (shrink related, margin on unrelated)\n"))
    feat = FeatureRelativeControl(
        weight_related=1e-2,
        weight_unrelated=1e-2,
        margin=0.15,
        enabled=True,
    )
    for step, (rel, unrel) in enumerate(
        [(0.40, 0.05), (0.25, 0.10), (0.10, 0.18), (0.05, 0.22)]
    ):
        pen = feat.penalty(rel, unrel)
        st = feat.status(rel, unrel)
        print(
            f"{tui.kv('step=', step)}  "
            f"{tui.kv('related=', f'{rel:.2f}')} "
            f"{tui.kv('unrelated=', f'{unrel:.2f}')}  "
            f"{tui.penalty_line(f'pen={st.penalty:.4g}')}  "
            f"(margin={st.margin})"
        )
        print(tui.penalty_line(f"         penalty.item()={float(pen):.6f}"))

    print()
    print(tui.ok("ok") + tui.dim("  hero train: loss = lm_loss + control.penalty_on_value(metric_t)"))
    print(tui.dim("    edit-time: spg edit-demo"))
    return 0


def _cmd_edit_demo(args: argparse.Namespace) -> int:
    """Thin edit-time GeoControl nudge (one MLP W_out)."""
    _ensure_repo_on_path()
    from experiments.edit_geo_nudge import main as edit_main

    argv: List[str] = []
    if args.config is not None:
        argv += ["--config", str(args.config)]
    old = sys.argv
    try:
        sys.argv = ["spg-edit-demo", *argv]
        edit_main()
    finally:
        sys.argv = old
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = _ensure_repo_on_path()
    try:
        from spg.paths import CONFIGS, RESULTS

        default_cfg = CONFIGS / "train_gpt2_small_geometry.yaml"
        default_csv = RESULTS / "train_gpt2_small_geometry" / "signals.csv"
        default_run = RESULTS / "train_gpt2_small_geometry"
    except Exception:
        default_cfg = root / "experiments" / "configs" / "train_gpt2_small_geometry.yaml"
        default_csv = (
            root / "experiments" / "results" / "train_gpt2_small_geometry" / "signals.csv"
        )
        default_run = root / "experiments" / "results" / "train_gpt2_small_geometry"

    parser = argparse.ArgumentParser(
        prog="spg",
        description="Superposition Geometry — track / diff / plot / view / control",
    )
    parser.add_argument("--version", action="version", version="spg 0.1.1")
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

    p_view = sub.add_parser(
        "view",
        help="Interactive SVG neighborhood scrubber (HTML, Aquin-style morph)",
    )
    p_view.add_argument(
        "--run-dir",
        type=Path,
        default=default_run,
        help="Training run dir with signals.csv (+ optional diffs)",
    )
    p_view.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output HTML (default: <run-dir>/neighborhood.html)",
    )
    p_view.add_argument(
        "--open",
        action="store_true",
        help="Open the HTML in the default browser",
    )
    p_view.set_defaults(func=_cmd_view)

    p_demo = sub.add_parser(
        "demo",
        help="Toy GeoControl: point / band / feature-relative (no model)",
        aliases=["control-demo"],
    )
    p_demo.set_defaults(func=_cmd_demo)

    try:
        from spg.paths import CONFIGS as _CONFIGS

        default_edit_cfg = _CONFIGS / "edit_geo_nudge.yaml"
    except Exception:
        default_edit_cfg = root / "experiments" / "configs" / "edit_geo_nudge.yaml"

    p_edit = sub.add_parser(
        "edit-demo",
        help="Thin edit-time geo nudge (one MLP W_out → target/band)",
    )
    p_edit.add_argument(
        "-c",
        "--config",
        type=Path,
        default=default_edit_cfg,
        help="Edit YAML under experiments/configs/",
    )
    p_edit.set_defaults(func=_cmd_edit_demo)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
