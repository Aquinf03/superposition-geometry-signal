"""Hero run: fine-tune a small LM and print live loss | geometry every N steps.

Uses TransformerLens GPT-2 small + a local text corpus (no edit algorithms).

Example live line (multi-layer):
  step=12  loss=0.41  |  geometry L4: interference=… spectral=… coact=…  |  L8: …  |  L11: …

Optional soft control (``spg.GeoControl``, off by default)::

  control:
    enabled: true
    layer: 8
    metric: interference_mean
    target: 0.35
    weight: 1.0e-2

Run (you run this):
  python experiments/train_with_geometry.py --config experiments/configs/train_gpt2_small_geometry.yaml
  python experiments/train_with_geometry.py --config experiments/configs/train_gpt2_small_geometry_control.yaml
  python -m scripts.helpers.plot_signals --csv experiments/results/train_gpt2_small_geometry/signals.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.helpers.checkpoint_geometry import save_aligned_checkpoint, write_aligned_geometry
from scripts.helpers.geometry import (
    DEFAULT_PROBE_PROMPTS,
    collect_mlp_out_banks,
    compute_geometry_metrics,
)
from scripts.helpers.paths import CONFIGS, resolve_under_experiments
from scripts.helpers.run_artifacts import save_run_artifacts
from scripts.helpers.seed import set_seed
from scripts.helpers.signals import SignalLogger


def resolve_probe_prompts(geo_cfg: Dict[str, Any]) -> List[str]:
    """Resolve watch/control probe bank.

    Priority:
      1. ``probe_prompts`` list in YAML
      2. ``probe_prompts_file`` — JSON list of strings, or list of
         ``{prompt: ...}`` objects
      3. ``DEFAULT_PROBE_PROMPTS``

    Optional ``max_probes`` keeps the first N (deterministic) so control
    stays tied to the task bank without exploding step cost.
    """
    probes: List[str] = []
    raw = geo_cfg.get("probe_prompts")
    if raw:
        probes = [str(p) for p in raw]
    else:
        file_key = geo_cfg.get("probe_prompts_file")
        if file_key:
            path = resolve_under_experiments(str(file_key))
            blob = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(blob, list):
                for item in blob:
                    if isinstance(item, str):
                        probes.append(item)
                    elif isinstance(item, dict) and item.get("prompt"):
                        probes.append(str(item["prompt"]))
            elif isinstance(blob, dict) and "prompts" in blob:
                probes = [str(p) for p in blob["prompts"]]
            else:
                raise ValueError(f"unrecognized probe_prompts_file format: {path}")
            if not probes:
                raise ValueError(f"no prompts in {path}")
        else:
            probes = list(DEFAULT_PROBE_PROMPTS)

    max_n = geo_cfg.get("max_probes")
    if max_n is not None:
        probes = probes[: int(max_n)]
    if len(probes) < 2:
        raise ValueError(f"need ≥2 probe prompts for geometry, got {len(probes)}")
    return probes


def _load_geocontrol(ctrl_cfg: Optional[Dict[str, Any]]):
    """Import GeoControl from stable ``spg``."""
    from spg import (
        GeoControl,
        collect_mlp_out_bank_grad,
        format_control_line,
        mean_interference_tensor,
    )

    control = GeoControl.from_config(ctrl_cfg)
    return control, collect_mlp_out_bank_grad, mean_interference_tensor, format_control_line


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_layers(geo_cfg: Dict[str, Any]) -> List[int]:
    """Prefer ``layers: [...]``; fall back to single ``layer``."""
    if geo_cfg.get("layers"):
        return [int(L) for L in geo_cfg["layers"]]
    return [int(geo_cfg.get("layer", 8))]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def load_corpus_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty corpus: {path}")
    return lines


def build_token_blocks(model, lines: List[str], seq_len: int) -> torch.Tensor:
    """Pack corpus into contiguous token blocks of shape [n_blocks, seq_len]."""
    ids: List[int] = []
    eos = int(model.tokenizer.eos_token_id)
    for line in lines:
        tok = model.to_tokens(line, prepend_bos=False)[0].tolist()
        ids.extend(int(i) for i in tok)
        ids.append(eos)

    if len(ids) < seq_len:
        reps = (seq_len // max(len(ids), 1)) + 2
        ids = ids * reps

    n_blocks = len(ids) // seq_len
    ids = ids[: n_blocks * seq_len]
    return torch.tensor(ids, dtype=torch.long).view(n_blocks, seq_len)


class BlockLoader:
    def __init__(self, blocks: torch.Tensor, batch_size: int, seed: int = 0) -> None:
        self.blocks = blocks
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self._perm = torch.randperm(blocks.shape[0], generator=self.g)
        self._i = 0

    def next(self) -> torch.Tensor:
        n = self.blocks.shape[0]
        if self._i + self.batch_size > n:
            self._perm = torch.randperm(n, generator=self.g)
            self._i = 0
        idx = self._perm[self._i : self._i + self.batch_size]
        self._i += self.batch_size
        return self.blocks[idx]


def _mean_geometry_on_bank(
    bank: torch.Tensor, *, top_k: int, metrics: List[str]
) -> Dict[str, float]:
    acc = {m: 0.0 for m in metrics}
    n = bank.shape[0]
    if n < 2:
        return {m: 0.0 for m in metrics}
    for i in range(n):
        query = bank[i]
        others = torch.cat([bank[:i], bank[i + 1 :]], dim=0)
        vals = compute_geometry_metrics(
            query, others, top_k=min(top_k, others.shape[0]), metrics=metrics
        )
        for m in metrics:
            acc[m] += vals[m]
    return {m: acc[m] / n for m in metrics}


@torch.no_grad()
def measure_geometry_layers(
    model,
    *,
    layers: List[int],
    probe_prompts: List[str],
    positions: str,
    top_k: int,
    metrics: List[str],
) -> Dict[str, float]:
    """Mean geometry per layer; keys are ``L{layer}_{metric}`` for the logger."""
    was_training = model.training
    model.eval()
    banks = collect_mlp_out_banks(
        model, layers=layers, prompts=probe_prompts, positions=positions
    )
    out: Dict[str, float] = {}
    for L in layers:
        vals = _mean_geometry_on_bank(banks[L], top_k=top_k, metrics=metrics)
        for m, v in vals.items():
            out[f"L{L}_{m}"] = v
    if was_training:
        model.train()
    return out


def causal_lm_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V], tokens: [B, T]
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
    )


def train(cfg: Dict[str, Any], source_config: Optional[Path] = None) -> Dict[str, Any]:
    seed = set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(str(cfg.get("device", "auto")))
    train_cfg = cfg["train"]
    geo_cfg = cfg.get("geometry", {})
    log_cfg = cfg.get("logging", {})
    ctrl_cfg = cfg.get("control") or {}

    control = None
    collect_bank_grad = None
    mean_interference_t = None
    format_control_line = None
    control_every = 1
    abort_on_warn = False
    warmup_steps = 0
    abort_on_warmup_fail = True
    if ctrl_cfg.get("enabled", False):
        control, collect_bank_grad, mean_interference_t, format_control_line = _load_geocontrol(
            ctrl_cfg
        )
        if control.metric != "interference_mean":
            raise ValueError(
                "spg soft regularizer currently supports metric=interference_mean only "
                f"(got {control.metric!r})"
            )
        control_every = int(ctrl_cfg.get("every_n_steps", geo_cfg.get("every_n_steps", 1)))
        abort_on_warn = bool(ctrl_cfg.get("abort_on_warn", False))
        warmup_steps = max(0, int(ctrl_cfg.get("warmup_steps", 0)))
        # Default: fail closed on a bad warmup when warmup_steps > 0
        abort_on_warmup_fail = bool(
            ctrl_cfg.get(
                "abort_on_warmup_fail",
                True if warmup_steps > 0 else abort_on_warn,
            )
        )

    from transformer_lens import HookedTransformer

    model_name = cfg.get("model", "gpt2-small")
    dtype_name = str(cfg.get("dtype") or "").lower().strip()
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    load_kwargs = {"device": device}
    if dtype_name in dtype_map:
        load_kwargs["dtype"] = dtype_map[dtype_name]
        print(f"loading {model_name} dtype={dtype_name}", flush=True)
    model = HookedTransformer.from_pretrained(model_name, **load_kwargs)
    model.train()

    corpus_path = resolve_under_experiments(train_cfg["corpus"])
    lines = load_corpus_lines(corpus_path)
    seq_len = int(train_cfg.get("seq_len", 64))
    batch_size = int(train_cfg.get("batch_size", 4))
    max_steps = int(train_cfg.get("max_steps", 40))
    lr = float(train_cfg.get("lr", 5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    blocks = build_token_blocks(model, lines, seq_len=seq_len)
    loader = BlockLoader(blocks, batch_size=batch_size, seed=seed)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    layers = resolve_layers(geo_cfg)
    every_n = int(geo_cfg.get("every_n_steps", 1))
    top_k = int(geo_cfg.get("top_k_neighbors", 8))
    positions = str(geo_cfg.get("bank_positions", "last"))
    metrics = list(
        geo_cfg.get(
            "metrics",
            ["interference_mean", "spectral_participation", "coactivation_overlap"],
        )
    )
    save_every = int(log_cfg.get("save_every_n_steps", 0) or 0)
    want_live = bool(log_cfg.get("live_print", True))
    probes = resolve_probe_prompts(geo_cfg)
    if want_live:
        src = (
            "probe_prompts_file"
            if geo_cfg.get("probe_prompts_file")
            else (
                "probe_prompts"
                if geo_cfg.get("probe_prompts")
                else "DEFAULT_PROBE_PROMPTS"
            )
        )
        print(f"geometry probes: n={len(probes)}  source={src}", flush=True)
    # When control is on we print via format_control_line (includes geo_target).
    logger = SignalLogger(
        results_dir=resolve_under_experiments(log_cfg.get("results_dir", "results")),
        run_name=log_cfg.get("run_name"),
        live_print=want_live and control is None,
    )
    artifacts = save_run_artifacts(
        logger.root,
        cfg,
        seed,
        source_config=source_config,
        extra_meta={
            "model": model_name,
            "device": device,
            "hero": "train_with_geometry",
            "layers": layers,
            "control_enabled": bool(control and control.enabled),
        },
    )
    note = "hero: fine-tune with live loss | multi-layer geometry"
    if control and control.enabled:
        note += " + spg soft geo control"
    logger.write_meta(
        seed=seed,
        model=model_name,
        device=device,
        train=train_cfg,
        geometry=geo_cfg,
        control=ctrl_cfg if control and control.enabled else {"enabled": False},
        layers=layers,
        n_blocks=int(blocks.shape[0]),
        artifacts=artifacts,
        note=note,
    )

    ckpt_dir = logger.root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    last_loss = None
    last_geo: Dict[str, float] = {}
    last_geo_step: Optional[int] = None
    last_control_status: Optional[Dict[str, Any]] = None
    aligned_ckpts: List[Dict[str, str]] = []
    # After warmup gate passes, regularizer may apply; stays False if gate fails soft.
    regularizer_armed = warmup_steps == 0
    control_phase = "off"

    def ensure_geometry(step: int) -> Dict[str, float]:
        nonlocal last_geo, last_geo_step
        if last_geo_step == step and last_geo:
            return last_geo
        last_geo = measure_geometry_layers(
            model,
            layers=layers,
            probe_prompts=probes,
            positions=positions,
            top_k=top_k,
            metrics=metrics,
        )
        last_geo_step = step
        return last_geo

    def freeze_aligned(out_dir: Path, step: int, stem: str) -> Dict[str, str]:
        geo = ensure_geometry(step)
        paths = save_aligned_checkpoint(
            out_dir,
            step=step,
            model_state=model.state_dict(),
            seed=seed,
            loss=last_loss,
            geometry=geo,
            layers=layers,
            metrics=metrics,
            bank_positions=positions,
            top_k_neighbors=top_k,
            probe_prompts=probes,
            stem=stem,
            extra={
                "model": model_name,
                "device": device,
                "control": last_control_status,
            },
        )
        aligned_ckpts.append(paths)
        return paths

    if control is not None and control.enabled and want_live:
        if control.mode == "band" and control.lo is not None and control.hi is not None:
            goal = f"[{control.lo:g},{control.hi:g}]"
        else:
            goal = f"{control.target}"
        if warmup_steps > 0:
            print(
                f"geo_control: watch warmup for {warmup_steps} steps, "
                f"then gate before λ on L{control.layer} {control.metric}→{goal} "
                f"(mode={control.mode})",
                flush=True,
            )
        else:
            print(
                f"geo_control: regularizer armed from step 0 "
                f"(L{control.layer} {control.metric}→{goal}, "
                f"mode={control.mode}, λ={control.weight:g})",
                flush=True,
            )
            control_phase = "active"

    for step in range(max_steps):
        batch = loader.next().to(device)
        logits = model(batch)
        lm_loss = causal_lm_loss(logits, batch)
        pen = lm_loss.new_zeros(())
        ctrl_metric_val: Optional[float] = None

        if control is not None and control.enabled:
            if step < warmup_steps:
                control_phase = "warmup"
            elif not regularizer_armed:
                # Gate once on watch geometry after warmup (no penalty yet this step).
                geo_gate = ensure_geometry(step)
                # Ensure history includes this gate sample if logging cadence skipped it.
                gate_reason = control.gate_enable(geo_gate)
                # If history is short, record status once then re-gate.
                if gate_reason == "warmup_too_short":
                    control.status(geo_gate)
                    gate_reason = control.gate_enable(geo_gate)
                if gate_reason is None:
                    regularizer_armed = True
                    control_phase = "active"
                    if want_live:
                        print(
                            f"geo_control: warmup OK at step={step} — regularizer ON",
                            flush=True,
                        )
                else:
                    control_phase = "blocked"
                    msg = (
                        f"geo_control: warmup gate FAILED ({gate_reason}) at step={step} "
                        "— regularizer stays OFF"
                    )
                    if abort_on_warmup_fail:
                        raise RuntimeError(msg)
                    if want_live:
                        print(msg, flush=True)

        apply_penalty = (
            control is not None
            and control.enabled
            and regularizer_armed
            and control_phase == "active"
            and step % max(control_every, 1) == 0
        )
        if apply_penalty:
            assert collect_bank_grad is not None and mean_interference_t is not None
            bank = collect_bank_grad(
                model,
                control.layer,
                probes,
                positions=positions,
            )
            metric_t = mean_interference_t(bank, top_k=top_k)
            pen = control.penalty_on_value(metric_t)
            ctrl_metric_val = float(metric_t.detach().item())

        loss = lm_loss + pen
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        last_loss = float(lm_loss.item())
        pen_f = float(pen.detach().item())

        if step % max(every_n, 1) == 0:
            geo = ensure_geometry(step)
            extras: Dict[str, Any] = {}
            st = None
            if control is not None and control.enabled:
                st = control.status(geo)
                # Prefer the differentiable metric used in the loss (same step).
                if ctrl_metric_val is not None:
                    st.current = ctrl_metric_val
                    if (
                        control.mode == "band"
                        and control.lo is not None
                        and control.hi is not None
                    ):
                        if ctrl_metric_val < control.lo:
                            st.error = ctrl_metric_val - control.lo
                        elif ctrl_metric_val > control.hi:
                            st.error = ctrl_metric_val - control.hi
                        else:
                            st.error = 0.0
                    else:
                        st.error = ctrl_metric_val - control.target
                # Live/CSV penalty = what was actually added to the loss (0 in warmup).
                st.penalty = pen_f
                last_control_status = {**st.as_dict(), "phase": control_phase}
                extras = {
                    "control_penalty": pen_f,
                    "control_total_loss": float(loss.detach().item()),
                    "control_target": control.target,
                    "control_weight": control.weight,
                    "control_layer": control.layer,
                    "control_metric": control.metric,
                    "control_mode": control.mode,
                    "control_lo": "" if control.lo is None else control.lo,
                    "control_hi": "" if control.hi is None else control.hi,
                    "control_current": st.current,
                    "control_warned": st.warned or "",
                    "control_phase": control_phase,
                    "control_armed": int(regularizer_armed),
                }
                if st.warned and control_phase == "active":
                    msg = f"geo_control WARN={st.warned} at step={step}"
                    if abort_on_warn:
                        raise RuntimeError(msg)
                    if want_live:
                        print(msg, flush=True)
            logger.log(step=step, loss=last_loss, geometry=geo, **extras)
            if want_live and control is not None and control.enabled and st is not None:
                assert format_control_line is not None
                line = format_control_line(
                    step=step, loss=last_loss, geometry=geo, control=st
                )
                if control_phase != "active":
                    from spg import tui

                    line = (
                        f"{line}{tui.pipe()}"
                        f"{tui.key('control_phase=')}"
                        f"{tui.warn(control_phase) if control_phase == 'warmup' else tui.val(control_phase)}"
                    )
                print(line, flush=True)

        if save_every > 0 and (step + 1) % save_every == 0:
            paths = freeze_aligned(ckpt_dir, step, f"step_{step:05d}")
            if want_live:
                print(
                    f"checkpoint-aligned: step={step}  "
                    f"weights={paths['weight_path']}  "
                    f"geometry={paths['geometry_path']}",
                    flush=True,
                )

    # Final: ensure an aligned pair under checkpoints/ + convenience aliases at run root
    final_step = max_steps - 1
    final_stem = f"step_{final_step:05d}"
    if not (ckpt_dir / f"{final_stem}.pt").is_file():
        ckpt_final = freeze_aligned(ckpt_dir, final_step, final_stem)
        if bool(log_cfg.get("live_print", True)):
            print(
                f"checkpoint-aligned: step={final_step}  "
                f"weights={ckpt_final['weight_path']}  "
                f"geometry={ckpt_final['geometry_path']}",
                flush=True,
            )
    else:
        # Already frozen this step (e.g. save_every landed on final); reuse paths
        geo = ensure_geometry(final_step)
        ckpt_final = {
            "weight_path": str(ckpt_dir / f"{final_stem}.pt"),
            "geometry_path": str(ckpt_dir / f"{final_stem}.geometry.json"),
            "manifest_path": str(ckpt_dir / "manifest.json"),
        }
        # Refresh sidecar in case live log ran after an earlier partial write
        write_aligned_geometry(
            Path(ckpt_final["weight_path"]),
            step=final_step,
            loss=last_loss,
            geometry=geo,
            layers=layers,
            metrics=metrics,
            bank_positions=positions,
            top_k_neighbors=top_k,
            probe_prompts=probes,
            seed=seed,
            extra={"model": model_name, "device": device},
        )

    final_weight = logger.root / "model_final.pt"
    shutil.copy2(ckpt_final["weight_path"], final_weight)
    final_geo = write_aligned_geometry(
        final_weight,
        step=final_step,
        loss=last_loss,
        geometry=last_geo,
        layers=layers,
        metrics=metrics,
        bank_positions=positions,
        top_k_neighbors=top_k,
        probe_prompts=probes,
        seed=seed,
        extra={"model": model_name, "device": device, "alias_of": ckpt_final["weight_path"]},
    )

    result = {
        "seed": seed,
        "model": model_name,
        "device": device,
        "max_steps": max_steps,
        "layers": layers,
        "final_loss": last_loss,
        "final_geometry": last_geo,
        "control": last_control_status,
        "signals_csv": str(logger.csv_path),
        "meta_json": str(logger.meta_path),
        "artifacts": artifacts,
        "model_final": str(final_weight),
        "model_final_geometry": str(final_geo),
        "checkpoints_dir": str(ckpt_dir),
        "checkpoints_manifest": str(ckpt_dir / "manifest.json"),
        "aligned_checkpoints": aligned_ckpts,
    }
    out = logger.root / "train_result.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    result["result_json"] = str(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune small LM with live loss|geometry")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS / "train_gpt2_small_geometry.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override YAML seed (for multi-seed suites)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override logging.run_name",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.run_name is not None:
        cfg.setdefault("logging", {})["run_name"] = args.run_name
    result = train(cfg, source_config=args.config)
    print(json.dumps(result, indent=2))
    print("\nDone (hero training run).")
    print(f"final_loss={result['final_loss']:.4f}")
    print(f"signals={result['signals_csv']}")
    print(f"wrote {result['result_json']}")
    print(f"checkpoints={result.get('checkpoints_manifest')}")
    print("Plot with:")
    print(f"  python -m scripts.helpers.plot_signals --csv {result['signals_csv']}")


if __name__ == "__main__":
    main()
