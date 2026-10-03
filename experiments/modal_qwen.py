"""Modal GPU runner for SPG Qwen matched baseline vs control.

Auth (once, on your machine)::

  pip install modal
  modal token set --token-id ak-... --token-secret as-...

Smoke (CPU, pennies)::

  modal run experiments/modal_qwen.py::smoke

Train matched pair on GPU (only when you ask)::

  modal run experiments/modal_qwen.py --no-smoke-only \\
    --model Qwen/Qwen2.5-7B --steps 200 --seed 0

Pull results volume to local disk::

  modal volume get spg-results / /tmp/spg-results --force
"""

from __future__ import annotations

import modal

APP_NAME = "spg-qwen"
RESULTS_MOUNT = "/vol/results"
REPO_MOUNT = "/root/super"

# Ignore heavy / local-only paths when packaging the repo into the image.
_IGNORE = [
    ".venv",
    ".git",
    ".mplconfig",
    "**/__pycache__",
    "**/*.pyc",
    "experiments/results/**",
    "paper/figures/**",
    "docs/**",
    "**/.DS_Store",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        # Pin TL 3.8.1 (local): TL 4.0 removed HookedTransformer.
        # TL 3.8.1 requires transformers>=5.9.0.
        "torch>=2.1.0",
        "transformer-lens==3.8.1",
        "transformers>=5.9.0",
        "accelerate>=0.30.0",
        "huggingface-hub>=0.23.0",
        "einops>=0.7.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "matplotlib>=3.8.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    )
    .env({"HF_HOME": "/vol/hf-cache", "TRANSFORMERS_CACHE": "/vol/hf-cache"})
    .add_local_dir(
        ".",
        remote_path=REPO_MOUNT,
        ignore=_IGNORE,
    )
)

results_vol = modal.Volume.from_name("spg-results", create_if_missing=True)
hf_vol = modal.Volume.from_name("spg-hf-cache", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.function(cpu=1.0, memory=2048, timeout=120)
def smoke() -> str:
    """Cheap auth + image check (no GPU)."""
    import torch

    msg = f"ok torch={torch.__version__} cuda={torch.cuda.is_available()}"
    print(msg)
    return msg


@app.function(
    # H200 (~141GB) — 7B control needs headroom for LM graph + batched probe bank.
    gpu="H200",
    timeout=60 * 60 * 4,
    memory=131072,
    volumes={
        RESULTS_MOUNT: results_vol,
        "/vol/hf-cache": hf_vol,
    },
)
def train_pair(
    model: str = "Qwen/Qwen2.5-7B",
    steps: int = 200,
    seed: int = 0,
    seq_len: int = 64,
    batch_size: int = 1,
    lr: float = 2.0e-5,
    control_target: float = 0.35,
    control_weight: float = 5.0e-2,
    warmup_steps: int = 20,
    dtype: str = "bf16",
    skip_existing: bool = False,
    run_tag: str = "mstar",
) -> dict:
    """Matched baseline + control fine-tune on Qwen; write artifacts to spg-results."""
    import copy
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, REPO_MOUNT)

    from experiments.train_with_geometry import train
    from experiments.eval_control_lm import evaluate_pair

    # Layer picks by depth (early / mid / late).
    n_layers_guess = {
        "Qwen/Qwen2.5-0.5B": 24,
        "Qwen/Qwen2.5-1.5B": 28,
        "Qwen/Qwen2.5-3B": 36,
        "Qwen/Qwen2.5-7B": 28,
        "Qwen/Qwen2.5-7B-Instruct": 28,
        "Qwen/Qwen2.5-14B": 48,
    }.get(model, 28)
    mid = n_layers_guess // 2
    layers = [max(1, mid // 2), mid, max(mid + 1, n_layers_guess - 2)]
    control_layer = mid

    # Absolute path → resolve_under_experiments passes through (writes onto volume).
    out_root = Path(RESULTS_MOUNT) / "qwen_modal"
    out_root.mkdir(parents=True, exist_ok=True)

    def _cfg(*, control: bool, run_name: str) -> dict:
        return {
            "seed": int(seed),
            "model": model,
            "device": "cuda",
            "dtype": dtype,
            "train": {
                "corpus": "data/train_corpus_extended.txt",
                "seq_len": int(seq_len),
                "batch_size": int(batch_size),
                "max_steps": int(steps),
                "lr": float(lr),
                "weight_decay": 0.01,
                "grad_clip": 1.0,
            },
            "geometry": {
                "layers": layers,
                # Same cadence + full probe bank for watch AND control (no max_probes).
                "every_n_steps": 5,
                "bank_positions": "last",
                "top_k_neighbors": 8,
                "metrics": [
                    "interference_mean",
                    "spectral_participation",
                    "coactivation_overlap",
                ],
                "probe_prompts": None,
            },
            "logging": {
                # Absolute → lands on Modal volume `/vol/results`
                "results_dir": str(out_root),
                "run_name": run_name,
                "live_print": True,
                "save_every_n_steps": max(50, int(steps) // 4),
            },
            "control": {
                "enabled": bool(control),
                "layer": int(control_layer),
                "metric": "interference_mean",
                "mode": "point",
                "target": float(control_target),
                "weight": float(control_weight),
                "every_n_steps": 5,
                "warmup_steps": int(warmup_steps),
                "abort_on_warmup_fail": True,
                "abort_on_warn": False,
            },
            "diff": {
                "top_k_neighbors": 8,
                "save_neighborhood_plot": True,
                "selected_features": [
                    {
                        "name": "eiffel_located_in",
                        "prompt": "The Eiffel Tower is located in",
                        "position": "last",
                    },
                    {
                        "name": "louvre_located_in",
                        "prompt": "The Louvre Museum is located in",
                        "position": "last",
                    },
                    {
                        "name": "superposition_pack",
                        "prompt": "Superposition allows neural networks to represent more features than",
                        "position": "last",
                    },
                ],
                "cross_feature_pairs": [
                    ["eiffel_located_in", "louvre_located_in"],
                    ["eiffel_located_in", "superposition_pack"],
                    ["louvre_located_in", "superposition_pack"],
                ],
            },
        }

    # Paths helper resolves under experiments/; chdir into mounted repo experiments.
    import os

    os.chdir(f"{REPO_MOUNT}/experiments")

    tag = model.split("/")[-1].replace(".", "").replace("-", "").lower()
    suffix = f"_{run_tag}" if run_tag else ""
    base_name = f"train_{tag}_geometry{suffix}_s{seed}"
    ctrl_name = f"train_{tag}_geometry_control{suffix}_s{seed}"
    base_dir = out_root / base_name
    ctrl_dir = out_root / ctrl_name

    import torch

    print(
        f"baseline → {base_name}  model={model} steps={steps} dtype={dtype} "
        f"λ={control_weight} target={control_target}",
        flush=True,
    )
    if skip_existing and (base_dir / "model_final.pt").is_file():
        print(f"skip baseline (exists): {base_dir}", flush=True)
        base_result = {"final_loss": None, "final_geometry": None, "skipped": True}
        tr = base_dir / "train_result.json"
        if tr.is_file():
            base_result = json.loads(tr.read_text())
            base_result["skipped"] = True
    else:
        if base_dir.exists():
            import shutil

            print(f"clearing old baseline dir {base_dir}", flush=True)
            shutil.rmtree(base_dir, ignore_errors=True)
        base_result = train(_cfg(control=False, run_name=base_name))
        results_vol.commit()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"control → {ctrl_name}", flush=True)
    if ctrl_dir.exists():
        import shutil

        print(f"clearing old control dir {ctrl_dir}", flush=True)
        shutil.rmtree(ctrl_dir, ignore_errors=True)

    ctrl_result = train(_cfg(control=True, run_name=ctrl_name))
    results_vol.commit()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_corpus = Path(REPO_MOUNT) / "experiments" / "data" / "eval_corpus.txt"
    train_corpus = Path(REPO_MOUNT) / "experiments" / "data" / "train_corpus_extended.txt"

    print("held-out LM eval…", flush=True)
    report = evaluate_pair(
        baseline_dir=base_dir,
        control_dir=ctrl_dir,
        eval_corpus=eval_corpus,
        train_corpus=train_corpus,
        model_name=model,
        device="cuda",
        seq_len=int(seq_len),
        batch_size=int(batch_size),
        seed=int(seed),
        include_pretrained=False,
    )

    v = report["verdict"]
    mid_key = f"L{control_layer}_interference_mean"
    b_mid = (base_result.get("final_geometry") or {}).get(mid_key)
    c_mid = (ctrl_result.get("final_geometry") or {}).get(mid_key)
    summary = {
        "model": model,
        "seed": seed,
        "steps": steps,
        "run_tag": run_tag,
        "control_weight": control_weight,
        "control_target": control_target,
        "layers": layers,
        "control_layer": control_layer,
        "baseline_dir": str(base_dir),
        "control_dir": str(ctrl_dir),
        "eval_loss_baseline": report["runs"]["baseline"]["eval"]["mean_loss"],
        "eval_loss_control": report["runs"]["control"]["eval"]["mean_loss"],
        "delta_eval_loss": v["delta_eval_loss_control_minus_baseline"],
        "control_wins_vs_baseline": v["control_lower_eval_loss"],
        "baseline_final_loss": base_result.get("final_loss"),
        "control_final_loss": ctrl_result.get("final_loss"),
        "baseline_final_geometry": base_result.get("final_geometry"),
        "control_final_geometry": ctrl_result.get("final_geometry"),
        "abs_err_baseline_to_target": (
            None if b_mid is None else abs(float(b_mid) - float(control_target))
        ),
        "abs_err_control_to_target": (
            None if c_mid is None else abs(float(c_mid) - float(control_target))
        ),
        "control_closer_to_target": (
            None
            if b_mid is None or c_mid is None
            else abs(float(c_mid) - float(control_target))
            < abs(float(b_mid) - float(control_target))
        ),
    }
    summary_path = out_root / f"summary_{tag}{suffix}_s{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    eval_path = out_root / f"held_out_lm_eval_{tag}{suffix}_s{seed}.json"
    eval_path.write_text(json.dumps(report, indent=2) + "\n")
    results_vol.commit()
    hf_vol.commit()

    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(
    gpu="H200",
    timeout=60 * 60 * 2,
    memory=131072,
    volumes={
        RESULTS_MOUNT: results_vol,
        "/vol/hf-cache": hf_vol,
    },
)
def postprocess_pair(
    baseline_dir: str = "/vol/results/qwen_modal/train_qwen257b_geometry_mstar_s0",
    control_dir: str = "/vol/results/qwen_modal/train_qwen257b_geometry_control_mstar_s0",
    pair_name: str = "qwen257b_mstar_pair_s0",
    dtype: str = "bf16",
) -> dict:
    """Cross-feature diffs + compare + out-of-metric validation for a finished pair."""
    import json
    import os
    import sys
    from pathlib import Path

    import torch
    import yaml

    os.chdir(f"{REPO_MOUNT}/experiments")
    sys.path.insert(0, REPO_MOUNT)
    sys.path.insert(0, f"{REPO_MOUNT}/experiments")

    from experiments.compare_control_vs_baseline import compare, summarize_run
    from experiments.diff_features import resolve_pairs, run_cross_feature_diff
    from experiments.validate_control_geometry import validate_control_pair
    from scripts.helpers.geometry import DEFAULT_PROBE_PROMPTS

    base_dir = Path(baseline_dir)
    ctrl_dir = Path(control_dir)
    out_root = base_dir.parent
    pair_dir = out_root / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    def _run_diff(run_dir: Path) -> dict:
        frozen = run_dir / "config.frozen.yaml"
        cfg = yaml.safe_load(frozen.read_text()) if frozen.is_file() else {}
        meta = json.loads((run_dir / "meta.json").read_text())
        geo_cfg = cfg.get("geometry") or meta.get("geometry") or {}
        diff_cfg = cfg.get("diff") or {}
        layers = [int(L) for L in (geo_cfg.get("layers") or [7, 14, 26])]
        metrics = list(
            geo_cfg.get(
                "metrics",
                ["interference_mean", "spectral_participation", "coactivation_overlap"],
            )
        )
        probes = list(geo_cfg.get("probe_prompts") or DEFAULT_PROBE_PROMPTS)
        features = list(diff_cfg.get("selected_features") or [])
        if not features:
            features = [
                {
                    "name": "eiffel_located_in",
                    "prompt": "The Eiffel Tower is located in",
                    "position": "last",
                },
                {
                    "name": "louvre_located_in",
                    "prompt": "The Louvre Museum is located in",
                    "position": "last",
                },
                {
                    "name": "superposition_pack",
                    "prompt": "Superposition allows neural networks to represent more features than",
                    "position": "last",
                },
            ]
        pairs = resolve_pairs(features, diff_cfg.get("cross_feature_pairs"), None)
        # Prefer final aligned ckpt step.
        from scripts.helpers.checkpoint_geometry import list_aligned_checkpoints

        rows = list_aligned_checkpoints(run_dir / "checkpoints")
        step = int(rows[-1]["step"]) if rows else 199
        print(f"diff_features → {run_dir.name} @ step={step}", flush=True)
        return run_cross_feature_diff(
            run_dir=run_dir,
            step=step,
            model_name=str(cfg.get("model") or meta.get("model")),
            device="cuda",
            layers=layers,
            features=features,
            pairs=pairs,
            probe_prompts=probes,
            bank_positions=str(geo_cfg.get("bank_positions", "last")),
            top_k=int(geo_cfg.get("top_k_neighbors", 8)),
            metrics=metrics,
            save_plots=True,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    base_diff = _run_diff(base_dir)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ctrl_diff = _run_diff(ctrl_dir)

    base_sum = summarize_run(base_dir)
    ctrl_sum = summarize_run(ctrl_dir)
    cmp = compare(base_sum, ctrl_sum)
    compare_payload = {
        "baseline": base_sum,
        "control": ctrl_sum,
        "compare": cmp,
        "note": "Qwen2.5-7B m★ matched pair (Modal postprocess).",
    }
    compare_path = pair_dir / "compare.json"
    compare_path.write_text(json.dumps(compare_payload, indent=2) + "\n")

    print("validate_control_geometry…", flush=True)
    report = validate_control_pair(
        base_dir,
        ctrl_dir,
        device="cuda",
        loss_slack=0.15,  # 7B LM loss scale; packing still constrained
        max_train_heldout_gap=0.25,
        skip_held_out=False,
    )
    val_path = pair_dir / "out_of_metric_validation.json"
    val_path.write_text(json.dumps(report, indent=2) + "\n")
    (ctrl_dir / "out_of_metric_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    results_vol.commit()
    hf_vol.commit()

    summary = {
        "pair_dir": str(pair_dir),
        "compare": str(compare_path),
        "validation": str(val_path),
        "baseline_diff": base_diff.get("summary_json"),
        "control_diff": ctrl_diff.get("summary_json"),
        "control_closer_to_target": cmp.get("control_closer_to_target"),
        "overall_pass": report.get("overall_pass"),
        "claims": [
            {"id": c.get("id"), "pass": c.get("pass"), "detail": c.get("detail")}
            for c in (report.get("claims") or [])
        ],
    }
    (pair_dir / "postprocess_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    results_vol.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(
    smoke_only: bool = True,
    model: str = "Qwen/Qwen2.5-7B",
    steps: int = 200,
    seed: int = 0,
    dtype: str = "bf16",
    control_weight: float = 5.0e-2,
    control_target: float = 0.35,
    run_tag: str = "mstar",
    postprocess_only: bool = False,
):
    """Default entry: smoke only. Pass --no-smoke-only to train (costs GPU $)."""
    if smoke_only and not postprocess_only:
        print(smoke.remote())
        return
    if postprocess_only:
        print(postprocess_pair.remote())
        return
    print(
        train_pair.remote(
            model=model,
            steps=steps,
            seed=seed,
            dtype=dtype,
            control_weight=control_weight,
            control_target=control_target,
            run_tag=run_tag,
            skip_existing=False,
        )
    )
