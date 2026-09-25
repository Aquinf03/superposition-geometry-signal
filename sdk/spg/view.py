"""Build an interactive HTML neighborhood viewer for a training run.

Minimal distill-style canvas: play/scrub through geometry morphs over a run.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


_LAYER_RE = re.compile(r"^geometry_L(?P<layer>\d+)_(?P<metric>.+)$")


def _read_signals_csv(path: Path) -> List[Dict[str, float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out: List[Dict[str, float]] = []
    for r in rows:
        row: Dict[str, float] = {}
        for k, v in r.items():
            if v in (None, ""):
                continue
            try:
                row[k] = float(v)
            except ValueError:
                continue
        out.append(row)
    return out


def _parse_layers(rows: List[Dict[str, float]]) -> List[int]:
    layers = set()
    for r in rows:
        for k in r:
            m = _LAYER_RE.match(k)
            if m:
                layers.add(int(m.group("layer")))
    return sorted(layers)


def _frame_from_row(row: Dict[str, float], layers: List[int]) -> Dict[str, Any]:
    by_layer: Dict[str, Dict[str, float]] = {}
    for L in layers:
        by_layer[str(L)] = {
            "interference": float(row.get(f"geometry_L{L}_interference_mean", 0.0)),
            "spectral": float(row.get(f"geometry_L{L}_spectral_participation", 0.0)),
            "coact": float(row.get(f"geometry_L{L}_coactivation_overlap", 0.0)),
        }
    return {
        "step": int(row.get("step", 0)),
        "loss": float(row.get("loss", float("nan"))),
        "layers": by_layer,
    }


def _load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if path.is_file():
        return json.loads(path.read_text())
    return None


def _find_latest(glob_root: Path, pattern: str) -> Optional[Path]:
    hits = sorted(glob_root.glob(pattern))
    return hits[-1] if hits else None


def build_view_payload(run_dir: Path) -> Dict[str, Any]:
    """Assemble JSON payload embedded in the HTML viewer."""
    run_dir = Path(run_dir)
    csv_path = run_dir / "signals.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing signals.csv in {run_dir}")

    rows = _read_signals_csv(csv_path)
    layers = _parse_layers(rows)
    frames = [_frame_from_row(r, layers) for r in rows]

    feat = None
    feat_path = _find_latest(run_dir, "diff_features_step_*/diff_summary.json")
    if feat_path:
        raw = _load_optional_json(feat_path) or {}
        pairs = []
        for key, by_L in (raw.get("pair_diffs") or {}).items():
            scores = {
                str(L).lstrip("L"): float(stats["summary_score"])
                for L, stats in by_L.items()
                if isinstance(stats, dict) and "summary_score" in stats
            }
            kl = key.lower()
            related = (
                "eiffel" in kl
                and "louvre" in kl
                and "superposition" not in kl
                and "pack" not in kl
            )
            vals = list(scores.values())
            mid = [scores[k] for k in scores if k != str(max(layers or [0]))]
            pairs.append(
                {
                    "id": key,
                    "related": related,
                    "scores": scores,
                    "mean": (sum(mid) / len(mid)) if mid else (sum(vals) / max(1, len(vals))),
                }
            )
        pairs.sort(key=lambda p: p["mean"])
        feat = {
            "step": raw.get("step"),
            "pairs": pairs,
            "source": feat_path.parent.name,
        }

    ckpt = None
    ckpt_path = _find_latest(run_dir, "diff_ckpt_*_vs_*/diff_summary.json")
    if ckpt_path:
        raw = _load_optional_json(ckpt_path) or {}
        feats = []
        for fid, by_L in (raw.get("feature_diffs") or {}).items():
            layers_out = {}
            for L, s in by_L.items():
                if not isinstance(s, dict):
                    continue
                layers_out[str(L).lstrip("L")] = {
                    "summary": float(s["summary_score"]),
                    "nbr": float(s["mean_abs_neighbor_delta"]),
                }
            feats.append({"id": fid, "layers": layers_out})
        ckpt = {
            "step_a": raw.get("step_a"),
            "step_b": raw.get("step_b"),
            "features": feats,
            "aligned_summary": raw.get("aligned_summary_score"),
            "source": ckpt_path.parent.name,
        }

    manifest = _load_optional_json(run_dir / "checkpoints" / "manifest.json")
    meta = _load_optional_json(run_dir / "meta.json") or {}

    return {
        "run_name": run_dir.name,
        "layers": layers,
        "frames": frames,
        "feature_diff": feat,
        "ckpt_diff": ckpt,
        "checkpoints": (manifest or {}).get("checkpoints", []),
        "meta": {
            "model": meta.get("model"),
            "note": meta.get("note"),
            "n_steps": len(frames),
        },
    }


def render_view_html(payload: Dict[str, Any]) -> str:
    """Return a full standalone HTML document."""
    data_json = json.dumps(payload, separators=(",", ":"))
    data_json = data_json.replace("<", "\\u003c")
    return _HTML_TEMPLATE.replace("@@DATA@@", data_json)


def write_view_html(run_dir: Path, out: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    out = Path(out) if out else run_dir / "neighborhood.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = render_view_html(build_view_payload(run_dir))
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Minimal white UI + canvas scrubber (distill / t-SNE explorer style)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>spg neighborhood</title>
<style>
  * { box-sizing: border-box; }
  html, body {
    margin: 0; height: 100%;
    background: #fff;
    color: #222;
    font: 14px/1.45 Helvetica, Arial, sans-serif;
  }
  .wrap {
    display: grid;
    grid-template-columns: 1fr 260px;
    gap: 28px;
    max-width: 1100px;
    margin: 0 auto;
    padding: 28px 32px;
    min-height: 100%;
  }
  @media (max-width: 800px) {
    .wrap { grid-template-columns: 1fr; padding: 16px; gap: 16px; }
  }
  canvas {
    width: 100%;
    height: auto;
    display: block;
    background: #fff;
    cursor: grab;
  }
  canvas:active { cursor: grabbing; }
  .side { display: flex; flex-direction: column; gap: 18px; padding-top: 4px; }
  .thumbs {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
  }
  .thumb {
    aspect-ratio: 1;
    border: 1px solid #ddd;
    background: #fafafa;
    cursor: pointer;
    padding: 0;
    position: relative;
  }
  .thumb.active { border-color: #1a73e8; box-shadow: inset 0 0 0 1px #1a73e8; }
  .thumb canvas { width: 100%; height: 100%; pointer-events: none; }
  .thumb span {
    position: absolute; left: 4px; bottom: 2px;
    font-size: 10px; color: #666; background: rgba(255,255,255,.8);
  }
  .playrow { display: flex; align-items: center; gap: 10px; }
  .iconbtn {
    width: 40px; height: 40px; border-radius: 50%;
    border: none; background: #1a73e8; color: #fff;
    cursor: pointer; display: grid; place-items: center;
    padding: 0;
  }
  .iconbtn:hover { background: #1557b0; }
  .iconbtn svg { width: 16px; height: 16px; fill: currentColor; }
  .stepn {
    font-size: 15px; color: #333;
    font-variant-numeric: tabular-nums;
  }
  .slider {
    display: grid; gap: 4px;
  }
  .slider label {
    display: flex; justify-content: space-between;
    font-size: 13px; color: #333;
  }
  .slider input[type=range] {
    width: 100%; margin: 0; accent-color: #1a73e8;
  }
  .blurb { font-size: 13px; color: #555; line-height: 1.5; margin: 0; }
  .metrics { font-size: 12px; color: #444; line-height: 1.55; }
  .metrics b { font-weight: 600; color: #222; }
  .diff { font-size: 12px; color: #555; line-height: 1.5; }
  .diff .rel { color: #0a7; }
  .diff .unrel { color: #c33; }
</style>
</head>
<body>
<div class="wrap">
  <canvas id="viz" width="720" height="560"></canvas>
  <aside class="side">
    <div class="thumbs" id="thumbs"></div>
    <div class="playrow">
      <button type="button" class="iconbtn" id="btnPlay" title="Play / Pause" aria-label="Play">
        <svg id="iconPlay" viewBox="0 0 16 16"><path d="M4 2l10 6-10 6z"/></svg>
        <svg id="iconPause" viewBox="0 0 16 16" style="display:none"><path d="M3 2h4v12H3zm6 0h4v12H9z"/></svg>
      </button>
      <button type="button" class="iconbtn" id="btnReset" title="Reset" aria-label="Reset">
        <svg viewBox="0 0 16 16"><path d="M8 2a6 6 0 1 0 5.3 3.2l-1.5.6A4.4 4.4 0 1 1 8 3.6V1l3 2.5L8 6V2z"/></svg>
      </button>
      <span class="stepn" id="stepLabel">Step 0</span>
    </div>
    <div class="slider">
      <label><span>Step</span><span id="stepVal">0</span></label>
      <input type="range" id="scrub" min="0" max="39" step="0.01" value="0"/>
    </div>
    <div class="slider">
      <label><span>Speed</span><span id="speedVal">1.0×</span></label>
      <input type="range" id="speed" min="0.2" max="4" step="0.1" value="1"/>
    </div>
    <div class="slider">
      <label><span>Spread</span><span id="spreadVal">1.0×</span></label>
      <input type="range" id="spread" min="0.5" max="2" step="0.05" value="1"/>
    </div>
    <p class="blurb" id="blurb">Geometry neighborhoods over training. Scrub or play to morph.</p>
    <div class="metrics" id="metrics"></div>
    <div class="diff" id="diff"></div>
  </aside>
</div>

<script id="payload" type="application/json">@@DATA@@</script>
<script>
(() => {
  const DATA = JSON.parse(document.getElementById("payload").textContent);
  const frames = DATA.frames;
  const layers = (DATA.layers || []).map(String);
  const N = frames.length;
  if (!N) {
    document.body.textContent = "No frames in signals.csv";
    return;
  }

  const canvas = document.getElementById("viz");
  const ctx = canvas.getContext("2d");
  const scrub = document.getElementById("scrub");
  const speedEl = document.getElementById("speed");
  const spreadEl = document.getElementById("spread");
  scrub.max = String(N - 1);

  const SPECTRUM = [
    [0.15, 0.10, 0.55],
    [0.15, 0.45, 0.95],
    [0.05, 0.80, 0.85],
    [0.35, 0.90, 0.25],
    [0.95, 0.85, 0.10],
  ];
  function spectrumRGB(u) {
    const x = Math.min(1, Math.max(0, u)) * (SPECTRUM.length - 1);
    const i = Math.floor(x), f = x - i;
    const a = SPECTRUM[i], b = SPECTRUM[Math.min(i + 1, SPECTRUM.length - 1)];
    return [
      Math.round((a[0] + (b[0] - a[0]) * f) * 255),
      Math.round((a[1] + (b[1] - a[1]) * f) * 255),
      Math.round((a[2] + (b[2] - a[2]) * f) * 255),
    ];
  }
  function hash(i) {
    const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
    return x - Math.floor(x);
  }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function ease(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  // Exaggerate geometry so step→step morph is visibly different.
  const GRID = 22; // points per side of sheet per layer
  function cloudForFrame(frame, amplify) {
    const pts = [];
    const nL = Math.max(1, layers.length);
    layers.forEach((L, li) => {
      const m = frame.layers[L] || { interference: 0.4, spectral: 0.1, coact: 0.1 };
      // Map metrics → sheet warp (big dynamic range)
      const pack = Math.pow(m.interference, 0.7);           // collapse
      const spread = 0.35 + m.spectral * amplify * 3.2;     // expand
      const ripple = 0.15 + m.coact * amplify * 1.8;        // wave amp
      const zBase = (li - (nL - 1) / 2) * 0.95;
      for (let yi = 0; yi < GRID; yi++) {
        for (let xi = 0; xi < GRID; xi++) {
          const u = xi / (GRID - 1);
          const v = yi / (GRID - 1);
          const g = li * 100003 + yi * 97 + xi;
          // Start as a grid sheet, then pack toward center + ripple
          let x = (u - 0.5) * 2.2 * spread;
          let y = (v - 0.5) * 2.2 * spread;
          const r = Math.hypot(x, y);
          const shrink = lerp(1, 0.22, pack);
          x *= shrink;
          y *= shrink;
          // fold / horseshoe based on packing
          const fold = pack * 1.1;
          const x2 = x * Math.cos(fold * y) - y * Math.sin(fold * x * 0.4);
          const y2 = x * Math.sin(fold * y * 0.5) + y * Math.cos(fold * 0.3);
          const z = zBase
            + Math.sin(u * Math.PI * 2 + pack * 3) * ripple * 0.55
            + Math.cos(v * Math.PI * 2) * ripple * 0.35
            + (hash(g) - 0.5) * 0.06;
          pts.push({
            x: x2 + (hash(g + 3) - 0.5) * 0.04,
            y: y2 + (hash(g + 7) - 0.5) * 0.04,
            z,
            u: (li + u) / nL,
            v,
          });
        }
      }
    });
    return pts;
  }

  // Precompute clouds once (amplify fixed at bake; spread slider scales in project)
  const clouds = frames.map(f => cloudForFrame(f, 1.35));

  function blended(tf) {
    const i0 = Math.floor(tf);
    const i1 = Math.min(N - 1, i0 + 1);
    const e = ease(tf - i0);
    const a = clouds[i0], b = clouds[i1];
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i++) {
      const p = a[i], q = b[i];
      out[i] = {
        x: lerp(p.x, q.x, e),
        y: lerp(p.y, q.y, e),
        z: lerp(p.z, q.z, e),
        u: p.u,
        v: p.v,
      };
    }
    return out;
  }

  let t = 0;
  let playing = true; // start playing so morph is obvious
  let rotY = 0.55, rotX = -0.35;
  let drag = null;

  function project(p, s, scale) {
    const cy = Math.cos(rotY), sy = Math.sin(rotY);
    const cx = Math.cos(rotX), sx = Math.sin(rotX);
    let x = p.x * s, y = p.y * s, z = p.z;
    const x1 = x * cy + z * sy;
    const z1 = -x * sy + z * cy;
    const y1 = y * cx - z1 * sx;
    const z2 = y * sx + z1 * cx;
    const persp = 3.0 / (3.0 + z2);
    return {
      x: canvas.width * 0.5 + x1 * scale * persp,
      y: canvas.height * 0.5 + y1 * scale * persp,
      z: z2,
      u: p.u,
    };
  }

  function paint(target, pts, w, h, scale, s) {
    const c = target.getContext("2d");
    c.clearRect(0, 0, w, h);
    c.fillStyle = "#fff";
    c.fillRect(0, 0, w, h);
    const drawn = pts.map(p => project(p, s, scale));
    drawn.sort((a, b) => a.z - b.z);
    for (const p of drawn) {
      const [r, g, b] = spectrumRGB(p.u);
      c.beginPath();
      c.fillStyle = `rgb(${r},${g},${b})`;
      c.arc(p.x * (w / canvas.width), p.y * (h / canvas.height), Math.max(1.1, 2.1 * (w / canvas.width)), 0, Math.PI * 2);
      c.fill();
    }
  }

  function fmt(x) {
    return Number.isFinite(x) ? x.toFixed(3) : "—";
  }

  function updateUI(tf) {
    const i = Math.floor(tf);
    const fr = frames[i];
    document.getElementById("stepLabel").textContent = "Step " + i.toLocaleString();
    document.getElementById("stepVal").textContent = String(i);
    scrub.value = String(tf);
    let html = `<div><b>loss</b> ${fmt(fr.loss)}</div>`;
    layers.forEach(L => {
      const m = fr.layers[L];
      html += `<div><b>L${L}</b> i ${fmt(m.interference)} · s ${fmt(m.spectral)} · c ${fmt(m.coact)}</div>`;
    });
    document.getElementById("metrics").innerHTML = html;

    const fd = DATA.feature_diff;
    if (fd && fd.pairs && fd.pairs.length) {
      document.getElementById("diff").innerHTML = fd.pairs.map(p => {
        const short = p.id.replace(/__/g, " ↔ ").replace(/_/g, " ");
        const cls = p.related ? "rel" : "unrel";
        return `<div><span class="${cls}">${p.related ? "related" : "unrelated"}</span> ${short} · ${fmt(p.mean)}</div>`;
      }).join("");
    }

    document.getElementById("blurb").textContent =
      `${DATA.run_name} · ${DATA.meta.model || "model"} · ${N} steps · layers [${layers.join(", ")}]. ` +
      `Play to morph neighborhoods as geometry drifts.`;
  }

  function setPlayIcon(on) {
    document.getElementById("iconPlay").style.display = on ? "none" : "block";
    document.getElementById("iconPause").style.display = on ? "block" : "none";
  }

  function draw() {
    const s = parseFloat(spreadEl.value);
    const pts = blended(t);
    paint(canvas, pts, canvas.width, canvas.height, 165, s);
    updateUI(t);
  }

  // Layer thumbs = jump to early / mid / late (and per-layer focus later)
  const thumbs = document.getElementById("thumbs");
  const thumbSteps = [0, Math.floor((N - 1) / 2), N - 1];
  // Also show each layer alone at mid step
  const thumbSpecs = [
    ...thumbSteps.map(st => ({ step: st, layer: null, label: "t" + st })),
    ...layers.map(L => ({ step: thumbSteps[1], layer: L, label: "L" + L })),
  ].slice(0, 9);

  function cloudLayerOnly(frame, layerId) {
    const all = cloudForFrame(frame, 1.35);
    if (layerId == null) return all;
    const idx = layers.indexOf(String(layerId));
    const per = GRID * GRID;
    return all.slice(idx * per, (idx + 1) * per);
  }

  thumbSpecs.forEach((spec, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "thumb" + (i === 0 ? " active" : "");
    const tc = document.createElement("canvas");
    tc.width = 64; tc.height = 64;
    btn.appendChild(tc);
    const lab = document.createElement("span");
    lab.textContent = spec.label;
    btn.appendChild(lab);
    btn.onclick = () => {
      playing = false; setPlayIcon(false);
      t = spec.step;
      document.querySelectorAll(".thumb").forEach(el => el.classList.remove("active"));
      btn.classList.add("active");
      draw();
    };
    thumbs.appendChild(btn);
    // mini paint
    const pts = cloudLayerOnly(frames[spec.step], spec.layer);
    // temporarily use mini projection onto thumb
    const prevW = canvas.width, prevH = canvas.height;
    // paint helper uses canvas.width for mapping — draw manually small
    const c = tc.getContext("2d");
    c.fillStyle = "#fff"; c.fillRect(0, 0, 64, 64);
    const s = 1;
    const scale = 18;
    const cy = Math.cos(0.5), sy = Math.sin(0.5);
    const cx = Math.cos(-0.3), sx = Math.sin(-0.3);
    const drawn = pts.map(p => {
      let x = p.x * s, y = p.y * s, z = p.z;
      const x1 = x * cy + z * sy;
      const z1 = -x * sy + z * cy;
      const y1 = y * cx - z1 * sx;
      const z2 = y * sx + z1 * cx;
      const persp = 3 / (3 + z2);
      return { x: 32 + x1 * scale * persp, y: 32 + y1 * scale * persp, z: z2, u: p.u };
    });
    drawn.sort((a, b) => a.z - b.z);
    for (const p of drawn) {
      const [r, g, b] = spectrumRGB(p.u);
      c.fillStyle = `rgb(${r},${g},${b})`;
      c.fillRect(p.x - 0.8, p.y - 0.8, 1.6, 1.6);
    }
  });

  // Drag to rotate
  canvas.addEventListener("pointerdown", e => {
    drag = { x: e.clientX, y: e.clientY, rotY, rotX };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", e => {
    if (!drag) return;
    rotY = drag.rotY + (e.clientX - drag.x) * 0.008;
    rotX = Math.max(-1.2, Math.min(0.6, drag.rotX + (e.clientY - drag.y) * 0.008));
    draw();
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("pointercancel", () => { drag = null; });

  scrub.addEventListener("input", () => {
    playing = false; setPlayIcon(false);
    t = parseFloat(scrub.value);
    draw();
  });
  speedEl.addEventListener("input", () => {
    document.getElementById("speedVal").textContent = parseFloat(speedEl.value).toFixed(1) + "×";
  });
  spreadEl.addEventListener("input", () => {
    document.getElementById("spreadVal").textContent = parseFloat(spreadEl.value).toFixed(1) + "×";
    draw();
  });

  document.getElementById("btnPlay").onclick = () => {
    playing = !playing;
    setPlayIcon(playing);
  };
  document.getElementById("btnReset").onclick = () => {
    playing = false; setPlayIcon(false);
    t = 0; rotY = 0.55; rotX = -0.35;
    draw();
  };

  setPlayIcon(true);
  draw();

  // setInterval — reliable scrubber playback (rAF gets throttled in some hosts)
  setInterval(() => {
    if (!playing) return;
    const spd = parseFloat(speedEl.value);
    t += 0.05 * spd * 3.0; // ~3 steps/sec at 1×
    if (t >= N - 1) t = 0;
    rotY += 0.012;
    draw();
  }, 50);
})();
</script>
</body>
</html>
"""
