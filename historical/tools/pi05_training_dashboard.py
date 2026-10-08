#!/usr/bin/env python3
"""Serve a tiny dashboard for G1-D pi0.5 fine-tuning progress."""

from __future__ import annotations

import argparse
import html
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


LOG_RE = re.compile(
    r"step=(?P<step>\d+)\s+loss=(?P<loss>[0-9.eE+-]+)\s+lr=(?P<lr>[0-9.eE+-]+)"
    r"(?:\s+grad_norm=(?P<grad_norm>[0-9.eE+-]+))?"
)
TQDM_RE = re.compile(
    r"loss=(?P<loss>[0-9.eE+-]+),\s+lr=(?P<lr>[0-9.eE+-]+),\s+step=(?P<step>[0-9.eE+-]+)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    points = []
    seen = set()
    for line in ANSI_RE.sub("", path.read_text(errors="ignore")).splitlines():
        match = LOG_RE.search(line)
        if match:
            item = {
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "lr": float(match.group("lr")),
            }
            if match.group("grad_norm") is not None:
                item["grad_norm"] = float(match.group("grad_norm"))
        else:
            match = TQDM_RE.search(line)
            if not match:
                continue
            item = {
                "step": int(float(match.group("step"))),
                "loss": float(match.group("loss")),
                "lr": float(match.group("lr")),
            }
        if item["step"] in seen:
            continue
        seen.add(item["step"])
        points.append(item)
    return points


def dataset_summary(path: Path) -> dict:
    info = json.loads((path / "meta/info.json").read_text())
    tasks = [json.loads(line) for line in (path / "meta/tasks.jsonl").read_text().splitlines() if line]
    episodes = [json.loads(line) for line in (path / "meta/episodes.jsonl").read_text().splitlines() if line]
    task_counts: dict[str, int] = {}
    for episode in episodes:
        idx = episode.get("tasks", [episode.get("task_index", 0)])[0]
        name = tasks[idx]["task"] if idx < len(tasks) else str(idx)
        task_counts[name] = task_counts.get(name, 0) + 1
    return {
        "name": path.name,
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "videos": info["total_videos"],
        "tasks": task_counts,
    }


def checkpoints(path: Path) -> list[int]:
    if not path.exists():
        return []
    return sorted(int(p.name) for p in path.iterdir() if p.is_dir() and p.name.isdigit())


def render(args: argparse.Namespace) -> str:
    points = parse_log(args.log)
    ckpts = checkpoints(args.checkpoints)
    data = dataset_summary(args.dataset)
    latest = points[-1] if points else None
    point_json = json.dumps(points[-1000:])
    data_json = json.dumps(data, ensure_ascii=False)
    ckpt_json = json.dumps(ckpts)
    latest_text = "waiting for first log line"
    if latest:
        latest_text = f"step {latest['step']} | loss {latest['loss']:.4f} | lr {latest['lr']:.2e}"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="10">
  <title>G1-D pi0.5 FT Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Inter, system-ui, sans-serif; background: #f7f7f4; color: #1f2933; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    .muted {{ color: #667085; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ background: white; border: 1px solid #d9dee3; border-radius: 8px; padding: 14px; }}
    .metric {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
    canvas {{ width: 100%; height: 360px; background: white; border: 1px solid #d9dee3; border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    td, th {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
    code {{ background: #eef2f6; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>G1-D pi0.5 Food Sorting FT</h1>
  <div class="muted">Auto refreshes every 10s. {html.escape(str(args.log))}</div>
  <div class="grid">
    <div class="card"><div class="muted">Latest</div><div class="metric">{html.escape(latest_text)}</div></div>
    <div class="card"><div class="muted">Episodes</div><div class="metric">{data["episodes"]}</div></div>
    <div class="card"><div class="muted">Frames</div><div class="metric">{data["frames"]:,}</div></div>
    <div class="card"><div class="muted">Checkpoints</div><div class="metric">{len(ckpts)}</div></div>
  </div>
  <canvas id="chart" width="1120" height="360"></canvas>
  <div class="grid" style="grid-template-columns: 1fr 1fr;">
    <div class="card">
      <h2>Dataset</h2>
      <table id="dataset"></table>
    </div>
    <div class="card">
      <h2>Checkpoints</h2>
      <div id="ckpts" class="muted"></div>
    </div>
  </div>
</main>
<script>
const points = {point_json};
const dataset = {data_json};
const ckpts = {ckpt_json};
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
ctx.clearRect(0, 0, canvas.width, canvas.height);
ctx.fillStyle = '#ffffff';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = '#cbd5df';
ctx.lineWidth = 1;
for (let i = 0; i <= 6; i++) {{
  const y = 30 + i * 48;
  ctx.beginPath(); ctx.moveTo(50, y); ctx.lineTo(1090, y); ctx.stroke();
}}
ctx.fillStyle = '#475467';
ctx.font = '14px system-ui';
ctx.fillText('loss', 18, 30);
if (points.length > 1) {{
  const xs = points.map(p => p.step);
  const ys = points.map(p => p.loss);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const pad = Math.max((ymax - ymin) * 0.08, 1e-4);
  function x(v) {{ return 50 + (v - xmin) / Math.max(1, xmax - xmin) * 1040; }}
  function y(v) {{ return 330 - (v - ymin + pad) / Math.max(1e-6, ymax - ymin + 2 * pad) * 290; }}
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, i) => {{ const px = x(p.step), py = y(p.loss); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }});
  ctx.stroke();
  ctx.fillStyle = '#111827';
  ctx.fillText(`step ${{xmin}}`, 50, 350);
  ctx.fillText(`step ${{xmax}}`, 1010, 350);
  ctx.fillText(`min ${{ymin.toFixed(4)}}`, 55, 320);
  ctx.fillText(`max ${{ymax.toFixed(4)}}`, 55, 48);
}} else {{
  ctx.fillStyle = '#667085';
  ctx.fillText('Waiting for training log...', 470, 180);
}}
const rows = [
  ['Dataset', dataset.name],
  ['Episodes', dataset.episodes],
  ['Frames', dataset.frames.toLocaleString()],
  ['Videos', dataset.videos],
  ['Tasks', Object.entries(dataset.tasks).map(([k,v]) => `${{v}} x ${{k}}`).join('<br>')]
];
document.getElementById('dataset').innerHTML = rows.map(r => `<tr><th>${{r[0]}}</th><td>${{r[1]}}</td></tr>`).join('');
document.getElementById('ckpts').innerHTML = ckpts.length ? ckpts.join(', ') : 'No checkpoint yet';
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = render(args).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *values):
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
