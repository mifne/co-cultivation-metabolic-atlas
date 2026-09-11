#!/usr/bin/env python3
"""Convert a resource-usage report into a compact inline SVG visualization."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


COLORS = {"highs": "var(--viz-series-1)", "surrogate_cuda": "var(--viz-series-2)"}
LABELS = {"highs": "CPU / HiGHS", "surrogate_cuda": "CUDA surrogate"}


def path_for(values: list[float], width: float, height: float, y_max: float) -> str:
    if len(values) < 2:
        return ""
    points = []
    for index, value in enumerate(values):
        x = 48 + (width - 64) * index / (len(values) - 1)
        y = 24 + (height - 52) * (1 - min(max(value, 0), y_max) / y_max)
        points.append(f"{x:.1f},{y:.1f}")
    return "M" + " L".join(points)


def chart(title: str, y_label: str, series: list[tuple[str, list[float]]], y_max: float, x_label: str = "seconds") -> str:
    width, height = 430, 230
    parts = [
        f'<svg class="resource-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f"<title>{html.escape(title)}</title>",
        f'<rect data-chart-frame x="48" y="24" width="{width - 64}" height="{height - 52}" fill="none" stroke="var(--border)"/>',
        f'<text x="48" y="16" fill="var(--foreground)" class="chart-title">{html.escape(title)}</text>',
        f'<text class="axis-title" data-axis="y" x="12" y="{height / 2}" transform="rotate(-90 12 {height / 2})">{html.escape(y_label)}</text>',
        f'<text class="axis-title" data-axis="x" x="{width / 2}" y="{height - 5}" text-anchor="middle">{html.escape(x_label)}</text>',
        f'<text x="42" y="29" text-anchor="end" class="tick">{y_max:g}</text>',
        '<text x="42" y="202" text-anchor="end" class="tick">0</text>',
    ]
    for backend, values in series:
        color = COLORS[backend]
        d = path_for(values, width, height, y_max)
        if d:
            parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    series: dict[str, dict[str, list[float]]] = {"highs": {}, "surrogate_cuda": {}}
    for backend in series:
        runs = report["runs"][backend]
        rows = runs[0]["resource_samples"] if runs else []
        series[backend]["gpu"] = [row["gpus"][0].get("gpu_utilization_percent", 0) if row["gpus"] else 0 for row in rows]
        series[backend]["vram"] = [row["gpus"][0].get("memory_used_mib", 0) if row["gpus"] else 0 for row in rows]
        series[backend]["cpu"] = [row["process_cpu_percent_sum"] for row in rows]
    gpu_max = max(100, max((max(v["gpu"]) if v["gpu"] else 0) for v in series.values()))
    vram_max = max(256, max((max(v["vram"]) if v["vram"] else 0) for v in series.values()) * 1.15)
    cpu_max = max(100, max((max(v["cpu"]) if v["cpu"] else 0) for v in series.values()) * 1.15)
    throughput = [(b, report["summary"][b]["steps_per_second_mean"]) for b in series]
    max_tp = max(v for _, v in throughput) * 1.2 if throughput else 1
    bars = []
    for index, (backend, value) in enumerate(throughput):
        x = 80 + index * 150
        bar_h = 150 * value / max_tp
        bars.append(
            f'<rect x="{x}" y="{190 - bar_h:.1f}" width="80" height="{bar_h:.1f}" fill="{COLORS[backend]}"/> '
            f'<text x="{x + 40}" y="{184 - bar_h:.1f}" text-anchor="middle" class="value">{value:.2f}</text> '
            f'<text x="{x + 40}" y="211" text-anchor="middle" class="tick">{html.escape(LABELS[backend])}</text>'
        )
    bar_svg = (
        '<svg class="resource-chart" viewBox="0 0 430 230" role="img" aria-label="Rollout throughput">'
        '<title>Rollout throughput</title><rect data-chart-frame x="48" y="24" width="366" height="166" fill="none" stroke="var(--border)"/>'
        '<text x="48" y="16" fill="var(--foreground)" class="chart-title">Rollout throughput</text>'
        '<text class="axis-title" data-axis="y" x="12" y="110" transform="rotate(-90 12 110)">steps / second</text>'
        + "".join(bars) + "</svg>"
    )
    legend = " ".join(
        f'<span class="legend-item"><span class="swatch" style="background:{COLORS[b]}"></span>{html.escape(LABELS[b])}</span>'
        for b in series
    )
    fragment = f'''<div id="resource-usage-short">
  <style>
    #resource-usage-short {{ color: var(--foreground); font-size: var(--font-size-base); }}
    #resource-usage-short .resource-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
    #resource-usage-short .resource-chart {{ width:100%; height:auto; overflow:visible; }}
    #resource-usage-short text {{ fill:var(--foreground); font-size:12px; }}
    #resource-usage-short .chart-title {{ font-size:14px; font-weight:500; }}
    #resource-usage-short .tick {{ fill:var(--muted-foreground); font-size:11px; }}
    #resource-usage-short .value {{ font-size:12px; }}
    #resource-usage-short .legend {{ display:flex; gap:16px; flex-wrap:wrap; margin:6px 0 12px 48px; }}
    #resource-usage-short .legend-item {{ display:inline-flex; align-items:center; gap:5px; }}
    #resource-usage-short .swatch {{ width:10px; height:10px; display:inline-block; }}
    @media (max-width:600px) {{ #resource-usage-short .resource-grid {{ grid-template-columns:1fr; }} }}
  </style>
  <h2>短時間実測：CPU版とCUDA版の資源使用状況</h2>
  <div class="legend">{legend}</div>
  <div class="resource-grid">
    {chart('GPU utilization', 'percent (%)', [(b, series[b]['gpu']) for b in series], gpu_max)}
    {chart('VRAM used', 'MiB', [(b, series[b]['vram']) for b in series], vram_max)}
    {chart('Process CPU utilization', 'percent (%)', [(b, series[b]['cpu']) for b in series], cpu_max)}
    {bar_svg}
  </div>
</div>
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment, encoding="utf-8")


if __name__ == "__main__":
    main()
