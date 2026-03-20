from __future__ import annotations

import json
from html import escape

import pandas as pd

from mysphinx_forge.clustering import ClusteringStats


def build_cluster_analysis_report(
    cluster_summary: pd.DataFrame,
    stats: ClusteringStats,
) -> pd.DataFrame:
    if cluster_summary.empty:
        return pd.DataFrame(
            columns=[
                "cluster_rank",
                "cluster_id",
                "cluster_size",
                "cluster_ratio",
                "cluster_label",
                "top_keywords",
                "representative_text",
                "example_texts",
            ]
        )

    analysis = cluster_summary.copy().sort_values(
        by=["cluster_size", "cluster_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    analysis.insert(0, "cluster_rank", range(1, len(analysis) + 1))
    denominator = stats.total_clustered if stats.total_clustered > 0 else max(stats.total_before, 1)
    analysis.insert(3, "cluster_ratio", (analysis["cluster_size"] / denominator).round(4))
    return analysis[
        [
            "cluster_rank",
            "cluster_id",
            "cluster_size",
            "cluster_ratio",
            "cluster_label",
            "top_keywords",
            "representative_text",
            "example_texts",
        ]
    ]


def render_cluster_report_html(
    *,
    analysis_report: pd.DataFrame,
    projection: pd.DataFrame,
    stats: ClusteringStats,
) -> str:
    analysis_records = analysis_report.to_dict(orient="records")
    point_records = projection.to_dict(orient="records")
    payload = {
        "stats": {
            "total_before": stats.total_before,
            "total_clustered": stats.total_clustered,
            "noise_rows": stats.noise_rows,
            "cluster_count": stats.cluster_count,
            "largest_cluster_size": stats.largest_cluster_size,
            "smallest_cluster_size": stats.smallest_cluster_size,
            "average_cluster_size": round(stats.average_cluster_size, 2),
            "cluster_mode": stats.cluster_mode,
            "target_column": stats.target_column,
        },
        "clusters": analysis_records,
        "points": point_records,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    table_rows = "\n".join(
        (
            "<tr>"
            f"<td>{record['cluster_rank']}</td>"
            f"<td>{record['cluster_id']}</td>"
            f"<td>{record['cluster_size']}</td>"
            f"<td>{record['cluster_ratio']:.2%}</td>"
            f"<td>{escape(str(record['cluster_label']))}</td>"
            f"<td>{escape(str(record['top_keywords']))}</td>"
            f"<td>{escape(str(record['representative_text']))}</td>"
            "</tr>"
        )
        for record in analysis_records
    )
    if not table_rows:
        table_rows = (
            '<tr><td colspan="7" class="empty">当前没有可展示的聚类结果，请调整聚类参数后重试。</td></tr>'
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cluster Report</title>
  <style>
    :root {{
      --bg: #f4efe8;
      --panel: rgba(255, 252, 247, 0.92);
      --ink: #1f2937;
      --muted: #6b7280;
      --accent: #1d4ed8;
      --accent-2: #ea580c;
      --line: rgba(31, 41, 55, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Noto Serif SC", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(29, 78, 216, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(234, 88, 12, 0.14), transparent 24%),
        linear-gradient(180deg, #f7f2eb 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 32px auto 48px;
    }}
    .hero {{
      padding: 28px 30px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--panel);
      backdrop-filter: blur(18px);
      box-shadow: 0 20px 60px rgba(31, 41, 55, 0.08);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(32px, 5vw, 54px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .sub {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 16px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 14px;
      margin-top: 24px;
    }}
    .card {{
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(31, 41, 55, 0.08);
    }}
    .card label {{
      display: block;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .card strong {{
      display: block;
      margin-top: 8px;
      font-size: 28px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .panel {{
      padding: 20px;
      border-radius: 24px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: 0 18px 40px rgba(31, 41, 55, 0.06);
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 20px;
    }}
    #chart {{
      width: 100%;
      height: 480px;
      display: block;
      border-radius: 18px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.86), rgba(247,242,235,0.96));
      border: 1px solid rgba(31, 41, 55, 0.08);
      cursor: grab;
    }}
    #chart.dragging {{ cursor: grabbing; }}
    .legend {{
      margin-top: 12px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }}
    .controls label {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }}
    select, button {{
      border: 1px solid rgba(31, 41, 55, 0.12);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255,255,255,0.8);
      color: var(--ink);
      font: inherit;
    }}
    button {{ cursor: pointer; }}
    .detail {{
      margin-top: 14px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(31,41,55,0.08);
      min-height: 122px;
    }}
    .detail h3 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .detail p {{
      margin: 6px 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .detail strong {{
      color: var(--ink);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      vertical-align: top;
      padding: 10px 8px;
      border-bottom: 1px solid rgba(31, 41, 55, 0.08);
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .empty {{
      color: var(--muted);
      text-align: center;
      padding: 32px 8px;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      #chart {{ height: 360px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Cluster Report</h1>
      <div class="sub">目标列：{escape(stats.target_column)} · 模式：{escape(stats.cluster_mode)}</div>
      <div class="stats">
        <div class="card"><label>Rows</label><strong>{stats.total_before}</strong></div>
        <div class="card"><label>Clusters</label><strong>{stats.cluster_count}</strong></div>
        <div class="card"><label>Clustered</label><strong>{stats.total_clustered}</strong></div>
        <div class="card"><label>Noise</label><strong>{stats.noise_rows}</strong></div>
        <div class="card"><label>Largest</label><strong>{stats.largest_cluster_size}</strong></div>
        <div class="card"><label>Average</label><strong>{stats.average_cluster_size:.2f}</strong></div>
      </div>
    </section>
    <section class="grid">
      <article class="panel">
        <h2>Projection</h2>
        <div class="controls">
          <label>Cluster
            <select id="clusterFilter">
              <option value="all">All</option>
            </select>
          </label>
          <label>
            <input id="noiseOnly" type="checkbox">
            Noise only
          </label>
          <button id="resetView" type="button">Reset View</button>
          <button id="resetSelection" type="button">Reset</button>
        </div>
        <canvas id="chart" width="720" height="480"></canvas>
        <div class="legend">
          <span>点颜色按簇区分，噪声点使用橙色高亮</span>
          <span>拖拽可旋转，滚轮可缩放</span>
          <span>当前使用 3D PCA 投影，空白或不可投影样本会被忽略</span>
        </div>
        <div class="detail" id="detailPanel">
          <h3>Point Detail</h3>
          <p>点击任意散点查看原文、簇编号和噪声状态。</p>
        </div>
      </article>
      <article class="panel">
        <h2>Cluster Summary</h2>
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>ID</th>
              <th>Size</th>
              <th>Ratio</th>
              <th>Label</th>
              <th>Keywords</th>
              <th>Representative</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
      </article>
    </section>
  </main>
  <script>
    const payload = {payload_json};
    const canvas = document.getElementById("chart");
    const context = canvas.getContext("2d");
    const clusterFilter = document.getElementById("clusterFilter");
    const noiseOnly = document.getElementById("noiseOnly");
    const resetView = document.getElementById("resetView");
    const resetSelection = document.getElementById("resetSelection");
    const detailPanel = document.getElementById("detailPanel");
    const width = canvas.width;
    const height = canvas.height;
    const palette = ["#1d4ed8", "#0f766e", "#7c3aed", "#be123c", "#0369a1", "#4d7c0f", "#c2410c", "#334155"];
    const projectedPoints = payload.points.filter(
      (point) => Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z)
    );
    const clusterOptions = [...new Set(projectedPoints.filter((point) => !point.is_noise).map((point) => point.cluster_id))].sort((a, b) => a - b);
    const defaultView = {{ rotationX: -0.58, rotationY: 0.72, zoom: 1 }};
    const view = {{ ...defaultView }};
    const pointRadius = 5.2;
    const cameraDistance = 3.2;
    const projectionStrength = 0.9;
    let selectedPoint = null;
    let renderedPoints = [];
    let isDragging = false;
    let lastPointer = null;
    let dragDistance = 0;

    clusterOptions.forEach((clusterId) => {{
      const option = document.createElement("option");
      option.value = String(clusterId);
      option.textContent = `Cluster ${{clusterId}}`;
      clusterFilter.appendChild(option);
    }});

    function updateDetail(point) {{
      if (!point) {{
        detailPanel.innerHTML = `
          <h3>Point Detail</h3>
          <p>点击任意散点查看原文、簇编号和噪声状态。</p>
        `;
        return;
      }}
      const text = String(point[payload.stats.target_column] ?? "");
      detailPanel.innerHTML = `
        <h3>Point Detail</h3>
        <p><strong>Text:</strong> ${{text || "(empty)"}}</p>
        <p><strong>Cluster:</strong> ${{point.cluster_id}}</p>
        <p><strong>Noise:</strong> ${{point.is_noise ? "Yes" : "No"}}</p>
        <p><strong>Coordinates:</strong> (${{Number(point.x).toFixed(3)}}, ${{Number(point.y).toFixed(3)}}, ${{Number(point.z).toFixed(3)}})</p>
      `;
    }}

    function filteredPoints() {{
      const clusterValue = clusterFilter.value;
      return projectedPoints.filter((point) => {{
        if (noiseOnly.checked && !point.is_noise) return false;
        if (clusterValue !== "all" && String(point.cluster_id) !== clusterValue) return false;
        return true;
      }});
    }}

    function normalizePoints(points) {{
      const center = {{
        x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
        y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
        z: points.reduce((sum, point) => sum + point.z, 0) / points.length,
      }};
      const maxExtent = Math.max(
        ...points.map((point) => Math.max(
          Math.abs(point.x - center.x),
          Math.abs(point.y - center.y),
          Math.abs(point.z - center.z),
        )),
        1e-6,
      );
      return points.map((point) => ({{
        point,
        x: (point.x - center.x) / maxExtent,
        y: (point.y - center.y) / maxExtent,
        z: (point.z - center.z) / maxExtent,
      }}));
    }}

    function rotatePoint(point) {{
      const cosY = Math.cos(view.rotationY);
      const sinY = Math.sin(view.rotationY);
      const cosX = Math.cos(view.rotationX);
      const sinX = Math.sin(view.rotationX);
      const x1 = point.x * cosY + point.z * sinY;
      const z1 = -point.x * sinY + point.z * cosY;
      const y2 = point.y * cosX - z1 * sinX;
      const z2 = point.y * sinX + z1 * cosX;
      return {{ x: x1, y: y2, z: z2 }};
    }}

    function projectToCanvas(point) {{
      const perspective = (projectionStrength * view.zoom) / Math.max(cameraDistance - point.z, 0.4);
      return {{
        x: width / 2 + point.x * perspective * width * 0.42,
        y: height / 2 - point.y * perspective * height * 0.42,
        scale: perspective,
        depth: point.z,
      }};
    }}

    function drawAxes() {{
      const axisLength = 1.25;
      const axes = [
        {{ label: "X", color: "rgba(29, 78, 216, 0.55)", from: {{ x: -axisLength, y: 0, z: 0 }}, to: {{ x: axisLength, y: 0, z: 0 }} }},
        {{ label: "Y", color: "rgba(15, 118, 110, 0.55)", from: {{ x: 0, y: -axisLength, z: 0 }}, to: {{ x: 0, y: axisLength, z: 0 }} }},
        {{ label: "Z", color: "rgba(234, 88, 12, 0.55)", from: {{ x: 0, y: 0, z: -axisLength }}, to: {{ x: 0, y: 0, z: axisLength }} }},
      ];
      axes.forEach((axis) => {{
        const from = projectToCanvas(rotatePoint(axis.from));
        const to = projectToCanvas(rotatePoint(axis.to));
        context.strokeStyle = axis.color;
        context.lineWidth = 1.2;
        context.beginPath();
        context.moveTo(from.x, from.y);
        context.lineTo(to.x, to.y);
        context.stroke();
        context.fillStyle = axis.color;
        context.font = "12px sans-serif";
        context.fillText(axis.label, to.x + 6, to.y - 6);
      }});
    }}

    function renderChart() {{
      const points = filteredPoints();
      renderedPoints = [];
      context.clearRect(0, 0, width, height);
      if (!points.length) {{
        context.fillStyle = "#6b7280";
        context.font = "18px serif";
        context.textAlign = "center";
        context.fillText("No points match the current filter", width / 2, height / 2);
        updateDetail(null);
        return;
      }}

      drawAxes();
      renderedPoints = normalizePoints(points).map((entry) => {{
        const rotated = rotatePoint(entry);
        const projected = projectToCanvas(rotated);
        return {{
          point: entry.point,
          screenX: projected.x,
          screenY: projected.y,
          depth: projected.depth,
          radius: Math.max(2.6, pointRadius * (0.72 + projected.scale * 1.15)),
        }};
      }}).sort((a, b) => a.depth - b.depth);

      renderedPoints.forEach((entry) => {{
        const point = entry.point;
        const color = point.is_noise ? "#ea580c" : palette[Math.abs(point.cluster_id) % palette.length];
        const isSelected = selectedPoint && selectedPoint.row_index === point.row_index;
        context.beginPath();
        context.arc(entry.screenX, entry.screenY, isSelected ? entry.radius + 2.2 : entry.radius, 0, Math.PI * 2);
        context.fillStyle = color;
        context.globalAlpha = point.is_noise ? 0.9 : 0.82;
        context.fill();
        context.globalAlpha = 1;
        context.lineWidth = isSelected ? 2.1 : 1.3;
        context.strokeStyle = isSelected ? "#111827" : "rgba(255,255,255,0.9)";
        context.stroke();
      }});

      if (selectedPoint) {{
        const stillVisible = points.some((point) => point.row_index === selectedPoint.row_index);
        if (!stillVisible) {{
          selectedPoint = null;
          updateDetail(null);
        }}
      }}
    }}

    function pickPoint(event) {{
      const bounds = canvas.getBoundingClientRect();
      const scaleX = canvas.width / bounds.width;
      const scaleY = canvas.height / bounds.height;
      const pointerX = (event.clientX - bounds.left) * scaleX;
      const pointerY = (event.clientY - bounds.top) * scaleY;
      let candidate = null;
      renderedPoints.forEach((entry) => {{
        const dx = entry.screenX - pointerX;
        const dy = entry.screenY - pointerY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance <= entry.radius + 4) {{
          if (!candidate || distance < candidate.distance || entry.depth > candidate.depth) {{
            candidate = {{ entry, distance }};
          }}
        }}
      }});
      return candidate ? candidate.entry.point : null;
    }}

    canvas.addEventListener("pointerdown", (event) => {{
      isDragging = true;
      lastPointer = {{ x: event.clientX, y: event.clientY }};
      dragDistance = 0;
      canvas.classList.add("dragging");
      canvas.setPointerCapture(event.pointerId);
    }});

    canvas.addEventListener("pointermove", (event) => {{
      if (!isDragging || !lastPointer) return;
      const deltaX = event.clientX - lastPointer.x;
      const deltaY = event.clientY - lastPointer.y;
      dragDistance += Math.abs(deltaX) + Math.abs(deltaY);
      lastPointer = {{ x: event.clientX, y: event.clientY }};
      view.rotationY += deltaX * 0.01;
      view.rotationX += deltaY * 0.01;
      renderChart();
    }});

    canvas.addEventListener("pointerup", (event) => {{
      isDragging = false;
      lastPointer = null;
      canvas.classList.remove("dragging");
      if (dragDistance <= 4) {{
        selectedPoint = pickPoint(event);
        updateDetail(selectedPoint);
        renderChart();
      }}
      dragDistance = 0;
    }});

    canvas.addEventListener("pointerleave", () => {{
      isDragging = false;
      lastPointer = null;
      dragDistance = 0;
      canvas.classList.remove("dragging");
    }});

    canvas.addEventListener("wheel", (event) => {{
      event.preventDefault();
      const delta = event.deltaY < 0 ? 1.08 : 0.92;
      view.zoom = Math.min(3.2, Math.max(0.45, view.zoom * delta));
      renderChart();
    }}, {{ passive: false }});

    clusterFilter.addEventListener("change", renderChart);
    noiseOnly.addEventListener("change", renderChart);
    resetView.addEventListener("click", () => {{
      view.rotationX = defaultView.rotationX;
      view.rotationY = defaultView.rotationY;
      view.zoom = defaultView.zoom;
      renderChart();
    }});
    resetSelection.addEventListener("click", () => {{
      clusterFilter.value = "all";
      noiseOnly.checked = false;
      selectedPoint = null;
      view.rotationX = defaultView.rotationX;
      view.rotationY = defaultView.rotationY;
      view.zoom = defaultView.zoom;
      updateDetail(null);
      renderChart();
    }});
    renderChart();
  </script>
</body>
</html>"""
