from __future__ import annotations

import json
from html import escape

import pandas as pd

from data_process.clustering import ClusteringStats


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
      border-radius: 18px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.86), rgba(247,242,235,0.96));
      border: 1px solid rgba(31, 41, 55, 0.08);
    }}
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
          <button id="resetSelection" type="button">Reset</button>
        </div>
        <svg id="chart" viewBox="0 0 720 480" preserveAspectRatio="none"></svg>
        <div class="legend">
          <span>点颜色按簇区分</span>
          <span>噪声点使用橙色高亮</span>
          <span>空白或不可投影样本会被忽略</span>
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
    const svg = document.getElementById("chart");
    const clusterFilter = document.getElementById("clusterFilter");
    const noiseOnly = document.getElementById("noiseOnly");
    const resetSelection = document.getElementById("resetSelection");
    const detailPanel = document.getElementById("detailPanel");
    const width = 720;
    const height = 480;
    const padding = 36;
    const palette = ["#1d4ed8", "#0f766e", "#7c3aed", "#be123c", "#0369a1", "#4d7c0f", "#c2410c", "#334155"];
    const projectedPoints = payload.points.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
    const clusterOptions = [...new Set(projectedPoints.filter((point) => !point.is_noise).map((point) => point.cluster_id))].sort((a, b) => a - b);
    let selectedPoint = null;

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
        <p><strong>Coordinates:</strong> (${{Number(point.x).toFixed(3)}}, ${{Number(point.y).toFixed(3)}})</p>
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

    function renderChart() {{
      const points = filteredPoints();
      if (!points.length) {{
        svg.innerHTML = '<text x="50%" y="50%" text-anchor="middle" fill="#6b7280" font-size="18">No points match the current filter</text>';
        updateDetail(null);
        return;
      }}

      const xs = points.map((point) => point.x);
      const ys = points.map((point) => point.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const scaleX = (value) => padding + ((value - minX) / Math.max(maxX - minX, 1e-9)) * (width - padding * 2);
      const scaleY = (value) => height - padding - ((value - minY) / Math.max(maxY - minY, 1e-9)) * (height - padding * 2);
      const circles = points.map((point, index) => {{
        const color = point.is_noise ? "#ea580c" : palette[Math.abs(point.cluster_id) % palette.length];
        const label = String(point[payload.stats.target_column] ?? "");
        const isSelected = selectedPoint && selectedPoint.row_index === point.row_index;
        const radius = isSelected ? 8 : 5.2;
        const stroke = isSelected ? "#111827" : "rgba(255,255,255,0.85)";
        return `<circle data-index="${{index}}" cx="${{scaleX(point.x)}}" cy="${{scaleY(point.y)}}" r="${{radius}}" fill="${{color}}" stroke="${{stroke}}" stroke-width="1.4" fill-opacity="0.82">
          <title>${{label}} | cluster=${{point.cluster_id}}</title>
        </circle>`;
      }}).join("");
      svg.innerHTML = `
        <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="transparent"></rect>
        <line x1="${{padding}}" y1="${{height-padding}}" x2="${{width-padding}}" y2="${{height-padding}}" stroke="rgba(31,41,55,0.18)"></line>
        <line x1="${{padding}}" y1="${{padding}}" x2="${{padding}}" y2="${{height-padding}}" stroke="rgba(31,41,55,0.18)"></line>
        ${{circles}}
      `;
      [...svg.querySelectorAll("circle")].forEach((circle) => {{
        circle.addEventListener("click", () => {{
          selectedPoint = points[Number(circle.dataset.index)];
          updateDetail(selectedPoint);
          renderChart();
        }});
      }});
      if (selectedPoint) {{
        const stillVisible = points.some((point) => point.row_index === selectedPoint.row_index);
        if (!stillVisible) {{
          selectedPoint = null;
          updateDetail(null);
        }}
      }}
    }}

    clusterFilter.addEventListener("change", renderChart);
    noiseOnly.addEventListener("change", renderChart);
    resetSelection.addEventListener("click", () => {{
      clusterFilter.value = "all";
      noiseOnly.checked = false;
      selectedPoint = null;
      updateDetail(null);
      renderChart();
    }});
    renderChart();
  </script>
</body>
</html>"""
