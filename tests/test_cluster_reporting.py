from __future__ import annotations

import pandas as pd

from mysphinx_forge.cluster_reporting import (
    build_cluster_analysis_report,
    render_cluster_report_html,
)
from mysphinx_forge.clustering import ClusteringStats


def test_build_cluster_analysis_report_adds_rank_and_ratio() -> None:
    cluster_summary = pd.DataFrame(
        {
            "cluster_id": [1, 0],
            "cluster_size": [1, 3],
            "cluster_label": ["发票", "退款"],
            "top_keywords": ["发票", "退款, 申请"],
            "representative_text": ["发票怎么开", "退款怎么申请"],
            "example_texts": ["发票怎么开", "退款怎么申请 | 怎么申请退款"],
        }
    )
    stats = ClusteringStats(
        total_before=5,
        total_clustered=4,
        cluster_count=2,
        noise_rows=1,
        largest_cluster_size=3,
        smallest_cluster_size=1,
        average_cluster_size=2.0,
        target_column="text",
        cluster_mode="hdbscan",
        embedding_model_path="models/m3e-base",
    )

    report = build_cluster_analysis_report(cluster_summary, stats)

    assert report["cluster_rank"].tolist() == [1, 2]
    assert report["cluster_id"].tolist() == [0, 1]
    assert report["cluster_ratio"].tolist() == [0.75, 0.25]


def test_render_cluster_report_html_embeds_summary_and_points() -> None:
    analysis_report = pd.DataFrame(
        {
            "cluster_rank": [1],
            "cluster_id": [0],
            "cluster_size": [2],
            "cluster_ratio": [1.0],
            "cluster_label": ["退款"],
            "top_keywords": ["退款, 申请"],
            "representative_text": ["退款怎么申请"],
            "example_texts": ["退款怎么申请 | 怎么申请退款"],
        }
    )
    projection = pd.DataFrame(
        {
            "row_index": [0, 1],
            "text": ["退款怎么申请", "怎么申请退款"],
            "cluster_id": [0, 0],
            "is_noise": [False, False],
            "x": [0.1, 0.2],
            "y": [0.3, 0.4],
            "z": [0.5, 0.6],
        }
    )
    stats = ClusteringStats(
        total_before=2,
        total_clustered=2,
        cluster_count=1,
        noise_rows=0,
        largest_cluster_size=2,
        smallest_cluster_size=2,
        average_cluster_size=2.0,
        target_column="text",
        cluster_mode="hdbscan",
        embedding_model_path="models/m3e-base",
    )

    html = render_cluster_report_html(
        analysis_report=analysis_report,
        projection=projection,
        stats=stats,
    )

    assert "Cluster Report" in html
    assert "退款怎么申请" in html
    assert '"cluster_id": 0' in html
    assert 'id="clusterFilter"' in html
    assert 'id="noiseOnly"' in html
    assert 'id="detailPanel"' in html
