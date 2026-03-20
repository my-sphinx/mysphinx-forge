from __future__ import annotations

import pandas as pd

from mysphinx_forge.clustering import TextClusterer


class FakeModel:
    def __init__(self, vector_map: dict[str, list[float]]) -> None:
        self.vector_map = vector_map

    def encode(
        self,
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        return [self.vector_map[text] for text in texts]


class FakeEstimator:
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels

    def fit_predict(self, _vectors):
        return self.labels


class FakeClusterLabelGenerator:
    def generate_label(self, context) -> str:
        if context.cluster_id == 0:
            return "售后退款流程"
        if context.cluster_id == 1:
            return "发票开具"
        return ""


def test_text_clusterer_builds_cluster_columns_and_summary() -> None:
    dataframe = pd.DataFrame(
        {
            "text": ["退款怎么申请", "怎么申请退款", "我要开发票", ""],
            "category": ["售后", "售后-重复", "财务", "空白"],
        }
    )
    clusterer = TextClusterer(
        model_path="models/m3e-base",
        cluster_mode="hdbscan",
        model=FakeModel(
            {
                "退款怎么申请": [1.0, 0.0],
                "怎么申请退款": [0.99, 0.01],
                "我要开发票": [0.0, 1.0],
            }
        ),
        estimator=FakeEstimator([0, 0, 1]),
    )

    clustered, summary, projection, stats = clusterer.cluster_dataframe(dataframe)

    assert clustered["cluster_id"].tolist() == [0, 0, 1, -1]
    assert clustered["is_noise"].tolist() == [False, False, False, True]
    assert clustered["cluster_size"].tolist() == [2, 2, 1, 1]
    assert clustered["cluster_representative_text"].tolist() == [
        "退款怎么申请",
        "退款怎么申请",
        "我要开发票",
        "",
    ]
    assert clustered["cluster_top_keywords"].tolist() == [
        "申请, 退款",
        "申请, 退款",
        "发票, 开发",
        "",
    ]
    assert clustered["cluster_label"].tolist() == [
        "退款怎么申请",
        "退款怎么申请",
        "我要开发票",
        "",
    ]
    assert summary["cluster_id"].tolist() == [0, 1]
    assert summary["cluster_size"].tolist() == [2, 1]
    assert summary["top_keywords"].tolist() == [
        "申请, 退款",
        "发票, 开发",
    ]
    assert summary["cluster_label"].tolist() == ["退款怎么申请", "我要开发票"]
    assert projection["cluster_id"].tolist() == [0, 0, 1, -1]
    assert projection["text"].tolist() == ["退款怎么申请", "怎么申请退款", "我要开发票", ""]
    assert projection["x"].notna().tolist() == [True, True, True, False]
    assert projection["y"].notna().tolist() == [True, True, True, False]
    assert projection["z"].notna().tolist() == [True, True, True, False]
    assert stats.total_before == 4
    assert stats.total_clustered == 3
    assert stats.noise_rows == 1
    assert stats.cluster_count == 2


def test_text_clusterer_validates_kmeans_cluster_count() -> None:
    dataframe = pd.DataFrame({"text": ["a", "b"]})
    clusterer = TextClusterer(
        model_path="models/m3e-base",
        cluster_mode="kmeans",
        num_clusters=3,
        model=FakeModel({"a": [1.0, 0.0], "b": [0.0, 1.0]}),
    )

    try:
        clusterer.cluster_dataframe(dataframe)
    except ValueError as exc:
        assert "KMeans 聚类需要至少 3 条非空文本" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_text_clusterer_supports_custom_cluster_label_generator() -> None:
    dataframe = pd.DataFrame(
        {
            "text": ["退款怎么申请", "怎么申请退款", "我要开发票", ""],
        }
    )
    clusterer = TextClusterer(
        model_path="models/m3e-base",
        cluster_mode="hdbscan",
        cluster_label_mode="llm",
        cluster_label_generator=FakeClusterLabelGenerator(),
        model=FakeModel(
            {
                "退款怎么申请": [1.0, 0.0],
                "怎么申请退款": [0.99, 0.01],
                "我要开发票": [0.0, 1.0],
            }
        ),
        estimator=FakeEstimator([0, 0, 1]),
    )

    clustered, summary, _, stats = clusterer.cluster_dataframe(dataframe)

    assert clustered["cluster_label"].tolist() == [
        "售后退款流程",
        "售后退款流程",
        "发票开具",
        "",
    ]
    assert summary["cluster_label"].tolist() == ["售后退款流程", "发票开具"]
    assert stats.cluster_label_mode == "llm"
    assert stats.cluster_label_model == "gpt-4.1-mini"
