from __future__ import annotations

import math

import pandas as pd

from data_process.semantic_deduplication import (
    SemanticDeduplicator,
    semantic_deduplicate_dataframe,
)


class FakeVector(list):
    def reshape(self, *_args):
        return [self]


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
        return [FakeVector(self.vector_map[text]) for text in texts]


class FakeFaissIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectors: list[list[float]] = []

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, vectors) -> None:
        self.vectors.append(list(vectors[0]))

    def search(self, query, top_k: int):
        vector = list(query[0])
        if not self.vectors:
            return [[0.0]], [[-1]]

        scored = [
            (_cosine_similarity(vector, candidate), index)
            for index, candidate in enumerate(self.vectors)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_index = scored[0]
        return [[best_score]], [[best_index]]


def test_semantic_deduplicate_dataframe_removes_semantic_duplicates() -> None:
    dataframe = pd.DataFrame(
        {"text": ["退款怎么申请", "怎么申请退款", "我要开发票", "发票如何申请"]}
    )
    vector_map = {
        "退款怎么申请": [1.0, 0.0],
        "怎么申请退款": [0.99, 0.01],
        "我要开发票": [0.0, 1.0],
        "发票如何申请": [0.02, 0.98],
    }
    deduplicator = SemanticDeduplicator(
        model_path="models/m3e-base",
        threshold=0.95,
        batch_size=8,
        model=FakeModel(vector_map),
        index=FakeFaissIndex(2),
    )

    deduplicated, stats, matches = deduplicator.deduplicate_dataframe(
        dataframe,
        collect_matches=True,
    )

    assert deduplicated["text"].tolist() == ["退款怎么申请", "我要开发票"]
    assert stats.dedupe_mode == "semantic"
    assert stats.semantic_threshold == 0.95
    assert stats.embedding_model_path == "models/m3e-base"
    assert stats.duplicate_rows == 2
    assert stats.unique_values == 2
    assert matches[0].duplicate_of_row_index == 0
    assert matches[1].duplicate_of_row_index == 2


def test_semantic_deduplicate_dataframe_tracks_blank_duplicates() -> None:
    dataframe = pd.DataFrame({"text": ["", " ", "有效内容"]})
    vector_map = {"有效内容": [1.0, 0.0]}
    deduplicator = SemanticDeduplicator(
        model_path="models/m3e-base",
        threshold=0.9,
        model=FakeModel(vector_map),
        index=FakeFaissIndex(2),
    )

    deduplicated, stats, matches = deduplicator.deduplicate_dataframe(
        dataframe,
        collect_matches=True,
    )

    assert deduplicated["text"].tolist() == ["", "有效内容"]
    assert stats.duplicate_rows == 1
    assert stats.unique_values == 2
    assert matches[0].similarity == 1.0


def test_semantic_deduplicate_dataframe_reports_progress() -> None:
    dataframe = pd.DataFrame({"text": ["a", "b", "c"]})
    vector_map = {
        "a": [1.0, 0.0],
        "b": [0.0, 1.0],
        "c": [0.7, 0.7],
    }
    reported: list[int] = []
    deduplicator = SemanticDeduplicator(
        model_path="models/m3e-base",
        threshold=0.95,
        model=FakeModel(vector_map),
        index=FakeFaissIndex(2),
    )

    deduplicator.deduplicate_dataframe(
        dataframe,
        progress_callback=reported.append,
        report_every=2,
    )

    assert reported == [2, 1]


def test_semantic_deduplicate_dataframe_uses_target_column() -> None:
    dataframe = pd.DataFrame(
        {
            "客户问题": ["退款怎么申请", "怎么申请退款"],
            "text": ["row1", "row2"],
        }
    )
    vector_map = {
        "退款怎么申请": [1.0, 0.0],
        "怎么申请退款": [0.98, 0.02],
    }

    deduplicated, stats, _ = semantic_deduplicate_dataframe(
        dataframe,
        target_column="客户问题",
        threshold=0.95,
        deduplicator=SemanticDeduplicator(
            model_path="models/m3e-base",
            threshold=0.95,
            model=FakeModel(vector_map),
            index=FakeFaissIndex(2),
        ),
    )

    assert deduplicated["text"].tolist() == ["row1"]
    assert stats.target_column == "客户问题"


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot_product / (left_norm * right_norm)
