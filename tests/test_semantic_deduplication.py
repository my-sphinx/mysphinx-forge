from __future__ import annotations

import io
import math
import sys
import types

import pandas as pd

from data_process.semantic_deduplication import (
    SemanticDeduplicator,
    _create_faiss_index,
    _capture_process_output,
    _load_embedding_model,
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


class RecordingFakeModel(FakeModel):
    def __init__(self, vector_map: dict[str, list[float]]) -> None:
        super().__init__(vector_map)
        self.calls: list[list[str]] = []

    def encode(
        self,
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        self.calls.append(list(texts))
        return super().encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar,
        )


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
        {
            "text": ["退款怎么申请", "怎么申请退款", "我要开发票", "发票如何申请"],
            "category": ["售后", "售后-重复", "财务", "财务-重复"],
        }
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
    assert matches[0].category == "售后-重复"
    assert matches[0].matched_category == "售后"
    assert matches[1].category == "财务-重复"
    assert matches[1].matched_category == "财务"


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


def test_semantic_deduplicate_dataframe_streams_embedding_batches() -> None:
    dataframe = pd.DataFrame({"text": ["a", "b", "c", " ", "d"]})
    vector_map = {
        "a": [1.0, 0.0],
        "b": [0.0, 1.0],
        "c": [0.7, 0.7],
        "d": [0.4, 0.9],
    }
    model = RecordingFakeModel(vector_map)
    deduplicator = SemanticDeduplicator(
        model_path="models/m3e-base",
        threshold=0.95,
        batch_size=2,
        model=model,
        index=FakeFaissIndex(2),
    )

    deduplicated, stats, _ = deduplicator.deduplicate_dataframe(dataframe)

    assert deduplicated["text"].tolist() == ["a", "b", "c", " ", "d"]
    assert stats.total_after == 5
    assert model.calls == [["a", "b"], ["c"], ["d"]]


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


def test_load_embedding_model_suppresses_known_benign_load_report(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    model_dir = tmp_path / "m3e-base"
    model_dir.mkdir()

    class FakeSentenceTransformer:
        def __init__(self, _model_path: str, **_kwargs) -> None:
            print("BertModel LOAD REPORT from: models/m3e-base")
            print("embeddings.position_ids | UNEXPECTED |")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    model = _load_embedding_model(model_dir)

    captured = capsys.readouterr()
    assert isinstance(model, FakeSentenceTransformer)
    assert captured.out == ""
    assert captured.err == ""


def test_load_embedding_model_replays_non_benign_output(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    model_dir = tmp_path / "m3e-base"
    model_dir.mkdir()

    class FakeSentenceTransformer:
        def __init__(self, _model_path: str, **_kwargs) -> None:
            print("loading custom backend")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    _load_embedding_model(model_dir)

    captured = capsys.readouterr()
    assert "loading custom backend" in captured.out


def test_load_embedding_model_tolerates_missing_stdio_fileno(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    model_dir = tmp_path / "m3e-base"
    model_dir.mkdir()

    class FakeSentenceTransformer:
        def __init__(self, _model_path: str, **_kwargs) -> None:
            print("loading without stdio fileno")

    class NoFilenoStream(io.StringIO):
        def fileno(self) -> int:
            raise io.UnsupportedOperation("no fileno")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setattr(sys, "__stdout__", NoFilenoStream())
    monkeypatch.setattr(sys, "__stderr__", NoFilenoStream())

    model = _load_embedding_model(model_dir)

    captured = capsys.readouterr()
    assert isinstance(model, FakeSentenceTransformer)
    assert "loading without stdio fileno" in captured.out


def test_capture_process_output_tolerates_flush_oserror(monkeypatch) -> None:
    class BrokenFlushStream(io.StringIO):
        def flush(self) -> None:
            raise OSError(6, "Invalid handle")

    monkeypatch.setattr(sys, "stdout", BrokenFlushStream())
    monkeypatch.setattr(sys, "stderr", BrokenFlushStream())

    with _capture_process_output() as output:
        assert output is not None


def test_create_faiss_index_uses_numpy_flat_backend_on_windows(monkeypatch) -> None:
    monkeypatch.setattr("data_process.semantic_deduplication.os.name", "nt")

    index = _create_faiss_index(dimension=2, index_type="flat")
    index.add([[1.0, 0.0], [0.0, 1.0]])
    scores, indices = index.search([[0.9, 0.1]], 1)

    assert index.ntotal == 2
    assert int(indices[0][0]) == 0
    assert float(scores[0][0]) > 0.8


def test_create_faiss_index_supports_hnsw(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    class FakeFaissModule:
        METRIC_INNER_PRODUCT = 0

        @staticmethod
        def IndexHNSWFlat(dimension: int, hnsw_m: int, metric: int):
            recorded["dimension"] = dimension
            recorded["hnsw_m"] = hnsw_m
            recorded["metric"] = metric
            return "hnsw-index"

    monkeypatch.setitem(sys.modules, "faiss", FakeFaissModule)

    index = _create_faiss_index(dimension=768, index_type="hnsw", hnsw_m=48)

    assert index == "hnsw-index"
    assert recorded == {"dimension": 768, "hnsw_m": 48, "metric": 0}


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot_product / (left_norm * right_norm)
