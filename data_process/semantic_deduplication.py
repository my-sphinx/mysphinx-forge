from __future__ import annotations

import io
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data_process.cleaning import resolve_target_column
from data_process.deduplication import DeduplicationStats, normalize_dedup_text


DEFAULT_EMBEDDING_MODEL_PATH = Path("models/m3e-base")
_BENIGN_MODEL_LOAD_OUTPUT_MARKERS = (
    "BertModel LOAD REPORT",
    "embeddings.position_ids",
    "UNEXPECTED",
)


@dataclass(slots=True)
class SemanticDeduplicationMatch:
    row_index: int
    duplicate_of_row_index: int
    text: str
    matched_text: str
    category: object | None
    matched_category: object | None
    similarity: float


class SemanticDeduplicator:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_EMBEDDING_MODEL_PATH,
        threshold: float = 0.9,
        batch_size: int = 64,
        index_type: str = "flat",
        hnsw_m: int = 32,
        model: object | None = None,
        index: object | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.batch_size = batch_size
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self._model = model
        self._index = index
        self._representative_row_indices: list[int] = []
        self._representative_texts: list[str] = []
        self._representative_categories: list[object | None] = []
        self._seen_blank = False
        self._blank_row_index: int | None = None
        self._blank_text = ""
        self._blank_category: object | None = None

    def deduplicate_dataframe(
        self,
        dataframe: pd.DataFrame,
        target_column: str = "text",
        progress_callback: Callable[[int], None] | None = None,
        report_every: int = 1_000,
        row_index_offset: int = 0,
        collect_matches: bool = False,
    ) -> tuple[pd.DataFrame, DeduplicationStats, list[SemanticDeduplicationMatch]]:
        resolved_target_column = resolve_target_column(dataframe, target_column)
        normalized_texts = [normalize_dedup_text(value) for value in dataframe[resolved_target_column].tolist()]
        categories = (
            dataframe["category"].tolist()
            if "category" in dataframe.columns
            else [None] * len(dataframe)
        )

        stats = DeduplicationStats(
            total_before=len(dataframe),
            total_after=0,
            target_column=resolved_target_column,
            dedupe_mode="semantic",
            semantic_threshold=self.threshold,
            embedding_model_path=str(self.model_path),
        )
        keep_mask: list[bool] = []
        processed_since_report = 0
        matches: list[SemanticDeduplicationMatch] = []
        pending_rows: list[tuple[int, str]] = []

        for row_index, normalized_text in enumerate(normalized_texts):
            processed_since_report += 1
            global_row_index = row_index_offset + row_index
            category = categories[row_index]

            if normalized_text == "":
                # Blank values do not need embeddings, but they still need to preserve
                # row order so the keep mask lines up with the original dataframe.
                self._flush_pending_rows(
                    pending_rows=pending_rows,
                    keep_mask=keep_mask,
                    stats=stats,
                    matches=matches,
                    collect_matches=collect_matches,
                )
                is_duplicate, match = self._handle_blank_text(
                    row_index=global_row_index,
                    text=normalized_text,
                    category=category,
                )
            else:
                pending_rows.append((global_row_index, normalized_text, category))
                if len(pending_rows) >= self.batch_size:
                    self._flush_pending_rows(
                        pending_rows=pending_rows,
                        keep_mask=keep_mask,
                        stats=stats,
                        matches=matches,
                        collect_matches=collect_matches,
                    )
                if progress_callback and processed_since_report >= report_every:
                    progress_callback(processed_since_report)
                    processed_since_report = 0
                continue

            self._record_row_result(
                is_duplicate=is_duplicate,
                match=match,
                keep_mask=keep_mask,
                stats=stats,
                matches=matches,
                collect_matches=collect_matches,
            )

            if progress_callback and processed_since_report >= report_every:
                progress_callback(processed_since_report)
                processed_since_report = 0

        self._flush_pending_rows(
            pending_rows=pending_rows,
            keep_mask=keep_mask,
            stats=stats,
            matches=matches,
            collect_matches=collect_matches,
        )

        if progress_callback and processed_since_report > 0:
            progress_callback(processed_since_report)

        deduplicated = dataframe.loc[keep_mask].reset_index(drop=True)
        stats.total_after = len(deduplicated)
        stats.unique_values = len(self._representative_row_indices) + int(self._seen_blank)
        return deduplicated, stats, matches

    def _handle_blank_text(
        self,
        row_index: int,
        text: str,
        category: object | None,
    ) -> tuple[bool, SemanticDeduplicationMatch | None]:
        if not self._seen_blank:
            self._seen_blank = True
            self._blank_row_index = row_index
            self._blank_text = text
            self._blank_category = category
            return False, None

        return True, SemanticDeduplicationMatch(
            row_index=row_index,
            duplicate_of_row_index=self._blank_row_index if self._blank_row_index is not None else row_index,
            text=text,
            matched_text=self._blank_text,
            category=category,
            matched_category=self._blank_category,
            similarity=1.0,
        )

    def _handle_vector(
        self,
        vector,
        row_index: int,
        text: str,
        category: object | None,
    ) -> tuple[bool, SemanticDeduplicationMatch | None]:
        if self._index is None or self._index.ntotal == 0:
            self._add_representative(vector, row_index, text, category)
            return False, None

        distances, indices = self._index.search(vector.reshape(1, -1), 1)
        similarity = float(distances[0][0])
        matched_index = int(indices[0][0])

        if matched_index >= 0 and similarity >= self.threshold:
            return True, SemanticDeduplicationMatch(
                row_index=row_index,
                duplicate_of_row_index=self._representative_row_indices[matched_index],
                text=text,
                matched_text=self._representative_texts[matched_index],
                category=category,
                matched_category=self._representative_categories[matched_index],
                similarity=similarity,
            )

        self._add_representative(vector, row_index, text, category)
        return False, None

    def _add_representative(
        self,
        vector,
        row_index: int,
        text: str,
        category: object | None,
    ) -> None:
        self._ensure_index(len(vector))
        self._index.add(vector.reshape(1, -1))
        self._representative_row_indices.append(row_index)
        self._representative_texts.append(text)
        self._representative_categories.append(category)

    def _encode_texts(self, texts: list[str]):
        model = self._ensure_model()
        return model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _flush_pending_rows(
        self,
        pending_rows: list[tuple[int, str, object | None]],
        keep_mask: list[bool],
        stats: DeduplicationStats,
        matches: list[SemanticDeduplicationMatch],
        collect_matches: bool,
    ) -> None:
        if not pending_rows:
            return

        # Encode and judge one batch at a time so large datasets do not hold
        # a full in-memory embedding matrix in addition to the FAISS index.
        batch_rows = pending_rows.copy()
        pending_rows.clear()
        texts = [text for _, text, _ in batch_rows]
        vectors = self._encode_texts(texts)

        for (row_index, text, category), vector in zip(batch_rows, vectors, strict=True):
            is_duplicate, match = self._handle_vector(
                vector=vector,
                row_index=row_index,
                text=text,
                category=category,
            )
            self._record_row_result(
                is_duplicate=is_duplicate,
                match=match,
                keep_mask=keep_mask,
                stats=stats,
                matches=matches,
                collect_matches=collect_matches,
            )

    def _record_row_result(
        self,
        is_duplicate: bool,
        match: SemanticDeduplicationMatch | None,
        keep_mask: list[bool],
        stats: DeduplicationStats,
        matches: list[SemanticDeduplicationMatch],
        collect_matches: bool,
    ) -> None:
        if is_duplicate:
            stats.duplicate_rows += 1
            keep_mask.append(False)
            if collect_matches and match is not None:
                matches.append(match)
            return

        keep_mask.append(True)

    def _ensure_model(self):
        if self._model is None:
            self._model = _load_embedding_model(self.model_path)
        return self._model

    def _ensure_index(self, dimension: int) -> None:
        if self._index is None:
            self._index = _create_faiss_index(
                dimension=dimension,
                index_type=self.index_type,
                hnsw_m=self.hnsw_m,
            )


def semantic_deduplicate_dataframe(
    dataframe: pd.DataFrame,
    target_column: str = "text",
    threshold: float = 0.9,
    model_path: str | Path = DEFAULT_EMBEDDING_MODEL_PATH,
    batch_size: int = 64,
    index_type: str = "flat",
    hnsw_m: int = 32,
    progress_callback: Callable[[int], None] | None = None,
    report_every: int = 1_000,
    row_index_offset: int = 0,
    collect_matches: bool = False,
    deduplicator: SemanticDeduplicator | None = None,
) -> tuple[pd.DataFrame, DeduplicationStats, list[SemanticDeduplicationMatch]]:
    active_deduplicator = deduplicator or SemanticDeduplicator(
        model_path=model_path,
        threshold=threshold,
        batch_size=batch_size,
        index_type=index_type,
        hnsw_m=hnsw_m,
    )
    return active_deduplicator.deduplicate_dataframe(
        dataframe,
        target_column=target_column,
        progress_callback=progress_callback,
        report_every=report_every,
        row_index_offset=row_index_offset,
        collect_matches=collect_matches,
    )


def _load_embedding_model(model_path: Path):
    if not model_path.exists():
        raise ValueError(f"未找到语义去重模型：{model_path}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ValueError(
            "未安装 sentence-transformers，请先执行 uv sync。"
        ) from exc

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    process_output: _CapturedProcessOutput | None = None

    try:
        with (
            _capture_process_output() as process_output,
            redirect_stdout(stdout_buffer),
            redirect_stderr(stderr_buffer),
        ):
            model = SentenceTransformer(str(model_path), local_files_only=True)
    except Exception:
        _replay_model_load_output(
            stdout_text=stdout_buffer.getvalue(),
            stderr_text=stderr_buffer.getvalue(),
            process_text=process_output.read() if process_output is not None else "",
        )
        raise

    _replay_model_load_output(
        stdout_text=stdout_buffer.getvalue(),
        stderr_text=stderr_buffer.getvalue(),
        process_text=process_output.read() if process_output is not None else "",
    )
    return model


def _replay_model_load_output(stdout_text: str, stderr_text: str, process_text: str) -> None:
    captured_output = "\n".join(
        text.strip()
        for text in (stdout_text, stderr_text, process_text)
        if text.strip()
    )
    if captured_output and _is_benign_model_load_output(captured_output):
        return

    if stdout_text:
        sys.stdout.write(stdout_text)
    if stderr_text:
        sys.stderr.write(stderr_text)
    if process_text:
        try:
            os.write(sys.__stdout__.fileno(), process_text.encode())
        except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
            sys.stdout.write(process_text)


def _is_benign_model_load_output(output: str) -> bool:
    return all(marker in output for marker in _BENIGN_MODEL_LOAD_OUTPUT_MARKERS)


class _CapturedProcessOutput:
    def __init__(self, stdout_file, stderr_file) -> None:
        self._stdout_file = stdout_file
        self._stderr_file = stderr_file

    def read(self) -> str:
        parts: list[str] = []
        for file_obj in (self._stdout_file, self._stderr_file):
            file_obj.flush()
            file_obj.seek(0)
            content = file_obj.read().decode()
            if content:
                parts.append(content)
        return "".join(parts)


class _capture_process_output:
    def __enter__(self) -> _CapturedProcessOutput:
        self._stdout_file = tempfile.TemporaryFile(mode="w+b")
        self._stderr_file = tempfile.TemporaryFile(mode="w+b")
        self._stdout_fd: int | None = None
        self._stderr_fd: int | None = None
        _safe_flush(sys.stdout)
        _safe_flush(sys.stderr)
        # Some model backends write directly to the process file descriptors
        # instead of Python's sys.stdout/sys.stderr, so redirect both layers.
        try:
            self._stdout_fd = os.dup(sys.__stdout__.fileno())
            self._stderr_fd = os.dup(sys.__stderr__.fileno())
            os.dup2(self._stdout_file.fileno(), sys.__stdout__.fileno())
            os.dup2(self._stderr_file.fileno(), sys.__stderr__.fileno())
        except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
            self._stdout_fd = None
            self._stderr_fd = None
        return _CapturedProcessOutput(self._stdout_file, self._stderr_file)

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        _safe_flush(sys.stdout)
        _safe_flush(sys.stderr)
        if self._stdout_fd is not None and self._stderr_fd is not None:
            os.dup2(self._stdout_fd, sys.__stdout__.fileno())
            os.dup2(self._stderr_fd, sys.__stderr__.fileno())
            os.close(self._stdout_fd)
            os.close(self._stderr_fd)
        self._stdout_file.flush()
        self._stderr_file.flush()


def _safe_flush(stream) -> None:
    try:
        stream.flush()
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        return


class _NumpyFlatIPIndex:
    def __init__(self, dimension: int, initial_capacity: int = 1024) -> None:
        self.dimension = dimension
        self._capacity = max(initial_capacity, 1)
        self._matrix = np.empty((self._capacity, dimension), dtype=np.float32)
        self._size = 0

    @property
    def ntotal(self) -> int:
        return self._size

    def add(self, vectors) -> None:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != self.dimension:
            raise ValueError(f"向量维度不匹配：期望 {self.dimension}，实际 {array.shape[1]}")

        needed = self._size + array.shape[0]
        if needed > self._capacity:
            while self._capacity < needed:
                self._capacity *= 2
            expanded = np.empty((self._capacity, self.dimension), dtype=np.float32)
            expanded[: self._size] = self._matrix[: self._size]
            self._matrix = expanded

        self._matrix[self._size : needed] = array
        self._size = needed

    def search(self, query, top_k: int):
        if self._size == 0:
            return np.zeros((1, top_k), dtype=np.float32), np.full((1, top_k), -1, dtype=np.int64)

        query_array = np.asarray(query, dtype=np.float32)
        if query_array.ndim == 1:
            query_array = query_array.reshape(1, -1)

        scores = query_array @ self._matrix[: self._size].T
        if top_k == 1:
            best_indices = np.argmax(scores, axis=1)
            best_scores = scores[np.arange(len(best_indices)), best_indices]
            return best_scores.reshape(-1, 1), best_indices.reshape(-1, 1)

        ranked = np.argsort(scores, axis=1)[:, -top_k:][:, ::-1]
        ranked_scores = np.take_along_axis(scores, ranked, axis=1)
        return ranked_scores, ranked


def _create_faiss_index(dimension: int, index_type: str = "flat", hnsw_m: int = 32):
    if index_type == "flat" and os.name == "nt":
        # Avoid Windows-native access violations observed in faiss flat search/add paths.
        return _NumpyFlatIPIndex(dimension)

    try:
        import faiss
    except ImportError as exc:
        raise ValueError("未安装 faiss-cpu，请先执行 uv sync。") from exc

    if index_type == "flat":
        return faiss.IndexFlatIP(dimension)
    if index_type == "hnsw":
        return faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    raise ValueError(f"不支持的语义索引类型：{index_type}")
