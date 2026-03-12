from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from data_process.cleaning import resolve_target_column
from data_process.deduplication import DeduplicationStats, normalize_dedup_text


DEFAULT_EMBEDDING_MODEL_PATH = Path("models/m3e-base")


@dataclass(slots=True)
class SemanticDeduplicationMatch:
    row_index: int
    duplicate_of_row_index: int
    text: str
    matched_text: str
    similarity: float


class SemanticDeduplicator:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_EMBEDDING_MODEL_PATH,
        threshold: float = 0.9,
        batch_size: int = 64,
        model: object | None = None,
        index: object | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.batch_size = batch_size
        self._model = model
        self._index = index
        self._representative_row_indices: list[int] = []
        self._representative_texts: list[str] = []
        self._seen_blank = False
        self._blank_row_index: int | None = None
        self._blank_text = ""

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
        embeddings = self._encode_non_blank_texts(normalized_texts)

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
        embedding_index = 0
        matches: list[SemanticDeduplicationMatch] = []

        for row_index, normalized_text in enumerate(normalized_texts):
            processed_since_report += 1
            global_row_index = row_index_offset + row_index

            if normalized_text == "":
                is_duplicate, match = self._handle_blank_text(
                    row_index=global_row_index,
                    text=normalized_text,
                )
            else:
                vector = embeddings[embedding_index]
                embedding_index += 1
                is_duplicate, match = self._handle_vector(
                    vector=vector,
                    row_index=global_row_index,
                    text=normalized_text,
                )

            if is_duplicate:
                stats.duplicate_rows += 1
                keep_mask.append(False)
                if collect_matches and match is not None:
                    matches.append(match)
            else:
                keep_mask.append(True)

            if progress_callback and processed_since_report >= report_every:
                progress_callback(processed_since_report)
                processed_since_report = 0

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
    ) -> tuple[bool, SemanticDeduplicationMatch | None]:
        if not self._seen_blank:
            self._seen_blank = True
            self._blank_row_index = row_index
            self._blank_text = text
            return False, None

        return True, SemanticDeduplicationMatch(
            row_index=row_index,
            duplicate_of_row_index=self._blank_row_index if self._blank_row_index is not None else row_index,
            text=text,
            matched_text=self._blank_text,
            similarity=1.0,
        )

    def _handle_vector(
        self,
        vector,
        row_index: int,
        text: str,
    ) -> tuple[bool, SemanticDeduplicationMatch | None]:
        if self._index is None or self._index.ntotal == 0:
            self._add_representative(vector, row_index, text)
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
                similarity=similarity,
            )

        self._add_representative(vector, row_index, text)
        return False, None

    def _add_representative(self, vector, row_index: int, text: str) -> None:
        self._ensure_index(len(vector))
        self._index.add(vector.reshape(1, -1))
        self._representative_row_indices.append(row_index)
        self._representative_texts.append(text)

    def _encode_non_blank_texts(self, normalized_texts: list[str]):
        texts_to_encode = [text for text in normalized_texts if text]
        if not texts_to_encode:
            return []

        model = self._ensure_model()
        return model.encode(
            texts_to_encode,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _ensure_model(self):
        if self._model is None:
            self._model = _load_embedding_model(self.model_path)
        return self._model

    def _ensure_index(self, dimension: int) -> None:
        if self._index is None:
            self._index = _create_faiss_index(dimension)


def semantic_deduplicate_dataframe(
    dataframe: pd.DataFrame,
    target_column: str = "text",
    threshold: float = 0.9,
    model_path: str | Path = DEFAULT_EMBEDDING_MODEL_PATH,
    batch_size: int = 64,
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

    return SentenceTransformer(str(model_path))


def _create_faiss_index(dimension: int):
    try:
        import faiss
    except ImportError as exc:
        raise ValueError("未安装 faiss-cpu，请先执行 uv sync。") from exc

    return faiss.IndexFlatIP(dimension)
