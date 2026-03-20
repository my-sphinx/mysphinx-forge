from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data_process.cleaning import resolve_target_column
from data_process.embedding import load_embedding_model


DEFAULT_CLUSTER_RANDOM_STATE = 42
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
STOPWORDS = {
    "的",
    "了",
    "呢",
    "吗",
    "啊",
    "呀",
    "和",
    "或",
    "及",
    "与",
    "是",
    "在",
    "把",
    "将",
    "就",
    "都",
    "也",
    "还",
    "要",
    "我",
    "你",
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
}
STOP_CHARS = {"的", "了", "呢", "吗", "啊", "呀", "么", "是", "在", "把", "将", "就", "都", "也", "还", "要"}


@dataclass(slots=True)
class ClusteringStats:
    total_before: int
    total_clustered: int
    cluster_count: int
    noise_rows: int
    largest_cluster_size: int
    smallest_cluster_size: int
    average_cluster_size: float
    target_column: str = ""
    cluster_mode: str = "hdbscan"
    embedding_model_path: str | None = None


class TextClusterer:
    def __init__(
        self,
        model_path: str | Path,
        cluster_mode: str = "hdbscan",
        batch_size: int = 64,
        min_cluster_size: int = 5,
        num_clusters: int = 8,
        cluster_selection_epsilon: float = 0.0,
        random_state: int = DEFAULT_CLUSTER_RANDOM_STATE,
        model: object | None = None,
        estimator: object | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.cluster_mode = cluster_mode
        self.batch_size = batch_size
        self.min_cluster_size = min_cluster_size
        self.num_clusters = num_clusters
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.random_state = random_state
        self._model = model
        self._estimator = estimator

    def cluster_dataframe(
        self,
        dataframe: pd.DataFrame,
        target_column: str = "text",
        progress_callback: Callable[[int], None] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ClusteringStats]:
        resolved_target_column = resolve_target_column(dataframe, target_column)
        texts = [_cell_to_text(value) for value in dataframe[resolved_target_column].tolist()]
        active_indices = [index for index, text in enumerate(texts) if text]

        labels = np.full(len(dataframe), -1, dtype=int)
        vectors = np.empty((0, 0), dtype=np.float32)
        if active_indices:
            active_texts = [texts[index] for index in active_indices]
            vectors = self._encode_texts(active_texts)
            active_labels = self._fit_predict(vectors)
            labels[np.asarray(active_indices, dtype=int)] = np.asarray(active_labels, dtype=int)
            if progress_callback:
                progress_callback(len(active_texts))
        elif progress_callback:
            progress_callback(0)

        cluster_sizes = _build_cluster_sizes(labels)
        representative_texts = _build_representative_texts(
            labels=labels,
            texts=texts,
            active_indices=active_indices,
            vectors=vectors,
            cluster_sizes=cluster_sizes,
        )
        cluster_keywords = _build_cluster_keywords(labels=labels, texts=texts, cluster_sizes=cluster_sizes)
        cluster_labels = _build_cluster_labels(representative_texts, cluster_keywords)

        clustered = dataframe.copy()
        clustered["cluster_id"] = labels.tolist()
        clustered["is_noise"] = [label == -1 for label in labels]
        clustered["cluster_size"] = [cluster_sizes.get(label, 1 if label == -1 else 0) for label in labels]
        clustered["cluster_representative_text"] = [
            representative_texts.get(label, "") if label != -1 else ""
            for label in labels
        ]
        clustered["cluster_top_keywords"] = [
            cluster_keywords.get(label, "") if label != -1 else ""
            for label in labels
        ]
        clustered["cluster_label"] = [
            cluster_labels.get(label, "") if label != -1 else ""
            for label in labels
        ]

        summary = _build_cluster_summary(
            cluster_sizes,
            representative_texts,
            cluster_keywords,
            cluster_labels,
            labels,
            texts,
        )
        projection = _build_projection_frame(
            dataframe=dataframe,
            target_column=resolved_target_column,
            texts=texts,
            labels=labels,
            active_indices=active_indices,
            vectors=vectors,
        )
        stats = _build_clustering_stats(
            labels=labels,
            cluster_sizes=cluster_sizes,
            total_before=len(dataframe),
            target_column=resolved_target_column,
            cluster_mode=self.cluster_mode,
            embedding_model_path=str(self.model_path),
        )
        return clustered, summary, projection, stats

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        encoded = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(encoded, dtype=np.float32)

    def _fit_predict(self, vectors: np.ndarray) -> np.ndarray:
        if len(vectors) == 0:
            return np.empty((0,), dtype=int)

        if self.cluster_mode == "kmeans" and len(vectors) < self.num_clusters:
            raise ValueError(
                f"KMeans 聚类需要至少 {self.num_clusters} 条非空文本，当前仅有 {len(vectors)} 条。"
            )

        estimator = self._ensure_estimator()
        labels = estimator.fit_predict(vectors)
        return np.asarray(labels, dtype=int)

    def _ensure_model(self):
        if self._model is None:
            try:
                self._model = load_embedding_model(self.model_path)
            except ValueError as exc:
                message = str(exc)
                if "未找到语义模型" in message:
                    raise ValueError(message.replace("未找到语义模型", "未找到聚类模型")) from exc
                raise
        return self._model

    def _ensure_estimator(self):
        if self._estimator is not None:
            return self._estimator

        try:
            from sklearn.cluster import HDBSCAN, KMeans
        except ImportError as exc:
            raise ValueError("未安装 scikit-learn，请先执行 uv sync。") from exc

        if self.cluster_mode == "hdbscan":
            self._estimator = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric="euclidean",
                n_jobs=-1,
            )
            return self._estimator
        if self.cluster_mode == "kmeans":
            self._estimator = KMeans(
                n_clusters=self.num_clusters,
                random_state=self.random_state,
                n_init="auto",
            )
            return self._estimator
        raise ValueError(f"不支持的聚类模式：{self.cluster_mode}")


def _cell_to_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _build_cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    cluster_sizes: dict[int, int] = {}
    for label in labels:
        if label == -1:
            continue
        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
    return cluster_sizes


def _build_representative_texts(
    labels: np.ndarray,
    texts: list[str],
    active_indices: list[int],
    vectors: np.ndarray,
    cluster_sizes: dict[int, int],
) -> dict[int, str]:
    if not active_indices or vectors.size == 0:
        return {}

    active_index_to_vector = {
        dataframe_index: vector
        for dataframe_index, vector in zip(active_indices, vectors, strict=True)
    }
    representative_texts: dict[int, str] = {}
    for label in sorted(cluster_sizes):
        member_indices = [index for index, current_label in enumerate(labels) if current_label == label]
        member_vectors = np.asarray([active_index_to_vector[index] for index in member_indices], dtype=np.float32)
        centroid = member_vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        scores = member_vectors @ centroid
        representative_index = member_indices[int(np.argmax(scores))]
        representative_texts[label] = texts[representative_index]
    return representative_texts


def _build_cluster_summary(
    cluster_sizes: dict[int, int],
    representative_texts: dict[int, str],
    cluster_keywords: dict[int, str],
    cluster_labels: dict[int, str],
    labels: np.ndarray,
    texts: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in sorted(cluster_sizes, key=lambda item: (-cluster_sizes[item], item)):
        member_texts = [text for text, current_label in zip(texts, labels, strict=True) if current_label == label]
        rows.append(
            {
                "cluster_id": label,
                "cluster_size": cluster_sizes[label],
                "cluster_label": cluster_labels.get(label, ""),
                "top_keywords": cluster_keywords.get(label, ""),
                "representative_text": representative_texts.get(label, ""),
                "example_texts": " | ".join(member_texts[:3]),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "cluster_id",
            "cluster_size",
            "cluster_label",
            "top_keywords",
            "representative_text",
            "example_texts",
        ],
    )


def _build_clustering_stats(
    labels: np.ndarray,
    cluster_sizes: dict[int, int],
    total_before: int,
    target_column: str,
    cluster_mode: str,
    embedding_model_path: str,
) -> ClusteringStats:
    cluster_values = list(cluster_sizes.values())
    return ClusteringStats(
        total_before=total_before,
        total_clustered=total_before - int(np.sum(labels == -1)),
        cluster_count=len(cluster_sizes),
        noise_rows=int(np.sum(labels == -1)),
        largest_cluster_size=max(cluster_values, default=0),
        smallest_cluster_size=min(cluster_values, default=0),
        average_cluster_size=(sum(cluster_values) / len(cluster_values)) if cluster_values else 0.0,
        target_column=target_column,
        cluster_mode=cluster_mode,
        embedding_model_path=embedding_model_path,
    )


def _build_projection_frame(
    dataframe: pd.DataFrame,
    target_column: str,
    texts: list[str],
    labels: np.ndarray,
    active_indices: list[int],
    vectors: np.ndarray,
) -> pd.DataFrame:
    x_values = np.full(len(dataframe), np.nan, dtype=np.float32)
    y_values = np.full(len(dataframe), np.nan, dtype=np.float32)

    if active_indices and vectors.size > 0:
        coordinates = _project_vectors(vectors)
        for dataframe_index, coordinate in zip(active_indices, coordinates, strict=True):
            x_values[dataframe_index] = float(coordinate[0])
            y_values[dataframe_index] = float(coordinate[1])

    return pd.DataFrame(
        {
            "row_index": list(range(len(dataframe))),
            target_column: texts,
            "cluster_id": labels.tolist(),
            "is_noise": [label == -1 for label in labels],
            "x": x_values.tolist(),
            "y": y_values.tolist(),
        }
    )


def _build_cluster_keywords(
    labels: np.ndarray,
    texts: list[str],
    cluster_sizes: dict[int, int],
) -> dict[int, str]:
    cluster_keywords: dict[int, str] = {}
    for label in sorted(cluster_sizes):
        member_texts = [text for text, current_label in zip(texts, labels, strict=True) if current_label == label]
        keywords = _extract_keywords(member_texts)
        cluster_keywords[label] = ", ".join(keywords)
    return cluster_keywords


def _build_cluster_labels(
    representative_texts: dict[int, str],
    cluster_keywords: dict[int, str],
) -> dict[int, str]:
    labels: dict[int, str] = {}
    for cluster_id, representative_text in representative_texts.items():
        keywords = cluster_keywords.get(cluster_id, "")
        first_keyword = keywords.split(", ")[0] if keywords else ""
        if first_keyword and first_keyword not in representative_text:
            labels[cluster_id] = f"{first_keyword} | {representative_text}"
        else:
            labels[cluster_id] = representative_text or first_keyword
    return labels


def _extract_keywords(texts: list[str], top_k: int = 5) -> list[str]:
    token_counts: dict[str, int] = {}
    for text in texts:
        for token in set(_tokenize_for_keywords(text)):
            token_counts[token] = token_counts.get(token, 0) + 1

    min_count = max(1, min(2, len(texts)))
    ranked_tokens = sorted(
        (
            (token, count)
            for token, count in token_counts.items()
            if count >= min_count
        ),
        key=lambda item: (-item[1], -len(item[0]), item[0]),
    )
    return [token for token, _ in ranked_tokens[:top_k]]


def _tokenize_for_keywords(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.findall(text.casefold()):
        if match in STOPWORDS:
            continue
        if _is_cjk_token(match):
            if len(match) <= 2:
                tokens.append(match)
                continue
            for index in range(len(match) - 1):
                piece = match[index : index + 2]
                if piece not in STOPWORDS and not any(char in STOP_CHARS for char in piece):
                    tokens.append(piece)
            continue
        if len(match) >= 2:
            tokens.append(match)
    return tokens


def _is_cjk_token(token: str) -> bool:
    return all("\u4e00" <= char <= "\u9fff" for char in token)


def _project_vectors(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(vectors) == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32)
    if vectors.shape[1] == 1:
        return np.column_stack((vectors[:, 0], np.zeros(len(vectors), dtype=np.float32))).astype(np.float32)

    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ValueError("未安装 scikit-learn，请先执行 uv sync。") from exc

    return np.asarray(PCA(n_components=2, random_state=DEFAULT_CLUSTER_RANDOM_STATE).fit_transform(vectors), dtype=np.float32)
