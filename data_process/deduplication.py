from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from data_process.cleaning import resolve_target_column


WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class DeduplicationStats:
    total_before: int
    total_after: int
    duplicate_rows: int = 0
    unique_values: int = 0
    target_column: str = ""
    dedupe_mode: str = "exact"
    semantic_threshold: float | None = None
    embedding_model_path: str | None = None


def deduplicate_dataframe(
    dataframe: pd.DataFrame,
    target_column: str = "text",
    seen_keys: set[str] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    report_every: int = 1_000,
) -> tuple[pd.DataFrame, DeduplicationStats]:
    resolved_target_column = resolve_target_column(dataframe, target_column)
    observed_keys = seen_keys if seen_keys is not None else set()

    stats = DeduplicationStats(
        total_before=len(dataframe),
        total_after=0,
        target_column=resolved_target_column,
    )
    keep_mask: list[bool] = []
    processed_since_report = 0

    for value in dataframe[resolved_target_column].tolist():
        processed_since_report += 1
        normalized_value = normalize_dedup_text(value)
        if normalized_value in observed_keys:
            stats.duplicate_rows += 1
            keep_mask.append(False)
        else:
            observed_keys.add(normalized_value)
            keep_mask.append(True)

        if progress_callback and processed_since_report >= report_every:
            progress_callback(processed_since_report)
            processed_since_report = 0

    if progress_callback and processed_since_report > 0:
        progress_callback(processed_since_report)

    deduplicated = dataframe.loc[keep_mask].reset_index(drop=True)
    stats.total_after = len(deduplicated)
    stats.unique_values = len(observed_keys)
    return deduplicated, stats


def normalize_dedup_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = WHITESPACE_RE.sub(" ", text)
    return text.casefold()
