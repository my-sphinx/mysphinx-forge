from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from typing import Callable, Iterable

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx", ".xlsm"}
MOJIBAKE_CHARS = set("ÃÂÐÑØÞßæøåçðþŒœ€™¢£¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿")
REPLACEMENT_CHARS = {"�", "\ufffd"}
EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x1F1E6, 0x1F1FF),
)


@dataclass(slots=True)
class CleaningStats:
    total_before: int
    total_after: int
    removed_blank_rows: int = 0
    removed_symbol_rows: int = 0
    removed_emoji_rows: int = 0
    removed_garbled_rows: int = 0

    @property
    def total_removed(self) -> int:
        return self.total_before - self.total_after

    def merge(self, other: "CleaningStats") -> None:
        self.total_before += other.total_before
        self.total_after += other.total_after
        self.removed_blank_rows += other.removed_blank_rows
        self.removed_symbol_rows += other.removed_symbol_rows
        self.removed_emoji_rows += other.removed_emoji_rows
        self.removed_garbled_rows += other.removed_garbled_rows


def load_dataframe(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "未检测到支持的输入文件，请提供 csv 或 Excel 文件（.csv/.xls/.xlsx/.xlsm）。"
        )

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, skip_blank_lines=False)

    return pd.read_excel(path)


def iter_dataframes(file_path: str | Path, chunksize: int = 50_000) -> Iterable[pd.DataFrame]:
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "未检测到支持的输入文件，请提供 csv 或 Excel 文件（.csv/.xls/.xlsx/.xlsm）。"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError("仅 csv 文件支持分块流式处理。")

    return pd.read_csv(path, skip_blank_lines=False, chunksize=chunksize)


def count_csv_rows(file_path: str | Path) -> int:
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "未检测到支持的输入文件，请提供 csv 或 Excel 文件（.csv/.xls/.xlsx/.xlsm）。"
        )
    if path.suffix.lower() != ".csv":
        raise ValueError("仅 csv 文件支持分块流式处理。")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def clean_dataframe(
    dataframe: pd.DataFrame,
    target_column: str = "text",
    progress_callback: Callable[[int], None] | None = None,
    report_every: int = 1_000,
) -> tuple[pd.DataFrame, CleaningStats]:
    if target_column not in dataframe.columns:
        raise ValueError(f"未找到目标列：{target_column}")

    stats = CleaningStats(total_before=len(dataframe), total_after=0)
    keep_mask: list[bool] = []
    processed_since_report = 0

    for value in dataframe[target_column].tolist():
        processed_since_report += 1
        row_text = _cell_to_text(value)
        if _is_blank_text(row_text):
            stats.removed_blank_rows += 1
            keep_mask.append(False)
        elif _is_emoji_only_text(row_text):
            stats.removed_emoji_rows += 1
            keep_mask.append(False)
        elif _is_garbled_only_text(row_text):
            stats.removed_garbled_rows += 1
            keep_mask.append(False)
        elif _is_symbol_only_text(row_text):
            stats.removed_symbol_rows += 1
            keep_mask.append(False)
        else:
            keep_mask.append(True)

        if progress_callback and processed_since_report >= report_every:
            progress_callback(processed_since_report)
            processed_since_report = 0

    if progress_callback and processed_since_report > 0:
        progress_callback(processed_since_report)

    cleaned = dataframe.loc[keep_mask].reset_index(drop=True)
    stats.total_after = len(cleaned)
    return cleaned, stats


def _cell_to_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _is_blank_text(text: str) -> bool:
    return not text.strip()


def _is_symbol_only_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return all(
        char.isspace() or unicodedata.category(char)[0] in {"P", "S"} for char in stripped
    )


def _is_emoji_only_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return all(_is_emoji_char(char) for char in stripped if not char.isspace())


def _is_garbled_only_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if re.search(r"[A-Za-z0-9\u4e00-\u9fff]", stripped):
        return False

    non_space_chars = [char for char in stripped if not char.isspace()]
    if not non_space_chars:
        return False

    strong_garble_hits = sum(
        char in REPLACEMENT_CHARS or unicodedata.category(char).startswith("C")
        for char in non_space_chars
    )
    mojibake_hits = sum(char in MOJIBAKE_CHARS for char in non_space_chars)

    if strong_garble_hits == 0 and mojibake_hits == 0:
        return False

    garble_like_chars = sum(_is_garble_like_char(char) for char in non_space_chars)
    return garble_like_chars == len(non_space_chars)


def _is_garble_like_char(char: str) -> bool:
    if char in REPLACEMENT_CHARS or char in MOJIBAKE_CHARS:
        return True

    category = unicodedata.category(char)
    if category.startswith("C"):
        return True

    codepoint = ord(char)
    if 0xE000 <= codepoint <= 0xF8FF:
        return True

    return category[0] in {"P", "S"}


def _is_emoji_char(char: str) -> bool:
    if char in {"\u200d", "\ufe0f"}:
        return True

    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in EMOJI_RANGES)
