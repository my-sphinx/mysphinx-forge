from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

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


def clean_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    stats = CleaningStats(total_before=len(dataframe), total_after=0)
    keep_mask: list[bool] = []

    for _, row in dataframe.iterrows():
        row_text = _row_to_text(row)
        if _is_blank_text(row_text):
            stats.removed_blank_rows += 1
            keep_mask.append(False)
            continue
        if _is_emoji_only_text(row_text):
            stats.removed_emoji_rows += 1
            keep_mask.append(False)
            continue
        if _is_garbled_only_text(row_text):
            stats.removed_garbled_rows += 1
            keep_mask.append(False)
            continue
        if _is_symbol_only_text(row_text):
            stats.removed_symbol_rows += 1
            keep_mask.append(False)
            continue
        keep_mask.append(True)

    cleaned = dataframe.loc[keep_mask].reset_index(drop=True)
    stats.total_after = len(cleaned)
    return cleaned, stats


def _row_to_text(row: pd.Series) -> str:
    parts: list[str] = []
    for value in row.tolist():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


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
