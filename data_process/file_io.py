from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import pandas as pd

if TYPE_CHECKING:
    from data_process.semantic_deduplication import SemanticDeduplicationMatch


SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx", ".xlsm"}


def validate_tabular_file(file_path: str | Path) -> Path:
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "未检测到支持的输入文件，请提供 csv 或 Excel 文件（.csv/.xls/.xlsx/.xlsm）。"
        )
    return path


def load_dataframe(file_path: str | Path) -> pd.DataFrame:
    path = validate_tabular_file(file_path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, skip_blank_lines=False)
    return pd.read_excel(path)


def iter_dataframes(file_path: str | Path, chunksize: int = 50_000) -> Iterable[pd.DataFrame]:
    path = validate_tabular_file(file_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("仅 csv 文件支持分块流式处理。")
    return pd.read_csv(path, skip_blank_lines=False, chunksize=chunksize)


def count_csv_rows(file_path: str | Path) -> int:
    path = validate_tabular_file(file_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("仅 csv 文件支持分块流式处理。")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def write_dataframe(dataframe: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    if path.suffix.lower() == ".csv":
        dataframe.to_csv(path, index=False)
        return
    dataframe.to_excel(path, index=False)


def append_dataframe_chunk(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    *,
    wrote_header: bool,
) -> bool:
    path = Path(output_path)
    dataframe.to_csv(path, mode="a", index=False, header=not wrote_header)
    return True


def write_match_rows(
    match_rows: list["SemanticDeduplicationMatch"],
    output_path: str | Path,
    *,
    append: bool = False,
) -> None:
    if not match_rows:
        return

    path = Path(output_path)
    build_match_frame(match_rows).to_csv(
        path,
        mode="a" if append else "w",
        index=False,
        header=not append or not path.exists(),
    )


def build_match_frame(match_rows: list["SemanticDeduplicationMatch"]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_index": match.row_index,
                "duplicate_of_row_index": match.duplicate_of_row_index,
                "text": match.text,
                "matched_text": match.matched_text,
                "category": match.category,
                "matched_category": match.matched_category,
                "same_category": match.category == match.matched_category,
                "similarity": match.similarity,
            }
            for match in match_rows
        ]
    )
