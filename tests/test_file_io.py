from __future__ import annotations

import pandas as pd
import pytest

from data_process.file_io import (
    append_dataframe_chunk,
    build_match_frame,
    count_csv_rows,
    iter_dataframes,
    load_dataframe,
    write_dataframe,
    write_match_rows,
)
from data_process.semantic_deduplication import SemanticDeduplicationMatch


def test_load_dataframe_supports_csv_and_excel(tmp_path) -> None:
    csv_file = tmp_path / "input.csv"
    excel_file = tmp_path / "input.xlsx"
    dataframe = pd.DataFrame({"text": ["a", "b"]})
    dataframe.to_csv(csv_file, index=False)
    dataframe.to_excel(excel_file, index=False)

    assert load_dataframe(csv_file)["text"].tolist() == ["a", "b"]
    assert load_dataframe(excel_file)["text"].tolist() == ["a", "b"]


def test_iter_dataframes_rejects_excel(tmp_path) -> None:
    excel_file = tmp_path / "input.xlsx"
    pd.DataFrame({"text": ["a"]}).to_excel(excel_file, index=False)

    with pytest.raises(ValueError, match="仅 csv 文件支持分块流式处理。"):
        list(iter_dataframes(excel_file))


def test_count_csv_rows_counts_data_rows(tmp_path) -> None:
    csv_file = tmp_path / "input.csv"
    pd.DataFrame({"text": ["a", "b", "c"]}).to_csv(csv_file, index=False)

    assert count_csv_rows(csv_file) == 3


def test_write_dataframe_writes_csv_and_excel(tmp_path) -> None:
    dataframe = pd.DataFrame({"text": ["a", "b"]})
    csv_file = tmp_path / "output.csv"
    excel_file = tmp_path / "output.xlsx"

    write_dataframe(dataframe, csv_file)
    write_dataframe(dataframe, excel_file)

    assert pd.read_csv(csv_file)["text"].tolist() == ["a", "b"]
    assert pd.read_excel(excel_file)["text"].tolist() == ["a", "b"]


def test_append_dataframe_chunk_writes_header_once(tmp_path) -> None:
    output_file = tmp_path / "output.csv"

    wrote_header = append_dataframe_chunk(
        pd.DataFrame({"text": ["a"]}),
        output_file,
        wrote_header=False,
    )
    wrote_header = append_dataframe_chunk(
        pd.DataFrame({"text": ["b"]}),
        output_file,
        wrote_header=wrote_header,
    )

    assert wrote_header is True
    assert pd.read_csv(output_file)["text"].tolist() == ["a", "b"]


def test_write_match_rows_supports_append_mode(tmp_path) -> None:
    output_file = tmp_path / "matches.csv"
    first_rows = [
        SemanticDeduplicationMatch(
            row_index=1,
            duplicate_of_row_index=0,
            text="怎么申请退款",
            matched_text="退款怎么申请",
            category="售后",
            matched_category="售后",
            similarity=0.95,
        )
    ]
    second_rows = [
        SemanticDeduplicationMatch(
            row_index=3,
            duplicate_of_row_index=2,
            text="怎么开发票",
            matched_text="发票怎么开",
            category="发票",
            matched_category="财务",
            similarity=0.93,
        )
    ]

    write_match_rows(first_rows, output_file)
    write_match_rows(second_rows, output_file, append=True)

    written = pd.read_csv(output_file)
    assert written["row_index"].tolist() == [1, 3]
    assert written["duplicate_of_row_index"].tolist() == [0, 2]
    assert written["category"].tolist() == ["售后", "发票"]
    assert written["matched_category"].tolist() == ["售后", "财务"]
    assert written["same_category"].tolist() == [True, False]


def test_write_match_rows_uses_dynamic_category_column_names(tmp_path) -> None:
    output_file = tmp_path / "matches.csv"
    match_rows = [
        SemanticDeduplicationMatch(
            row_index=1,
            duplicate_of_row_index=0,
            text="怎么申请退款",
            matched_text="退款怎么申请",
            category="售后",
            matched_category="售后",
            similarity=0.95,
        )
    ]

    write_match_rows(match_rows, output_file, category_column="label")

    written = pd.read_csv(output_file)
    assert written.columns.tolist() == [
        "row_index",
        "duplicate_of_row_index",
        "text",
        "matched_text",
        "label",
        "matched_label",
        "same_label",
        "similarity",
    ]
    assert written["label"].tolist() == ["售后"]
    assert written["matched_label"].tolist() == ["售后"]
    assert written["same_label"].tolist() == [True]


def test_build_match_frame_omits_category_columns_when_all_categories_missing() -> None:
    match_rows = [
        SemanticDeduplicationMatch(
            row_index=1,
            duplicate_of_row_index=0,
            text="怎么申请退款",
            matched_text="退款怎么申请",
            category=None,
            matched_category=None,
            similarity=0.95,
        )
    ]

    written = build_match_frame(match_rows)

    assert written.columns.tolist() == [
        "row_index",
        "duplicate_of_row_index",
        "text",
        "matched_text",
        "similarity",
    ]


def test_build_match_frame_uses_dynamic_category_column_names() -> None:
    match_rows = [
        SemanticDeduplicationMatch(
            row_index=1,
            duplicate_of_row_index=0,
            text="怎么申请退款",
            matched_text="退款怎么申请",
            category="售后",
            matched_category="售后",
            similarity=0.95,
        )
    ]

    written = build_match_frame(match_rows, category_column="label")

    assert written.columns.tolist() == [
        "row_index",
        "duplicate_of_row_index",
        "text",
        "matched_text",
        "label",
        "matched_label",
        "same_label",
        "similarity",
    ]
