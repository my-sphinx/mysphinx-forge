import pandas as pd
import pytest

from data_process.deduplication import deduplicate_dataframe, normalize_dedup_text


def test_deduplicate_dataframe_normalizes_text_before_comparing() -> None:
    dataframe = pd.DataFrame(
        {
            "text": [
                " Hello   World ",
                "hello world",
                "HELLO\tWORLD",
                "Another Value",
            ]
        }
    )

    deduplicated, stats = deduplicate_dataframe(dataframe)

    assert deduplicated["text"].tolist() == [" Hello   World ", "Another Value"]
    assert stats.target_column == "text"
    assert stats.total_before == 4
    assert stats.total_after == 2
    assert stats.duplicate_rows == 2
    assert stats.unique_values == 2


def test_deduplicate_dataframe_uses_target_column_only() -> None:
    dataframe = pd.DataFrame(
        {
            "text": ["A", "A"],
            "客户问题": ["同一个 问题", "同一个\t问题"],
        }
    )

    deduplicated, stats = deduplicate_dataframe(dataframe, target_column="客户问题")

    assert deduplicated["text"].tolist() == ["A"]
    assert deduplicated["客户问题"].tolist() == ["同一个 问题"]
    assert stats.target_column == "客户问题"
    assert stats.duplicate_rows == 1


def test_deduplicate_dataframe_auto_detects_user_input_column() -> None:
    dataframe = pd.DataFrame(
        {
            "用户输入": [" Foo ", "foo", "Bar"],
            "其他列": [1, 2, 3],
        }
    )

    deduplicated, stats = deduplicate_dataframe(dataframe)

    assert deduplicated["用户输入"].tolist() == [" Foo ", "Bar"]
    assert deduplicated["其他列"].tolist() == [1, 3]
    assert stats.target_column == "用户输入"
    assert stats.unique_values == 2


def test_deduplicate_dataframe_reports_progress() -> None:
    dataframe = pd.DataFrame({"text": ["A", "a", "B"]})
    reported: list[int] = []

    deduplicate_dataframe(dataframe, progress_callback=reported.append, report_every=2)

    assert reported == [2, 1]


def test_deduplicate_dataframe_rejects_missing_target_column() -> None:
    dataframe = pd.DataFrame({"text": ["value"]})

    with pytest.raises(ValueError, match="未找到目标列：客户问题"):
        deduplicate_dataframe(dataframe, target_column="客户问题")


def test_normalize_dedup_text_handles_blank_like_values() -> None:
    assert normalize_dedup_text(None) == ""
    assert normalize_dedup_text(" \tHello \n World ") == "hello world"
