import pandas as pd
import pytest

from data_process.cleaning import clean_dataframe, load_dataframe


def test_clean_dataframe_removes_invalid_rows() -> None:
    dataframe = pd.DataFrame(
        {
            "text": [
                None,
                "!!!",
                "😂🤣",
                "���",
                "正常内容",
                "abc123",
            ]
        }
    )

    cleaned, stats = clean_dataframe(dataframe)

    assert cleaned["text"].tolist() == ["正常内容", "abc123"]
    assert stats.total_before == 6
    assert stats.total_after == 2
    assert stats.removed_blank_rows == 1
    assert stats.removed_symbol_rows == 1
    assert stats.removed_emoji_rows == 1
    assert stats.removed_garbled_rows == 1


def test_load_dataframe_rejects_unsupported_file() -> None:
    with pytest.raises(ValueError):
        load_dataframe("input.txt")


def test_clean_dataframe_reports_progress() -> None:
    dataframe = pd.DataFrame({"text": ["正常内容", "!!!", "abc123"]})
    reported: list[int] = []

    clean_dataframe(dataframe, progress_callback=reported.append, report_every=2)

    assert reported == [2, 1]


def test_clean_dataframe_uses_target_column_only() -> None:
    dataframe = pd.DataFrame(
        {
            "text": ["正常内容", "!!!"],
            "客户问题": ["???", "正常问题"],
        }
    )

    cleaned, stats = clean_dataframe(dataframe, target_column="客户问题")

    assert cleaned["text"].tolist() == ["!!!"]
    assert cleaned["客户问题"].tolist() == ["正常问题"]
    assert stats.total_before == 2
    assert stats.total_after == 1
    assert stats.removed_symbol_rows == 1


def test_clean_dataframe_rejects_missing_target_column() -> None:
    dataframe = pd.DataFrame({"text": ["正常内容"]})

    with pytest.raises(ValueError, match="未找到目标列：客户问题"):
        clean_dataframe(dataframe, target_column="客户问题")
