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
