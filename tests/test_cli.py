from __future__ import annotations

import sys

import pandas as pd

from data_process.cli import main


def test_main_supports_action_flag(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame({"text": ["正常内容", "!!!"]}).to_csv(input_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--action", "clean", "--input-file", str(input_file)],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "清洗完成" in captured.out
    assert (tmp_path / "input_cleaned.csv").exists()
