from __future__ import annotations

import sys

import pandas as pd

from data_process import cli
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
    assert "统计总行数" in captured.err
    assert "分块清洗" in captured.err
    assert "写出结果" in captured.err
    assert "总数" in captured.err
    assert "删除" in captured.err
    assert "保留" in captured.err
    assert "空行" in captured.err
    assert "符号" in captured.err
    assert "表情" in captured.err
    assert "乱码" in captured.err
    assert (tmp_path / "input_cleaned.csv").exists()


def test_main_supports_target_column(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "text": ["!!!", "正常内容"],
            "客户问题": ["正常问题", "???"],
        }
    ).to_csv(input_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "clean",
            "--input-file",
            str(input_file),
            "--target-column",
            "客户问题",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()
    output_file = tmp_path / "input_cleaned.csv"

    assert exit_code == 0
    assert "清洗完成" in captured.out
    cleaned = pd.read_csv(output_file)
    assert cleaned["text"].tolist() == ["!!!"]
    assert cleaned["客户问题"].tolist() == ["正常问题"]


def test_main_auto_detects_user_input_column(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "用户输入": ["正常内容", "!!!"],
            "其他列": ["x", "y"],
        }
    ).to_csv(input_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--action", "clean", "--input-file", str(input_file)],
    )

    exit_code = main()
    captured = capsys.readouterr()
    output_file = tmp_path / "input_cleaned.csv"

    assert exit_code == 0
    assert "清洗完成" in captured.out
    cleaned = pd.read_csv(output_file)
    assert cleaned["用户输入"].tolist() == ["正常内容"]
    assert cleaned["其他列"].tolist() == ["x"]


def test_main_streams_csv_cleaning(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame({"text": ["正常内容", "!!!", "abc123"]}).to_csv(input_file, index=False)

    calls = {"iterated": False, "loaded": False}
    original_iter = cli.iter_dataframes

    def wrapped_iter_dataframes(file_path, chunksize=50_000):
        calls["iterated"] = True
        return original_iter(file_path, chunksize)

    def fail_load_dataframe(_file_path):
        calls["loaded"] = True
        raise AssertionError("csv should not use load_dataframe")

    monkeypatch.setattr(cli, "iter_dataframes", wrapped_iter_dataframes)
    monkeypatch.setattr(cli, "load_dataframe", fail_load_dataframe)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--action", "clean", "--input-file", str(input_file)],
    )

    exit_code = main()
    captured = capsys.readouterr()
    output_file = tmp_path / "input_cleaned.csv"

    assert exit_code == 0
    assert "清洗完成" in captured.out
    assert "统计总行数" in captured.err
    assert "分块清洗" in captured.err
    assert "写出结果" in captured.err
    assert "总数" in captured.err
    assert "删除" in captured.err
    assert "保留" in captured.err
    assert "空行" in captured.err
    assert "符号" in captured.err
    assert "表情" in captured.err
    assert "乱码" in captured.err
    assert calls["iterated"] is True
    assert calls["loaded"] is False
    assert pd.read_csv(output_file)["text"].tolist() == ["正常内容", "abc123"]


def test_main_passes_custom_chunk_size(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame({"text": ["正常内容", "!!!", "abc123"]}).to_csv(input_file, index=False)

    calls = {"chunk_size": None}
    original_iter = cli.iter_dataframes

    def wrapped_iter_dataframes(file_path, chunksize=50_000):
        calls["chunk_size"] = chunksize
        return original_iter(file_path, chunksize)

    monkeypatch.setattr(cli, "iter_dataframes", wrapped_iter_dataframes)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "clean",
            "--input-file",
            str(input_file),
            "--chunk-size",
            "2",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "清洗完成" in captured.out
    assert calls["chunk_size"] == 2


def test_main_rejects_invalid_chunk_size(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "clean",
            "--input-file",
            "input.csv",
            "--chunk-size",
            "0",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--chunk-size 必须是大于 0 的整数。" in captured.out


def test_main_rejects_missing_target_column(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame({"text": ["正常内容"]}).to_csv(input_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "clean",
            "--input-file",
            str(input_file),
            "--target-column",
            "客户问题",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "未找到目标列：客户问题" in captured.out


def test_main_shows_multistage_progress_for_excel(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.xlsx"
    pd.DataFrame({"text": ["正常内容", "!!!"]}).to_excel(input_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--action", "clean", "--input-file", str(input_file)],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "读取文件" in captured.err
    assert "清洗数据" in captured.err
    assert "写出结果" in captured.err
    assert "总数" in captured.err
    assert "删除" in captured.err
    assert "保留" in captured.err
    assert "空行" in captured.err
    assert "符号" in captured.err
    assert "表情" in captured.err
    assert "乱码" in captured.err
    assert (tmp_path / "input_cleaned.xlsx").exists()
