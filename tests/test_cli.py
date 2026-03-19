from __future__ import annotations

import json
import sys

import pandas as pd

from data_process import cli
from data_process.cli import main
from data_process.deduplication import DeduplicationStats
from data_process.semantic_deduplication import SemanticDeduplicationMatch


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
    log_file = tmp_path / "data-process.log"
    assert log_file.exists()
    log_text = log_file.read_text(encoding="utf-8")
    assert "开始执行 action=clean" in log_text
    assert "开始阶段：统计总行数" in log_text
    assert "完成阶段：分块清洗" in log_text
    assert "清洗完成，输出文件" in log_text
    meta_file = tmp_path / "input_cleaned.meta.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["action"] == "clean"
    assert meta["input_file"] == str(input_file)
    assert meta["output_file"] == str(tmp_path / "input_cleaned.csv")
    assert meta["cleaning_stats"]["total_before"] == 2


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


def test_main_supports_deduplicate_action(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame({"text": [" Hello  World ", "hello world", "Another"]}).to_csv(
        input_file, index=False
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--action", "deduplicate", "--input-file", str(input_file)],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "去重完成" in captured.out
    assert "分块去重" in captured.err
    assert "写出结果" in captured.err
    assert "总数" in captured.err
    assert "重复" in captured.err
    assert "保留" in captured.err
    assert "唯一值" in captured.err

    output_file = tmp_path / "input_deduplicated.csv"
    deduplicated = pd.read_csv(output_file)
    assert deduplicated["text"].tolist() == [" Hello  World ", "Another"]
    log_file = tmp_path / "data-process.log"
    assert log_file.exists()
    log_text = log_file.read_text(encoding="utf-8")
    assert "开始执行 action=deduplicate" in log_text
    assert "开始阶段：统计总行数" in log_text
    assert "完成阶段：分块去重" in log_text
    assert "去重完成，输出文件" in log_text


def test_main_deduplicates_with_target_column(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "text": ["row1", "row2"],
            "客户问题": ["同一个 问题", "同一个\t问题"],
        }
    ).to_csv(input_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "deduplicate",
            "--input-file",
            str(input_file),
            "--target-column",
            "客户问题",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()
    output_file = tmp_path / "input_deduplicated.csv"

    assert exit_code == 0
    assert "使用目标列：客户问题" in captured.out
    deduplicated = pd.read_csv(output_file)
    assert deduplicated["text"].tolist() == ["row1"]


def test_main_auto_detects_user_input_for_deduplicate(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame({"用户输入": [" Foo ", "foo", "Bar"]}).to_csv(input_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--action", "deduplicate", "--input-file", str(input_file)],
    )

    exit_code = main()
    captured = capsys.readouterr()
    output_file = tmp_path / "input_deduplicated.csv"

    assert exit_code == 0
    assert "使用目标列：用户输入" in captured.out
    deduplicated = pd.read_csv(output_file)
    assert deduplicated["用户输入"].tolist() == [" Foo ", "Bar"]


def test_main_streams_csv_deduplication(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame({"text": ["A", "a", "B"]}).to_csv(input_file, index=False)

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
        ["main.py", "--action", "deduplicate", "--input-file", str(input_file)],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "去重完成" in captured.out
    assert calls["iterated"] is True
    assert calls["loaded"] is False
    assert "统计总行数" in captured.err
    assert "分块去重" in captured.err


def test_main_rejects_missing_target_column_for_deduplicate(
    tmp_path, monkeypatch, capsys
) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame({"text": ["正常内容"]}).to_csv(input_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "deduplicate",
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


def test_main_supports_semantic_deduplicate_action(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "text": ["退款怎么申请", "怎么申请退款", "发票怎么开"],
            "category": ["售后", "售后-重复", "财务"],
        }
    ).to_csv(
        input_file, index=False
    )

    class FakeSemanticDeduplicator:
        def __init__(
            self,
            model_path: str,
            threshold: float,
            batch_size: int,
            index_type: str = "flat",
            hnsw_m: int = 32,
        ) -> None:
            self.model_path = model_path
            self.threshold = threshold
            self.batch_size = batch_size
            self.index_type = index_type
            self.hnsw_m = hnsw_m

    def fake_semantic_deduplicate_dataframe(
        dataframe,
        target_column="text",
        threshold=0.9,
        model_path="models/m3e-base",
        batch_size=64,
        progress_callback=None,
        report_every=1_000,
        row_index_offset=0,
        collect_matches=False,
        deduplicator=None,
    ):
        if progress_callback:
            progress_callback(len(dataframe))
        stats = DeduplicationStats(
            total_before=3,
            total_after=2,
            duplicate_rows=1,
            unique_values=2,
            target_column=target_column,
            dedupe_mode="semantic",
            semantic_threshold=deduplicator.threshold,
            embedding_model_path=deduplicator.model_path,
        )
        matches = [
            SemanticDeduplicationMatch(
                row_index=row_index_offset + 1,
                duplicate_of_row_index=row_index_offset,
                text="怎么申请退款",
                matched_text="退款怎么申请",
                category="售后-重复",
                matched_category="售后",
                similarity=0.96,
            )
        ]
        return dataframe.iloc[[0, 2]].reset_index(drop=True), stats, matches

    monkeypatch.setattr(cli, "SemanticDeduplicator", FakeSemanticDeduplicator)
    monkeypatch.setattr(cli, "semantic_deduplicate_dataframe", fake_semantic_deduplicate_dataframe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "deduplicate",
            "--input-file",
            str(input_file),
            "--dedupe-mode",
            "semantic",
            "--semantic-threshold",
            "0.95",
            "--embedding-model-path",
            "models/m3e-base",
            "--batch-size",
            "16",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()
    output_file = tmp_path / "input_deduplicated.csv"
    match_file = tmp_path / "input_deduplicated_matches.csv"

    assert exit_code == 0
    assert "去重模式：semantic" in captured.out
    assert "语义阈值：0.95" in captured.out
    assert "语义模型路径：models/m3e-base" in captured.out
    assert output_file.exists()
    assert match_file.exists()
    deduplicated = pd.read_csv(output_file)
    assert deduplicated["text"].tolist() == ["退款怎么申请", "发票怎么开"]
    match_rows = pd.read_csv(match_file)
    assert match_rows["duplicate_of_row_index"].tolist() == [0]
    assert match_rows["category"].tolist() == ["售后-重复"]
    assert match_rows["matched_category"].tolist() == ["售后"]
    assert match_rows["same_category"].tolist() == [False]
    meta = json.loads((tmp_path / "input_deduplicated.meta.json").read_text(encoding="utf-8"))
    assert meta["action"] == "deduplicate"
    assert meta["deduplication_stats"]["dedupe_mode"] == "semantic"
    assert meta["match_file"] == str(match_file)
    assert meta["parameters"]["semantic_index_type"] == "flat"
    assert meta["parameters"]["semantic_hnsw_m"] == 32


def test_main_supports_clean_deduplicate_action(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "text": ["!!!", "退款怎么申请", "怎么申请退款"],
            "category": ["噪声", "售后", "售后-重复"],
        }
    ).to_csv(
        input_file, index=False
    )

    class FakeSemanticDeduplicator:
        def __init__(
            self,
            model_path: str,
            threshold: float,
            batch_size: int,
            index_type: str = "flat",
            hnsw_m: int = 32,
        ) -> None:
            self.model_path = model_path
            self.threshold = threshold
            self.batch_size = batch_size
            self.index_type = index_type
            self.hnsw_m = hnsw_m

    def fake_semantic_deduplicate_dataframe(
        dataframe,
        target_column="text",
        threshold=0.9,
        model_path="models/m3e-base",
        batch_size=64,
        progress_callback=None,
        report_every=1_000,
        row_index_offset=0,
        collect_matches=False,
        deduplicator=None,
    ):
        if progress_callback:
            progress_callback(len(dataframe))
        stats = DeduplicationStats(
            total_before=2,
            total_after=1,
            duplicate_rows=1,
            unique_values=1,
            target_column=target_column,
            dedupe_mode="semantic",
            semantic_threshold=deduplicator.threshold,
            embedding_model_path=deduplicator.model_path,
        )
        matches = [
            SemanticDeduplicationMatch(
                row_index=row_index_offset + 1,
                duplicate_of_row_index=row_index_offset,
                text="怎么申请退款",
                matched_text="退款怎么申请",
                category="售后-重复",
                matched_category="售后",
                similarity=0.96,
            )
        ]
        return dataframe.iloc[[0]].reset_index(drop=True), stats, matches

    monkeypatch.setattr(cli, "SemanticDeduplicator", FakeSemanticDeduplicator)
    monkeypatch.setattr(cli, "semantic_deduplicate_dataframe", fake_semantic_deduplicate_dataframe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "clean-deduplicate",
            "--input-file",
            str(input_file),
            "--dedupe-mode",
            "semantic",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()
    output_file = tmp_path / "input_deduplicated.csv"
    match_file = tmp_path / "input_deduplicated_matches.csv"

    assert exit_code == 0
    assert "清洗完成" in captured.out
    assert "去重完成" in captured.out
    assert "分块清洗" in captured.err
    assert "分块去重" in captured.err
    assert output_file.exists()
    assert match_file.exists()
    deduplicated = pd.read_csv(output_file)
    assert deduplicated["text"].tolist() == ["退款怎么申请"]
    match_rows = pd.read_csv(match_file)
    assert match_rows["category"].tolist() == ["售后-重复"]
    assert match_rows["matched_category"].tolist() == ["售后"]
    assert match_rows["same_category"].tolist() == [False]
    log_text = (tmp_path / "data-process.log").read_text(encoding="utf-8")
    assert "开始执行 action=clean-deduplicate" in log_text
    meta = json.loads((tmp_path / "input_deduplicated.meta.json").read_text(encoding="utf-8"))
    assert meta["action"] == "clean-deduplicate"
    assert meta["cleaning_stats"]["removed_symbol_rows"] == 1
    assert meta["deduplication_stats"]["dedupe_mode"] == "semantic"
    assert meta["match_file"] == str(match_file)
    assert meta["parameters"]["semantic_index_type"] == "flat"
    assert meta["parameters"]["semantic_hnsw_m"] == 32


def test_main_rejects_invalid_semantic_threshold(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "deduplicate",
            "--input-file",
            "input.csv",
            "--dedupe-mode",
            "semantic",
            "--semantic-threshold",
            "1.5",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--semantic-threshold 必须在 0 到 1 之间。" in captured.out


def test_main_rejects_invalid_semantic_hnsw_m(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "deduplicate",
            "--input-file",
            "input.csv",
            "--dedupe-mode",
            "semantic",
            "--semantic-index-type",
            "hnsw",
            "--semantic-hnsw-m",
            "0",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--semantic-hnsw-m 必须是大于 0 的整数。" in captured.out


def test_main_reports_unexpected_semantic_deduplicate_error(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.xlsx"
    pd.DataFrame({"text": ["退款怎么申请"]}).to_excel(input_file, index=False)

    def boom(*_args, **_kwargs):
        raise RuntimeError("model backend crashed")

    monkeypatch.setattr(cli, "semantic_deduplicate_dataframe", boom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "deduplicate",
            "--input-file",
            str(input_file),
            "--dedupe-mode",
            "semantic",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()
    log_text = (tmp_path / "data-process.log").read_text(encoding="utf-8")

    assert exit_code == 1
    assert "执行去重失败：RuntimeError: model backend crashed" in captured.out
    assert "执行去重失败" in log_text


def test_main_reports_unexpected_clean_deduplicate_error(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.xlsx"
    pd.DataFrame({"text": ["退款怎么申请"]}).to_excel(input_file, index=False)

    def boom(*_args, **_kwargs):
        raise RuntimeError("model backend crashed")

    monkeypatch.setattr(cli, "semantic_deduplicate_dataframe", boom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "clean-deduplicate",
            "--input-file",
            str(input_file),
            "--dedupe-mode",
            "semantic",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()
    log_text = (tmp_path / "data-process.log").read_text(encoding="utf-8")

    assert exit_code == 1
    assert "执行清洗去重失败：RuntimeError: model backend crashed" in captured.out
    assert "执行清洗去重失败" in log_text
