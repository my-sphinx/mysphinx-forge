from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas as pd

from mysphinx_forge import cli
from mysphinx_forge.cli import main
from mysphinx_forge.clustering import ClusteringStats
from mysphinx_forge.deduplication import DeduplicationStats
from mysphinx_forge.model_testing import BatchModelTestStats
from mysphinx_forge.semantic_deduplication import SemanticDeduplicationMatch


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
    log_file = tmp_path / "mysphinx-forge.log"
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
    log_file = tmp_path / "mysphinx-forge.log"
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
        category_column="category",
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


def test_main_supports_cluster_action(tmp_path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "text": ["退款怎么申请", "怎么申请退款", "我要开发票", ""],
            "category": ["售后", "售后-重复", "财务", "空白"],
        }
    ).to_csv(input_file, index=False)

    class FakeTextClusterer:
        def __init__(
            self,
            model_path: str,
            cluster_mode: str,
            batch_size: int,
            min_cluster_size: int,
            num_clusters: int,
            cluster_selection_epsilon: float,
            cluster_label_mode: str,
            cluster_label_model: str,
            cluster_label_api_base: str | None,
            cluster_label_sample_size: int,
        ) -> None:
            self.model_path = model_path
            self.cluster_mode = cluster_mode
            self.batch_size = batch_size
            self.min_cluster_size = min_cluster_size
            self.num_clusters = num_clusters
            self.cluster_selection_epsilon = cluster_selection_epsilon
            self.cluster_label_mode = cluster_label_mode
            self.cluster_label_model = cluster_label_model
            self.cluster_label_api_base = cluster_label_api_base
            self.cluster_label_sample_size = cluster_label_sample_size

        def cluster_dataframe(self, dataframe, target_column="text", progress_callback=None):
            if progress_callback:
                progress_callback(len(dataframe))
            clustered = dataframe.copy()
            clustered["cluster_id"] = [0, 0, 1, -1]
            clustered["is_noise"] = [False, False, False, True]
            clustered["cluster_size"] = [2, 2, 1, 1]
            clustered["cluster_representative_text"] = [
                "退款怎么申请",
                "退款怎么申请",
                "我要开发票",
                "",
            ]
            summary = pd.DataFrame(
                {
                    "cluster_id": [0, 1],
                    "cluster_size": [2, 1],
                    "cluster_label": ["退款怎么申请", "我要开发票"],
                    "top_keywords": ["申请, 退款", "发票, 开发"],
                    "representative_text": ["退款怎么申请", "我要开发票"],
                    "example_texts": ["退款怎么申请 | 怎么申请退款", "我要开发票"],
                }
            )
            projection = pd.DataFrame(
                {
                    "row_index": [0, 1, 2, 3],
                    target_column: ["退款怎么申请", "怎么申请退款", "我要开发票", ""],
                    "cluster_id": [0, 0, 1, -1],
                    "is_noise": [False, False, False, True],
                    "x": [0.1, 0.2, 0.9, float("nan")],
                    "y": [0.3, 0.4, 0.8, float("nan")],
                    "z": [0.5, 0.6, 0.7, float("nan")],
                }
            )
            stats = ClusteringStats(
                total_before=4,
                total_clustered=3,
                cluster_count=2,
                noise_rows=1,
                largest_cluster_size=2,
                smallest_cluster_size=1,
                average_cluster_size=1.5,
                target_column=target_column,
                cluster_mode=self.cluster_mode,
                embedding_model_path=self.model_path,
                cluster_label_mode=self.cluster_label_mode,
                cluster_label_model=self.cluster_label_model,
            )
            return clustered, summary, projection, stats

    monkeypatch.setattr(cli, "TextClusterer", FakeTextClusterer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "cluster",
            "--input-file",
            str(input_file),
            "--cluster-mode",
            "hdbscan",
            "--min-cluster-size",
            "2",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()
    output_file = tmp_path / "input_clustered.csv"
    cluster_summary_file = tmp_path / "input_clustered_clusters.csv"
    projection_file = tmp_path / "input_clustered_projection.csv"
    analysis_file = tmp_path / "input_clustered_analysis.csv"
    html_report_file = tmp_path / "input_clustered_report.html"
    meta_file = tmp_path / "input_clustered.meta.json"

    assert exit_code == 0
    assert "聚类完成" in captured.out
    assert "聚类模式：hdbscan" in captured.out
    assert "标签模式：rule" in captured.out
    assert "聚类簇数量：2" in captured.out
    assert "执行聚类" in captured.err
    assert output_file.exists()
    assert cluster_summary_file.exists()
    assert projection_file.exists()
    assert analysis_file.exists()
    assert html_report_file.exists()
    clustered = pd.read_csv(output_file)
    assert clustered["cluster_id"].tolist() == [0, 0, 1, -1]
    cluster_summary = pd.read_csv(cluster_summary_file)
    assert cluster_summary["cluster_id"].tolist() == [0, 1]
    projection = pd.read_csv(projection_file)
    assert projection["cluster_id"].tolist() == [0, 0, 1, -1]
    analysis = pd.read_csv(analysis_file)
    assert analysis["cluster_rank"].tolist() == [1, 2]
    assert "Cluster Report" in html_report_file.read_text(encoding="utf-8")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["action"] == "cluster"
    assert meta["clustering_stats"]["cluster_mode"] == "hdbscan"
    assert meta["clustering_stats"]["cluster_label_mode"] == "rule"
    assert meta["parameters"]["cluster_label_mode"] == "rule"
    assert meta["cluster_summary_file"] == str(cluster_summary_file)
    assert meta["projection_file"] == str(projection_file)
    assert meta["analysis_file"] == str(analysis_file)
    assert meta["html_report_file"] == str(html_report_file)


def test_main_rejects_invalid_cluster_label_sample_size(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "cluster",
            "--input-file",
            "input.csv",
            "--cluster-label-sample-size",
            "0",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--cluster-label-sample-size 必须是大于 0 的整数。" in captured.out


def test_main_supports_custom_category_column_for_semantic_deduplicate(
    tmp_path, monkeypatch, capsys
) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "text": ["退款怎么申请", "怎么申请退款", "发票怎么开"],
            "label": ["售后", "售后-重复", "财务"],
        }
    ).to_csv(input_file, index=False)

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
        category_column="category",
        threshold=0.9,
        model_path="models/m3e-base",
        batch_size=64,
        progress_callback=None,
        report_every=1_000,
        row_index_offset=0,
        collect_matches=False,
        deduplicator=None,
    ):
        assert category_column == "label"
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
            "--category-column",
            "label",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()
    match_file = tmp_path / "input_deduplicated_matches.csv"
    meta = json.loads((tmp_path / "input_deduplicated.meta.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "去重完成" in captured.out
    assert match_file.exists()
    match_rows = pd.read_csv(match_file)
    assert match_rows.columns.tolist() == [
        "row_index",
        "duplicate_of_row_index",
        "text",
        "matched_text",
        "label",
        "matched_label",
        "same_label",
        "similarity",
    ]
    assert match_rows["label"].tolist() == ["售后-重复"]
    assert match_rows["matched_label"].tolist() == ["售后"]
    assert match_rows["same_label"].tolist() == [False]
    assert meta["parameters"]["category_column"] == "label"


def test_main_omits_category_columns_when_semantic_input_has_no_category_column(
    tmp_path, monkeypatch, capsys
) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "text": ["退款怎么申请", "怎么申请退款"],
        }
    ).to_csv(input_file, index=False)

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
        category_column="category",
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
                category=None,
                matched_category=None,
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
            "deduplicate",
            "--input-file",
            str(input_file),
            "--dedupe-mode",
            "semantic",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()
    match_rows = pd.read_csv(tmp_path / "input_deduplicated_matches.csv")

    assert exit_code == 0
    assert "去重完成" in captured.out
    assert match_rows.columns.tolist() == [
        "row_index",
        "duplicate_of_row_index",
        "text",
        "matched_text",
        "similarity",
    ]


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
        category_column="category",
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
    log_text = (tmp_path / "mysphinx-forge.log").read_text(encoding="utf-8")
    assert "开始执行 action=clean-deduplicate" in log_text
    meta = json.loads((tmp_path / "input_deduplicated.meta.json").read_text(encoding="utf-8"))
    assert meta["action"] == "clean-deduplicate"
    assert meta["cleaning_stats"]["removed_symbol_rows"] == 1
    assert meta["deduplication_stats"]["dedupe_mode"] == "semantic"
    assert meta["match_file"] == str(match_file)
    assert meta["parameters"]["semantic_index_type"] == "flat"
    assert meta["parameters"]["semantic_hnsw_m"] == 32


def test_main_supports_custom_category_column_for_clean_deduplicate(
    tmp_path, monkeypatch, capsys
) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "text": ["!!!", "退款怎么申请", "怎么申请退款"],
            "label": ["噪声", "售后", "售后-重复"],
        }
    ).to_csv(input_file, index=False)

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
        category_column="category",
        threshold=0.9,
        model_path="models/m3e-base",
        batch_size=64,
        progress_callback=None,
        report_every=1_000,
        row_index_offset=0,
        collect_matches=False,
        deduplicator=None,
    ):
        assert category_column == "label"
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
            "--category-column",
            "label",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()
    match_file = tmp_path / "input_deduplicated_matches.csv"
    meta = json.loads((tmp_path / "input_deduplicated.meta.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "清洗完成" in captured.out
    assert match_file.exists()
    match_rows = pd.read_csv(match_file)
    assert match_rows.columns.tolist() == [
        "row_index",
        "duplicate_of_row_index",
        "text",
        "matched_text",
        "label",
        "matched_label",
        "same_label",
        "similarity",
    ]
    assert match_rows["matched_label"].tolist() == ["售后"]
    assert match_rows["same_label"].tolist() == [False]
    assert meta["parameters"]["category_column"] == "label"


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
    log_text = (tmp_path / "mysphinx-forge.log").read_text(encoding="utf-8")

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
    log_text = (tmp_path / "mysphinx-forge.log").read_text(encoding="utf-8")

    assert exit_code == 1
    assert "执行清洗去重失败：RuntimeError: model backend crashed" in captured.out
    assert "执行清洗去重失败" in log_text


def test_main_supports_model_test_action(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    calls = {}

    def fake_run_model_test(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            model_path=kwargs["model_path"],
            user_input="请问退款怎么申请？",
            model_class="AutoModelForCausalLM",
            tokenizer_class="AutoTokenizer",
            device="cuda",
            generated_text="您可以在订单详情页提交退款申请。",
        )

    monkeypatch.setattr(cli, "run_model_test", fake_run_model_test)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "model-test",
            "--test-model-path",
            "models/custom-model",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls["model_path"] == "models/custom-model"
    assert calls["max_new_tokens"] == 64
    assert calls["do_sample"] is False
    assert calls["temperature"] == 1.0
    assert calls["top_p"] == 1.0
    assert calls["top_k"] == 0
    assert calls["repetition_penalty"] == 1.05
    assert "模型测试完成" in captured.out
    assert "模型路径：models/custom-model" in captured.out
    assert "测试输入：请问退款怎么申请？" in captured.out
    assert "模型类型：AutoModelForCausalLM" in captured.out
    assert "Tokenizer 类型：AutoTokenizer" in captured.out
    assert "推理设备：cuda" in captured.out
    assert "生成参数：max_new_tokens=64, do_sample=False, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.05" in captured.out
    assert "模型输出：您可以在订单详情页提交退款申请。" in captured.out
    log_text = (tmp_path / "mysphinx-forge.log").read_text(encoding="utf-8")
    assert "开始执行 action=model-test model=models/custom-model" in log_text


def test_main_rejects_missing_model_path_for_model_test(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--action", "model-test"],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--test-model-path 为必填参数，--model-path 可作为别名，且仅用于 model-test。" in captured.out


def test_main_supports_model_path_alias_for_model_test(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    calls = {}

    def fake_run_model_test(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            model_path=kwargs["model_path"],
            user_input="请问退款怎么申请？",
            model_class="AutoModelForCausalLM",
            tokenizer_class="AutoTokenizer",
            device="cuda",
            generated_text="您可以在订单详情页提交退款申请。",
        )

    monkeypatch.setattr(cli, "run_model_test", fake_run_model_test)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "model-test",
            "--model-path",
            "models/alias-model",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls["model_path"] == "models/alias-model"
    assert calls["max_new_tokens"] == 64
    assert "模型路径：models/alias-model" in captured.out


def test_main_passes_custom_generation_parameters_for_model_test(
    monkeypatch, capsys, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)

    calls = {}

    def fake_run_model_test(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            model_path=kwargs["model_path"],
            user_input="请问退款怎么申请？",
            model_class="AutoModelForCausalLM",
            tokenizer_class="AutoTokenizer",
            device="cuda",
            generated_text="您可以在订单详情页提交退款申请。",
        )

    monkeypatch.setattr(cli, "run_model_test", fake_run_model_test)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "model-test",
            "--test-model-path",
            "models/custom-model",
            "--max-new-tokens",
            "128",
            "--temperature",
            "0.8",
            "--top-p",
            "0.95",
            "--top-k",
            "40",
            "--repetition-penalty",
            "1.1",
            "--no-do-sample",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls["model_path"] == "models/custom-model"
    assert calls["max_new_tokens"] == 128
    assert calls["do_sample"] is False
    assert calls["temperature"] == 0.8
    assert calls["top_p"] == 0.95
    assert calls["top_k"] == 40
    assert calls["repetition_penalty"] == 1.1
    assert "生成参数：max_new_tokens=128, do_sample=False, temperature=0.8, top_p=0.95, top_k=40, repetition_penalty=1.1" in captured.out


def test_main_rejects_invalid_generation_parameters_for_model_test(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "model-test",
            "--test-model-path",
            "models/custom-model",
            "--top-p",
            "1.2",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--top-p 必须在 0 到 1 之间。" in captured.out


def test_main_reports_model_test_error(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    def boom(**_kwargs):
        raise RuntimeError("model backend crashed")

    monkeypatch.setattr(cli, "run_model_test", boom)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--action", "model-test", "--test-model-path", "models/custom-model"],
    )

    exit_code = main()
    captured = capsys.readouterr()
    log_text = (tmp_path / "mysphinx-forge.log").read_text(encoding="utf-8")

    assert exit_code == 1
    assert "执行模型测试失败：RuntimeError: model backend crashed" in captured.out
    assert "执行模型测试失败" in log_text


def test_main_supports_file_based_model_test_with_expected_result(
    tmp_path, monkeypatch, capsys
) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "用户输入": ["退款怎么申请", "发票怎么开"],
            "预期结果": ["请在订单页提交退款申请", "请联系财务开票"],
        }
    ).to_csv(input_file, index=False)

    def fake_model_test_dataframe(
        dataframe,
        model_path,
        *,
        runtime_config,
        target_column="text",
        progress_callback=None,
    ):
        if progress_callback:
            progress_callback(len(dataframe))
        tested = dataframe.copy()
        tested["模型结果"] = ["请在订单页提交退款申请", "请走人工工单"]
        tested["模型调用时间"] = [0.12, 0.15]
        tested["匹配预期"] = [True, False]
        stats = BatchModelTestStats(
            total_rows=2,
            target_column="用户输入",
            has_expected_result=True,
            matched_expected_count=1,
            average_call_time_seconds=0.135,
            model_path=str(model_path),
            device="cuda:0",
            num_workers=1,
            batch_size=runtime_config.batch_size,
        )
        return tested, stats

    monkeypatch.setattr(cli, "model_test_dataframe", fake_model_test_dataframe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "model-test",
            "--input-file",
            str(input_file),
            "--test-model-path",
            "models/custom-model",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()
    output_file = tmp_path / "input_model_tested.csv"

    assert exit_code == 0
    assert "模型测试完成，输出文件" in captured.out
    assert "模型调用时间列：模型调用时间" in captured.out
    assert "匹配预期数量：1" in captured.out
    assert "实际 worker 数：1" in captured.out
    tested = pd.read_csv(output_file)
    assert tested["模型结果"].tolist() == ["请在订单页提交退款申请", "请走人工工单"]
    assert "模型调用时间" in tested.columns
    assert tested["匹配预期"].tolist() == [True, False]


def test_main_supports_file_based_model_test_without_expected_result(
    tmp_path, monkeypatch, capsys
) -> None:
    input_file = tmp_path / "input.csv"
    pd.DataFrame({"用户输入": ["退款怎么申请"]}).to_csv(input_file, index=False)

    def fake_model_test_dataframe(
        dataframe,
        model_path,
        *,
        runtime_config,
        target_column="text",
        progress_callback=None,
    ):
        if progress_callback:
            progress_callback(len(dataframe))
        tested = dataframe.copy()
        tested["模型结果"] = ["回答:退款怎么申请"]
        tested["模型调用时间"] = [0.11]
        stats = BatchModelTestStats(
            total_rows=1,
            target_column="用户输入",
            has_expected_result=False,
            matched_expected_count=0,
            average_call_time_seconds=0.11,
            model_path=str(model_path),
            device="cuda:0",
            num_workers=1,
            batch_size=runtime_config.batch_size,
        )
        return tested, stats

    monkeypatch.setattr(cli, "model_test_dataframe", fake_model_test_dataframe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--action",
            "model-test",
            "--input-file",
            str(input_file),
            "--test-model-path",
            "models/custom-model",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()
    output_file = tmp_path / "input_model_tested.csv"

    assert exit_code == 0
    assert "模型结果列：模型结果" in captured.out
    tested = pd.read_csv(output_file)
    assert tested["模型结果"].tolist() == ["回答:退款怎么申请"]
    assert "模型调用时间" in tested.columns
    assert "匹配预期" not in tested.columns
