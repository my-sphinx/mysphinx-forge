from __future__ import annotations

import argparse
import json
import math
import tempfile
from datetime import UTC, datetime
from logging import Logger
from pathlib import Path

import pandas as pd

from mysphinx_forge.cleaning import (
    CleaningStats,
    clean_dataframe,
)
from mysphinx_forge.cluster_labeling import DEFAULT_CLUSTER_LABEL_MODEL
from mysphinx_forge.clustering import ClusteringStats, TextClusterer
from mysphinx_forge.cluster_reporting import (
    build_cluster_analysis_report,
    render_cluster_report_html,
)
from mysphinx_forge.deduplication import DeduplicationStats, deduplicate_dataframe
from mysphinx_forge.file_io import (
    append_dataframe_chunk,
    count_csv_rows,
    iter_dataframes,
    load_dataframe,
    write_dataframe,
    write_match_rows,
)
from mysphinx_forge.logging_utils import close_logger, configure_logger
from mysphinx_forge.model_testing import (
    BatchModelTestStats,
    ModelTestRuntimeConfig,
    model_test_dataframe,
    run_model_test,
)
from mysphinx_forge.progress import ProgressBar, run_stage
from mysphinx_forge.semantic_deduplication import (
    DEFAULT_EMBEDDING_MODEL_PATH,
    SemanticDeduplicator,
    semantic_deduplicate_dataframe,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="数据处理工具")
    parser.add_argument(
        "--action",
        required=True,
        choices=["clean", "deduplicate", "clean-deduplicate", "cluster", "model-test"],
        help="要执行的功能。当前支持 clean、deduplicate、clean-deduplicate、cluster、model-test。",
    )
    parser.add_argument(
        "--input-file",
        dest="input_file",
        help="输入文件路径，支持 csv 和 Excel。",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="输出文件路径。未指定时，默认在原文件旁生成 *_cleaned 文件。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="csv 分块流式处理时每块读取的行数。仅对 csv 生效，默认 50000。",
    )
    parser.add_argument(
        "--target-column",
        default="text",
        help="指定执行处理的目标列名，默认 text。",
    )
    parser.add_argument(
        "--category-column",
        default="category",
        help="指定语义去重分类列名。仅影响 *_matches.csv 中 category 相关导出，默认 category。",
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=["exact", "semantic"],
        default="exact",
        help="去重模式。exact 为标准化后精确匹配，semantic 为基于向量语义相似度去重。",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.9,
        help="语义去重阈值，仅对 --dedupe-mode semantic 生效，默认 0.9。",
    )
    parser.add_argument(
        "--embedding-model-path",
        default=str(DEFAULT_EMBEDDING_MODEL_PATH),
        help="语义去重使用的本地 embedding 模型路径，默认 models/m3e-base。",
    )
    parser.add_argument(
        "--train-model-path",
        default="",
        help="模型训练使用的本地模型路径。仅对后续训练相关 action 生效。",
    )
    parser.add_argument(
        "--test-model-path",
        "--model-path",
        dest="test_model_path",
        default="",
        help="模型测试使用的本地模型路径。--model-path 为其别名，仅对 --action model-test 生效。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="语义去重时 embedding 编码批大小，默认 64。",
    )
    parser.add_argument(
        "--semantic-index-type",
        choices=["flat", "hnsw"],
        default="flat",
        help="语义去重向量索引类型。flat 为精确检索，hnsw 为近似检索，默认 flat。",
    )
    parser.add_argument(
        "--semantic-hnsw-m",
        type=int,
        default=32,
        help="语义索引为 hnsw 时的图连接度参数 M，默认 32。",
    )
    parser.add_argument(
        "--cluster-mode",
        choices=["hdbscan", "kmeans"],
        default="hdbscan",
        help="聚类模式。hdbscan 为密度聚类，kmeans 为固定簇数聚类。",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN 最小簇大小，默认 5。",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=8,
        help="KMeans 聚类簇数，默认 8。",
    )
    parser.add_argument(
        "--cluster-selection-epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN 的 cluster_selection_epsilon，默认 0。",
    )
    parser.add_argument(
        "--cluster-label-mode",
        choices=["rule", "llm"],
        default="rule",
        help="聚类标签模式。rule 为规则拼接，llm 为基于簇样本生成摘要标签。",
    )
    parser.add_argument(
        "--cluster-label-model",
        default=DEFAULT_CLUSTER_LABEL_MODEL,
        help=f"LLM 聚类标签使用的模型名，默认 {DEFAULT_CLUSTER_LABEL_MODEL}。",
    )
    parser.add_argument(
        "--cluster-label-api-base",
        default="",
        help="LLM 聚类标签接口基地址。未指定时优先读取 OPENAI_BASE_URL，否则使用官方默认地址。",
    )
    parser.add_argument(
        "--cluster-label-sample-size",
        type=int,
        default=8,
        help="生成聚类标签时每个簇送给 LLM 的示例问题数量，默认 8。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="模型测试时最大生成 token 数，默认 64。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="模型测试采样温度，默认 1.0。",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="模型测试 nucleus sampling 的 top_p，默认 1.0。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="模型测试采样时的 top_k，默认 0。",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="模型测试重复惩罚系数，默认 1.05。",
    )
    parser.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        default=False,
        help="模型测试时启用采样生成，默认关闭。",
    )
    parser.add_argument(
        "--no-do-sample",
        dest="do_sample",
        action="store_false",
        help="模型测试时关闭采样，改为确定性生成。默认即为关闭。",
    )
    parser.add_argument(
        "--model-test-batch-size",
        type=int,
        default=8,
        help="批量模型测试时单个 worker 的推理批大小，默认 8。",
    )
    parser.add_argument(
        "--model-test-num-workers",
        default="auto",
        help="批量模型测试时的 worker 数。默认 auto，会按可见 GPU 数自动决定；无 GPU 时退化为 1。",
    )

    args = parser.parse_args()
    if args.action != "model-test" and not args.input_file:
        print("未检测到支持的输入文件，请提供 csv 或 Excel 文件（.csv/.xls/.xlsx/.xlsm）。")
        return 1
    if args.action == "model-test" and not args.test_model_path:
        print("--test-model-path 为必填参数，--model-path 可作为别名，且仅用于 model-test。")
        return 1
    if args.model_test_batch_size <= 0:
        print("--model-test-batch-size 必须是大于 0 的整数。")
        return 1
    if args.max_new_tokens <= 0:
        print("--max-new-tokens 必须是大于 0 的整数。")
        return 1
    if args.chunk_size <= 0:
        print("--chunk-size 必须是大于 0 的整数。")
        return 1
    if args.batch_size <= 0:
        print("--batch-size 必须是大于 0 的整数。")
        return 1
    if args.semantic_hnsw_m <= 0:
        print("--semantic-hnsw-m 必须是大于 0 的整数。")
        return 1
    if args.min_cluster_size <= 0:
        print("--min-cluster-size 必须是大于 0 的整数。")
        return 1
    if args.num_clusters <= 0:
        print("--num-clusters 必须是大于 0 的整数。")
        return 1
    if args.cluster_selection_epsilon < 0:
        print("--cluster-selection-epsilon 不能小于 0。")
        return 1
    if args.cluster_label_sample_size <= 0:
        print("--cluster-label-sample-size 必须是大于 0 的整数。")
        return 1
    if args.temperature <= 0:
        print("--temperature 必须是大于 0 的数值。")
        return 1
    if not 0 < args.top_p <= 1:
        print("--top-p 必须在 0 到 1 之间。")
        return 1
    if args.top_k < 0:
        print("--top-k 不能小于 0。")
        return 1
    if args.repetition_penalty <= 0:
        print("--repetition-penalty 必须是大于 0 的数值。")
        return 1
    if args.model_test_num_workers != "auto":
        try:
            parsed_worker_count = int(args.model_test_num_workers)
        except ValueError:
            print("--model-test-num-workers 必须是 auto 或大于 0 的整数。")
            return 1
        if parsed_worker_count <= 0:
            print("--model-test-num-workers 必须是 auto 或大于 0 的整数。")
            return 1
        args.model_test_num_workers = parsed_worker_count
    if not 0 < args.semantic_threshold <= 1:
        print("--semantic-threshold 必须在 0 到 1 之间。")
        return 1

    if args.action == "clean":
        return _run_clean(args.input_file, args.output, args.chunk_size, args.target_column)
    if args.action == "deduplicate":
        return _run_deduplicate(
            args.input_file,
            args.output,
            args.chunk_size,
            args.target_column,
            args.category_column,
            args.dedupe_mode,
            args.semantic_threshold,
            args.embedding_model_path,
            args.batch_size,
            args.semantic_index_type,
            args.semantic_hnsw_m,
        )
    if args.action == "clean-deduplicate":
        return _run_clean_deduplicate(
            args.input_file,
            args.output,
            args.chunk_size,
            args.target_column,
            args.category_column,
            args.dedupe_mode,
            args.semantic_threshold,
            args.embedding_model_path,
            args.batch_size,
            args.semantic_index_type,
            args.semantic_hnsw_m,
        )
    if args.action == "cluster":
        return _run_cluster(
            args.input_file,
            args.output,
            args.target_column,
            args.embedding_model_path,
            args.batch_size,
            args.cluster_mode,
            args.min_cluster_size,
            args.num_clusters,
            args.cluster_selection_epsilon,
            args.cluster_label_mode,
            args.cluster_label_model,
            args.cluster_label_api_base or None,
            args.cluster_label_sample_size,
        )
    if args.action == "model-test":
        return _run_model_test(
            model_path=args.test_model_path,
            input_file=args.input_file,
            output_arg=args.output,
            target_column=args.target_column,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            model_test_batch_size=args.model_test_batch_size,
            model_test_num_workers=args.model_test_num_workers,
        )

    parser.print_help()
    return 1


def _run_clean(
    input_file: str,
    output_arg: str | None,
    chunk_size: int,
    target_column: str,
) -> int:
    input_path = Path(input_file)
    output_path = _resolve_output_path(input_path, output_arg)
    logger = configure_logger(_resolve_log_path(output_path))
    logger.info("开始执行 action=clean input=%s output=%s", input_path, output_path)

    if input_path.suffix.lower() == ".csv":
        try:
            stats = _run_clean_csv_stream(input_path, output_path, chunk_size, target_column, logger)
        except ValueError as exc:
            _emit_error(str(exc), logger)
            close_logger()
            return 1
        _write_meta(
            output_path=output_path,
            action="clean",
            input_path=input_path,
            parameters={
                "chunk_size": chunk_size,
                "target_column": target_column,
            },
            cleaning_stats=stats,
        )
        _print_stats(stats, output_path, logger)
        close_logger()
        return 0

    try:
        run_stage("读取文件", logger=logger)
        dataframe = load_dataframe(input_file)
    except ValueError as exc:
        _emit_error(str(exc), logger)
        close_logger()
        return 1

    progress_bar = ProgressBar(total=len(dataframe), description="清洗数据", logger=logger)
    try:
        cleaned, stats = clean_dataframe(
            dataframe,
            target_column=target_column,
            progress_callback=progress_bar.advance,
        )
        progress_bar.set_summary(
            total_before=stats.total_before,
            total_removed=stats.total_removed,
            total_after=stats.total_after,
            removed_blank_rows=stats.removed_blank_rows,
            removed_symbol_rows=stats.removed_symbol_rows,
            removed_emoji_rows=stats.removed_emoji_rows,
            removed_garbled_rows=stats.removed_garbled_rows,
        )
    finally:
        progress_bar.close()

    run_stage("写出结果", logger=logger)
    write_dataframe(cleaned, output_path)
    _write_meta(
        output_path=output_path,
        action="clean",
        input_path=input_path,
        parameters={
            "chunk_size": chunk_size,
            "target_column": target_column,
        },
        cleaning_stats=stats,
    )
    _print_stats(stats, output_path, logger)
    close_logger()
    return 0


def _run_deduplicate(
    input_file: str,
    output_arg: str | None,
    chunk_size: int,
    target_column: str,
    category_column: str,
    dedupe_mode: str,
    semantic_threshold: float,
    embedding_model_path: str,
    batch_size: int,
    semantic_index_type: str,
    semantic_hnsw_m: int,
) -> int:
    input_path = Path(input_file)
    output_path = _resolve_deduplicate_output_path(input_path, output_arg)
    logger = configure_logger(_resolve_log_path(output_path))
    logger.info(
        "开始执行 action=deduplicate input=%s output=%s mode=%s",
        input_path,
        output_path,
        dedupe_mode,
    )

    semantic_deduplicator = _build_semantic_deduplicator(
        dedupe_mode,
        embedding_model_path,
        semantic_threshold,
        batch_size,
        semantic_index_type,
        semantic_hnsw_m,
    )

    if input_path.suffix.lower() == ".csv":
        try:
            stats = _run_deduplicate_csv_stream(
                input_path,
                output_path,
                chunk_size,
                target_column,
                category_column,
                logger,
                dedupe_mode,
                semantic_deduplicator,
            )
        except ValueError as exc:
            _emit_error(str(exc), logger)
            close_logger()
            return 1
        _write_meta(
            output_path=output_path,
            action="deduplicate",
            input_path=input_path,
            parameters=_build_deduplication_parameters(
                chunk_size=chunk_size,
                target_column=target_column,
                category_column=category_column,
                dedupe_mode=dedupe_mode,
                semantic_threshold=semantic_threshold,
                embedding_model_path=embedding_model_path,
                batch_size=batch_size,
                semantic_index_type=semantic_index_type,
                semantic_hnsw_m=semantic_hnsw_m,
            ),
            deduplication_stats=stats,
            match_output_path=_resolve_match_output_path(output_path) if dedupe_mode == "semantic" else None,
        )
        _print_deduplication_stats(stats, output_path, logger)
        close_logger()
        return 0

    try:
        run_stage("读取文件", logger=logger)
        dataframe = load_dataframe(input_file)
        progress_bar = ProgressBar(total=len(dataframe), description="执行去重", logger=logger)
        try:
            deduplicated, stats, match_rows = _deduplicate_dataframe(
                dataframe=dataframe,
                target_column=target_column,
                category_column=category_column,
                dedupe_mode=dedupe_mode,
                semantic_deduplicator=semantic_deduplicator,
                progress_callback=progress_bar.advance,
            )
            progress_bar.set_postfix(
                {
                    "总数": stats.total_before,
                    "重复": stats.duplicate_rows,
                    "保留": stats.total_after,
                    "唯一值": stats.unique_values,
                }
            )
        finally:
            progress_bar.close()
    except ValueError as exc:
        _emit_error(str(exc), logger)
        close_logger()
        return 1
    except Exception as exc:
        logger.exception("执行去重失败")
        _emit_error(f"执行去重失败：{type(exc).__name__}: {exc}", logger)
        close_logger()
        return 1

    run_stage("写出结果", logger=logger)
    write_dataframe(deduplicated, output_path)
    write_match_rows(
        match_rows,
        _resolve_match_output_path(output_path),
        category_column=category_column,
    )
    _write_meta(
        output_path=output_path,
        action="deduplicate",
        input_path=input_path,
        parameters=_build_deduplication_parameters(
            chunk_size=chunk_size,
            target_column=target_column,
            category_column=category_column,
            dedupe_mode=dedupe_mode,
            semantic_threshold=semantic_threshold,
            embedding_model_path=embedding_model_path,
            batch_size=batch_size,
            semantic_index_type=semantic_index_type,
            semantic_hnsw_m=semantic_hnsw_m,
        ),
        deduplication_stats=stats,
        match_output_path=_resolve_match_output_path(output_path) if dedupe_mode == "semantic" else None,
    )
    _print_deduplication_stats(stats, output_path, logger)
    close_logger()
    return 0


def _run_clean_deduplicate(
    input_file: str,
    output_arg: str | None,
    chunk_size: int,
    target_column: str,
    category_column: str,
    dedupe_mode: str,
    semantic_threshold: float,
    embedding_model_path: str,
    batch_size: int,
    semantic_index_type: str,
    semantic_hnsw_m: int,
) -> int:
    input_path = Path(input_file)
    output_path = _resolve_deduplicate_output_path(input_path, output_arg)
    logger = configure_logger(_resolve_log_path(output_path))
    logger.info(
        "开始执行 action=clean-deduplicate input=%s output=%s mode=%s",
        input_path,
        output_path,
        dedupe_mode,
    )

    semantic_deduplicator = _build_semantic_deduplicator(
        dedupe_mode,
        embedding_model_path,
        semantic_threshold,
        batch_size,
        semantic_index_type,
        semantic_hnsw_m,
    )

    if input_path.suffix.lower() == ".csv":
        return _run_clean_deduplicate_csv(
            input_path=input_path,
            output_path=output_path,
            chunk_size=chunk_size,
            target_column=target_column,
            category_column=category_column,
            dedupe_mode=dedupe_mode,
            semantic_deduplicator=semantic_deduplicator,
            logger=logger,
        )

    try:
        run_stage("读取文件", logger=logger)
        dataframe = load_dataframe(input_file)
        clean_bar = ProgressBar(total=len(dataframe), description="清洗数据", logger=logger)
        try:
            cleaned, clean_stats = clean_dataframe(
                dataframe,
                target_column=target_column,
                progress_callback=clean_bar.advance,
            )
            clean_bar.set_summary(
                total_before=clean_stats.total_before,
                total_removed=clean_stats.total_removed,
                total_after=clean_stats.total_after,
                removed_blank_rows=clean_stats.removed_blank_rows,
                removed_symbol_rows=clean_stats.removed_symbol_rows,
                removed_emoji_rows=clean_stats.removed_emoji_rows,
                removed_garbled_rows=clean_stats.removed_garbled_rows,
            )
        finally:
            clean_bar.close()

        dedupe_bar = ProgressBar(total=len(cleaned), description="执行去重", logger=logger)
        try:
            deduplicated, dedupe_stats, match_rows = _deduplicate_dataframe(
                dataframe=cleaned,
                target_column=target_column,
                category_column=category_column,
                dedupe_mode=dedupe_mode,
                semantic_deduplicator=semantic_deduplicator,
                progress_callback=dedupe_bar.advance,
            )
            dedupe_bar.set_postfix(
                {
                    "总数": dedupe_stats.total_before,
                    "重复": dedupe_stats.duplicate_rows,
                    "保留": dedupe_stats.total_after,
                    "唯一值": dedupe_stats.unique_values,
                }
            )
        finally:
            dedupe_bar.close()
    except ValueError as exc:
        _emit_error(str(exc), logger)
        close_logger()
        return 1
    except Exception as exc:
        logger.exception("执行清洗去重失败")
        _emit_error(f"执行清洗去重失败：{type(exc).__name__}: {exc}", logger)
        close_logger()
        return 1

    run_stage("写出结果", logger=logger)
    write_dataframe(deduplicated, output_path)
    write_match_rows(
        match_rows,
        _resolve_match_output_path(output_path),
        category_column=category_column,
    )
    _write_meta(
        output_path=output_path,
        action="clean-deduplicate",
        input_path=input_path,
        parameters={
            "chunk_size": chunk_size,
            "target_column": target_column,
            "category_column": category_column,
            "dedupe_mode": dedupe_mode,
            "semantic_threshold": semantic_threshold,
            "embedding_model_path": embedding_model_path,
            "batch_size": batch_size,
        },
        cleaning_stats=clean_stats,
        deduplication_stats=dedupe_stats,
        match_output_path=_resolve_match_output_path(output_path) if dedupe_mode == "semantic" else None,
    )
    _print_stats(clean_stats, output_path, logger)
    _print_deduplication_stats(dedupe_stats, output_path, logger)
    close_logger()
    return 0


def _run_cluster(
    input_file: str,
    output_arg: str | None,
    target_column: str,
    embedding_model_path: str,
    batch_size: int,
    cluster_mode: str,
    min_cluster_size: int,
    num_clusters: int,
    cluster_selection_epsilon: float,
    cluster_label_mode: str,
    cluster_label_model: str,
    cluster_label_api_base: str | None,
    cluster_label_sample_size: int,
) -> int:
    input_path = Path(input_file)
    output_path = _resolve_cluster_output_path(input_path, output_arg)
    cluster_summary_path = _resolve_cluster_summary_output_path(output_path)
    projection_path = _resolve_projection_output_path(output_path)
    analysis_path = _resolve_cluster_analysis_output_path(output_path)
    html_report_path = _resolve_cluster_report_html_output_path(output_path)
    logger = configure_logger(_resolve_log_path(output_path))
    logger.info(
        "开始执行 action=cluster input=%s output=%s mode=%s",
        input_path,
        output_path,
        cluster_mode,
    )

    clusterer = TextClusterer(
        model_path=embedding_model_path,
        cluster_mode=cluster_mode,
        batch_size=batch_size,
        min_cluster_size=min_cluster_size,
        num_clusters=num_clusters,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_label_mode=cluster_label_mode,
        cluster_label_model=cluster_label_model,
        cluster_label_api_base=cluster_label_api_base,
        cluster_label_sample_size=cluster_label_sample_size,
    )

    try:
        run_stage("读取文件", logger=logger)
        dataframe = load_dataframe(input_file)
        progress_bar = ProgressBar(total=len(dataframe), description="执行聚类", logger=logger)
        try:
            clustered, cluster_summary, projection, stats = clusterer.cluster_dataframe(
                dataframe,
                target_column=target_column,
                progress_callback=progress_bar.advance,
            )
            progress_bar.set_postfix(
                {
                    "总数": stats.total_before,
                    "簇数": stats.cluster_count,
                    "噪声": stats.noise_rows,
                    "入簇": stats.total_clustered,
                }
            )
        finally:
            progress_bar.close()
    except ValueError as exc:
        _emit_error(str(exc), logger)
        close_logger()
        return 1
    except Exception as exc:
        logger.exception("执行聚类失败")
        _emit_error(f"执行聚类失败：{type(exc).__name__}: {exc}", logger)
        close_logger()
        return 1

    run_stage("写出结果", logger=logger)
    analysis_report = build_cluster_analysis_report(cluster_summary, stats)
    write_dataframe(clustered, output_path)
    write_dataframe(cluster_summary, cluster_summary_path)
    write_dataframe(projection, projection_path)
    write_dataframe(analysis_report, analysis_path)
    html_report_path.write_text(
        render_cluster_report_html(
            analysis_report=analysis_report,
            projection=projection,
            stats=stats,
        ),
        encoding="utf-8",
    )
    _write_meta(
        output_path=output_path,
        action="cluster",
        input_path=input_path,
        parameters={
            "target_column": target_column,
            "embedding_model_path": embedding_model_path,
            "batch_size": batch_size,
            "cluster_mode": cluster_mode,
            "min_cluster_size": min_cluster_size,
            "num_clusters": num_clusters,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "cluster_label_mode": cluster_label_mode,
            "cluster_label_model": cluster_label_model,
            "cluster_label_api_base": cluster_label_api_base,
            "cluster_label_sample_size": cluster_label_sample_size,
        },
        clustering_stats=stats,
        cluster_summary_path=cluster_summary_path,
        projection_path=projection_path,
        analysis_path=analysis_path,
        html_report_path=html_report_path,
    )
    _print_clustering_stats(
        stats,
        output_path,
        cluster_summary_path,
        projection_path,
        analysis_path,
        html_report_path,
        logger,
    )
    close_logger()
    return 0


def _run_model_test(
    model_path: str,
    input_file: str | None,
    output_arg: str | None,
    target_column: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    model_test_batch_size: int,
    model_test_num_workers: int | str,
) -> int:
    if input_file:
        return _run_model_test_on_file(
            input_file=input_file,
            output_arg=output_arg,
            target_column=target_column,
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            model_test_batch_size=model_test_batch_size,
            model_test_num_workers=model_test_num_workers,
        )

    log_path = Path.cwd() / "mysphinx-forge.log"
    logger = configure_logger(log_path)
    logger.info("开始执行 action=model-test model=%s", model_path)

    try:
        result = run_model_test(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
    except ValueError as exc:
        _emit_error(str(exc), logger)
        close_logger()
        return 1
    except Exception as exc:
        logger.exception("执行模型测试失败")
        _emit_error(f"执行模型测试失败：{type(exc).__name__}: {exc}", logger)
        close_logger()
        return 1

    _emit_message("模型测试完成", logger)
    _emit_message(f"模型路径：{result.model_path}", logger)
    _emit_message(f"测试输入：{result.user_input}", logger)
    _emit_message(f"模型类型：{result.model_class}", logger)
    _emit_message(f"Tokenizer 类型：{result.tokenizer_class}", logger)
    _emit_message(f"推理设备：{result.device}", logger)
    _emit_message(
        "生成参数："
        f"max_new_tokens={max_new_tokens}, "
        f"do_sample={do_sample}, "
        f"temperature={temperature}, "
        f"top_p={top_p}, "
        f"top_k={top_k}, "
        f"repetition_penalty={repetition_penalty}",
        logger,
    )
    _emit_message(f"模型输出：{result.generated_text}", logger)
    close_logger()
    return 0


def _run_model_test_on_file(
    input_file: str,
    output_arg: str | None,
    target_column: str,
    model_path: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    model_test_batch_size: int,
    model_test_num_workers: int | str,
) -> int:
    input_path = Path(input_file)
    output_path = _resolve_model_test_output_path(input_path, output_arg)
    logger = configure_logger(_resolve_log_path(output_path))
    logger.info("开始执行 action=model-test input=%s output=%s model=%s", input_path, output_path, model_path)

    try:
        run_stage("读取文件", logger=logger)
        dataframe = load_dataframe(input_file)
        progress_bar = ProgressBar(total=len(dataframe), description="执行模型测试", logger=logger)
        try:
            tested, stats = model_test_dataframe(
                dataframe=dataframe,
                model_path=model_path,
                runtime_config=ModelTestRuntimeConfig(
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    batch_size=model_test_batch_size,
                    num_workers=model_test_num_workers,
                ),
                target_column=target_column,
                progress_callback=progress_bar.advance,
            )
            progress_bar.set_postfix(
                {
                    "总数": stats.total_rows,
                    "命中预期": stats.matched_expected_count if stats.has_expected_result else "-",
                }
            )
        finally:
            progress_bar.close()
    except ValueError as exc:
        _emit_error(str(exc), logger)
        close_logger()
        return 1
    except Exception as exc:
        logger.exception("执行模型文件测试失败")
        _emit_error(f"执行模型文件测试失败：{type(exc).__name__}: {exc}", logger)
        close_logger()
        return 1

    run_stage("写出结果", logger=logger)
    write_dataframe(tested, output_path)
    _print_batch_model_test_stats(
        stats=stats,
        output_path=output_path,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        model_test_batch_size=model_test_batch_size,
        model_test_num_workers=model_test_num_workers,
        logger=logger,
    )
    close_logger()
    return 0


def _run_clean_csv_stream(
    input_path: Path,
    output_path: Path,
    chunk_size: int,
    target_column: str,
    logger: Logger,
) -> CleaningStats:
    total_stats = CleaningStats(total_before=0, total_after=0)
    run_stage("统计总行数", logger=logger)
    total_rows = count_csv_rows(input_path)
    if output_path.exists():
        output_path.unlink()
    if total_rows == 0:
        empty_bar = ProgressBar(total=1, description="分块清洗", logger=logger)
        empty_bar.set_summary(
            total_before=0,
            total_removed=0,
            total_after=0,
            removed_blank_rows=0,
            removed_symbol_rows=0,
            removed_emoji_rows=0,
            removed_garbled_rows=0,
        )
        empty_bar.advance(1)
        empty_bar.close()
        write_bar = ProgressBar(total=1, description="写出结果", logger=logger)
        write_dataframe(pd.DataFrame(), output_path)
        write_bar.advance(1)
        write_bar.close()
        return total_stats

    progress_bar = ProgressBar(total=total_rows, description="分块清洗", logger=logger)
    chunk_total = math.ceil(total_rows / chunk_size)
    write_bar = ProgressBar(total=chunk_total, description="写出结果", logger=logger)
    wrote_header = False

    try:
        for chunk in iter_dataframes(input_path, chunksize=chunk_size):
            cleaned_chunk, chunk_stats = clean_dataframe(
                chunk,
                target_column=target_column,
                progress_callback=progress_bar.advance,
            )
            total_stats.merge(chunk_stats)
            wrote_header = append_dataframe_chunk(
                cleaned_chunk,
                output_path,
                wrote_header=wrote_header,
            )
            write_bar.advance(1)
        progress_bar.set_summary(
            total_before=total_stats.total_before,
            total_removed=total_stats.total_removed,
            total_after=total_stats.total_after,
            removed_blank_rows=total_stats.removed_blank_rows,
            removed_symbol_rows=total_stats.removed_symbol_rows,
            removed_emoji_rows=total_stats.removed_emoji_rows,
            removed_garbled_rows=total_stats.removed_garbled_rows,
        )
    finally:
        progress_bar.close()
        write_bar.close()

    return total_stats


def _run_deduplicate_csv_stream(
    input_path: Path,
    output_path: Path,
    chunk_size: int,
    target_column: str,
    category_column: str,
    logger: Logger,
    dedupe_mode: str,
    semantic_deduplicator: SemanticDeduplicator | None,
) -> DeduplicationStats:
    total_stats = DeduplicationStats(total_before=0, total_after=0)
    seen_keys: set[str] = set()
    processed_rows = 0

    run_stage("统计总行数", logger=logger)
    total_rows = count_csv_rows(input_path)
    if output_path.exists():
        output_path.unlink()
    match_output_path = _resolve_match_output_path(output_path)
    if match_output_path.exists():
        match_output_path.unlink()
    if total_rows == 0:
        dedupe_bar = ProgressBar(total=1, description="分块去重", logger=logger)
        dedupe_bar.set_postfix({"总数": 0, "重复": 0, "保留": 0, "唯一值": 0})
        dedupe_bar.advance(1)
        dedupe_bar.close()
        write_bar = ProgressBar(total=1, description="写出结果", logger=logger)
        write_dataframe(pd.DataFrame(), output_path)
        write_bar.advance(1)
        write_bar.close()
        return total_stats

    progress_bar = ProgressBar(total=total_rows, description="分块去重", logger=logger)
    chunk_total = math.ceil(total_rows / chunk_size)
    write_bar = ProgressBar(total=chunk_total, description="写出结果", logger=logger)
    wrote_header = False

    try:
        for chunk in iter_dataframes(input_path, chunksize=chunk_size):
            deduplicated_chunk, chunk_stats, chunk_match_rows = _deduplicate_dataframe(
                dataframe=chunk,
                target_column=target_column,
                category_column=category_column,
                dedupe_mode=dedupe_mode,
                seen_keys=seen_keys,
                semantic_deduplicator=semantic_deduplicator,
                progress_callback=progress_bar.advance,
                row_index_offset=processed_rows,
            )
            processed_rows += chunk_stats.total_before
            total_stats.total_before += chunk_stats.total_before
            total_stats.total_after += chunk_stats.total_after
            total_stats.duplicate_rows += chunk_stats.duplicate_rows
            total_stats.unique_values = chunk_stats.unique_values
            total_stats.target_column = chunk_stats.target_column
            total_stats.dedupe_mode = chunk_stats.dedupe_mode
            total_stats.semantic_threshold = chunk_stats.semantic_threshold
            total_stats.embedding_model_path = chunk_stats.embedding_model_path
            # Keep semantic audit rows on disk while streaming so duplicate-heavy
            # datasets do not accumulate a second large in-memory result set.
            write_match_rows(
                chunk_match_rows,
                match_output_path,
                category_column=category_column,
                append=True,
            )
            wrote_header = append_dataframe_chunk(
                deduplicated_chunk,
                output_path,
                wrote_header=wrote_header,
            )
            write_bar.advance(1)
        progress_bar.set_postfix(
            {
                "总数": total_stats.total_before,
                "重复": total_stats.duplicate_rows,
                "保留": total_stats.total_after,
                "唯一值": total_stats.unique_values,
            }
        )
    finally:
        progress_bar.close()
        write_bar.close()

    return total_stats


def _run_clean_deduplicate_csv(
    input_path: Path,
    output_path: Path,
    chunk_size: int,
    target_column: str,
    category_column: str,
    dedupe_mode: str,
    semantic_deduplicator: SemanticDeduplicator | None,
    logger: Logger,
) -> int:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{input_path.stem}_cleaned_",
            suffix=".csv",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)

        clean_stats = _run_clean_csv_stream(
            input_path=input_path,
            output_path=temp_path,
            chunk_size=chunk_size,
            target_column=target_column,
            logger=logger,
        )
        dedupe_stats = _run_deduplicate_csv_stream(
            input_path=temp_path,
            output_path=output_path,
            chunk_size=chunk_size,
            target_column=target_column,
            category_column=category_column,
            logger=logger,
            dedupe_mode=dedupe_mode,
            semantic_deduplicator=semantic_deduplicator,
        )
        _write_meta(
            output_path=output_path,
            action="clean-deduplicate",
            input_path=input_path,
            parameters=_build_deduplication_parameters(
                chunk_size=chunk_size,
                target_column=target_column,
                category_column=category_column,
                dedupe_mode=dedupe_mode,
                semantic_threshold=semantic_deduplicator.threshold if semantic_deduplicator else None,
                embedding_model_path=(
                    str(semantic_deduplicator.model_path) if semantic_deduplicator else None
                ),
                batch_size=semantic_deduplicator.batch_size if semantic_deduplicator else None,
                semantic_index_type=semantic_deduplicator.index_type if semantic_deduplicator else None,
                semantic_hnsw_m=semantic_deduplicator.hnsw_m if semantic_deduplicator else None,
            ),
            cleaning_stats=clean_stats,
            deduplication_stats=dedupe_stats,
            match_output_path=_resolve_match_output_path(output_path) if dedupe_mode == "semantic" else None,
        )
        _print_stats(clean_stats, output_path, logger)
        _print_deduplication_stats(dedupe_stats, output_path, logger)
        close_logger()
        return 0
    except ValueError as exc:
        _emit_error(str(exc), logger)
        close_logger()
        return 1
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def _resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)
    return input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")


def _resolve_deduplicate_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)
    return input_path.with_name(f"{input_path.stem}_deduplicated{input_path.suffix}")


def _resolve_cluster_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)
    return input_path.with_name(f"{input_path.stem}_clustered{input_path.suffix}")


def _resolve_model_test_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)
    return input_path.with_name(f"{input_path.stem}_model_tested{input_path.suffix}")


def _resolve_match_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_matches.csv")


def _resolve_cluster_summary_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_clusters.csv")


def _resolve_projection_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_projection.csv")


def _resolve_cluster_analysis_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_analysis.csv")


def _resolve_cluster_report_html_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_report.html")


def _resolve_meta_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.meta.json")


def _resolve_log_path(output_path: Path) -> Path:
    return output_path.parent / "mysphinx-forge.log"


def _print_stats(stats: CleaningStats, output_path: Path, logger: Logger) -> None:
    _emit_message(f"清洗完成，输出文件：{output_path}", logger)
    _emit_message(f"清洗前总行数：{stats.total_before}", logger)
    _emit_message(f"删除空行：{stats.removed_blank_rows}", logger)
    _emit_message(f"删除全符号/标点行：{stats.removed_symbol_rows}", logger)
    _emit_message(f"删除全表情行：{stats.removed_emoji_rows}", logger)
    _emit_message(f"删除全乱码行：{stats.removed_garbled_rows}", logger)
    _emit_message(f"共删除：{stats.total_removed}", logger)
    _emit_message(f"清洗后总行数：{stats.total_after}", logger)


def _print_deduplication_stats(
    stats: DeduplicationStats,
    output_path: Path,
    logger: Logger,
) -> None:
    _emit_message(f"去重完成，输出文件：{output_path}", logger)
    _emit_message(f"去重模式：{stats.dedupe_mode}", logger)
    _emit_message(f"使用目标列：{stats.target_column}", logger)
    if stats.dedupe_mode == "semantic":
        _emit_message(f"语义阈值：{stats.semantic_threshold}", logger)
        _emit_message(f"语义模型路径：{stats.embedding_model_path}", logger)
        _emit_message(f"语义命中明细：{_resolve_match_output_path(output_path)}", logger)
    _emit_message(f"去重前总行数：{stats.total_before}", logger)
    _emit_message(f"删除重复行数：{stats.duplicate_rows}", logger)
    if stats.dedupe_mode == "semantic":
        _emit_message(f"语义代表值数量：{stats.unique_values}", logger)
    else:
        _emit_message(f"标准化后唯一值数量：{stats.unique_values}", logger)
    _emit_message(f"去重后总行数：{stats.total_after}", logger)


def _print_clustering_stats(
    stats: ClusteringStats,
    output_path: Path,
    cluster_summary_path: Path,
    projection_path: Path,
    analysis_path: Path,
    html_report_path: Path,
    logger: Logger,
) -> None:
    _emit_message(f"聚类完成，输出文件：{output_path}", logger)
    _emit_message(f"聚类模式：{stats.cluster_mode}", logger)
    _emit_message(f"标签模式：{stats.cluster_label_mode}", logger)
    if stats.cluster_label_model:
        _emit_message(f"标签模型：{stats.cluster_label_model}", logger)
    _emit_message(f"使用目标列：{stats.target_column}", logger)
    _emit_message(f"语义模型路径：{stats.embedding_model_path}", logger)
    _emit_message(f"聚类前总行数：{stats.total_before}", logger)
    _emit_message(f"成功入簇行数：{stats.total_clustered}", logger)
    _emit_message(f"噪声点行数：{stats.noise_rows}", logger)
    _emit_message(f"聚类簇数量：{stats.cluster_count}", logger)
    _emit_message(f"最大簇大小：{stats.largest_cluster_size}", logger)
    _emit_message(f"最小簇大小：{stats.smallest_cluster_size}", logger)
    _emit_message(f"平均簇大小：{stats.average_cluster_size:.2f}", logger)
    _emit_message(f"聚类汇总文件：{cluster_summary_path}", logger)
    _emit_message(f"聚类投影文件：{projection_path}", logger)
    _emit_message(f"聚类分析报表：{analysis_path}", logger)
    _emit_message(f"聚类可视化报告：{html_report_path}", logger)


def _print_batch_model_test_stats(
    stats: BatchModelTestStats,
    output_path: Path,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    model_test_batch_size: int,
    model_test_num_workers: int | str,
    logger: Logger,
) -> None:
    _emit_message(f"模型测试完成，输出文件：{output_path}", logger)
    _emit_message(f"测试模型路径：{stats.model_path}", logger)
    _emit_message(f"使用目标列：{stats.target_column}", logger)
    _emit_message(f"推理设备：{stats.device}", logger)
    _emit_message(f"实际 worker 数：{stats.num_workers}", logger)
    _emit_message(f"批量推理大小：{stats.batch_size}", logger)
    _emit_message(
        "生成参数："
        f"max_new_tokens={max_new_tokens}, "
        f"do_sample={do_sample}, "
        f"temperature={temperature}, "
        f"top_p={top_p}, "
        f"top_k={top_k}, "
        f"repetition_penalty={repetition_penalty}, "
        f"worker_setting={model_test_num_workers}, "
        f"batch_setting={model_test_batch_size}",
        logger,
    )
    _emit_message(f"测试总行数：{stats.total_rows}", logger)
    _emit_message(f"模型结果列：{stats.model_result_column}", logger)
    _emit_message(f"模型调用时间列：{stats.model_call_time_column}", logger)
    _emit_message(f"平均模型调用时间（秒）：{stats.average_call_time_seconds}", logger)
    if stats.has_expected_result:
        _emit_message(f"预期结果列：{stats.expected_result_column}", logger)
        _emit_message(f"匹配结果列：{stats.match_expected_column}", logger)
        _emit_message(f"匹配预期数量：{stats.matched_expected_count}", logger)


def _emit_message(message: str, logger: Logger) -> None:
    print(message)
    logger.info(message)


def _emit_error(message: str, logger: Logger) -> None:
    print(message)
    logger.error(message)


def _write_meta(
    output_path: Path,
    action: str,
    input_path: Path,
    parameters: dict[str, object],
    cleaning_stats: CleaningStats | None = None,
    deduplication_stats: DeduplicationStats | None = None,
    clustering_stats: ClusteringStats | None = None,
    match_output_path: Path | None = None,
    cluster_summary_path: Path | None = None,
    projection_path: Path | None = None,
    analysis_path: Path | None = None,
    html_report_path: Path | None = None,
) -> None:
    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "action": action,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "log_file": str(_resolve_log_path(output_path)),
        "parameters": parameters,
    }
    if cleaning_stats is not None:
        meta["cleaning_stats"] = {
            "total_before": cleaning_stats.total_before,
            "removed_blank_rows": cleaning_stats.removed_blank_rows,
            "removed_symbol_rows": cleaning_stats.removed_symbol_rows,
            "removed_emoji_rows": cleaning_stats.removed_emoji_rows,
            "removed_garbled_rows": cleaning_stats.removed_garbled_rows,
            "total_removed": cleaning_stats.total_removed,
            "total_after": cleaning_stats.total_after,
        }
    if deduplication_stats is not None:
        meta["deduplication_stats"] = {
            "target_column": deduplication_stats.target_column,
            "dedupe_mode": deduplication_stats.dedupe_mode,
            "semantic_threshold": deduplication_stats.semantic_threshold,
            "embedding_model_path": deduplication_stats.embedding_model_path,
            "total_before": deduplication_stats.total_before,
            "duplicate_rows": deduplication_stats.duplicate_rows,
            "unique_values": deduplication_stats.unique_values,
            "total_after": deduplication_stats.total_after,
        }
    if clustering_stats is not None:
        meta["clustering_stats"] = {
            "target_column": clustering_stats.target_column,
            "cluster_mode": clustering_stats.cluster_mode,
            "embedding_model_path": clustering_stats.embedding_model_path,
            "cluster_label_mode": clustering_stats.cluster_label_mode,
            "cluster_label_model": clustering_stats.cluster_label_model,
            "total_before": clustering_stats.total_before,
            "total_clustered": clustering_stats.total_clustered,
            "noise_rows": clustering_stats.noise_rows,
            "cluster_count": clustering_stats.cluster_count,
            "largest_cluster_size": clustering_stats.largest_cluster_size,
            "smallest_cluster_size": clustering_stats.smallest_cluster_size,
            "average_cluster_size": clustering_stats.average_cluster_size,
        }
    if match_output_path is not None and match_output_path.exists():
        meta["match_file"] = str(match_output_path)
    if cluster_summary_path is not None and cluster_summary_path.exists():
        meta["cluster_summary_file"] = str(cluster_summary_path)
    if projection_path is not None and projection_path.exists():
        meta["projection_file"] = str(projection_path)
    if analysis_path is not None and analysis_path.exists():
        meta["analysis_file"] = str(analysis_path)
    if html_report_path is not None and html_report_path.exists():
        meta["html_report_file"] = str(html_report_path)

    _resolve_meta_output_path(output_path).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_deduplication_parameters(
    chunk_size: int,
    target_column: str,
    category_column: str,
    dedupe_mode: str,
    semantic_threshold: float | None,
    embedding_model_path: str | None,
    batch_size: int | None,
    semantic_index_type: str | None,
    semantic_hnsw_m: int | None,
) -> dict[str, object]:
    return {
        "chunk_size": chunk_size,
        "target_column": target_column,
        "category_column": category_column,
        "dedupe_mode": dedupe_mode,
        "semantic_threshold": semantic_threshold,
        "embedding_model_path": embedding_model_path,
        "batch_size": batch_size,
        "semantic_index_type": semantic_index_type,
        "semantic_hnsw_m": semantic_hnsw_m,
    }


def _build_semantic_deduplicator(
    dedupe_mode: str,
    embedding_model_path: str,
    semantic_threshold: float,
    batch_size: int,
    semantic_index_type: str,
    semantic_hnsw_m: int,
) -> SemanticDeduplicator | None:
    if dedupe_mode != "semantic":
        return None

    return SemanticDeduplicator(
        model_path=embedding_model_path,
        threshold=semantic_threshold,
        batch_size=batch_size,
        index_type=semantic_index_type,
        hnsw_m=semantic_hnsw_m,
    )


def _deduplicate_dataframe(
    dataframe: pd.DataFrame,
    target_column: str,
    category_column: str,
    dedupe_mode: str,
    progress_callback=None,
    seen_keys: set[str] | None = None,
    semantic_deduplicator: SemanticDeduplicator | None = None,
    row_index_offset: int = 0,
) -> tuple[pd.DataFrame, DeduplicationStats, list[SemanticDeduplicationMatch]]:
    if dedupe_mode == "semantic":
        return semantic_deduplicate_dataframe(
            dataframe,
            target_column=target_column,
            category_column=category_column,
            progress_callback=progress_callback,
            row_index_offset=row_index_offset,
            deduplicator=semantic_deduplicator,
            collect_matches=True,
        )

    deduplicated, stats = deduplicate_dataframe(
        dataframe,
        target_column=target_column,
        seen_keys=seen_keys,
        progress_callback=progress_callback,
    )
    return deduplicated, stats, []
