from __future__ import annotations

import argparse
import json
import math
import tempfile
from datetime import UTC, datetime
from logging import Logger
from pathlib import Path

import pandas as pd

from data_process.cleaning import (
    CleaningStats,
    clean_dataframe,
)
from data_process.deduplication import DeduplicationStats, deduplicate_dataframe
from data_process.file_io import (
    append_dataframe_chunk,
    count_csv_rows,
    iter_dataframes,
    load_dataframe,
    write_dataframe,
    write_match_rows,
)
from data_process.logging_utils import close_logger, configure_logger
from data_process.progress import ProgressBar, run_stage
from data_process.semantic_deduplication import (
    DEFAULT_EMBEDDING_MODEL_PATH,
    SemanticDeduplicator,
    semantic_deduplicate_dataframe,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="数据处理工具")
    parser.add_argument(
        "--action",
        required=True,
        choices=["clean", "deduplicate", "clean-deduplicate"],
        help="要执行的功能。当前支持 clean、deduplicate、clean-deduplicate。",
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

    args = parser.parse_args()
    if not args.input_file:
        print("未检测到支持的输入文件，请提供 csv 或 Excel 文件（.csv/.xls/.xlsx/.xlsm）。")
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
            args.dedupe_mode,
            args.semantic_threshold,
            args.embedding_model_path,
            args.batch_size,
            args.semantic_index_type,
            args.semantic_hnsw_m,
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
    write_match_rows(match_rows, _resolve_match_output_path(output_path))
    _write_meta(
        output_path=output_path,
        action="deduplicate",
        input_path=input_path,
        parameters=_build_deduplication_parameters(
            chunk_size=chunk_size,
            target_column=target_column,
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
    write_match_rows(match_rows, _resolve_match_output_path(output_path))
    _write_meta(
        output_path=output_path,
        action="clean-deduplicate",
        input_path=input_path,
        parameters={
            "chunk_size": chunk_size,
            "target_column": target_column,
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
            write_match_rows(chunk_match_rows, match_output_path, append=True)
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


def _resolve_match_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_matches.csv")


def _resolve_meta_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.meta.json")


def _resolve_log_path(output_path: Path) -> Path:
    return output_path.parent / "data-process.log"
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
    match_output_path: Path | None = None,
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
    if match_output_path is not None and match_output_path.exists():
        meta["match_file"] = str(match_output_path)

    _resolve_meta_output_path(output_path).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_deduplication_parameters(
    chunk_size: int,
    target_column: str,
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
