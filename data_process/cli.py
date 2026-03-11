from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from data_process.cleaning import (
    CleaningStats,
    clean_dataframe,
    count_csv_rows,
    iter_dataframes,
    load_dataframe,
)
from data_process.progress import ProgressBar, run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="数据处理工具")
    parser.add_argument(
        "--action",
        required=True,
        choices=["clean"],
        help="要执行的功能。当前支持 clean。",
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
        help="csv 分块流式清洗时每块读取的行数。仅对 csv 生效，默认 50000。",
    )
    parser.add_argument(
        "--target-column",
        default="text",
        help="指定执行清洗判断的目标列名，默认 text。",
    )

    args = parser.parse_args()
    if not args.input_file:
        print("未检测到支持的输入文件，请提供 csv 或 Excel 文件（.csv/.xls/.xlsx/.xlsm）。")
        return 1
    if args.chunk_size <= 0:
        print("--chunk-size 必须是大于 0 的整数。")
        return 1

    if args.action == "clean":
        return _run_clean(args.input_file, args.output, args.chunk_size, args.target_column)

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

    if input_path.suffix.lower() == ".csv":
        try:
            stats = _run_clean_csv_stream(input_path, output_path, chunk_size, target_column)
        except ValueError as exc:
            print(str(exc))
            return 1
        _print_stats(stats, output_path)
        return 0

    try:
        run_stage("读取文件")
        dataframe = load_dataframe(input_file)
    except ValueError as exc:
        print(str(exc))
        return 1

    progress_bar = ProgressBar(total=len(dataframe), description="清洗数据")
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

    run_stage("写出结果")
    _write_dataframe(cleaned, output_path)
    _print_stats(stats, output_path)
    return 0


def _run_clean_csv_stream(
    input_path: Path,
    output_path: Path,
    chunk_size: int,
    target_column: str,
) -> CleaningStats:
    total_stats = CleaningStats(total_before=0, total_after=0)
    run_stage("统计总行数")
    total_rows = count_csv_rows(input_path)
    if output_path.exists():
        output_path.unlink()
    if total_rows == 0:
        empty_bar = ProgressBar(total=1, description="分块清洗")
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
        write_bar = ProgressBar(total=1, description="写出结果")
        pd.DataFrame().to_csv(output_path, index=False)
        write_bar.advance(1)
        write_bar.close()
        return total_stats

    progress_bar = ProgressBar(total=total_rows, description="分块清洗")
    chunk_total = math.ceil(total_rows / chunk_size)
    write_bar = ProgressBar(total=chunk_total, description="写出结果")
    wrote_header = False

    try:
        for chunk in iter_dataframes(input_path, chunksize=chunk_size):
            cleaned_chunk, chunk_stats = clean_dataframe(
                chunk,
                target_column=target_column,
                progress_callback=progress_bar.advance,
            )
            total_stats.merge(chunk_stats)
            cleaned_chunk.to_csv(output_path, mode="a", index=False, header=not wrote_header)
            wrote_header = True
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


def _resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)
    return input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")


def _write_dataframe(dataframe, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        dataframe.to_csv(output_path, index=False)
        return
    dataframe.to_excel(output_path, index=False)


def _print_stats(stats: CleaningStats, output_path: Path) -> None:
    print(f"清洗完成，输出文件：{output_path}")
    print(f"清洗前总行数：{stats.total_before}")
    print(f"删除空行：{stats.removed_blank_rows}")
    print(f"删除全符号/标点行：{stats.removed_symbol_rows}")
    print(f"删除全表情行：{stats.removed_emoji_rows}")
    print(f"删除全乱码行：{stats.removed_garbled_rows}")
    print(f"共删除：{stats.total_removed}")
    print(f"清洗后总行数：{stats.total_after}")
