from __future__ import annotations

import argparse
from pathlib import Path

from data_process.cleaning import CleaningStats, clean_dataframe, load_dataframe


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

    args = parser.parse_args()
    if not args.input_file:
        print("未检测到支持的输入文件，请提供 csv 或 Excel 文件（.csv/.xls/.xlsx/.xlsm）。")
        return 1

    if args.action == "clean":
        return _run_clean(args.input_file, args.output)

    parser.print_help()
    return 1


def _run_clean(input_file: str, output_arg: str | None) -> int:
    try:
        dataframe = load_dataframe(input_file)
    except ValueError as exc:
        print(str(exc))
        return 1

    cleaned, stats = clean_dataframe(dataframe)
    output_path = _resolve_output_path(Path(input_file), output_arg)
    _write_dataframe(cleaned, output_path)
    _print_stats(stats, output_path)
    return 0


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
